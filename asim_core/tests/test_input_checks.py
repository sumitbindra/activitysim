"""
Input table definitions, the reader and the checks before the run.

The region tests copy an example's data, inject one error, and check that
the report names it (and nothing else); the clean data must pass.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from asim_core.inputs import DefinitionError, InputError, check_inputs, load_definitions
from asim_core.inputs.checks import Report
from asim_core.inputs.reader import input_tables_from_settings, read_tables, recode_order, to_pandas

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "activitysim/examples"
OURS = REPO / "asim_core/configs"


# --- small made-up region --------------------------------------------------------


def _region(tmp: Path, households, persons, land_use, definitions: dict | None = None,
            size_terms: str | None = None) -> tuple[list[Path], list[Path]]:
    configs, data = tmp / "configs", tmp / "data"
    configs.mkdir()
    data.mkdir()
    settings = {"input_table_list": [
        {"tablename": "households", "filename": "households.csv", "index_col": "household_id"},
        {"tablename": "persons", "filename": "persons.csv", "index_col": "person_id"},
        {"tablename": "land_use", "filename": "land_use.csv", "index_col": "zone_id"},
    ]}
    (configs / "settings.yaml").write_text(yaml.safe_dump(settings))
    if definitions is not None:
        (configs / "input_tables.yaml").write_text(yaml.safe_dump(definitions))
    if size_terms is not None:
        (configs / "destination_choice_size_terms.csv").write_text(size_terms)
    pd.DataFrame(households).to_csv(data / "households.csv", index=False)
    pd.DataFrame(persons).to_csv(data / "persons.csv", index=False)
    pd.DataFrame(land_use).to_csv(data / "land_use.csv", index=False)
    return [configs], [data]


HH = {"household_id": [1, 2, 3], "home_zone_id": [10, 20, 20], "hhsize": [2, 1, 1]}
PER = {"person_id": [1, 2, 3, 4], "household_id": [1, 1, 2, 3], "age": [40, 8, 70, 25]}
LU = {"zone_id": [10, 20], "jobs": [100, 0], "lot_spaces": [50, 0]}


def _messages(report: Report) -> list[tuple[str, str, str | None]]:
    return [(p.severity, p.table, p.column) for p in report.problems]


def test_clean_made_up_region_passes(tmp_path):
    c, d = _region(tmp_path, HH, PER, LU)
    r = check_inputs(c, d, skims=False)
    assert r.ok and not r.problems, r.text()
    assert r.rows_checked == {"households": 3, "persons": 4, "land_use": 2}
    assert r.undefined_columns["persons"] == ["age"]


def test_each_rule(tmp_path):
    definitions = {"tables": {
        "persons": {"columns": {"age": {"type": "int64", "min": 0, "max": 110, "warn_above": 99}}},
        "land_use": {"columns": {
            "jobs": {"type": "int64", "min": 0},
            "lot_spaces": {"type": "int64", "warn_below": 1, "description": "park-and-ride capacity"}}},
    }}
    per = dict(PER, age=[40, -1, 150, 100])
    per["person_id"] = [1, 2, 2, 4]  # a repeated ID
    hh = dict(HH, home_zone_id=[10, 30, 20], hhsize=[2, 1, 2])  # unknown zone; wrong size
    c, d = _region(tmp_path, hh, per, dict(LU, jobs=[100, -5]), definitions)
    r = check_inputs(c, d, skims=False)
    found = {(p.severity, p.table, p.column, p.message.split(" (")[0]): p for p in r.problems}
    assert ("error", "persons", "age", "below the minimum 0") in found
    assert ("error", "persons", "age", "above the maximum 110") in found
    assert ("warning", "persons", "age", "above 99") in found
    assert ("error", "persons", "person_id", "1 value used by more than one row") in found
    assert ("error", "households", "home_zone_id", "not found in land_use.zone_id") in found
    assert ("error", "households", "hhsize", "differs from the number of persons rows with this household_id") in found
    assert ("error", "land_use", "jobs", "below the minimum 0") in found
    assert ("warning", "land_use", "lot_spaces", "below 1") in found
    assert not r.ok
    zone = found[("error", "households", "home_zone_id", "not found in land_use.zone_id")]
    assert zone.rows == 1 and zone.examples == ["household_id 2: 30"]
    size = found[("error", "households", "hhsize", "differs from the number of persons rows with this household_id")]
    assert size.examples == ["household_id 3: hhsize 2, persons rows 1"]
    with pytest.raises(InputError, match="home_zone_id"):
        r.raise_if_errors()


def test_types_and_missing_values(tmp_path):
    definitions = {"tables": {"persons": {"columns": {
        "age": {"type": "int64"}, "income": {"type": "float64", "missing_allowed": True},
        "code": {"type": "int8"}}}}}
    per = dict(PER, age=[40, 8.5, None, 25], income=[1.0, None, 3.0, 4.0], code=[1, 2, 300, 4])
    c, d = _region(tmp_path, HH, per, LU, definitions)
    r = check_inputs(c, d, skims=False)
    msgs = {(p.column, p.message.split(" (")[0]) for p in r.problems}
    assert ("age", "missing values") in msgs
    assert ("age", "not whole numbers") in msgs
    assert ("code", "outside the range of int8") in msgs
    assert not any(p.column == "income" for p in r.problems)
    per = dict(PER, age=["40", "eight", "70", "25"])
    shutil.rmtree(tmp_path / "configs"), shutil.rmtree(tmp_path / "data")
    c, d = _region(tmp_path, HH, per, LU, definitions)
    r = check_inputs(c, d, skims=False)
    bad = [p for p in r.problems if p.column == "age"]
    assert bad and bad[0].message.startswith("not numbers") and bad[0].examples == ["person_id 2: eight"]


def test_sum_and_allowed_values(tmp_path):
    definitions = {"tables": {"land_use": {"columns": {
        "jobs": {"type": "int64", "sum_of": ["a", "b"], "sum_tolerance": 1},
        "a": {"type": "int64"}, "b": {"type": "int64"},
        "kind": {"type": "str", "values": ["urban", "rural"]}}}}}
    lu = dict(LU, a=[60, 0], b=[39, 5], kind=["urban", "town"])
    c, d = _region(tmp_path, HH, PER, lu, definitions)
    r = check_inputs(c, d, skims=False)
    assert _messages(r) == [("warning", "land_use", "jobs"), ("error", "land_use", "kind")]
    assert r.problems[0].examples == ["zone_id 20: 0, sum 5"]


def test_size_terms_need_no_definitions(tmp_path):
    terms = ("model_selector,segment,jobs,lot_spaces\n"
             "workplace,all,1,0\n"
             "non_mandatory,parking,0,1\n")
    c, d = _region(tmp_path, HH, PER, dict(LU, jobs=[-3, 0]), size_terms=terms)
    r = check_inputs(c, d, skims=False)
    msgs = [(p.column, p.message.split(":")[0]) for p in r.problems]
    assert ("jobs", "negative values in a size term (all)") in msgs
    assert (None, "size of segment 'all' (workplace) is zero in every zone, so no destination can be chosen "
                  "for it (size terms") in [(c, m.split(": jobs")[0]) for c, m in msgs]


def test_definition_files_are_checked(tmp_path):
    bad = {"tables": {"land_use": {"columns": {"jobs": {"type": "int64", "mn": 0}}}}}
    c, d = _region(tmp_path, HH, PER, LU, bad)
    with pytest.raises(DefinitionError, match="'mn' is not a setting"):
        load_definitions(c)
    bad = {"tables": {"land_use": {"columns": {"jobs": {"references": "zones.zone"}}}}}
    (c[0] / "input_tables.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(DefinitionError, match="is not a defined column"):
        load_definitions(c)
    bad = {"tables": {"land_use": {"columns": {"jobs": {"min": 5, "max": 1}}}}}
    (c[0] / "input_tables.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(DefinitionError, match="above max"):
        load_definitions(c)


def test_region_rule_changes_keep_the_rest(tmp_path):
    defs = load_definitions([])
    assert defs.tables["households"].columns["hhsize"].warn_above == 20
    c, _ = _region(tmp_path, HH, PER, LU, {"tables": {"households": {"columns": {"hhsize": {"warn_above": 30}}}}})
    hhsize = load_definitions(c).tables["households"].columns["hhsize"]
    assert hhsize.warn_above == 30 and hhsize.min == 1 and hhsize.count_of == "persons.household_id"


def test_report_is_reused_for_unchanged_inputs(tmp_path):
    c, d = _region(tmp_path, HH, PER, LU)
    first = check_inputs(c, d, cache_dir=tmp_path / "cache", skims=False)
    again = check_inputs(c, d, cache_dir=tmp_path / "cache", skims=False)
    assert not first.cached and again.cached and again.key == first.key
    pd.DataFrame(dict(HH, hhsize=[3, 1, 1])).to_csv(d[0] / "households.csv", index=False)
    changed = check_inputs(c, d, cache_dir=tmp_path / "cache", skims=False)
    assert not changed.cached and not changed.ok


# --- the three example regions -----------------------------------------------------

REGIONS = {
    # example: (total employment, an employment size term, university size terms)
    "prototype_mtc": ("TOTEMP", "RETEMPN", ["COLLFTE", "COLLPTE"]),
    "prototype_mwcog": ("TOTEMP", "RETEMP", ["COLLEGE"]),
    "prototype_arc": ("emp", "N4445", ["univ"]),
}
INJECTIONS = ["negative_total_employment", "negative_size_term", "duplicate_person_id",
              "unknown_home_zone", "household_size", "missing_skim_value", "zero_university_capacity"]


def _example_paths(example):
    configs = [OURS / example, EXAMPLES / example / "configs"]
    return configs, [EXAMPLES / example / "data"]


@pytest.mark.parametrize("example", REGIONS)
def test_example_inputs_pass(example):
    pytest.importorskip("tables")
    configs, data = _example_paths(example)
    r = check_inputs(configs, data)
    assert r.ok and not r.problems, r.text()
    assert not any(r.undefined_columns.values()), r.undefined_columns


def _copy_data(example, tmp: Path) -> Path:
    src = EXAMPLES / example / "data"
    dst = tmp / "data"
    dst.mkdir()
    for p in src.iterdir():
        if p.is_file():
            (dst / p.name).symlink_to(p)
    return dst


def _rewrite(data: Path, configs, table: str, change) -> None:
    spec = {s.tablename: s for s in input_tables_from_settings(configs)}[table]
    p = data / spec.filename
    df = pd.read_csv(p.resolve())
    rename = spec.rename_columns or {}
    raw = {v: k for k, v in rename.items()}
    change(df, lambda c: raw.get(c, c))
    p.unlink()
    df.to_csv(p, index=False)


@pytest.mark.parametrize("injection", INJECTIONS)
@pytest.mark.parametrize("example", REGIONS)
def test_injected_errors_are_caught(example, injection, tmp_path):
    tables = pytest.importorskip("tables")
    configs, _ = _example_paths(example)
    data = _copy_data(example, tmp_path)
    total_emp, size_term, univ = REGIONS[example]

    def first_zone_negative(column):
        def change(df, raw):
            df.loc[df.index[0], raw(column)] = -abs(df.loc[df.index[0], raw(column)]) - 100
        return change

    if injection == "negative_total_employment":
        _rewrite(data, configs, "land_use", first_zone_negative(total_emp))
        expect = {("error", "land_use", total_emp), ("warning", "land_use", total_emp)}
    elif injection == "negative_size_term":
        _rewrite(data, configs, "land_use", first_zone_negative(size_term))
        expect = {("error", "land_use", size_term)}  # its own rule and the size term rule
        expect_extra = {("warning", "land_use", total_emp)}  # total no longer the sum
    elif injection == "duplicate_person_id":
        def change(df, raw):
            df.loc[df.index[1], raw("person_id")] = df.loc[df.index[0], raw("person_id")]
        _rewrite(data, configs, "persons", change)
        expect = {("error", "persons", "person_id")}
    elif injection == "unknown_home_zone":
        def change(df, raw):
            df.loc[df.index[0], raw("home_zone_id")] = 999_999
        _rewrite(data, configs, "households", change)
        expect = {("error", "households", "home_zone_id")}
    elif injection == "household_size":
        def change(df, raw):
            df.loc[df.index[0], raw("hhsize")] += 1
        _rewrite(data, configs, "households", change)
        expect = {("error", "households", "hhsize")}
    elif injection == "missing_skim_value":
        from asim_core.skims import omx_files_from_settings
        src = omx_files_from_settings(configs, [EXAMPLES / example / "data"])[0]
        dst = data / src.name
        dst.unlink()
        shutil.copy(src, dst)
        with tables.open_file(dst, "a") as h:
            node = h.list_nodes("/data")[0]
            a = node[:]
            a[1, 2] = np.nan
            node[:] = a
            name = node.name
        expect = {("error", "skims", name)}
    elif injection == "zero_university_capacity":
        def change(df, raw):
            for c in univ:
                df[raw(c)] = 0
        _rewrite(data, configs, "land_use", change)
        expect = {("error", "land_use", None)}
    r = check_inputs(configs, [data])
    found = set(_messages(r))
    if injection == "negative_size_term":
        assert found == expect | expect_extra, r.text()
        assert len([p for p in r.problems if p.column == size_term and p.severity == "error"]) == 2
    else:
        assert found == expect, r.text()
    assert not r.ok
    if injection == "zero_university_capacity":
        msg = r.problems[0].message
        assert "is zero in every zone" in msg and any(c in msg for c in univ), r.text()
    if injection == "negative_size_term":
        assert "destination_choice_size_terms" in " ".join(r.problems[0].used_by + r.problems[1].used_by)


# --- skim problems the models would use without an error -----------------------------

# example: (a highway time skim used with each tour's period, the distance skim for the zone order)
SKIM_REGIONS = {
    "prototype_mtc": ("SOV_TIME", "DIST"),
    "prototype_mwcog": ("SOV_TIME", "DIST"),
    "prototype_arc": ("SOV_FREE_TIME", "SOV_FREE_DISTANCE__MD"),
}
# injection: (expected problems as (severity, column), words in the message)
SKIM_INJECTIONS = {
    "skims_in_another_order": ("error", "distance", "doesn't follow the land use"),
    "missing_period": ("error", "time", "has no skim for period"),
    "skim_not_in_files": ("error", "time", "not in the skims"),
    "unknown_period": ("warning", None, "isn't in network_los.yaml"),
    "duplicate_skim": ("error", "time__AM", "different values"),
    "time_in_seconds": ("warning", "time", "per unit of distance"),
    "no_path_placeholder": ("warning", "time", "placeholder"),
    "zero_within_zone": ("warning", "time", "within a zone"),
}


def _write_omx(path: Path, mats: dict) -> None:
    import tables

    with tables.open_file(str(path), "w") as f:
        f.root._v_attrs["OMX_VERSION"] = b"0.2"
        shape = next(iter(mats.values())).shape
        f.root._v_attrs["SHAPE"] = np.array(shape, dtype=np.int32)
        f.create_group("/", "data")
        for name, m in mats.items():
            f.create_carray("/data", name, obj=np.asarray(m))


@pytest.mark.parametrize("injection", SKIM_INJECTIONS)
@pytest.mark.parametrize("example", SKIM_REGIONS)
def test_injected_skim_problems_are_caught(example, injection, tmp_path):
    tables = pytest.importorskip("tables")
    from asim_core.skims import omx_files_from_settings, skim_period_labels

    configs, _ = _example_paths(example)
    data = _copy_data(example, tmp_path)
    time, distance = SKIM_REGIONS[example]
    labels = skim_period_labels(configs)
    src = omx_files_from_settings(configs, [EXAMPLES / example / "data"])[0]
    dst = data / src.name
    dst.unlink()
    shutil.copy(src, dst)

    with tables.open_file(dst, "a") as h:
        def matrix(name):
            return h.get_node("/data/" + name)

        if injection == "skims_in_another_order":
            # the same skims, rows and columns in another zone order, the zone list unchanged
            n = matrix(f"{time}__AM").shape[0]
            p = np.random.default_rng(1).permutation(n)
            for node in h.list_nodes("/data"):
                node[:] = node[:][np.ix_(p, p)]
        elif injection == "missing_period":
            h.remove_node("/data", f"{time}__{labels[-1]}")
        elif injection == "skim_not_in_files":
            for label in labels:
                h.remove_node("/data", f"{time}__{label}")
        elif injection == "unknown_period":
            h.create_carray("/data", "EXTRA__XX", obj=matrix(f"{time}__AM")[:])
        elif injection == "duplicate_skim":
            other = matrix(f"{time}__AM")[:] * 2
        elif injection == "time_in_seconds":
            node = matrix(f"{time}__AM")
            node[:] = node[:] * 60
        elif injection == "no_path_placeholder":
            node = matrix(f"{time}__AM")
            a = node[:]
            a[0, 1:4] = 999_999
            node[:] = a
        elif injection == "zero_within_zone":
            node = matrix(f"{time}__AM")
            a = node[:]
            a[0, 0] = 0
            node[:] = a

    if injection == "duplicate_skim":
        _write_omx(data / "skims_extra.omx", {f"{time}__AM": other})
        if "*" not in str(yaml.safe_load((EXAMPLES / example / "configs/network_los.yaml").read_text())
                          ["taz_skims"]["omx"]):
            # this region names one file: list both
            overlay = tmp_path / "configs"
            overlay.mkdir()
            los = yaml.safe_load((EXAMPLES / example / "configs/network_los.yaml").read_text())
            los["taz_skims"]["omx"] = [src.name, "skims_extra.omx"]
            (overlay / "network_los.yaml").write_text(yaml.safe_dump(los))
            configs = [overlay] + configs

    r = check_inputs(configs, [data])
    severity, which, words = SKIM_INJECTIONS[injection]
    column = {"time": time, "distance": distance, "time__AM": f"{time}__AM", None: None}[which]
    assert set(_messages(r)) == {(severity, "skims", column)}, r.text()
    assert any(words in p.message for p in r.problems), r.text()
    assert r.ok == (severity == "warning")


def test_skim_definitions_are_checked_when_loaded(tmp_path):
    (tmp_path / "input_tables.yaml").write_text(
        "skims:\n  measures:\n    '*_TIME': {unit: minutes, typical: [300, 1]}\n")
    with pytest.raises(DefinitionError, match="typical"):
        load_definitions([tmp_path])
    (tmp_path / "input_tables.yaml").write_text("skims:\n  zone_order: {x: X}\n")
    with pytest.raises(DefinitionError, match="x and y"):
        load_definitions([tmp_path])


# --- the reader against 1.x's -----------------------------------------------------


@pytest.mark.parametrize("like_asim1", [True, False], ids=["like_asim1", "correctly_rounded"])
@pytest.mark.parametrize("example", REGIONS)
def test_reader_gives_1x_tables(example, like_asim1, tmp_path):
    """
    Same tables as 1.x's reader (read_input_table): columns, index and
    values. Read like 1.x, every value is identical; read correctly
    rounded, floats may differ from 1.x's in the last few bits.
    """
    asim1 = pytest.importorskip("activitysim.core.input")
    from activitysim.core import workflow

    configs, data = _example_paths(example)
    configs = configs[1:]
    tables, _, specs = read_tables(configs, data, like_asim1=like_asim1)
    recode = bool(yaml.safe_load((configs[0] / "settings.yaml").read_text()).get("recode_pipeline_columns", False))
    ours = to_pandas(tables, specs, recode=recode)
    state = workflow.State.make_default(working_dir=EXAMPLES / example, configs_dir=configs,
                                        data_dir=data, output_dir=tmp_path)
    for spec in recode_order(specs):
        ref = asim1.read_input_table(state, spec.tablename)
        state.add_table(spec.tablename, ref)
        got = ours[spec.tablename]
        assert list(got.columns) == list(ref.columns), spec.tablename
        assert got.index.name == ref.index.name
        np.testing.assert_array_equal(got.index.to_numpy(), ref.index.to_numpy())
        for c in ref.columns:
            a, b = got[c].to_numpy(), ref[c].to_numpy()
            where = f"{spec.tablename}.{c}"
            if a.dtype.kind in "iufb" and b.dtype.kind in "iufb":
                if like_asim1 or b.dtype.kind != "f":
                    np.testing.assert_array_equal(a, b, err_msg=where)
                    assert a.dtype == b.dtype or not like_asim1, where
                else:
                    np.testing.assert_array_max_ulp(a.astype(float), b, maxulp=8)
            else:
                assert (pd.Series(a).fillna("<NA>").astype(str) == pd.Series(b).fillna("<NA>").astype(str)).all(), where
