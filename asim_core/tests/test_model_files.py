"""
asim_core's formula language, the converter from 1.x's formulas, and the
model files. With a saved 1.x run, each region's six ported steps are
converted and run from the converted files, and must match 1.x exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from asim_core.spec.convert import LegacyScope, NewNames, convert_formula
from asim_core.spec.language import Scope, SkimScope, compile_formula, split_assignment
from asim_core.spec.model_file import ModelFileError, read_coefficients_table, read_model_yaml
from asim_core.spec.translate import FormulaError, compile_function

REPO = Path(__file__).resolve().parents[2]
RUNS = {
    "prototype_mtc": REPO / "asim_core/work/asim1_legacy",
    "prototype_mwcog": REPO / "asim_core/work/mwcog/asim1_legacy",
    "prototype_arc": REPO / "asim_core/work/arc/asim1_legacy",
}


def _scope(**kw) -> Scope:
    base = dict(
        roles={"person": {"age", "ptype"}, "household": {"income", "home_zone_id"}},
        derived={"out_period"}, constants={"PTYPE_FULL": 1, "BINS": {1: 5.0}}, coefficients={"c_ivt"},
        parameters={"has_tour_type": False}, tables={"land_use": {"TOPOLOGY", "TOTPOP"}},
        skims=SkimScope({("o", "d", "out_period"): "odt_skims", ("o", "d", None): "od_skims",
                         ("d", "o", None): "do_skims"}, lookup="skim_dict"),
        zones={"o": "home_zone_id", "d": "destination"}, settings={"x": 3})
    base.update(kw)
    return Scope(**base)


@pytest.mark.parametrize("formula, source", [
    ("person.age > 18 and household.income < 50000", "((c['age'] > 18) & (c['income'] < 50000))"),
    ("18 <= person.age < 65", "((18 <= c['age']) & (c['age'] < 65))"),
    ("person.ptype in [PTYPE_FULL, 2]", "np.isin(c['ptype'], [1, 2])"),
    ("person.ptype not in [1]", "(~np.isin(c['ptype'], [1]))"),
    ("where(person.age > 5, 1, 0) * c_ivt", "(np.where((c['age'] > 5), 1, 0) * k['c_ivt'])"),
    ("skim.SOV_TIME[o, d, out_period]", "s('odt_skims', 'SOV_TIME')"),
    ("max(skim.DIST[d, o], skim.DIST[o, d])", "np.maximum(s('do_skims', 'DIST'), s('od_skims', 'DIST'))"),
    ("skim.DIST[o, d, 'AM']", "s('od_skims', 'DIST__AM')"),
    ("skim.DIST[household.home_zone_id, d, 'MD']", "s.lookup('DIST__MD', c['home_zone_id'], c['destination'])"),
    ("land_use.TOPOLOGY[d]", "t('land_use', 'TOPOLOGY', c['destination'])"),
    ("1 if has_tour_type else 2", "2"),
    ("not person.age > 3", "np.logical_not((c['age'] > 3))"),
    ("clip(household.income / 1000, upper=30)", "_clip((c['income'] / 1000), None, 30)"),
    ("setting('x') + 1", "(3 + 1)"),
    ("map(person.ptype, BINS)", "_map(c['ptype'], {1: 5.0})"),
    ("int8(count_by(land_use.TOTPOP, household.home_zone_id))",
     "_astype(t.count('land_use', 'TOTPOP', c['home_zone_id']), 'int8')"),
    ("log1p(out_period)", "np.log1p(c['out_period'])"),
    ("fillna(sum_by(land_use.TOTPOP, land_use.TOPOLOGY, household.home_zone_id), 0.0)",
     "_fillna(t.sum_by('land_use', 'TOTPOP', 'TOPOLOGY', c['home_zone_id']), 0.0)"),
])
def test_language_compiles_to_numpy(formula, source):
    assert compile_formula(formula, _scope()).source == source


@pytest.mark.parametrize("formula, message", [
    ("person.agee", "person has no column 'agee'; did you mean age?"),
    ("persn.age", "unknown table 'persn'"),
    ("foo + 1", "unknown name 'foo'"),
    ("where(person.age > 1 if person.age else 2, 1, 2)", "must be known when the model loads"),
    ("person.age & 1", "unsupported operator (use and, or, not for logic)"),
    ("skim.DIST", "a skim needs zones"),
    ("land_use.TOPOLOGY", "looked up by ID"),
    ("random_lognormal(mean=1, sd=2)", "only available in derived values"),
    ("wher(person.age, 1, 2)", "did you mean where?"),
])
def test_language_errors_say_what_is_wrong(formula, message):
    with pytest.raises(FormulaError, match=message.replace("?", r"\?").replace("(", r"\(").replace(")", r"\)")):
        compile_formula(formula, _scope())


def test_numpy_number_rules_whatever_the_row_count():
    """A float32 column times an int64 column: 64-bit results at any size (no pandas.eval)."""
    tr = compile_formula("person.age * household.income", _scope())
    fn = compile_function(f"def f(c):\n    return {tr.source}\n", "f")
    for n in (10, 2_000_000):
        out = fn({"age": np.full(n, 0.1, dtype=np.float32), "income": np.full(n, 3, dtype=np.int64)})
        assert out.dtype == np.float64


def test_split_assignment_keeps_the_text():
    assert split_assignment("_DIST = skim.DIST[o,  d]") == ("_DIST", "skim.DIST[o,  d]")
    assert split_assignment("a == b") == (None, "a == b")


ROLES = {"age": "person", "income": "household", "ptype": "person", "tour_type": "tour",
         "destination": "tour", "home_zone_id": "household", "household_id": "household"}.get


@pytest.mark.parametrize("legacy, scope, new", [
    ("@df.income.clip(0, 30)", {}, "clip(household.income, 0, 30)"),
    ("(ptype == 1) & (age > 18)", {}, "ptype == 1 and age > 18".replace("ptype", "person.ptype")
     .replace("age", "person.age")),
    ("@~df.ptype.isin([1, 2])", {}, "not person.ptype in [1, 2]"),
    ("@np.where(df.age.between(5, 10), 1, 0)", {}, "where(between(person.age, 5, 10), 1, 0)"),
    ("@df.age.between(5, 10, inclusive='left')", {}, "5 <= person.age < 10"),
    ("@reindex(land_use.TOTPOP + land_use.TOTEMP, df[dest_col_name])",
     {"tables": {"land_use"}, "constants": {"dest_col_name": "destination"}},
     "land_use.TOTPOP[d] + land_use.TOTEMP[d]"),
    ("@df.tour_type if 'tour_type' in df.columns else 1", {}, "tour.tour_type if has_tour_type else 1"),
    ("@odt_skims['SOV_TIME'] + dot_skims['SOV_TIME']",
     {"skims": {"odt_skims": ("o", "d", "out_period"), "dot_skims": ("d", "o", "in_period")}},
     "skim.SOV_TIME[o, d, out_period] + skim.SOV_TIME[d, o, in_period]"),
    ("@od_skims.max('DIST')", {"skims": {"od_skims": ("o", "d", None)}}, "max(skim.DIST[d, o], skim.DIST[o, d])"),
    ("@skim_od[('SOVTOLL_TIME', 'AM')]", {"skims": {"skim_od": ("o", "d", None)}}, "skim.SOVTOLL_TIME[o, d, 'AM']"),
    ("@rng.lognormal_for_df(df, mu=1, sigma=2, broadcast=True, scale=True)", {},
     "random_lognormal(mean=1, sd=2)"),
    ("@reindex_i(jtp.groupby('person_id').size(), df.age)", {"tables": {"jtp"}},
     "int8(count_by(jtp.person_id, person.age))"),
    ("@pd.cut(df.age, bins=[0, 5], labels=[1]).astype(int)", {}, "int(cut(person.age, bins=[0, 5], labels=[1]))"),
    ("persons.x.groupby(persons.household_id).sum().reindex(households.index).fillna(0.0)",
     {"python_mode": True, "tables": {"persons"}, "df_names": {"households"}, "index_name": "household_id"},
     "fillna(sum_by(persons.x, persons.household_id, household.household_id), 0.0)"),
])
def test_converter(legacy, scope, new):
    assert convert_formula(legacy, LegacyScope(**scope), NewNames(role_of=ROLES)).text == new


def test_converter_turns_column_tests_into_parameters():
    out = convert_formula("@df.tour_type if 'tour_type' in df.columns else 1", LegacyScope(), NewNames(ROLES))
    assert out.parameters == {"has_tour_type": "tour_type"}


def test_converter_settles_tests_of_input_tables():
    legacy = LegacyScope(tables={"land_use"}, table_columns={"land_use": {"a"}}, constants={"x": 1})
    out = convert_formula("@reindex(land_use.b, df.age) if 'b' in land_use else x", legacy, NewNames(ROLES))
    assert out.text == "x"  # the branch that applies to this region's land use


# --- model files ----------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_extends_and_coefficient_values(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    _write(base / "m_terms.csv", "Label,Description,Formula,a,b\nt1,,household.income,c1,\n")
    _write(base / "m_coefficients.csv", "coefficient_name,value\nc1,0.5\nc2,1.25\n")
    _write(base / "m.yaml", "kind: auto_ownership\nalternatives: [a, b]\nterms: m_terms.csv\n"
                            "coefficients: m_coefficients.csv\noptions: {utility_shift: false}\n")
    region = tmp_path / "region"
    region.mkdir()
    _write(region / "m.yaml", f"extends: {base / 'm.yaml'}\ncoefficient_values: {{c1: 0.75}}\n"
                              "options: {drop_zero_terms: false}\n")
    data = read_model_yaml(region / "m.yaml")
    assert data["terms"] == str((base / "m_terms.csv").resolve())  # relative to the file naming it
    assert data["options"] == {"utility_shift": False, "drop_zero_terms": False}  # merged
    coefs, _ = read_coefficients_table(data["coefficients"], data["coefficient_values"])
    assert coefs["c1"].value == 0.75 and coefs["c2"].value == 1.25
    with pytest.raises(ModelFileError, match="unknown coefficients"):
        read_coefficients_table(data["coefficients"], {"c9": 1})


def test_misspelled_option_is_an_error(tmp_path):
    from asim_core.components.auto_ownership import load_model_file

    _write(tmp_path / "m_terms.csv", "Label,Description,Formula,a\nt1,,household.income,c1\n")
    _write(tmp_path / "m_coefficients.csv", "coefficient_name,value\nc1,0.5\n")
    path = _write(tmp_path / "m.yaml", "kind: auto_ownership\nalternatives: [a]\nterms: m_terms.csv\n"
                                       "coefficients: m_coefficients.csv\noptions: {utilty_shift: true}\n")
    with pytest.raises(ModelFileError, match="options.utilty_shift: not a setting"):
        load_model_file(path, [])


def test_wrong_column_is_reported_at_startup(tmp_path):
    """Formulas are checked against the run's tables before anything runs."""
    from asim_core.components.auto_ownership import auto_ownership

    _write(tmp_path / "m_terms.csv", "Label,Description,Formula,a,b\nt1,,household.incme,c1,\n")
    _write(tmp_path / "m_coefficients.csv", "coefficient_name,value\nc1,0.5\n")
    _write(tmp_path / "auto_ownership.yaml", "kind: auto_ownership\nalternatives: [a, b]\n"
                                             "terms: m_terms.csv\ncoefficients: m_coefficients.csv\n")
    comp = auto_ownership([tmp_path])
    with pytest.raises(ValueError, match="household has no column 'incme'; did you mean income"):
        comp.resolve_needs({"households": {"income", "home_zone_id"}, "land_use": set(), "accessibility": set()})


# --- the three regions, converted and run ------------------------------------------------------


@pytest.mark.parametrize("example", list(RUNS))
def test_converted_steps_match_1x_exactly(example, tmp_path):
    pytest.importorskip("tables")
    run = RUNS[example]
    if not (run / "pipeline.parquetpipeline").exists():
        pytest.skip(f"no saved 1.x run for {example}")
    import sys

    sys.path.insert(0, str(REPO / "asim_core/harness"))
    from run_milestone1 import skim_store

    from asim_core.asim1 import Asim1State
    from asim_core.compare import compare_column
    from asim_core.converter import STEPS, convert_step
    from asim_core.runner import Runner

    ex = REPO / "activitysim/examples" / example
    legacy = [ex / "configs"]
    st = Asim1State(run)
    skims = skim_store(ex, legacy)
    out = tmp_path / "configs"
    for step in STEPS:
        convert_step(step, legacy, st, out)
    for step, entry in STEPS.items():
        comp = entry["component"]([out] + legacy)
        store = st.load_store(st.previous_step(step))
        Runner([comp], tmp_path / step, skims=skims).run([step], store)
        written = st.columns_at(step)
        for table, cols in comp.provides.items():
            ref = st.read(table, step)
            for col in cols:
                if col in written.get(table, ()):
                    result = compare_column(step, store[table], col, ref[col])
                    assert result.status == "exact", result


def test_committed_model_files_are_up_to_date(tmp_path):
    """The converted MTC files in asim_core/configs are what the converter writes."""
    run = RUNS["prototype_mtc"]
    if not (run / "pipeline.parquetpipeline").exists():
        pytest.skip("no saved 1.x run")
    from asim_core.asim1 import Asim1State
    from asim_core.converter import STEPS, convert_step

    st = Asim1State(run)
    legacy = [REPO / "activitysim/examples/prototype_mtc/configs"]
    for step in STEPS:
        for path in convert_step(step, legacy, st, tmp_path):
            committed = REPO / "asim_core/configs/prototype_mtc" / path.name
            assert committed.read_text() == path.read_text(), path.name
    assert yaml.safe_load((tmp_path / "tour_mode_choice.yaml").read_text())["kind"] == "tour_mode_choice"


def test_lookups_of_missing_ids_and_no_zone_give_missing_values():
    """As 1.x: reindex() of an ID not in the table, and skims for zone -1, are missing values."""
    from asim_core.component import Component, StepContext
    from asim_core.components.runtime import TableLookup, ZoneSkims
    from asim_core.seams import Engine
    from asim_core.store import Store, Table

    store = Store([Table("land_use", np.array([1, 2, 3]), "zone_id", {"area_type": np.array([5, 6, 7])}),
                   Table("persons", np.array([10, 11, 12]), "person_id",
                         {"x": np.array([1.0, np.nan, 2.0]), "household_id": np.array([7, 7, 8])})])
    comp = Component(name="t", run=lambda ctx: {})
    ctx = StepContext(comp, store, {"land_use": ["area_type"], "persons": ["x", "household_id"]}, Engine(),
                      None, 0)
    t = TableLookup(ctx)
    np.testing.assert_array_equal(t("land_use", "area_type", np.array([2, -1])), [6.0, np.nan])
    assert t("land_use", "area_type", np.array([3, 1])).dtype.kind == "i"  # all found: the column's type
    # pandas' groupby sum: missing values skipped; keys with no rows missing
    np.testing.assert_array_equal(t.sum_by("persons", "x", "household_id", np.array([7, 8, 9])), [1.0, 2.0, np.nan])

    class FakeSkims(ZoneSkims):
        def __init__(self):
            pass

    s = FakeSkims()
    s.ctx = type("C", (), {"skim_positions": staticmethod(lambda z: np.asarray(z) - 1),
                           "skim": staticmethod(lambda name, o, d: (o * 10 + d).astype(np.float32))})()
    np.testing.assert_array_equal(s.lookup("DIST", np.array([1, 2]), np.array([3, -1])), [2.0, np.nan])
