"""
The initialize steps on a small made-up region: reading, household sampling
as 1.x does it, and the annotation functions 1.x's initialize files use,
checked against pandas doing what 1.x's formulas do.
"""

import ast
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from asim_core.components.initialize import (
    InitializeError,
    initialize_households,
    initialize_landuse,
    sample_households,
)
from asim_core.inputs.reader import input_tables_from_settings, read_table, table_columns, to_pandas
from asim_core.profiler import Profiler
from asim_core.random import hash32
from asim_core.runner import Runner
from asim_core.spec.translate import parse_pandas_query
from asim_core.store import Store

REPO = Path(__file__).resolve().parents[2]


def test_pandas_query_precedence():
    # pandas reads & and | with the precedence of and/or, unlike Python
    assert ast.unparse(parse_pandas_query("age>=6 & age<=15")) == "age >= 6 and age <= 15"
    assert ast.unparse(parse_pandas_query("a==1 | b>2 & c")) == "a == 1 or (b > 2 and c)"
    assert ast.unparse(parse_pandas_query(" ~adult ")) == "~adult"


def _people(n_hh=20, seed=1):
    rng = np.random.default_rng(seed)
    hh = pd.DataFrame({"home_zone_id": rng.integers(1, 4, n_hh), "income": rng.integers(0, 200_000, n_hh)},
                      index=pd.Index(np.arange(101, 101 + n_hh), name="household_id"))
    sizes = rng.integers(1, 5, n_hh)
    persons = pd.DataFrame({"household_id": np.repeat(hh.index, sizes),
                            "age": rng.integers(0, 90, sizes.sum())},
                           index=pd.Index(np.arange(1, sizes.sum() + 1), name="person_id"))
    # 1.x keeps the persons file's order: shuffle it so that it differs from the households'
    persons = persons.sample(frac=1.0, random_state=3)
    return hh, persons


def test_sampling_as_1x():
    hh, persons = _people()
    s_hh, s_persons = sample_households(hh, persons, 5, base_seed=7)
    idx = np.random.RandomState([7, hash32("sample_households")]).choice(len(hh), size=5, replace=False)
    assert list(s_hh.index) == list(hh.index[idx])  # in the order drawn
    assert (s_hh.sample_rate == 5 / 20).all()
    expected = persons[persons.household_id.isin(s_hh.index)]
    assert list(s_persons.index) == list(expected.index)  # in the file's order


def test_sampling_keeps_the_traced_household():
    hh, persons = _people()
    s_hh, _ = sample_households(hh, persons, 5, base_seed=7)
    trace = next(i for i in hh.index if i not in s_hh.index)
    t_hh, _ = sample_households(hh, persons, 5, trace=trace, base_seed=7)
    assert t_hh.index[0] == trace and list(t_hh.index[1:]) == list(s_hh.index[1:])
    one, one_p = sample_households(hh, persons, 1, trace=trace, base_seed=7)
    assert list(one.index) == [trace] and set(one_p.household_id) == {trace}


def test_no_sampling():
    hh, persons = _people()
    for n, rate in ((0, 1), (20, 1.0), (50, 50 / 20)):  # 1.x: the literal ratio, even above 1
        s_hh, s_persons = sample_households(hh, persons, n)
        assert list(s_hh.index) == list(hh.index) and len(s_persons) == len(persons)
        assert (s_hh.sample_rate == rate).all()


def test_sampling_checks():
    hh, persons = _people()
    bad = persons.copy()
    bad.iloc[0, bad.columns.get_loc("household_id")] = 999
    with pytest.raises(InitializeError, match="persons with bad household_id"):
        sample_households(hh, bad, 0)
    with pytest.raises(InitializeError, match="households with no persons"):
        sample_households(hh, persons[persons.household_id != hh.index[3]], 0)


@pytest.mark.parametrize("example", ["prototype_mtc", "prototype_mwcog", "prototype_arc"])
def test_table_columns_without_reading(example):
    configs = [REPO / "activitysim/examples" / example / "configs"]
    data = [REPO / "activitysim/examples" / example / "data"]
    specs = input_tables_from_settings(configs)
    tables = {s.tablename: read_table(s, data)[0] for s in specs}
    for recode in (False, True):
        dfs = to_pandas(tables, specs, recode=recode)
        for s in specs:
            assert table_columns(s, data, recode=recode) == list(dfs[s.tablename].columns), s.tablename


def _region(tmp: Path, sample_size: int):
    configs, data = tmp / "configs", tmp / "data"
    configs.mkdir()
    data.mkdir()
    (configs / "settings.yaml").write_text(dedent(f"""
        households_sample_size: {sample_size}
        rng_base_seed: 3
        input_table_list:
          - tablename: land_use
            filename: land_use.csv
            index_col: zone_id
          - tablename: households
            filename: households.csv
            index_col: household_id
          - tablename: persons
            filename: persons.csv
            index_col: person_id
    """))
    (configs / "constants.yaml").write_text("ADULT_AGE: 18\n")
    (configs / "initialize_landuse.yaml").write_text(dedent("""
        annotate_tables:
          - tablename: land_use
            annotate:
              SPEC: annotate_landuse
    """))
    (configs / "annotate_landuse.csv").write_text(dedent("""\
        Description,Target,Expression
        ,density,df.pop / df.area
    """))
    (configs / "initialize_households.yaml").write_text(dedent("""
        annotate_tables:
          - tablename: persons
            annotate:
              SPEC: annotate_persons
              DF: persons
              TABLES:
                - households
          - tablename: households
            annotate:
              SPEC: annotate_households
              DF: households
              TABLES:
                - persons
                - land_use
    """))
    (configs / "annotate_persons.csv").write_text(dedent("""\
        Description,Target,Expression
        ,is_adult,persons.age >= ADULT_AGE
        ,has_other_adult,"other_than(persons.household_id, persons.age >= ADULT_AGE)"
        ,age_top,"persons.age.where(persons.age < 80, 80)"
        ,log_vot,"rng.lognormal_for_df(df, mu=np.log(10), sigma=0.5, broadcast=False)"
    """))
    (configs / "annotate_households.csv").write_text(dedent("""\
        Description,Target,Expression
        ,_PERSON_COUNT,"lambda query, persons, households: persons.query(query).groupby('household_id').size().reindex(households.index).fillna(0).astype(np.int8)"
        ,num_school,"_PERSON_COUNT('age>=6 & age<=15', persons, households)"
        ,num_adults,"_PERSON_COUNT('is_adult', persons, households)"
        ,num_children,"_PERSON_COUNT('~is_adult', persons, households)"
        ,zeros,np.zeros(len(households))
        ,density,"reindex(land_use.density, households.home_zone_id)"
    """))
    pd.DataFrame({"start": [5, 5, 6], "end": [5, 23, 23]}).to_csv(
        configs / "tour_departure_and_duration_alternatives.csv", index=False)
    lu = pd.DataFrame({"zone_id": [3, 1, 2], "pop": [100.0, 50.0, 10.0], "area": [2.0, 4.0, 3.0]})
    lu.to_csv(data / "land_use.csv", index=False)
    hh, persons = _people()
    hh.to_csv(data / "households.csv")
    persons.to_csv(data / "persons.csv")
    return [configs], [data], lu, hh, persons


@pytest.mark.parametrize("sample_size", [0, 6])
def test_initialize_steps(tmp_path, sample_size):
    configs, data, lu, hh, persons = _region(tmp_path, sample_size)
    comps = [initialize_landuse(configs, data), initialize_households(configs, data)]
    store = Store()
    Runner(comps, tmp_path / "out", base_seed=3).run([c.name for c in comps], store, Profiler())

    land_use = store["land_use"].to_pandas()
    assert list(land_use.index) == [1, 2, 3]  # sorted, as 1.x
    np.testing.assert_array_equal(land_use.density, (lu.set_index("zone_id").sort_index().eval("pop / area")))
    assert store["accessibility"].column_names == [] and list(store["accessibility"].index) == [1, 2, 3]

    s_hh, s_persons = sample_households(hh, persons, sample_size, base_seed=3)
    h, p = store["households"].to_pandas(), store["persons"].to_pandas()
    assert list(h.index) == list(s_hh.index) and list(p.index) == list(s_persons.index)

    # persons, as pandas does 1.x's formulas
    adult = s_persons.age >= 18
    n_adults = s_persons.household_id.map(s_persons[adult].household_id.value_counts()).fillna(0)
    np.testing.assert_array_equal(p.is_adult, adult)
    np.testing.assert_array_equal(p.has_other_adult, np.where(adult, n_adults > 1, n_adults > 0))
    np.testing.assert_array_equal(p.age_top, s_persons.age.where(s_persons.age < 80, 80))
    assert (p.log_vot > 0).all() and p.log_vot.nunique() == len(p)  # one draw per person

    # households
    def count(q):
        return (s_persons.assign(is_adult=adult).query(q).groupby("household_id").size()
                .reindex(s_hh.index).fillna(0).astype(np.int8))

    for col, q in (("num_school", "age>=6 & age<=15"), ("num_adults", "is_adult"),
                   ("num_children", "~is_adult")):
        np.testing.assert_array_equal(h[col], count(q))
        assert h[col].dtype == np.int8
    np.testing.assert_array_equal(h.zeros, np.zeros(len(h)))
    np.testing.assert_array_equal(h.density, land_use.density.reindex(s_hh.home_zone_id).to_numpy())
    assert list(h.columns) == ["home_zone_id", "income", "sample_rate", "num_school", "num_adults",
                               "num_children", "zeros", "density"]
    assert store["households_skipped"].column_names == ["home_zone_id", "income", "sample_rate"]
    assert len(store["households_skipped"].index) == 0

    windows = store["person_windows"]
    assert windows.column_names == [str(w) for w in range(4, 25)]
    assert all(np.asarray(windows[c]).dtype == np.int8 and not np.asarray(windows[c]).any()
               for c in windows.column_names)
