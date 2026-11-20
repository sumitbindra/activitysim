"""
Preprocessor features used by mode choice: tests decided when the model
loads, scalar promotion, table lookups, dictionaries, pd.cut, random
lognormal draws. Each is checked against the same rows evaluated the way 1.x
does (Python eval over pandas Series).
"""

import numpy as np
import pandas as pd
import pytest

from asim_core.spec.definition import Assignment, AssignmentSpec
from asim_core.spec.translate import FormulaError, _lognormal, translate

CONSTANTS = {"shortWalk": 0.333, "waits": {1: 5.5, 2: 9.5, 3: 13.3}, "dest_col_name": "destination"}


def spec(rows, **kw):
    return AssignmentSpec(
        name="pre",
        assignments=[Assignment(target=t, formula=f) for t, f in rows],
        constants=CONSTANTS,
        tables=["land_use"],
        settings={"cbd_threshold": 2},
        **kw,
    )


ROWS = [
    ("_DF_IS_TOUR", "'tour_type' in df.columns"),
    ("number_of_participants", "df.number_of_participants if _DF_IS_TOUR else 1"),
    ("_parent_mode", "reindex(land_use.mode, df.parent_id) if 'parent_id' in df.columns else ''"),
    ("parent_is_bike", "_parent_mode == 'BIKE'"),
    ("_dist", "reindex(land_use.access, df[dest_col_name]) if 'access' in land_use else shortWalk"),
    ("walk_time", "_dist * 60 / 3"),
    ("topology", "reindex(land_use.TOPOLOGY, df[dest_col_name])"),
    ("in_cbd", "(reindex(land_use.area_type, df[dest_col_name]) < setting('cbd_threshold')) * 1"),
    ("density", "pd.cut(df.density, bins=[-np.inf, 10, 20, np.inf], labels=[3, 2, 1]).astype(int)"),
    ("wait_mean", "density.map({k: v for k, v in waits.items()})"),
]


def test_tests_are_decided_when_bound():
    s = spec(ROWS)
    with pytest.raises(ValueError, match="depend on the run's columns"):
        s.function()  # needs the run's columns first
    b = s.bind({"tour_type", "number_of_participants", "destination", "density"},
               {"land_use": {"TOPOLOGY", "area_type"}})
    assert "_DF_IS_TOUR = True" in b.source  # worked out, not computed
    code = "\n".join(line for line in b.source.splitlines() if not line.strip().startswith("#"))
    assert "parent_id" not in code and "access" not in code  # branches not taken are gone
    assert "v['_parent_mode'] = _full('', n)" in code
    assert b.columns_used() == {"number_of_participants", "destination", "density"}
    assert b.tables_used() == {("land_use", "TOPOLOGY"), ("land_use", "area_type")}


def test_values_match_python_eval():
    rng = np.random.default_rng(1)
    n = 40
    zones = np.arange(1, 6)
    land_use = pd.DataFrame({"TOPOLOGY": [1, 2, 3, 1, 2], "area_type": [0, 1, 2, 3, 4]}, index=zones)
    df = pd.DataFrame({
        "tour_type": "work", "number_of_participants": rng.integers(1, 4, n),
        "destination": rng.choice(zones, n), "density": rng.random(n) * 30,
    })
    b = spec(ROWS).bind(set(df.columns), {"land_use": set(land_use.columns)})

    def t(table, column, keys):
        return land_use[column].to_numpy()[land_use.index.get_indexer(keys)]

    v = b.function({c: df[c].to_numpy() for c in df.columns}, None, n, t)

    # 1.x: eval with pandas, scalar results made into Series
    from activitysim.core.util import reindex  # noqa: F401  (skip if 1.x is not installed)

    loc = {"df": df, "pd": pd, "np": np, "reindex": reindex, "land_use": land_use,
           "setting": lambda k: {"cbd_threshold": 2}[k], **CONSTANTS}
    for target, formula in ROWS:
        val = eval(formula, {}, loc)
        if not (target.startswith("_") and target.isupper()) and np.isscalar(val):
            val = pd.Series(val, index=df.index)
        loc[target] = val
    for target in b.targets:
        ours, ref = np.asarray(v[target]), np.asarray(loc[target])
        np.testing.assert_array_equal(ours, ref)
        assert ours.dtype == ref.dtype, target


def test_random_draws_are_numbered_in_order():
    rows = [("a", "rng.lognormal_for_df(df, mu=df.m, sigma=df.s, broadcast=True, scale=True)"),
            ("b", "df.m * 2"),
            ("c", "rng.lognormal_for_df(df, mu=df.m, sigma=df.s, broadcast=True, scale=True).clip(0, 50)")]
    b = AssignmentSpec(name="r", assignments=[Assignment(target=t, formula=f) for t, f in rows]).bind(set())
    assert b.n_random == 2
    assert "r[:, 0]" in b.source and "r[:, 1]" in b.source
    m, s = np.array([5.5, 9.5]), np.array([0, 2])
    draws = np.array([[0.1, -0.3], [1.2, 0.4]])
    v = b.function({"m": m, "s": s}, None, 2, None, draws)
    np.testing.assert_array_equal(v["a"], _lognormal(draws[:, 0], m, s, scale=True))
    # with no spread, the value is the mean (up to exp(log(x)) rounding)
    np.testing.assert_allclose(v["a"][0], 5.5)


def test_undecidable_condition_is_rejected():
    with pytest.raises(FormulaError, match="decided"):
        translate("df.a if df.b > 0 else 1", {}, python_mode=True, df_columns={"a", "b"})


def test_lognormal_broadcast_modes():
    def bound(*formulas):
        return AssignmentSpec(name="r", assignments=[
            Assignment(target=f"a{i}", formula=f) for i, f in enumerate(formulas)]).bind(set())

    # 1.x's default (broadcast=False): one draw per row per call, from each row's stream
    b = bound("rng.lognormal_for_df(df, mu=1, sigma=0)", "rng.lognormal_for_df(df, mu=2, sigma=1)")
    assert b.n_random == 2 and b.draws_per_call
    assert not bound("rng.lognormal_for_df(df, mu=1, sigma=0, broadcast=True)").draws_per_call
    with pytest.raises(ValueError, match="mixes lognormal draws"):
        bound("rng.lognormal_for_df(df, mu=1, sigma=0, broadcast=True)",
              "rng.lognormal_for_df(df, mu=1, sigma=0)").draws_per_call
