import numpy as np
import pandas as pd
import pytest

from asim_core.spec.definition import Assignment, AssignmentSpec
from asim_core.spec.translate import FormulaError, translate

CONSTANTS = {"weight": 2.0, "SCALE": 100}


def spec(rows, **kw):
    return AssignmentSpec(
        name="toy",
        assignments=[Assignment(target=t, formula=f) for t, f in rows],
        constants=CONSTANTS,
        skim_accessors=["skim_od", "skim_do"],
        **kw,
    )


def test_skim_keys_and_names_translate():
    tr = translate("skim_od[('TIME', 'AM')] + skim_do['DIST'] * weight + _x", CONSTANTS,
                   python_mode=True, assigned={"_x"}, skim_accessors={"skim_od", "skim_do"})
    assert tr.skims == {"TIME__AM", "DIST"}
    assert tr.names == {"_x"}
    assert "s('skim_od', 'TIME__AM')" in tr.source and "v['_x']" in tr.source


def test_rows_only_see_earlier_variables():
    with pytest.raises(ValueError, match="unknown name '_later'"):
        spec([("_a", "_later + 1"), ("_later", "df.x"), ("out", "_a")])


def test_bad_skim_key_is_rejected():
    with pytest.raises(FormulaError, match="skim keys"):
        translate("skim_od[1]", {}, python_mode=True, skim_accessors={"skim_od"})


def test_targets_and_evaluation_match_python():
    rows = [
        ("_t", "skim_od[('TIME', 'AM')] + skim_do[('TIME', 'PM')]"),
        ("_decay", "exp(_t * -0.05)"),
        ("a", "df.jobs * _decay"),
        ("_t", "skim_od['DIST'] * weight"),
        ("_decay", "(_t <= 3) * exp(_t * -1.0)"),
        ("b", "df.jobs * _decay / SCALE"),
        ("a", "a + 1"),  # reassigning keeps the first position
        ("_", "a * 0"),  # thrown away
    ]
    s = spec(rows)
    assert s.targets == ["a", "b"]
    assert s.skims_used() == {"TIME__AM", "TIME__PM", "DIST"}

    rng = np.random.default_rng(0)
    n = 50
    skims = {"TIME__AM": rng.random(n).astype(np.float32) * 20,
             "TIME__PM": rng.random(n).astype(np.float32) * 20,
             "DIST": rng.random(n).astype(np.float32) * 5}
    od = {k: v for k, v in skims.items()}
    do = {k: v[::-1].copy() for k, v in skims.items()}
    jobs = rng.integers(0, 1000, n)
    f, source = s.function()
    v = f({"jobs": jobs}, lambda acc, name: od[name] if acc == "skim_od" else do[name])

    # the same rows evaluated the way 1.x does: Python eval over pandas Series
    df = pd.DataFrame({"jobs": jobs})
    loc = {"df": df, "exp": np.exp, **CONSTANTS,
           "skim_od": {("TIME", "AM"): pd.Series(od["TIME__AM"]), ("TIME", "PM"): pd.Series(od["TIME__PM"]),
                       "DIST": pd.Series(od["DIST"])},
           "skim_do": {("TIME", "AM"): pd.Series(do["TIME__AM"]), ("TIME", "PM"): pd.Series(do["TIME__PM"]),
                       "DIST": pd.Series(do["DIST"])}}
    for target, formula in rows:
        val = eval(formula, {}, loc)
        if target != "_":
            loc[target] = val
    for t in s.targets:
        np.testing.assert_array_equal(np.asarray(v[t]), np.asarray(loc[t]))
        assert np.asarray(v[t]).dtype == np.asarray(loc[t]).dtype
    assert "def toy_assign(c, s):" in source
