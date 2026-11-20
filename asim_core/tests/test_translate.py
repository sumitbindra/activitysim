import numpy as np
import pandas as pd
import pytest

from asim_core.spec.translate import FormulaError, build_function, compile_function, translate

DF = pd.DataFrame(
    {
        "num_workers": np.array([0, 1, 2, 5], dtype=np.int64),
        "num_drivers": np.array([1, 2, 3, 4], dtype=np.int8),
        "income_in_thousands": np.array([10.5, 40.0, 80.0, 200.0]),
        "density_index": np.array([3.0, 12.0, 0.5, 25.0], dtype=np.float32),
        "ratio": np.array([0.2, 0.4, np.nan, 0.8]),
        "county_id": np.array([1, 9, 6, 9]),
    }
)
CONSTANTS = {"ID_MARIN": 9, "HHT_FAMILY": [1, 2, 3]}

# formulas in both 1.x styles, checked against what pandas / Python give
FORMULAS = [
    "num_drivers==2",
    "num_drivers>3",
    "(num_workers==0)*(0.66*income_in_thousands+0.34*density_index)",
    "@df.num_workers.clip(upper=3)",
    "@df.income_in_thousands.clip(0, 30)",
    "@(df.income_in_thousands-30).clip(0, 45)",
    "@(df.num_workers==0)*(df.density_index-10).clip(0)",
    "@df.county_id == ID_MARIN",
    "@df.county_id.isin(HHT_FAMILY)",
    "@df.income_in_thousands.between(10, 80)",
    "@df.ratio.fillna(0)",
    "@np.where(df.num_workers > 0, df.income_in_thousands / df.num_workers, 0)",
    "@np.minimum(df.num_workers, 2) + np.maximum(df.num_drivers, 3)",
    "@df['num_workers'] * 2",
]


def reference(formula):
    if formula.startswith("@"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return eval(formula[1:], {}, {"df": DF, "np": np, **CONSTANTS})
    return DF.eval(formula)


def evaluate(formula):
    tr = translate(formula, CONSTANTS)
    fn = compile_function(build_function("f", [("t", tr)]), "f")
    with np.errstate(divide="ignore", invalid="ignore"):
        return next(fn({c: DF[c].to_numpy() for c in DF.columns})), tr


@pytest.mark.parametrize("formula", FORMULAS)
def test_matches_pandas_exactly(formula):
    ours, _ = evaluate(formula)
    ref = np.asarray(reference(formula))
    # compare as the float64 term values 1.x builds
    a = np.broadcast_to(np.asarray(ours, dtype=np.float64), (len(DF),))
    b = np.broadcast_to(ref.astype(np.float64), (len(DF),))
    np.testing.assert_array_equal(a, b)


def test_columns_and_constants_are_recorded():
    _, tr = evaluate("@(df.num_workers==0)*df.density_index.clip(0, 10) + (df.county_id == ID_MARIN)")
    assert tr.columns == {"num_workers", "density_index", "county_id"}
    assert tr.constants == {"ID_MARIN"}
    assert "9" in tr.source  # constants are written into the code


def test_plain_style_names_are_columns():
    tr = translate("(num_children_5_to_15+num_children_16_to_17)>0")
    assert tr.columns == {"num_children_5_to_15", "num_children_16_to_17"}


@pytest.mark.parametrize(
    "formula, message",
    [
        ("@df.x.rolling(3)", "rolling"),
        ("@np.random.rand()", "unsupported"),
        ("@UNKNOWN_CONSTANT * df.x", "unknown name"),
        ("@df.x and df.y", "&"),
        ("@__import__('os')", "unsupported"),
        ("num_workers ==", "not valid syntax"),
    ],
)
def test_unsupported_formulas_are_rejected(formula, message):
    with pytest.raises(FormulaError, match=message):
        translate(formula, CONSTANTS)


def test_plain_style_arithmetic_uses_pandas_eval():
    """numexpr (used by pandas.eval, as in 1.x) computes float32 x int64 in 32 bits; NumPy wouldn't."""
    tr = translate("a * i")
    assert tr.source.startswith("_pandas_eval(")
    fn = compile_function(build_function("f", [("t", tr)]), "f")
    a = np.array([0.1, 1.7, 2.3], dtype=np.float32)
    i = np.array([1, 2, 3])
    (value,) = list(fn({"a": a, "i": i}))
    expected = pd.eval("a * i", resolvers=[{"a": pd.Series(a), "i": pd.Series(i)}]).to_numpy()
    np.testing.assert_array_equal(value, expected)
    # comparisons and logic give the same values either way: plain NumPy code
    assert not translate("(a > 1) & (i < 3)").source.startswith("_pandas_eval(")


def test_interaction_style_names_and_timetable_calls():
    tr = translate("_x * start", {"K": 2}, assigned={"_x"}, pandas_locals=True)
    assert "'_x': v['_x']" in tr.source and "'start': c['start']" in tr.source
    tr = translate("@tt.previous_tour_ends(df.person_id, df.start)", objects={"tt": {"previous_tour_ends"}})
    assert tr.source == "tt.previous_tour_ends(c['person_id'], c['start'])"
    with pytest.raises(FormulaError, match="not supported"):
        translate("@tt.rollback()", objects={"tt": {"previous_tour_ends"}})


def test_pandas_mode_reads_and_or_as_pandas_does():
    # 1.x evaluates utility expressions without @ with pandas.eval: & and | bind like and/or
    import numpy as np
    import pandas as pd

    from asim_core.spec.translate import RUNTIME_NAMESPACE

    df = pd.DataFrame({"ptype": [7, 7, 1, 7], "age": [5, 8, 8, 12]})
    for formula in ["ptype == 7 & age >= 6", "ptype == 1 | age < 6 & ptype == 7", "~(age > 6) | ptype == 1"]:
        tr = translate(formula, {}, python_mode=False, df_columns={"ptype", "age"})
        c = {k: df[k].to_numpy() for k in df}
        got = eval(tr.source, dict(RUNTIME_NAMESPACE), {"c": c})
        np.testing.assert_array_equal(np.asarray(got), df.eval(formula).to_numpy(), err_msg=formula)
