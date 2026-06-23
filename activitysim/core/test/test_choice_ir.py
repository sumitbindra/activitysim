# ActivitySim
# See full license in LICENSE.txt.
"""
Parity tests for the reference choice IR (Milestone 1).

For each representative destination/location-choice expression we evaluate it
two ways and assert the results match:

  (a) ground truth: stock Python ``eval`` of the original string against pandas
      DataFrame / numpy / skim objects (what ActivitySim does today), and
  (b) the IR: ``compile_expression`` -> ``evaluate`` over the same data.

This de-risks the IR design before any Rust evaluator is built: it confirms the
IR covers the real expression vocabulary and reproduces pandas/numpy semantics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from activitysim.core import choice_ir
from activitysim.core.choice_ir import (
    EvalContext,
    can_compile,
    compile_expression,
    evaluate,
    split_temp_assignment,
)

N = 8
RNG = np.random.RandomState(42)


class SizeTermLookup:
    """Mimics ActivitySim's ``DataFrameMatrix`` size-term accessor.

    ``get(zone_ids, segments)`` returns one size term per row.
    """

    def __init__(self):
        # size term = base[segment] * zone (deterministic, easy to check)
        self._base = {"work": 10.0, "school": 4.0, "shop": 2.5}

    def get(self, zones, segments):
        zones = np.asarray(zones)
        segments = np.asarray(segments)
        base = np.array([self._base[s] for s in segments], dtype=float)
        return base * zones


def _make_data():
    """Build a synthetic interaction dataframe + skim/size/constant context."""
    df = pd.DataFrame(
        {
            "size_term": RNG.uniform(0, 100, N),
            "dest_taz": RNG.randint(1, 6, N).astype(float),
            "purpose": RNG.choice(["work", "school", "shop"], N),
            "is_joint": RNG.randint(0, 2, N).astype(bool),
            "outbound": RNG.randint(0, 2, N).astype(bool),
            "tour_mode_is_walk": RNG.randint(0, 2, N).astype(bool),
            "mode_choice_logsum": RNG.uniform(-5, 5, N),
            "pick_count": RNG.randint(1, 10, N).astype(float),
            "prob": RNG.uniform(0.01, 1.0, N),
        }
    )
    # a couple of rows with size_term == 0 to exercise availability gates
    df.loc[0, "size_term"] = 0.0

    skim_dist = RNG.uniform(0, 30, N)
    skim_distwalk = RNG.uniform(0, 5, N)
    skim_od = RNG.uniform(0, 30, N)
    skim_dp = RNG.uniform(0, 30, N)

    # ground-truth skim objects: dict access returns numpy arrays
    gt_skims = {"DIST": skim_dist, "DISTWALK": skim_distwalk}
    gt_od = {"DIST": skim_od, "DISTWALK": skim_distwalk}
    gt_dp = {"DIST": skim_dp}

    size_terms = SizeTermLookup()
    constants = {"max_walk_distance": 3.0, "max_bike_distance": 12.0}

    # ground-truth eval namespace
    gt_env = {
        "df": df,
        "np": np,
        "pd": pd,
        "skims": gt_skims,
        "od_skims": gt_od,
        "dp_skims": gt_dp,
        "size_terms": size_terms,
        **constants,
    }

    # IR evaluation context
    ctx = EvalContext(
        columns={c: df[c].to_numpy() for c in df.columns},
        skims={
            "skims": gt_skims,
            "od_skims": gt_od,
            "dp_skims": gt_dp,
        },
        constants=constants,
        size_terms=size_terms.get,
    )
    return df, gt_env, ctx, set(constants)


# Representative expressions spanning every IR category in recon §4.
EXPRESSIONS = [
    "@skims['DIST']",
    "@skims['DIST'].clip(0,1)",
    "@(skims['DIST']-1).clip(0,1)",
    "@(skims['DIST']-2).clip(0,3)",
    "@df['size_term'].apply(np.log1p)",
    "@df['size_term']==0",
    "@np.log1p(size_terms.get(df.dest_taz, df.purpose))",
    "@size_terms.get(df.dest_taz, df.purpose) == 0",
    "@(~df.is_joint & df.outbound) * (od_skims['DIST'] + dp_skims['DIST'])",
    "@df.outbound * od_skims['DIST']",
    "mode_choice_logsum",
    "@np.minimum(np.log(df.pick_count/df.prob), 60)",
    "@(df.tour_mode_is_walk) & (od_skims['DISTWALK'] > max_walk_distance)",
    "@np.where(df.outbound, od_skims['DIST'], dp_skims['DIST'])",
]


def _eval_ground_truth(expr: str, gt_env: dict) -> np.ndarray:
    src = expr.strip()
    if src.startswith("@"):
        src = src[1:]
    # Non-@ expressions are evaluated by ActivitySim via fast_eval in the df
    # namespace (column names directly in scope); mirror that here.
    ns = dict(gt_env)
    ns.update({c: gt_env["df"][c] for c in gt_env["df"].columns})
    return np.asarray(eval(src, ns))


def test_expression_parity():
    df, gt_env, ctx, consts = _make_data()
    for expr in EXPRESSIONS:
        gt = _eval_ground_truth(expr, gt_env)
        node = compile_expression(expr, constants=consts)
        ir = np.asarray(evaluate(node, ctx))

        # broadcast scalars/bools to comparable float arrays
        gt_f = np.broadcast_to(gt, (N,)).astype(float)
        ir_f = np.broadcast_to(ir, (N,)).astype(float)
        assert np.allclose(gt_f, ir_f, rtol=1e-12, atol=1e-12), (
            f"parity mismatch for {expr!r}\n gt={gt_f}\n ir={ir_f}"
        )


def test_spec_utility_assembly():
    """A whole (expr, coefficient) spec: IR-summed utility == pandas-summed."""
    df, gt_env, ctx, consts = _make_data()
    spec = {
        "@skims['DIST'].clip(0,1)": -0.8,
        "@(skims['DIST']-1).clip(0,1)": -0.3,
        "@df['size_term'].apply(np.log1p)": 1.0,
        "mode_choice_logsum": 0.7,
    }

    gt_util = np.zeros(N)
    ir_util = np.zeros(N)
    for expr, coef in spec.items():
        gt_util += coef * np.broadcast_to(
            _eval_ground_truth(expr, gt_env), (N,)
        ).astype(float)
        node = compile_expression(expr, constants=consts)
        ir_util += coef * np.broadcast_to(
            np.asarray(evaluate(node, ctx)), (N,)
        ).astype(float)

    assert np.allclose(gt_util, ir_util, rtol=1e-12, atol=1e-12)


def test_temp_assignment_split():
    name, body = split_temp_assignment("_od_DIST@od_skims['DIST']")
    assert name == "_od_DIST"
    assert body == "od_skims['DIST']"
    name, body = split_temp_assignment("@skims['DIST']")
    assert name is None
    assert body == "@skims['DIST']"


def test_unsupported_falls_back():
    # arbitrary python the IR intentionally does not model -> fallback signal
    assert not can_compile("@df.x.rolling(3).mean()")
    assert not can_compile("@mystery_func(df.x)")
    assert not can_compile("@df.x @ df.y")  # matmul
    # but the real ones do compile
    for expr in EXPRESSIONS:
        assert can_compile(expr, constants={"max_walk_distance", "max_bike_distance"}), expr


if __name__ == "__main__":
    # Allow running without pytest for quick verification.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} IR parity tests passed.")
