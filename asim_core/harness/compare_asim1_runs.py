#!/usr/bin/env python
"""
Compare two ActivitySim 1.x runs table by table, column by column, at their
last saved step (or ``--step``). Used to check that a change to 1.x (see
asim1_patches.py) leaves results identical.

    python asim_core/harness/compare_asim1_runs.py RUN_A RUN_B

Prints one line per table and exits with 0 only if every table has the same
rows and every column is identical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "asim_core" / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asim_core.asim1 import Asim1State  # noqa: E402


def _equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        x, y = a.to_numpy(), b.to_numpy()
        return (x == y) | (np.isnan(x) & np.isnan(y))
    return (a.astype(object).to_numpy() == b.astype(object).to_numpy()) | (a.isna().to_numpy() & b.isna().to_numpy())


def compare_runs(run_a, run_b, step: str | None = None) -> pd.DataFrame:
    a, b = Asim1State(run_a), Asim1State(run_b)
    step_a = step or a.steps[-1]
    step_b = step or b.steps[-1]
    rows = []
    tables = sorted(set(a.tables_at(step_a)) | set(b.tables_at(step_b)))
    for t in tables:
        if t not in a.tables_at(step_a) or t not in b.tables_at(step_b):
            rows.append({"table": t, "rows_a": None, "rows_b": None, "columns": None,
                         "columns_different": None, "note": "only in one run"})
            continue
        da, db = a.read(t, step_a), b.read(t, step_b)
        note = ""
        same_rows = len(da) == len(db) and da.index.equals(db.index)
        if not same_rows:
            note = "different rows"
            common = da.index.intersection(db.index)
            da, db = da.loc[common], db.loc[common]
        cols = sorted(set(da.columns) | set(db.columns))
        different = []
        for c in cols:
            if c not in da or c not in db:
                different.append(f"{c} (missing)")
            elif not _equal(da[c], db[c]).all():
                different.append(f"{c} ({int((~_equal(da[c], db[c])).sum())} rows)")
        rows.append({"table": t, "rows_a": len(a.read(t, step_a)), "rows_b": len(b.read(t, step_b)),
                     "columns": len(cols), "columns_different": ", ".join(different), "note": note})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--step", default=None)
    args = p.parse_args(argv)
    df = compare_runs(args.run_a, args.run_b, args.step)
    print(df.to_string(index=False))
    identical = (df["columns_different"].fillna("x") == "").all() and (df["note"] == "").all()
    print("IDENTICAL" if identical else "DIFFERENT")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
