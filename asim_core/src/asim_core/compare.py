"""
Compare asim_core results with ActivitySim 1.x results.

Columns are matched by row ID. Each comparison gets a status:

* ``exact``             every value identical (NaN counts as equal to NaN)
* ``within tolerance``  numeric values differ, but by no more than the tolerance
* ``different``         at least one value differs by more than the tolerance
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from .store import Table


@dataclass
class ColumnComparison:
    step: str
    table: str
    column: str
    rows: int
    mismatched: int
    max_abs_diff: float | None
    status: str
    dtype: str
    reference_dtype: str
    note: str = ""


def _values(x):
    if isinstance(x, pd.Categorical):
        return np.asarray(x.astype(object))
    return np.asarray(x)


def compare_column(
    step: str,
    table: Table,
    column: str,
    reference: pd.Series,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> ColumnComparison:
    ref_ids = reference.index.to_numpy()
    note = ""
    if len(ref_ids) != len(table) or not np.isin(ref_ids, table.index).all():
        note = f"row IDs differ: ours {len(table)}, reference {len(ref_ids)}"
        common = np.intersect1d(ref_ids, table.index)
        reference = reference.loc[common]
        ref_ids = common
    ours = _values(table[column])[table.positions(ref_ids)]
    ref = _values(reference)

    numeric = ours.dtype.kind in "biuf" and ref.dtype.kind in "biuf"
    if numeric:
        a = ours.astype(np.float64)
        b = ref.astype(np.float64)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.abs(a - b)
        diff[both_nan] = 0.0
        diff[np.isnan(diff)] = np.inf
        mismatched = int((diff != 0).sum())
        max_diff = float(diff.max()) if len(diff) else 0.0
        if mismatched == 0:
            status = "exact"
        elif np.all(diff <= atol + rtol * np.abs(b)):
            status = "within tolerance"
        else:
            status = "different"
    else:
        # missing text is missing whether NaN (pandas in memory) or None (after a Parquet round trip)
        eq = (ours == ref) | (pd.isna(ours) & pd.isna(ref))
        mismatched = int((~eq).sum())
        max_diff = None
        status = "exact" if mismatched == 0 else "different"
    # categories: the same values in a different category list are not 1.x's column
    ours_cat, ref_cat = _categories(table[column]), _categories(reference)
    if ours_cat is not None and ref_cat is not None and ours_cat != ref_cat:
        status = "different"
        note = (note + "; " if note else "") + f"categories differ: ours {ours_cat}, reference {ref_cat}"

    return ColumnComparison(
        step=step,
        table=table.name,
        column=column,
        rows=len(ref_ids),
        mismatched=mismatched,
        max_abs_diff=max_diff,
        status=status,
        dtype=str(ours.dtype),
        reference_dtype=str(reference.dtype),
        note=note,
    )


def value_shares(ours, reference) -> pd.DataFrame:
    """Share of rows taking each value, side by side (for statistical comparisons)."""
    a = pd.Series(_values(ours)).value_counts(normalize=True).rename("asim_core")
    b = pd.Series(_values(reference)).value_counts(normalize=True).rename("asim1")
    out = pd.concat([a, b], axis=1).fillna(0.0).sort_index()
    out["difference"] = out["asim_core"] - out["asim1"]
    return out


def write_comparisons(rows: list[ColumnComparison], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(ColumnComparison)])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path


def _categories(values):
    dtype = getattr(values, "dtype", None)
    if isinstance(dtype, pd.CategoricalDtype):
        return [str(c) for c in dtype.categories]
    return None
