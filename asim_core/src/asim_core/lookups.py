"""Implementations of the ``lookup``, ``skim_lookup``, ``sample_columns`` and ``size_lookup`` seams."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .seams import register


@register("lookup", "numpy", reference=True,
          description="NumPy indexing: values[positions]")
def lookup(values, positions: np.ndarray):
    return values[positions]


@register("skim_lookup", "flat_take", reference=True,
          description="one-dimensional gather: matrix.ravel().take(orig * zones + dest), the flat "
                      "positions worked out once per set of zone pairs and shared by its skims")
def skim_lookup(matrix: np.ndarray, orig: np.ndarray, dest: np.ndarray, pairs=None) -> np.ndarray:
    """
    ``pairs`` (a ``skims.ZonePairs`` for the same positions), when given, holds
    the flat positions, so a step looking up many skims for the same zone pairs
    works them out once. Faster than two-dimensional indexing on random zone
    pairs (30 to 40% on its own; docs/skim_experiments.md), with the same values.
    """
    if pairs is not None:
        flat = pairs.flat
    else:
        flat = np.asarray(orig, dtype=np.intp) * matrix.shape[1] + np.asarray(dest, dtype=np.intp)
    return matrix.reshape(-1).take(flat)


@register("skim_lookup", "index_2d", agreement="exact",
          description="NumPy two-dimensional indexing: matrix[orig, dest] (the default until the "
                      "skim experiment)")
def skim_lookup_2d(matrix: np.ndarray, orig: np.ndarray, dest: np.ndarray, pairs=None) -> np.ndarray:
    return matrix[orig, dest]


# --- sampling every zone for each chooser (trip destination) ----------------------------


@register("sample_columns", "spec_reads", reference=True,
          description="only the columns the sample spec reads (and the period, if it uses time-of-day "
                      "skims), as 1.x's drop_unused_columns does before sampling")
def sample_columns_spec_reads(available, read, alt_col, period_col, uses_period) -> list[str]:
    return [k for k in available if k != alt_col and (k in read or (uses_period and k == period_col))]


@register("sample_columns", "all_columns", agreement="exact",
          description="every column the choosers carry (the default until the trip destination "
                      "measurement; the same results)")
def sample_columns_all(available, read, alt_col, period_col, uses_period) -> list[str]:
    return [k for k in available if k != alt_col]


class ColumnsByPosition:
    """
    ``get(zones, purposes)`` for one batch of sampling (every chooser x every
    zone, zones in the table's order). When ``zones`` is the batch's zone column
    and ``purposes`` a column copied out from the choosers, a chooser's values
    are its purpose's column of the table: taken whole, once per batch (the spec
    may ask twice), instead of matching every row's zone and purpose. Other
    calls go to the table's ``get``. The values are the same.

    ``table``: ``.values`` (zones x columns), ``.column_positions(names)`` and
    ``.get(zones, names)``; ``repeated``: for each column copied out to the
    rows, (the rows' values, the choosers' values).
    """

    def __init__(self, table, zone_rows: np.ndarray, repeated):
        self.table = table
        self.zone_rows = zone_rows
        self.repeated = dict(repeated)
        self._cache: dict = {}
        self.fast_calls = 0

    def get(self, zones, purposes) -> np.ndarray:
        if zones is self.zone_rows:
            for name, (rows_values, chooser_values) in self.repeated.items():
                if purposes is rows_values:
                    if name not in self._cache:
                        cols = self.table.column_positions(chooser_values)
                        out = self.table.values.T[cols].reshape(-1)  # (choosers, zones), row by row
                        out.flags.writeable = False
                        self._cache[name] = out
                    self.fast_calls += 1
                    return self._cache[name]
        return self.table.get(zones, purposes)


@register("size_lookup", "by_position", reference=True,
          description="each chooser's sizes taken whole as its purpose's column of the size table, once "
                      "per batch (the same values as matching every row)")
def size_lookup_by_position(table, zone_rows, repeated, zones):
    if not np.array_equal(np.asarray(table.zones), np.asarray(zones)):
        return table  # zones not in the table's order: match every row
    return ColumnsByPosition(table, zone_rows, repeated)


@register("size_lookup", "match", agreement="exact",
          description="every row's zone and purpose matched in the size table, as 1.x's "
                      "DataFrameMatrix.get (the default until the trip destination measurement)")
def size_lookup_match(table, zone_rows, repeated, zones):
    return table


# --- trip matrices: how the trips are held while each matrix is added up -----------------


class OneTableTrips:
    """
    Every column the trips carry and the preprocessor made, in one pandas
    table, as 1.x holds them; each matrix's columns cut from it; a file's
    matrices all made, then written.
    """

    def __init__(self, cols, ids):
        df = pd.DataFrame({k: v for k, v in cols.items() if np.ndim(v)}, index=pd.Index(ids, name="trip_id"))
        for k, v in cols.items():
            if not np.ndim(v):  # a constant from the preprocessor
                df[k] = v
        self.df = df

    @property
    def columns(self) -> set[str]:
        return set(self.df.columns)

    def __len__(self) -> int:
        return len(self.df)

    def get(self, name: str) -> pd.Series:
        return self.df[name]

    def set(self, name: str, values) -> None:
        self.df[name] = values

    def keep_rows(self, mask) -> None:
        self.df = self.df[mask]

    def frame(self, names, weight: str | None) -> pd.DataFrame:
        work = self.df[list(names)].copy()  # 1.x: trips_df[[ocol, dcol, col]].copy()
        if weight:
            work[weight] = self.df[weight]
        return work


class PerMatrixTrips:
    """
    Only the columns some matrix reads (``keep``), as arrays; each matrix's
    small table (origin, destination, its value, the weight) built from them
    when that matrix is added up: the same values, types and row order as
    the table 1.x cuts from its big one. Each matrix is written to its file
    as soon as it is made (in 1.x's order), so one is in memory at a time.
    """

    write_each_matrix = True  # each matrix written once made: one in memory at a time

    def __init__(self, cols, ids, keep):
        n = len(ids)
        self.index = pd.Index(ids, name="trip_id")
        self.cols = {k: (np.asarray(v) if np.ndim(v) else np.full(n, v)) for k, v in cols.items() if k in keep}

    @property
    def columns(self) -> set[str]:
        return set(self.cols)

    def __len__(self) -> int:
        return len(self.index)

    def get(self, name: str) -> pd.Series:
        return pd.Series(self.cols[name], index=self.index, name=name)

    def set(self, name: str, values) -> None:
        self.cols[name] = np.asarray(values)

    def keep_rows(self, mask) -> None:
        mask = np.asarray(mask)
        self.cols = {k: v[mask] for k, v in self.cols.items()}
        self.index = self.index[mask]

    def frame(self, names, weight: str | None) -> pd.DataFrame:
        data = {n: self.cols[n] for n in names}
        if weight:
            data[weight] = self.cols[weight]
        return pd.DataFrame(data, index=self.index)


@register("trip_matrix_table", "per_matrix", reference=True,
          description="only the columns the matrices read; each matrix's table built from its own columns, "
                      "each matrix written once made (the same sums as one table)")
def trip_matrix_table_per_matrix(cols, ids, keep):
    return PerMatrixTrips(cols, ids, keep)


@register("trip_matrix_table", "one_table", agreement="exact",
          description="every trip column and preprocessor result in one pandas table, as 1.x (the default "
                      "until the trip matrices memory change)")
def trip_matrix_table_one_table(cols, ids, keep):
    return OneTableTrips(cols, ids)

