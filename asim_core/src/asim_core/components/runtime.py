"""
What generated code calls while a component runs: lookups in other tables
(``t``) and skim lookups by zone ID (``s.lookup``), going through the
step's declared needs.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..component import StepContext

NO_ZONE = -1  # 1.x's zone for "none" (NOT_IN_SKIM_ZONE_ID)


def plain(values):
    """Categoricals as plain arrays of their values, as formulas see them."""
    if isinstance(values, pd.Categorical):
        return np.asarray(values, dtype=object if values.categories.dtype == object else None)
    return values


class TableLookup:
    """
    ``t(table, column, ids)``: a column of another table for the given IDs
    (1.x's ``reindex``; an ID not in the table is an error).
    ``t.fill(table, column, ids, value)``: the same, with ``value`` for IDs
    not in the table. ``t.count(table, column, keys)``: how many rows of the
    table have each key in ``column`` (0 if none).
    """

    def __init__(self, ctx: StepContext, overrides: dict | None = None):
        self.ctx = ctx
        # table -> {column: values}: columns a step has just worked out (not in the store yet)
        self.overrides = overrides or {}

    def _column(self, table: str, column: str):
        over = self.overrides.get(table, {})
        return over[column] if column in over else self.ctx.column(table, column)

    def __call__(self, table: str, column: str, ids):
        """
        As pandas' reindex: an ID not in the table gives a missing value
        (1.x, e.g. reindex(land_use.area_type, persons.workplace_zone_id)
        for persons without a workplace, whose zone is -1).
        """
        ids = np.asarray(ids)
        if column not in self.overrides.get(table, {}):
            try:
                return plain(self.ctx.column_at(table, column, self.ctx.positions(table, ids)))
            except KeyError:
                pass
        values = pd.Series(np.asarray(plain(self._column(table, column))), index=self.ctx.ids(table))
        return values.reindex(ids).to_numpy()

    def fill(self, table: str, column: str, ids, value):
        index = pd.Index(self.ctx.ids(table))
        pos = index.get_indexer(np.asarray(ids))
        found = pos >= 0
        values = np.asarray(plain(self.ctx.column(table, column)))
        out = np.where(found, values[np.where(found, pos, 0)], value)
        return out

    def count(self, table: str, column: str, keys):
        counts = pd.Series(np.asarray(self._column(table, column))).value_counts()
        return counts.reindex(np.asarray(keys), fill_value=0).to_numpy().astype(np.int64)

    def col(self, table: str, column: str) -> np.ndarray:
        """A whole column of another table, in its row order."""
        return np.asarray(plain(self._column(table, column)))

    def count_where(self, table: str, key: str, keys, mask, as_int8: bool = True, offset: int = 0):
        """
        For each key, how many rows of the table with that key in ``key`` have
        ``mask``, as 8-bit integers (0 if none): 1.x's
        ``T.query(q).groupby(key).size().reindex(keys).fillna(0).astype(np.int8)``;
        without ``as_int8``, as pandas gives them (64-bit, floats if a key has none).
        """
        by = np.asarray(plain(self._column(table, key)))[np.asarray(mask, dtype=bool)]
        counts = pd.Series(by).value_counts() + offset  # ARC: (... .size() - 1).reindex(...)
        out = counts.reindex(np.asarray(keys)).fillna(0).to_numpy()
        return out.astype(np.int8) if as_int8 else out

    # --- 1.x's time-window functions (overlap.py), on person_windows -------------

    def _windows(self):
        if getattr(self, "_window_cache", None) is None:
            names = sorted(self.ctx.declared("person_windows"), key=int)
            windows = np.column_stack([np.asarray(self.ctx.column("person_windows", c)) for c in names])
            self._window_cache = (windows, pd.Index(self.ctx.ids("person_windows")))
        return self._window_cache

    def hh_time_window_overlap(self, table: str, household_ids) -> dict:
        from .overlap import hh_time_window_overlap

        household_ids = np.asarray(household_ids)
        persons = pd.DataFrame({"household_id": np.asarray(plain(self._column(table, "household_id"))),
                                "adult": np.asarray(plain(self._column(table, "adult")))},
                               index=pd.Index(self.ctx.ids(table), name="person_id"))
        persons = persons[persons.household_id.isin(household_ids)]  # 1.x: these households' persons
        out = hh_time_window_overlap(pd.Index(household_ids), persons, *self._windows())
        return {c: out[c].to_numpy() for c in out.columns}

    def person_time_window_overlap(self, person_ids, household_id, adult) -> dict:
        from .overlap import person_time_window_overlap

        persons = pd.DataFrame({"household_id": np.asarray(household_id), "adult": np.asarray(adult)},
                               index=pd.Index(np.asarray(person_ids), name="person_id"))
        out = person_time_window_overlap(persons, *self._windows())
        return {c: out[c].to_numpy() for c in out.columns}

    def person_time_window_overlap_of(self, table: str):
        """person_time_window_overlap for every person of a table, with their IDs."""
        from ..spec.translate import IndexedColumns
        from .overlap import person_time_window_overlap

        ids = self.ctx.ids(table)
        persons = pd.DataFrame({"household_id": np.asarray(plain(self._column(table, "household_id"))),
                                "adult": np.asarray(plain(self._column(table, "adult")))},
                               index=pd.Index(ids, name="person_id"))
        out = person_time_window_overlap(persons, *self._windows())
        return IndexedColumns(out.index.to_numpy(), {c: out[c].to_numpy() for c in out.columns})

    def view(self, table: str, mask):
        """Some rows of another table: those where ``mask`` (one value per row) holds."""
        from ..spec.translate import TableView

        return TableView(self, table, np.flatnonzero(np.asarray(mask, dtype=bool)))

    def merged_view(self, table: str, key: str, other: str, other_columns):
        """1.x's pd.merge(table[[...]], other[[...]], left_on=key, right_index=True, how='left')."""
        from ..spec.translate import TableView

        return TableView(self, table, np.arange(len(self.ctx.ids(table))),
                         {c: (other, key) for c in other_columns})

    def group_size(self, table: str, key: str, keys):
        """1.x's T.groupby(key).size().reindex(keys): rows per key, missing (a float) where none."""
        counts = pd.Series(np.asarray(plain(self._column(table, key)))).groupby(
            np.asarray(plain(self._column(table, key)))).size()
        return counts.reindex(np.asarray(keys)).to_numpy()

    def person_max_window(self, person_ids):
        from .overlap import person_max_window

        return person_max_window(np.asarray(person_ids), *self._windows()).to_numpy()

    def sum_by(self, table: str, column: str, key: str, keys):
        """
        For each key, the sum of ``table.column`` over the rows whose ``key``
        column equals it; missing where there are none. pandas' groupby sum,
        as 1.x's ``T.column.groupby(T.key).sum().reindex(keys)`` (missing
        values are skipped, as pandas does).
        """
        return self.sum_values(table, key, plain(self._column(table, column)), keys)

    def sum_values(self, table: str, key: str, values, keys):
        """``sum_by`` for values worked out from the table's columns (one per row), e.g.
        1.x's ``(persons.adult & persons.active).astype(int).groupby(persons.household_id).sum()``."""
        values = pd.Series(np.asarray(values))
        by = np.asarray(plain(self._column(table, key)))
        return values.groupby(by).sum().reindex(np.asarray(keys)).to_numpy()


class ZoneSkims:
    """``s.lookup(name, origins, destinations)``: skim values by zone ID (1.x's skim_dict.lookup)."""

    def __init__(self, ctx: StepContext):
        self.ctx = ctx

    def lookup(self, name: str, origins, destinations):
        """Zone -1 (no zone, e.g. a person without a workplace) gives a missing value, as in 1.x."""
        origins, destinations = np.asarray(origins), np.asarray(destinations)
        none = (origins == NO_ZONE) | (destinations == NO_ZONE)
        if not none.any():
            return self.ctx.skim(name, self.ctx.skim_positions(origins), self.ctx.skim_positions(destinations))
        ok = ~none
        values = self.ctx.skim(name, self.ctx.skim_positions(origins[ok]), self.ctx.skim_positions(destinations[ok]))
        out = np.full(len(origins), np.nan, dtype=values.dtype if values.dtype.kind == "f" else np.float64)
        out[ok] = values
        return out


class WrappedSkims(ZoneSkims):
    """
    ``ZoneSkims`` plus 1.x's skim wrappers without time periods, for the
    rows being annotated: ``wrappers`` maps an accessor name to the rows'
    origin and destination zone IDs (1.x: ``skim_dict.wrap("origin",
    "destination")`` as ``od_skims``). ``s(accessor, key, how)`` gets the
    skim from origin to destination, ``reverse`` from destination to origin,
    ``max`` the larger of both.
    """

    def __init__(self, ctx: StepContext, wrappers: Mapping[str, tuple]):
        super().__init__(ctx)
        self.wrappers = dict(wrappers)
        self._pairs: dict = {}

    def __call__(self, accessor: str, key: str, how: str = "get"):
        if accessor not in self._pairs:
            if accessor not in self.wrappers:
                raise KeyError(f"no skim wrapper {accessor!r} here (there are {sorted(self.wrappers)})")
            o, d = self.wrappers[accessor]
            self._pairs[accessor] = self.ctx.zone_pairs(self.ctx.skim_positions(np.asarray(o)),
                                                        self.ctx.skim_positions(np.asarray(d)))
        pairs = self._pairs[accessor]
        if how == "reverse":
            return self.ctx.skim(key, pairs.reversed())
        if how == "max":
            return np.maximum(self.ctx.skim(key, pairs), self.ctx.skim(key, pairs.reversed()))
        return self.ctx.skim(key, pairs)
