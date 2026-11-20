"""
What generated code calls while a component runs: lookups in other tables
(``t``) and skim lookups by zone ID (``s.lookup``), going through the
step's declared needs.
"""

from __future__ import annotations

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

    def sum_by(self, table: str, column: str, key: str, keys):
        """
        For each key, the sum of ``table.column`` over the rows whose ``key``
        column equals it; missing where there are none. pandas' groupby sum,
        as 1.x's ``T.column.groupby(T.key).sum().reindex(keys)`` (missing
        values are skipped, as pandas does).
        """
        values = pd.Series(np.asarray(plain(self._column(table, column))))
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
