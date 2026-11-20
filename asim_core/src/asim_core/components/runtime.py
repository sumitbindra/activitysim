"""
What generated code calls while a component runs: lookups in other tables
(``t``) and skim lookups by zone ID (``s.lookup``), going through the
step's declared needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..component import StepContext


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

    def __init__(self, ctx: StepContext):
        self.ctx = ctx

    def __call__(self, table: str, column: str, ids):
        return plain(self.ctx.column_at(table, column, self.ctx.positions(table, ids)))

    def fill(self, table: str, column: str, ids, value):
        index = pd.Index(self.ctx.ids(table))
        pos = index.get_indexer(np.asarray(ids))
        found = pos >= 0
        values = np.asarray(plain(self.ctx.column(table, column)))
        out = np.where(found, values[np.where(found, pos, 0)], value)
        return out

    def count(self, table: str, column: str, keys):
        counts = pd.Series(np.asarray(self.ctx.column(table, column))).value_counts()
        return counts.reindex(np.asarray(keys), fill_value=0).to_numpy().astype(np.int64)


class ZoneSkims:
    """``s.lookup(name, origins, destinations)``: skim values by zone ID (1.x's skim_dict.lookup)."""

    def __init__(self, ctx: StepContext):
        self.ctx = ctx

    def lookup(self, name: str, origins, destinations):
        return self.ctx.skim(name, self.ctx.skim_positions(origins), self.ctx.skim_positions(destinations))
