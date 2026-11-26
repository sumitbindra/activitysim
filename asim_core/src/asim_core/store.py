"""
The data store: tables held as plain columns.

A ``Table`` is an ID index plus named columns. Columns are NumPy arrays, or a
pandas ``Categorical`` for category-coded data. There is exactly one copy of
each table; steps add columns rather than rewriting tables, and joins are done
with index lookups (``Table.positions``) instead of building merged copies.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


class Table:
    def __init__(
        self,
        name: str,
        index: np.ndarray,
        index_name: str,
        columns: Mapping[str, object] | None = None,
    ):
        self.name = name
        self.index = np.asarray(index)
        self.index_name = index_name
        self._columns: dict[str, object] = {}
        self._lookup: pd.Index | None = None
        for k, v in (columns or {}).items():
            self.set_column(k, v)

    # --- basic access -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, column: str) -> bool:
        return column in self._columns

    def __getitem__(self, column: str):
        try:
            return self._columns[column]
        except KeyError:
            raise KeyError(f"table '{self.name}' has no column '{column}'") from None

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def set_column(self, column: str, values) -> None:
        if not isinstance(values, pd.Categorical):
            values = np.asarray(values)
        if len(values) != len(self.index):
            raise ValueError(
                f"column '{column}' has {len(values)} values, "
                f"table '{self.name}' has {len(self.index)} rows"
            )
        self._columns[column] = values

    def drop_column(self, column: str) -> None:
        del self._columns[column]

    def nbytes(self) -> int:
        total = self.index.nbytes
        for v in self._columns.values():
            total += v.nbytes
        return total

    # --- lookups ------------------------------------------------------------

    def positions(self, ids) -> np.ndarray:
        """Row positions for the given IDs. Raises if any ID is missing."""
        if self._lookup is None:
            self._lookup = pd.Index(self.index)
        pos = self._lookup.get_indexer(np.asarray(ids))
        if (pos < 0).any():
            missing = np.asarray(ids)[pos < 0][:5]
            raise KeyError(f"IDs not found in table '{self.name}': {missing.tolist()} ...")
        return pos

    # --- conversion ---------------------------------------------------------

    @classmethod
    def from_pandas(cls, name: str, df: pd.DataFrame) -> Table:
        index_name = df.index.name or "index"
        cols = {}
        for c in df.columns:
            s = df[c]
            if isinstance(s.dtype, pd.CategoricalDtype):
                cols[str(c)] = s.array  # pandas Categorical
            else:
                cols[str(c)] = s.to_numpy()
        return cls(name, df.index.to_numpy(), index_name, cols)

    def to_pandas(self, columns: Iterable[str] | None = None) -> pd.DataFrame:
        names = self.column_names if columns is None else list(columns)
        df = pd.DataFrame(
            {c: self._columns[c] for c in names},
            index=pd.Index(self.index, name=self.index_name),
        )
        return df

    def __repr__(self) -> str:
        return f"Table({self.name!r}, rows={len(self)}, columns={len(self._columns)})"


class Store:
    """All tables of a model run, by name."""

    def __init__(self, tables: Iterable[Table] = ()):
        self.tables: dict[str, Table] = {}
        for t in tables:
            self.add(t)

    def add(self, table: Table) -> None:
        self.tables[table.name] = table

    def __getitem__(self, name: str) -> Table:
        try:
            return self.tables[name]
        except KeyError:
            raise KeyError(f"no table named '{name}' in the store") from None

    def __contains__(self, name: str) -> bool:
        return name in self.tables

    def has_column(self, table: str, column: str) -> bool:
        return table in self.tables and column in self.tables[table]

    def columns(self) -> dict[str, set[str]]:
        return {name: set(t.column_names) for name, t in self.tables.items()}

    def nbytes(self) -> int:
        return sum(t.nbytes() for t in self.tables.values())

    def __repr__(self) -> str:
        return f"Store({', '.join(repr(t) for t in self.tables.values())})"
