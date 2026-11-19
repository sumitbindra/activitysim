"""
Read the state ActivitySim 1.x saves after every step.

ActivitySim 1.x (with ``checkpoint_format: parquet``) writes a folder
``pipeline.parquetpipeline`` containing:

* ``checkpoints.parquet``: one row per step, in run order. For each table it
  names the step whose saved version is current at that point.
* ``<table>/<step>.parquet``: the table as saved after ``<step>``.

This module only reads those files; it does not import ActivitySim.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .store import Store, Table

_NON_TABLE_COLUMNS = {"checkpoint_name", "timestamp"}


class Asim1State:
    def __init__(self, path: str | Path):
        path = Path(path)
        if path.is_dir() and path.suffix != ".parquetpipeline":
            candidate = path / "pipeline.parquetpipeline"
            if candidate.exists():
                path = candidate
        if not (path / "checkpoints.parquet").exists():
            raise FileNotFoundError(f"no ActivitySim 1.x saved state found at {path}")
        self.path = path
        self.checkpoints = pd.read_parquet(path / "checkpoints.parquet")
        self.steps: list[str] = self.checkpoints["checkpoint_name"].tolist()
        self.table_names = [c for c in self.checkpoints.columns if c not in _NON_TABLE_COLUMNS]

    def _row(self, step: str) -> pd.Series:
        if step not in self.steps:
            raise KeyError(f"step '{step}' is not in the saved state (steps: {self.steps})")
        return self.checkpoints.loc[self.checkpoints["checkpoint_name"] == step].iloc[0]

    def previous_step(self, step: str) -> str | None:
        i = self.steps.index(step)
        return self.steps[i - 1] if i > 0 else None

    def tables_at(self, step: str) -> dict[str, str]:
        """{table: step whose saved version is current} for tables existing after ``step``."""
        row = self._row(step)
        return {t: row[t] for t in self.table_names if isinstance(row[t], str) and row[t]}

    def written_at(self, step: str) -> list[str]:
        """Tables that ``step`` created or changed."""
        return [t for t, saved in self.tables_at(step).items() if saved == step]

    def read(self, table: str, step: str) -> pd.DataFrame:
        """A table as it was after ``step``."""
        saved = self.tables_at(step).get(table)
        if saved is None:
            raise KeyError(f"table '{table}' does not exist after step '{step}'")
        return pd.read_parquet(self.path / table / f"{saved}.parquet")

    def columns_at(self, step: str) -> dict[str, set[str]]:
        """Column names of every table after ``step`` (reads only file metadata)."""
        import pyarrow.parquet as pq

        out = {}
        for table, saved in self.tables_at(step).items():
            schema = pq.read_schema(self.path / table / f"{saved}.parquet")
            index_cols = set()
            meta = schema.pandas_metadata or {}
            for ic in meta.get("index_columns", []):
                if isinstance(ic, str):
                    index_cols.add(ic)
            out[table] = {n for n in schema.names if n not in index_cols}
        return out

    def load_store(self, after_step: str, tables: list[str] | None = None) -> Store:
        """All tables (or the listed ones) as they were after ``after_step``."""
        store = Store()
        for table in tables or list(self.tables_at(after_step)):
            store.add(Table.from_pandas(table, self.read(table, after_step)))
        return store
