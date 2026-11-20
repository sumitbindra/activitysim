"""
Reading the input tables.

Which file holds each table, and how its columns are renamed, comes from
the region's settings.yaml (``input_table_list``), exactly as 1.x reads it:
``filename``, ``index_col``, ``drop_columns``, ``rename_columns``,
``keep_columns``, ``recode_columns``. Files are read with PyArrow (CSV,
compressed CSV or Parquet), only the columns that are kept.

``read_tables`` gives the tables as the checks see them: renamed, kept
columns plus the ID, original values (nothing recoded). ``to_pandas`` gives
what 1.x's reader gives: the ID as the index and recodes applied.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

# 1.x's canonical ID columns (activitysim.abm.models.util.canonical_ids)
CANONICAL_IDS = {
    "households": "household_id",
    "persons": "person_id",
    "tours": "tour_id",
    "joint_tour_participants": "participant_id",
    "trips": "trip_id",
    "land_use": "zone_id",
}

NOTSET = "NOTSET"

# pandas' default missing-value markers (1.x reads with pandas), so an empty
# text cell is missing here too, not ""
PANDAS_NA_VALUES = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
]


class InputTable(BaseModel):
    """One entry of settings.yaml's ``input_table_list``."""

    model_config = ConfigDict(extra="forbid")

    tablename: str
    filename: str | None = None
    index_col: str | None = NOTSET
    rename_columns: dict[str, str] | None = None
    recode_columns: dict[str, str] | None = None
    keep_columns: list[str] | None = None
    drop_columns: list[str] | None = None
    h5_tablename: str | None = None
    dtypes: dict[str, str] | None = None

    @property
    def id_column(self) -> str | None:
        """The ID column, as 1.x settles it (a canonical table gets its canonical ID)."""
        if self.index_col != NOTSET:
            return self.index_col
        return CANONICAL_IDS.get(self.tablename)


def input_tables_from_settings(configs_dirs: Sequence[str | Path]) -> list[InputTable]:
    from ..spec.legacy import load_top_settings

    settings = load_top_settings(configs_dirs)
    return [InputTable(**t) for t in settings.get("input_table_list") or []]


def find_data_file(name: str, data_dirs: Sequence[str | Path]) -> Path:
    """The first data folder that has the file (or its .csv.gz / .parquet form), as in 1.x."""
    base = Path(name)
    candidates = [base]
    if base.suffix == ".csv":
        candidates += [base.with_suffix(".csv.gz"), base.with_suffix(".parquet")]
    for d in data_dirs:
        for c in candidates:
            p = Path(d) / c
            if p.is_file():
                return p
    raise FileNotFoundError(f"input file '{name}' not found in {[str(d) for d in data_dirs]}")


def file_info(path: Path) -> dict:
    st = os.stat(path)
    return {"path": str(Path(path).resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _wanted_raw_columns(spec: InputTable, raw_names: list[str]) -> list[str] | None:
    """The file's columns that end up in the table, or None for all."""
    if not spec.keep_columns:
        return None
    rename = spec.rename_columns or {}
    drop = set(spec.drop_columns or [])
    wanted = set(spec.keep_columns) | {spec.id_column} | set(_recode_original_names(spec))
    return [c for c in raw_names if c not in drop and rename.get(c, c) in wanted]


def _recode_original_names(spec: InputTable) -> list[str]:
    return list((spec.recode_columns or {}).keys())


def _csv_header(path: Path, encoding: str) -> list[str]:
    reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(encoding=encoding, block_size=1 << 16))
    return reader.schema.names


def _read_csv_like_asim1(path: Path, spec: InputTable) -> pa.Table:
    """pandas' reader, as 1.x calls it: the same floats to the last bit."""
    for encoding in ("utf8", "cp1252"):
        try:
            header = pd.read_csv(path, comment="#", nrows=0, encoding=encoding).columns.tolist()
            df = pd.read_csv(path, comment="#", encoding=encoding, low_memory=False,
                             usecols=_wanted_raw_columns(spec, header), dtype=spec.dtypes)
            return pa.Table.from_pandas(df[[c for c in header if c in df.columns]], preserve_index=False)
        except UnicodeDecodeError:
            if encoding == "cp1252":
                raise
    raise AssertionError("unreachable")


def _read_file(path: Path, spec: InputTable, like_asim1: bool = False) -> pa.Table:
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        raw = pq.read_schema(path).names
        return pq.read_table(path, columns=_wanted_raw_columns(spec, raw))
    if name.endswith((".h5", ".hdf5")):
        df = pd.read_hdf(path, spec.h5_tablename or spec.tablename)
        return pa.Table.from_pandas(df, preserve_index=False)
    if like_asim1:
        return _read_csv_like_asim1(path, spec)
    for encoding in ("utf8", "cp1252"):  # 1.x falls back to cp1252
        try:
            raw = _csv_header(path, encoding)
            convert = pacsv.ConvertOptions(include_columns=_wanted_raw_columns(spec, raw),
                                           null_values=PANDAS_NA_VALUES, strings_can_be_null=True)
            return pacsv.read_csv(path, read_options=pacsv.ReadOptions(encoding=encoding),
                                  convert_options=convert)
        except (pa.ArrowInvalid, UnicodeDecodeError) as e:
            if encoding == "cp1252" or "utf8" not in str(e).lower().replace("-", ""):
                raise
    raise AssertionError("unreachable")


class ReadProblem(ValueError):
    pass


def read_table(spec: InputTable, data_dirs: Sequence[str | Path], like_asim1: bool = False
               ) -> tuple[pa.Table, Path]:
    """
    One input table: renamed, its kept columns and ID, values as in the file.

    CSV numbers are read correctly rounded (the closest float to the text).
    pandas' default parser, which 1.x uses, is not: some values differ in
    the last few bits (MTC's land use: 22 of 250). ``like_asim1=True`` reads with
    pandas, for results identical to 1.x's.
    """
    path = find_data_file(spec.filename or f"{spec.tablename}.csv", data_dirs)
    table = _read_file(path, spec, like_asim1)
    names = [n for n in table.column_names]
    if spec.drop_columns:
        keep = [i for i, n in enumerate(names) if n not in set(spec.drop_columns)]
        table = table.select(keep)
        names = table.column_names
    if spec.rename_columns:
        table = table.rename_columns([spec.rename_columns.get(n, n) for n in names])
    names = table.column_names
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ReadProblem(f"{spec.tablename} ({path.name}): columns named more than once after renaming: {dup}")
    id_col = spec.id_column
    if id_col is not None and id_col not in names:
        raise ReadProblem(f"{spec.tablename} ({path.name}): ID column '{id_col}' not in the file "
                          f"(columns after renaming: {names})")
    if spec.keep_columns:
        missing = [c for c in spec.keep_columns if c not in names]
        if missing:
            raise ReadProblem(f"{spec.tablename} ({path.name}): keep_columns not in the file: {missing}")
        order = ([id_col] if id_col else []) + [c for c in spec.keep_columns if c != id_col]
        table = table.select(order)
    return table, path


def table_columns(spec: InputTable, data_dirs: Sequence[str | Path], recode: bool = True) -> list[str]:
    """
    The columns ``to_pandas`` gives this table (the ID is the index, not a
    column), worked out from the file's header and the settings, without
    reading the data. With ``recode``, zero-based recodes add
    ``_original_<column>``.
    """
    path = find_data_file(spec.filename or f"{spec.tablename}.csv", data_dirs)
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        names = pq.read_schema(path).names
    elif name.endswith((".h5", ".hdf5")):
        names = list(pd.read_hdf(path, spec.h5_tablename or spec.tablename, stop=0).columns)
    else:
        for encoding in ("utf8", "cp1252"):
            try:
                names = pd.read_csv(path, comment="#", nrows=0, encoding=encoding).columns.tolist()
                break
            except UnicodeDecodeError:
                if encoding == "cp1252":
                    raise
    wanted = _wanted_raw_columns(spec, names)
    if wanted is not None:
        names = [n for n in names if n in set(wanted)]
    if spec.drop_columns:
        names = [n for n in names if n not in set(spec.drop_columns)]
    if spec.rename_columns:
        names = [spec.rename_columns.get(n, n) for n in names]
    id_col = spec.id_column
    if spec.keep_columns:
        names = ([id_col] if id_col else []) + [c for c in spec.keep_columns if c != id_col]
    if recode:
        for col, how in (spec.recode_columns or {}).items():
            if how == "zero-based":
                names.append(f"_original_{col}")
    return [n for n in names if n != id_col]


def read_tables(configs_dirs: Sequence[str | Path], data_dirs: Sequence[str | Path], like_asim1: bool = False
                ) -> tuple[dict[str, pa.Table], dict[str, Path], list[InputTable]]:
    specs = input_tables_from_settings(configs_dirs)
    tables, paths = {}, {}
    for spec in specs:
        tables[spec.tablename], paths[spec.tablename] = read_table(spec, data_dirs, like_asim1)
    return tables, paths, specs


def to_pandas(tables: dict[str, pa.Table], specs: Sequence[InputTable], recode: bool = True
              ) -> dict[str, pd.DataFrame]:
    """
    The tables as 1.x's reader returns them: the ID as the index and, when
    ``recode`` (1.x's ``recode_pipeline_columns``), the recodes applied, in
    table order.
    """
    out: dict[str, pd.DataFrame] = {}
    for spec in recode_order(specs):
        df = tables[spec.tablename].to_pandas()
        if recode and spec.recode_columns:
            for col, how in spec.recode_columns.items():
                if how == "zero-based":
                    original = df[col].to_numpy()
                    uniq = np.unique(original)
                    df[f"_original_{col}"] = original
                    df[col] = np.searchsorted(uniq, original)
                else:
                    parent_name, lookup = how.split(".")
                    parent = out[parent_name]
                    key = parent[f"_original_{lookup}"] if f"_original_{lookup}" in parent else parent[lookup]
                    df[col] = df[col].map(pd.Series(parent.index, index=key.to_numpy()))
        id_col = spec.id_column
        if id_col is not None:
            df[id_col] = df[id_col].astype(np.int64)
            df = df.set_index(id_col)
        out[spec.tablename] = df
    return {s.tablename: out[s.tablename] for s in specs}


def recode_order(specs: Sequence[InputTable]) -> list[InputTable]:
    """Tables ordered so that a table recoded from another comes after it (1.x
    reads land use first, in its own step)."""
    by_name = {s.tablename: s for s in specs}
    done: list[InputTable] = []

    def visit(spec: InputTable, path: tuple = ()):
        if spec in done:
            return
        if spec.tablename in path:
            raise ReadProblem(f"recode_columns refer to each other: {' -> '.join(path)}")
        for how in (spec.recode_columns or {}).values():
            parent = how.split(".")[0] if how != "zero-based" else None
            if parent in by_name:
                visit(by_name[parent], path + (spec.tablename,))
        done.append(spec)

    for spec in specs:
        visit(spec)
    return done
