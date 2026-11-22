"""
The output steps of 1.x's ``core/steps/output.py``: ``write_tables``,
``write_data_dictionary`` and ``track_skim_usage``.

Writing the tables (``write_tables``), as
settings.yaml's ``output_tables`` says: which tables (``action`` include
or skip, ``tables``), the file names' ``prefix``, CSV or Parquet
(``file_type``). Each table goes through PyArrow as 1.x writes it: its
columns in order, then its row IDs as the last column. With
``recode_pipeline_columns``, ``decode_columns`` turn zone positions back
into the input's zone IDs (``land_use.zone_id``; ``nonnegative |`` keeps
-1), or period positions into labels (``time_period``); the decoded column
moves to the end, as in 1.x.

``checkpoints`` is the run's step list as 1.x's checkpoint table: per step,
when it finished, and for every table the last step that wrote it. Its
times are this run's, so they never match another run's.

Not supported yet (reported before the run): HDF5 output (``h5_store``,
``file_type: h5``) and ``sort``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, StepContext
from ..skims import time_periods_from_settings
from ..spec.legacy import find_config_file, load_top_settings, read_settings, read_yaml

STEP = "write_tables"
CHECKPOINTS = "checkpoints"


def write_tables(configs_dirs: Sequence[str | Path]) -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    top = load_top_settings(configs_dirs)
    out = top.get("output_tables")
    if out is None:  # 1.x writes nothing
        return Component(STEP, lambda ctx: {}, provides={}, description="no output_tables: nothing written")
    action = out.get("action")
    if action not in ("include", "skip"):
        raise ValueError(f"settings.yaml: output_tables action '{action}' must be include or skip")
    file_type = out.get("file_type", "csv")
    unsupported = []
    if out.get("h5_store") or file_type == "h5":
        unsupported.append("HDF5 output (h5_store, file_type: h5)")
    elif file_type not in ("csv", "parquet"):
        raise ValueError(f"settings.yaml: output_tables file_type '{file_type}' must be csv, parquet or h5")
    if out.get("sort"):
        unsupported.append("output_tables sort")
    if unsupported:
        raise NotImplementedError(f"settings.yaml: not supported yet: {', '.join(unsupported)}")
    prefix = out.get("prefix", "final_")
    listed = []  # (table, {column: decode instruction})
    for t in out.get("tables") or []:
        if isinstance(t, str):
            listed.append((t, {}))
        else:
            listed.append((t["tablename"], dict(t.get("decode_columns") or {})))
    recode = bool(top.get("recode_pipeline_columns", False))
    period_labels = None
    if recode and any(v.split("|")[-1].strip() == "time_period" for _, d in listed for v in d.values()):
        period_labels = list(time_periods_from_settings(configs_dirs).unique_labels)
    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]):
        names = [t for t in available if t != SKIMS]
        if action == "include":
            chosen = listed if out.get("tables") is not None else [(t, {}) for t in names]
        else:
            skip = {t for t, _ in listed}
            chosen = [(t, {}) for t in names if t not in skip]
        resolved["chosen"] = chosen
        needs = {}
        for table, decode in chosen:
            if table == CHECKPOINTS or table not in available:
                continue  # 1.x: skipped with a warning if not there
            needs[table] = set(available[table])
            if recode:
                for instruction in decode.values():
                    source = instruction.split("|")[-1].strip()
                    if "." in source:
                        src_table, col = source.split(".")
                        if src_table not in available:
                            raise ValueError(f"decode_columns: no table '{src_table}' ({instruction})")
                        needs.setdefault(src_table, set()).update(
                            c for c in (col, f"_original_{col}") if c in available[src_table])
        return {t: sorted(c) for t, c in needs.items()}

    def run(ctx: StepContext):
        for table, decode in resolved["chosen"]:
            if table == CHECKPOINTS:
                frame = checkpoint_table(ctx.history)
            elif table not in ctx.table_names():
                ctx.warn(f"skipping '{table}': table not found")
                continue
            else:
                frame = None
            write_table(ctx, table, frame, decode if recode else {}, prefix, file_type, period_labels)
        return {}

    return Component(STEP, run, provides={}, resolve_needs=resolve_needs,
                     description=f"output tables as {file_type}, prefix '{prefix}'")


def checkpoint_table(history) -> pd.DataFrame:
    """1.x's checkpoint inventory: per step, its time and each table's last writer ("" before it exists)."""
    tables: list[str] = []
    rows = []
    last: dict[str, str] = {}
    for record in history:
        for t in record.tables:
            if t not in tables:
                tables.append(t)
            last[t] = record.step
        rows.append({"checkpoint_name": record.step, "timestamp": record.finished,
                     **{t: last.get(t, "") for t in tables}})
    frame = pd.DataFrame(rows, columns=["checkpoint_name", "timestamp", *tables])
    frame[tables] = frame[tables].fillna("")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).astype("datetime64[ns]")
    frame.index = pd.Index(np.arange(len(frame), dtype=np.int64))  # 1.x's index is written as a column
    return frame


def write_table(ctx: StepContext, table: str, frame: pd.DataFrame | None, decode: Mapping[str, str], prefix: str,
                file_type: str, period_labels) -> None:
    import pyarrow as pa
    from pyarrow import csv, parquet

    if frame is None:
        frame = ctx.table_frame(table)
    dt = pa.Table.from_pandas(frame, preserve_index=True)
    for colname, instruction in decode.items():
        decode_filter, source = (s.strip() for s in instruction.split("|")) if "|" in instruction \
            else (None, instruction.strip())
        if source == "time_period":
            map_col = period_labels
            if colname in dt.column_names:
                dt = dt.drop([colname]).append_column(colname, pa.array(_decoded(dt.column(colname), map_col,
                                                                                    decode_filter)))
            continue
        if "." in source:
            src_table, lookup_col = source.split(".")
            parent = pa.Table.from_pandas(ctx.table_frame(src_table), preserve_index=True)
        else:
            src_table, lookup_col, parent = table, source, dt
        name = f"_original_{lookup_col}" if f"_original_{lookup_col}" in parent.column_names else lookup_col
        map_col = np.asarray(parent.column(name))
        if colname in dt.column_names:
            dt = dt.drop([colname]).append_column(colname, pa.array(_decoded(dt.column(colname), map_col,
                                                                            decode_filter)))
        if src_table == table and f"_original_{lookup_col}" in dt.column_names:
            dt = dt.drop([f"_original_{lookup_col}"])  # 1.x: duplicates the decoded column
    path = ctx.output_path(f"{prefix}{table}.{file_type}")
    if file_type == "csv":
        csv.write_csv(dt, path)
    else:
        parquet.write_table(dt, path)


def _decoded(column, map_col, decode_filter):
    """1.x's _decode_output_column: positions (as int) mapped to map_col's values; nonnegative keeps -1."""
    series = pd.Series(column)
    if decode_filter is None:
        map_func = map_col.__getitem__
    elif decode_filter in ("nonnegative", "nullable_nonnegative"):
        def map_func(x):
            return x if x < 0 else map_col[x]
    else:
        raise ValueError(f"unknown decode filter {decode_filter}")
    if decode_filter == "nullable_nonnegative":
        revised = series.copy()
        notna = series.notna()
        revised.loc[notna] = series.loc[notna].astype(int).map(map_func)
        return revised
    return series.astype(int).map(map_func)


# --- the data dictionary -----------------------------------------------------------------


def write_data_dictionary(configs_dirs: Sequence[str | Path],
                          settings_file: str = "write_data_dictionary.yaml") -> Component:
    """
    1.x's write_data_dictionary: each table's columns (row IDs first), dtype
    and the step that first wrote them (from the run's steps), as a CSV
    (``csv_format``, default data_dict.csv) and a text file (``txt_format``,
    default data_dict.txt); ``tables`` limits it to those tables.
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    try:
        settings = read_settings(configs_dirs, settings_file)
    except FileNotFoundError:
        settings = {}  # 1.x: the settings file is optional
    txt_format = settings.get("txt_format", "data_dict.txt")
    csv_format = settings.get("csv_format", "data_dict.csv")
    wanted = settings.get("tables")
    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]):
        names = [t for t in available if t != SKIMS]
        if wanted:
            names = [t for t in wanted if t in names]
        resolved["tables"] = names
        return {t: sorted(available[t]) for t in names}

    def run(ctx: StepContext):
        if not (csv_format or txt_format):
            ctx.warn("neither txt_format nor csv_format: nothing written")
            return {}
        tables = resolved["tables"] if wanted else [t for t in ctx.table_names() if t in resolved["tables"]]
        schema, shapes = data_dictionary(ctx, tables)
        schema_df = pd.concat(schema.values())
        if csv_format:
            schema_df.to_csv(ctx.output_path(csv_format), header=True, index=False)
        if txt_format:
            col_width = {c: int(schema_df[c].str.len().fillna(0).max()) + 2 for c in schema_df}
            with open(ctx.output_path(txt_format), "w") as f:
                for table in tables:  # 1.x: the order the tables were made in, or the settings' list
                    info = schema.get(table)
                    if info is None:
                        continue
                    info = info[["column_name", "dtype", "checkpoint"]].copy()
                    for c in info:  # one width per column across every table, as 1.x
                        info[c] = info[c].str.pad(col_width[c], side="right")
                    info.columns = [c.ljust(col_width[c]) for c in info.columns]
                    print(f"###\n### {table} {shapes[table]}\n###\n", file=f)
                    print(f"{info.to_string(index=False)}\n", file=f)
        return {}

    return Component("write_data_dictionary", run, provides={}, resolve_needs=resolve_needs,
                     description="each table's columns, dtypes and the step that first wrote them")


def data_dictionary(ctx: StepContext, tables: Sequence[str]):
    """({table: frame of table_name, column_name, dtype, checkpoint}, {table: shape}), as 1.x."""
    schema, shapes = {}, {}
    for table in tables:
        df = ctx.table_frame(table)
        shapes[table] = df.shape
        if df.index.name and df.index.name not in df.columns:
            df = df.reset_index()
        info = df.dtypes.astype(str).to_frame("dtype").reset_index().rename(columns={"index": "column_name"})
        info["checkpoint"] = ""
        info.insert(loc=0, column="table_name", value=table)
        schema[table] = info
    # each column's first step: the first step after which the table had it
    for record in ctx.history:
        for table, columns in record.tables.items():
            info = schema.get(table)
            if info is None:
                continue
            index_name = ctx.index_name(table)
            columns = ([index_name] if index_name and index_name not in columns else []) + list(columns)
            prev = info[info.checkpoint != ""].column_name.values
            new = [c for c in columns if c not in prev]
            info.checkpoint = np.where(info.column_name.isin(new), record.step, info.checkpoint)
    return schema, shapes


# --- skim usage --------------------------------------------------------------------------


def track_skim_usage(configs_dirs: Sequence[str | Path] = ()) -> Component:
    """
    1.x's track_skim_usage: skim_usage.txt lists the skims the run looked up,
    then the skim file's names it didn't. As 1.x's skim_dict records them: a
    lookup by each row's period gives the skim's name (SOV_TIME), one
    period's skim named in full gives ('SOV_TIME', 'MD'), whose name SOV_TIME
    is then listed as not looked up unless another lookup used it. 1.x writes
    them in Python set order, which differs from run to run; here sorted.
    """

    def run(ctx: StepContext):
        used = {_skim_key_text(s) for record in ctx.history for s in record.skims}
        names = {n.rsplit("__", 1)[0] if "__" in n else n for n in ctx.skim_names()}  # 1.x's base keys
        with open(ctx.output_path("skim_usage.txt"), "w") as f:
            print("\n### skim_dict usage", file=f)
            for key in sorted(used):
                print(key, file=f)
            for key in sorted(names - used):
                print(key, file=f)
        return {}

    return Component("track_skim_usage", run, provides={}, resolve_needs=lambda available: {},
                     description="the skims the run read, and those it didn't")


def _skim_key_text(name: str) -> str:
    """'SOV_TIME__AM' -> "('SOV_TIME', 'AM')" (1.x's key for a time-of-day skim); 'DIST' stays."""
    if "__" in name:
        key, period = name.rsplit("__", 1)
        return str((key, period))
    return name

