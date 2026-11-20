"""
Checks on the input data, before the first step runs.

Everything is checked a column at a time with NumPy, so a population of
millions takes seconds. Three sources of rules:

1. The table definitions (standard and the region's): types, ranges,
   allowed values, missing values, unique IDs, references between tables,
   counts (household size against its persons), sums.
2. The model files: every land use column used as a size term must be
   there, never negative, and each size segment must be positive somewhere
   (otherwise nothing can be chosen); the land use zones must match the skims.
3. The skims: no missing or infinite values (errors); negative values are
   reported as warnings.

Errors stop the run; warnings are reported and the run goes on. The report
lists every problem once, with how many rows, the first few examples, and
which config files mention the column. A report is kept for each version of
the inputs, so unchanged inputs are not checked again.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from .definitions import Column, Definitions, load_definitions
from .reader import InputTable, file_info, read_tables

ERROR = "error"
WARNING = "warning"
EXAMPLES = 5
SIZE_TERMS_FILE = "destination_choice_size_terms.csv"
# config files that name columns without using them in a model
NOT_MODEL_FILES = {"settings.yaml", "input_tables.yaml", "logging.yaml", "network_los.yaml"}


@dataclass
class Problem:
    severity: str
    table: str
    column: str | None
    message: str
    rows: int
    examples: list[str] = field(default_factory=list)
    used_by: list[str] = field(default_factory=list)


class InputError(ValueError):
    pass


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)
    rows_checked: dict[str, int] = field(default_factory=dict)
    undefined_columns: dict[str, list[str]] = field(default_factory=dict)
    skims_checked: int = 0
    seconds: float = 0.0
    key: str = ""
    cached: bool = False

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == ERROR]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        n_err, n_warn = len(self.errors), len(self.warnings)
        rows = ", ".join(f"{t} {n:,}" for t, n in self.rows_checked.items())
        skims = f", {self.skims_checked:,} skims" if self.skims_checked else ""
        how = "unchanged inputs, report reused" if self.cached else f"{self.seconds:.2f} s"
        return (f"Input checks: {n_err} error{'s' * (n_err != 1)}, {n_warn} warning{'s' * (n_warn != 1)}"
                f" (rows: {rows}{skims}; {how})")

    def text(self, show_undefined: bool = True) -> str:
        lines = [self.summary()]
        for p in sorted(self.problems, key=lambda p: (p.severity != ERROR,)):
            where = p.table if p.column is None else f"{p.table}.{p.column}"
            unit = "cell" if p.table == "skims" else "row"
            rows = f"{p.rows:,} {unit}{'s' * (p.rows != 1)}" if p.rows else ""
            lines.append(f"\n{p.severity.upper():7} {where}: {p.message}" + (f" -- {rows}" if rows else ""))
            for e in p.examples:
                lines.append(f"          {e}")
            if p.used_by:
                more = len(p.used_by) - 8
                shown = ", ".join(p.used_by[:8]) + (f", and {more} more" if more > 0 else "")
                lines.append(f"          used in: {shown}")
        if show_undefined and any(self.undefined_columns.values()):
            lines.append("\nColumns with no definition (describe them in input_tables.yaml):")
            for t, cols in self.undefined_columns.items():
                if cols:
                    lines.append(f"  {t}: {', '.join(cols)}")
        return "\n".join(lines)

    def raise_if_errors(self) -> None:
        if not self.ok:
            raise InputError(self.text(show_undefined=False))

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("cached")
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, text: str) -> Report:
        d = json.loads(text)
        d["problems"] = [Problem(**p) for p in d["problems"]]
        return cls(**d)


# --- helpers ------------------------------------------------------------------------


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "missing"
        return f"{v:.6g}" if not float(v).is_integer() else str(int(v))
    return str(v)


class _Labels:
    """Names rows by the table's ID column ('zone_id 8'), or by row number."""

    def __init__(self, table: str, ids: np.ndarray | None, id_name: str | None):
        self.table, self.ids, self.id_name = table, ids, id_name

    def __call__(self, rows: np.ndarray, *values: np.ndarray, sep: str = ": ") -> list[str]:
        rows = rows[:EXAMPLES]
        out = []
        for k, r in enumerate(rows):
            who = f"{self.id_name} {_fmt(self.ids[r])}" if self.ids is not None else f"row {r + 1}"
            out.append(who + sep + ", ".join(_fmt(v[k]) for v in values))
        return out


def _count_matches(keys: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """How many times each of ``ids`` appears in ``keys``."""
    keys = keys[~pd.isna(keys)] if keys.dtype.kind == "f" else keys
    if keys.dtype.kind in "iuf" and len(keys) and ids.dtype.kind in "iuf":
        k = keys.astype(np.int64, copy=False)
        lo, hi = int(k.min()), int(k.max())
        if lo >= 0 and hi < 4 * (len(k) + len(ids)) + 1_000_000:
            counts = np.bincount(k, minlength=hi + 1)
            i = ids.astype(np.int64)
            inside = (i >= 0) & (i <= hi)
            out = np.zeros(len(ids), dtype=np.int64)
            out[inside] = counts[i[inside]]
            return out
    uniq, counts = np.unique(keys, return_counts=True)
    pos = np.searchsorted(uniq, ids)
    pos = np.clip(pos, 0, max(len(uniq) - 1, 0))
    found = (uniq[pos] == ids) if len(uniq) else np.zeros(len(ids), bool)
    return np.where(found, counts[pos] if len(uniq) else 0, 0)


def _duplicated(values: np.ndarray) -> np.ndarray:
    """Rows whose value appears more than once."""
    order = np.argsort(values, kind="stable")
    s = values[order]
    same = s[1:] == s[:-1]
    dup_sorted = np.zeros(len(s), bool)
    dup_sorted[1:] |= same
    dup_sorted[:-1] |= same
    out = np.zeros(len(values), bool)
    out[order] = dup_sorted
    return out


def _isin(values: np.ndarray, allowed) -> np.ndarray:
    allowed = np.asarray(allowed)
    if values.dtype.kind == "O" or allowed.dtype.kind in "OUS":
        return pd.Series(values).isin(list(allowed)).to_numpy()
    if values.dtype.kind in "iu" and allowed.dtype.kind in "iu" and len(values) and len(allowed):
        lo = min(int(values.min()), int(allowed.min()))
        hi = max(int(values.max()), int(allowed.max()))
        if hi - lo < 50_000_000:  # a lookup table, one pass over the values
            return np.isin(values, allowed, kind="table")
    return np.isin(values, allowed)


def _where(present: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    return mask if present is None else present & mask


INT_RANGES = {t: (np.iinfo(t).min, np.iinfo(t).max) for t in
              ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")}


def _type_problem(values: np.ndarray, present: np.ndarray | None, col: Column
                  ) -> tuple[str, np.ndarray, np.ndarray] | None:
    """(message, bad rows, values as numbers) when the values don't fit the declared type."""
    t = col.type
    if t is None or t in ("str", "category"):
        return None
    if values.dtype.kind in "OUS":
        num = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
        bad = _where(present, np.isnan(num))
        if bad.any():
            return f"not numbers (the column is defined as {t})", bad, values
        values = num
    if t == "bool":
        bad = _where(present, ~_isin(values, [0, 1]))
        return ("not true/false or 0/1", bad, values) if bad.any() else None
    if t in INT_RANGES:
        v = values if present is None else np.where(present, values, 0)
        if v.dtype.kind == "f":
            bad = v != np.floor(v)
            if bad.any():
                return f"not whole numbers (the column is defined as {t})", _where(present, bad), values
        lo, hi = INT_RANGES[t]
        if len(v) and (v.min() < lo or v.max() > hi):
            return f"outside the range of {t} ({lo} to {hi})", _where(present, (v < lo) | (v > hi)), values
    return None


# --- the checks ---------------------------------------------------------------------


class _Checker:
    def __init__(self, definitions: Definitions, tables: dict[str, pa.Table],
                 specs: Sequence[InputTable], used_by):
        self.defs = definitions
        self.tables = tables
        self.specs = {s.tablename: s for s in specs}
        self.used_by = used_by
        self.problems: list[Problem] = []
        self._values: dict[tuple[str, str], np.ndarray] = {}

    def add(self, severity, table, column, message, rows=0, examples=(), used_by=None):
        if used_by is None:
            used_by = self.used_by(column) if column else []
        used_by = list(dict.fromkeys(used_by))
        self.problems.append(Problem(severity, table, column, message, int(rows), list(examples), used_by))

    def values(self, table: str, column: str, keep: bool = True) -> np.ndarray:
        """A column as NumPy. Kept for reuse (IDs, columns other rules point to)
        unless ``keep=False``, so memory holds a few columns at a time."""
        key = (table, column)
        if key in self._values:
            return self._values[key]
        arr = self.tables[table].column(column)
        if pa.types.is_dictionary(arr.type):
            arr = arr.cast(arr.type.value_type)
        values = arr.to_numpy()
        if keep:
            self._values[key] = values
        return values

    def labels(self, table: str) -> _Labels:
        spec = self.specs.get(table)
        id_col = spec.id_column if spec else None
        if id_col and id_col in self.tables[table].column_names:
            return _Labels(table, self.values(table, id_col), id_col)
        return _Labels(table, None, None)

    def run(self) -> None:
        for tname, tdef in self.defs.tables.items():
            if tname not in self.tables:
                self.add(ERROR, tname, None, "table is defined but not in settings.yaml's input_table_list",
                         used_by=[])
                continue
            names = set(self.tables[tname].column_names)
            for cname, col in tdef.columns.items():
                if cname not in names:
                    self.add(ERROR, tname, cname, "column is missing from the input")
                    continue
                self.check_column(tname, cname, col)

    def check_column(self, tname: str, cname: str, col: Column) -> None:
        arr = self.tables[tname].column(cname)
        values = self.values(tname, cname, keep=False)
        label = self.labels(tname)
        missing = None
        if arr.null_count:
            missing = np.asarray(arr.is_null().to_numpy(zero_copy_only=False))
        if values.dtype.kind == "f":
            nan = np.isnan(values)
            if nan.any():
                missing = nan if missing is None else missing | nan
        present = None if missing is None else ~missing
        if missing is not None and not col.missing_allowed:
            rows = np.flatnonzero(missing)
            self.add(ERROR, tname, cname, "missing values", len(rows), label(rows, sep=""))

        typ = _type_problem(values, present, col)
        if typ is not None:
            msg, bad, _ = typ
            rows = np.flatnonzero(bad)
            self.add(ERROR, tname, cname, msg, len(rows), label(rows, values[rows]))
            return
        if values.dtype.kind in "OUS" and col.type not in (None, "str", "category"):
            values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()

        numeric = values.dtype.kind in "iufb"
        v = values
        if numeric and len(v):
            # the column's range first; rows are looked for only when a limit is crossed
            vmin, vmax = (np.nanmin(v), np.nanmax(v)) if present is not None else (v.min(), v.max())
        rules = [("min", ERROR, lambda x, m: x < m, "below the minimum {}"),
                 ("max", ERROR, lambda x, m: x > m, "above the maximum {}"),
                 ("warn_below", WARNING, lambda x, m: x < m, "below {} (allowed, but unusual)"),
                 ("warn_above", WARNING, lambda x, m: x > m, "above {} (allowed, but unusual)")]
        for name, severity, test, text in rules:
            limit = getattr(col, name)
            if limit is None or not numeric or not len(v):
                continue
            if not test(vmin if name in ("min", "warn_below") else vmax, limit):
                continue
            with np.errstate(invalid="ignore"):
                bad = _where(present, test(v, limit))
            rows = np.flatnonzero(bad)
            self.add(severity, tname, cname, text.format(_fmt(limit)), len(rows), label(rows, v[rows]))

        if col.values is not None:
            bad = _where(present, ~_isin(v, col.values))
            if bad.any():
                rows = np.flatnonzero(bad)
                shown = ", ".join(_fmt(x) for x in col.values[:12]) + (" ..." if len(col.values) > 12 else "")
                self.add(ERROR, tname, cname, f"not one of the allowed values ({shown})", len(rows),
                         label(rows, v[rows]))

        if col.unique:
            dup = _where(present, _duplicated(v if present is None else np.where(present, v, -1)))
            if dup.any():
                rows = np.flatnonzero(dup)
                n_vals = len(np.unique(v[rows]))
                self.add(ERROR, tname, cname, f"{n_vals:,} value{'s' * (n_vals != 1)} used by more than one row"
                         " (must be unique)", len(rows), label(rows, v[rows]))

        if col.references is not None:
            t, c = col.references.split(".")
            if t in self.tables and c in self.tables[t].column_names:
                target = self.values(t, c)
                bad = _where(present, ~_isin(v, target[~pd.isna(target)]))
                if bad.any():
                    rows = np.flatnonzero(bad)
                    n_vals = len(np.unique(v[rows]))
                    self.add(ERROR, tname, cname, f"not found in {t}.{c} ({n_vals:,} different "
                             f"value{'s' * (n_vals != 1)})", len(rows), label(rows, v[rows]))

        if col.count_of is not None:
            t, c = col.count_of.split(".")
            spec = self.specs.get(tname)
            id_col = spec.id_column if spec else None
            if t in self.tables and c in self.tables[t].column_names and id_col:
                ids = self.values(tname, id_col)
                counts = _count_matches(self.values(t, c), ids)
                bad = _where(present, v != counts)
                if bad.any():
                    rows = np.flatnonzero(bad)
                    ex = [f"{id_col} {_fmt(ids[r])}: {cname} {_fmt(v[r])}, {t} rows {counts[r]}"
                          for r in rows[:EXAMPLES]]
                    self.add(ERROR, tname, cname, f"differs from the number of {t} rows with this {c}",
                             len(rows), ex)

        if col.sum_of:
            parts = [c for c in col.sum_of if c in self.tables[tname].column_names]
            if len(parts) == len(col.sum_of) and numeric:
                total = np.zeros(len(v))
                for c in parts:
                    total = total + np.nan_to_num(np.asarray(self.values(tname, c, keep=False), dtype=float))
                bad = _where(present, np.abs(v - total) > col.sum_tolerance + 1e-9)
                if bad.any():
                    rows = np.flatnonzero(bad)
                    ex = [f"{who}, sum {_fmt(t)}" for who, t in zip(label(rows, v[rows]), total[rows])]
                    self.add(WARNING, tname, cname,
                             f"differs from the sum of {', '.join(parts)} by more than {_fmt(col.sum_tolerance)}",
                             len(rows), ex)

    # --- worked out from the model files ---

    def check_size_terms(self, path: Path | None) -> None:
        if path is None or "land_use" not in self.tables:
            return
        terms = pd.read_csv(path, comment="#")
        if "segment" not in terms or "model_selector" not in terms:
            return
        # rows without a segment are comments (a quoted '#' line reads as a row)
        terms = terms[terms.segment.notna() & terms.model_selector.notna()]
        cols = [c for c in terms.columns if c not in ("model_selector", "segment")]
        weights = terms[cols].fillna(0).astype(float)
        used = [c for c in cols if (weights[c] != 0).any()]
        lu = self.tables["land_use"]
        label = self.labels("land_use")
        src = [path.stem]
        present_cols = []
        for c in used:
            segs = terms.loc[weights[c] != 0, "segment"].astype(str).tolist()
            seg_text = ", ".join(segs[:6]) + (" ..." if len(segs) > 6 else "")
            if c not in lu.column_names:
                self.add(ERROR, "land_use", c, f"is a size term ({seg_text}) but is not in the land use table",
                         used_by=src)
                continue
            present_cols.append(c)
            v = np.asarray(self.values("land_use", c), dtype=float)
            if np.isnan(v).any():
                rows = np.flatnonzero(np.isnan(v))
                self.add(ERROR, "land_use", c, f"missing values in a size term ({seg_text})", len(rows),
                         label(rows, sep=""), used_by=src + self.used_by(c))
            neg = v < 0
            if neg.any():
                rows = np.flatnonzero(neg)
                self.add(ERROR, "land_use", c, f"negative values in a size term ({seg_text}): "
                         "a zone's size must not be negative", len(rows), label(rows, v[rows]),
                         used_by=src + self.used_by(c))
        if not present_cols:
            return
        data = np.column_stack([np.nan_to_num(np.asarray(self.values("land_use", c), float))
                                for c in present_cols])
        sizes = data @ weights[present_cols].to_numpy().T  # zones x segments
        totals = np.clip(sizes, 0, None).sum(axis=0)
        for k, (_, row) in enumerate(terms.iterrows()):
            if totals[k] <= 0:
                cols_k = [c for c in present_cols if weights[c].iloc[k] != 0]
                self.add(ERROR, "land_use", None,
                         f"size of segment '{row.segment}' ({row.model_selector}) is zero in every zone, "
                         f"so no destination can be chosen for it (size terms: {', '.join(cols_k)})",
                         used_by=src)

    def check_skims(self, omx_paths: Sequence[Path], zones_recoded: bool = False) -> int:
        """
        Checks every skim value, and that the land use zones can be found in the
        skims as 1.x finds them: through the file's zone lookup if it has one;
        otherwise, with zones renumbered from 0 (``zones_recoded``), row i is the
        i-th smallest zone number, and without renumbering, zone z is row z
        counting from 1.
        """
        try:
            import tables as pytables
        except ImportError:  # optional dependency
            self.add(WARNING, "skims", None, "skims not checked: PyTables is not installed", used_by=[])
            return 0
        zones = None
        if "land_use" in self.tables:
            spec = self.specs.get("land_use")
            if spec and spec.id_column in self.tables["land_use"].column_names:
                zones = np.sort(self.values("land_use", spec.id_column))
        n_checked = 0
        shape = None
        for path in omx_paths:
            with pytables.open_file(path, "r") as h:
                lookup = None
                if "/lookup" in h:
                    nodes = h.list_nodes("/lookup")
                    if nodes:
                        lookup = np.asarray(nodes[0][:])
                for node_path in [n._v_pathname for n in h.list_nodes("/data")]:
                    node = h.get_node(node_path)
                    a = node[:]
                    node_name = node.name
                    node._f_close()  # frees its chunk cache (up to 16 MB per open matrix)
                    n_checked += 1
                    if shape is None:
                        shape = a.shape
                        self._check_skim_zones(path, shape, lookup, zones, zones_recoded)
                    elif a.shape != shape:
                        self.add(ERROR, "skims", node_name, f"is {a.shape}, other skims are {shape}", used_by=[])
                        continue
                    names = lookup if lookup is not None else (
                        np.arange(1, a.shape[0] + 1) if not zones_recoded else
                        zones if zones is not None and len(zones) == a.shape[0] else None)
                    base = node_name.split("__")[0]
                    finite = np.isfinite(a)
                    if not finite.all():
                        cells = np.argwhere(~finite)
                        self.add(ERROR, "skims", node_name, "missing or infinite values", len(cells),
                                 [_cell(names, i, j, a[i, j]) for i, j in cells[:EXAMPLES]],
                                 used_by=self.used_by(base))
                    with np.errstate(invalid="ignore"):
                        neg = a < 0
                    if neg.any():
                        cells = np.argwhere(neg)
                        self.add(WARNING, "skims", node_name, "negative values", len(cells),
                                 [_cell(names, i, j, a[i, j]) for i, j in cells[:EXAMPLES]],
                                 used_by=self.used_by(base))
        return n_checked

    def _check_skim_zones(self, path, shape, lookup, zones, zones_recoded=False) -> None:
        if zones is None:
            return
        if lookup is not None:
            # without renumbering, any order: zones are matched by number
            missing = zones[~np.isin(zones, lookup)]
            if len(missing):
                self.add(ERROR, "land_use", "zone_id", f"zones not in the skims' zone list ({path.name})",
                         len(missing), [f"zone_id {_fmt(z)}" for z in missing[:EXAMPLES]], used_by=[])
            elif zones_recoded:
                # renumbered from 0: zone i is row i, whatever the lookup says (1.x findings, #13)
                rows = lookup[: len(zones)]
                wrong = np.flatnonzero(rows != zones)
                if len(wrong):
                    self.add(ERROR, "land_use", "zone_id", f"the skims' zone lookup ({path.name}) isn't in "
                             "ascending zone order: with zones renumbered from 0, the i-th smallest zone "
                             "is read from row i, so these zones would get another zone's skims",
                             len(wrong), [f"zone_id {_fmt(zones[i])} would read row {i + 1}, zone "
                                          f"{_fmt(rows[i])} in the lookup" for i in wrong[:EXAMPLES]],
                             used_by=[])
            return
        if not zones_recoded:
            # no lookup, original numbers: 1.x reads zone z from row z (counting from 1)
            outside = zones[(zones < 1) | (zones > shape[0])]
            if len(outside):
                self.add(ERROR, "land_use", "zone_id", f"zone numbers outside 1 to {shape[0]:,}: the skims "
                         f"({path.name}) have no zone lookup, so zone z is read from row z, counting "
                         "from 1 (as in 1.x)", len(outside),
                         [f"zone_id {_fmt(z)}" for z in outside[:EXAMPLES]], used_by=[])
            elif shape[0] > len(zones):
                self.add(WARNING, "land_use", "zone_id", f"{len(zones):,} zones, the skims ({path.name}) "
                         f"{shape[0]:,}: {shape[0] - len(zones):,} skim zones have no land use, so they "
                         "can't be chosen", used_by=[])
            return
        # no lookup, zones renumbered from 0: row i is the i-th smallest zone number
        if len(zones) and zones[-1] - zones[0] + 1 != len(zones):
            gaps = np.flatnonzero(np.diff(zones) > 1)
            self.add(WARNING, "land_use", "zone_id", f"zone numbers have gaps, and the skims ({path.name}) "
                     "have no zone lookup: their rows are taken to be the zones in ascending order "
                     "(as in 1.x), which can't be checked; add a zone lookup to the skims to be sure",
                     len(gaps), [f"after zone_id {_fmt(zones[i])}: {_fmt(zones[i + 1])}" for i in gaps[:EXAMPLES]],
                     used_by=[])
        if shape[0] < len(zones):
            self.add(ERROR, "land_use", "zone_id", f"{len(zones):,} zones, but the skims ({path.name}) "
                     f"have only {shape[0]:,}", used_by=[])
        elif shape[0] > len(zones):
            # e.g. MTC's full skims: 1,475 zones, the last 21 external stations without land use
            self.add(WARNING, "land_use", "zone_id", f"{len(zones):,} zones, the skims ({path.name}) "
                     f"{shape[0]:,}: the last {shape[0] - len(zones):,} skim zones have no land use "
                     "(e.g. external stations), so they can't be chosen", used_by=[])


def _cell(names, i, j, value) -> str:
    if names is not None:
        return f"zone {_fmt(names[i])} to zone {_fmt(names[j])}: {_fmt(value)}"
    return f"row {i + 1}, column {j + 1}: {_fmt(value)}"


class _ConfigIndex:
    """Which config files mention a name (as a whole word)."""

    def __init__(self, configs_dirs: Sequence[str | Path]):
        self.words: dict[str, set[str]] = {}
        seen = set()
        for d in configs_dirs:
            for p in sorted(Path(d).glob("*")):
                if p.suffix not in (".csv", ".yaml", ".yml") or p.name in NOT_MODEL_FILES or p.name in seen:
                    continue
                seen.add(p.name)
                try:
                    text = p.read_text(errors="replace")
                except OSError:
                    continue
                for w in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)):
                    self.words.setdefault(w, set()).add(p.stem)

    def __call__(self, name: str | None) -> list[str]:
        return sorted(self.words.get(name, ())) if name else []


# --- entry point --------------------------------------------------------------------


CODE_FILES = ("checks.py", "definitions.py", "reader.py", "standard_tables.yaml")


def _code_version() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def _find(configs_dirs, name) -> Path | None:
    for d in configs_dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def check_inputs(
    configs_dirs: Sequence[str | Path],
    data_dirs: Sequence[str | Path],
    cache_dir: str | Path | None = None,
    skims: bool = True,
    definitions: Definitions | None = None,
) -> Report:
    """
    Checks the region's input tables and skims. With ``cache_dir``, a report
    is kept per version of the inputs (files, definitions, model files,
    this code) and reused while nothing has changed.
    """
    from ..skims import omx_files_from_settings
    from .reader import find_data_file, input_tables_from_settings

    t0 = time.perf_counter()
    configs_dirs = [Path(d) for d in configs_dirs]
    defs = definitions if definitions is not None else load_definitions(configs_dirs)
    specs = input_tables_from_settings(configs_dirs)
    size_terms = _find(configs_dirs, SIZE_TERMS_FILE)
    omx = omx_files_from_settings(configs_dirs, data_dirs) if skims else []

    key = ""
    cache_file = None
    if cache_dir is not None:
        files = [find_data_file(s.filename or f"{s.tablename}.csv", data_dirs) for s in specs]
        payload = {
            "code": _code_version(),
            "definitions": defs.model_dump(exclude={"source_files"}),
            "tables": [s.model_dump() for s in specs],
            "files": [file_info(p) for p in files],
            "size_terms": None if size_terms is None else hashlib.sha256(size_terms.read_bytes()).hexdigest(),
            "skims": [file_info(p) for p in omx],
            "configs": [str(d.resolve()) for d in configs_dirs],
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
        cache_file = Path(cache_dir) / f"input_checks_{key}.json"
        if cache_file.exists():
            report = Report.from_json(cache_file.read_text())
            report.cached = True
            return report

    tables, _, specs = read_tables(configs_dirs, data_dirs)
    checker = _Checker(defs, tables, specs, _ConfigIndex(configs_dirs))
    checker.run()
    checker.check_size_terms(size_terms)
    from ..skims import land_use_zones_recoded

    n_skims = checker.check_skims(omx, land_use_zones_recoded(configs_dirs)) if omx else 0

    undefined = {}
    for tname, table in tables.items():
        tdef = defs.tables.get(tname)
        defined = set(tdef.columns) if tdef else set()
        undefined[tname] = [c for c in table.column_names if c not in defined]
    report = Report(
        problems=checker.problems,
        rows_checked={t: tables[t].num_rows for t in tables},
        undefined_columns=undefined,
        skims_checked=n_skims,
        seconds=time.perf_counter() - t0,
        key=key,
    )
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(report.to_json())
    return report
