"""
Comparing the files the output steps write with 1.x's (harness code; reads
1.x's run folder):

* OMX trip matrices: every matrix (name, dtype, every cell), the zone
  mappings and the file's shape and version;
* CSV tables and the data dictionary: byte for byte, except the checkpoint
  table's timestamps (each run's own) and the times in nothing else;
* skim_usage.txt: the same lines (1.x writes them in Python set order,
  which changes from run to run);
* summarize/*.csv: byte for byte.

``compare_output_files(ours, theirs)`` returns rows of (file, status, note),
status exact / different / missing (1.x wrote it, we didn't) / extra.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

OUTPUT_PATTERNS = ("*.omx", "final_*.csv", "data_dict.csv", "data_dict.txt", "skim_usage.txt", "summarize/*.csv")


# the files each output step writes
STEP_PATTERNS = {"write_trip_matrices": ["*.omx"], "write_tables": ["final_*.csv"],
                 "write_data_dictionary": ["data_dict.csv", "data_dict.txt"], "track_skim_usage": ["skim_usage.txt"],
                 "summarize": ["summarize/*.csv"]}
# files that depend on the steps run before (when each column first appeared, which skims were read):
# only comparable in a run of the whole model, not for one step from 1.x's saved state
HISTORY_FILES = {"final_checkpoints.csv", "data_dict.csv", "data_dict.txt", "skim_usage.txt"}


def _files(folder: Path, patterns=OUTPUT_PATTERNS) -> dict[str, Path]:
    folder = Path(folder)
    return {str(p.relative_to(folder)): p for pattern in patterns for p in folder.glob(pattern)}


def compare_output_files(ours: Path, theirs: Path, steps=None, skip=()) -> list[tuple[str, str, str]]:
    """``steps``: only these output steps' files (default: every output file); ``skip``: file names left out."""
    patterns = OUTPUT_PATTERNS if steps is None else [p for s in steps for p in STEP_PATTERNS.get(s, [])]
    mine, ref = _files(ours, patterns), _files(theirs, patterns)
    rows = []
    for name in sorted(set(mine) | set(ref)):
        if name in skip:
            continue
        if name not in mine:
            rows.append((name, "missing", "1.x wrote it; this run didn't"))
        elif name not in ref:
            rows.append((name, "extra", "not in 1.x's run"))
        else:
            rows.append((name, *_compare(mine[name], ref[name])))
    return rows


def _compare(a: Path, b: Path) -> tuple[str, str]:
    if a.suffix == ".omx":
        return _compare_omx(a, b)
    if a.name == "skim_usage.txt":
        la = a.read_text(encoding="utf-8", errors="replace").splitlines()  # 1.x writes text files in the system encoding
        lb = b.read_text(encoding="utf-8", errors="replace").splitlines()
        if sorted(la) == sorted(lb):
            return "exact", f"{len(la) - 2} skims, the same lines (1.x's order varies by run)"
        return "different", f"lines only here: {sorted(set(la) - set(lb))[:4]}; only in 1.x: {sorted(set(lb) - set(la))[:4]}"
    if a.name == "final_checkpoints.csv":
        ra, rb = _without(a, "timestamp"), _without(b, "timestamp")
        if ra == rb:
            return "exact", f"{len(ra) - 1} steps (timestamps aside)"
        return "different", _first_difference([",".join(r) for r in ra], [",".join(r) for r in rb])
    da, db = a.read_bytes(), b.read_bytes()
    if da == db:
        return "exact", f"byte for byte ({len(da):,} bytes)"
    return "different", _first_difference(da.decode().splitlines(), db.decode().splitlines())


def _without(path: Path, column: str) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8", errors="replace"))))
    if not rows or column not in rows[0]:
        return rows
    i = rows[0].index(column)
    return [r[:i] + r[i + 1:] for r in rows]


def _first_difference(la: list[str], lb: list[str]) -> str:
    first = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
    ours = la[first][:160] if first < len(la) else "<end>"
    theirs = lb[first][:160] if first < len(lb) else "<end>"
    return f"{len(la)} vs {len(lb)} lines; first difference at line {first + 1}: ours {ours!r}, 1.x {theirs!r}"


def _compare_omx(a: Path, b: Path) -> tuple[str, str]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from asim_core.omx import read_omx

    ma, pa, aa = read_omx(a)
    mb, pb, ab = read_omx(b)
    problems = []
    if sorted(ma) != sorted(mb):
        problems.append(f"matrices only here {sorted(set(ma) - set(mb))[:3]}, only in 1.x {sorted(set(mb) - set(ma))[:3]}")
    differ = [k for k in set(ma) & set(mb) if ma[k].dtype != mb[k].dtype or not np.array_equal(ma[k], mb[k])]
    if differ:
        k = sorted(differ)[0]
        cells = int((ma[k] != mb[k]).sum()) if ma[k].shape == mb[k].shape else "shape"
        problems.append(f"{len(differ)} matrices differ (e.g. {k}: {cells} cells)")
    if sorted(pa) != sorted(pb) or any(pa[k].dtype != pb[k].dtype or not np.array_equal(pa[k], pb[k])
                                       for k in set(pa) & set(pb)):
        problems.append(f"zone mappings differ ({sorted(pa)} vs {sorted(pb)})")
    if not np.array_equal(aa.get("SHAPE"), ab.get("SHAPE")) or aa.get("OMX_VERSION") != ab.get("OMX_VERSION"):
        problems.append("shape or OMX version differ")
    if problems:
        return "different", "; ".join(problems)
    total = sum(float(m.sum()) for m in mb.values())
    return "exact", f"{len(mb)} matrices, every cell (total {total:,.2f}), mapping {sorted(pb)}"



def as_comparisons(rows, step: str):
    """The rows as the harness's column comparisons (table '(file)', the file name as the column)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from asim_core.compare import ColumnComparison

    return [ColumnComparison(step=step, table="(file)", column=name, rows=1, mismatched=int(status != "exact"),
                             max_abs_diff=None, status=status, dtype="", reference_dtype="", note=note)
            for name, status, note in rows]
