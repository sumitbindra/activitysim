"""The runs index: ``runs/index.jsonl``, one JSON object per run, oldest first.

``asim run`` appends a row when a run starts and replaces it when the run
finishes; ``asim reindex`` rebuilds the whole file from the manifests.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: best effort, no inter-process lock
    fcntl = None

from . import paths
from .jsonio import read_json_if_exists
from .manifest import MANIFEST_NAME

LEDGER_FIELDS = (
    "run_id", "label", "created_at", "status", "exit_code", "sample_size",
    "duration_s", "resume_after", "parent_run_id", "scorecard_pass", "failed_step",
)


def row_from_manifest(manifest: dict, scorecard: dict | None = None, error: dict | None = None) -> dict:
    ov = manifest.get("settings_overrides") or {}
    return {
        "run_id": manifest.get("run_id"),
        "label": manifest.get("label"),
        "created_at": manifest.get("created_at"),
        "status": manifest.get("status"),
        "exit_code": manifest.get("exit_code"),
        "sample_size": manifest.get("sample_size", ov.get("households_sample_size")),
        "duration_s": manifest.get("duration_s"),
        "resume_after": ov.get("resume_after"),
        "parent_run_id": manifest.get("parent_run_id"),
        "scorecard_pass": (scorecard or {}).get("passed"),
        "failed_step": (error or {}).get("failed_step"),
    }


def row_for_run_dir(run_dir: Path) -> dict | None:
    run_dir = Path(run_dir)
    manifest = read_json_if_exists(run_dir / MANIFEST_NAME)
    if manifest is None:
        return None
    scorecard = read_json_if_exists(run_dir / "scorecard.json")
    error = read_json_if_exists(run_dir / "error.json")
    return row_from_manifest(manifest, scorecard, error)


def read_rows(ledger: Path | None = None) -> list[dict]:
    ledger = Path(ledger) if ledger else paths.ledger_path()
    rows: list[dict] = []
    if not ledger.is_file():
        return rows
    with open(ledger) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn line should not take the whole ledger down
    return rows


@contextlib.contextmanager
def _locked(ledger: Path):
    """Serialise writers: several runs may finish at the same time (MCP wait=False)."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger.with_name(ledger.name + ".lock"), "w") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _write_rows(rows: list[dict], ledger: Path) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger.with_name(ledger.name + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, ledger)


def append(row: dict, ledger: Path | None = None) -> None:
    ledger = Path(ledger) if ledger else paths.ledger_path()
    with _locked(ledger), open(ledger, "a") as f:
        f.write(json.dumps(row) + "\n")


def upsert(row: dict, ledger: Path | None = None) -> None:
    """Replace the row with the same run_id, or append if there is none."""
    ledger = Path(ledger) if ledger else paths.ledger_path()
    with _locked(ledger):
        rows = read_rows(ledger)
        for i, existing in enumerate(rows):
            if existing.get("run_id") == row.get("run_id"):
                rows[i] = row
                break
        else:
            rows.append(row)
        _write_rows(rows, ledger)


def reindex(runs: Path | None = None, ledger: Path | None = None) -> list[dict]:
    """Rebuild the ledger from every runs/<id>/manifest.json, oldest first."""
    runs = Path(runs) if runs else paths.runs_dir()
    ledger = Path(ledger) if ledger else paths.ledger_path()
    rows = []
    if runs.is_dir():
        for d in sorted(runs.iterdir()):
            if d.is_dir():
                row = row_for_run_dir(d)
                if row:
                    rows.append(row)
    rows.sort(key=lambda r: (r.get("created_at") or "", r.get("run_id") or ""))
    with _locked(ledger):
        _write_rows(rows, ledger)
    return rows


def list_runs(limit: int | None = 20, ledger: Path | None = None) -> list[dict]:
    """Ledger rows, newest first."""
    rows = read_rows(ledger)
    rows.reverse()
    return rows[:limit] if limit else rows


def resolve_run_id(run_id: str, runs: Path | None = None) -> str:
    """Accept an exact run id or a unique prefix; raise KeyError otherwise."""
    runs = Path(runs) if runs else paths.runs_dir()
    run_id = (run_id or "").strip()
    if not run_id:
        raise KeyError("empty run id")
    if (runs / run_id / MANIFEST_NAME).is_file():
        return run_id
    matches = sorted(
        d.name for d in runs.iterdir()
        if d.is_dir() and d.name.startswith(run_id) and (d / MANIFEST_NAME).is_file()
    ) if runs.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"no run matches {run_id!r} under {runs}")
    raise KeyError(f"run id {run_id!r} is ambiguous: {', '.join(matches)}")
