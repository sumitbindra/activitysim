"""Run manifests and the fingerprinting helpers they use.

A manifest is a plain dict written to ``runs/<run_id>/manifest.json``. The
fields are listed in PLAN.md (Phase 1); ``runner.py`` assembles them.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .jsonio import read_json, write_json

MANIFEST_NAME = "manifest.json"
STATUSES = ("running", "succeeded", "failed")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def manifest_path(run_dir: Path) -> Path:
    return Path(run_dir) / MANIFEST_NAME


def read_manifest(run_dir: Path) -> dict:
    return read_json(manifest_path(run_dir))


def write_manifest(run_dir: Path, manifest: dict) -> None:
    write_json(manifest_path(run_dir), manifest)


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def config_hash(dirs: Iterable[Path]) -> str:
    """sha256 over the sorted (dir position, relative path, content hash) of every file.

    ``dirs`` is given in precedence order (override dir first, base configs
    last), and the position is part of the key so moving a file between
    directories changes the hash.
    """
    entries: list[tuple[str, str]] = []
    for i, d in enumerate(dirs):
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(x for x in d.rglob("*") if x.is_file()):
            entries.append((f"{i}:{p.relative_to(d).as_posix()}", hash_file(p)))
    entries.sort()
    h = hashlib.sha256()
    for key, digest in entries:
        h.update(f"{key}\0{digest}\n".encode())
    return h.hexdigest()


def data_fingerprint(data_dir: Path) -> list[dict]:
    """name, size, mtime for every file under data_dir (hashing skims is slow)."""
    data_dir = Path(data_dir)
    rows = []
    if not data_dir.is_dir():
        return rows
    for p in sorted(x for x in data_dir.rglob("*") if x.is_file()):
        st = p.stat()
        rows.append(
            {
                "name": p.relative_to(data_dir).as_posix(),
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
    return rows


def parse_step_timings(timing_log: Path | None) -> dict[str, float]:
    """ActivitySim's timing_log.csv -> {model_name: seconds}, in run order."""
    if timing_log is None:
        return {}
    timing_log = Path(timing_log)
    if not timing_log.is_file():
        return {}
    timings: dict[str, float] = {}
    with open(timing_log, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("model_name") or "").strip()
            if not name:
                continue
            try:
                timings[name] = float(row.get("seconds") or 0.0)
            except ValueError:
                continue
    return timings


def harness_git_sha(root: Path) -> str | None:
    """HEAD sha of the harness repo, with '+dirty' if tracked files are modified."""
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        if head.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = head.stdout.strip()
    if status.returncode == 0 and status.stdout.strip():
        sha += "+dirty"
    return sha


def activitysim_version() -> str | None:
    try:
        return importlib.metadata.version("activitysim")
    except importlib.metadata.PackageNotFoundError:
        return None


def python_version() -> str:
    return platform.python_version()


def summary_of(manifest: dict) -> dict:
    """The compact view of a manifest used by `asim list`, the MCP tools and reports."""
    ov = manifest.get("settings_overrides") or {}
    return {
        "run_id": manifest.get("run_id"),
        "label": manifest.get("label"),
        "status": manifest.get("status"),
        "exit_code": manifest.get("exit_code"),
        "created_at": manifest.get("created_at"),
        "finished_at": manifest.get("finished_at"),
        "duration_s": manifest.get("duration_s"),
        "sample_size": manifest.get("sample_size"),
        "resume_after": ov.get("resume_after"),
        "parent_run_id": manifest.get("parent_run_id"),
        "models": manifest.get("models"),
        "override_files": manifest.get("override_files") or [],
        "settings_overrides": ov,
        "activitysim_version": manifest.get("activitysim_version"),
        "config_hash": manifest.get("config_hash"),
        "steps_run": len(manifest.get("step_timings") or {}),
    }
