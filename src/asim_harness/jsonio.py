"""Tiny JSON helpers: atomic writes, tolerant reads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _default(obj: Any):
    # numpy scalars and anything else with .item() / .isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def dumps(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(obj, indent=indent, default=_default, allow_nan=False)


def write_json(path: Path, obj: Any, indent: int | None = 2) -> None:
    """Write JSON atomically (temp file + rename) so readers never see a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(dumps(obj, indent=indent) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def read_json_if_exists(path: Path) -> Any | None:
    path = Path(path)
    return read_json(path) if path.exists() else None
