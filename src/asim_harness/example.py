"""Read-only access to the example directory (configs, data, settings).

Nothing in this module writes under ``example/`` except ``init_example``, which
only fills in pieces that are missing (data is gitignored, so a fresh clone
needs it recreated from the ActivitySim package).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import yaml

from . import paths

EXAMPLE_PIECES = ("configs", "data", "output", "configs_mp", "README.MD")


class ExampleMissing(Exception):
    pass


def check_example() -> None:
    """Raise ExampleMissing with an actionable message if the example is incomplete."""
    missing = [
        str(p)
        for p in (paths.configs_dir() / "settings.yaml", paths.data_dir())
        if not p.exists()
    ]
    if missing:
        raise ExampleMissing(
            "example is incomplete (missing: " + ", ".join(missing) + "). "
            "Run `asim init` to recreate it from the installed ActivitySim package."
        )


def init_example() -> list[str]:
    """Recreate missing pieces of example/ from the packaged prototype_mtc.

    Returns the names of the pieces that were created. Existing pieces are
    never touched.
    """
    ex = paths.example_dir()
    missing = [p for p in EXAMPLE_PIECES if not (ex / p).exists()]
    if not missing:
        return []
    from activitysim.cli.create import get_example  # slow import, keep local

    tmp = Path(tempfile.mkdtemp(prefix="asim-init-"))
    try:
        src = Path(get_example(paths.EXAMPLE_NAME, str(tmp)))
        ex.mkdir(parents=True, exist_ok=True)
        for piece in missing:
            shutil.move(str(src / piece), str(ex / piece))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return missing


def base_settings() -> dict:
    """The example's settings.yaml as a dict (no inheritance resolution needed: it has none)."""
    with open(paths.configs_dir() / "settings.yaml") as f:
        return yaml.safe_load(f) or {}


def base_models() -> list[str]:
    return list(base_settings().get("models") or [])


def list_configs() -> list[dict]:
    """File names under example/configs with sizes, sorted, relative paths."""
    cfg = paths.configs_dir()
    out = []
    for p in sorted(x for x in cfg.rglob("*") if x.is_file()):
        out.append({"path": p.relative_to(cfg).as_posix(), "size": p.stat().st_size})
    return out


def resolve_config_path(relative_path: str) -> Path:
    """Resolve a path inside example/configs, rejecting anything that escapes it."""
    cfg = paths.configs_dir().resolve()
    rel = str(relative_path).strip()
    if not rel or rel.startswith(("/", "\\")) or ":" in rel.split("/")[0]:
        raise ValueError(f"path must be relative to example/configs: {relative_path!r}")
    target = (cfg / rel).resolve()
    if target != cfg and cfg not in target.parents:
        raise ValueError(f"path escapes example/configs: {relative_path!r}")
    if not target.is_file():
        raise FileNotFoundError(f"no such config file: {relative_path!r}")
    return target


def read_config(relative_path: str, max_bytes: int | None = None) -> str:
    target = resolve_config_path(relative_path)
    text = target.read_text(errors="replace")
    if max_bytes is not None and len(text) > max_bytes:
        text = text[:max_bytes] + f"\n... [truncated, {len(text)} chars total]"
    return text
