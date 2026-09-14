"""Where things live.

Every other module goes through these functions so a test (or an agency
deployment) can relocate the example, the runs directory, or the whole harness
with environment variables instead of code changes:

- ``ASIM_HARNESS_ROOT``  repo root (default: auto-detected from this package)
- ``ASIM_EXAMPLE_DIR``   the example directory (default: ``<root>/example``)
- ``ASIM_RUNS_DIR``      where run directories go (default: ``<root>/runs``)
"""

from __future__ import annotations

import os
from pathlib import Path

EXAMPLE_NAME = "prototype_mtc"
MODEL_NAME = "prototype_mtc"

# ActivitySim 1.4 defaults: checkpoint_format=parquet, pipeline_file_name=pipeline
PIPELINE_DIRNAME = "pipeline.parquetpipeline"
CHECKPOINTS_FILENAME = "checkpoints.parquet"
LOG_RELPATH = Path("log") / "activitysim.log"
TIMING_LOG_RELPATH = Path("log") / "timing_log.csv"
OUTPUT_TABLE_PREFIX = "final_"


def root() -> Path:
    env = os.environ.get("ASIM_HARNESS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    # editable install: <root>/src/asim_harness/paths.py
    candidate = here.parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    cwd = Path.cwd().resolve()
    for p in (cwd, *cwd.parents):
        if (p / "example" / "configs").is_dir() or (p / "runs").is_dir():
            return p
    return cwd


def example_dir() -> Path:
    env = os.environ.get("ASIM_EXAMPLE_DIR")
    return Path(env).resolve() if env else root() / "example"


def configs_dir() -> Path:
    return example_dir() / "configs"


def data_dir() -> Path:
    return example_dir() / "data"


def runs_dir() -> Path:
    env = os.environ.get("ASIM_RUNS_DIR")
    return Path(env).resolve() if env else root() / "runs"


def targets_dir() -> Path:
    return root() / "targets"


def targets_file(model: str = MODEL_NAME) -> Path:
    return targets_dir() / f"{model}.yaml"


def ledger_path() -> Path:
    return runs_dir() / "index.jsonl"


def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def output_dir(run_id: str) -> Path:
    return run_dir(run_id) / "output"


# ActivitySim writes log files into output/log/ only if that folder already
# exists, otherwise straight into output/. The runner pre-creates the folder,
# but look in both places so older runs and other layouts still work.
OUTPUT_SUBDIRS = ("log", "trace")


def find_output_file(output_dir: Path, name: str, subdir: str = "log") -> Path | None:
    output_dir = Path(output_dir)
    for candidate in (output_dir / subdir / name, output_dir / name):
        if candidate.is_file():
            return candidate
    return None


def find_log_file(output_dir: Path) -> Path | None:
    return find_output_file(output_dir, LOG_RELPATH.name)


def find_timing_log(output_dir: Path) -> Path | None:
    return find_output_file(output_dir, TIMING_LOG_RELPATH.name)
