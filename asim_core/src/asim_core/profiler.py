"""
Per-step runtime and memory profiling.

The same profiler is used for asim_core runs and for ActivitySim 1.x runs
(see ``harness/run_asim1.py``), so the numbers are directly comparable.

For each step it records:

* ``wall_s``      wall-clock seconds
* ``cpu_s``       CPU seconds used by this process (can exceed wall time when
                  libraries use several threads)
* ``rss_before_mb`` memory held by the process when the step started
* ``peak_mb``     highest memory held by the process during the step
* ``step_extra_mb`` ``peak_mb - rss_before_mb``: how much extra memory the
                  step needed on top of what was already held
* ``rss_after_mb`` memory still held when the step finished

Exact per-step peaks use Linux's ``/proc`` interface (the peak counter is reset
before each step). On other systems the memory columns are left empty for now.
Only the Python standard library is used.
"""

from __future__ import annotations

import csv
import gc
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path

_PROC_STATUS = "/proc/self/status"
_PROC_CLEAR_REFS = "/proc/self/clear_refs"


def _read_status_mb(key: str) -> float | None:
    try:
        with open(_PROC_STATUS) as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except OSError:
        return None
    return None


def current_rss_mb() -> float | None:
    """Memory currently held by this process (resident set size), in MB."""
    return _read_status_mb("VmRSS")


def peak_rss_mb() -> float | None:
    """Highest memory held since the last ``reset_peak()``, in MB."""
    return _read_status_mb("VmHWM")


def reset_peak() -> bool:
    """Reset the peak-memory counter. Returns False if not supported."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open(_PROC_CLEAR_REFS, "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


@dataclass
class StepProfile:
    step: str
    mode: str  # "computed", "replayed", "asim1", ...
    wall_s: float
    cpu_s: float
    rss_before_mb: float | None
    peak_mb: float | None
    step_extra_mb: float | None
    rss_after_mb: float | None


class Profiler:
    """
    Collects one ``StepProfile`` per step. With ``live_csv``, each step is
    also appended to that file as soon as it finishes, so a run that is
    stopped (e.g. killed for running out of memory) keeps its profile.
    """

    def __init__(self, collect_garbage: bool = True, live_csv: str | os.PathLike | None = None):
        self.records: list[StepProfile] = []
        self.collect_garbage = collect_garbage
        self.live_csv = Path(live_csv) if live_csv else None
        if self.live_csv:
            self.live_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(self.live_csv, "w", newline="") as f:
                csv.writer(f).writerow([fld.name for fld in fields(StepProfile)])

    @contextmanager
    def step(self, name: str, mode: str = "computed"):
        if self.collect_garbage:
            gc.collect()  # outside the timed region, so steps start from a clean slate
        rss_before = current_rss_mb()
        peak_ok = reset_peak()
        c0 = time.process_time()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            wall = time.perf_counter() - t0
            cpu = time.process_time() - c0
            peak = peak_rss_mb() if peak_ok else None
            rss_after = current_rss_mb()
            extra = None
            if peak is not None and rss_before is not None:
                extra = max(peak - rss_before, 0.0)
            rec = StepProfile(
                step=name,
                mode=mode,
                wall_s=wall,
                cpu_s=cpu,
                rss_before_mb=rss_before,
                peak_mb=peak,
                step_extra_mb=extra,
                rss_after_mb=rss_after,
            )
            self.records.append(rec)
            if self.live_csv:
                with open(self.live_csv, "a", newline="") as f:
                    row = asdict(rec)
                    csv.writer(f).writerow(
                        [round(v, 4) if isinstance(v, float) else v for v in row.values()])

    def write_csv(self, path: str | os.PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = [f.name for f in fields(StepProfile)]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=names)
            w.writeheader()
            for r in self.records:
                row = asdict(r)
                for k, v in row.items():
                    if isinstance(v, float):
                        row[k] = round(v, 4)
                w.writerow(row)
        return path

    def total_wall_s(self) -> float:
        return sum(r.wall_s for r in self.records)

    def max_peak_mb(self) -> float | None:
        peaks = [r.peak_mb for r in self.records if r.peak_mb is not None]
        return max(peaks) if peaks else None


def read_profile_csv(path: str | os.PathLike) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
