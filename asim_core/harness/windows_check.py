#!/usr/bin/env python
"""
Check asim_core on this computer (meant for Windows; works anywhere): the
tests, then the MTC example in one process and in 2, each against 1.x run
here, on this machine. Writes ``asim_core/work/windows_check/report.md``
(paste it back) and keeps going after a failed stage.

From the repository root, in the environment with ActivitySim 1.x and
asim_core installed::

    python asim_core/harness/windows_check.py

About 15 to 25 minutes. Stages already done (their summary exists) are
skipped; ``--again`` redoes them all, ``--redo`` the ones named (``tests``,
``asim1``, ``asim_core``, ``asim1_mp``, ``asim_core_mp``), e.g. after an
update::

    python asim_core/harness/windows_check.py --redo tests asim_core asim_core_mp
"""

from __future__ import annotations

import argparse
import json
import locale
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = REPO / "asim_core" / "work"
OUT = WORK / "windows_check"


def run(name: str, args: list[str], log: Path) -> tuple[int, float]:
    """Run a harness command, its output to ``log``; (exit code, seconds)."""
    print(f"== {name}: {' '.join(args)}", flush=True)
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([sys.executable, *args], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    secs = time.perf_counter() - t0
    print(f"   exit {rc}, {secs:.0f} s, log {log}", flush=True)
    return rc, secs


def headline(summary: Path) -> list[str]:
    if not summary.exists():
        return ["(no summary written)"]
    return [ln for ln in summary.read_text(encoding="utf-8", errors="replace").splitlines() if ln.startswith("**")]


def tail(log: Path, n: int = 25) -> str:
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
    return "\n".join(lines[-n:])


def environment() -> list[str]:
    import numpy as np

    rows = [f"- {platform.platform()}, Python {platform.python_version()}, {os.cpu_count()} CPUs",
            f"- encodings: preferred {locale.getpreferredencoding(False)}, filesystem {sys.getfilesystemencoding()}, "
            f"stdout {sys.stdout.encoding}"]
    for mod in ("numpy", "pandas", "pyarrow", "tables", "yaml", "numexpr", "activitysim"):
        try:
            m = __import__(mod)
            rows.append(f"- {mod} {getattr(m, '__version__', '?')}")
        except Exception as err:  # noqa: BLE001
            rows.append(f"- {mod}: not importable ({type(err).__name__}: {err})")
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        rows.append(f"- BLAS: {blas.get('name')} {blas.get('version', '')}")
    except Exception:  # noqa: BLE001
        pass
    rows.append(f"- default integer: {np.array([1]).dtype} (must be int64)")
    try:
        import yaml

        rows.append(f"- PyYAML C loader: {hasattr(yaml, 'CSafeLoader')}")
    except Exception:  # noqa: BLE001
        pass
    sys.path.insert(0, str(REPO / "asim_core" / "src"))
    try:
        from asim_core.profiler import memory_now

        rows.append(f"- asim_core memory counters now: {memory_now()}")
    except Exception as err:  # noqa: BLE001
        rows.append(f"- asim_core memory counters: failed ({type(err).__name__}: {err})")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--again", action="store_true", help="redo stages already done")
    p.add_argument("--redo", nargs="*", default=[], choices=["tests", "asim1", "asim_core", "asim1_mp", "asim_core_mp"],
                   help="redo these stages even if done")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    py = ["-X", "utf8=0"]  # the system's own encoding, as an agency's Python would have it
    h = "asim_core/harness/"
    stages = [  # (name, command, the file that shows it's done)
        ("tests", py + ["-m", "pytest", "-q", "-p", "no:cacheprovider", "asim_core/tests"], None),
        ("1.x MTC example, one process", py + [h + "run_asim1.py", "--out", str(WORK / "asim1_legacy")],
         WORK / "asim1_legacy" / "final_trips.csv"),
        ("asim_core MTC example, one process", py + [h + "run_standalone.py", "--through", "summarize",
                                                     "--out", str(OUT / "standalone_mtc")],
         OUT / "standalone_mtc" / "summary.md"),
        ("1.x MTC example, 2 processes", py + [h + "run_asim1.py", "--configs", "configs_mp", "configs",
                                               "--out", str(WORK / "asim1_mp2")],
         WORK / "asim1_mp2" / "final_trips.csv"),
        ("asim_core MTC example, 2 processes", py + [h + "run_multiprocess.py", "--asim1-out", str(WORK / "asim1_mp2"),
                                                     "--asim1-single", str(WORK / "asim1_legacy"),
                                                     "--out", str(OUT / "multiprocess_mtc")],
         OUT / "multiprocess_mtc" / "summary.md"),
    ]
    report = ["# asim_core check on this computer", "", *environment(), ""]
    results = {}
    keys = ["tests", "asim1", "asim_core", "asim1_mp", "asim_core_mp"]
    for key, (name, cmd, done_file) in zip(keys, stages):
        log = OUT / (name.replace(" ", "_").replace(",", "").replace(".", "") + ".log")
        if done_file is not None and done_file.exists() and not args.again and key not in args.redo:
            print(f"== {name}: already done ({done_file})", flush=True)
            rc, secs = 0, None
        else:
            rc, secs = run(name, cmd, log)
        results[name] = {"exit": rc, "seconds": None if secs is None else round(secs)}
        report += [f"## {name}", "", f"exit code {rc}" + ("" if secs is None else f", {secs:.0f} s"), ""]
        if name == "tests":
            report += ["```", tail(log, 12), "```", ""]
        elif done_file is not None and done_file.suffix == ".md":
            report += headline(done_file) + [""]
        if rc != 0 and name != "tests":
            report += ["Last lines of the log:", "", "```", tail(log), "```", ""]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(report))
    print(f"\nReport: {OUT / 'report.md'}")
    return 0 if all(r["exit"] == 0 for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
