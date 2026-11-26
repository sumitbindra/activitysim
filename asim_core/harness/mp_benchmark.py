#!/usr/bin/env python
"""
How a model's time and memory change with the number of processes, on this
computer: the same model run with 1 process, then with more, every run
checked against the 1-process run, value by value by ID.

Defaults: full MTC zones, 50,000 households, 1, 8, 16 and 24 processes, one
thread per process. Change the settings below, or pass them::

    python asim_core/harness/mp_benchmark.py
    python asim_core/harness/mp_benchmark.py --processes 1 4 8 12 --households 50000
    python asim_core/harness/mp_benchmark.py --example-dir activitysim/examples/prototype_mtc --households 0

Writes ``<out>/report.md`` (paste it back) and each run's own folder. No 1.x
run is needed. A warm-up run first (not counted) imports the skims into
asim_core's cache if needed and brings the skim file into memory, so every
timed run starts the same way.

What to expect (see docs/multiprocess.md): the households' steps (school
location to trip mode choice) are split among the processes and get faster
with more of them; the steps before (land use, households) and after (the
output files) run in the main process and don't. Each process holds its
own share of the households and its own copy of the model: memory grows
with the number of processes. The skims are one file shared by all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# --- settings: change these, or pass them on the command line ---------------------------------------------
EXAMPLE_DIR = REPO / "asim_core" / "work" / "mtc_full" / "prototype_mtc_full"  # full MTC zones
CONFIGS = ["configs"]                       # the region's configs folders, inside EXAMPLE_DIR
DATA = ["data"]                             # its data folders, inside EXAMPLE_DIR
HOUSEHOLDS = 50_000                         # households sample size (0: all)
PROCESSES = [1, 8, 16, 24]                  # process counts to time; 1 first: the others are checked against it
THREADS = 1                                 # threads per process for the numeric libraries (1, as 1.x)
MULTIPROCESS_SETTINGS = REPO / "asim_core" / "configs" / "multiprocess"  # listed first: the groups, as 1.x
OUT = REPO / "asim_core" / "work" / "mp_benchmark"
# ------------------------------------------------------------------------------------------------------------

KEYS = {"households": "household_id", "persons": "person_id", "tours": "tour_id", "trips": "trip_id",
        "joint_tour_participants": "participant_id"}


def run_model(n: int, out: Path, args) -> tuple[int, float]:
    """One run with n processes; (exit code, seconds) and its log in out/run.log."""
    cmd = [sys.executable, str(HERE / "run_multiprocess.py"),
           "--example-dir", str(args.example_dir),
           "--configs", str(MULTIPROCESS_SETTINGS), *args.configs,
           "--data", *args.data,
           "--households", str(args.households),
           "--multiprocess", str(n),
           "--threads", str(args.threads),
           "--read-skim-cache", "true",
           "--out", str(out)]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(out / "run.log", "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.call(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    return rc, time.perf_counter() - t0


def summary(out: Path) -> dict:
    """Times and memory of one run, from what run_multiprocess.py wrote."""
    import pandas as pd

    t = json.loads((out / "timings.json").read_text(encoding="utf-8"))
    groups = {g["group"]: g for g in t["groups"]}
    hh = groups.get("mp_households", {})
    main = pd.read_csv(out / "run" / "profile.csv")
    workers = [m for m in hh.get("process_own_peak_mb", []) if m]
    before = main.set_index("step").anon_peak_mb.loc[:"compute_accessibility"].max()
    return {
        "run_s": t["run_s"],
        "households_s": hh.get("wall_s"),
        "main_s": t["run_s"] - (hh.get("wall_s") or 0.0),
        "main_peak_mb": main.anon_peak_mb.max(),
        "workers_peak_mb": workers,
        # while the workers run: the main process (as before them) and every worker at its peak
        "together_mb": (before + sum(workers)) if workers else main.anon_peak_mb.max(),
    }


def differences(out: Path, base: Path) -> str:
    """Every final table's values by ID against the 1-process run: 'identical' or what differs."""
    import pandas as pd

    notes = []
    for table, key in KEYS.items():
        a, b = base / "run" / "output" / f"final_{table}.csv", out / "run" / "output" / f"final_{table}.csv"
        if not a.exists() or not b.exists():
            continue
        x = pd.read_csv(a, low_memory=False).set_index(key).sort_index()
        y = pd.read_csv(b, low_memory=False).set_index(key).sort_index()
        if not x.index.equals(y.index):
            notes.append(f"{table}: {len(y) - len(x):+,} rows")
            continue
        cols = [c for c in x.columns if c in y.columns]
        bad = {c: int(((x[c] != y[c]) & ~(x[c].isna() & y[c].isna())).sum()) for c in cols if not x[c].equals(y[c])}
        notes += [f"{table}.{c}: {n:,} rows" for c, n in bad.items()]
    return "identical" if not notes else "; ".join(notes)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--example-dir", type=Path, default=EXAMPLE_DIR)
    p.add_argument("--configs", nargs="+", default=CONFIGS)
    p.add_argument("--data", nargs="+", default=DATA)
    p.add_argument("--households", type=int, default=HOUSEHOLDS)
    p.add_argument("--processes", nargs="+", type=int, default=PROCESSES)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--no-warm-up", action="store_true")
    args = p.parse_args(argv)
    if not args.example_dir.exists():
        print(f"example folder not found: {args.example_dir} (pass --example-dir)")
        return 2
    counts = sorted(set(args.processes))
    if counts[0] != 1:
        counts = [1] + counts  # the others are checked against it
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_warm_up:
        print(f"== warm-up, {max(counts)} processes (not counted)", flush=True)
        rc, secs = run_model(max(counts), args.out / "warm_up", args)
        print(f"   exit {rc}, {secs:.0f} s", flush=True)
    rows = []
    for n in counts:
        out = args.out / f"processes_{n}"
        print(f"== {n} process{'es' if n > 1 else ''}", flush=True)
        rc, secs = run_model(n, out, args)
        print(f"   exit {rc}, {secs:.0f} s", flush=True)
        row = {"processes": n, "exit": rc}
        if (out / "timings.json").exists():
            row.update(summary(out))
            row["vs 1 process"] = "(reference)" if n == 1 else differences(out, args.out / "processes_1")
        rows.append(row)

    base = next((r for r in rows if r["processes"] == 1 and "run_s" in r), None)
    lines = ["# asim_core: time and memory against the number of processes", "",
             f"{args.example_dir.name}, configs {args.configs}, {args.households:,} households "
             f"(0: all), {args.threads} thread(s) per process. Run time: from setting up to the output files, "
             "skims already in asim_core's cache and in memory (warm-up run first).", "",
             "| Processes | Run | Speed-up | Households steps | Main process (set-up, output files) | "
             "Own memory, main process peak | Own memory, each process | Own memory, all processes together | "
             "Values by ID vs 1 process |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "run_s" not in r:
            lines.append(f"| {r['processes']} | failed (exit {r['exit']}, see processes_{r['processes']}/run.log) "
                         "| | | | | | | |")
            continue
        speed = f"{base['run_s'] / r['run_s']:.2f}x" if base else ""
        each = (f"{min(r['workers_peak_mb']) / 1024:.2f} to {max(r['workers_peak_mb']) / 1024:.2f} GB"
                if r["workers_peak_mb"] else "(no workers)")
        lines.append(f"| {r['processes']} | {r['run_s']:.1f} s | {speed} | {r['households_s']:.1f} s | "
                     f"{r['main_s']:.1f} s | {r['main_peak_mb'] / 1024:.2f} GB | {each} | "
                     f"{r['together_mb'] / 1024:.2f} GB | {r['vs 1 process']} |")
    lines += ["", "Each run's details: `processes_<n>/summary.md`, `processes_<n>/timings.json`, "
              "`processes_<n>/run.log`.", "",
              "A few last-digit differences, or a stop frequency and its trips, against 1 process are 1.x's own "
              "behaviour too (asim1 findings 23: MTC's stop frequency looks household tour counts up by person "
              "ID, which finds another household only in the same process)."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport: {args.out / 'report.md'}")
    return 0 if all(r["exit"] == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
