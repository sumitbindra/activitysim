#!/usr/bin/env python
"""
Run a model in several processes, as 1.x does, from the input files, and
compare it with 1.x.

The plan is the configs' own 1.x settings (``multiprocess_steps``,
``num_processes``), e.g. MTC's ``configs_mp``. Two comparisons:

* with a 1.x run made with the same settings and number of processes
  (``--asim1-out``): every output file, byte for byte or cell by cell, so
  row order included;
* with a 1.x run in one process (``--asim1-single``, optional): every table's
  values by ID (row order differs between 1.x's single and multiprocess runs).

Example (MTC, 2 processes)::

    python asim_core/harness/run_asim1.py --configs configs_mp configs --out asim_core/work/asim1_mp2
    python asim_core/harness/run_multiprocess.py --configs configs_mp configs \\
        --asim1-out asim_core/work/asim1_mp2 --asim1-single asim_core/work/asim1_legacy \\
        --out asim_core/work/multiprocess_mtc
"""

from __future__ import annotations

if __name__ == "__main__":  # threads for the numeric libraries, before NumPy loads (--threads, default 1; --fast)
    import _threads

    _threads.limit_from_argv()

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "asim_core" / "src"))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from asim_core.asim1 import Asim1State  # noqa: E402
from asim_core.compare import compare_column  # noqa: E402
from asim_core.multiprocess import ModelFiles, plan_groups, run_in_groups  # noqa: E402
from asim_core.profiler import Profiler  # noqa: E402
from asim_core.seams import Engine  # noqa: E402
from asim_core.spec.legacy import load_top_settings  # noqa: E402
from asim_core.store import Store  # noqa: E402
from compare_outputs import compare_output_files  # noqa: E402
from asim_core.model import build_steps  # noqa: E402
from run_milestone1 import NOT_BUILT, asim1_sample_size, md_table, skim_store  # noqa: E402

KEYS = {"households": "household_id", "persons": "person_id", "tours": "tour_id", "trips": "trip_id",
        "joint_tour_participants": "participant_id", "land_use": "zone_id", "accessibility": "zone_id"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--example-dir", default=str(REPO / "activitysim/examples/prototype_mtc"))
    p.add_argument("--configs", nargs="+", default=["configs_mp", "configs"])
    p.add_argument("--data", nargs="+", default=["data"])
    p.add_argument("--asim1-out", default=None,
                   help="1.x run with the same settings and processes: its output files are compared, and its "
                        "households sample size used unless --households is given")
    p.add_argument("--households", type=int, default=None, help="households sample size (0: all)")
    p.add_argument("--asim1-single", default=None, help="1.x run in one process, to compare values by ID")
    p.add_argument("--out", required=True)
    p.add_argument("-m", "--multiprocess", "--processes", dest="processes", type=int, default=None,
                   help="number of processes, overriding settings.yaml's num_processes (as 1.x's -m N)")
    p.add_argument("--read-skim-cache", choices=["settings", "true", "false"], default="settings")
    p.add_argument("--seams", default="{}")
    import _threads

    _threads.add_arguments(p)
    p.add_argument("--no-file-compare", action="store_true",
                   help="don't compare the output files with --asim1-out (e.g. it ran in another number of processes)")
    p.add_argument("--threads-per-process", type=int, default=None,
                   help="threads for each worker's numeric libraries (default: as the main process, as 1.x); "
                        "can change results' last digits")
    args = p.parse_args(argv)
    import run_milestone1

    run_milestone1.DATA_DIRS[:] = args.data
    example = Path(args.example_dir).resolve()
    configs = [example / c for c in args.configs]
    data_dirs = [example / d for d in args.data]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = load_top_settings(configs)
    groups = plan_groups(settings, settings["models"])
    if args.processes:
        for g in groups:
            g.num_processes = args.processes
    sample = args.households if args.households is not None else asim1_sample_size(args.asim1_out)
    components = {c.name: c for c in build_steps(configs, data_dirs, sample, NOT_BUILT)}
    missing = [s for g in groups for s in g.steps if s not in components]
    if missing:
        print(f"these steps can't run in asim_core: { {s: NOT_BUILT.get(s, 'not ported') for s in missing} }")
        return 2
    t0 = time.perf_counter()
    rsc = {"settings": None, "true": True, "false": False}[args.read_skim_cache]
    skims = skim_store(example, configs, read_skim_cache=rsc)
    skims_s = time.perf_counter() - t0
    seams = json.loads(args.seams)
    base_seed = int(settings.get("rng_base_seed") or 0)
    model = ModelFiles(tuple(str(c) for c in configs), tuple(str(d) for d in data_dirs), str(skims.path),
                       tuple(sorted(seams.items())), base_seed, sample)
    engine = Engine(seams)

    t_run = time.perf_counter()
    store, history, timings = run_in_groups(groups, components, model, Store(), out, skims, engine,
                                            Profiler(split_memory=True), args.threads_per_process)
    run_s = time.perf_counter() - t_run
    (out / "timings.json").write_text(json.dumps({
        "processes": max(g.num_processes for g in groups), "threads": _threads.describe(),
        "threads_per_process": args.threads_per_process, "run_s": run_s, "skims_s": skims_s,
        "groups": timings}, indent=2, default=str), encoding="utf-8")
    timings = [{"group": t["group"], "steps": f"{t['first_step']} .. {t['last_step']}", "where": t["where"],
                "wall_s": round(t["wall_s"], 2),
                "per process, s": ", ".join(f"{w:.1f}" for w in t["process_wall_s"]),
                "per process, own peak MB": ", ".join(f"{m:,.0f}" for m in t["process_own_peak_mb"] if m)}
               for t in timings]

    lines = [f"# asim_core in several processes: {example.name}", "",
             f"Configs {args.configs}; groups as 1.x's multiprocess_steps; skims ready in {skims_s:.1f} s; "
             f"run {run_s:.1f} s.", "", md_table(pd.DataFrame(timings)), ""]
    ok = True
    if args.asim1_out and not args.no_file_compare:
        files = compare_output_files(out / "run" / "output", Path(args.asim1_out))
        same = sum(status == "exact" for _, status, _ in files)
        lines += [f"**Output files compared with 1.x in {groups[-2].num_processes if len(groups) > 1 else 1} "
                  f"processes: {same} of {len(files)} exact.**", "",
                  md_table(pd.DataFrame(files, columns=["file", "status", "note"])), ""]
        ok = same == len(files)
    if args.asim1_single:  # every value by ID against 1.x in one process
        st = Asim1State(args.asim1_single)
        last = st.steps[-1]
        comps = []
        for table, key in KEYS.items():
            if table not in store:
                continue
            ref = st.read(table, last).sort_index()
            ours = store[table].to_pandas().sort_index()
            for col in ours.columns:
                if col in ref.columns:
                    from asim_core.store import Table

                    comps.append(compare_column("by ID", Table.from_pandas(table, ours), col, ref[col]))
        exact = sum(c.status == "exact" for c in comps)
        lines += [f"**Values by ID compared with 1.x in one process: {exact} of {len(comps)} columns exact.**", ""]
        bad = [c for c in comps if c.status != "exact"]
        if bad:
            lines += [md_table(pd.DataFrame([c.__dict__ for c in bad])), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
