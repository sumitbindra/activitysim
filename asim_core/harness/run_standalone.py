#!/usr/bin/env python
"""
Run asim_core on its own: the ported steps from the input files, in one
process, with nothing taken from a 1.x run; then compare its tables with
1.x's at the last step. (The milestone script checks each step from 1.x's
saved state and replays the steps not ported; this one shows the ported
steps run together without 1.x.)

    python asim_core/harness/run_standalone.py --through mandatory_tour_scheduling

The 1.x run (``--asim1-out``) gives the step list, the households sample
size, and the tables to compare with; it is read only after asim_core has
run. Writes summary.md, comparisons.csv and profile.csv to ``--out``.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asim_core.asim1 import Asim1State  # noqa: E402
from asim_core.compare import compare_column, write_comparisons  # noqa: E402
from asim_core.profiler import Profiler  # noqa: E402
from asim_core.runner import Runner  # noqa: E402
from asim_core.seams import Engine  # noqa: E402
from asim_core.spec.legacy import load_top_settings  # noqa: E402
from asim_core.store import Store  # noqa: E402
from compare_outputs import compare_output_files  # noqa: E402
from run_milestone1 import OUTPUT_STEPS, NOT_BUILT, build_components, load_asim1_profile, md_table, skim_store  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--example-dir", default=str(REPO / "activitysim/examples/prototype_mtc"))
    p.add_argument("--configs", nargs="+", default=["configs"])
    p.add_argument("--asim1-out", default=str(REPO / "asim_core/work/asim1_legacy"))
    p.add_argument("--through", default="mandatory_tour_scheduling", help="the last step to run")
    p.add_argument("--out", default=str(REPO / "asim_core/work/standalone"))
    p.add_argument("--read-skim-cache", choices=["settings", "true", "false"], default="settings")
    p.add_argument("--asim1-profile-run", default=None,
                   help="take 1.x's step times from this run instead (e.g. 1.x as shipped, when the run "
                        "compared with was made with numexpr off)")
    p.add_argument("--seams", default="{}",
                   help='choices for the seams, as JSON, e.g. \'{"sample_columns": "all_columns", '
                        '"size_lookup": "match"}\' (python -m asim_core.seams lists them); '
                        'unnamed seams use the reference')
    import _threads

    _threads.add_arguments(p)
    p.add_argument("--data", nargs="+", default=["data"],
                   help="the example's data folders, inside it (default: data; SANDAG's one-zone setup: data_1)")
    args = p.parse_args(argv)
    import run_milestone1

    run_milestone1.DATA_DIRS[:] = args.data

    example = Path(args.example_dir).resolve()
    configs = [example / c for c in args.configs]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    st = Asim1State(args.asim1_out)
    if args.through not in st.steps:
        print(f"'{args.through}' is not a step of the 1.x run ({st.steps})")
        return 2
    steps = st.steps[: st.steps.index(args.through) + 1]
    components = {c.name: c for c in build_components(configs, args.asim1_out)}
    missing = [s for s in steps if s not in components]
    if missing:
        why = {s: NOT_BUILT.get(s, "not ported") for s in missing}
        print(f"these steps can't run in asim_core alone: {why}")
        return 2

    rsc = {"settings": None, "true": True, "false": False}[args.read_skim_cache]
    t0 = time.perf_counter()
    skims = skim_store(example, configs, read_skim_cache=rsc)
    skims_s = time.perf_counter() - t0
    base_seed = int(load_top_settings(configs).get("rng_base_seed") or 0)
    profiler = Profiler(split_memory=True)
    store = Store()
    t0 = time.perf_counter()
    report = Runner([components[s] for s in steps], out / "run", base_seed=base_seed, skims=skims,
                    engine=Engine(json.loads(args.seams))).run(
        steps, store, profiler, start_note="input files only; every step computed")
    run_s = time.perf_counter() - t0

    # compare with 1.x after the last step: every column asim_core made, every row of 1.x's table
    comparisons = []
    for s in steps:
        for table, cols in components[s].provides.items():
            ref = st.read(table, args.through)
            for col in cols:
                if col in ref.columns:
                    comparisons.append(compare_column(f"after {args.through}", store[table], col, ref[col]))
    comparisons = list({(c.table, c.column): c for c in comparisons}.values())
    write_comparisons(comparisons, out / "comparisons.csv")
    profiler.write_csv(out / "profile.csv")

    ours = pd.DataFrame([r.__dict__ for r in profiler.records])
    theirs = load_asim1_profile(Path(args.asim1_profile_run or args.asim1_out))
    table = ours[["step", "wall_s", "cpu_s", "anon_peak_mb", "peak_mb"]].rename(
        columns={"wall_s": "asim_core_s", "cpu_s": "asim_core_cpu_s", "anon_peak_mb": "asim_core_own_peak_mb",
                 "peak_mb": "asim_core_peak_mb"})
    if theirs is not None:
        table = table.merge(theirs[["step", "wall_s", "peak_mb"]].rename(
            columns={"wall_s": "asim1_s", "peak_mb": "asim1_peak_mb"}), on="step", how="left")
    total = {"step": "all", "asim_core_s": table.asim_core_s.sum(), "asim_core_cpu_s": table.asim_core_cpu_s.sum()}
    if "asim1_s" in table:
        total["asim1_s"] = table.asim1_s.sum()
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    exact = sum(c.status == "exact" for c in comparisons)
    rows = sum(c.rows for c in comparisons)
    lines = [
        f"# asim_core alone, {steps[0]} through {args.through}",
        "",
        f"Example `{example.name}`, 1.x run `{Path(args.asim1_out).name}` (its step list and sample size; "
        f"read only to compare, after the run). Skims ready in {skims_s:.1f} s (not included below).",
        "",
        f"**Compared with 1.x after `{args.through}`: {exact} of {len(comparisons)} columns exact** "
        f"({rows:,} values).",
        "",
    ]
    bad = [c for c in comparisons if c.status != "exact"]
    if bad:
        lines += ["Not exact:", "", md_table(pd.DataFrame([c.__dict__ for c in bad])), ""]
    files = []
    if any(s in OUTPUT_STEPS for s in steps):  # the output steps' files against 1.x's
        files = compare_output_files(out / "run" / "output", Path(args.asim1_out))
        same = sum(status == "exact" for _, status, _ in files)
        lines += [f"**Output files compared with 1.x: {same} of {len(files)} exact.**", "",
                  md_table(pd.DataFrame(files, columns=["file", "status", "note"])), ""]
    if report.warnings:
        lines += ["Warnings:", ""] + [f"- {w['step']}: {w['message']}" for w in report.warnings] + [""]
    lines += [
        f"Per step, one process (seconds; memory in MB; 1.x from its full run "
        f"`{Path(args.asim1_profile_run or args.asim1_out).name}`, where `initialize_landuse` includes "
        f"loading skims). asim_core run total: {run_s:.1f} s.",
        "",
        md_table(table.round(2)),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({"example": str(example), "asim1_run": str(args.asim1_out),
                                                   "through": args.through, "steps": steps, "run_s": run_s,
                                                   "skims_s": skims_s, "threads": _threads.describe()},
                                                  indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0 if not bad and all(status == "exact" for _, status, _ in files) else 1


if __name__ == "__main__":
    sys.exit(main())
