#!/usr/bin/env python
"""
Where a step's time goes, piece by piece: the ported steps run in one process
(other steps replayed from 1.x's saved run), with each swappable piece timed
(asim_core.seam_timing). The run is repeated in the same process: the first
time brings the skim pages into it, so later runs show the work itself, as in
any step of a whole run after the first ones that read the skims.

Writes ``pieces.csv`` (step, run, piece, seconds, calls) and ``pieces.md``
(the last run, each piece's share of each step) to ``--out``.
Harness code: replaying reads 1.x's saved run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "asim_core" / "src"))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from asim_core.asim1 import Asim1State  # noqa: E402
from asim_core.profiler import Profiler  # noqa: E402
from asim_core.runner import Runner  # noqa: E402
from asim_core.seam_timing import SeamTimes, TimedEngine  # noqa: E402
from asim_core.seams import Engine  # noqa: E402
from asim_core.spec.legacy import load_top_settings  # noqa: E402
from asim_core.store import Store  # noqa: E402
from run_milestone1 import build_components, md_table, skim_store  # noqa: E402

ORDER = ["skim_lookup", "formulas", "utilities", "interaction_utilities", "probabilities",
         "nested_probabilities", "nested_logsums", "sample", "choose", "random", "lookup",
         "logsum_keys", "skim_access"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--example-dir", required=True)
    p.add_argument("--configs", nargs="+", default=["configs"])
    p.add_argument("--asim1-out", required=True, help="1.x run to replay the other steps from")
    p.add_argument("--out", required=True)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--seams", default="{}", help="implementations to use, as JSON")
    args = p.parse_args(argv)

    import json

    example = Path(args.example_dir).resolve()
    configs = [example / c for c in args.configs]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    st = Asim1State(args.asim1_out)
    base_seed = int(load_top_settings(configs).get("rng_base_seed") or 0)
    components = build_components(configs)
    skims = skim_store(example, configs, read_skim_cache=True)
    choices = json.loads(args.seams)

    rows = []
    for run in range(1, args.runs + 1):
        times = {c.name: SeamTimes() for c in components}
        engines = {c.name: TimedEngine(Engine(choices), times[c.name]) for c in components}
        prof = Profiler()
        t0 = time.perf_counter()
        Runner(components, out / f"run{run}", replay=st, base_seed=base_seed, skims=skims,
               step_engines=engines).run(st.steps, Store(), prof)
        print(f"run {run}: {time.perf_counter() - t0:.1f} s", flush=True)
        for rec in prof.records:
            if rec.step not in times:
                continue
            t = times[rec.step]
            for piece, sec in t.seconds.items():
                rows.append({"step": rec.step, "run": run, "piece": piece, "seconds": sec,
                             "calls": t.calls[piece]})
            rows.append({"step": rec.step, "run": run, "piece": "other", "seconds": rec.wall_s - t.total,
                         "calls": 0})
            rows.append({"step": rec.step, "run": run, "piece": "step (wall)", "seconds": rec.wall_s,
                         "calls": 0})
    df = pd.DataFrame(rows)
    df.to_csv(out / "pieces.csv", index=False)

    last = df[df.run == args.runs]
    wide = last.pivot_table(index="step", columns="piece", values="seconds", sort=False).fillna(0.0)
    pieces = [c for c in ORDER if c in wide.columns] + ["other"]
    total = wide["step (wall)"]
    table = pd.DataFrame({"step": wide.index, "wall_s": total.round(2)})
    for c in pieces:
        table[c] = [f"{s:.2f} ({100 * s / w:.0f}%)" if s >= 0.005 else "" for s, w in zip(wide[c], total)]
    all_steps = wide[pieces].sum()
    summary = pd.DataFrame({"piece": pieces, "seconds": all_steps.round(2).to_numpy(),
                            "share": (100 * all_steps / total.sum()).round(1).to_numpy()})
    text = ["# Where the time goes, piece by piece", "",
            f"Example `{example.name}`, run {args.runs} of {args.runs} in one process "
            f"(seams: {choices or 'reference'}).", "",
            md_table(table.reset_index(drop=True), "{:.2f}"), "", "All ported steps together:", "",
            md_table(summary, "{:.2f}"), ""]
    (out / "pieces.md").write_text("\n".join(text))
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
