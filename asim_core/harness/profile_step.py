#!/usr/bin/env python
"""
Profile one step in a fresh process, starting from ActivitySim 1.x's saved
state just before that step. Runs either engine, so both are measured the
same way:

    python profile_step.py --engine asim1     --step auto_ownership_simulate --out row.csv
    python profile_step.py --engine asim_core --step auto_ownership_simulate --out row.csv

Each call appends one row to ``--out``. Loading the saved state happens before
timing starts; the step itself, including saving its results, is timed.
Harness code: the asim1 engine imports ActivitySim 1.x.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "asim_core" / "src"))

from asim_core.profiler import Profiler  # noqa: E402


def profile_asim1(step, prev, example, configs, pipeline, scratch, profiler):
    from activitysim import abm  # noqa: F401
    from activitysim.core import workflow
    from activitysim.core.workflow.runner import Runner

    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    # 1.x writes new checkpoints into the pipeline it resumes from: use a copy
    shutil.copytree(pipeline, scratch / "pipeline.parquetpipeline")

    # 1.x compiles its choice function with Numba on first use. In a full run
    # that happens in an earlier step, so do it here before timing starts.
    import numpy as np
    from activitysim.core.choosing import choice_maker

    rn = np.full((2, 1), 0.3)
    choice_maker(np.full((2, 2), 0.5), rn)
    choice_maker(np.asfortranarray(np.full((2, 2), 0.5)), rn)

    original = Runner.by_name

    def profiled(self, model_name, **kwargs):
        with profiler.step(model_name, mode="asim1"):
            return original(self, model_name, **kwargs)

    Runner.by_name = profiled
    state = workflow.State.make_default(
        working_dir=example,
        configs_dir=tuple(configs),
        data_dir=("data",),
        output_dir=scratch,
    )
    state.run(models=[step], resume_after=prev)
    state.checkpoint.close_store()


def profile_asim_core(step, prev, configs, pipeline, scratch, profiler):
    from run_milestone1 import build_components

    from asim_core.asim1 import Asim1State
    from asim_core.runner import Runner

    comps = {c.name: c for c in build_components([Path(c) for c in configs])}
    st = Asim1State(pipeline)
    store = st.load_store(prev)
    Runner([comps[step]], scratch).run([step], store, profiler)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["asim1", "asim_core"], required=True)
    p.add_argument("--step", required=True)
    p.add_argument("--label", default=None, help="label for this engine/config in the output")
    p.add_argument("--example-dir", default=str(REPO / "activitysim/examples/prototype_mtc"))
    p.add_argument("--configs", nargs="+", default=["configs"])
    p.add_argument("--asim1-out", default=str(REPO / "asim_core/work/asim1_legacy"))
    p.add_argument("--scratch", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    sys.path.insert(0, str(HERE))
    from asim_core.asim1 import Asim1State

    example = Path(args.example_dir).resolve()
    configs = [example / c for c in args.configs]
    st = Asim1State(args.asim1_out)
    prev = st.previous_step(args.step)
    profiler = Profiler()
    scratch = Path(args.scratch).resolve()
    if args.engine == "asim1":
        profile_asim1(args.step, prev, example, configs, st.path, scratch, profiler)
    else:
        profile_asim_core(args.step, prev, configs, st.path, scratch, profiler)

    rec = [r for r in profiler.records if r.step == args.step][-1]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a") as f:
        if new:
            f.write("label,step,wall_s,cpu_s,rss_before_mb,peak_mb,step_extra_mb,rss_after_mb\n")
        f.write(
            f"{args.label or args.engine},{rec.step},{rec.wall_s:.6f},{rec.cpu_s:.6f},"
            f"{rec.rss_before_mb},{rec.peak_mb},{rec.step_extra_mb},{rec.rss_after_mb}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
