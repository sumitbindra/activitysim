#!/usr/bin/env python
"""
Profile one step in a fresh process, starting from ActivitySim 1.x's saved
state just before that step. Runs either engine, so both are measured the
same way:

    python profile_step.py --engine asim1     --step auto_ownership_simulate --out row.csv
    python profile_step.py --engine asim_core --step auto_ownership_simulate --out row.csv

Each call appends one row to ``--out``. Loading the saved state, getting skims
ready and one-time Numba compiling happen before timing starts (the time to
get skims ready is recorded separately, as ``skims_ready_s``); the step
itself, including saving its results, is timed.
Harness code: the asim1 engine imports ActivitySim 1.x.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "asim_core" / "src"))

from asim_core.profiler import Profiler  # noqa: E402


def _with_read_skim_cache(configs, scratch: Path, value: bool):
    """Configs with network_los.yaml's read_skim_cache overridden (an overlay folder placed first)."""
    import yaml

    src = next(Path(c) / "network_los.yaml" for c in configs if (Path(c) / "network_los.yaml").exists())
    los = yaml.safe_load(src.read_text())
    los["read_skim_cache"] = value
    overlay = scratch.parent / f"{scratch.name}_overlay_configs"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "network_los.yaml").write_text(yaml.safe_dump(los, sort_keys=False))
    return [overlay] + list(configs)


def _warm_up_timetable():
    """
    1.x's timetable functions (used by every scheduling step) are compiled by
    Numba on first use. In a full run that happens in the first scheduling
    step; compile them here, with the argument types scheduling uses.
    """
    import numpy as np
    import pandas as pd
    from activitysim.core import timetable as tt1

    alts = pd.DataFrame({"start": [5, 5], "end": [5, 6]})
    alts["duration"] = alts.end - alts.start
    alts = alts.astype(np.int8)
    windows = pd.DataFrame(0, index=pd.Index([1, 2], name="person_id"),
                           columns=[str(p) for p in range(4, 8)], dtype=np.int8)
    t = tt1.TimeTable(windows, alts, "person_windows")
    ids = pd.Series(np.array([1, 2], dtype=np.int64))
    periods = pd.Series(np.array([5, 6], dtype=np.int8))
    # a timetable read from a saved run is column-major; one built in memory may not be
    for layout in (np.ascontiguousarray, np.asfortranarray):
        t.windows = layout(t.windows)
        t.tour_available(ids, np.array([0, 1]))
        t.adjacent_window_before(ids, periods)
        t.adjacent_window_after(ids, periods)
        t.previous_tour_ends(ids, periods)
        t.previous_tour_begins(ids, periods)
        t.remaining_periods_available(ids, periods, periods)
        t.max_time_block_available(ids)


def profile_asim1(step, prev, example, configs, pipeline, scratch, profiler, read_skim_cache=None,
                  patches=()):
    from activitysim import abm  # noqa: F401
    from activitysim.core import workflow
    from activitysim.core.workflow.runner import Runner

    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    if read_skim_cache is not None:
        configs = _with_read_skim_cache(configs, scratch, read_skim_cache)
    # 1.x writes new checkpoints into the pipeline it resumes from: use a copy
    shutil.copytree(pipeline, scratch / "pipeline.parquetpipeline")
    # reuse the skim cache the earlier full run wrote (as a repeated run would);
    # whether 1.x reads it depends on its settings (e.g. read_skim_cache)
    cache = Path(pipeline).parent / "cache"
    if cache.exists():
        shutil.copytree(cache, scratch / "cache")

    # 1.x compiles some functions with Numba on first use. In a full run that
    # happens in an earlier step (or is the same in every step), so do it here
    # before timing starts.
    import numpy as np
    from activitysim.abm.models.accessibility import _accumulate_accessibility
    from activitysim.core.choosing import choice_maker

    rn = np.full((2, 1), 0.3)
    for dtype in (np.float64, np.float32):  # sharrow produces 32-bit probabilities
        probs = np.full((2, 2), 0.5, dtype=dtype)
        choice_maker(probs, rn)
        choice_maker(np.asfortranarray(probs), rn)
        _accumulate_accessibility(np.zeros(4, dtype=dtype), 2, 2)
    _warm_up_timetable()

    original = Runner.by_name

    def profiled(self, model_name, **kwargs):
        with profiler.step(model_name, mode="asim1"):
            return original(self, model_name, **kwargs)

    Runner.by_name = profiled
    if patches:  # run-time changes to 1.x being tested (asim1_patches.py)
        import asim1_patches

        asim1_patches.apply(list(patches))
    state = workflow.State.make_default(
        working_dir=example,
        configs_dir=tuple(configs),
        data_dir=("data",),
        output_dir=scratch,
    )
    state.checkpoint.restore(prev)
    # Load skims before timing: in a full run this happens once, at the first
    # step that needs them. Timed separately. With sharrow, the first model
    # run in a process also loads the whole skim dataset, even for models
    # that use no skims.
    t0 = time.perf_counter()
    state.get("network_los")
    if state.settings.sharrow:
        state.get("skim_dataset")
    skims_ready_s = time.perf_counter() - t0
    state.run(models=[step], resume_after=prev)
    state.checkpoint.close_store()
    return skims_ready_s


def profile_asim_core(step, prev, example, configs, pipeline, scratch, profiler, seams,
                      read_skim_cache=None):
    from run_milestone1 import build_components, skim_store

    from asim_core.asim1 import Asim1State
    from asim_core.runner import Runner
    from asim_core.seams import Engine

    configs = [Path(c) for c in configs]
    comps = {c.name: c for c in build_components(configs)}
    st = Asim1State(pipeline)
    store = st.load_store(prev)
    t0 = time.perf_counter()
    # as 1.x would: import from OMX every run, or reuse an unchanged cache (read_skim_cache)
    skims = skim_store(example, configs, read_skim_cache=read_skim_cache)
    skims_ready_s = time.perf_counter() - t0
    Runner([comps[step]], scratch, engine=Engine(seams), skims=skims).run([step], store, profiler)
    return skims_ready_s


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
    p.add_argument("--read-skim-cache", choices=["settings", "true", "false"], default="settings",
                   help="override network_los.yaml's read_skim_cache for either engine")
    p.add_argument("--patch", nargs="*", default=[],
                   help="asim1 only: run-time patches to apply (see asim1_patches.py)")
    p.add_argument("--seams", default="{}",
                   help='asim_core only: implementations to use, as JSON, e.g. \'{"random": "counter"}\'')
    args = p.parse_args(argv)

    sys.path.insert(0, str(HERE))
    from asim_core.asim1 import Asim1State

    example = Path(args.example_dir).resolve()
    configs = [example / c for c in args.configs]
    st = Asim1State(args.asim1_out)
    prev = st.previous_step(args.step)
    profiler = Profiler()
    scratch = Path(args.scratch).resolve()
    rsc = {"settings": None, "true": True, "false": False}[args.read_skim_cache]
    if args.engine == "asim1":
        skims_ready_s = profile_asim1(args.step, prev, example, configs, st.path, scratch, profiler, rsc,
                                      args.patch)
    else:
        import json

        skims_ready_s = profile_asim_core(args.step, prev, example, configs, st.path, scratch,
                                          profiler, json.loads(args.seams), rsc)

    rec = [r for r in profiler.records if r.step == args.step][-1]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a") as f:
        if new:
            f.write("label,step,wall_s,cpu_s,rss_before_mb,peak_mb,step_extra_mb,rss_after_mb,"
                    "skims_ready_s\n")
        f.write(
            f"{args.label or args.engine},{rec.step},{rec.wall_s:.6f},{rec.cpu_s:.6f},"
            f"{rec.rss_before_mb},{rec.peak_mb},{rec.step_extra_mb},{rec.rss_after_mb},"
            f"{skims_ready_s:.6f}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
