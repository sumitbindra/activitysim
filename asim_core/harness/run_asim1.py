#!/usr/bin/env python
"""
Run ActivitySim 1.x with every step saved, profiling each step with
asim_core's profiler so the numbers are directly comparable with asim_core runs.

This is a harness script: unlike the asim_core package itself, it imports
ActivitySim 1.x.

Examples (from the repository root)::

    # legacy engine (sharrow off), single process
    python asim_core/harness/run_asim1.py --out asim_core/work/asim1_legacy

    # sharrow: compile once (warm-up), then a timed run that uses the compiled code
    python asim_core/harness/run_asim1.py --configs configs_sh_compile configs \\
        --out asim_core/work/asim1_sharrow_compile
    python asim_core/harness/run_asim1.py --configs configs_sh configs \\
        --out asim_core/work/asim1_sharrow

Outputs in ``--out``: ActivitySim's usual outputs, the saved state after every
step (``pipeline.parquetpipeline``), ``asim_core_profile.csv`` (one row per
step) and ``run_info.json``.
"""

from __future__ import annotations

if __name__ == "__main__":  # threads for the numeric libraries, before NumPy loads (--threads, default 1; --fast)
    import _threads

    _threads.limit_from_argv()

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "asim_core" / "src"))

from asim_core.profiler import Profiler  # noqa: E402


def _write_overlay(out: Path, overrides: dict) -> Path:
    """A settings overlay directory, placed first in the configs list."""
    overlay = out / "_overlay_configs"
    overlay.mkdir(parents=True, exist_ok=True)
    lines = ["inherit_settings: True"]
    lines += [f"{k}: {v}" for k, v in overrides.items()]
    (overlay / "settings.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overlay


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--example-dir",
        default=str(REPO / "activitysim" / "examples" / "prototype_mtc"),
        help="ActivitySim example folder (default: prototype_mtc)",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=["configs"],
        help="config folders inside the example folder; the first takes precedence",
    )
    p.add_argument("--data", default="data", help="data folder inside the example")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--households", type=int, default=None, help="household sample size")
    p.add_argument("--trace-hh-id", type=int, default=None, help="household to trace")
    p.add_argument("--stop-after", default=None, help="run the model list only up to this step")
    p.add_argument("--resume-after", default=None,
                   help="start after this step, from the saved run already in --out (1.x's resume_after)")
    p.add_argument("--patch", nargs="*", default=[], help="optional changes to 1.x (see asim1_patches.py)")
    import _threads

    _threads.add_arguments(p)
    args = p.parse_args(argv)

    import activitysim
    from activitysim import abm  # noqa: F401  (registers the model steps)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import asim1_patches

    asim1_patches.apply(args.patch)
    from activitysim.cli.run import add_run_args, run
    from activitysim.core.workflow.runner import Runner

    out_early = Path(args.out).resolve()
    out_early.mkdir(parents=True, exist_ok=True)
    # written step by step, so a run killed partway (e.g. out of memory) keeps its profile
    profiler = Profiler(live_csv=out_early / "asim_core_profile.csv")
    original_by_name = Runner.by_name

    def profiled_by_name(self, model_name, **kwargs):
        with profiler.step(model_name, mode="asim1"):
            return original_by_name(self, model_name, **kwargs)

    Runner.by_name = profiled_by_name

    example = Path(args.example_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    config_dirs = [example / c for c in args.configs]
    overrides = {}
    if args.trace_hh_id is not None:
        overrides["trace_hh_id"] = args.trace_hh_id
    if args.resume_after:
        overrides["resume_after"] = args.resume_after
    if args.stop_after:
        import yaml

        from asim_core.spec.legacy import load_top_settings

        models = load_top_settings(config_dirs)["models"]
        models = models[: models.index(args.stop_after) + 1]
        overrides["models"] = yaml.safe_dump(models, default_flow_style=True).strip()
    if overrides:
        config_dirs.insert(0, _write_overlay(out, overrides))

    cli_args = []
    for c in config_dirs:
        cli_args += ["-c", str(c)]
    cli_args += ["-d", str(example / args.data), "-o", str(out)]
    if args.households is not None:
        cli_args += ["--households_sample_size", str(args.households)]

    cli = argparse.ArgumentParser()
    add_run_args(cli)
    ns = cli.parse_args(cli_args)

    t0 = time.perf_counter()
    error = None
    try:
        rc = run(ns)
    except Exception as err:  # keep the profile and saved state of the steps that ran
        rc, error = 1, f"{type(err).__name__}: {err}"
    total = time.perf_counter() - t0

    profile_path = profiler.write_csv(out / "asim_core_profile.csv")
    info = {
        "engine": "activitysim-1.x",
        "activitysim_version": activitysim.__version__,
        "example": str(example),
        "configs": args.configs,
        "households_sample_size": args.households,
        "trace_hh_id": args.trace_hh_id,
        "patches": args.patch,
        "threads": _threads.describe(),
        "exit_code": rc,
        "error": error,
        "steps_completed": len(profiler.records),
        "total_wall_s": round(total, 3),
        "steps_wall_s": round(profiler.total_wall_s(), 3),
        "max_peak_mb": profiler.max_peak_mb(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"profile written to {profile_path}")
    print(json.dumps(info, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
