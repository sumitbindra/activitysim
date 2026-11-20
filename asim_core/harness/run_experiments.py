#!/usr/bin/env python
"""
Run experiments: the same step with different implementations of its
swappable pieces, each checked against ActivitySim 1.x and profiled, next to
the 1.x legacy and sharrow baselines.

Experiments are listed in ``experiments.yaml`` (or ``--config``). For each:

* **Match:** run from 1.x's saved state before the step and compare with 1.x's
  result: rows with a different result, the largest difference in the share of
  rows taking any value (with the sampling error expected by chance), and the
  largest utility difference from the reference implementation.
* **Profile:** run the step in fresh processes, ``repeat`` times
  (``profile_step.py``), the same way as the 1.x baselines; with
  ``--page-cache cold warm``, once for each state of the skim file in the
  operating system's page cache (see page_cache.py).
* **Seam benchmarks:** time each implementation of a piece on its own with
  synthetic data at larger row counts, to show how costs grow.

Writes ``experiments.csv``, ``seam_benchmarks.csv`` and ``experiments.md``
to ``--out``. Harness code: the baselines import ActivitySim 1.x.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "asim_core" / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from asim_core.asim1 import Asim1State  # noqa: E402
from asim_core.compare import compare_column, value_shares  # noqa: E402
from asim_core.profiler import Profiler  # noqa: E402
from asim_core.runner import Runner  # noqa: E402
from asim_core.seams import Engine, implementations  # noqa: E402
from asim_core.spec.legacy import load_top_settings  # noqa: E402
from run_milestone1 import build_components, few_values, md_table, skim_store  # noqa: E402


# --- match ------------------------------------------------------------------------


def match_check(comp, st: Asim1State, engine: Engine, base_seed: int, out: Path, skims=None) -> dict:
    prev = st.previous_step(comp.name)
    store = st.load_store(prev)
    before = None
    if "compute" in comp.info:
        view = comp.info["view"]
        before = (view.gather_from_store(store, comp.info["columns"]), len(store[view.table]))
    Runner([comp], out, engine=engine, base_seed=base_seed, skims=skims).run([comp.name], store, Profiler())

    row = {"rows": 0, "rows_different": 0, "largest_share_diff_pp": 0.0,
           "sampling_error_2se_pp": 0.0, "shares_within_noise": True,
           # choices (integer, text, categories) apart from decimal values (logsums, ...)
           "choice_rows_different": 0, "max_value_diff": 0.0}
    for table, cols in comp.provides.items():
        ref = st.read(table, comp.name)
        for col in cols:
            c = compare_column(comp.name, store[table], col, ref[col])
            row["rows"] += c.rows
            row["rows_different"] += c.mismatched
            row["max_output_diff"] = max(row.get("max_output_diff", 0.0), c.max_abs_diff or 0.0)
            if np.asarray(store[table][col]).dtype.kind == "f" and np.asarray(ref[col]).dtype.kind == "f":
                row["max_value_diff"] = max(row["max_value_diff"], c.max_abs_diff or 0.0)
            else:
                row["choice_rows_different"] += c.mismatched
            if not few_values(store[table][col]):
                continue  # shares only make sense for columns with few distinct values
            shares = value_shares(store[table][col], ref[col])
            n = c.rows
            # two independent samples of the same households: sd of a share difference
            p = shares["asim1"].to_numpy()
            two_se = 2 * np.sqrt(2 * p * (1 - p) / n)
            diff = shares["difference"].abs().to_numpy()
            worst = int(np.argmax(diff))
            if diff[worst] * 100 >= row["largest_share_diff_pp"]:
                row["largest_share_diff_pp"] = diff[worst] * 100
                row["sampling_error_2se_pp"] = two_se[worst] * 100
            row["shares_within_noise"] &= bool(np.all(diff <= np.maximum(two_se, 1.0 / n)))

    row["max_utility_diff"] = np.nan
    if before is not None:
        cols, n = before
        ours, _, _ = comp.info["compute"](cols, n, engine)
        ref_u, _, _ = comp.info["compute"](cols, n, Engine())
        row["max_utility_diff"] = float(np.abs(ours - ref_u).max())

    if row["rows_different"] == 0:
        row["result_vs_1x"] = "same as 1.x"
    elif engine.agreement == "approximate" and row["choice_rows_different"] == 0:
        row["result_vs_1x"] = f"choices same as 1.x; values within {row['max_value_diff']:.1e}"
    elif engine.agreement == "approximate":
        row["result_vs_1x"] = (f"{row['choice_rows_different']:,} choices differ; values within "
                               f"{row['max_value_diff']:.1e}")
    elif engine.agreement == "statistical":
        row["result_vs_1x"] = ("same distribution (rows differ by design)"
                               if row["shares_within_noise"] else "distribution differs: check")
    else:
        row["result_vs_1x"] = "differs: check"
    return row


# --- profiles ---------------------------------------------------------------------


def profile(step, label, engine_kind, configs, run_dir, seams, repeat, out: Path,
            example: Path, read_skim_cache: str = "settings", page_cache: str = "as_is",
            env: dict | None = None) -> pd.DataFrame:
    rows_csv = out / "runs" / f"{label}.csv"
    rows_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows_csv.exists():
        rows_csv.unlink()
    for _ in range(repeat):
        cmd = [
            sys.executable, str(HERE / "profile_step.py"),
            "--engine", engine_kind, "--step", step, "--label", label,
            "--example-dir", str(example),
            "--configs", *configs, "--asim1-out", str(run_dir),
            "--scratch", str(out / "scratch" / label), "--out", str(rows_csv),
            "--seams", json.dumps(seams),
            "--read-skim-cache", read_skim_cache if engine_kind == "asim_core" else "settings",
            "--page-cache", page_cache if engine_kind == "asim_core" else "as_is",
        ]
        subprocess.run(cmd, check=True, capture_output=True, env={**os.environ, **(env or {})})
    return pd.read_csv(rows_csv)


def summarize_runs(runs: pd.DataFrame) -> dict:
    return {
        "median_s": float(runs["wall_s"].median()),
        "min_s": float(runs["wall_s"].min()),
        "max_s": float(runs["wall_s"].max()),
        "median_extra_mb": float(runs["step_extra_mb"].median()),
        "median_peak_mb": float(runs["peak_mb"].median()),
        "median_jax_compile_s": (float(runs["jax_compile_s"].median())
                                 if "jax_compile_s" in runs and runs["jax_compile_s"].notna().any() else np.nan),
        "median_own_extra_mb": (float(runs["own_extra_mb"].median())
                                if "own_extra_mb" in runs else np.nan),
        "median_file_peak_mb": (float(runs["file_peak_mb"].median())
                                if "file_peak_mb" in runs else np.nan),
        "skims_resident": (float(runs["skims_resident_share"].median())
                           if "skims_resident_share" in runs else np.nan),
        "runs": len(runs),
    }


# --- seam benchmarks --------------------------------------------------------------


def seam_benchmarks(cfg: dict) -> pd.DataFrame:
    rows = []
    sizes = cfg.get("rows", [100_000])
    k = cfg.get("utilities", {}).get("terms", 29)
    a = cfg.get("utilities", {}).get("alternatives", 5)
    rng = np.random.default_rng(0)
    for n in sizes:
        ids = np.arange(1, n + 1, dtype=np.int64)
        for impl in implementations("random"):
            prof = Profiler()
            with prof.step("random"):
                impl.fn("benchmark_step", 0).uniform("households", ids)
            rows.append(_bench_row("random", impl, n, prof))
        terms = [rng.normal(size=n) for _ in range(k)]
        coef = rng.normal(size=(k, a))
        for impl in implementations("utilities"):
            prof = Profiler()
            with prof.step("utilities"):
                impl.fn(iter(terms), coef, n)
            rows.append(_bench_row("utilities", impl, n, prof))
        del terms
    return pd.DataFrame(rows)


def _bench_row(seam, impl, n, prof):
    r = prof.records[-1]
    return {"seam": seam, "implementation": impl.name,
            "agreement": "reference" if impl.reference else impl.agreement,
            "rows": n, "seconds": r.wall_s, "extra_mb": r.step_extra_mb}


# --- report -----------------------------------------------------------------------


def write_report(out: Path, step: str, exp: pd.DataFrame, bench: pd.DataFrame) -> Path:
    s = [f"# Experiments: `{step}`", ""]
    s.append("Each experiment swaps some pieces for alternatives; the rest use the reference "
             "implementation. Match: run from ActivitySim 1.x's saved state before the step and "
             "compared with 1.x. Timing: fresh processes from the same saved state "
             "(see profile_step.py), alongside the 1.x baselines.")
    s += ["", "## Match and timing", ""]
    cols = ["experiment", "pieces_changed", "page_cache", "skims_resident", "result_vs_1x",
            "rows_different", "rows", "largest_share_diff_pp", "sampling_error_2se_pp",
            "max_utility_diff", "median_s", "min_s", "max_s", "median_extra_mb",
            "median_own_extra_mb", "median_file_peak_mb", "median_jax_compile_s", "runs"]
    table = exp[cols].copy()
    table["max_utility_diff"] = table["max_utility_diff"].map(
        lambda v: "" if pd.isna(v) else f"{v:.1e}")
    for c in ["rows_different", "rows", "runs"]:
        table[c] = table[c].map(lambda v: "" if pd.isna(v) else str(int(v)))
    for c in ["largest_share_diff_pp", "sampling_error_2se_pp", "skims_resident"]:
        table[c] = table[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    s.append(md_table(table, "{:.3f}"))
    s += ["", "`page_cache`: the skim file's state in the operating system's page cache when "
          "each run starts (`cold`: dropped, `warm`: read once before; 1.x holds its skims in "
          "memory, loaded before timing). `skims_resident`: the share of the file in the cache "
          "then (median of the runs). `median_extra_mb`: the step's peak memory above where it "
          "started, counting pages of memory-mapped skims it read; `median_own_extra_mb`: the "
          "same for the process's own memory only; `median_file_peak_mb`: the most pages of "
          "mapped files it held (sampled; see profile_step.py).", "",
          "`largest_share_diff_pp`: largest difference, in percentage points, in the share "
          "of rows taking any one value. `sampling_error_2se_pp`: two standard errors of that "
          "difference if both runs drew independently, a rough line for \"within chance\"."]
    if len(bench):
        s += ["", "## Each piece on its own, synthetic data", ""]
        s.append(md_table(bench, "{:.4g}"))
    path = out / "experiments.md"
    path.write_text("\n".join(s) + "\n")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(HERE / "experiments.yaml"))
    p.add_argument("--list", action="store_true", help="list the swappable pieces and implementations")
    p.add_argument("--example-dir", default=str(REPO / "activitysim/examples/prototype_mtc"))
    p.add_argument("--configs", nargs="+", default=["configs"])
    p.add_argument("--asim1-out", default=str(REPO / "asim_core/work/asim1_legacy"))
    p.add_argument("--asim1-sharrow-out", default=str(REPO / "asim_core/work/asim1_sharrow"))
    p.add_argument("--sharrow-configs", nargs="+", default=["configs_sh", "configs"])
    p.add_argument("--out", default=str(REPO / "asim_core/work/experiments"))
    p.add_argument("--repeat", type=int, default=None, help="override the config's repeat")
    p.add_argument("--step", default=None, help="override the config's step")
    p.add_argument("--match-run", default=None,
                   help="1.x run to compare results with (default: --asim1-out); e.g. a run with "
                        "numexpr off, which asim_core matches exactly on large zone systems")
    p.add_argument("--page-cache", nargs="+", choices=["as_is", "cold", "warm"], default=["as_is"],
                   help="for asim_core: profile with the skim file in each of these states in the "
                        "OS page cache (cold: dropped first; warm: read once first)")
    p.add_argument("--no-baselines", action="store_true",
                   help="don't profile the 1.x baselines (e.g. when they were profiled already)")
    p.add_argument("--read-skim-cache", choices=["settings", "true", "false"], default="settings",
                   help="for asim_core: reuse an unchanged skim cache (preparing skims isn't timed)")
    args = p.parse_args(argv)

    if args.list:
        for i in implementations():
            tag = "reference" if i.reference else i.agreement
            print(f"{i.seam:14s} {i.name:14s} {tag:12s} {i.description}")
        return 0

    cfg = yaml.safe_load(open(args.config))
    step = args.step or cfg["step"]
    repeat = args.repeat or int(cfg.get("repeat", 5))
    example = Path(args.example_dir).resolve()
    configs = [example / c for c in args.configs]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    st = Asim1State(args.match_run or args.asim1_out)
    base_seed = int(load_top_settings(configs).get("rng_base_seed") or 0)
    comp = {c.name: c for c in build_components(configs)}[step]
    rsc = {"settings": None, "true": True, "false": False}[args.read_skim_cache]
    skims = skim_store(example, configs, read_skim_cache=rsc)

    # an experiment is its seams ({seam: implementation}), or {seams: ..., env: ...} when its
    # runs need environment variables (e.g. ASIM_JAX_CACHE_DIR)
    experiments, envs = {}, {}
    for n, spec in (cfg.get("experiments") or {"reference": {}}).items():
        spec = spec or {}
        if set(spec) and set(spec) <= {"seams", "env"}:
            experiments[n], envs[n] = Engine(spec.get("seams") or {}), {k: str(v) for k, v in
                                                                      (spec.get("env") or {}).items()}
        else:
            experiments[n], envs[n] = Engine(spec), {}
    # profiles first: the match checks map skim pages into this process, and pages a
    # process has mapped can't be dropped from the page cache for a cold run
    timed = {}
    for state in args.page_cache:
        for name, engine in experiments.items():
            print(f"profile {name} ({state}): {engine.changed or 'reference'}", flush=True)
            label = name if len(args.page_cache) == 1 else f"{name}_{state}"
            timed[name, state] = summarize_runs(profile(
                step, label, "asim_core", args.configs, Path(args.asim1_out), engine.changed,
                repeat, out, example, args.read_skim_cache, state, envs[name]))

    rows = []
    for name, engine in experiments.items():
        print(f"match {name}", flush=True)
        match = match_check(comp, st, engine, base_seed, out / "match" / name, skims)
        for state in args.page_cache:
            rows.append({"experiment": name,
                         "pieces_changed": ", ".join(f"{k}={v}" for k, v in engine.changed.items())
                         or "none",
                         "agreement": engine.agreement, "page_cache": state,
                         **match, **timed[name, state]})

    baselines = [] if args.no_baselines else [("1.x legacy (baseline)", args.configs, Path(args.asim1_out))]
    if not args.no_baselines and (Path(args.asim1_sharrow_out) / "pipeline.parquetpipeline").exists():
        baselines.append(("1.x sharrow (baseline)", args.sharrow_configs, Path(args.asim1_sharrow_out)))
    for label, cfgs, run_dir in baselines:
        print(f"baseline {label}", flush=True)
        safe = label.split(" (")[0].replace(" ", "_").replace(".", "")
        runs = profile(step, safe, "asim1", cfgs, run_dir, {}, repeat, out, example)
        rows.append({"experiment": label, "pieces_changed": "", "page_cache": "in memory",
                     "result_vs_1x": "",
                     "rows_different": np.nan, "rows": np.nan, "largest_share_diff_pp": np.nan,
                     "sampling_error_2se_pp": np.nan, "max_utility_diff": np.nan,
                     **summarize_runs(runs)})

    exp = pd.DataFrame(rows)
    exp.to_csv(out / "experiments.csv", index=False)
    print("seam benchmarks", flush=True)
    bench = seam_benchmarks(cfg["seam_benchmarks"]) if cfg.get("seam_benchmarks") else pd.DataFrame()
    bench.to_csv(out / "seam_benchmarks.csv", index=False)
    report = write_report(out, step, exp, bench)
    print(report.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
