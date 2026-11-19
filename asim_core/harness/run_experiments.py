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
  (``profile_step.py``), the same way as the 1.x baselines.
* **Seam benchmarks:** time each implementation of a piece on its own with
  synthetic data at larger row counts, to show how costs grow.

Writes ``experiments.csv``, ``seam_benchmarks.csv`` and ``experiments.md``
to ``--out``. Harness code: the baselines import ActivitySim 1.x.
"""

from __future__ import annotations

import argparse
import json
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
           "sampling_error_2se_pp": 0.0, "shares_within_noise": True}
    for table, cols in comp.provides.items():
        ref = st.read(table, comp.name)
        for col in cols:
            c = compare_column(comp.name, store[table], col, ref[col])
            row["rows"] += c.rows
            row["rows_different"] += c.mismatched
            row["max_output_diff"] = max(row.get("max_output_diff", 0.0), c.max_abs_diff or 0.0)
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
    elif engine.agreement == "statistical":
        row["result_vs_1x"] = ("same distribution (rows differ by design)"
                               if row["shares_within_noise"] else "distribution differs: check")
    else:
        row["result_vs_1x"] = "differs: check"
    return row


# --- profiles ---------------------------------------------------------------------


def profile(step, label, engine_kind, configs, run_dir, seams, repeat, out: Path) -> pd.DataFrame:
    rows_csv = out / "runs" / f"{label}.csv"
    rows_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows_csv.exists():
        rows_csv.unlink()
    for _ in range(repeat):
        cmd = [
            sys.executable, str(HERE / "profile_step.py"),
            "--engine", engine_kind, "--step", step, "--label", label,
            "--configs", *configs, "--asim1-out", str(run_dir),
            "--scratch", str(out / "scratch" / label), "--out", str(rows_csv),
            "--seams", json.dumps(seams),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    return pd.read_csv(rows_csv)


def summarize_runs(runs: pd.DataFrame) -> dict:
    return {
        "median_s": float(runs["wall_s"].median()),
        "min_s": float(runs["wall_s"].min()),
        "max_s": float(runs["wall_s"].max()),
        "median_extra_mb": float(runs["step_extra_mb"].median()),
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
    cols = ["experiment", "pieces_changed", "result_vs_1x", "rows_different", "rows",
            "largest_share_diff_pp", "sampling_error_2se_pp", "max_utility_diff",
            "median_s", "min_s", "max_s", "median_extra_mb", "runs"]
    table = exp[cols].copy()
    table["max_utility_diff"] = table["max_utility_diff"].map(
        lambda v: "" if pd.isna(v) else f"{v:.1e}")
    for c in ["rows_different", "rows", "runs"]:
        table[c] = table[c].map(lambda v: "" if pd.isna(v) else str(int(v)))
    for c in ["largest_share_diff_pp", "sampling_error_2se_pp"]:
        table[c] = table[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    s.append(md_table(table, "{:.3f}"))
    s += ["", "`largest_share_diff_pp`: largest difference, in percentage points, in the share "
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
    args = p.parse_args(argv)

    if args.list:
        for i in implementations():
            tag = "reference" if i.reference else i.agreement
            print(f"{i.seam:14s} {i.name:14s} {tag:12s} {i.description}")
        return 0

    cfg = yaml.safe_load(open(args.config))
    step = cfg["step"]
    repeat = args.repeat or int(cfg.get("repeat", 5))
    example = Path(args.example_dir).resolve()
    configs = [example / c for c in args.configs]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    st = Asim1State(args.asim1_out)
    base_seed = int(load_top_settings(configs).get("rng_base_seed") or 0)
    comp = {c.name: c for c in build_components(configs)}[step]
    skims = skim_store(example, configs)

    rows = []
    for name, seams in (cfg.get("experiments") or {"reference": {}}).items():
        engine = Engine(seams or {})
        print(f"experiment {name}: {engine.changed or 'reference'}", flush=True)
        row = {"experiment": name,
               "pieces_changed": ", ".join(f"{k}={v}" for k, v in engine.changed.items()) or "none",
               "agreement": engine.agreement}
        row.update(match_check(comp, st, engine, base_seed, out / "match" / name, skims))
        runs = profile(step, name, "asim_core", args.configs, Path(args.asim1_out),
                       engine.changed, repeat, out)
        row.update(summarize_runs(runs))
        rows.append(row)

    baselines = [("1.x legacy (baseline)", args.configs, Path(args.asim1_out))]
    if (Path(args.asim1_sharrow_out) / "pipeline.parquetpipeline").exists():
        baselines.append(("1.x sharrow (baseline)", args.sharrow_configs, Path(args.asim1_sharrow_out)))
    for label, cfgs, run_dir in baselines:
        print(f"baseline {label}", flush=True)
        safe = label.split(" (")[0].replace(" ", "_").replace(".", "")
        runs = profile(step, safe, "asim1", cfgs, run_dir, {}, repeat, out)
        rows.append({"experiment": label, "pieces_changed": "", "result_vs_1x": "",
                     "rows_different": np.nan, "rows": np.nan, "largest_share_diff_pp": np.nan,
                     "sampling_error_2se_pp": np.nan, "max_utility_diff": np.nan,
                     **summarize_runs(runs)})

    exp = pd.DataFrame(rows)
    exp.to_csv(out / "experiments.csv", index=False)
    print("seam benchmarks", flush=True)
    bench = seam_benchmarks(cfg.get("seam_benchmarks") or {})
    bench.to_csv(out / "seam_benchmarks.csv", index=False)
    report = write_report(out, step, exp, bench)
    print(report.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
