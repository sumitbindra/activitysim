#!/usr/bin/env python
"""
Milestone 1: check asim_core against ActivitySim 1.x and profile both.

Quick correctness check after a change (a few minutes):

    python asim_core/harness/run_milestone1.py --profiles none

Profiles are cached by default: a (step, engine) is measured again only
when something it depends on changed, so 1.x is measured once.

Needs a 1.x run with every step saved (see run_asim1.py). Then:

1. Step check: for each ported component, start from 1.x's saved state just
   before that step, run the component (``--repeat`` times, profiled), and
   compare its output with 1.x's output for the same step.
2. Utility check: compare utilities and probabilities for every chooser with
   the values 1.x's own functions compute from the same saved state.
3. End to end: run 1.x's full step list with ported components computed and
   every other step replayed from the 1.x run; compare at the end.
4. Profiles: put 1.x's per-step runtime and memory next to asim_core's.

Results go to ``--out``: comparisons.csv, utility_check.csv,
profile_comparison.csv, asim1_profiles.csv, summary.md, plus run folders with
saved outputs, traces and the generated NumPy code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "asim_core" / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asim_core.asim1 import Asim1State  # noqa: E402
from asim_core.compare import compare_column, value_shares, write_comparisons  # noqa: E402
from asim_core.inputs import check_inputs  # noqa: E402
from asim_core.components import (  # noqa: E402
    atwork_subtour_destination,
    atwork_subtour_frequency,
    atwork_subtour_mode_choice,
    atwork_subtour_scheduling,
    auto_ownership,
    cdap_simulate,
    compute_accessibility,
    free_parking,
    initialize_households,
    initialize_landuse,
    joint_tour_composition,
    joint_tour_destination,
    joint_tour_frequency,
    joint_tour_participation,
    joint_tour_scheduling,
    mandatory_tour_frequency,
    mandatory_tour_scheduling,
    non_mandatory_tour_destination,
    non_mandatory_tour_frequency,
    non_mandatory_tour_scheduling,
    school_location,
    stop_frequency,
    tour_mode_choice,
    trip_destination,
    trip_mode_choice,
    trip_purpose,
    trip_purpose_and_destination,
    trip_scheduling,
    workplace_location,
)
from asim_core.spec.translate import FormulaError  # noqa: E402
from asim_core.profiler import Profiler  # noqa: E402
from asim_core.runner import Runner  # noqa: E402
from asim_core.seams import Engine  # noqa: E402
from asim_core.skims import SkimStore, prepare_skims_from_settings  # noqa: E402
from asim_core.spec.legacy import load_top_settings  # noqa: E402
from asim_core.store import Store  # noqa: E402


PORTED = [initialize_landuse, initialize_households, compute_accessibility, school_location,
          workplace_location, auto_ownership, free_parking, cdap_simulate, mandatory_tour_frequency,
          mandatory_tour_scheduling, joint_tour_frequency, joint_tour_composition,
          joint_tour_participation, joint_tour_destination, joint_tour_scheduling,
          non_mandatory_tour_frequency, non_mandatory_tour_destination,
          non_mandatory_tour_scheduling, tour_mode_choice, atwork_subtour_frequency,
          atwork_subtour_destination, atwork_subtour_scheduling, atwork_subtour_mode_choice,
          stop_frequency, trip_purpose, trip_destination, trip_purpose_and_destination, trip_scheduling,
          trip_mode_choice]
# steps that read the input files: they also take the data folders (and the sample size 1.x ran with)
READS_INPUTS = {initialize_landuse, initialize_households}
NOT_BUILT: dict[str, str] = {}  # component -> why it could not be built for this example


def asim1_sample_size(asim1_run) -> int | None:
    """The households_sample_size a 1.x run was started with (None: settings.yaml's)."""
    p = Path(asim1_run) / "run_info.json" if asim1_run else None
    if p is None or not p.exists():
        return None
    return json.loads(p.read_text()).get("households_sample_size")


def build_components(configs, asim1_run=None, data_dirs=None):
    """
    The ported components for this example's configs. A component whose
    configs use something asim_core doesn't support yet is left out (that
    step is then replayed from 1.x), and the reason recorded in NOT_BUILT.
    Steps reading the inputs read ``data_dirs`` (default: the example's data
    folder) with the households sample size of the 1.x run they are compared with.
    """
    data_dirs = data_dirs or [Path(configs[-1]).parent / "data"]
    out = []
    for make in PORTED:
        try:
            if make is initialize_households:
                out.append(make(configs, data_dirs, households_sample_size=asim1_sample_size(asim1_run)))
            elif make in READS_INPUTS:
                out.append(make(configs, data_dirs))
            else:
                out.append(make(configs))
        except (NotImplementedError, FormulaError, ValueError) as err:
            NOT_BUILT[make.__name__] = str(err).splitlines()[0][:300]
            print(f"not ported for this example: {make.__name__}: {NOT_BUILT[make.__name__]}")
    return out


def skim_store(example: Path, configs, read_skim_cache: bool | None = None) -> SkimStore:
    """
    The example's skims, prepared as 1.x would from its network_los.yaml:
    imported from OMX on every run unless read_skim_cache is on, in which case
    an unchanged cache is reused. The cache lives in asim_core/work/<example>/cache.
    """
    return prepare_skims_from_settings(
        configs, [example / "data"], REPO / "asim_core" / "work" / example.name,
        read_skim_cache=read_skim_cache,
    )


def few_values(values, limit: int = 50) -> bool:
    """True for integer or categorical columns with few distinct values (shares make sense)."""
    arr = np.asarray(values)
    # pandas counts distinct values with missing ones (text columns can have None)
    return arr.dtype.kind in "biuO" and pd.Series(arr).nunique(dropna=False) <= limit


# settings needed to ask 1.x for its own utilities, per ported step
def _asim1_reference_settings(step):
    if step == "auto_ownership_simulate":
        from activitysim.abm.models.auto_ownership import AutoOwnershipSettings

        return AutoOwnershipSettings, "auto_ownership.yaml", "households_merged"
    return None


# --- 1. step check -------------------------------------------------------------


def step_check(comp, st: Asim1State, out: Path, base_seed: int, skims, trace_ids=None):
    prev = st.previous_step(comp.name)
    store = st.load_store(prev)
    table = comp.info.get("trace_table")
    if trace_ids is None and table in store:  # default: trace the first row
        trace_ids = {table: [store[table].index[0]]}
    report = Runner([comp], out, trace_ids=trace_ids, base_seed=base_seed, skims=skims).run(
        [comp.name], store, Profiler(), start_note=f"1.x state after {prev}"
    )
    STEP_WARNINGS.extend(report.warnings)
    comparisons = []
    shares = {}
    for t, cols in comp.provides.items():
        ref = st.read(t, comp.name)
        for col in cols:
            comparisons.append(compare_column(comp.name, store[t], col, ref[col]))
            if t != "person_windows" and few_values(store[t][col]):
                shares[f"{t}.{col}"] = value_shares(store[t][col], ref[col])
    if "generated_source" in comp.info:
        gen = out / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        (gen / f"{comp.name}.py").write_text(comp.info["generated_source"])
    return comparisons, shares


# warnings the steps reported in the step check (the end-to-end run's are in its report)
STEP_WARNINGS: list[dict] = []


# --- 2. utility check ----------------------------------------------------------


def utility_check(comp, st: Asim1State, example: Path, configs_names, out: Path):
    ref_settings = _asim1_reference_settings(comp.name)
    if ref_settings is None or "compute" not in comp.info:
        return []
    from asim1_reference import simple_model_internals

    settings_cls, settings_file, choosers_table = ref_settings
    prev = st.previous_step(comp.name)
    ref_utils, ref_probs = simple_model_internals(
        example, configs_names, st.path, prev, settings_cls, settings_file,
        choosers_table, out / "asim1_scratch",
    )
    store = st.load_store(prev)
    view = comp.info["view"]
    cols = view.gather_from_store(store, comp.info["columns"])
    ids = store[view.table].index
    utils, probs, _ = comp.info["compute"](cols, len(ids), Engine())
    rows = []
    for name, ours, ref in (
        ("utilities", utils, ref_utils.loc[ids].to_numpy()),
        ("probabilities", probs, ref_probs.loc[ids].to_numpy()),
    ):
        diff = np.abs(ours - ref)
        rows.append({
            "step": comp.name,
            "quantity": name,
            "values": int(ours.size),
            "bit_identical": int((ours == ref).sum()),
            "max_abs_diff": float(diff.max()),
        })
    return rows


# --- 3. end to end -------------------------------------------------------------


def end_to_end(components, st: Asim1State, out: Path, base_seed: int, skims):
    runner = Runner(components, out, replay=st, base_seed=base_seed, skims=skims)
    store = Store()
    report = runner.run(
        st.steps, store, Profiler(),
        start_note="empty store; ported steps computed, all other steps replayed from 1.x",
    )
    final = st.steps[-1]
    comparisons = list(report.carried_over)
    for comp in components:
        for t, cols in comp.provides.items():
            ref = st.read(t, final)
            for col in cols:
                if col not in ref.columns and col not in store[t].column_names:
                    continue  # dropped by a later step in both runs (e.g. trips' failed)
                comparisons.append(compare_column(f"end of run ({final})", store[t], col,
                                                  ref[col] if col in ref.columns else pd.Series(dtype=object)))
    return report, comparisons


# --- 4. profiles ----------------------------------------------------------------


def load_asim1_profile(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "asim_core_profile.csv"
    return pd.read_csv(p) if p.exists() else None


def _hash_files(paths) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path).encode())
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def _code_fingerprints() -> dict[str, str]:
    """What a profile depends on besides the step: the code of each engine and the harness."""
    return {
        "asim1": _hash_files((REPO / "activitysim").rglob("*.py")),
        "asim_core": _hash_files((REPO / "asim_core" / "src").rglob("*.py")),
        "harness": _hash_files([HERE / "profile_step.py"]),
    }


def _profile_key(step, label, engine, configs, run_dir, repeat, example, code) -> dict:
    checkpoints = Path(run_dir) / "pipeline.parquetpipeline" / "checkpoints.parquet"
    stat = checkpoints.stat() if checkpoints.exists() else None
    config_files = [f for c in configs for f in (example / c).rglob("*") if f.is_file()]
    return {
        "step": step, "label": label, "engine": engine, "repeat": repeat,
        "example": str(example), "configs": list(configs), "configs_hash": _hash_files(config_files),
        "asim1_run": str(run_dir), "asim1_run_saved": [stat.st_size, stat.st_mtime_ns] if stat else None,
        "engine_code": code[engine], "harness_code": code["harness"],
        "machine": [platform.platform(), os.cpu_count()],
    }


def profile_steps(steps, variants, repeat: int, out: Path, example: Path,
                  mode: str = "cached") -> pd.DataFrame:
    """
    Time each step in fresh processes (see profile_step.py), ``repeat`` times
    per variant. A variant is (label, engine, configs, 1.x run folder).

    ``mode="cached"`` reuses earlier measurements of a (step, variant) when
    nothing they depend on has changed: the engine's code, the harness, the
    configs, the saved 1.x run and the machine. So 1.x is measured once, and
    asim_core again only when its code changes. ``mode="all"`` measures
    everything again (and refreshes the cache).
    """
    import subprocess

    cache_dir = REPO / "asim_core" / "work" / "profile_cache" / example.name
    code = _code_fingerprints()
    frames = []
    for step in steps:
        for label, engine, configs, run_dir in variants:
            key = _profile_key(step, label, engine, configs, run_dir, repeat, example, code)
            cached_csv = cache_dir / label / f"{step}.csv"
            cached_key = cached_csv.with_suffix(".json")
            if (mode == "cached" and cached_csv.exists() and cached_key.exists()
                    and json.loads(cached_key.read_text()) == key):
                frames.append(pd.read_csv(cached_csv).assign(reused=True))
                continue
            cached_csv.parent.mkdir(parents=True, exist_ok=True)
            if cached_csv.exists():
                cached_csv.unlink()
            for k in range(repeat):
                cmd = [
                    sys.executable, str(HERE / "profile_step.py"),
                    "--engine", engine, "--step", step, "--label", label,
                    "--example-dir", str(example),
                    "--configs", *configs, "--asim1-out", str(run_dir),
                    "--scratch", str(out / "scratch" / label), "--out", str(cached_csv),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            cached_key.write_text(json.dumps(key, indent=2))
            frames.append(pd.read_csv(cached_csv).assign(reused=False))
    runs = pd.concat(frames, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "step_profile_runs.csv", index=False)
    return runs


def profile_tables(runs: pd.DataFrame, legacy: pd.DataFrame, sharrow: pd.DataFrame | None):
    g = runs.groupby(["step", "label"], sort=False)
    comparison = pd.DataFrame({
        "median_wall_s": g["wall_s"].median(),
        "min_wall_s": g["wall_s"].min(),
        "max_wall_s": g["wall_s"].max(),
        "median_extra_mb": g["step_extra_mb"].median(),
        "max_extra_mb": g["step_extra_mb"].max(),
        "median_skims_ready_s": g["skims_ready_s"].median(),
        "runs": g["wall_s"].size(),
    }).reset_index()
    # the same steps inside a full 1.x run, for reference
    in_run = []
    for _, r in comparison.iterrows():
        src = legacy if r["label"] == "asim1_legacy" else sharrow if r["label"] == "asim1_sharrow" else None
        if src is None:
            in_run.append(np.nan)
        else:
            in_run.append(float(src.set_index("step").loc[r["step"], "wall_s"]))
    comparison["wall_s_in_full_1x_run"] = in_run

    full = legacy[["step", "wall_s", "cpu_s", "peak_mb", "step_extra_mb"]].rename(
        columns={"wall_s": "legacy_wall_s", "cpu_s": "legacy_cpu_s",
                 "peak_mb": "legacy_peak_mb", "step_extra_mb": "legacy_extra_mb"})
    if sharrow is not None:
        sh = sharrow[["step", "wall_s", "cpu_s", "peak_mb", "step_extra_mb"]].rename(
            columns={"wall_s": "sharrow_wall_s", "cpu_s": "sharrow_cpu_s",
                     "peak_mb": "sharrow_peak_mb", "step_extra_mb": "sharrow_extra_mb"})
        full = full.merge(sh, on="step", how="outer", sort=False)
    return comparison, full


# --- summary -----------------------------------------------------------------------


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_summary(out: Path, info: dict, step_cmp, util_rows, e2e_cmp, e2e_report, prof_cmp,
                  full_prof, shares, inputs=None) -> Path:
    s = ["# Milestone 1 results", ""]
    s.append(f"Example: `{info['example']}`, configs {info['configs']}. "
             f"Machine: {info['platform']}, {info['cpu_count']} CPUs. "
             f"Python {info['python']}.")
    if inputs is not None:
        s += ["", "## Input checks", "", "```", inputs.text(show_undefined=False), "```"]
    warned = list({(w["step"], w["message"]): w for w in STEP_WARNINGS + e2e_report.warnings}.values())
    if warned:
        s += ["", "## Warnings from the steps", "",
              "Reported by the steps while they ran; they didn't stop the run.", ""]
        s += [f"- `{w['step']}`: {w['message']}" for w in warned]
    if NOT_BUILT:
        s += ["", "Not ported for this example (replayed from 1.x instead):", ""]
        s += [f"- `{k}`: {v}" for k, v in NOT_BUILT.items()]
    s += ["", "## 1. Step check (start from 1.x's saved state before the step)", ""]
    s.append(md_table(pd.DataFrame([c.__dict__ for c in step_cmp])))
    for name, df in shares.items():
        s += ["", f"Share of rows by value, `{name}`:", "", md_table(df.reset_index().rename(columns={"index": "value"}), "{:.4f}")]
    if util_rows:
        s += ["", "## 2. Utility and probability check (against 1.x's own functions)", ""]
        s.append(md_table(pd.DataFrame(util_rows), "{:.3g}"))
    s += ["", "## 3. End to end (ported steps computed, all others replayed from 1.x)", ""]
    modes = pd.Series([m for _, m in e2e_report.plan]).value_counts().to_dict()
    s.append(f"Steps: {len(e2e_report.plan)} ({', '.join(f'{v} {k}' for k, v in modes.items())}).")
    s += ["", md_table(pd.DataFrame([c.__dict__ for c in e2e_cmp]))]
    s += ["", "## 4. Profiles", "",
          "Ported steps, each run in a fresh process starting from 1.x's saved state. "
          "Not timed: loading the saved state, 1.x's one-time Numba compiles, and getting "
          "skims ready, which in a full run happen once, in earlier steps or at the first step "
          "that needs them. `median_skims_ready_s` is the time to get skims ready: 1.x loads "
          "them from OMX; asim_core opens its memory-mapped store. "
          "`wall_s_in_full_1x_run` is the same step inside the full 1.x run, for reference. "
          "`reused` is True where earlier measurements were reused because nothing they depend "
          "on changed (engine code, harness, configs, saved 1.x run, machine).", ""]
    s.append(md_table(prof_cmp) if len(prof_cmp) else "Not run (`--profiles none`).")
    if len(full_prof):
        s += ["", "ActivitySim 1.x, every step:", "", md_table(full_prof)]
    path = out / "summary.md"
    path.write_text("\n".join(s) + "\n")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--example-dir", default=str(REPO / "activitysim/examples/prototype_mtc"))
    p.add_argument("--configs", nargs="+", default=["configs"])
    p.add_argument("--asim1-out", default=str(REPO / "asim_core/work/asim1_legacy"))
    p.add_argument("--asim1-sharrow-out", default=str(REPO / "asim_core/work/asim1_sharrow"))
    p.add_argument("--sharrow-configs", nargs="+", default=["configs_sh", "configs"])
    p.add_argument("--out", default=str(REPO / "asim_core/work/milestone1"))
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--trace-hh-id", type=int, default=None)
    p.add_argument("--skip-utility-check", action="store_true")
    p.add_argument("--profiles", choices=["cached", "all", "none"], default="cached",
                   help="cached (default): reuse earlier measurements when the code, configs, "
                        "1.x run and machine are unchanged; all: measure everything again; "
                        "none: skip profiling (the quick correctness check)")
    p.add_argument("--skim-check-values", choices=["cache", "omx"], default="cache",
                   help="where the input checks read skim values: the run's skim cache (default; "
                        "the values the model uses, no second decompression) or the OMX files")
    p.add_argument("--model-files", action="store_true",
                   help="run the ported steps from asim_core's model files (asim_core/configs/<example>, "
                        "made by harness/check_model_files.py) instead of today's files")
    args = p.parse_args(argv)

    example = Path(args.example_dir).resolve()
    configs = [example / c for c in args.configs]
    if args.model_files:
        configs = [REPO / "asim_core/configs" / example.name] + configs
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    st = Asim1State(args.asim1_out)
    top = load_top_settings(configs)
    base_seed = int(top.get("rng_base_seed") or 0)
    components = build_components(configs, args.asim1_out)
    absent = [c.name for c in components if c.name not in st.steps]
    if absent:  # e.g. a 1.x run stopped early: nothing to compare these with
        print(f"not in the 1.x run, so not checked: {', '.join(absent)}")
        components = [c for c in components if c.name in st.steps]

    # skims first (imported from OMX unless an unchanged cache exists), so the input
    # checks read the cache instead of decompressing the OMX files a second time
    try:
        skims = skim_store(example, configs)
    except Exception as err:  # the checks below report what's wrong with the files
        print(f"skims could not be prepared ({err}); checking the OMX files")
        skims = None
    # the inputs are checked before anything runs; errors stop here
    ours = REPO / "asim_core/configs" / example.name
    inputs = check_inputs(([ours] if ours.is_dir() else []) + configs, [example / "data"],
                          cache_dir=REPO / "asim_core/work" / example.name / "input_checks",
                          skim_values=args.skim_check_values, skim_store=skims)
    (out / "input_checks.txt").write_text(inputs.text() + "\n")
    if inputs.problems:  # warnings are shown too: the run goes on, but they're worth a look
        print(inputs.text(show_undefined=False))
        print(f"(also in {out / 'input_checks.txt'} and summary.md)")
    else:
        print(inputs.summary())
    if not inputs.ok:
        return 2

    trace_ids = {"households": [args.trace_hh_id]} if args.trace_hh_id is not None else None

    step_cmp, util_rows, shares = [], [], {}
    for comp in components:
        ids = trace_ids if trace_ids and comp.info.get("trace_table") in trace_ids else None
        cmp, sh = step_check(comp, st, out / "step_check" / comp.name, base_seed, skims, ids)
        step_cmp += cmp
        shares.update(sh)
        if not args.skip_utility_check:
            util_rows += utility_check(comp, st, example, args.configs, out / "utility_check")

    e2e_report, e2e_cmp = end_to_end(components, st, out / "end_to_end", base_seed, skims)

    legacy = load_asim1_profile(Path(args.asim1_out))
    sharrow = load_asim1_profile(Path(args.asim1_sharrow_out))
    variants = [
        ("asim_core", "asim_core", args.configs, Path(args.asim1_out)),
        ("asim1_legacy", "asim1", args.configs, Path(args.asim1_out)),
    ]
    if sharrow is not None:
        variants.append(("asim1_sharrow", "asim1", args.sharrow_configs, Path(args.asim1_sharrow_out)))
    if args.profiles == "none":
        prof_cmp, full_prof = pd.DataFrame(), legacy if legacy is not None else pd.DataFrame()
        reused = None
    else:
        runs = profile_steps([c.name for c in components], variants, args.repeat, out / "step_profiles",
                             example, mode=args.profiles)
        prof_cmp, full_prof = profile_tables(runs, legacy, sharrow)
        reused = runs.groupby(["step", "label"])["reused"].first().reset_index()
        prof_cmp = prof_cmp.merge(reused, on=["step", "label"], how="left")

    write_comparisons(step_cmp + e2e_cmp, out / "comparisons.csv")
    pd.DataFrame(util_rows).to_csv(out / "utility_check.csv", index=False)
    prof_cmp.to_csv(out / "profile_comparison.csv", index=False)
    full_prof.to_csv(out / "asim1_profiles.csv", index=False)
    info = {
        "example": str(example.relative_to(REPO)) if example.is_relative_to(REPO) else str(example),
        "configs": args.configs,
        "trace_hh_id": args.trace_hh_id,
        "repeat": args.repeat,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "not_ported": dict(NOT_BUILT),
        "profiles": args.profiles,
        "input_checks": inputs.summary(),
        "model_files": args.model_files,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    summary = write_summary(out, info, step_cmp, util_rows, e2e_cmp, e2e_report, prof_cmp, full_prof, shares,
                            inputs)

    all_cmp = step_cmp + e2e_cmp
    ok = all(c.status == "exact" for c in all_cmp) and all(r["bit_identical"] == r["values"] for r in util_rows)
    print(summary.read_text())
    print("ALL EXACT" if ok else "SOME DIFFERENCES - see comparisons.csv")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
