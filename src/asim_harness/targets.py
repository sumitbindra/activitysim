"""Targets: load targets/<model>.yaml, bootstrap it from a run, score a run against it.

Target file format (PLAN.md, Phase 2)::

    model: prototype_mtc
    source_run: <run_id>
    metrics:
      trip_mode_share.overall:
        tolerance_abs: 0.02
        targets: {<mode>: <share>, ...}

Scoring reads only a run's summary.json, never the raw tables.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import ledger, paths, summarize
from .jsonio import read_json, write_json

SCORECARD_NAME = "scorecard.json"
DEFAULT_TOLERANCE = 0.02
DEFAULT_TOLERANCES = {
    "trip_mode_share": 0.02,
    "tour_mode_share": 0.02,
    "auto_ownership_share": 0.03,
    "cdap_share": 0.03,
    "tour_frequency": 0.02,
}


class TargetsError(Exception):
    pass


def tolerance_for(metric_path: str) -> float:
    return DEFAULT_TOLERANCES.get(metric_path.split(".")[0], DEFAULT_TOLERANCE)


def load_targets(path: Path | None = None) -> dict:
    path = Path(path) if path else paths.targets_file()
    if not path.is_file():
        raise TargetsError(f"no targets file at {path}; create one with `asim targets bootstrap <run_id>`")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data.get("metrics"), dict) or not data["metrics"]:
        raise TargetsError(f"{path} has no 'metrics' mapping")
    data["_path"] = str(path)
    return data


def bootstrap(run_id: str, path: Path | None = None, force: bool = False) -> dict:
    """Write every share metric of a run's summary as targets with default tolerances."""
    run_id = ledger.resolve_run_id(run_id)
    path = Path(path) if path else paths.targets_file()
    if path.exists() and not force:
        raise TargetsError(f"{path} already exists; pass --force to overwrite it")
    summary = summarize.load_summary(run_id)
    metrics = {}
    for metric_path in summarize.share_metric_paths(summary):
        shares = summarize.get_metric(summary, metric_path)
        metrics[metric_path] = {
            "tolerance_abs": tolerance_for(metric_path),
            "n": summarize.metric_n(summary, metric_path),
            "targets": {str(k): float(v) for k, v in shares.items()},
        }
    doc = {
        "model": paths.MODEL_NAME,
        "source_run": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Bootstrapped from a harness run; there is no observed data for the example.\n")
        f.write("# tolerance_abs is the maximum absolute share difference allowed per category.\n")
        yaml.safe_dump(doc, f, sort_keys=False)
    doc["_path"] = str(path)
    return doc


def score(summary: dict, targets: dict) -> dict:
    """Per-metric deltas, max |delta|, pass/fail per metric, and an overall verdict."""
    results = {}
    for name, spec in targets["metrics"].items():
        tol = float(spec.get("tolerance_abs", tolerance_for(name)))
        target = {str(k): float(v) for k, v in (spec.get("targets") or {}).items()}
        try:
            actual = summarize.get_metric(summary, name)
        except KeyError:
            actual = None
        if not isinstance(actual, dict):
            results[name] = {"status": "missing", "passed": False, "tolerance_abs": tol,
                             "max_abs_delta": None, "worst_category": None, "deltas": {}, "n": None}
            continue
        actual = {str(k): float(v) for k, v in actual.items()}
        cats = sorted(set(target) | set(actual))
        deltas = {c: round(actual.get(c, 0.0) - target.get(c, 0.0), 6) for c in cats}
        worst = max(cats, key=lambda c: abs(deltas[c])) if cats else None
        max_abs = abs(deltas[worst]) if worst is not None else 0.0
        results[name] = {
            "status": "pass" if max_abs <= tol else "fail",
            "passed": bool(max_abs <= tol),
            "tolerance_abs": tol,
            "max_abs_delta": round(max_abs, 6),
            "worst_category": worst,
            "deltas": deltas,
            "actual": {c: round(actual.get(c, 0.0), 6) for c in cats},
            "target": {c: round(target.get(c, 0.0), 6) for c in cats},
            "n": summarize.metric_n(summary, name),
        }
    failed = [n for n, r in results.items() if not r["passed"]]
    return {
        "run_id": summary.get("run_id"),
        "model": targets.get("model"),
        "targets_file": targets.get("_path"),
        "source_run": targets.get("source_run"),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": not failed,
        "n_metrics": len(results),
        "n_failed": len(failed),
        "failed_metrics": failed,
        "checks_passed": bool((summary.get("checks") or {}).get("all_passed", True)),
        "metrics": results,
    }


def scorecard_path(run_id: str) -> Path:
    return paths.run_dir(run_id) / SCORECARD_NAME


def check_run(run_id: str, targets_path: Path | None = None) -> dict:
    """Score a run from its summary.json against the targets file and persist scorecard.json."""
    run_id = ledger.resolve_run_id(run_id)
    summary = summarize.load_summary(run_id)
    targets = load_targets(targets_path)
    card = score(summary, targets)
    card["run_id"] = run_id
    write_json(scorecard_path(run_id), card)
    ledger.upsert(ledger.row_for_run_dir(paths.run_dir(run_id)))
    return card


def load_scorecard(run_id: str) -> dict | None:
    run_id = ledger.resolve_run_id(run_id)
    path = scorecard_path(run_id)
    return read_json(path) if path.exists() else None


def finalize_hook(run_dir: Path, manifest: dict) -> None:
    """After a successful run with a summary, score it if a targets file exists."""
    run_dir = Path(run_dir)
    if manifest.get("status") != "succeeded":
        return
    summary_file = run_dir / summarize.SUMMARY_NAME
    if not summary_file.exists() or not paths.targets_file().is_file():
        return
    summary = read_json(summary_file)
    card = score(summary, load_targets())
    card["run_id"] = manifest.get("run_id")
    write_json(run_dir / SCORECARD_NAME, card)


def _categories(r: dict, min_abs: float = 0.0) -> dict:
    """{category: {share, target, delta}} sorted by |delta| desc; share/target absent on old scorecards."""
    actual, target = r.get("actual") or {}, r.get("target") or {}
    out = {}
    for cat, delta in sorted(r.get("deltas", {}).items(), key=lambda kv: -abs(kv[1])):
        if abs(delta) < min_abs:
            continue
        row = {"delta": delta}
        if cat in actual:
            row["share"] = actual[cat]
        if cat in target:
            row["target"] = target[cat]
        out[cat] = row
    return out


def compact(card: dict, detail: str = "failed", metric: str | None = None, max_metrics: int = 40) -> dict:
    """A scorecard sized for a tool result.

    detail="failed" (default): failed metrics carry per-category share/target/delta (deltas below
    0.0005 dropped), passing metrics are one line each. detail="all": every metric carries all
    categories. detail="summary": one line per metric, no categories (what get_run attaches).
    metric="<dotted name or prefix>" selects metrics and always gives all categories.
    """
    if detail not in ("failed", "all", "summary"):
        raise ValueError("detail must be 'failed', 'all' or 'summary'")
    items = sorted(card["metrics"].items(), key=lambda kv: -(kv[1].get("max_abs_delta") or 0))
    if metric:
        items = [(n, r) for n, r in items if n == metric or n.startswith(metric.rstrip(".") + ".")]
        if not items:
            raise KeyError(f"no metric named or starting with {metric!r}; metrics are: "
                           + ", ".join(sorted(card["metrics"])))
    metrics = []
    for name, r in items[:max_metrics]:
        row = {"metric": name, "status": r.get("status"), "passed": r.get("passed"),
               "max_abs_delta": r.get("max_abs_delta"), "tolerance_abs": r.get("tolerance_abs"),
               "worst_category": r.get("worst_category"), "n": r.get("n")}
        if metric or detail == "all":
            row["categories"] = _categories(r)
        elif detail == "failed" and not r.get("passed"):
            row["categories"] = _categories(r, min_abs=0.0005)
        metrics.append(row)
    return {
        "run_id": card.get("run_id"), "passed": card.get("passed"), "n_metrics": card.get("n_metrics"),
        "n_failed": card.get("n_failed"), "targets_file": card.get("targets_file"),
        "source_run": card.get("source_run"), "checks_passed": card.get("checks_passed"),
        "detail": "all" if (metric or detail == "all") else detail,
        "note": "delta = run share minus target share; check_targets(run_id, metric=<name or prefix>) or "
                "detail='all' gives share, target and delta for every category of a metric",
        "failed": [m for m in metrics if not m["passed"]],
        "passed_metrics": [m for m in metrics if m["passed"]],
    }


def text(card: dict) -> str:
    lines = [
        f"scorecard for {card.get('run_id')} vs {card.get('targets_file')} (source run {card.get('source_run')}): "
        f"{'PASS' if card.get('passed') else 'FAIL'} — {card.get('n_failed')} of {card.get('n_metrics')} metrics out of tolerance"
    ]
    if not card.get("checks_passed", True):
        lines.append("  sanity checks FAILED in the summary (see `asim summarize`)")
    rows = sorted(card["metrics"].items(), key=lambda kv: (kv[1]["passed"], -(kv[1].get("max_abs_delta") or 0)))
    for name, r in rows:
        if r.get("status") == "missing":
            lines.append(f"  MISSING {name}")
            continue
        worst = r.get("worst_category")
        delta = r["deltas"].get(worst, 0.0) if worst is not None else 0.0
        n = f" n={r['n']}" if r.get("n") is not None else ""
        lines.append(
            f"  {'FAIL' if not r['passed'] else 'pass'} {name:<42s} max|Δ| {r['max_abs_delta']:.3f} "
            f"({worst} {delta:+.3f}) tol {r['tolerance_abs']:.2f}{n}"
        )
    return "\n".join(lines)
