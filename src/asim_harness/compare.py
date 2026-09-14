"""Diff two runs' summaries: per-category deltas and a one-line verdict per metric."""

from __future__ import annotations

from . import ledger, summarize

UNCHANGED_TOLERANCE = 0.005
VERDICTS = ("unchanged", "drifted", "missing")


def compare_summaries(a: dict, b: dict, threshold: float = UNCHANGED_TOLERANCE) -> dict:
    """Deltas are b - a for every share vector present in either summary."""
    names = sorted(set(summarize.share_metric_paths(a)) | set(summarize.share_metric_paths(b)))
    metrics = {}
    for name in names:
        try:
            va = summarize.get_metric(a, name)
        except KeyError:
            va = None
        try:
            vb = summarize.get_metric(b, name)
        except KeyError:
            vb = None
        if not isinstance(va, dict) or not isinstance(vb, dict):
            metrics[name] = {"verdict": "missing", "missing_in": "a" if va is None else "b",
                             "max_abs_delta": None, "worst_category": None, "deltas": {}}
            continue
        cats = sorted(set(va) | set(vb))
        deltas = {c: round(float(vb.get(c, 0.0)) - float(va.get(c, 0.0)), 6) for c in cats}
        worst = max(cats, key=lambda c: abs(deltas[c])) if cats else None
        max_abs = abs(deltas[worst]) if worst is not None else 0.0
        metrics[name] = {
            "verdict": "unchanged" if max_abs <= threshold else "drifted",
            "max_abs_delta": round(max_abs, 6),
            "worst_category": worst,
            "deltas": deltas,
            "n_a": summarize.metric_n(a, name),
            "n_b": summarize.metric_n(b, name),
        }
    counts_a, counts_b = a.get("counts", {}), b.get("counts", {})
    counts = {k: {"a": counts_a.get(k), "b": counts_b.get(k)} for k in ("households", "persons", "tours", "trips")}
    tally = {v: sum(1 for m in metrics.values() if m["verdict"] == v) for v in VERDICTS}
    return {
        "run_a": a.get("run_id"),
        "run_b": b.get("run_id"),
        "threshold_abs": threshold,
        "counts": counts,
        "n_unchanged": tally["unchanged"],
        "n_drifted": tally["drifted"],
        "n_missing": tally["missing"],
        "metrics": metrics,
    }


def compare_runs(run_a: str, run_b: str, threshold: float = UNCHANGED_TOLERANCE) -> dict:
    a = summarize.load_summary(ledger.resolve_run_id(run_a))
    b = summarize.load_summary(ledger.resolve_run_id(run_b))
    return compare_summaries(a, b, threshold)


def compact(result: dict, max_drifted: int = 25) -> dict:
    """Sized for a tool result: drifted metrics with their non-trivial deltas, the rest as counts."""
    drifted = []
    for name, m in sorted(result["metrics"].items(), key=lambda kv: -(kv[1].get("max_abs_delta") or 0)):
        if m["verdict"] == "drifted":
            drifted.append({
                "metric": name, "max_abs_delta": m["max_abs_delta"], "worst_category": m["worst_category"],
                "n_a": m.get("n_a"), "n_b": m.get("n_b"),
                "deltas": {k: v for k, v in sorted(m["deltas"].items(), key=lambda kv: -abs(kv[1])) if abs(v) > 0.0005},
            })
    return {
        "run_a": result["run_a"], "run_b": result["run_b"], "threshold_abs": result["threshold_abs"],
        "counts": result["counts"], "n_unchanged": result["n_unchanged"], "n_drifted": result["n_drifted"],
        "n_missing": result["n_missing"],
        "missing": [n for n, m in result["metrics"].items() if m["verdict"] == "missing"],
        "unchanged": [n for n, m in result["metrics"].items() if m["verdict"] == "unchanged"],
        "drifted": drifted[:max_drifted],
    }


def text(result: dict) -> str:
    c = result["counts"]
    lines = [
        f"compare {result['run_a']} (a) -> {result['run_b']} (b), deltas are b - a, unchanged within {result['threshold_abs']}",
        "  counts: " + ", ".join(f"{k} {v['a']} -> {v['b']}" for k, v in c.items()),
        f"  {result['n_unchanged']} unchanged, {result['n_drifted']} drifted, {result['n_missing']} missing",
    ]
    order = {"drifted": 0, "missing": 1, "unchanged": 2}
    for name, m in sorted(result["metrics"].items(), key=lambda kv: (order[kv[1]["verdict"]], -(kv[1].get("max_abs_delta") or 0))):
        if m["verdict"] == "missing":
            lines.append(f"  missing   {name:<42s} (absent in run {m.get('missing_in')})")
        else:
            worst = m["worst_category"]
            lines.append(f"  {m['verdict']:<9s} {name:<42s} max|Δ| {m['max_abs_delta']:.3f} ({worst} {m['deltas'].get(worst, 0.0):+.3f})")
    return "\n".join(lines)
