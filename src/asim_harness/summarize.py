"""Reduce a run's final output tables to a small, JSON-serialisable dict of metrics.

Everything model-specific (file names, column names, category labels) is in
the constants at the top so an agency model can be remapped in one place.
Shares are rounded to SHARE_DECIMALS for output; the "sums to one" check is
made on the unrounded values.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import ledger, paths
from .jsonio import read_json, write_json

SUMMARY_NAME = "summary.json"
SHARE_DECIMALS = 6
SUM_TOLERANCE = 1e-6
MAX_TOURS_BIN = 3  # tour-count distributions are binned 0, 1, 2, "3+"

# ---- model-specific names (prototype_mtc, ActivitySim 1.4 output_tables prefix "final_") ----
TABLE_FILES = {
    "households": "final_households.csv",
    "persons": "final_persons.csv",
    "tours": "final_tours.csv",
    "trips": "final_trips.csv",
}
COLS = {
    "households": {"id": "household_id", "auto_ownership": "auto_ownership"},
    "persons": {"id": "person_id", "cdap": "cdap_activity"},
    "tours": {
        "id": "tour_id", "person_id": "person_id", "mode": "tour_mode",
        "purpose": "primary_purpose", "category": "tour_category", "destination": "destination",
    },
    "trips": {
        "id": "trip_id", "mode": "trip_mode", "purpose": "purpose",
        "tour_id": "tour_id", "destination": "destination",
    },
}
TOUR_CATEGORIES = {"mandatory": "mandatory", "non_mandatory": "non_mandatory"}
NULL_LABEL = "null"


class SummaryError(Exception):
    pass


def _read(output_dir: Path, table: str, cols: list[str]) -> pd.DataFrame:
    path = Path(output_dir) / TABLE_FILES[table]
    if not path.is_file():
        raise SummaryError(f"missing output table {path}")
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in cols if c not in header]
    if missing:
        raise SummaryError(f"{path.name} lacks expected column(s) {missing}; columns are {list(header)}")
    return pd.read_csv(path, usecols=cols)


def _label(value) -> str:
    if pd.isna(value):
        return NULL_LABEL
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _shares(series: pd.Series) -> tuple[dict[str, float], float]:
    """(rounded share dict sorted by label, unrounded sum)."""
    if len(series) == 0:
        return {}, 0.0
    counts = series.map(_label).value_counts()
    total = float(counts.sum())
    raw = {k: v / total for k, v in counts.items()}
    return {k: round(raw[k], SHARE_DECIMALS) for k in sorted(raw)}, float(sum(raw.values()))


def _shares_by(df: pd.DataFrame, value_col: str, by_col: str) -> tuple[dict[str, dict[str, float]], list[float]]:
    out: dict[str, dict[str, float]] = {}
    sums: list[float] = []
    for key, group in df.groupby(df[by_col].map(_label), sort=True):
        shares, total = _shares(group[value_col])
        out[str(key)] = shares
        sums.append(total)
    return out, sums


def _counts_by(series: pd.Series) -> dict[str, int]:
    counts = series.map(_label).value_counts()
    return {k: int(counts[k]) for k in sorted(counts.index)}


def _tours_per_person(tours: pd.DataFrame, persons: pd.DataFrame, category: str) -> tuple[dict[str, float], float]:
    c = COLS["tours"]
    per_person = (
        tours[tours[c["category"]] == category]
        .groupby(c["person_id"])[c["id"]].size()
        .reindex(persons[COLS["persons"]["id"]], fill_value=0)
    )
    binned = per_person.clip(upper=MAX_TOURS_BIN).map(lambda n: f"{MAX_TOURS_BIN}+" if n >= MAX_TOURS_BIN else str(int(n)))
    return _shares(binned)


def summarize_output(output_dir: Path) -> dict:
    """Compute the metrics dict from the final tables in an ActivitySim output directory."""
    output_dir = Path(output_dir)
    hh_c, pp_c, tt_c, tr_c = COLS["households"], COLS["persons"], COLS["tours"], COLS["trips"]
    hh = _read(output_dir, "households", [hh_c["id"], hh_c["auto_ownership"]])
    pp = _read(output_dir, "persons", [pp_c["id"], pp_c["cdap"]])
    tt = _read(output_dir, "tours", [tt_c["id"], tt_c["person_id"], tt_c["mode"], tt_c["purpose"],
                                     tt_c["category"], tt_c["destination"]])
    tr = _read(output_dir, "trips", [tr_c["id"], tr_c["mode"], tr_c["purpose"], tr_c["tour_id"], tr_c["destination"]])

    sums: list[float] = []
    auto, s = _shares(hh[hh_c["auto_ownership"]]); sums.append(s)
    cdap, s = _shares(pp[pp_c["cdap"]]); sums.append(s)
    tour_overall, s = _shares(tt[tt_c["mode"]]); sums.append(s)
    tour_by_purpose, ss = _shares_by(tt, tt_c["mode"], tt_c["purpose"]); sums += ss
    trip_overall, s = _shares(tr[tr_c["mode"]]); sums.append(s)
    trip_by_purpose, ss = _shares_by(tr, tr_c["mode"], tr_c["purpose"]); sums += ss
    mand, s = _tours_per_person(tt, pp, TOUR_CATEGORIES["mandatory"]); sums.append(s)
    nonmand, s = _tours_per_person(tt, pp, TOUR_CATEGORIES["non_mandatory"]); sums.append(s)

    checks = {
        "no_null_tour_modes": bool(tt[tt_c["mode"]].notna().all()),
        "no_null_trip_modes": bool(tr[tr_c["mode"]].notna().all()),
        "no_null_tour_destinations": bool(tt[tt_c["destination"]].notna().all()),
        "no_null_trip_destinations": bool(tr[tr_c["destination"]].notna().all()),
        "every_trip_has_tour": bool(tr[tr_c["tour_id"]].isin(tt[tt_c["id"]]).all()),
        "shares_sum_to_one": all(abs(x - 1.0) <= SUM_TOLERANCE for x in sums if x),
    }
    checks["all_passed"] = all(checks.values())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "counts": {
            "households": int(len(hh)),
            "persons": int(len(pp)),
            "tours": int(len(tt)),
            "trips": int(len(tr)),
            "tours_by_purpose": _counts_by(tt[tt_c["purpose"]]),
            "trips_by_purpose": _counts_by(tr[tr_c["purpose"]]),
        },
        "auto_ownership_share": auto,
        "cdap_share": cdap,
        "tour_mode_share": {"overall": tour_overall, "by_purpose": tour_by_purpose},
        "trip_mode_share": {"overall": trip_overall, "by_purpose": trip_by_purpose},
        "tour_frequency": {"mandatory": mand, "non_mandatory": nonmand},
        "checks": checks,
    }


def summary_path(run_id: str) -> Path:
    return paths.run_dir(run_id) / SUMMARY_NAME


def summarize_run(run_id: str, force: bool = False) -> dict:
    """Compute (or load, unless force) and persist runs/<id>/summary.json."""
    run_id = ledger.resolve_run_id(run_id)
    path = summary_path(run_id)
    if path.exists() and not force:
        return read_json(path)
    summary = summarize_output(paths.output_dir(run_id))
    summary["run_id"] = run_id
    write_json(path, summary)
    ledger.upsert(ledger.row_for_run_dir(paths.run_dir(run_id)))
    return summary


def load_summary(run_id: str) -> dict:
    """Read summary.json only (never the raw tables); raise if it is missing."""
    run_id = ledger.resolve_run_id(run_id)
    path = summary_path(run_id)
    if not path.exists():
        raise SummaryError(f"run {run_id} has no {SUMMARY_NAME} (did it succeed? try `asim summarize {run_id}`)")
    return read_json(path)


def finalize_hook(run_dir: Path, manifest: dict) -> None:
    """Called by the runner after every run: write summary.json for successful runs."""
    if manifest.get("status") != "succeeded":
        return
    summary = summarize_output(Path(run_dir) / "output")
    summary["run_id"] = manifest.get("run_id")
    write_json(Path(run_dir) / SUMMARY_NAME, summary)


SHARE_METRIC_ROOTS = ("auto_ownership_share", "cdap_share", "tour_mode_share", "trip_mode_share", "tour_frequency")


def share_metric_paths(summary: dict) -> list[str]:
    """Dotted paths of every share vector in a summary (the things targets and compare work on).

    A share vector is a dict whose values are all numbers; nested dicts
    ("by_purpose") are walked one level at a time.
    """
    paths_out: list[str] = []

    def walk(node, prefix: str):
        if not isinstance(node, dict) or not node:
            return
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in node.values()):
            paths_out.append(prefix)
            return
        for key, child in node.items():
            walk(child, f"{prefix}.{key}")

    for root in SHARE_METRIC_ROOTS:
        if root in summary:
            walk(summary[root], root)
    return paths_out


def metric_n(summary: dict, dotted: str) -> int | None:
    """How many records a share vector was computed from (for judging sampling noise)."""
    c = summary.get("counts", {})
    parts = dotted.split(".")
    root = parts[0]
    if root == "auto_ownership_share":
        return c.get("households")
    if root in ("cdap_share", "tour_frequency"):
        return c.get("persons")
    if root in ("tour_mode_share", "trip_mode_share"):
        unit = "tours" if root.startswith("tour") else "trips"
        if len(parts) >= 3 and parts[1] == "by_purpose":
            return (c.get(f"{unit}_by_purpose") or {}).get(parts[2])
        return c.get(unit)
    return None


def get_metric(summary: dict, dotted: str):
    """Resolve 'trip_mode_share.overall' style paths; KeyError if absent."""
    node = summary
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def compact_text(summary: dict, top: int = 8) -> str:
    """A short human-readable rendering for the CLI."""
    lines = []
    c = summary.get("counts", {})
    lines.append(f"run {summary.get('run_id', '?')}: {c.get('households')} households, {c.get('persons')} persons, "
                 f"{c.get('tours')} tours, {c.get('trips')} trips")

    def fmt(d: dict) -> str:
        items = sorted(d.items(), key=lambda kv: -kv[1])[:top]
        rest = len(d) - len(items)
        text = ", ".join(f"{k} {v:.3f}" for k, v in items)
        return text + (f", +{rest} more" if rest > 0 else "")

    lines.append(f"auto ownership: {fmt(summary.get('auto_ownership_share', {}))}")
    lines.append(f"cdap: {fmt(summary.get('cdap_share', {}))}")
    lines.append(f"tour modes: {fmt(summary.get('tour_mode_share', {}).get('overall', {}))}")
    lines.append(f"trip modes: {fmt(summary.get('trip_mode_share', {}).get('overall', {}))}")
    tf = summary.get("tour_frequency", {})
    lines.append(f"tours/person mandatory: {fmt(tf.get('mandatory', {}))}; non-mandatory: {fmt(tf.get('non_mandatory', {}))}")
    checks = summary.get("checks", {})
    failed = [k for k, v in checks.items() if k != "all_passed" and not v]
    lines.append("checks: " + ("all passed" if checks.get("all_passed") else f"FAILED {failed}"))
    return "\n".join(lines)
