"""
Summaries for checking and visualising a run (1.x: ``abm/models/summarize.py``;
MTC runs it).

Reads today's summarize.yaml and its expression file (``SPECIFICATION``).
As 1.x: the tables are merged as its ``persons_merged``,
``households_merged`` and ``tours_merged`` (each joined table's columns the
left one already has are dropped) and ``trips_merged`` (trips with their
tour's columns, suffixes ``_trip``/``_tour``); the preprocessor adds columns
to ``trips_merged``, with the trip's and the tour's skims; the settings'
``AGGREGATE`` (a mapped column) and ``BIN`` (manual breaks, quantiles,
spaced or equal intervals) add more. Then each expression is evaluated, in
order, as Python code over those tables (pandas), as 1.x evaluates it:
outputs starting with ``_`` are kept for later expressions, ``_del`` drops
them, the others are written to ``OUTPUT/<name>.csv``.

The expressions are the region's own code, run as 1.x runs it (Python
``eval``), so they are trusted like the rest of its configs.

Not supported yet (reported before the run): ``EXPORT_PIPELINE_TABLES``,
disaggregate accessibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, MissingInputError, StepContext
from ..skims import time_periods_from_settings
from ..spec.legacy import find_config_file, load_global_constants, load_top_settings, read_yaml
from ..spec.translate import INDEX
from .annotate import _update, load_preprocessor
from .runtime import TableLookup
from .tour_mode_choice import SkimLookup

STEP = "summarize"
TABLES = ["persons", "households", "land_use", "accessibility", "tours", "trips"]
# 1.x's wrap_skims: (origin role, destination role, period column) for each skim wrapper
SKIM_WRAPPERS = {
    "tour_odt_skims": ("o_tour", "d_tour", "start_tour_period"),
    "tour_dot_skims": ("d_tour", "o_tour", "end_tour_period"),
    "trip_odt_skims": ("o_trip", "d_trip", "trip_period"),
    "tour_od_skims": ("o_tour", "d_tour", None),
    "trip_od_skims": ("o_trip", "d_trip", None),
}
ZONE_COLUMNS = {"o_tour": "origin_tour", "d_tour": "destination_tour", "o_trip": "origin_trip",
                "d_trip": "destination_trip"}
PERIOD_COLUMNS = {"start_tour_period": "start", "end_tour_period": "end", "trip_period": "depart"}


def summarize(configs_dirs: Sequence[str | Path], settings_file: str = "summarize.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    if settings.get("EXPORT_PIPELINE_TABLES", True):
        raise NotImplementedError(f"{settings_file}: not supported yet: EXPORT_PIPELINE_TABLES (1.x's default: "
                                  f"True; set it to False)")
    output = settings.get("OUTPUT", "summarize")
    spec = pd.read_csv(find_config_file(configs_dirs, settings.get("SPECIFICATION", "summarize.csv")), comment="#")
    constants = dict(load_global_constants(configs_dirs))
    pre = settings.get("preprocessor")
    pre_spec = None
    if pre:
        # 1.x evaluates the preprocessor with every merged table in scope, trips_merged among them
        pre = dict(pre, TABLES=[])
        pre_spec, _ = load_preprocessor(configs_dirs, pre, constants, top, STEP, "preprocessor",
                                        skim_accessors=list(SKIM_WRAPPERS), df_aliases=["trips_merged"])
    periods = time_periods_from_settings(configs_dirs)
    meta = {t: settings[t] for t in ("persons", "persons_merged", "households", "households_merged", "trips",
                                     "trips_merged", "tours", "tours_merged", "land_use") if t in settings}
    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]):
        missing = [t for t in TABLES if t not in available]
        if missing:
            raise MissingInputError(f"needs the tables {missing}")
        if "disaggregate_accessibility" in available:
            raise NotImplementedError(f"{STEP}: not supported yet: disaggregate accessibility")
        needs = {t: set(available[t]) for t in TABLES}  # the expressions may read any column
        bound = None
        if pre_spec is not None:
            merged = _merged_columns(available)
            bound = pre_spec.bind(merged, {})
            if bound.n_random:
                raise NotImplementedError(f"{STEP}: a preprocessor with random draws")
            uses = bound.skim_uses()
            if any(how != "get" for _, _, how in uses):
                raise NotImplementedError(f"{STEP}: only skim lookups by [key] in the preprocessor")
            skims = set()
            for accessor, key, _ in uses:
                if SKIM_WRAPPERS[accessor][2] is None:
                    skims.add(key)
                else:
                    skims |= {f"{key}__{label}" for label in periods.unique_labels}
            if skims:
                needs[SKIMS] = skims
        resolved["bound"] = bound
        return {t: sorted(c) for t, c in needs.items()}

    def run(ctx: StepContext):
        frames = {t: ctx.table_frame(t) for t in TABLES}
        persons, households, land_use = frames["persons"], frames["households"], frames["land_use"]
        households_merged = simple_table_join(simple_table_join(households, land_use, "home_zone_id"),
                                              frames["accessibility"], "home_zone_id")
        persons_merged = simple_table_join(persons, households_merged, "household_id")
        tours_merged = simple_table_join(frames["tours"], persons_merged, "person_id")
        trips = frames["trips"]
        trips_merged = pd.merge(trips, tours_merged.drop(columns=["person_id", "household_id"]), left_on="tour_id",
                                right_index=True, suffixes=("_trip", "_tour"), how="left")
        locals_d = {"persons": persons, "persons_merged": persons_merged, "households": households,
                    "households_merged": households_merged, "trips": trips, "trips_merged": trips_merged,
                    "tours": tours_merged, "tours_merged": tours_merged, "land_use": land_use}
        for col, source in PERIOD_COLUMNS.items():  # 1.x's wrap_skims labels the periods first
            trips_merged[col] = periods.labels_for(trips_merged[source].to_numpy())
        bound = resolved["bound"]
        if bound is not None and len(trips_merged):
            _preprocess(ctx, bound, trips_merged, periods)
        for table_name, df in locals_d.items():
            spec_meta = meta.get(table_name)
            if not spec_meta:
                continue
            for agg in spec_meta.get("AGGREGATE") or []:
                df[agg["label"]] = df[agg["column"]].map(agg["map"]).fillna(df[agg["column"]])
            for slicer in spec_meta.get("BIN") or []:
                df[slicer["label"]] = _bin(df[slicer["column"]], slicer)
        locals_d.update(quantiles=quantiles, spaced_intervals=spaced_intervals, equal_intervals=equal_intervals,
                        manual_breaks=manual_breaks)
        scope = {"np": np, "pd": pd}
        for _, row in spec.iterrows():
            out_file, expr = row["Output"], row["Expression"]
            if out_file == "_del":
                for var in str(expr).split(","):
                    locals_d.pop(var.strip(), None)
                continue
            if out_file.startswith("_"):
                locals_d[out_file] = eval(expr, scope, locals_d)  # the region's expression, as 1.x
                continue
            result = eval(expr, scope, locals_d)
            result.to_csv(ctx.output_path(f"{output}/{out_file}.csv"), index=False)
        return {}

    return Component(STEP, run, provides={}, resolve_needs=resolve_needs,
                     description=f"{sum(~spec.Output.str.startswith('_'))} summaries to {output}/")


def simple_table_join(left: pd.DataFrame, right: pd.DataFrame, left_on: str) -> pd.DataFrame:
    """1.x's simple_table_join: an inner merge on the key, the right table's duplicate columns dropped."""
    intersection = set(left.columns).intersection(right.columns)
    intersection.discard(left_on)
    return pd.merge(left, right.drop(intersection, axis=1), left_on=left_on, right_index=True)


def _merged_columns(available: Mapping[str, set[str]]) -> set[str]:
    """The columns trips_merged has (for binding the preprocessor), worked out from the tables'."""
    persons, households = available["persons"], available["households"]
    hh = households | (available["land_use"] - households) | (available["accessibility"] - households)
    per = persons | (hh - persons)
    tours = available["tours"] | (per - available["tours"])
    trips = available["trips"]
    tour_side = tours - {"person_id", "household_id"}
    both = trips & tour_side - {"tour_id"}
    return ((trips - both) | (tour_side - both) | {f"{c}_trip" for c in both} | {f"{c}_tour" for c in both}
            | set(PERIOD_COLUMNS))


def _preprocess(ctx: StepContext, bound, trips_merged: pd.DataFrame, periods) -> None:
    """The preprocessor's columns, added to trips_merged (1.x keeps those not starting with '_')."""
    cols = {k: trips_merged[k].to_numpy() for k in bound.columns_used() - {INDEX}}
    cols[INDEX] = trips_merged.index.to_numpy()
    zones = {role: ctx.skim_positions(trips_merged[col].to_numpy().astype(np.int64))
             for role, col in ZONE_COLUMNS.items() if col in trips_merged}
    labels = {col: trips_merged[col].to_numpy() for col in PERIOD_COLUMNS}
    skims = SkimLookup(ctx, periods, None, None, labels, wrappers=SKIM_WRAPPERS, zones=zones)
    with np.errstate(all="ignore"):  # 1.x logs numpy warnings and carries on
        v = bound.function(cols, skims, len(trips_merged), TableLookup(ctx), None, {})
    for target in bound.targets:
        if not target.startswith("_"):
            values = _update(trips_merged[target].to_numpy() if target in trips_merged else None,
                             np.asarray(v[target]))
            trips_merged[target] = values if np.ndim(values) else np.full(len(trips_merged), values)


# --- 1.x's binning functions (the expressions call them too) ---------------------------

DEFAULT_BIN_LABEL_FORMAT = "{left:,.2f} - {right:,.2f}"


def _bin(data: pd.Series, slicer: Mapping) -> pd.Series:
    kind = slicer["type"]
    if kind == "manual_breaks":
        return manual_breaks(data, slicer["bin_breaks"], slicer["bin_labels"])
    if kind == "quantiles":
        return quantiles(data, slicer["bins"], slicer["label_format"])
    if kind == "spaced_intervals":
        return spaced_intervals(data, slicer["lower_bound"], slicer["interval"], slicer["label_format"])
    if kind == "equal_intervals":
        return equal_intervals(data, slicer["bins"], slicer["label_format"])
    raise ValueError(f"summarize: unknown BIN type '{kind}'")


def construct_bin_labels(bins: pd.Series, label_format: str) -> pd.Series:
    left = bins.apply(lambda x: x.left)
    mid = bins.apply(lambda x: x.mid)
    right = bins.apply(lambda x: x.right)
    rank = mid.map({x: sorted(mid.unique().tolist()).index(x) + 1 if pd.notnull(x) else np.nan
                    for x in mid.unique()}, na_action="ignore")

    def construct_label(fmt, bounds):
        return fmt.format(**{x: b for x, b in bounds.items() if x in fmt})

    labels = pd.Series([construct_label(label_format, {"left": lt, "mid": md, "right": rt, "rank": rk})
                        for lt, md, rt, rk in zip(left, mid, right, rank)], index=bins.index)
    try:  # 1.x: pd.to_numeric(labels, errors="ignore")
        return pd.to_numeric(labels)
    except (ValueError, TypeError):
        return labels


def quantiles(data: pd.Series, bins, label_format: str = DEFAULT_BIN_LABEL_FORMAT) -> pd.Series:
    vals = data.sort_values()
    ranks = vals.rank(method="first")
    return construct_bin_labels(pd.qcut(ranks, bins, duplicates="drop"), label_format)


def spaced_intervals(data: pd.Series, lower_bound, interval, label_format: str = DEFAULT_BIN_LABEL_FORMAT):
    if lower_bound == "min":
        lower_bound = data.min()
    breaks = np.arange(lower_bound, data.max() + interval, interval)
    return construct_bin_labels(pd.cut(data, breaks, include_lowest=True), label_format)


def equal_intervals(data: pd.Series, bins, label_format: str = DEFAULT_BIN_LABEL_FORMAT) -> pd.Series:
    return construct_bin_labels(pd.cut(data, bins, include_lowest=True), label_format)


def manual_breaks(data: pd.Series, bin_breaks, labels=None, label_format: str = DEFAULT_BIN_LABEL_FORMAT):
    if isinstance(labels, list):
        return pd.cut(data, bin_breaks, labels=labels, include_lowest=True)
    return construct_bin_labels(pd.cut(data, bin_breaks, include_lowest=True), label_format)
