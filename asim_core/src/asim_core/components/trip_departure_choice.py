"""
Trip departure choice: when each trip leaves, given the leg durations trip
scheduling choice chose (1.x: ``abm/models/trip_departure_choice.py``; ARC).

As 1.x:

* a leg's first and last trips are fixed: the outbound leg starts at the
  tour's start and its last trip leaves ``outbound_duration`` later; the
  inbound leg's first trip leaves ``inbound_duration`` before the tour's
  end and its last trip at the end;
* for the trips in between (by direction, inbound first): every way of
  splitting the leg's duration into whole periods over its stops is a
  pattern; each stop's time gets a utility from the direction's spec
  column (the preprocessor adds the trip's free-flow time and purpose
  flags), a pattern's utility is the sum over its stops, and one pattern is
  drawn per leg (from the ``tour_legs`` random stream: leg ID ``tour_id +
  10**9`` outbound, ``+ 2 * 10**9`` inbound);
* each trip's departure is the leg's start plus the stop times before it.

The steps follow 1.x's pandas code (merges, grouped sums), so that sums
come out in the same order.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, MissingInputError, StepContext
from ..logit import FormulaValues
from ..spec.legacy import load_global_constants, load_interaction_model, load_top_settings, read_settings
from ..spec.translate import INDEX
from . import annotate
from .runtime import TableLookup, WrappedSkims, plain
from .trip_scheduling_choice import get_time_windows

STEP = "trip_departure_choice"
CHANNEL = "tour_legs"
MAX_TOUR_ID = int(1e9)  # 1.x's module constant (the step sets another value on the function, not the module)
PAD_UTILITY = -999.0
TRIP_COLUMNS = ["tour_id", "outbound", "trip_num", "trip_count", "origin", "destination", "purpose"]
TOUR_COLUMNS = ["start", "end", "outbound_duration", "inbound_duration"]


def tour_leg_ids(tour_id, outbound) -> np.ndarray:
    return np.asarray(tour_id, dtype=np.int64) + np.where(np.asarray(outbound, dtype=bool), MAX_TOUR_ID,
                                                          2 * MAX_TOUR_ID)


def trip_departure_choice(configs_dirs: Sequence[str | Path],
                          settings_file: str = "trip_departure_choice.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_settings(configs_dirs, settings_file)
    top = load_top_settings(configs_dirs)
    for key in ("alts_preprocessor", "annotate_trips", "annotate_tours"):
        if settings.get(key):
            raise NotImplementedError(f"{settings_file}: {key} is not supported yet")
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    spec_file = settings.get("SPECIFICATION", "trip_departure_choice.csv")
    specs = {d: load_interaction_model(configs_dirs, spec_file, settings.get("COEFFICIENTS"), constants,
                                       name=f"{STEP}_{d}", value_column=d) for d in ("outbound", "inbound")}
    terms = {d: spec.utility_terms() for d, spec in specs.items()}
    pre = settings.get("PREPROCESSOR")
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, pre, constants, top, STEP, "preprocessor",
                                                      skim_accessors=["od_skims", "do_skims"])
    shift = not top.get("skip_failed_choices", True)
    resolved: dict = {}

    def resolve_needs(available):
        trip_cols = set(available.get("trips", set()))
        missing = [c for c in TRIP_COLUMNS if c not in trip_cols]
        if missing:
            raise MissingInputError(f"trips lack {missing}")
        tour_cols = set(available.get("tours", set()))
        missing = [c for c in TOUR_COLUMNS if c not in tour_cols]
        if missing:
            raise MissingInputError(f"tours lack {missing} (made by trip_scheduling_choice)")
        df_cols = trip_cols | tour_cols
        needs: dict[str, set[str]] = {"trips": set(TRIP_COLUMNS), "tours": set(TOUR_COLUMNS)}
        skims: set[str] = set()
        bound = None
        targets: set[str] = set()
        if pre_spec is not None:
            bound = pre_spec.bind(df_cols, {t: available.get(t, set()) for t in pre_tables})
            if bound.n_random:
                raise NotImplementedError(f"{STEP}: preprocessor with random draws")
            targets = set(bound.targets)
            for c in bound.columns_used() - {INDEX}:
                needs["trips" if c in trip_cols else "tours"].add(c)
            for t, c in bound.tables_used():
                needs.setdefault(t, set()).add(c)
            skims |= {name for _, name, _ in bound.skim_uses()}
        for spec in specs.values():
            for c in spec.columns_used() - targets - {"stop_time_duration"}:
                if c in trip_cols:
                    needs["trips"].add(c)
                elif c in tour_cols:
                    needs["tours"].add(c)
                else:
                    raise MissingInputError(f"{STEP}: the spec reads '{c}', which trips_merged lacks")
            skims |= {name for _, name, _ in spec.skim_uses()}
        if skims:
            needs[SKIMS] = skims
        resolved.update(bound=bound, needs=needs)
        return {t: sorted(c) for t, c in needs.items()}

    def run(ctx: StepContext):
        trips = ctx.table_frame("trips")
        order = trips.index
        merged = trips.copy()
        pos = pd.Index(ctx.ids("tours")).get_indexer(merged.tour_id.to_numpy())
        for c in resolved["needs"]["tours"]:
            if c not in merged.columns:  # 1.x's trips_merged: the trip's own column wins
                merged[c] = np.asarray(ctx.column("tours", c))[pos]
        bound = resolved["bound"]
        if bound is not None and len(merged):
            c = {k: np.asarray(plain(merged[k].to_numpy())) for k in bound.columns_used() - {INDEX}}
            c[INDEX] = merged.index.to_numpy()
            s = WrappedSkims(ctx, {"od_skims": (merged.origin.to_numpy(), merged.destination.to_numpy()),
                                   "do_skims": (merged.destination.to_numpy(), merged.origin.to_numpy())})
            with np.errstate(all="ignore"):
                v = bound.function(c, s, len(merged), TableLookup(ctx), None, {})
            for target in bound.targets:
                merged[target] = annotate._update(merged[target].to_numpy() if target in merged else None,
                                                  np.asarray(v[target]))
        depart = stage_two(ctx, merged, terms, shift)
        return {"trips": {"depart": depart.reindex(order).to_numpy()}}

    return Component(STEP, run, provides={"trips": ["depart"]}, resolve_needs=resolve_needs,
                     description="trip departures: stop time patterns drawn per tour leg",
                     info={"specs": specs, "trace_table": "trips"})


def stage_two(ctx, trips: pd.DataFrame, terms, shift) -> pd.Series:
    """1.x's apply_stage_two_model."""
    if not trips.index.is_monotonic_increasing:
        trips = trips.sort_index()
    trips["trip_duration"] = np.where(trips.outbound, trips.outbound_duration, trips.inbound_duration)
    trips["depart"] = -1
    first, last = trips.trip_num == 1, trips.trip_count == trips.trip_num
    trips.loc[first & trips.outbound, "depart"] = trips["start"]
    trips.loc[first & ~trips.outbound, "depart"] = trips["end"] - trips.inbound_duration
    trips.loc[last & trips.outbound, "depart"] = trips["start"] + trips.outbound_duration
    trips.loc[last & ~trips.outbound, "depart"] = trips["end"]
    side = trips[~first & ~last]
    if side.empty:
        return trips["depart"].astype(int)
    windows = get_time_windows(side.trip_duration.max(), side.trip_count.max() - 1)
    chosen = []
    for is_outbound, segment in side.groupby("outbound"):  # inbound (False) first
        direction = "outbound" if is_outbound else "inbound"
        patterns = build_patterns(segment, windows)
        choices = choose_patterns(ctx, segment, patterns, terms[direction], shift)
        choices = pd.merge(choices.reset_index(), patterns.reset_index(), on=["tour_leg_id", "pattern_id"],
                           how="left")
        chosen.append(choices[["trip_id", "stop_time_duration"]].copy())
    chosen = pd.concat(chosen, sort=True).set_index("trip_id")
    trips["stop_time_duration"] = 0
    trips.update(chosen)
    trips.loc[trips.trip_num == 1, "stop_time_duration"] = trips["depart"]
    trips["stop_time_duration"] = trips.groupby(["tour_id", "outbound"])["stop_time_duration"].cumsum()
    trips.loc[trips.trip_num != trips.trip_count, "depart"] = trips["stop_time_duration"]
    return trips["depart"].astype(int)


def build_patterns(trips: pd.DataFrame, windows) -> pd.DataFrame:
    """1.x's build_patterns: every split of each leg's duration over its stops, a row per (pattern, trip)."""
    tours = trips.groupby(["tour_id"])[["trip_duration", "trip_count"]].first()
    max_trip_count = trips.trip_count.max() - 1
    patterns, sizes = [], []
    for duration, count in tours[["trip_duration", "trip_count"]].values:
        possible = windows[: count - 1, np.where(windows[: count - 1].sum(axis=0) == duration)[0]]
        possible = np.unique(possible, axis=1).transpose()
        filler = np.full((possible.shape[0], max_trip_count), np.nan)
        filler[: possible.shape[0], : possible.shape[1]] = possible
        patterns.append(filler)
        sizes.append(filler.shape[0])
    patterns = np.concatenate(patterns)
    names = ["_".join("%0.0f" % x for x in y[~np.isnan(y)]) for y in patterns]
    patterns = pd.DataFrame(index=np.repeat(tours.index, sizes), data=patterns)
    patterns.index.name = tours.index.name
    patterns["pattern_id"] = names
    patterns = patterns.melt(id_vars="pattern_id", value_name="stop_time_duration", var_name="trip_num",
                             ignore_index=False).reset_index()
    patterns = patterns[~patterns.stop_time_duration.isnull()].copy()
    patterns["trip_num"] = patterns["trip_num"] + 1
    patterns["stop_time_duration"] = patterns["stop_time_duration"].astype(int)
    patterns = pd.merge(patterns, trips.reset_index()[["tour_id", "trip_id", "trip_num", "outbound"]],
                        on=["tour_id", "trip_num"])
    patterns.index = tour_leg_ids(patterns.tour_id, patterns.outbound)
    patterns.index.name = "tour_leg_id"
    return patterns


def choose_patterns(ctx, segment: pd.DataFrame, patterns: pd.DataFrame, terms, shift) -> pd.Series:
    """1.x's choose_tour_leg_pattern: a pattern per leg."""
    terms_fn, coefs, _ = terms
    # each trip's stop time options: 0 .. its leg duration (1.x's generate_alternatives)
    durations = np.where(segment.outbound, segment.outbound_duration, segment.inbound_duration).astype(int)
    alts = pd.DataFrame({"stop_time_duration": np.concatenate([np.arange(0, d + 1) for d in durations])},
                        index=pd.Index(np.repeat(segment.index.to_numpy(), durations + 1), name="trip_id"))
    alts = alts.sort_index(kind="stable") if not alts.index.is_monotonic_increasing else alts
    rows = segment.index.get_indexer(alts.index)
    n = len(alts)
    c = {"stop_time_duration": alts.stop_time_duration.to_numpy()}
    for col in segment.columns:
        if col not in c:
            c[col] = np.asarray(plain(segment[col].to_numpy()))[rows]
    with np.errstate(all="ignore"):
        utils = ctx.engine.interaction_utilities(FormulaValues(terms_fn, c, n, None, None), coefs, n)
    utilities = pd.DataFrame({"stop_time_duration": alts.stop_time_duration.to_numpy(), "utility": utils},
                             index=alts.index)
    utilities = pd.merge(utilities.reset_index(), patterns[patterns.trip_id.isin(utilities.index)],
                         on=["trip_id", "stop_time_duration"], how="left")
    utilities = utilities.groupby(["tour_id", "outbound", "pattern_id"], as_index=False)[["utility"]].sum()
    utilities["tour_leg_id"] = tour_leg_ids(utilities.tour_id, utilities.outbound)
    choosers = utilities.set_index("tour_leg_id")
    counts = choosers.groupby(choosers.index).size().values
    width = counts.max()
    last_offsets = counts.cumsum()
    first_offsets = np.insert(last_offsets[:-1], 0, 0)
    padded = np.insert(choosers.utility.values, np.repeat(last_offsets, width - counts), PAD_UTILITY)
    padded = padded.reshape(-1, width)
    legs = choosers.index.unique()
    probs, bad = ctx.engine.probabilities(padded, shift=shift)
    if bad.any():
        raise ValueError(f"{STEP}: tour legs with no usable probabilities, e.g. {legs[bad][:5].tolist()}")
    positions = ctx.engine.choose(probs, ctx.random.uniform(CHANNEL, legs.to_numpy()))
    return choosers["pattern_id"].take(positions + first_offsets)
