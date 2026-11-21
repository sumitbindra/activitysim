"""
Trip scheduling: a departure period for every trip, within its tour (1.x:
``abm/models/trip_scheduling.py`` and ``util/probabilistic_scheduling.py``,
departure mode).

As 1.x:

* each trip's window is its tour's (start to end), cut by any at-work
  subtours: a work tour's outbound trips leave before its first subtour
  starts, its inbound trips after its last one ends. ``tour_hour`` is the
  tour's start (outbound) or end (inbound);
* each direction is scheduled in turn: the first outbound trip leaves at
  the tour's start and the last inbound trip at its end (as do at-work
  subtours' trips); the others draw a period from
  ``trip_scheduling_probs.csv`` (the row matching primary purpose,
  direction, tour hour and trip number), with periods outside the trip's
  window given no probability. Outbound trips go in trip number order,
  each later trip's window starting at the earlier trip's departure;
  inbound trips go from the last one back, each earlier trip's window
  ending at the later one's. Only a direction's first scheduled trip has its
  probabilities scaled back up to 1; for the others, the probability
  outside the window is the chance of failing;
* trips of a direction with a failed trip are scheduled again (with the next
  draws on their streams), up to ``MAX_ITERATIONS`` times; on the last
  attempt a trip that still fails leaves at its window's edge (1.x's
  ``choose_most_initial``).

``logic_version`` 1 (1.x's default) and 2 are supported; ``stop_duration``
and ``relative`` scheduling, a preprocessor and ``drop_and_cleanup`` are not
yet.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import Component, MissingInputError, StepContext
from ..spec.legacy import find_config_file, read_settings
from .runtime import plain

STEP = "trip_scheduling"
CHANNEL = "trips"
JOIN_COLUMNS = ["primary_purpose", "outbound", "tour_hour", "trip_num"]  # 1.x: departure mode
NO_TRIP_ID = 0
TRIP_COLUMNS = ["tour_id", "person_id", "primary_purpose", "outbound", "trip_num", "trip_count"]
TOUR_COLUMNS = ["start", "end", "primary_purpose", "tour_num", "tour_count", "parent_tour_id"]


def trip_scheduling(configs_dirs: Sequence[str | Path], settings_file: str = "trip_scheduling.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_settings(configs_dirs, settings_file)
    mode = settings.get("scheduling_mode", "departure")
    if mode != "departure":
        raise NotImplementedError(f"{settings_file}: scheduling_mode '{mode}' is not supported yet")
    if settings.get("preprocessor"):
        raise NotImplementedError(f"{settings_file}: a trip scheduling preprocessor is not supported yet")
    failfix = settings.get("FAILFIX", "choose_most_initial")
    if failfix != "choose_most_initial":
        raise NotImplementedError(f"{settings_file}: FAILFIX '{failfix}' is not supported yet")
    for key in ("annotate_trips", "annotate_tours", "annotate_persons", "annotate_households"):
        if settings.get(key):
            raise NotImplementedError(f"{settings_file}: {key} is not supported yet")
    logic_version = settings.get("logic_version")
    if logic_version is None:
        logic_version = 1  # 1.x's default (it warns that it may change)
    if logic_version not in (1, 2):
        raise ValueError(f"{settings_file}: bad logic_version {logic_version}")
    join_cols = list(settings.get("probs_join_cols") or JOIN_COLUMNS)
    base = int(settings.get("DEPART_ALT_BASE", 5))
    max_iterations = int(settings.get("MAX_ITERATIONS", 1))
    if max_iterations < 1:
        raise ValueError(f"{settings_file}: MAX_ITERATIONS must be at least 1")
    probs_spec = pd.read_csv(find_config_file(configs_dirs, settings.get("PROBS_SPEC", "trip_scheduling_probs.csv")),
                             comment="#")
    missing = [c for c in join_cols if c not in probs_spec.columns]
    if missing:
        raise ValueError(f"{settings_file}: the probabilities table lacks {missing}")
    info: dict = {}

    def resolve_needs(available):
        for table, cols in (("trips", TRIP_COLUMNS), ("tours", TOUR_COLUMNS)):
            missing = [c for c in cols if c not in available.get(table, set())]
            if missing:
                raise MissingInputError(f"{table} lack {missing}")
        extra = [c for c in join_cols if c not in JOIN_COLUMNS and c not in available["trips"]]
        if extra:
            raise MissingInputError(f"trips lack the probability join columns {extra}")
        return {"trips": sorted(set(TRIP_COLUMNS) | {c for c in join_cols if c not in JOIN_COLUMNS}),
                "tours": sorted(TOUR_COLUMNS)}

    def run(ctx: StepContext):
        trips = ctx.table_frame("trips")
        tours = ctx.table_frame("tours")
        stats = info["stats"] = {"iterations": 0, "coerced": 0}
        choices_list = []
        chunk = trips.copy()
        set_tour_hour(chunk, tours)
        i = 0
        while i < max_iterations and not chunk.empty:
            i += 1
            last = i == max_iterations
            choices = run_trip_scheduling(ctx, chunk, tours, probs_spec, join_cols, base, last, logic_version, stats)
            failed = choices.reindex(chunk.index).isnull()
            if not last:
                cohorts = failed_trip_cohorts(chunk, failed)
                chunk = chunk[cohorts].copy()
                choices = choices[~cohorts]
            choices_list.append(choices)
        stats["iterations"] = i
        choices = pd.concat(choices_list).reindex(trips.index)
        if choices.isnull().any():
            raise ValueError(f"{STEP}: {int(choices.isnull().sum())} trips could not be scheduled "
                             f"(1.x's drop_and_cleanup is not supported yet)")
        return {"trips": {"depart": choices.to_numpy(dtype=np.float64)}}

    component = Component(
        STEP, run, provides={"trips": ["depart"]}, resolve_needs=resolve_needs,
        description=f"trip departure periods drawn within each tour ({len(probs_spec)} probability rows, "
                    f"up to {max_iterations} attempts)",
    )
    info.update(trace_table="trips", probs_spec=probs_spec)
    component.info = info
    return component


def set_tour_hour(trips: pd.DataFrame, tours: pd.DataFrame) -> None:
    """1.x's set_tour_hour: earliest, latest and tour_hour for each trip (in place)."""
    trips["earliest"] = tours.start.reindex(trips.tour_id).to_numpy()
    trips["latest"] = tours.end.reindex(trips.tour_id).to_numpy()
    trips["tour_hour"] = np.where(trips.outbound, trips["earliest"], trips["latest"]).astype(np.int8)
    subtours = tours.loc[np.asarray(tours.primary_purpose == "atwork"),
                         ["tour_num", "tour_count", "parent_tour_id", "start", "end"]].copy()
    subtours.parent_tour_id = subtours.parent_tour_id.astype(np.int64)
    subtours = subtours.set_index("parent_tour_id").astype(np.int16)
    has_subtours = trips.tour_id.isin(subtours.index)
    outbound = has_subtours & trips.outbound
    trips.loc[outbound, "latest"] = (subtours[subtours.tour_num == 1]["start"]
                                     .reindex(trips[outbound].tour_id).to_numpy())
    inbound = has_subtours & ~trips.outbound
    trips.loc[inbound, "earliest"] = (subtours[subtours.tour_num == subtours.tour_count]["end"]
                                      .reindex(trips[inbound].tour_id).to_numpy())


def set_stop_num(trips: pd.DataFrame) -> None:
    trips["stop_num"] = trips["trip_num"] - 1
    trips["stop_num"] = trips.stop_num.where(trips["outbound"], trips["trip_num"])


def update_tour_earliest(trips: pd.DataFrame, outbound_choices: pd.Series, logic_version: int) -> None:
    """1.x: inbound trips leave no earlier than the tour's last outbound departure."""
    tmp = trips.copy()
    tmp["outbound_departure"] = outbound_choices.reindex(tmp.index)
    latest_out = tmp.groupby(["person_id", "tour_id"])["outbound_departure"].max()
    latest_out.name = "max_outbound_departure"
    tmp = tmp.merge(latest_out, left_on=["person_id", "tour_id"], right_index=True)
    if logic_version == 1:
        # 1.x's default: also when the tour's outbound trips aren't being scheduled (NaN)
        tmp["earliest"] = tmp["earliest"].where(tmp["outbound"], tmp["max_outbound_departure"])
    else:
        tmp["earliest"] = np.where(~tmp["outbound"] & ~tmp["max_outbound_departure"].isna(),
                                   tmp["max_outbound_departure"], tmp["earliest"])
    trips["earliest"] = tmp["earliest"].reindex(trips.index)


def failed_trip_cohorts(trips: pd.DataFrame, failed: pd.Series) -> pd.Series:
    """1.x: the trips of each direction of a tour with a failed trip."""
    bad_out = trips.outbound & trips.tour_id.isin(trips.tour_id[failed & trips.outbound])
    bad_in = ~trips.outbound & trips.tour_id.isin(trips.tour_id[failed & ~trips.outbound])
    return bad_out | bad_in


def run_trip_scheduling(ctx, trips, tours, probs_spec, join_cols, base, last, logic_version, stats) -> pd.Series:
    set_tour_hour(trips, tours)
    set_stop_num(trips)
    results = []
    if trips.outbound.any():
        out = schedule_trips_in_leg(ctx, True, trips[trips.outbound].copy(), probs_spec, join_cols, base, last, stats)
        results.append(out)
        update_tour_earliest(trips, out, logic_version)
    if (~trips.outbound).any():
        results.append(schedule_trips_in_leg(ctx, False, trips[~trips.outbound].copy(), probs_spec, join_cols, base,
                                             last, stats))
    return _concat(results)


def schedule_trips_in_leg(ctx, outbound, trips, probs_spec, join_cols, base, last, stats) -> pd.Series:
    """1.x's schedule_trips_in_leg (departure mode): departures for one direction's trips."""
    to_from_origin = (trips.trip_num == 1) if outbound else (trips.trip_num == trips.trip_count)
    do_not_schedule = to_from_origin | (trips.primary_purpose == "atwork")
    choices = trips.tour_hour[do_not_schedule]
    if do_not_schedule.all():
        return choices
    results = [choices]
    trips = trips[~do_not_schedule].sort_index()
    if outbound:
        trips["next_trip_id"] = np.roll(trips.index, -1)
        is_final = trips.trip_num == trips.trip_count
        adjust = "earliest"
    else:
        trips["next_trip_id"] = np.roll(trips.index, 1)
        is_final = trips.trip_num == 1
        adjust = "latest"
    trips.next_trip_id = trips.next_trip_id.where(~is_final, NO_TRIP_ID)
    first = True
    for i in range(trips.trip_num.min(), trips.trip_num.max() + 1):
        nth = trips[trips.trip_num == i] if outbound else trips[trips.trip_num == trips.trip_count - i]
        choices = scheduling_choices(ctx, nth, probs_spec, join_cols, base, first)
        if last:  # choose_most_initial
            choices = choices.reindex(nth.index)
            stats["coerced"] += int(choices.isna().sum())
            choices = choices.fillna(trips[adjust])
        has_next = nth.next_trip_id != NO_TRIP_ID
        if has_next.any():
            next_ids = nth.next_trip_id[has_next]
            trips.loc[next_ids, adjust] = choices.reindex(next_ids.index).fillna(trips[adjust]).values
        results.append(choices)
        first = False
    return _concat(results)


def _concat(parts):
    """pd.concat, leaving out empty parts (as pandas does today when working out the type)."""
    kept = [p for p in parts if len(p)] or parts[:1]
    return pd.concat(kept)


def scheduling_choices(ctx, nth, probs_spec, join_cols, base, first_in_leg) -> pd.Series:
    """1.x's make_scheduling_choices (departure mode): departures, without the trips that failed."""
    choosers = pd.merge(nth.reset_index(), probs_spec, on=join_cols, how="left").set_index(nth.index.name)
    probs_cols = [c for c in probs_spec.columns if c not in join_cols]
    probs = choosers[probs_cols]
    if probs.isna().any(axis=None):
        raise ValueError(f"{STEP}: trips with no row in the probabilities table, "
                         f"e.g. {probs.index[probs.isna().any(axis=1)][:5].tolist()}")
    # 1.x's _clip_probs: scaled to 1, then nothing outside the trip's window
    probs = probs.div(probs.sum(axis=1), axis=0)
    n_rows, n_cols = probs.shape
    periods = np.tile(np.arange(0, n_cols), n_rows).reshape(n_rows, n_cols) + base
    mask = ((periods >= nth.earliest.values.reshape(n_rows, 1))
            & (periods <= nth.latest.values.reshape(n_rows, 1))) * 1
    probs = probs * mask
    if first_in_leg:
        probs = probs.div(probs.sum(axis=1), axis=0).fillna(0)
    probs["fail"] = 1 - probs.sum(axis=1).clip(0, 1)
    p = probs.to_numpy(dtype=np.float64)
    bad = np.abs(p.sum(axis=1) - 1) > 0.001  # 1.x's make_choices check
    if bad.any():
        raise ValueError(f"{STEP}: probabilities that don't add up to 1, e.g. trips {probs.index[bad][:5].tolist()}")
    rands = ctx.random.uniform(CHANNEL, probs.index.to_numpy())
    raw = ctx.engine.choose(p, rands)
    failed = raw == probs.columns.get_loc("fail")
    choices = pd.Series(raw + base, index=probs.index).where(~failed, -1)
    choices = choices[~failed]
    if not ((choices >= nth.earliest[~failed]).all() and (choices <= nth.latest[~failed]).all()):
        raise ValueError(f"{STEP}: departures outside the trips' windows")
    return choices
