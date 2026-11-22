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

Two seams, both with the same results as 1.x:

* ``trip_attempts``: how an attempt's departures are worked out. ``arrays``
  (the default) finds each trip's row of the probability table once and does
  the rounds on NumPy arrays, adding up each row's probabilities left to
  right as pandas does here, so every number is the same; ``pandas`` is 1.x's
  code as ported (a merge with the probability table in every round).
* ``trip_retries``: which retried trips go through each attempt.
  ``skip_certain_failures`` (the default) holds back, while any outbound trip
  is still being retried, the inbound trips of tours whose outbound trips are
  not: with ``logic_version`` 1 they have no earliest period, so they fail on
  every such attempt whatever is drawn (1.x finding 18). They rejoin when no
  outbound trip is left or on the last attempt, their streams moved on by the
  draws they skipped. ``every_attempt`` runs them through every attempt, as
  1.x. (The one case the two could differ: a held trip's skipped draw exactly
  0.0, which would pick the first period instead of failing; about one chance
  in 10**16 per draw.)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import Component, MissingInputError, StepContext
from ..seams import register
from ..spec.legacy import find_config_file, read_settings

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
        attempts = ctx.engine.trip_attempts(ctx, trips, tours, probs_spec, join_cols, base, logic_version, stats)
        depart = schedule_attempts(ctx, attempts, max_iterations, logic_version, stats)
        if np.isnan(depart).any():
            raise ValueError(f"{STEP}: {int(np.isnan(depart).sum())} trips could not be scheduled "
                             f"(1.x's drop_and_cleanup is not supported yet)")
        return {"trips": {"depart": depart}}

    component = Component(
        STEP, run, provides={"trips": ["depart"]}, resolve_needs=resolve_needs,
        description=f"trip departure periods drawn within each tour ({len(probs_spec)} probability rows, "
                    f"up to {max_iterations} attempts)",
    )
    info.update(trace_table="trips", probs_spec=probs_spec)
    component.info = info
    return component


def schedule_attempts(ctx, attempts, max_iterations: int, logic_version: int, stats: dict) -> np.ndarray:
    """
    1.x's attempts: every trip in the first; then the trips of each direction
    of a tour with a failed trip, up to ``max_iterations`` attempts, the last
    giving a trip that still fails its window's edge. Returns each trip's
    departure (missing if never scheduled). Which trips sit out attempts they
    are certain to fail is the ``trip_retries`` seam; their streams are moved
    on by the draws they skip, so later draws are 1.x's.
    """
    a = attempts
    depart = np.full(len(a.ids), np.nan)
    m = np.arange(len(a.ids))  # this attempt's trips (positions, in table order)
    held = np.zeros(0, dtype=np.int64)
    held_from = np.zeros(0, dtype=np.int64)  # the first attempt each held trip sat out
    i = 0
    while i < max_iterations and (len(m) or len(held)):
        i += 1
        last = i == max_iterations
        has_outbound = bool(a.outbound[m].any())
        if len(held) and (last or not has_outbound):
            # back in: certain to fail no longer, or the last attempt
            drawn = a.drawn[held]
            skipped = i - held_from[drawn]
            for k in np.unique(skipped):
                ctx.random.skip(CHANNEL, a.ids[held[drawn][skipped == k]], int(k))
            m = np.sort(np.concatenate([m, held]))
            held = held_from = np.zeros(0, dtype=np.int64)
        elif has_outbound and i > 1:
            hold = ctx.engine.trip_retries(a.tour_id[m], a.outbound[m], logic_version)
            if hold.any():
                held = np.concatenate([held, m[hold]])
                held_from = np.concatenate([held_from, np.full(int(hold.sum()), i)])
                m = m[~hold]
        got = a.attempt(m, last)
        if last:
            depart[m] = got
            break
        failed = np.isnan(got)
        out = a.outbound[m]
        tours = a.tour_id[m]
        retry = ((np.isin(tours, tours[failed & out]) & out)  # failed_trip_cohorts
                 | (np.isin(tours, tours[failed & ~out]) & ~out))
        depart[m[~retry]] = got[~retry]
        m = m[retry]
    stats["iterations"] = i
    return depart


@register("trip_retries", "skip_certain_failures", reference=True,
          description="while outbound trips are retried, the inbound trips of tours whose outbound trips are not "
                      "(logic_version 1) sit out the attempts they are certain to fail, their streams moved on by "
                      "the draws skipped (the default since the trip scheduling measurement; the same departures)")
def retries_skip_certain_failures(tour_id: np.ndarray, outbound: np.ndarray, logic_version: int) -> np.ndarray:
    """
    Trips certain to fail this attempt whatever is drawn: with logic_version
    1, an inbound trip's earliest period becomes its tour's last outbound
    departure among the trips being scheduled, missing for a tour whose
    outbound trips are not among them, so no period is inside its window.
    (Called only when the attempt has outbound trips: without any, 1.x
    leaves inbound windows as they are.)
    """
    if logic_version != 1:
        return np.zeros(len(tour_id), dtype=bool)
    return ~outbound & ~np.isin(tour_id, tour_id[outbound])


@register("trip_retries", "every_attempt", agreement="exact",
          description="every retried trip through every attempt, as 1.x (the default before; the same departures)")
def retries_every_attempt(tour_id: np.ndarray, outbound: np.ndarray, logic_version: int) -> np.ndarray:
    return np.zeros(len(tour_id), dtype=bool)


def _trip_arrays(trips: pd.DataFrame):
    """The columns the attempts driver uses: IDs, tours, direction, and which trips are drawn for."""
    outbound = trips.outbound.to_numpy(dtype=bool)
    num, count = trips.trip_num.to_numpy(), trips.trip_count.to_numpy()
    not_drawn = np.where(outbound, num == 1, num == count) | (trips.primary_purpose == "atwork").to_numpy()
    return trips.index.to_numpy(), trips.tour_id.to_numpy(), outbound, ~not_drawn


@register("trip_attempts", "pandas", agreement="exact",
          description="1.x's pandas code as ported: a merge with the probability table in every round of every "
                      "attempt (the default before the trip scheduling measurement; the same departures)")
class PandasAttempts:
    """Each attempt run with 1.x's functions (run_trip_scheduling and those it calls)."""

    def __init__(self, ctx, trips, tours, probs_spec, join_cols, base, logic_version, stats):
        self.ctx, self.tours, self.probs_spec, self.join_cols = ctx, tours, probs_spec, join_cols
        self.base, self.logic_version, self.stats = base, logic_version, stats
        self.trips = trips.copy()
        set_tour_hour(self.trips, tours)
        self.ids, self.tour_id, self.outbound, self.drawn = _trip_arrays(self.trips)

    def attempt(self, m: np.ndarray, last: bool) -> np.ndarray:
        chunk = self.trips.iloc[m].copy()
        choices = run_trip_scheduling(self.ctx, chunk, self.tours, self.probs_spec, self.join_cols, self.base, last,
                                      self.logic_version, self.stats)
        return choices.reindex(chunk.index).to_numpy(dtype=np.float64)


@register("trip_attempts", "arrays", reference=True,
          description="each trip's row of the probability table found once, rounds on NumPy arrays, row sums added "
                      "left to right as pandas does (the default since the trip scheduling measurement; the same "
                      "departures)")
class ArrayAttempts:
    """The same attempts as PandasAttempts, on arrays: the same rounds, windows, draws and arithmetic."""

    def __init__(self, ctx, trips, tours, probs_spec, join_cols, base, logic_version, stats):
        self.ctx, self.base, self.logic_version, self.stats = ctx, base, logic_version, stats
        t = trips.copy()
        set_tour_hour(t, tours)
        set_stop_num(t)
        self.ids, self.tour_id, self.outbound, self.drawn = _trip_arrays(t)
        self.trip_num, self.trip_count = t.trip_num.to_numpy(), t.trip_count.to_numpy()
        self.atwork = (t.primary_purpose == "atwork").to_numpy()
        self.earliest = t.earliest.to_numpy(dtype=np.float64)
        self.latest = t.latest.to_numpy(dtype=np.float64)
        self.tour_hour = t.tour_hour.to_numpy(dtype=np.float64)
        table = probs_spec[join_cols].copy()
        table["_row"] = np.arange(len(probs_spec))
        if table.duplicated(join_cols).any():
            raise ValueError(f"{STEP}: the probabilities table repeats a combination of {join_cols}")
        found = t[join_cols].reset_index(drop=True).merge(table, on=join_cols, how="left")["_row"]
        self.prob_row = found.fillna(-1).to_numpy().astype(np.int64)
        probs = probs_spec[[c for c in probs_spec.columns if c not in join_cols]].to_numpy(dtype=np.float64)
        self.probs = probs / _row_sums(probs)[:, None]  # 1.x's _clip_probs scaling, row by row, so per table row
        self.periods = np.arange(probs.shape[1]) + base
        self.fail = probs.shape[1]

    def attempt(self, m: np.ndarray, last: bool) -> np.ndarray:
        """run_trip_scheduling on arrays: departures for trips m (missing where a trip failed)."""
        out = np.full(len(m), np.nan)
        earliest, latest = self.earliest[m], self.latest[m]  # set_tour_hour, afresh each attempt
        is_out = self.outbound[m]
        if is_out.any():
            leg = np.flatnonzero(is_out)
            out[leg] = self._leg(True, m[leg], earliest[leg].copy(), latest[leg].copy(), last)
            # update_tour_earliest: inbound trips leave no earlier than their tour's last outbound departure
            tours = self.tour_id[m]
            order = np.argsort(tours, kind="stable")
            keys, starts = np.unique(tours[order], return_index=True)
            latest_out = np.fmax.reduceat(np.where(is_out, out, np.nan)[order], starts)
            max_out = latest_out[np.searchsorted(keys, tours)]
            if self.logic_version == 1:  # also when the tour's outbound trips aren't being scheduled (missing)
                earliest = np.where(is_out, earliest, max_out)
            else:
                earliest = np.where(~is_out & ~np.isnan(max_out), max_out, earliest)
        if (~is_out).any():
            leg = np.flatnonzero(~is_out)
            out[leg] = self._leg(False, m[leg], earliest[leg].copy(), latest[leg].copy(), last)
        return out

    def _leg(self, outbound, m, earliest, latest, last) -> np.ndarray:
        """schedule_trips_in_leg on arrays: departures for one direction's trips m (missing where failed)."""
        out = np.full(len(m), np.nan)
        num, count = self.trip_num[m], self.trip_count[m]
        do_not_schedule = ((num == 1) if outbound else (num == count)) | self.atwork[m]
        out[do_not_schedule] = self.tour_hour[m[do_not_schedule]]
        rest = np.flatnonzero(~do_not_schedule)
        if not len(rest):
            return out
        rest = rest[np.argsort(self.ids[m[rest]], kind="stable")]  # in trip ID order
        nxt = np.roll(rest, -1 if outbound else 1)  # the next trip scheduled in the direction
        nxt = np.where((num[rest] == count[rest]) if outbound else (num[rest] == 1), -1, nxt)
        adjust = earliest if outbound else latest  # a trip's choice bounds the next one's window
        first = True
        for i in range(int(num[rest].min()), int(num[rest].max()) + 1):
            sel = (num[rest] == i) if outbound else (num[rest] == count[rest] - i)
            nth, nth_next = rest[sel], nxt[sel]
            ids = self.ids[m[nth]]
            rows = self.prob_row[m[nth]]
            if (rows < 0).any():
                raise ValueError(f"{STEP}: trips with no row in the probabilities table, e.g. {ids[rows < 0][:5].tolist()}")
            # 1.x's _clip_probs: nothing outside the trip's window; only the first round scaled back up to 1
            p = self.probs[rows] * ((self.periods >= earliest[nth, None]) & (self.periods <= latest[nth, None]))
            if first:
                with np.errstate(invalid="ignore", divide="ignore"):
                    p = p / _row_sums(p)[:, None]
                p[np.isnan(p)] = 0.0
            p = np.column_stack([p, 1 - np.clip(_row_sums(p), 0, 1)])  # the chance of failing
            bad = np.abs(_row_sums(p) - 1) > 0.001  # 1.x's make_choices check
            if bad.any():
                raise ValueError(f"{STEP}: probabilities that don't add up to 1, e.g. trips {ids[bad][:5].tolist()}")
            raw = self.ctx.engine.choose(p, self.ctx.random.uniform(CHANNEL, ids))
            failed = raw == self.fail
            choice = np.where(failed, np.nan, (raw + self.base).astype(np.float64))
            ok = ~failed
            if not ((choice[ok] >= earliest[nth[ok]]).all() and (choice[ok] <= latest[nth[ok]]).all()):
                raise ValueError(f"{STEP}: departures outside the trips' windows")
            if last:  # choose_most_initial
                self.stats["coerced"] += int(failed.sum())
                choice = np.where(failed, adjust[nth], choice)
            has_next = nth_next >= 0
            if has_next.any():
                adjust[nth_next[has_next]] = np.where(np.isnan(choice[has_next]), adjust[nth[has_next]],
                                                      choice[has_next])
            out[nth] = choice
            first = False
        return out


def _row_sums(p: np.ndarray) -> np.ndarray:
    """Row sums as pandas makes them for these tables: columns added left to right, missing values as 0
    (NumPy's own sum adds in another order, and its last bits can differ)."""
    p = np.where(np.isnan(p), 0.0, p)
    total = p[:, 0].copy()
    for j in range(1, p.shape[1]):
        total += p[:, j]
    return total


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
