"""
Trip scheduling's two seams on made-up trips: every combination gives the same departures.

The probability table puts most of each row's chance on a few periods, so many trips fail and are
retried: inbound trips are held back while outbound trips are still retried (trip_retries), and rejoin
either when no outbound trip is left (their streams moved on by the draws they skipped) or on the last
attempt. Both paths are checked to happen.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from asim_core.components import trip_scheduling as _  # noqa: F401  (registers the seams)
from asim_core.random import StepRandom
from asim_core.seams import Engine
import importlib

ts = importlib.import_module("asim_core.components.trip_scheduling")
BASE = 5
PERIODS = [f"HR{h}" for h in range(5, 24)]


def made_up(seed: int, n_tours: int = 400):
    rng = np.random.default_rng(seed)
    start = rng.integers(5, 15, n_tours)
    end = np.minimum(23, start + rng.integers(1, 9, n_tours))
    purpose = rng.choice(["work", "shopping"], n_tours)
    tours = pd.DataFrame({"start": start, "end": end, "primary_purpose": purpose, "tour_num": 1, "tour_count": 1,
                          "parent_tour_id": np.nan}, index=pd.Index(np.arange(1, n_tours + 1) * 10, name="tour_id"))
    rows = []
    for tid, t in tours.iterrows():
        for outbound in (True, False):
            count = int(rng.integers(1, 5))
            for num in range(1, count + 1):
                rows.append({"trip_id": tid * 100 + (0 if outbound else 50) + num, "tour_id": tid,
                             "person_id": tid // 10, "primary_purpose": t.primary_purpose, "outbound": outbound,
                             "trip_num": num, "trip_count": count})
    trips = pd.DataFrame(rows).set_index("trip_id").sort_index()
    spec = []
    for purpose_, outbound, hour, num in itertools.product(["work", "shopping"], [True, False], range(5, 24),
                                                           range(1, 5)):
        if outbound:  # the tour's first two hours: outbound retries end after a few attempts
            p = np.zeros(len(PERIODS))
            p[hour - 5:hour - 3] = 1.0
            p /= p.sum()
        else:  # most of the chance on a few periods: many inbound trips fail
            p = rng.dirichlet(np.full(len(PERIODS), 0.15))
        spec.append({"primary_purpose": purpose_, "outbound": outbound, "tour_hour": hour, "trip_num": num,
                     **dict(zip(PERIODS, p))})
    return trips, tours, pd.DataFrame(spec)


def schedule(trips, tours, spec, seams, max_iterations, logic_version=1):
    engine = Engine(seams)
    rand = StepRandom("trip_scheduling", 7, engine.random_positions())
    events = []  # ("skip", draws) when held trips rejoin, ("attempt", last) for each attempt
    real_skip = rand.skip
    rand.skip = lambda channel, ids, n: (events.append(("skip", n)), real_skip(channel, ids, n))
    ctx = SimpleNamespace(engine=engine, random=rand)
    stats = {"iterations": 0, "coerced": 0}
    attempts = engine.trip_attempts(ctx, trips, tours, spec, ts.JOIN_COLUMNS, BASE, logic_version, stats)
    real_attempt = attempts.attempt
    attempts.attempt = lambda m, last: (events.append(("attempt", last)), real_attempt(m, last))[1]
    depart = ts.schedule_attempts(ctx, attempts, max_iterations, logic_version, stats)
    return depart, stats, events


@pytest.mark.parametrize("seed,max_iterations,logic_version", [(1, 40, 1), (2, 3, 1), (3, 40, 2), (4, 100, 1)])
def test_every_combination_gives_the_same_departures(seed, max_iterations, logic_version):
    trips, tours, spec = made_up(seed)
    ref, ref_stats, _ = schedule(trips, tours, spec, {"trip_attempts": "pandas", "trip_retries": "every_attempt"},
                                 max_iterations, logic_version)
    assert not np.isnan(ref).any()
    for attempts, retries in itertools.product(["pandas", "arrays"], ["every_attempt", "skip_certain_failures"]):
        got, stats, events = schedule(trips, tours, spec, {"trip_attempts": attempts, "trip_retries": retries},
                                      max_iterations, logic_version)
        np.testing.assert_array_equal(got, ref, err_msg=f"{attempts}, {retries}")
        assert stats == ref_stats
        if retries == "every_attempt" or logic_version != 1:
            assert not [e for e in events if e[0] == "skip"]


@pytest.mark.parametrize("seed,max_iterations,on_last", [(1, 40, False), (2, 3, True)])
def test_held_trips_rejoin(seed, max_iterations, on_last):
    """Held trips rejoin when no outbound trip is left to retry (seed 1) or on the last attempt (seed 2)."""
    trips, tours, spec = made_up(seed)
    _, _, events = schedule(trips, tours, spec, {}, max_iterations)
    rejoined = {next(x for x in events[k + 1:] if x[0] == "attempt") for k, e in enumerate(events) if e[0] == "skip"}
    assert rejoined, "no held trips rejoined"
    assert ("attempt", on_last) in rejoined


def test_row_sums_add_left_to_right():
    """_row_sums adds columns in order, as pandas does for the tables the pandas attempts build (a merge of the
    trips with the probability table); NumPy's own sum adds in another order and can differ in the last bit."""
    trips, tours, spec = made_up(5)
    choosers = trips.reset_index().assign(tour_hour=8).merge(spec, on=ts.JOIN_COLUMNS, how="left")
    probs = choosers[PERIODS]
    np.testing.assert_array_equal(ts._row_sums(probs.to_numpy()), probs.sum(axis=1).to_numpy())
    rows_in_order = np.ascontiguousarray(probs.to_numpy())  # as the arrays attempts hold them
    assert not np.array_equal(rows_in_order.sum(axis=1), probs.sum(axis=1).to_numpy())  # why it matters
