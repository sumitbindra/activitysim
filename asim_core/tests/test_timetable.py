"""The timetable against ActivitySim 1.x's, on random days and on a saved 1.x run."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asim_core.timetable import I_END, I_MIDDLE, I_START, TimeTable, read_tdd_alternatives

REPO = Path(__file__).resolve().parents[2]
ALTS_FILE = REPO / "activitysim/examples/prototype_mtc/configs/tour_departure_and_duration_alternatives.csv"


def random_days(alts, n_persons=300, seed=0):
    """Timetables with up to three tours booked per person, as 1.x would book them."""
    tt1 = pytest.importorskip("activitysim.core.timetable")
    periods = list(range(int(alts["start"].min()) - 1, int(alts["end"].max()) + 2))
    windows = pd.DataFrame(0, index=pd.Index(np.arange(n_persons) + 100, name="person_id"),
                           columns=[str(p) for p in periods], dtype=np.int8)
    ref = tt1.TimeTable(windows, pd.DataFrame(alts), "person_windows")
    rng = np.random.default_rng(seed)
    A = len(alts["start"])
    for _ in range(3):
        ids = pd.Series(windows.index.to_numpy())
        fits = ref.tour_available(np.repeat(ids, A), np.tile(np.arange(A), len(ids))).reshape(len(ids), A)
        pick = np.array([rng.choice(np.flatnonzero(f)) if f.any() else -1 for f in fits])
        keep = pick >= 0
        ref.assign(ids[keep], pd.Series(pick[keep]))
    return ref, periods


@pytest.mark.skipif(not ALTS_FILE.exists(), reason="example configs not present")
def test_matches_activitysim_1x():
    alts = read_tdd_alternatives(ALTS_FILE)
    ref, periods = random_days(alts)
    df = ref.get_windows_df()
    ours = TimeTable.from_columns(df.index.to_numpy(), {c: df[c].to_numpy() for c in df.columns}, alts)
    np.testing.assert_array_equal(ours.footprints, ref.tdd_footprints)
    A = len(alts["start"])
    ids = df.index.to_numpy()
    fits = ref.tour_available(pd.Series(np.repeat(ids, A)), np.tile(np.arange(A), len(ids)))
    np.testing.assert_array_equal(ours.available(ours.rows(ids)), fits.reshape(len(ids), A))

    rng = np.random.default_rng(1)
    pid = pd.Series(rng.choice(ids, 20000))
    a = rng.integers(0, A, 20000)
    start, end = pd.Series(alts["start"][a]), pd.Series(alts["end"][a])
    for name in ("adjacent_window_before", "adjacent_window_after", "previous_tour_ends",
                 "previous_tour_begins"):
        for periods_ in (start, end):
            r = getattr(ref, name)(pid, periods_).to_numpy()
            o = getattr(ours, name)(pid.to_numpy(), periods_.to_numpy())
            np.testing.assert_array_equal(o, r)
            assert o.dtype == r.dtype, name
    r = ref.remaining_periods_available(pid, start, end).to_numpy()
    np.testing.assert_array_equal(ours.remaining_periods_available(pid.to_numpy(), start.to_numpy(), end.to_numpy()), r)
    r = ref.max_time_block_available(pid).to_numpy()
    o = ours.max_time_block_available(pid.to_numpy())
    np.testing.assert_array_equal(o, r)
    assert o.dtype == r.dtype


def test_booking_and_collisions():
    alts = {"start": np.array([5, 5, 7, 9], dtype=np.int8), "end": np.array([5, 7, 9, 9], dtype=np.int8)}
    alts["duration"] = (alts["end"] - alts["start"]).astype(np.int8)
    tt = TimeTable.from_columns([1], {str(p): np.zeros(1, np.int8) for p in range(4, 11)}, alts)
    tt.assign(np.array([0]), np.array([1]))  # 5 -> 7
    assert tt.windows[0].tolist() == [0, I_START, I_MIDDLE, I_END, 0, 0, 0]
    # 1.x's rules: a tour may share its first or last period with a booked
    # tour's end or start, but not overlap its middle or start where it starts
    # (5->5 shares period 5 with the booked start: allowed; 5->7 starts where
    # it starts: not allowed; 7->9 starts where it ends: allowed)
    assert tt.available(np.array([0])).tolist() == [[True, False, True, True]]
