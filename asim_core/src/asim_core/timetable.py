"""
Timetables: each person's day as a row of time periods, as in ActivitySim
1.x (activitysim/core/timetable.py, table ``person_windows``).

Each period holds a state: empty, the start of a tour, its middle, its end,
or start and end (a tour that starts and ends in the same period). States
are bit patterns, so booking a tour is a bitwise OR of its footprint into
the person's row.

A scheduling option (a "tdd": departure, duration) fits a person's day if
its footprint collides with nothing already booked, using 1.x's list of
collisions. 1.x checks options one person-option pair at a time (a Numba
loop); here it's done for a batch of tours at once with small matrix
products, which give the same answer.

The functions scheduling formulas call (``tt.previous_tour_ends(...)`` and
so on) take one person ID and period per row, like 1.x's, and are
vectorized.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

I_EMPTY = 0x0
I_START = 0x2
I_END = 0x4
I_START_END = 0x6
I_MIDDLE = 0x7

# (footprint state, window state) pairs that collide, as in 1.x
COLLISIONS = [
    (I_START, I_START), (I_END, I_END), (I_MIDDLE, I_MIDDLE),
    (I_START, I_MIDDLE), (I_MIDDLE, I_START),
    (I_END, I_MIDDLE), (I_MIDDLE, I_END),
    (I_START_END, I_MIDDLE), (I_MIDDLE, I_START_END),
]

# what formulas may call as tt.<name>(...)
FORMULA_FUNCTIONS = {
    "adjacent_window_before", "adjacent_window_after",
    "previous_tour_ends", "previous_tour_begins",
    "remaining_periods_available", "max_time_block_available",
}


def read_tdd_alternatives(path) -> dict[str, np.ndarray]:
    """
    Scheduling options from tour_departure_and_duration_alternatives.csv:
    start, end and duration (end - start), as 8-bit integers like 1.x.
    """
    df = pd.read_csv(path)
    df["duration"] = df.end - df.start
    df = df.astype(np.int8)
    if not (df.index == np.arange(len(df))).all():
        raise ValueError(f"{path}: options must be numbered 0, 1, 2, ...")
    return {c: df[c].to_numpy() for c in ("start", "end", "duration")}


class TimeTable:
    def __init__(self, person_ids, windows: np.ndarray, periods: list[int],
                 alts: Mapping[str, np.ndarray]):
        self.ids = np.asarray(person_ids)
        self.index = pd.Index(self.ids)
        self.windows = np.ascontiguousarray(windows, dtype=np.int8)
        self.periods = [int(p) for p in periods]
        if self.periods != sorted(self.periods):
            raise ValueError("timetable periods must be in increasing order")
        self.period_col = {p: j for j, p in enumerate(self.periods)}
        self._period_lookup = np.full(max(self.periods) + 1, -1, dtype=np.int64)
        self._period_lookup[self.periods] = np.arange(len(self.periods))
        self.alts = dict(alts)
        self.footprints = self._footprints()
        self._rows_cache: tuple | None = None

    # --- building -----------------------------------------------------------

    @classmethod
    def from_columns(cls, person_ids, columns: Mapping[str, np.ndarray], alts) -> TimeTable:
        """From person_windows columns named by period ('4', '5', ...)."""
        names = sorted(columns, key=int)
        windows = np.column_stack([np.asarray(columns[n]) for n in names]) if names else None
        return cls(person_ids, windows, [int(n) for n in names], alts)

    def to_columns(self) -> dict[str, np.ndarray]:
        return {str(p): self.windows[:, j].copy() for j, p in enumerate(self.periods)}

    def _footprints(self) -> np.ndarray:
        """Each option's states over the timetable's periods, as 1.x builds them."""
        start, end, duration = self.alts["start"], self.alts["end"], self.alts["duration"]
        lo, hi = self.periods[0], self.periods[-1]
        if self.periods != list(range(lo, hi + 1)):
            raise ValueError("timetable periods must be consecutive")
        fp = np.zeros((len(start), len(self.periods)), dtype=np.int8)
        for a in range(len(start)):
            s, e, d = int(start[a]) - lo, int(end[a]) - lo, int(duration[a])
            if d > 0:
                fp[a, s] = I_START
                fp[a, s + 1:e] = I_MIDDLE
                fp[a, e] = I_END
            else:
                fp[a, s] = I_START_END
        return fp

    # --- rows -----------------------------------------------------------------

    def rows(self, person_ids) -> np.ndarray:
        """Row positions for person IDs (the last lookup is remembered: formulas repeat it)."""
        person_ids = np.asarray(person_ids)
        cached = self._rows_cache
        if cached is not None and cached[0] is person_ids:
            return cached[1]
        pos = self.index.get_indexer(person_ids)
        if (pos < 0).any():
            raise KeyError(f"persons not in the timetable: {person_ids[pos < 0][:5].tolist()}")
        self._rows_cache = (person_ids, pos)
        return pos

    def cols(self, periods) -> np.ndarray:
        periods = np.asarray(periods).astype(np.int64)
        if (periods < 0).any() or (periods >= len(self._period_lookup)).any():
            raise KeyError("periods outside the timetable")
        cols = self._period_lookup[periods]
        if (cols < 0).any():
            raise KeyError("periods outside the timetable")
        return cols

    # --- booking --------------------------------------------------------------

    def available(self, rows: np.ndarray) -> np.ndarray:
        """
        (len(rows), number of options) True where the option fits the
        person's day. Same rule as 1.x's tour_available.
        """
        w = self.windows[rows]
        fp = self.footprints
        clash = np.zeros((len(rows), fp.shape[0]), dtype=np.float32)
        for state in np.unique(w):
            hits = [f for f, ws in COLLISIONS if ws == state]
            if not hits:
                continue
            in_state = (w == state).astype(np.float32)               # (rows, periods)
            collides = np.isin(fp, hits).astype(np.float32)          # (options, periods)
            clash += in_state @ collides.T
        return clash == 0

    def assign(self, rows: np.ndarray, tdds: np.ndarray) -> None:
        """Book each person's chosen option (rows must be distinct)."""
        if len(np.unique(rows)) != len(rows):
            raise ValueError("assign() needs one tour per person")
        self.windows[rows] = np.bitwise_or(self.windows[rows], self.footprints[np.asarray(tdds)])

    # --- functions formulas call (one person ID and period per row) ----------

    def _run_length(self, person_ids, periods, before: bool) -> np.ndarray:
        rows = self.rows(person_ids)
        cols = self.cols(periods)
        uniq, inv = np.unique(rows, return_inverse=True)
        w = self.windows[uniq]
        n_cols = w.shape[1]
        unavailable = w == I_MIDDLE
        unavailable[:, 0] = True
        unavailable[:, -1] = True
        idx = np.arange(n_cols)
        if before:
            # last unavailable period strictly before each period (0 if none)
            last = np.maximum.accumulate(np.where(unavailable, idx, 0), axis=1)
            strictly = np.zeros_like(last)
            strictly[:, 1:] = last[:, :-1]
            first_unavailable = strictly[inv, cols]
            return (cols - first_unavailable - 1).astype(np.int32)
        nxt = np.minimum.accumulate(np.where(unavailable, idx, n_cols)[:, ::-1], axis=1)[:, ::-1]
        strictly = np.full_like(nxt, n_cols)
        strictly[:, :-1] = nxt[:, 1:]
        first_unavailable = strictly[inv, cols]
        return (first_unavailable - cols - 1).astype(np.int32)

    def adjacent_window_before(self, person_ids, periods) -> np.ndarray:
        """Free periods just before each period (1.x: adjacent_window_before)."""
        return self._run_length(person_ids, periods, before=True)

    def adjacent_window_after(self, person_ids, periods) -> np.ndarray:
        """Free periods just after each period (1.x: adjacent_window_after)."""
        return self._run_length(person_ids, periods, before=False)

    def _in_states(self, person_ids, periods, states) -> np.ndarray:
        return np.isin(self.windows[self.rows(person_ids), self.cols(periods)], states)

    def previous_tour_ends(self, person_ids, periods) -> np.ndarray:
        """True where a tour already booked ends in the period."""
        return self._in_states(person_ids, periods, [I_END, I_START_END])

    def previous_tour_begins(self, person_ids, periods) -> np.ndarray:
        """True where a tour already booked begins in the period."""
        return self._in_states(person_ids, periods, [I_START, I_START_END])

    def remaining_periods_available(self, person_ids, starts, ends) -> np.ndarray:
        """
        Periods still free in the person's day, after a tour from start to
        end (1.x: count of periods not in the middle of a tour, minus 2,
        minus the tour's inner periods).
        """
        rows = self.rows(person_ids)
        uniq, inv = np.unique(rows, return_inverse=True)
        free = (self.windows[uniq] != I_MIDDLE).sum(axis=1).astype(np.int64)
        block = np.asarray(ends).astype(np.int64) - np.asarray(starts).astype(np.int64) - 1
        return free[inv] - 2 - np.maximum(block, 0)

    def max_time_block_available(self, person_ids) -> np.ndarray:
        """Longest run of free periods in each person's day (1.x: 8-bit unsigned)."""
        rows = self.rows(person_ids)
        uniq, inv = np.unique(rows, return_inverse=True)
        free = self.windows[uniq][:, 1:-1] != I_MIDDLE
        current = np.zeros(len(uniq), dtype=np.int64)
        best = np.zeros(len(uniq), dtype=np.int64)
        for j in range(free.shape[1]):
            current = (current + 1) * free[:, j]
            np.maximum(best, current, out=best)
        return best.astype(np.uint8)[inv]
