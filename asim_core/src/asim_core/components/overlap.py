"""
1.x's time-window overlap functions (``abm/models/util/overlap.py``), for
formulas that ask how much free time household members share:

* ``hh_time_window_overlap(households, persons)``: per household, the longest
  stretch of hours two members are both free, for pairs of adults ('aa'),
  children ('cc') and an adult and a child ('ac');
* ``person_time_window_overlap(persons)``: the same per person, over the
  pairs the person is in;
* ``person_max_window(persons)``: each person's longest free stretch.

"Free" is any hour not in the middle of a tour (a tour's first and last
hours can be shared with another tour). The windows are ``person_windows``,
whose first and last columns are padding. Kept as 1.x computes them
(pandas merges and group maxima), with the same types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..timetable import I_MIDDLE


def window_columns_sorted(names) -> list[str]:
    return sorted(names, key=int)


def rle(a):
    """1.x's rle: runs of equal values in each row, the first and last columns being padding."""
    a = np.asarray(a)
    changed = np.array(a[..., :-1] != a[..., 1:])
    changed[..., 0] = True
    changed[..., -1] = True
    i = np.where(changed)
    row_id = i[0][1:]
    row_changed, run_length = np.diff(i)
    start_pos = np.cumsum(np.append(0, run_length))[:-1] + 1
    real_rows = np.where(1 - row_changed)[0]
    row_id = row_id[real_rows]
    run_length = run_length[real_rows]
    start_pos = start_pos[real_rows]
    run_val = a[(row_id, start_pos)]
    return row_id, start_pos, run_length, run_val


def _longest_run(available: np.ndarray, index) -> pd.Series:
    row_ids, _, run_length, run_val = rle(available)
    target = np.where(run_val == 1)
    df = pd.DataFrame({"row_ids": row_ids[target], "run_length": run_length[target]})
    longest = df.groupby("row_ids").run_length.max()
    longest = longest.reindex(np.arange(len(index))).fillna(0)
    longest.index = index
    return longest


def p2p_time_window_overlap(windows: np.ndarray, rows1: np.ndarray, rows2: np.ndarray, index) -> pd.Series:
    """Longest stretch both persons of each pair are free (1.x's p2p_time_window_overlap)."""
    available = (windows[rows1] != I_MIDDLE) * 1 * ((windows[rows2] != I_MIDDLE) * 1)
    return _longest_run(available, index)


def person_pairs(persons: pd.DataFrame) -> pd.DataFrame:
    """Every pair of members of a household, by person ID (1.x's person_pairs)."""
    p = persons[["household_id", "adult"]].reset_index()
    p2p = pd.merge(p, p, left_on="household_id", right_on="household_id", how="outer")
    p2p.rename(columns={f"{persons.index.name}_x": "person1", f"{persons.index.name}_y": "person2"},
               inplace=True)
    p2p = p2p[p2p.person1 < p2p.person2]
    p2p.reset_index(drop=True, inplace=True)
    p2p["p2p_type"] = (p2p.adult_x * 1 + p2p.adult_y * 1).map({0: "cc", 1: "ac", 2: "aa"})
    return p2p[["household_id", "person1", "person2", "p2p_type"]]


def _pair_overlaps(persons, windows, window_index):
    p2p = person_pairs(persons)
    rows1 = window_index.get_indexer(p2p.person1.to_numpy())
    rows2 = window_index.get_indexer(p2p.person2.to_numpy())
    if (rows1 < 0).any() or (rows2 < 0).any():
        raise KeyError("persons without time windows")
    p2p["max_overlap"] = p2p_time_window_overlap(windows, rows1, rows2, p2p.person1.index)
    return p2p


def hh_time_window_overlap(households_index, persons: pd.DataFrame, windows, window_index) -> pd.DataFrame:
    """1.x's hh_time_window_overlap: columns aa, cc, ac per household, 8-bit."""
    p2p = _pair_overlaps(persons, windows, window_index)
    hh_overlap = (p2p.groupby(["household_id", "p2p_type"]).max_overlap.max()
                  .unstack(level=-1, fill_value=0))
    hh_overlap = hh_overlap.reindex(households_index).fillna(0).astype(np.int8)
    for c in ["aa", "cc", "ac"]:
        if c not in hh_overlap.columns:
            hh_overlap[c] = 0
    return hh_overlap


def person_time_window_overlap(persons: pd.DataFrame, windows, window_index) -> pd.DataFrame:
    """1.x's person_time_window_overlap: columns aa, cc, ac per person, 8-bit."""
    p2p = _pair_overlaps(persons, windows, window_index)
    p_overlap = (
        pd.concat([
            p2p[["person1", "p2p_type", "max_overlap"]].rename(columns={"person1": "person_id"}),
            p2p[["person2", "p2p_type", "max_overlap"]].rename(columns={"person2": "person_id"}),
        ])
        .groupby(["person_id", "p2p_type"]).max_overlap.max()
    )
    p_overlap = p_overlap.unstack(level=-1, fill_value=0)
    for c in ["aa", "cc", "ac"]:
        if c not in p_overlap.columns:
            p_overlap[c] = 0
    return p_overlap.reindex(persons.index).fillna(0).astype(np.int8)


def person_max_window(person_ids, windows, window_index) -> pd.Series:
    """1.x's person_max_window: each person's longest free stretch."""
    rows = window_index.get_indexer(np.asarray(person_ids))
    if (rows < 0).any():
        raise KeyError("persons without time windows")
    available = (windows[rows] != I_MIDDLE) * 1
    return _longest_run(available, pd.Index(person_ids))
