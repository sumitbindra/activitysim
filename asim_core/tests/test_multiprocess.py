"""
The multiprocess runner's pieces, as 1.x does them: the plan from multiprocess_steps, how tables are split
(households by stride, other tables following by index or column), and how they are put back together
(split tables in process order, the others from process 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asim_core.multiprocess import (
    Group,
    coalesce,
    plan_groups,
    reorder_as_split,
    slice_rules,
    split_store,
    take_rows,
)
from asim_core.store import Store, Table


def made_up_store():
    hh = Table("households", np.array([11, 12, 13, 14, 15]), "household_id", {"size": np.array([1, 2, 3, 1, 2])})
    persons = Table("persons", np.array([101, 102, 103, 104, 105, 106, 107, 108, 109]), "person_id",
                    {"household_id": np.array([11, 12, 12, 13, 13, 13, 14, 15, 15])})
    tours = Table("tours", np.array([1001, 1002, 1003, 1004]), "tour_id",
                  {"person_id": np.array([102, 104, 107, 109]), "household_id": np.array([12, 13, 14, 15])})
    land_use = Table("land_use", np.array([1, 2, 3]), "zone_id", {"area": np.array([1.0, 2.0, 3.0])})
    return Store([land_use, hh, persons, tours])


def test_plan_from_1x_multiprocess_steps():
    settings = {"num_processes": 3, "multiprocess_steps": [
        {"name": "mp_initialize", "begin": "a"},
        {"name": "mp_households", "begin": "c", "slice": {"tables": ["households", "persons"]}},
        {"name": "mp_summarize", "begin": "e"}]}
    groups = plan_groups(settings, ["a", "b", "c", "d", "e"])
    assert [(g.name, g.steps, g.sliced, g.num_processes) for g in groups] == [
        ("mp_initialize", ["a", "b"], False, 3), ("mp_households", ["c", "d"], True, 3),
        ("mp_summarize", ["e"], False, 3)]
    with pytest.raises(ValueError, match="not in models"):
        plan_groups({"multiprocess_steps": [{"name": "x", "begin": "zz"}]}, ["a"])


def test_slice_rules_follow_1x():
    store = made_up_store()
    rules = slice_rules(store, ["households", "persons"])
    assert rules["households"] == ("primary",)
    assert rules["persons"] == ("column", "household_id", "households")
    assert rules["tours"] == ("column", "household_id", "households")
    assert rules["land_use"] is None  # whole to every process
    assert slice_rules(store, ["households", "persons"], exclude=["tours"])["tours"] is None
    assert slice_rules(store, ["land_use"], exclude=True) == {
        "land_use": ("primary",), "households": None, "persons": None, "tours": None}


def test_split_by_stride_and_put_back_in_process_order():
    store = made_up_store()
    rules = slice_rules(store, ["households", "persons"])
    parts = split_store(store, rules, 2)
    # households by stride: rows 0, 2, 4 to process 0; 1, 3 to process 1
    assert store["households"].index[parts[0]["households"]].tolist() == [11, 13, 15]
    assert store["persons"].index[parts[1]["persons"]].tolist() == [102, 103, 107]
    assert parts[0]["land_use"] is None
    stores = [Store([take_rows(store[t], parts[k][t]) for t in rules]) for k in range(2)]
    merged = coalesce(stores, slice_rules(stores[0], ["households", "persons"]))
    assert merged["households"].index.tolist() == [11, 13, 15, 12, 14]  # process 0's rows, then process 1's
    assert merged["tours"].index.tolist() == [1002, 1004, 1001, 1003]
    assert merged["land_use"] is stores[0]["land_use"]
    # every row, every value kept
    for t in ("households", "persons", "tours"):
        a = store[t].to_pandas().sort_index()
        b = merged[t].to_pandas().sort_index()
        pd.testing.assert_frame_equal(a, b)


def test_categories_survive_the_split():
    t = Table("households", np.array([1, 2, 3]), "household_id",
              {"kind": pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"])})
    part = take_rows(t, np.array([0, 2]))
    assert isinstance(part["kind"], pd.Categorical) and list(part["kind"].categories) == ["a", "b", "c"]


def test_a_group_split_by_zones_is_reordered_as_1x_leaves_it():
    store = made_up_store()
    g = Group("mp_accessibility", ["compute_accessibility"], ["land_use"], True, 2)
    out = reorder_as_split(store, g)
    assert out["land_use"].index.tolist() == [1, 3, 2]
    assert out["households"] is store["households"]


def test_more_processes_than_households_is_an_error():
    store = made_up_store()
    with pytest.raises(ValueError, match="fewer rows"):
        split_store(store, slice_rules(store, ["households"]), 6)


def test_mtc_in_two_processes_matches_1x_in_two_processes(tmp_path):
    """The MTC example run as configs_mp says (2 processes): every output file 1.x's, byte for byte or cell by cell."""
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    mp_run = repo / "asim_core" / "work" / "asim1_mp2"
    if not (mp_run / "final_trips.csv").exists():
        pytest.skip("needs 1.x's 2-process MTC run (run_asim1.py --configs configs_mp configs --out asim_core/work/asim1_mp2)")
    sys.path.insert(0, str(repo / "asim_core" / "harness"))
    import run_multiprocess

    rc = run_multiprocess.main(["--asim1-out", str(mp_run), "--out", str(tmp_path / "mp"), "--read-skim-cache", "true"])
    summary = (tmp_path / "mp" / "summary.md").read_text(encoding="utf-8")
    assert rc == 0, summary
    assert "39 of 39 exact" in summary
