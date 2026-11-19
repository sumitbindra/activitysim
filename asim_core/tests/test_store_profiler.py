import sys

import numpy as np
import pandas as pd
import pytest

from asim_core.profiler import Profiler
from asim_core.store import Table


def test_positions_and_missing_ids():
    t = Table("t", np.array([30, 10, 20]), "id", {"v": np.array([3, 1, 2])})
    assert t["v"][t.positions([10, 20, 30])].tolist() == [1, 2, 3]
    with pytest.raises(KeyError, match="IDs not found"):
        t.positions([99])


def test_pandas_round_trip_keeps_dtypes_and_categories():
    df = pd.DataFrame(
        {"a": np.array([1, 2], dtype=np.int8), "b": pd.Categorical(["x", "y"])},
        index=pd.Index([5, 6], name="household_id"),
    )
    t = Table.from_pandas("households", df)
    back = t.to_pandas()
    pd.testing.assert_frame_equal(back, df)


def test_column_length_is_checked():
    t = Table("t", np.array([1, 2]), "id")
    with pytest.raises(ValueError, match="2 rows"):
        t.set_column("x", np.zeros(3))


def test_profiler_records_time_and_memory():
    p = Profiler()
    with p.step("allocate"):
        x = np.ones(20_000_000)  # 160 MB
        del x
    (r,) = p.records
    assert r.wall_s > 0
    if sys.platform.startswith("linux"):
        assert r.step_extra_mb > 100
