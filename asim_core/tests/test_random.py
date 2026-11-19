import numpy as np
import pandas as pd
import pytest

from asim_core.random import StepRandom, hash32, row_seeds


def test_same_inputs_same_draws():
    ids = np.array([5, 17, 123456])
    a = StepRandom("some_step").uniform("households", ids)
    b = StepRandom("some_step").uniform("households", ids)
    np.testing.assert_array_equal(a, b)


def test_streams_continue_within_a_step():
    ids = np.array([5, 17, 123456])
    one = StepRandom("s")
    first, second = one.uniform("households", ids), one.uniform("households", ids)
    both = StepRandom("s").uniform("households", ids, n=2)
    np.testing.assert_array_equal(np.hstack([first, second]), both)


def test_draws_do_not_depend_on_row_order_or_other_rows():
    a = StepRandom("s").uniform("households", np.array([1, 2, 3]))
    b = StepRandom("s").uniform("households", np.array([3, 1]))
    np.testing.assert_array_equal(a[[2, 0]], b)


def test_step_and_channel_change_the_stream():
    ids = np.array([1, 2, 3])
    base = StepRandom("s").uniform("households", ids)
    assert not np.array_equal(base, StepRandom("t").uniform("households", ids))
    assert not np.array_equal(base, StepRandom("s").uniform("persons", ids))


def test_seed_formula():
    assert row_seeds(0, "households", "s", [0])[0] == (hash32("households") + hash32("s")) % (1 << 32)


def test_matches_activitysim_1x():
    """Direct check against ActivitySim 1.x's own random module, when installed."""
    asim_random = pytest.importorskip("activitysim.core.random")
    ids = np.array([2717868, 763899, 2222791, 112064, 370497])
    domain = pd.DataFrame({"x": 0}, index=pd.Index(ids, name="household_id"))
    channel = asim_random.SimpleChannel("households", 0, domain, "auto_ownership_simulate")
    first = channel.random_for_df(domain, "auto_ownership_simulate", n=1)
    second = channel.random_for_df(domain, "auto_ownership_simulate", n=2)

    ours = StepRandom("auto_ownership_simulate", base_seed=0)
    np.testing.assert_array_equal(ours.uniform("households", ids, n=1), first)
    np.testing.assert_array_equal(ours.uniform("households", ids, n=2), second)
