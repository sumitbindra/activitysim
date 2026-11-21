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


def test_normal_draws_match_activitysim_1x():
    """Normal draws, then uniform draws in the same step, as tour mode choice makes them."""
    asim_random = pytest.importorskip("activitysim.core.random")
    ids = np.array([1359025, 1359026, 1453204, 7, 99])
    domain = pd.DataFrame({"x": 0}, index=pd.Index(ids, name="tour_id"))
    step = "tour_mode_choice_simulate"
    channel = asim_random.SimpleChannel("tours", 0, domain, step)
    normals = channel.normal_for_df(domain, step, mu=0, sigma=1, lognormal=False, size=6)
    after = channel.random_for_df(domain, step, n=1)

    ours = StepRandom(step, base_seed=0)
    np.testing.assert_array_equal(ours.normal("tours", ids, n=6), normals)
    np.testing.assert_array_equal(ours.uniform("tours", ids, n=1), after)


def test_counter_normals_look_normal():
    from asim_core.random import CounterRandom

    z = CounterRandom("s").normal("tours", np.arange(20000), n=5).ravel()
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02


@pytest.mark.parametrize("make", ["asim1", "vectorized"])
def test_both_exact_implementations_match_activitysim_1x(make):
    """1.x's own random module: first draws, then more (the streams continue), then normal and uniform."""
    from asim_core.random import VectorizedRandom

    asim_random = pytest.importorskip("activitysim.core.random")
    cls = {"asim1": StepRandom, "vectorized": VectorizedRandom}[make]
    ids = np.array([2717868, 763899, 2222791, 112064, 370497])
    domain = pd.DataFrame({"x": 0}, index=pd.Index(ids, name="household_id"))
    step = "auto_ownership_simulate"
    channel = asim_random.SimpleChannel("households", 0, domain, step)
    ours = cls(step, base_seed=0)
    np.testing.assert_array_equal(ours.uniform("households", ids, n=1), channel.random_for_df(domain, step, n=1))
    np.testing.assert_array_equal(ours.uniform("households", ids, n=30), channel.random_for_df(domain, step, n=30))
    np.testing.assert_array_equal(ours.normal("households", ids, n=3),
                                  channel.normal_for_df(domain, step, mu=0, sigma=1, lognormal=False, size=3))
    np.testing.assert_array_equal(ours.uniform("households", ids, n=2), channel.random_for_df(domain, step, n=2))


def test_vectorized_draws_are_the_reference_bit_for_bit():
    """The same calls in the same order: first draws, subsets, many draws, past the first 113 (row by row)."""
    from asim_core.random import FIRST_DRAWS, VectorizedRandom

    rng = np.random.default_rng(11)
    ids = rng.choice(10**9, size=20_000, replace=False)
    ref, new = StepRandom("trip_destination", 3), VectorizedRandom("trip_destination", 3)
    calls = [("trips", ids, 1), ("trips", ids[:500], 30), ("persons", ids[::3], 2), ("trips", ids[:300], 70),
             ("trips", ids[:200], FIRST_DRAWS), ("trips", ids[100:400], 5)]
    for channel, rows, n in calls:
        np.testing.assert_array_equal(new.uniform(channel, rows, n), ref.uniform(channel, rows, n))
    np.testing.assert_array_equal(new.normal("trips", ids[:50], 4), ref.normal("trips", ids[:50], 4))
    seeds = rng.integers(0, 2**32, size=2000, dtype=np.int64)  # every seed value, incl. 0 and the largest
    seeds[:2] = [0, 2**32 - 1]
    first = VectorizedRandom("s")._draw(seeds, np.zeros(len(seeds), dtype=np.int64), 3)
    np.testing.assert_array_equal(first, StepRandom("s")._draw(seeds, np.zeros(len(seeds), dtype=np.int64), 3))


def test_vectorized_is_an_exact_seam_choice():
    from asim_core.seams import Engine

    engine = Engine({"random": "vectorized"})
    assert engine.agreement == "exact" and engine.changed == {"random": "vectorized"}
