import numpy as np
import pytest

from asim_core import logit
from asim_core.random import CounterRandom, StepRandom
from asim_core.seams import Engine, implementations, reference_name, register


def test_every_seam_has_a_reference():
    for seam in ["lookup", "utilities", "probabilities", "choose", "random"]:
        assert reference_name(seam)


def test_engine_defaults_to_reference_and_reports_changes():
    e = Engine()
    assert e.changed == {}
    assert e.agreement == "exact"
    assert e.utilities is logit.utilities
    e = Engine({"utilities": "accumulate", "random": "counter"})
    assert e.changed == {"utilities": "accumulate", "random": "counter"}
    assert e.agreement == "statistical"  # the loosest of the chosen pieces


def test_unknown_names_are_clear_errors():
    with pytest.raises(KeyError, match="unknown seam"):
        Engine({"utility": "matmul"})
    with pytest.raises(KeyError, match="available"):
        Engine({"utilities": "gpu"})


def test_new_alternatives_can_be_registered():
    @register("choose", "test_always_first", agreement="statistical")
    def always_first(probs, rands):
        return np.zeros(len(probs), dtype=np.int32)

    assert "test_always_first" in [i.name for i in implementations("choose")]
    assert Engine({"choose": "test_always_first"}).choose(np.ones((3, 2)) / 2, None).tolist() == [0, 0, 0]


def test_accumulate_matches_matmul_to_rounding():
    rng = np.random.default_rng(0)
    n, k, a = 1000, 30, 5
    terms = [rng.normal(size=n) for _ in range(k - 1)] + [1]  # a constant term too
    coef = rng.normal(size=(k, a))
    coef[3] = 0.0  # a term that applies to no alternative
    ref = Engine().utilities(iter(terms), coef, n)
    alt = Engine({"utilities": "accumulate"}).utilities(iter(terms), coef, n)
    np.testing.assert_allclose(alt, ref, rtol=1e-12, atol=1e-12)


def test_counter_random_keeps_the_stream_rules():
    ids = np.array([5, 17, 123456, 7])
    a = CounterRandom("s").uniform("households", ids)
    np.testing.assert_array_equal(a, CounterRandom("s").uniform("households", ids))  # repeatable
    b = CounterRandom("s").uniform("households", ids[[2, 0]])
    np.testing.assert_array_equal(a[[2, 0]], b)  # independent of order and other rows
    one = CounterRandom("s")
    first, second = one.uniform("households", ids), one.uniform("households", ids)
    np.testing.assert_array_equal(np.hstack([first, second]),
                                  CounterRandom("s").uniform("households", ids, n=2))  # streams continue
    assert not np.array_equal(a, CounterRandom("t").uniform("households", ids))  # step matters
    assert not np.array_equal(a, StepRandom("s").uniform("households", ids))  # differs from 1.x


def test_counter_random_looks_uniform():
    draws = CounterRandom("s").uniform("persons", np.arange(200_000)).ravel()
    assert draws.min() >= 0.0 and draws.max() < 1.0
    counts, _ = np.histogram(draws, bins=20, range=(0, 1))
    expected = len(draws) / 20
    chi2 = (((counts - expected) ** 2) / expected).sum()
    assert chi2 < 45  # 19 degrees of freedom; 45 is far in the tail
    assert abs(np.corrcoef(draws[:-1], draws[1:])[0, 1]) < 0.01  # neighbouring rows unrelated
