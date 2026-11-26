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


def _size_table():
    from asim_core.components.trip_destination import SizeTerms

    zones = np.array([3, 5, 9, 11])
    return SizeTerms(zones, ["work", "shopping", "escort"],
                     np.arange(12, dtype=np.float64).reshape(4, 3) + 0.5), zones


def test_size_lookup_by_position_gives_the_matched_values():
    table, zones = _size_table()
    purposes = np.array(["shopping", "work", "shopping"], dtype=object)  # three choosers
    zone_rows = np.tile(zones, 3)
    rows_purposes = np.repeat(purposes, 4)
    expected = Engine({"size_lookup": "match"}).size_lookup(table, zone_rows, {}, zones).get(zone_rows, rows_purposes)
    fast = Engine().size_lookup(table, zone_rows, {"purpose": (rows_purposes, purposes)}, zones)
    got = fast.get(zone_rows, rows_purposes)
    assert np.array_equal(got, expected) and fast.fast_calls == 1
    assert fast.get(zone_rows, rows_purposes) is got  # asked twice by the spec: taken once
    # anything else (other zones, purposes not copied out) is matched row by row
    other = np.array(["escort"] * 12, dtype=object)
    assert np.array_equal(fast.get(zone_rows, other), table.get(zone_rows, other)) and fast.fast_calls == 2
    # zones not in the table's order: every row matched
    assert Engine().size_lookup(table, zone_rows, {}, zones[::-1]) is table
    bad = np.array(["school"], dtype=object)
    bad_rows = np.repeat(bad, 4)
    with pytest.raises(ValueError, match="no trip size terms"):
        Engine().size_lookup(table, zones, {"purpose": (bad_rows, bad)}, zones).get(zones, bad_rows)


def test_sample_columns_choices():
    available = ["dest", "purpose", "income", "trip_period", "age"]
    reads = {"purpose", "income"}
    assert Engine().sample_columns(available, reads, "dest", "trip_period", True) == ["purpose", "income",
                                                                                      "trip_period"]
    assert Engine().sample_columns(available, reads, "dest", "trip_period", False) == ["purpose", "income"]
    assert Engine({"sample_columns": "all_columns"}).sample_columns(available, reads, "dest", "trip_period",
                                                                     False) == ["purpose", "income",
                                                                                "trip_period", "age"]
