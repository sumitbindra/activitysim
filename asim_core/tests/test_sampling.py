"""Sampling of alternatives against ActivitySim 1.x's sampler."""

import numpy as np
import pytest

from asim_core.components.tour_destination import _dedupe
from asim_core.logit import sample


def random_probs(rng, n, k, zeros=0.2):
    p = rng.random((n, k))
    p[rng.random((n, k)) < zeros] = 0.0
    p[:, 0] += 1e-9  # no row of all zeros
    return p / p.sum(axis=1, keepdims=True)


def test_matches_activitysim_1x_sampler():
    choosing = pytest.importorskip("activitysim.core.choosing")
    rng = np.random.default_rng(3)
    probs = random_probs(rng, 500, 40)
    rands = rng.random((500, 30))
    rands[0, :3] = [0.0, 0.999999999999, 1.0]  # edges: the last draw is above every running sum
    alts = np.arange(100, 140, dtype=np.int64)
    ref_choices, ref_probs = choosing.sample_choices_maker_preserve_ordering(probs, rands, alts)
    picks, picked = sample(probs, rands, max_cells=5_000)  # small pieces: same answer
    np.testing.assert_array_equal(alts[picks], ref_choices.T)
    np.testing.assert_array_equal(picked, ref_probs.T)
    assert picked.dtype == np.float32


def test_dedupe_counts_and_orders_by_zone():
    picks = np.array([[2, 0, 2, 1], [3, 3, 3, 3]])
    probs = np.array([[0.5, 0.1, 0.5, 0.2], [0.9, 0.9, 0.9, 0.9]], dtype=np.float32)
    zones = np.array([10, 11, 12, 13])
    out = _dedupe(np.array([7, 9]), picks, probs, zones)
    assert out["tour"].tolist() == [7, 7, 7, 9]
    assert out["zone"].tolist() == [10, 11, 12, 13]
    assert out["pick_count"].tolist() == [1, 1, 2, 4]
    assert out["prob"].tolist() == pytest.approx([0.1, 0.2, 0.5, 0.9])


@pytest.mark.parametrize("n_alts", [1, 2, 7, 40, 1475])
def test_binary_search_picks_the_same_alternatives(n_alts):
    """The alternative must agree with the reference exactly, edges included."""
    from asim_core.logit import sample_binary_search

    rng = np.random.default_rng(n_alts)
    probs = random_probs(rng, 300, n_alts, zeros=0.5)
    probs[1, :] = 0.0
    probs[1, -1] = 1.0  # all mass on the last alternative
    probs[2, n_alts // 2:] = 0.0  # zeros at the end: draws near 1 fall past the total
    probs[2] /= probs[2].sum()
    rands = rng.random((300, 30))
    cum = np.cumsum(probs, axis=1)
    m = min(30, n_alts)
    rands[3, :m] = cum[3, :m]  # draws exactly equal to running sums
    rands[4, :3] = [0.0, 0.999999999999, 1.0]
    if n_alts > 2:
        probs[5, 1] = np.nan  # missing probabilities from here on
    ref = sample(probs, rands)
    alt = sample_binary_search(probs, rands)
    np.testing.assert_array_equal(alt[0], ref[0])
    np.testing.assert_array_equal(alt[1], ref[1])
