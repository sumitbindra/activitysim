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
