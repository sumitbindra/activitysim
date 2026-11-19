"""
Random numbers for a step: the ``random`` seam.

The reference implementation (``asim1``) reproduces ActivitySim 1.x exactly.
Every row (household, person, tour, ...) gets its own random stream for each
step. The stream's seed depends only on:

    (base_seed + hash(channel name) + hash(step name) + row ID) mod 2**32

and the stream restarts at the beginning of every step. Nothing has to be
saved from a previous run to reproduce the draws: a component that makes the
same draws, in the same order, for the same rows, under the same step name,
gets exactly the numbers ActivitySim 1.x used (see activitysim/core/random.py).
It uses NumPy's legacy ``RandomState`` (Mersenne Twister, a frozen and stable
API) reseeded for every row, which is slow: a Python loop over rows.

Alternatives keep the same per-row, per-step independence (results never
depend on row order, batching or which other rows are present) but draw
different numbers, so they can only match 1.x statistically.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .seams import register

_MAX_SEED = 1 << 32
_SEED_MASK = 0xFFFFFFFF


def hash32(s: str) -> int:
    """32-bit hash of a string (same as ActivitySim 1.x)."""
    return int(hashlib.md5(s.encode("utf8")).hexdigest(), base=16) & _SEED_MASK


def row_seeds(base_seed: int, channel: str, step: str, ids) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    return (base_seed + hash32(channel) + hash32(step) + ids) % _MAX_SEED


@register("random", "asim1", reference=True,
          description="ActivitySim 1.x per-row Mersenne Twister streams, reseeded for every row")
class StepRandom:
    """The random streams for one step. Created fresh by the runner for each step."""

    def __init__(self, step: str, base_seed: int = 0):
        self.step = step
        self.base_seed = base_seed
        self._offsets: dict[str, pd.Series] = {}

    def uniform(self, channel: str, ids, n: int = 1) -> np.ndarray:
        """
        ``n`` uniform draws in [0, 1) for each ID, from that ID's stream.
        Repeated calls in the same step continue each row's stream.
        Returns an array of shape (len(ids), n).
        """
        ids = np.asarray(ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("uniform() needs unique IDs")
        offsets = self._offsets.get(channel)
        if offsets is None:
            offsets = pd.Series(dtype=np.int64)
        current = offsets.reindex(ids, fill_value=0).to_numpy()
        seeds = row_seeds(self.base_seed, channel, self.step, ids)
        out = self._draw(seeds, current, n)
        updated = pd.Series(current + n, index=ids)
        self._offsets[channel] = updated.combine_first(offsets).astype(np.int64)
        return out

    def _draw(self, seeds: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
        out = np.empty((len(seeds), n), dtype=np.float64)
        prng = np.random.RandomState()
        for i in range(len(seeds)):
            prng.seed(int(seeds[i]))
            if offsets[i]:
                prng.rand(int(offsets[i]))  # fast-forward past draws already used
            out[i] = prng.rand(n)
        return out


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer: a well-known 64-bit mixing function (vectorized)."""
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


@register("random", "counter", agreement="statistical",
          description="vectorized counter-based draws: SplitMix64 hash of (row seed, draw number); "
                      "no per-row loop")
class CounterRandom(StepRandom):
    """
    Same seeds and stream bookkeeping as the reference, but each draw is a
    hash of (row seed, draw number), computed for all rows at once. Draws are
    different from 1.x's, so results match 1.x only statistically.

    Experimental: the hash's statistical quality is good for simulation use,
    but would need review before production use.
    """

    def _draw(self, seeds: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
        counters = (seeds.astype(np.uint64) << np.uint64(32))[:, None] + (
            offsets.astype(np.uint64)[:, None] + np.arange(n, dtype=np.uint64)[None, :]
        )
        bits = _splitmix64(counters)
        return (bits >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
