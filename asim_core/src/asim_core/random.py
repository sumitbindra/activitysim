"""
Random numbers that reproduce ActivitySim 1.x exactly.

Every row (household, person, tour, ...) gets its own random stream for each
step. The stream's seed depends only on:

    (base_seed + hash(channel name) + hash(step name) + row ID) mod 2**32

and the stream restarts at the beginning of every step. Nothing has to be
saved from a previous run to reproduce the draws: a component that makes the
same draws, in the same order, for the same rows, under the same step name,
gets exactly the numbers ActivitySim 1.x used (see activitysim/core/random.py).

Only NumPy's legacy ``RandomState`` (Mersenne Twister, a frozen and stable
API) and the standard library's ``hashlib`` are used.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

_MAX_SEED = 1 << 32
_SEED_MASK = 0xFFFFFFFF


def hash32(s: str) -> int:
    """32-bit hash of a string (same as ActivitySim 1.x)."""
    return int(hashlib.md5(s.encode("utf8")).hexdigest(), base=16) & _SEED_MASK


def row_seeds(base_seed: int, channel: str, step: str, ids) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    return (base_seed + hash32(channel) + hash32(step) + ids) % _MAX_SEED


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

        out = np.empty((len(ids), n), dtype=np.float64)
        prng = np.random.RandomState()
        for i in range(len(ids)):
            prng.seed(int(seeds[i]))
            if current[i]:
                prng.rand(int(current[i]))  # fast-forward past draws already used
            out[i] = prng.rand(n)

        updated = pd.Series(current + n, index=ids)
        self._offsets[channel] = updated.combine_first(offsets).astype(np.int64)
        return out
