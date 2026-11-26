"""
Random numbers for a step: the ``random`` seam.

Both exact implementations reproduce ActivitySim 1.x exactly. Every row (household, person, tour, ...) gets its own random stream for each
step. The stream's seed depends only on:

    (base_seed + hash(channel name) + hash(step name) + row ID) mod 2**32

and the stream restarts at the beginning of every step. Nothing has to be
saved from a previous run to reproduce the draws: a component that makes the
same draws, in the same order, for the same rows, under the same step name,
gets exactly the numbers ActivitySim 1.x used (see activitysim/core/random.py).
1.x uses NumPy's legacy ``RandomState`` (Mersenne Twister, a frozen and
stable API) reseeded for every row, in a Python loop over rows: ``asim1``
here, which does the same. The reference, ``vectorized``, works the same
numbers out for all rows at once (the default since the random draws
measurement; about 7 times faster at a million rows).

``counter`` keeps the same per-row, per-step independence (results never
depend on row order, batching or which other rows are present) but draws
different numbers, so it can only match 1.x statistically.
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


@register("random_positions", "kept_index", reference=True,
          description="each channel's IDs in one pandas index, reused while no new IDs appear, and their "
                      "positions in a NumPy array updated in place (the default since the trip scheduling "
                      "measurement)")
class KeptIndexPositions:
    """
    Where each row is in its stream: one index of the IDs drawn for so far
    in each channel, and their positions. pandas keeps a lookup table with
    an index once built, so finding a call's IDs again costs only those
    IDs, however many the channel has seen; new IDs are appended.
    """

    def __init__(self):
        self._channels: dict[str, tuple[pd.Index, np.ndarray]] = {}

    def take(self, channel: str, ids: np.ndarray, n: int) -> np.ndarray:
        """Each ID's position before this call (0 if new); moves each on by ``n``."""
        index, positions = self._channels.get(channel, (pd.Index([], dtype=np.int64), np.zeros(0, np.int64)))
        rows = index.get_indexer(ids)
        new = rows < 0
        current = np.zeros(len(ids), dtype=np.int64)
        current[~new] = positions[rows[~new]]
        if new.any():
            start = len(index)
            index = index.append(pd.Index(ids[new]))
            positions = np.concatenate([positions, np.zeros(int(new.sum()), dtype=np.int64)])
            rows[new] = np.arange(start, len(index))
        positions[rows] = current + n
        self._channels[channel] = (index, positions)
        return current


@register("random_positions", "rebuilt", agreement="exact",
          description="each channel's positions in a pandas Series, reindexed and rebuilt from every ID seen "
                      "so far on every draw (the default until the trip scheduling measurement; the same "
                      "numbers)")
class RebuiltPositions:
    """The way positions were kept before: a Series per channel, rebuilt on each call."""

    def __init__(self):
        self._offsets: dict[str, pd.Series] = {}

    def take(self, channel: str, ids: np.ndarray, n: int) -> np.ndarray:
        offsets = self._offsets.get(channel)
        if offsets is None:
            offsets = pd.Series(dtype=np.int64)
        current = offsets.reindex(ids, fill_value=0).to_numpy()
        updated = pd.Series(current + n, index=ids)
        self._offsets[channel] = updated.combine_first(offsets).astype(np.int64)
        return current


@register("random", "asim1", agreement="exact",
          description="ActivitySim 1.x per-row Mersenne Twister streams, reseeded for every row in a loop, as "
                      "1.x (the default until the random draws measurement; the same numbers)")
class StepRandom:
    """
    The random streams for one step. Created fresh by the runner for each
    step. ``positions`` keeps each row's place in its streams between draws
    (the ``random_positions`` seam; its reference if not given).
    """

    def __init__(self, step: str, base_seed: int = 0, positions=None):
        self.step = step
        self.base_seed = base_seed
        self.positions = positions if positions is not None else KeptIndexPositions()

    def uniform(self, channel: str, ids, n: int = 1) -> np.ndarray:
        """
        ``n`` uniform draws in [0, 1) for each ID, from that ID's stream.
        Repeated calls in the same step continue each row's stream.
        Returns an array of shape (len(ids), n).
        """
        return self._next(channel, ids, n, self._draw)

    def skip(self, channel: str, ids, n: int) -> None:
        """Move each ID's stream on by ``n`` draws without drawing them, as if they had been drawn."""
        ids = np.asarray(ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("random draws need unique IDs")
        self.positions.take(channel, ids, n)

    def normal(self, channel: str, ids, n: int = 1) -> np.ndarray:
        """
        ``n`` standard normal draws for each ID, as 1.x's
        ``normal_for_df(df, broadcast=True, size=n)``. As in 1.x, each
        stream's position then moves on by ``n``, although NumPy's normal
        generator uses more uniform numbers than that; later draws in the
        same step depend on this, so it is reproduced as is.
        Returns an array of shape (len(ids), n).
        """
        return self._next(channel, ids, n, self._draw_normal)

    def _next(self, channel, ids, n, draw) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("random draws need unique IDs")
        current = self.positions.take(channel, ids, n)
        seeds = row_seeds(self.base_seed, channel, self.step, ids)
        return draw(seeds, current, n)

    def _draw(self, seeds: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
        out = np.empty((len(seeds), n), dtype=np.float64)
        prng = np.random.RandomState()
        for i in range(len(seeds)):
            prng.seed(int(seeds[i]))
            if offsets[i]:
                prng.rand(int(offsets[i]))  # fast-forward past draws already used
            out[i] = prng.rand(n)
        return out

    def _draw_normal(self, seeds: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
        out = np.empty((len(seeds), n), dtype=np.float64)
        prng = np.random.RandomState()
        for i in range(len(seeds)):
            prng.seed(int(seeds[i]))
            if offsets[i]:
                prng.rand(int(offsets[i]))
            out[i] = prng.normal(loc=0, scale=1, size=n)
        return out


# MT19937, as NumPy's RandomState (legacy, frozen) runs it
_N, _M = 624, 397
_MULT = np.uint32(1812433253)
_MATRIX_A, _UPPER, _LOWER = np.uint32(0x9908B0DF), np.uint32(0x80000000), np.uint32(0x7FFFFFFF)
# the doubles a freshly seeded generator gives before its state needs more than one pass of the
# twist: output i (i < N - M) depends only on seeding words i, i + 1 and i + M; a double takes two
FIRST_DRAWS = (_N - _M) // 2  # 113


def _temper(y: np.ndarray) -> np.ndarray:
    y = y ^ (y >> np.uint32(11))
    y = y ^ ((y << np.uint32(7)) & np.uint32(0x9D2C5680))
    y = y ^ ((y << np.uint32(15)) & np.uint32(0xEFC60000))
    return y ^ (y >> np.uint32(18))


def _first_doubles(seeds: np.ndarray, k: int) -> np.ndarray:
    """
    ``RandomState(seed).rand(k)`` for every seed at once (k <= FIRST_DRAWS),
    bit for bit: the seeding recurrence (``init_genrand``) as far as word
    397 + 2k, one pass of the twist for the outputs used, tempering, and two
    32-bit outputs per double as ``random_sample``. Returns (len(seeds), k).
    """
    words = 2 * k
    cur = np.asarray(seeds).astype(np.uint32)  # a copy, updated in place
    tmp = np.empty_like(cur)
    low = np.empty((words + 1, len(cur)), dtype=np.uint32)  # seeding words 0 .. 2k
    high = np.empty((words, len(cur)), dtype=np.uint32)  # seeding words 397 .. 396 + 2k
    low[0] = cur
    for pos in range(1, _M + words):
        np.right_shift(cur, 30, out=tmp)
        np.bitwise_xor(cur, tmp, out=cur)
        np.multiply(cur, _MULT, out=cur)  # mod 2**32, as the C code
        np.add(cur, np.uint32(pos), out=cur)
        if pos <= words:
            low[pos] = cur
        elif pos >= _M:
            high[pos - _M] = cur
    y = (low[:-1] & _UPPER) | (low[1:] & _LOWER)
    out = _temper(high ^ (y >> np.uint32(1)) ^ ((y & np.uint32(1)) * _MATRIX_A))
    a = (out[0::2] >> np.uint32(5)).astype(np.float64)
    b = (out[1::2] >> np.uint32(6)).astype(np.float64)
    return ((a * 67108864.0 + b) / 9007199254740992.0).T


@register("random", "vectorized", reference=True,
          description="1.x's per-row streams, the same numbers, worked out for all rows at once (each row's "
                      "first 113 uniform draws in a step; later ones, and normal draws, row by row as asim1)")
class VectorizedRandom(StepRandom):
    """
    The reference's numbers without its loop: each row's stream is seeded
    and drawn from for all rows at once (``_first_doubles``), in batches of
    rows so the working arrays stay small. Rows whose stream is past its
    first 113 draws in the step, and normal draws, go through the
    reference's per-row loop.
    """

    BATCH_ROWS = 1 << 16  # fastest measured: fewer NumPy calls per row, arrays still in cache
    SAVED_WORDS = 1 << 24  # at most 64 MB of seeding words kept per batch (4k + 1 per row)

    def _draw(self, seeds: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
        out = np.empty((len(seeds), n), dtype=np.float64)
        fast = offsets + n <= FIRST_DRAWS
        rows = np.flatnonzero(fast)
        if len(rows):
            k_all = int(offsets[rows].max()) + n
            batch = min(self.BATCH_ROWS, max(4096, self.SAVED_WORDS // (4 * k_all + 1)))
            for start in range(0, len(rows), batch):
                r = rows[start:start + batch]
                off = offsets[r]
                k = int(off.max()) + n
                first = _first_doubles(seeds[r], k)
                out[r] = first if k == n else np.take_along_axis(first, off[:, None] + np.arange(n), axis=1)
        slow = np.flatnonzero(~fast)
        if len(slow):
            out[slow] = super()._draw(seeds[slow], offsets[slow], n)
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

    def _draw_normal(self, seeds: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
        # Box-Muller transform of two independent counter-based uniforms
        u1 = self._draw(seeds, offsets, n)
        u2 = self._draw(seeds ^ np.int64(0x5DEECE66D), offsets, n)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)
