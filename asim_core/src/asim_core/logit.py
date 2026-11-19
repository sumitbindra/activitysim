"""
Multinomial logit: utilities -> probabilities -> choices.

The arithmetic follows ActivitySim 1.x step for step (activitysim/core/
simulate.py ``eval_utilities`` and logit.py ``utils_to_probs`` /
``make_choices``), because floating-point results depend on the order of
operations and exact matching needs the same order.
"""

from __future__ import annotations

import numpy as np

EXP_UTIL_MIN = 1e-300
PROB_MIN = 0.0
PROB_MAX = 1.0
BAD_PROB_THRESHOLD = 0.001
BAD_CHOICE = -99


def term_values(values: list, n: int) -> np.ndarray:
    """Stack per-term values (arrays or scalars) into a (terms x choosers) float64 array."""
    out = np.empty((len(values), n), dtype=np.float64)
    for i, v in enumerate(values):
        out[i] = v
    return out


def utilities(values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """(terms x choosers) values and (terms x alternatives) coefficients -> (choosers x alternatives)."""
    return np.dot(values.transpose(), coefficients)


def probabilities(utils: np.ndarray, shift: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Probabilities from utilities, and a mask of rows that cannot be used
    (all probabilities zero, or infinite / NaN exponentiated utilities).

    ``shift`` subtracts each row's largest utility first (overflow protection).
    ActivitySim 1.x turns this off when ``skip_failed_choices`` is on, which
    is its default.
    """
    u = np.array(utils, dtype=np.float64, copy=True)
    if shift:
        u -= u.max(1, keepdims=True)
    e = np.exp(u)
    np.putmask(e, e <= EXP_UTIL_MIN, 0)
    total = e.sum(axis=1)
    bad = (total == 0.0) | np.isinf(total) | np.isnan(total)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(e, total.reshape(len(e), 1), out=e)
    np.putmask(e, np.isnan(e), PROB_MIN)
    np.clip(e, PROB_MIN, PROB_MAX, out=e)
    return e, bad


def choose(probs: np.ndarray, rands: np.ndarray) -> np.ndarray:
    """
    Pick an alternative (column position) per row: subtract each alternative's
    probability from the random draw in turn and take the first one that takes
    it to zero or below. Rows whose probabilities do not sum to 1 (within
    0.001) get -99, as in ActivitySim 1.x.
    """
    n, n_alts = probs.shape
    z = np.asarray(rands, dtype=np.float64).reshape(n).copy()
    out = np.full(n, -1, dtype=np.int32)
    undecided = np.ones(n, dtype=bool)
    for col in range(n_alts):
        z = z - probs[:, col]
        hit = undecided & (z <= 0)
        out[hit] = col
        undecided &= ~hit
    if undecided.any():  # rare: rounding left the draw just above the total
        out[undecided] = np.argmax(probs[undecided], axis=1)
    bad = np.abs(probs.sum(axis=1) - 1.0) > BAD_PROB_THRESHOLD
    out[bad] = BAD_CHOICE
    return out
