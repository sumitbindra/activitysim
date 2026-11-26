"""
Alternative implementations of seams, for experiments.

Each one has the same signature as its seam's reference implementation and
declares how its results relate to the reference. Use one with, e.g.,
``Engine({"utilities": "accumulate"})``, or name it in
``harness/experiments.yaml``.

The ``counter`` random-number alternative lives in ``random.py``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .seams import register


@register("utilities", "accumulate", agreement="approximate",
          description="add each term's contribution into a running total of utilities; "
                      "never holds all term values at once")
def utilities_accumulate(terms: Iterable, coefficients: np.ndarray, n: int) -> np.ndarray:
    """
    Memory: the utilities themselves plus one term at a time, instead of a
    (terms x choosers) table. Utilities are kept as one contiguous row per
    alternative so each addition is a simple in-place vector operation. The
    sums are added in a different order from a matrix product, so results can
    differ from 1.x in the last decimal places.
    """
    by_alt = np.zeros((coefficients.shape[1], n), dtype=np.float64)
    scratch = np.empty(n, dtype=np.float64)
    k = -1
    for k, v in enumerate(terms):
        v = np.asarray(v, dtype=np.float64)
        for j in np.flatnonzero(coefficients[k]):
            if v.ndim == 0:
                by_alt[j] += v * coefficients[k, j]
            else:
                np.multiply(v, coefficients[k, j], out=scratch)
                by_alt[j] += scratch
    if k + 1 != coefficients.shape[0]:
        raise ValueError(f"expected {coefficients.shape[0]} term values, got {k + 1}")
    return by_alt.T
