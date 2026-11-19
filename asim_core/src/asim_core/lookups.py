"""Reference implementations of the ``lookup`` and ``skim_lookup`` seams."""

from __future__ import annotations

import numpy as np

from .seams import register


@register("lookup", "numpy", reference=True,
          description="NumPy indexing: values[positions]")
def lookup(values, positions: np.ndarray):
    return values[positions]


@register("skim_lookup", "numpy", reference=True,
          description="NumPy indexing: matrix[orig, dest]")
def skim_lookup(matrix: np.ndarray, orig: np.ndarray, dest: np.ndarray) -> np.ndarray:
    return matrix[orig, dest]
