"""Implementations of the ``lookup`` and ``skim_lookup`` seams."""

from __future__ import annotations

import numpy as np

from .seams import register


@register("lookup", "numpy", reference=True,
          description="NumPy indexing: values[positions]")
def lookup(values, positions: np.ndarray):
    return values[positions]


@register("skim_lookup", "flat_take", reference=True,
          description="one-dimensional gather: matrix.ravel().take(orig * zones + dest), the flat "
                      "positions worked out once per set of zone pairs and shared by its skims")
def skim_lookup(matrix: np.ndarray, orig: np.ndarray, dest: np.ndarray, pairs=None) -> np.ndarray:
    """
    ``pairs`` (a ``skims.ZonePairs`` for the same positions), when given, holds
    the flat positions, so a step looking up many skims for the same zone pairs
    works them out once. Faster than two-dimensional indexing on random zone
    pairs (30 to 40% on its own; docs/skim_experiments.md), with the same values.
    """
    if pairs is not None:
        flat = pairs.flat
    else:
        flat = np.asarray(orig, dtype=np.intp) * matrix.shape[1] + np.asarray(dest, dtype=np.intp)
    return matrix.reshape(-1).take(flat)


@register("skim_lookup", "index_2d", agreement="exact",
          description="NumPy two-dimensional indexing: matrix[orig, dest] (the default until the "
                      "skim experiment)")
def skim_lookup_2d(matrix: np.ndarray, orig: np.ndarray, dest: np.ndarray, pairs=None) -> np.ndarray:
    return matrix[orig, dest]
