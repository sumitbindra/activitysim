"""
Checks that every chooser belongs to a segment of the model.

Models split their choosers into segments (income groups, school types,
tour purposes), each with its own coefficients or size terms. 1.x runs each
segment it is given and leaves choosers whose value has no segment without a
choice, with no message (1.x findings, #8 for scheduling). asim_core does the
same, so results stay 1.x's, but reports it as a warning of the step, naming
the values and how many choosers have them.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def segment_problem(choosers: str, column: str, values, segments: Iterable) -> str | None:
    """A message if some ``values`` (one per chooser) aren't among ``segments``' values."""
    values = np.asarray(values)
    segments = list(segments)
    missing = ~pd.Series(values).isin(segments).to_numpy()
    if not missing.any():
        return None
    counts = pd.Series(values[missing]).value_counts(dropna=False)
    shown = ", ".join(f"{v!r}: {n:,}" for v, n in counts.head(5).items())
    return (f"{int(missing.sum()):,} {choosers} have a {column} with no segment in the model "
            f"({shown}{', ...' if len(counts) > 5 else ''}; segments: {segments}): they get no "
            "choice, as in 1.x")
