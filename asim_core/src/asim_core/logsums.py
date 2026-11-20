"""
Which rows of a scheduling round get their own mode choice logsum: the
``logsum_keys`` piece.

Tour scheduling needs a mode choice logsum for every (tour, departure-return
option) row. Many rows share a logsum: it depends on the option only through
the option columns the mode choice formulas read. An implementation returns
the option columns to group rows by (within each tour); one logsum is
computed per group and copied to its rows.

``read`` is the set of option columns the mode choice model reads
(``out_period``, ``in_period``, ``duration``, ``start``, ``end``);
``has_segments`` is True when tour_departure_and_duration_segments.csv gives
representative start and end times per period (they then depend on the
periods only).
"""

from __future__ import annotations

from .seams import register


@register("logsum_keys", "asim1", reference=True,
          description="as 1.x's dedupe_alt_tdd: one logsum per tour and (departure period, "
                      "return period), plus duration when there is no segments file")
def logsum_keys(read: set[str], has_segments: bool) -> list[str]:
    if has_segments:
        return ["out_period", "in_period"]
    return ["out_period", "in_period", "duration"]


@register("logsum_keys", "distinct_inputs", agreement="exact",
          description="one logsum per tour and distinct value of the option columns the mode "
                      "choice formulas actually read")
def logsum_keys_distinct(read: set[str], has_segments: bool) -> list[str]:
    if has_segments:
        # representative start, end and duration depend on both periods
        if read & {"start", "end", "duration"}:
            return ["out_period", "in_period"]
        return [c for c in ("out_period", "in_period") if c in read]
    return [c for c in ("out_period", "in_period", "duration") if c in read]
