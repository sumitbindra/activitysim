"""
The ``skim_reuse`` seam: whether a model's formulas share the skim values
they read.

A model's formulas read skims through ``s(accessor, key, how)``, one call
per place a formula names a skim. Formulas often name the same skim more
than once (MTC's tour mode choice: 281 lookups written, 211 distinct; its
non-mandatory destination: 26 written, 6 distinct), so the same values are
gathered from the skim again.

* ``as_written`` (reference): every lookup is made as the formulas are
  written.
* ``once``: each skim the formulas name more than once (counted in their
  generated source) is looked up once per batch and shared; it is kept only
  until its last use. Shared values are read-only, so a formula that tried
  to change one would fail instead of changing another formula's input.

Both give the same values: the same gather, made fewer times.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from .seams import register

# s('odt_skims', 'SOV_TIME') or s('skims', 'DIST', 'reverse') in generated code
_SKIM = re.compile(r"""\bs\((['"])([^'"]+)\1,\s*(['"])([^'"]+)\3(?:,\s*(['"])([^'"]+)\5)?\)""")

_COUNTS: dict[str, Counter] = {}


def skim_uses(source: str) -> Counter:
    """How many times each (accessor, key, how) is looked up in generated code (comments skipped)."""
    counts = _COUNTS.get(source)
    if counts is None:
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        counts = Counter((m.group(2), m.group(4), m.group(6) or "get") for m in _SKIM.finditer(code))
        _COUNTS[source] = counts
    return counts


class SharedSkims:
    """``s`` for a batch, sharing the values of skims looked up more than once."""

    def __init__(self, s, uses: Counter):
        self._s = s
        self._left = {k: n for k, n in uses.items() if n > 1}  # lookups still to come
        self._values: dict = {}

    def __call__(self, accessor, key, how="get"):
        k = (accessor, key, how)
        left = self._left.get(k, 0)
        if left <= 0:  # used once, or not foreseen: as written
            return self._s(accessor, key) if how == "get" else self._s(accessor, key, how)
        values = self._values.get(k)
        if values is None:
            values = self._s(accessor, key) if how == "get" else self._s(accessor, key, how)
            if hasattr(values, "flags"):
                values = values.view()
                values.flags.writeable = False
            self._values[k] = values
        self._left[k] = left - 1
        if left == 1:  # its last use: no longer kept
            del self._values[k]
        return values

    def __getattr__(self, name):  # anything else the lookups offer (e.g. pairs)
        return getattr(self._s, name)


@register("skim_reuse", "as_written", reference=True,
          description="every skim lookup made as the formulas are written")
def as_written(s, sources: Iterable[str]):
    return s


@register("skim_reuse", "once", agreement="exact",
          description="each skim the formulas name more than once is looked up once per batch and "
                      "shared until its last use")
def once(s, sources: Iterable[str]):
    uses = Counter()
    for src in sources:
        if src:
            uses.update(skim_uses(src))
    if not any(n > 1 for n in uses.values()):
        return s
    return SharedSkims(s, uses)
