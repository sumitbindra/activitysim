"""
Time spent in each swappable piece while a step runs: profiling by piece.

``TimedEngine(engine, times)`` behaves like ``engine`` and records, in
``times`` (a ``SeamTimes``), the time spent in each seam's implementation,
excluding pieces called from inside it (a skim lookup made while a formula
is evaluated counts as ``skim_lookup``, not as ``formulas``). Two more
entries:

* ``formulas``: producing the formula values (terms) that ``utilities`` and
  ``interaction_utilities`` consume; they are generated lazily, inside those
  calls, so this is split out of them;
* ``random``: drawing random numbers (the ``random`` seam's ``uniform``).

The rest of a step's time (outside every piece: preprocessors, data
movement, timetables, ...) is the step's wall time minus the total.
The timing itself costs a few microseconds per call.
"""

from __future__ import annotations

import time
from collections import defaultdict

from .seams import Engine

FORMULAS = "formulas"
RANDOM = "random"
_WITH_TERMS = {"utilities", "interaction_utilities"}


class SeamTimes:
    def __init__(self):
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._stack: list[list[float]] = []

    def _enter(self) -> None:
        self._stack.append([time.perf_counter(), 0.0])

    def _exit(self, piece: str) -> None:
        start, inner = self._stack.pop()
        elapsed = time.perf_counter() - start
        self.seconds[piece] += elapsed - inner
        self.calls[piece] += 1
        if self._stack:
            self._stack[-1][1] += elapsed

    def wrap(self, piece: str, fn):
        def timed(*args, **kwargs):
            self._enter()
            try:
                return fn(*args, **kwargs)
            finally:
                self._exit(piece)
        return timed

    @property
    def total(self) -> float:
        return sum(self.seconds.values())


class _TimedTerms:
    """An iterator of formula values, timing the work of producing each."""

    def __init__(self, terms, times: SeamTimes):
        self._it = None
        self._terms = terms
        self._times = times
        # what produced the values, for implementations that compute them another way
        self.fn = getattr(terms, "fn", None)
        self.args = getattr(terms, "args", None)

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = iter(self._terms)
        self._times._enter()
        try:
            return next(self._it)
        finally:
            self._times._exit(FORMULAS)


class _TimedRandom:
    def __init__(self, inner, times: SeamTimes):
        self._inner = inner
        self.uniform = times.wrap(RANDOM, inner.uniform)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TimedEngine(Engine):
    """``engine``'s implementations, each timed into ``times``."""

    def __init__(self, engine: Engine, times: SeamTimes):  # noqa: super().__init__ not wanted
        self.__dict__["_impl"] = engine._impl
        self.__dict__["times"] = times

    def __getattr__(self, seam: str):
        impl = self.__dict__.get("_impl", {}).get(seam)
        if impl is None:
            raise AttributeError(seam)
        times = self.__dict__["times"]
        fn = impl.fn
        if seam in _WITH_TERMS:
            inner = times.wrap(seam, fn)

            def with_terms(terms, *args, **kwargs):
                return inner(_TimedTerms(terms, times), *args, **kwargs)
            return with_terms
        if seam == "random":
            make = times.wrap(seam, fn)
            return lambda *a, **k: _TimedRandom(make(*a, **k), times)
        return times.wrap(seam, fn)
