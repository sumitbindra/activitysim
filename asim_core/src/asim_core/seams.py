"""
Swappable pieces ("seams").

Each seam is one job inside a model step. Every seam has a **reference**
implementation: plain NumPy, readable, and the one that must match
ActivitySim 1.x. Alternatives register under a name, and which one runs is a
setting, not a code change::

    Engine({"utilities": "accumulate", "random": "counter"})

Seams not named use the reference. Each implementation declares how its
results relate to the reference, which tells the experiment harness how to
compare them:

* ``exact``        identical results
* ``approximate``  identical up to floating-point rounding, so individual
                   choices almost always match
* ``statistical``  different individual results with the same distribution
                   (for example, different random numbers), so only shares and
                   distributions can be compared

To add an alternative, write a function with the same signature as the
reference and decorate it with ``@register(seam, name, agreement=...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

SEAMS = {
    "lookup": "values of a column at given row positions (joins between tables): "
              "lookup(values, positions) -> values",
    "skim_access": "how a step holds the skims it reads while it runs (memory-mapped, pages mapped "
                   "up front, or copied into memory): ",
    "skim_lookup": "skim values for origin/destination positions: "
                   "skim_lookup(matrix, orig_positions, dest_positions, pairs=None) -> values "
                   "(pairs: the same positions as a ZonePairs, shared by the skims looked up for them)",
    "utilities": "formula values and coefficients -> utilities: "
                 "utilities(terms, coefficients, n) -> (n x alternatives) array",
    "interaction_utilities": "one utility per chooser-alternative row (e.g. a tour's departure "
                             "options): interaction_utilities(terms, coefficients, n) -> (n,) array",
    "probabilities": "utilities -> probabilities: "
                     "probabilities(utils, shift) -> (probs, unusable_rows)",
    "nested_probabilities": "utilities and a nest tree -> probabilities (nested logit): "
                            "nested_probabilities(utils, alternatives, nests) -> (probs, unusable_rows, logsums)",
    "nested_logsums": "utilities and a nest tree -> logsums only (nested logit): "
                      "nested_logsums(utils, alternatives, nests) -> (n,) array",
    "logsum_keys": "which scheduling option rows share a mode choice logsum: "
                   "logsum_keys(read_columns, has_segments) -> option columns to group by",
    "sample": "probabilities and k random draws per row -> k sampled alternatives (with "
              "replacement): sample(probs, rands) -> (positions, probabilities as float32)",
    "choose": "probabilities and one random draw per row -> chosen position: "
              "choose(probs, rands) -> int32 array",
    "random": "per-row uniform random draws for a step: "
              "random(step, base_seed) -> object with .uniform(channel, ids, n)",
}

AGREEMENT_ORDER = ["exact", "approximate", "statistical"]


@dataclass(frozen=True)
class Implementation:
    seam: str
    name: str
    fn: Callable
    agreement: str
    description: str
    reference: bool


_REGISTRY: dict[str, dict[str, Implementation]] = {s: {} for s in SEAMS}


def register(seam: str, name: str, agreement: str = "exact", description: str = "",
             reference: bool = False):
    """Decorator: register ``fn`` as implementation ``name`` of ``seam``."""
    if seam not in SEAMS:
        raise KeyError(f"unknown seam '{seam}'; seams are {sorted(SEAMS)}")
    if agreement not in AGREEMENT_ORDER:
        raise ValueError(f"agreement must be one of {AGREEMENT_ORDER}")

    def deco(fn):
        if reference and any(i.reference for i in _REGISTRY[seam].values()):
            raise ValueError(f"seam '{seam}' already has a reference implementation")
        _REGISTRY[seam][name] = Implementation(
            seam, name, fn, "exact" if reference else agreement,
            description or (fn.__doc__ or "").strip().split("\n")[0], reference,
        )
        return fn

    return deco


def _load_builtin() -> None:
    # importing these modules registers the built-in implementations
    from . import alternatives, logit, logsums, lookups, random, skim_access  # noqa: F401


def implementations(seam: str | None = None) -> list[Implementation]:
    _load_builtin()
    seams = [seam] if seam else list(SEAMS)
    return [i for s in seams for i in _REGISTRY[s].values()]


def reference_name(seam: str) -> str:
    _load_builtin()
    for i in _REGISTRY[seam].values():
        if i.reference:
            return i.name
    raise LookupError(f"seam '{seam}' has no reference implementation")


class Engine:
    """The implementation chosen for each seam. Unnamed seams use the reference."""

    def __init__(self, choices: Mapping[str, str] | None = None):
        _load_builtin()
        choices = dict(choices or {})
        unknown = set(choices) - set(SEAMS)
        if unknown:
            raise KeyError(f"unknown seam(s) {sorted(unknown)}; seams are {sorted(SEAMS)}")
        self._impl: dict[str, Implementation] = {}
        for seam in SEAMS:
            name = choices.get(seam) or reference_name(seam)
            if name not in _REGISTRY[seam]:
                raise KeyError(
                    f"no implementation '{name}' for seam '{seam}'; "
                    f"available: {sorted(_REGISTRY[seam])}"
                )
            self._impl[seam] = _REGISTRY[seam][name]

    def __getattr__(self, seam: str) -> Callable:
        impl = self.__dict__.get("_impl", {}).get(seam)
        if impl is None:
            raise AttributeError(seam)
        return impl.fn

    @property
    def choices(self) -> dict[str, str]:
        return {s: i.name for s, i in self._impl.items()}

    @property
    def changed(self) -> dict[str, str]:
        """Seams using something other than the reference."""
        return {s: i.name for s, i in self._impl.items() if not i.reference}

    @property
    def agreement(self) -> str:
        """The loosest agreement with the reference among the chosen implementations."""
        levels = [AGREEMENT_ORDER.index(i.agreement) for i in self._impl.values()]
        return AGREEMENT_ORDER[max(levels)]

    def __repr__(self) -> str:
        return f"Engine({self.changed or 'reference'})"
