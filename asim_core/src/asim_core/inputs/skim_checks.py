"""
Checks of the skims against the model files and the land use, run with the
other input checks (checks.py). They look for skims that would be used
wrongly without any error:

* **References** (warnings): every skim the model files use is in the skim
  files, for every time period it can be used with. Found by a text search,
  so a warning: a step that does use a missing skim stops anyway.
* **Time periods** (warnings): skims for periods network_los.yaml doesn't
  list are not used (as in 1.x).
* **Zone order** (warning): with land use columns given in the region's
  definitions (``skims.zone_order``), the distances must follow them better
  than zones in a random order would: each zone's distance to itself grows
  with its area, distances grow with straight-line distances between zone
  coordinates, and a zone's nearest zone is mostly in its own district. A
  file listing its zones in another order than the model assumes fails all
  three. (A shift by a few zones can't be told apart this way: neighbouring
  zone numbers are neighbouring places. The zone numbering rules are
  checked separately, see ``checks._check_skim_zones``.)
* **Measures** (warnings): with the region's descriptions of its skims
  (``skims.measures``): typical values (to catch a skim in other units),
  largest values, and values within a zone. Without them, values more than
  1,000 times a skim's typical value are reported (a placeholder for "no
  path", used by the models as a number).

Harness-free: only NumPy and pandas.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VARIABLE = "<variable>"  # a skim used with a period from the data (each tour's, ...)
# references are found by a text search of the config files, so they can be wrong; a step
# that does look the skim up stops (asim_core before it starts, 1.x when it gets there)
FOUND_BY_TEXT = "(found by a text search of the config files; a step that uses it stops)"
PLACEHOLDER_RATIO = 1000  # values this many times the typical value look like placeholders
ZONE_ORDER_Z = 2.0  # how far above a random order (in standard deviations) distances must be

# 1.x specs and settings: odt_skims['SOV_TIME'], skim_od[('SOV_TIME', 'AM')], skims[('X', period)]
_LEGACY = re.compile(r"""(\w*skim\w*)\s*\[\s*\(?\s*['"]([A-Za-z0-9_]+)['"]\s*(?:,\s*(['"]?)([A-Za-z0-9_]+)\3)?""")
# skim_dict.lookup(orig, dest, 'DIST') or lookup(orig, dest, ('SOV_TIME', 'AM'))
_LOOKUP = re.compile(r"""\blookup\([^()]*,\s*\(?\s*['"]([A-Za-z0-9_]+)['"]\s*(?:,\s*['"]([A-Za-z0-9_]+)['"])?""")
# asim_core model files: skim.SOV_TIME[o, d, out_period], skim.DIST[o, d]
_NEW = re.compile(r"""\bskim\.([A-Za-z_][A-Za-z0-9_]*)\[([^\]]*)\]""")


def skim_references(configs_dirs: Sequence[str | Path], skip: Iterable[str] = ()
                    ) -> dict[tuple[str, str | None], set[str]]:
    """
    Skims named in the config files: {(name, period): files}. ``period`` is a
    period label when the file names one, ``VARIABLE`` when it comes from the
    data, None when the skim is used without a period. Commented-out lines
    (``#`` first) are skipped, as 1.x skips them; the first configs folder with
    a file name wins, as for every config file.
    """
    refs: dict[tuple[str, str | None], set[str]] = defaultdict(set)
    seen = set(skip)
    for d in configs_dirs:
        for p in sorted(Path(d).glob("*")):
            if p.suffix not in (".csv", ".yaml", ".yml") or p.name in seen:
                continue
            seen.add(p.name)
            try:
                lines = p.read_text(errors="replace").splitlines()
            except OSError:
                continue
            text = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
            for m in _LEGACY.finditer(text):
                name, quote, period = m.group(2), m.group(3), m.group(4)
                refs[(name, None if period is None else period if quote else VARIABLE)].add(p.name)
            for m in _LOOKUP.finditer(text):
                refs[(m.group(1), m.group(2))].add(p.name)
            for m in _NEW.finditer(text):
                args = [a.strip() for a in m.group(2).split(",")]
                period = None
                if len(args) >= 3:
                    a = args[2]
                    period = a.strip("'\"") if a[:1] in "'\"" else VARIABLE
                refs[(m.group(1), period)].add(p.name)
    return dict(refs)


@dataclass
class Found:
    severity: str
    column: str | None
    message: str
    rows: int = 0
    examples: list[str] = field(default_factory=list)
    used_by: list[str] | None = None


def reference_problems(names: Iterable[str], labels: Sequence[str] | None,
                       refs: dict[tuple[str, str | None], set[str]]) -> list[Found]:
    names = set(names)
    periods_of: dict[str, set[str]] = defaultdict(set)
    for n in names:
        base, sep, period = n.partition("__")
        if sep:
            periods_of[base].add(period)
    out = []
    by_name: dict[str, list] = defaultdict(list)
    for (name, period), files in refs.items():
        by_name[name].append((period, files))
    for name in sorted(by_name):
        uses = by_name[name]
        files = sorted(set().union(*(f for _, f in uses)))
        if name in names:
            continue  # a skim without periods: any use finds it
        have = periods_of.get(name, set())
        if not have:
            out.append(Found("warning", name, "is used by the model files but is not in the skims "
                             f"{FOUND_BY_TEXT}", used_by=files))
            continue
        named = sorted({p for p, _ in uses if p not in (None, VARIABLE)} - have)
        if named:
            out.append(Found("warning", name, f"is used for time period(s) {named}, but there is no "
                             f"skim {name}__{named[0]} {FOUND_BY_TEXT}", 0, [f"{name}__{p}" for p in named],
                             used_by=files))
        if labels is not None and any(p in (None, VARIABLE) for p, _ in uses):
            missing = [p for p in labels if p not in have]
            if missing:
                out.append(Found("warning", name, f"is used with each row's time period, but has no "
                                 f"skim for period(s) {missing}: rows in those periods can't be looked up "
                                 f"{FOUND_BY_TEXT}",
                                 0, [f"{name}__{p}" for p in missing], used_by=files))
    if labels is not None:
        unknown: dict[str, list[str]] = defaultdict(list)
        for n in sorted(names):
            base, sep, period = n.partition("__")
            if sep and period not in labels:
                unknown[period].append(n)
        for period, skims in sorted(unknown.items()):
            out.append(Found("warning", None, f"{len(skims):,} skims have time period '{period}', which "
                             f"isn't in network_los.yaml's skim_time_periods {list(labels)}: they are "
                             "not used (as in 1.x)", 0, skims[:5], used_by=[]))
    return out


# --- zone order ------------------------------------------------------------------------


def _ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class OrderTest:
    name: str
    value: float  # the statistic for the zones as they are
    random_mean: float  # and for zones in a random order
    random_sd: float

    @property
    def z(self) -> float:
        return (self.value - self.random_mean) / self.random_sd if self.random_sd > 0 else float("nan")

    def text(self) -> str:
        return (f"{self.name} {self.value:.2f} (in a random zone order: {self.random_mean:.2f} "
                f"+/- {self.random_sd:.2f})")


def zone_order_tests(distance: np.ndarray, area=None, x=None, y=None, district=None,
                     permutations: int = 30, seed: int = 0) -> list[OrderTest]:
    """
    How well a distance skim (rows and columns = land use zones, in the model's
    order) follows land use columns, against the same zones in random orders.
    """
    n = len(distance)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n) for _ in range(permutations)]
    tests = []

    def test(name, stat):
        value = stat(np.arange(n))
        if not np.isfinite(value):
            return
        null = np.array([stat(p) for p in perms])
        null = null[np.isfinite(null)]
        if len(null) < 2:
            return
        tests.append(OrderTest(name, value, float(null.mean()), float(null.std(ddof=1))))

    if area is not None:
        diag = np.diag(distance).astype(np.float64)
        size = np.sqrt(np.clip(np.asarray(area, dtype=np.float64), 0, None))
        ok = np.isfinite(diag) & np.isfinite(size)
        idx = np.flatnonzero(ok)
        # zone i's land use is at position p[i] when zones are in a random order
        test("each zone's distance to itself against the square root of its area",
             lambda p: _rank_corr(diag[idx], size[p][idx]))
    if x is not None and y is not None:
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        m = min(200_000, n * (n - 1) // 2)
        i = rng.integers(0, n, m)
        j = rng.integers(0, n, m)
        keep = i != j
        i, j = i[keep], j[keep]
        d = distance[i, j].astype(np.float64)
        test("distances against straight-line distances between zone coordinates",
             lambda p: _rank_corr(d, np.hypot(x[p][i] - x[p][j], y[p][i] - y[p][j])))
    if district is not None:
        district = np.asarray(district)
        off = distance.astype(np.float64) + np.diag(np.full(n, np.inf))
        nearest = off.argmin(axis=1)
        _, counts = np.unique(district, return_counts=True)
        if (counts / n).max() < 0.5:  # one district for most zones says nothing
            test("share of zones whose nearest zone is in the same district",
                 lambda p: float((district[p][nearest] == district[p]).mean()))
    return tests


# --- measures ---------------------------------------------------------------------------


@dataclass
class MatrixStats:
    name: str
    typical: float  # median of positive values (on a sample of cells)
    largest: float
    placeholder_cells: int = 0
    placeholder_examples: list[float] = field(default_factory=list)
    above_max: int = 0
    above_max_examples: list[float] = field(default_factory=list)
    zero_within: int = 0  # land use zones whose value to itself is <= 0
    zero_within_examples: list[str] = field(default_factory=list)
    per_distance: float | None = None  # median of value / distance, where both are positive


def matrix_stats(name: str, a: np.ndarray, measure=None, own_rows=None, zone_labels=None,
                 distance: np.ndarray | None = None) -> MatrixStats:
    """
    ``own_rows``: each land use zone's row (= column), for values within a zone;
    ``zone_labels``: their zone IDs, for examples; ``distance``: the region's
    distance skim, for values per unit of distance.
    """
    no_path = None if measure is None else measure.no_path
    real = a if no_path is None else np.where(a == no_path, 0, a)  # 'no path' left out
    step = max(1, a.shape[0] // 300)
    sample = real[::step, ::step]
    pos = sample[sample > 0]
    typical = float(np.median(pos)) if len(pos) else 0.0
    largest = float(np.nanmax(real)) if a.size else 0.0
    st = MatrixStats(name, typical, largest)
    a = real
    if typical > 0 and largest >= PLACEHOLDER_RATIO * typical:
        big = a[a >= PLACEHOLDER_RATIO * typical]
        st.placeholder_cells = int(big.size)
        vals, counts = np.unique(big, return_counts=True)
        st.placeholder_examples = [float(v) for v in vals[np.argsort(-counts)][:3]]
    if measure is not None and measure.max is not None and largest > measure.max:
        over = a[a > measure.max]
        st.above_max = int(over.size)
        vals, counts = np.unique(over, return_counts=True)
        st.above_max_examples = [float(v) for v in vals[np.argsort(-counts)][:3]]
    if measure is not None and measure.per_distance is not None and distance is not None \
            and distance.shape == a.shape:
        d = distance[::step, ::step]
        both = (sample > 0) & (d > 0)
        if both.any():
            st.per_distance = float(np.median(sample[both] / d[both]))
    if measure is not None and measure.intrazonal == "positive" and own_rows is not None:
        within = a[own_rows, own_rows]
        bad = np.flatnonzero(~(within > 0))
        st.zero_within = int(len(bad))
        if zone_labels is not None:
            st.zero_within_examples = [f"zone {zone_labels[k]}: {within[k]:.6g}" for k in bad[:5]]
    return st
