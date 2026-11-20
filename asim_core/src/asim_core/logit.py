"""
Multinomial logit: utilities -> probabilities -> choices.

These are the **reference** implementations of the ``utilities``,
``probabilities`` and ``choose`` seams (see ``seams.py``). Their arithmetic
follows ActivitySim 1.x step for step (activitysim/core/simulate.py
``eval_utilities`` and logit.py ``utils_to_probs`` / ``make_choices``),
because floating-point results depend on the order of operations and exact
matching needs the same order.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .seams import register

EXP_UTIL_MIN = 1e-300
PROB_MIN = 0.0
PROB_MAX = 1.0
BAD_PROB_THRESHOLD = 0.001
BAD_CHOICE = -99


def term_values(terms: Iterable, n: int, n_terms: int | None = None) -> np.ndarray:
    """Stack per-term values (arrays or scalars) into a (terms x choosers) float64 array."""
    terms = list(terms) if n_terms is None else terms
    k = len(terms) if n_terms is None else n_terms
    out = np.empty((k, n), dtype=np.float64)
    i = -1
    for i, v in enumerate(terms):
        out[i] = v
    if i + 1 != k:
        raise ValueError(f"expected {k} term values, got {i + 1}")
    return out


@register("utilities", "matmul", reference=True,
          description="stack all term values (terms x choosers, 64-bit) and multiply by the "
                      "coefficients in one matrix product, as ActivitySim 1.x does")
def utilities(terms: Iterable, coefficients: np.ndarray, n: int) -> np.ndarray:
    """(choosers x alternatives) utilities from term values and (terms x alternatives) coefficients."""
    values = term_values(terms, n, n_terms=coefficients.shape[0])
    return np.dot(values.transpose(), coefficients)


@register("interaction_utilities", "sequential", reference=True,
          description="add term value x coefficient one term at a time into a 64-bit total, "
                      "as 1.x's eval_interaction_utilities does")
def interaction_utilities(terms: Iterable, coefficients: list[float], n: int) -> np.ndarray:
    """
    Utilities of ``n`` chooser-alternative rows. Each product is computed in
    the term's own type (a 32-bit column times its coefficient stays 32-bit,
    as in 1.x), then added to the 64-bit total. ``coefficients`` should be
    Python floats (see InteractionModelDefinition.utility_terms).
    """
    total = np.zeros(n, dtype=np.float64)
    k = -1
    for k, (v, c) in enumerate(zip(terms, coefficients)):
        total += np.asarray(v * c, dtype=np.float64)
    if k + 1 != len(coefficients):
        raise ValueError(f"expected {len(coefficients)} term values, got {k + 1}")
    return total


@register("probabilities", "asim1", reference=True,
          description="exponentiate, zero out values below 1e-300, divide by row totals")
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


@register("choose", "sequential", reference=True,
          description="subtract each alternative's probability from the draw in turn; "
                      "first one to reach zero or below is chosen, as ActivitySim 1.x does")
def choose(probs: np.ndarray, rands: np.ndarray) -> np.ndarray:
    """
    Pick an alternative (column position) per row. Rows whose probabilities
    do not sum to 1 (within 0.001) get -99, as in ActivitySim 1.x.
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


# --- nested logit ---------------------------------------------------------------------


@dataclass
class _Node:
    name: str
    is_leaf: bool
    coefficient: float
    product: float  # product of nesting coefficients from the root down
    alternatives: list[str] = field(default_factory=list)
    ancestors: list[str] = field(default_factory=list)


def walk_nests(tree, post_order: bool = False) -> list[_Node]:
    """
    Every nest and alternative in the tree, as ActivitySim 1.x walks it
    (logit.each_nest): pre-order or post-order, carrying the product of
    nesting coefficients and the list of ancestors.
    """
    out: list[_Node] = []

    def visit(spec, parent: _Node):
        if isinstance(spec, str):  # an alternative (leaf)
            out.append(_Node(spec, True, parent.coefficient, parent.product,
                             ancestors=parent.ancestors + [spec]))
            return
        node = _Node(spec.name, False, spec.coefficient, parent.product * spec.coefficient,
                     [a if isinstance(a, str) else a.name for a in spec.alternatives],
                     parent.ancestors + [spec.name])
        if not post_order:
            out.append(node)
        for a in spec.alternatives:
            visit(a, node)
        if post_order:
            out.append(node)

    visit(tree, _Node("", False, 0, 1))
    return out


def _sum_columns(columns: list[np.ndarray]) -> np.ndarray:
    """Add columns left to right (the order 1.x's row sums use for nests)."""
    total = np.array(columns[0], dtype=np.float64, copy=True)
    for c in columns[1:]:
        total += c
    return total


@register("nested_probabilities", "asim1", reference=True,
          description="nested logit as ActivitySim 1.x computes it: exponentiated nest "
                      "utilities, conditional probabilities per nest, multiplied down the tree")
def nested_probabilities(utils: np.ndarray, alternatives: list[str], nests
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Probabilities of each alternative (columns in ``alternatives`` order), a
    mask of rows that cannot be used, and each row's logsum (log of the
    root's exponentiated utility, as 1.x computes it). Follows 1.x's
    compute_nested_exp_utilities, compute_nested_probabilities and
    compute_base_probabilities (activitysim/core/simulate.py) operation by
    operation.
    """
    n = utils.shape[0]
    col = {a: j for j, a in enumerate(alternatives)}
    bad = np.zeros(n, dtype=bool)

    # exponentiated utilities of alternatives and nests, from the leaves up
    exp_u: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for node in walk_nests(nests, post_order=True):
            if node.is_leaf:
                x = utils[:, col[node.name]].astype(float) / node.product
            else:
                kids = [np.where(np.isnan(exp_u[c]), 0.0, exp_u[c]) for c in node.alternatives]
                x = node.coefficient * np.log(_sum_columns(kids))  # pandas sum skips NaN
            exp_u[node.name] = np.exp(x)

        # probability of each child within its nest, from the root down
        cond: dict[str, np.ndarray] = {}
        for node in walk_nests(nests, post_order=False):
            if node.is_leaf:
                continue
            arr = np.column_stack([exp_u[c] for c in node.alternatives]).astype(np.float64)
            np.putmask(arr, arr <= EXP_UTIL_MIN, 0)
            total = _sum_columns([arr[:, k] for k in range(arr.shape[1])])
            bad |= np.isinf(total) | np.isnan(total)
            np.divide(arr, total.reshape(n, 1), out=arr)
            np.putmask(arr, np.isnan(arr), PROB_MIN)
            np.clip(arr, PROB_MIN, PROB_MAX, out=arr)
            for k, c in enumerate(node.alternatives):
                cond[c] = arr[:, k]

    # each alternative: product of the conditional probabilities on its path
    probs = np.empty((n, len(alternatives)), dtype=np.float64)
    for node in walk_nests(nests, post_order=False):
        if not node.is_leaf:
            continue
        path = node.ancestors[1:]  # not the root
        p = cond[path[0]].copy()
        for a in path[1:]:
            p = p * cond[a]
        probs[:, col[node.name]] = p
    bad |= np.abs(probs.sum(axis=1) - 1.0) > BAD_PROB_THRESHOLD
    with np.errstate(divide="ignore", invalid="ignore"):
        logsums = np.log(exp_u[nests.name])
    return probs, bad, logsums
