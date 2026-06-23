# ActivitySim
# See full license in LICENSE.txt.
"""
Reference intermediate representation (IR) for destination/location choice
utility expressions.

This module is the Python "crux" of the Rust kernel project (Milestone 1). It
does two things, both in pure Python with **no Rust involved**:

1. Compiles a single utility-spec expression string into a small, restricted
   IR (a tree of dataclasses covering only the operations actually used in the
   destination/location choice specs -- see ``docs/rust_kernel_recon.md`` §4).
   Expressions outside that finite set raise :class:`UnsupportedExpression`,
   which is how the dispatcher decides to fall back to the existing Python path.

2. Provides a slow, simple **reference evaluator** that walks the IR and
   produces numpy arrays. The evaluator deliberately interprets the IR data
   structure (not the original Python string), so it exercises exactly the
   semantics a future Rust evaluator must reproduce.

The accompanying parity test asserts that, for representative expressions, the
IR evaluator's output equals stock pandas/numpy evaluation of the original
expression. Proving that here de-risks the IR design before any Rust is written.

The IR is intentionally minimal; it grows as coverage expands (Milestone 7).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


class UnsupportedExpression(Exception):
    """Raised when an expression uses an operation the IR does not support.

    The dispatcher catches this at spec-compile time and falls back to the
    existing pure-Python evaluation for the affected model component.
    """


# ---------------------------------------------------------------------------
# IR node types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    """Base class for IR nodes."""


@dataclass(frozen=True)
class Num(Node):
    """A numeric constant."""

    value: float


@dataclass(frozen=True)
class ColRef(Node):
    """Reference to a column of the interaction dataframe (chooser or alt).

    Covers ``df.colname``, ``df['colname']`` and bare ``colname``.
    """

    name: str


@dataclass(frozen=True)
class ConstRef(Node):
    """Reference to a named scalar constant from the model ``CONSTANTS`` block."""

    name: str


@dataclass(frozen=True)
class SkimRef(Node):
    """A 2-D skim lookup, e.g. ``skims['DIST']`` (resolved per row upstream)."""

    accessor: str  # e.g. "skims", "od_skims", "dp_skims"
    key: str


@dataclass(frozen=True)
class SizeTerm(Node):
    """A size-term lookup, e.g. ``size_terms.get(df.dest_taz, df.purpose)``."""

    zone: Node
    segment: Node


@dataclass(frozen=True)
class BinOp(Node):
    op: str  # '+', '-', '*', '/', '**', '&', '|'
    left: Node
    right: Node


@dataclass(frozen=True)
class UnaryOp(Node):
    op: str  # 'neg', 'invert'
    operand: Node


@dataclass(frozen=True)
class Compare(Node):
    op: str  # '==', '!=', '<', '<=', '>', '>='
    left: Node
    right: Node


@dataclass(frozen=True)
class Func(Node):
    """A supported function application (numpy free function or array method)."""

    name: str  # 'log', 'log1p', 'exp', 'abs', 'clip', 'minimum', 'maximum', 'where'
    args: tuple = field(default_factory=tuple)


# function name -> required arg count (None means variadic-ish handled explicitly)
_SUPPORTED_FUNCS = {
    "log": 1,
    "log1p": 1,
    "exp": 1,
    "abs": 1,
    "sqrt": 1,
    "clip": 3,  # (x, lo, hi)
    "minimum": 2,
    "maximum": 2,
    "where": 3,
}

_SKIM_ACCESSORS = {"skims", "od_skims", "dp_skims", "do_skims", "pd_skims", "dpt_skims"}

_BINOPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
    ast.BitAnd: "&",
    ast.BitOr: "|",
}

_CMPOPS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


# ---------------------------------------------------------------------------
# Compiler: Python expression string -> IR
# ---------------------------------------------------------------------------
def compile_expression(expr: str, constants: set[str] | None = None) -> Node:
    """Compile a single spec expression into IR.

    Parameters
    ----------
    expr : str
        The expression as it appears in a spec ``Expression`` cell. A leading
        ``@`` (python-eval marker) is stripped. Temp-assignment expressions
        (leading ``_name@...``) should be split by the caller before this.
    constants : set of str, optional
        Names that should be treated as scalar model constants rather than
        dataframe columns.

    Returns
    -------
    Node
        The root IR node.

    Raises
    ------
    UnsupportedExpression
        If the expression uses anything outside the supported IR.
    """
    constants = constants or set()
    src = expr.strip()
    if src.startswith("@"):
        src = src[1:]
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise UnsupportedExpression(f"cannot parse {expr!r}: {e}") from e
    return _Compiler(constants).visit(tree.body)


class _Compiler:
    def __init__(self, constants: set[str]):
        self.constants = constants

    def visit(self, node: ast.AST) -> Node:
        method = getattr(self, f"v_{type(node).__name__}", None)
        if method is None:
            raise UnsupportedExpression(
                f"unsupported syntax node: {type(node).__name__}"
            )
        return method(node)

    # literals -------------------------------------------------------------
    def v_Constant(self, node: ast.Constant) -> Node:
        if isinstance(node.value, bool):
            return Num(1.0 if node.value else 0.0)
        if isinstance(node.value, (int, float)):
            return Num(float(node.value))
        raise UnsupportedExpression(f"unsupported constant: {node.value!r}")

    # names ----------------------------------------------------------------
    def v_Name(self, node: ast.Name) -> Node:
        if node.id in self.constants:
            return ConstRef(node.id)
        # a bare name in a dest-choice spec is a column passthrough
        return ColRef(node.id)

    # df.col  /  np.func is handled in Call ; here handle attribute access ---
    def v_Attribute(self, node: ast.Attribute) -> Node:
        # df.colname -> ColRef
        if isinstance(node.value, ast.Name) and node.value.id == "df":
            return ColRef(node.attr)
        raise UnsupportedExpression(
            f"unsupported attribute access: .{node.attr}"
        )

    # skims['KEY'] / df['col'] --------------------------------------------
    def v_Subscript(self, node: ast.Subscript) -> Node:
        base = node.value
        key = _literal_str(node.slice)
        if isinstance(base, ast.Name):
            if base.id in _SKIM_ACCESSORS:
                if key is None:
                    raise UnsupportedExpression("non-literal skim key")
                return SkimRef(base.id, key)
            if base.id == "df":
                if key is None:
                    raise UnsupportedExpression("non-literal column key")
                return ColRef(key)
        raise UnsupportedExpression("unsupported subscript")

    # arithmetic / bitwise -------------------------------------------------
    def v_BinOp(self, node: ast.BinOp) -> Node:
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise UnsupportedExpression(f"unsupported binop: {type(node.op).__name__}")
        return BinOp(op, self.visit(node.left), self.visit(node.right))

    def v_UnaryOp(self, node: ast.UnaryOp) -> Node:
        if isinstance(node.op, ast.USub):
            return UnaryOp("neg", self.visit(node.operand))
        if isinstance(node.op, ast.Invert):
            return UnaryOp("invert", self.visit(node.operand))
        if isinstance(node.op, ast.UAdd):
            return self.visit(node.operand)
        raise UnsupportedExpression(f"unsupported unaryop: {type(node.op).__name__}")

    def v_Compare(self, node: ast.Compare) -> Node:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise UnsupportedExpression("chained comparisons not supported")
        op = _CMPOPS.get(type(node.ops[0]))
        if op is None:
            raise UnsupportedExpression("unsupported comparison")
        return Compare(op, self.visit(node.left), self.visit(node.comparators[0]))

    # function / method calls ---------------------------------------------
    def v_Call(self, node: ast.Call) -> Node:
        func = node.func
        # np.func(...)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in ("np", "numpy")
        ):
            return self._numpy_call(func.attr, node)
        # size_terms.get(zone, segment)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "size_terms"
            and func.attr == "get"
        ):
            if len(node.args) != 2:
                raise UnsupportedExpression("size_terms.get needs (zone, segment)")
            return SizeTerm(self.visit(node.args[0]), self.visit(node.args[1]))
        # method call on an expression: x.clip(lo, hi), x.apply(np.log1p),
        # x.astype(...)
        if isinstance(func, ast.Attribute):
            return self._method_call(func, node)
        raise UnsupportedExpression("unsupported call form")

    def _numpy_call(self, name: str, node: ast.Call) -> Node:
        if name not in _SUPPORTED_FUNCS:
            raise UnsupportedExpression(f"unsupported numpy function: np.{name}")
        nargs = _SUPPORTED_FUNCS[name]
        if len(node.args) != nargs:
            raise UnsupportedExpression(
                f"np.{name} expects {nargs} args, got {len(node.args)}"
            )
        return Func(name, tuple(self.visit(a) for a in node.args))

    def _method_call(self, func: ast.Attribute, node: ast.Call) -> Node:
        recv = self.visit(func.value)
        attr = func.attr
        if attr == "clip":
            # x.clip(lo, hi)
            if len(node.args) != 2:
                raise UnsupportedExpression(".clip needs (lo, hi)")
            lo = self.visit(node.args[0])
            hi = self.visit(node.args[1])
            return Func("clip", (recv, lo, hi))
        if attr == "apply":
            # x.apply(np.log1p) and similar single-ufunc applies
            if len(node.args) != 1:
                raise UnsupportedExpression(".apply needs one function arg")
            fn = node.args[0]
            if (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id in ("np", "numpy")
                and fn.attr in _SUPPORTED_FUNCS
                and _SUPPORTED_FUNCS[fn.attr] == 1
            ):
                return Func(fn.attr, (recv,))
            raise UnsupportedExpression("unsupported .apply target")
        if attr == "astype":
            # treat astype(float)/astype('float') as identity for numeric IR
            return recv
        raise UnsupportedExpression(f"unsupported method: .{attr}")


def _literal_str(slice_node: ast.AST) -> str | None:
    """Extract a string literal subscript key, or None if not a literal."""
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


# ---------------------------------------------------------------------------
# Reference evaluator: IR -> numpy array
# ---------------------------------------------------------------------------
@dataclass
class EvalContext:
    """Inputs the reference evaluator reads from.

    Attributes
    ----------
    columns : mapping of column name -> 1-D numpy array (length n_rows)
    skims : mapping of accessor name -> mapping of key -> 1-D numpy array
        Skim values already resolved per interaction row (as the real pipeline
        provides via the skim wrappers).
    constants : mapping of constant name -> scalar
    size_terms : callable (zone_array, segment_array) -> 1-D numpy array, optional
    """

    columns: dict[str, np.ndarray]
    skims: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    constants: dict[str, float] = field(default_factory=dict)
    size_terms: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None


def evaluate(node: Node, ctx: EvalContext) -> np.ndarray:
    """Evaluate an IR node against a context, returning a numpy array."""
    t = type(node)
    if t is Num:
        return np.float64(node.value)
    if t is ColRef:
        try:
            return np.asarray(ctx.columns[node.name])
        except KeyError as e:
            raise UnsupportedExpression(f"unknown column {node.name!r}") from e
    if t is ConstRef:
        return np.float64(ctx.constants[node.name])
    if t is SkimRef:
        return np.asarray(ctx.skims[node.accessor][node.key])
    if t is SizeTerm:
        if ctx.size_terms is None:
            raise UnsupportedExpression("size_terms not provided")
        zone = evaluate(node.zone, ctx)
        seg = evaluate(node.segment, ctx)
        return np.asarray(ctx.size_terms(zone, seg))
    if t is BinOp:
        a = evaluate(node.left, ctx)
        b = evaluate(node.right, ctx)
        return _apply_binop(node.op, a, b)
    if t is UnaryOp:
        v = evaluate(node.operand, ctx)
        if node.op == "neg":
            return -v
        if node.op == "invert":
            return ~np.asarray(v, dtype=bool) if v.dtype == bool else np.invert(v)
    if t is Compare:
        a = evaluate(node.left, ctx)
        b = evaluate(node.right, ctx)
        return _apply_compare(node.op, a, b)
    if t is Func:
        return _apply_func(node.name, [evaluate(a, ctx) for a in node.args])
    raise UnsupportedExpression(f"cannot evaluate node {t.__name__}")


def _apply_binop(op: str, a, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    if op == "**":
        return a**b
    if op == "&":
        return np.asarray(a, dtype=bool) & np.asarray(b, dtype=bool)
    if op == "|":
        return np.asarray(a, dtype=bool) | np.asarray(b, dtype=bool)
    raise UnsupportedExpression(f"binop {op}")


def _apply_compare(op: str, a, b):
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    raise UnsupportedExpression(f"compare {op}")


def _apply_func(name: str, args: list):
    if name == "log":
        return np.log(args[0])
    if name == "log1p":
        return np.log1p(args[0])
    if name == "exp":
        return np.exp(args[0])
    if name == "abs":
        return np.abs(args[0])
    if name == "sqrt":
        return np.sqrt(args[0])
    if name == "clip":
        return np.clip(args[0], args[1], args[2])
    if name == "minimum":
        return np.minimum(args[0], args[1])
    if name == "maximum":
        return np.maximum(args[0], args[1])
    if name == "where":
        return np.where(args[0], args[1], args[2])
    raise UnsupportedExpression(f"func {name}")


# ---------------------------------------------------------------------------
# Spec-level helpers
# ---------------------------------------------------------------------------
def split_temp_assignment(expr: str) -> tuple[str | None, str]:
    """Split a spec expression into an optional temp name and the expression.

    ActivitySim spec rows beginning with ``_name@expr`` bind a temporary into
    the locals namespace rather than contributing to the utility. Returns
    ``(temp_name, expr_without_marker)`` or ``(None, expr)``.
    """
    s = expr.strip()
    if s.startswith("_") and "@" in s:
        name, _, rhs = s.partition("@")
        return name.strip(), rhs.strip()
    return None, s


def can_compile(expr: str, constants: set[str] | None = None) -> bool:
    """Return True if ``expr`` compiles to IR (used by the fallback dispatcher)."""
    _, body = split_temp_assignment(expr)
    try:
        compile_expression(body, constants)
        return True
    except UnsupportedExpression:
        return False
