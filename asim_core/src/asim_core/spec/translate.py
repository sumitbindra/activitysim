"""
Formula translator: turns today's spec formulas into plain NumPy code.

ActivitySim 1.x formulas come in two styles:

* plain, e.g. ``num_drivers==2``: evaluated by pandas against the chooser
  table, so every bare name is a column;
* starting with ``@``, e.g. ``@df.income_in_thousands.clip(0, 30)``: run as
  Python, with ``df`` (the choosers), ``np``, and named constants available.

Both are parsed with Python's built-in ``ast`` module (nothing is executed),
checked against a small list of allowed operations, and rewritten as a NumPy
expression over a mapping of column arrays ``c``. For example::

    @(df.num_workers==0)*df.density_index.clip(0, 10)
    ->  ((c['num_workers'] == 0) * _clip(c['density_index'], 0, 10))

Constants are written into the code as literal values. Anything outside the
supported subset raises ``FormulaError`` naming the formula and the part that
is not supported, so problems show up when a model loads, not mid-run.

Assignment specs (preprocessors, annotations, accessibility) are always
Python style, and can also use variables assigned by earlier rows (``v[name]``),
skim lookups such as ``skim_od[('SOVTOLL_TIME', 'AM')]`` (``s(accessor,
'SOVTOLL_TIME__AM')``) and bare functions such as ``exp`` (``np.exp``).
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


class FormulaError(ValueError):
    pass


# numpy functions a formula may call as np.<name>(...)
NP_FUNCTIONS = {
    "where", "minimum", "maximum", "clip", "isin",
    "log", "log1p", "exp", "sqrt", "abs", "floor", "ceil",
}

_BINOPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
}
_CMPOPS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
}
_UNARY = {ast.USub: "-", ast.UAdd: "+", ast.Invert: "~"}


@dataclass
class Translated:
    formula: str
    source: str
    columns: set[str] = field(default_factory=set)
    constants: set[str] = field(default_factory=set)
    skims: set[str] = field(default_factory=set)  # skim names, e.g. 'SOVTOLL_TIME__AM'
    names: set[str] = field(default_factory=set)  # earlier assigned variables used


def _literal(value) -> str:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    raise FormulaError(f"constant value {value!r} is not a number, string or list")


class _Translator:
    def __init__(
        self,
        formula: str,
        constants: Mapping[str, object],
        python_mode: bool,
        assigned: set[str] = frozenset(),
        functions: Mapping[str, str] | None = None,
        skim_accessors: set[str] = frozenset(),
    ):
        self.formula = formula
        self.constants = constants
        self.python_mode = python_mode
        self.assigned = assigned
        self.functions = functions or {}
        self.skim_accessors = skim_accessors
        self.columns: set[str] = set()
        self.used_constants: set[str] = set()
        self.skims: set[str] = set()
        self.names: set[str] = set()

    def fail(self, node: ast.AST, why: str):
        part = ast.unparse(node) if isinstance(node, ast.AST) else str(node)
        raise FormulaError(f"formula {self.formula!r}: {why} (in '{part}')")

    def column(self, name: str) -> str:
        self.columns.add(name)
        return f"c[{name!r}]"

    def visit(self, node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return self.visit(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float, str)):
                return repr(node.value)
            self.fail(node, "unsupported literal")

        if isinstance(node, (ast.List, ast.Tuple)):
            return "[" + ", ".join(self.visit(e) for e in node.elts) + "]"

        if isinstance(node, ast.Name):
            if not self.python_mode:
                return self.column(node.id)  # pandas style: bare names are columns
            if node.id in self.assigned:  # a variable assigned by an earlier row
                self.names.add(node.id)
                return f"v[{node.id!r}]"
            if node.id in self.constants:
                self.used_constants.add(node.id)
                return _literal(self.constants[node.id])
            self.fail(node, f"unknown name '{node.id}' (not a constant or an earlier "
                            f"variable; columns are written df.name)")

        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "df":
                return self.column(node.attr)
            self.fail(node, "unsupported attribute")

        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "df"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                return self.column(node.slice.value)
            if isinstance(node.value, ast.Name) and node.value.id in self.skim_accessors:
                return self.skim(node)
            self.fail(node, "unsupported subscript (only df['column'] and skim lookups are allowed)")

        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                self.fail(node, "unsupported operator")
            return f"({self.visit(node.left)} {op} {self.visit(node.right)})"

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return f"np.logical_not({self.visit(node.operand)})"
            return f"({_UNARY[type(node.op)]}{self.visit(node.operand)})"

        if isinstance(node, ast.BoolOp):
            if self.python_mode:
                self.fail(node, "'and'/'or' do not work on columns; use & and |")
            op = " & " if isinstance(node.op, ast.And) else " | "
            return "(" + op.join(self.visit(v) for v in node.values) + ")"

        if isinstance(node, ast.Compare):
            return self.compare(node)

        if isinstance(node, ast.Call):
            return self.call(node)

        self.fail(node, f"unsupported syntax ({type(node).__name__})")

    def compare(self, node: ast.Compare) -> str:
        if len(node.ops) > 1 and self.python_mode:
            self.fail(node, "chained comparisons are not supported")
        parts = []
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            lhs, rhs = self.visit(left), self.visit(right)
            if isinstance(op, ast.In) and not self.python_mode:
                parts.append(f"np.isin({lhs}, {rhs})")
            elif isinstance(op, ast.NotIn) and not self.python_mode:
                parts.append(f"(~np.isin({lhs}, {rhs}))")
            elif type(op) in _CMPOPS:
                parts.append(f"({lhs} {_CMPOPS[type(op)]} {rhs})")
            else:
                self.fail(node, "unsupported comparison")
            left = right
        return parts[0] if len(parts) == 1 else "(" + " & ".join(parts) + ")"

    def skim(self, node: ast.Subscript) -> str:
        """skim_od['DIST'] or skim_od[('SOVTOLL_TIME', 'AM')] -> s('skim_od', 'SOVTOLL_TIME__AM')"""
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            name = key.value
        elif (
            isinstance(key, ast.Tuple)
            and len(key.elts) == 2
            and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in key.elts)
        ):
            name = f"{key.elts[0].value}__{key.elts[1].value}"  # skim name and time period
        else:
            self.fail(node, "skim keys must be 'NAME' or ('NAME', 'PERIOD')")
        self.skims.add(name)
        return f"s({node.value.id!r}, {name!r})"

    def call(self, node: ast.Call) -> str:
        func = node.func
        # bare functions made available by the model, e.g. exp(...) -> np.exp(...)
        if isinstance(func, ast.Name) and func.id in self.functions:
            if node.keywords:
                self.fail(node, f"{func.id}() takes no keyword arguments")
            return f"{self.functions[func.id]}({', '.join(self.visit(a) for a in node.args)})"
        # np.<function>(...)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "np":
            if func.attr not in NP_FUNCTIONS:
                self.fail(node, f"np.{func.attr} is not in the supported list {sorted(NP_FUNCTIONS)}")
            args = [self.visit(a) for a in node.args]
            args += [f"{k.arg}={self.visit(k.value)}" for k in node.keywords]
            return f"np.{func.attr}({', '.join(args)})"
        # <column expression>.<method>(...)
        if isinstance(func, ast.Attribute):
            target = self.visit(func.value)
            method = func.attr
            kw = {k.arg: k.value for k in node.keywords}
            if method == "clip":
                names = ["lower", "upper"]
                vals = dict(zip(names, node.args))
                vals.update({k: v for k, v in kw.items() if k in names})
                if set(kw) - set(names) or len(node.args) > 2:
                    self.fail(node, "clip() only supports lower and upper")
                lower = self.visit(vals["lower"]) if "lower" in vals else "None"
                upper = self.visit(vals["upper"]) if "upper" in vals else "None"
                return f"_clip({target}, {lower}, {upper})"
            if method == "isin":
                if len(node.args) != 1 or kw:
                    self.fail(node, "isin() takes one list")
                return f"np.isin({target}, {self.visit(node.args[0])})"
            if method == "between":
                if len(node.args) != 2:
                    self.fail(node, "between() needs a lower and an upper bound")
                inclusive = "both"
                if "inclusive" in kw:
                    v = kw["inclusive"]
                    if not (isinstance(v, ast.Constant) and v.value in ("both", "neither", "left", "right")):
                        self.fail(node, "inclusive must be 'both', 'neither', 'left' or 'right'")
                    inclusive = v.value
                lo, hi = self.visit(node.args[0]), self.visit(node.args[1])
                lop = ">=" if inclusive in ("both", "left") else ">"
                hop = "<=" if inclusive in ("both", "right") else "<"
                return f"(({target} {lop} {lo}) & ({target} {hop} {hi}))"
            if method == "fillna":
                if len(node.args) != 1 or kw:
                    self.fail(node, "fillna() takes one value")
                return f"_fillna({target}, {self.visit(node.args[0])})"
            self.fail(node, f"method .{method}() is not supported")
        self.fail(node, "unsupported function call")


def translate(
    formula: str,
    constants: Mapping[str, object] | None = None,
    *,
    python_mode: bool | None = None,
    assigned: set[str] = frozenset(),
    functions: Mapping[str, str] | None = None,
    skim_accessors: set[str] = frozenset(),
) -> Translated:
    """
    Translate one spec formula to a NumPy expression over column arrays ``c``.

    ``python_mode`` defaults to "starts with @" (utility specs). Assignment
    specs are always Python style; for those, ``assigned`` names earlier
    variables (referenced as ``v[name]``), ``functions`` maps bare function
    names to NumPy (e.g. ``{'exp': 'np.exp'}``) and ``skim_accessors`` names
    the skim lookups available (referenced as ``s(accessor, skim_name)``).
    """
    text = formula.strip()
    if python_mode is None:
        python_mode = text.startswith("@")
    if text.startswith("@"):
        text = text[1:]
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as err:
        raise FormulaError(f"formula {formula!r} is not valid syntax: {err.msg}") from None
    t = _Translator(formula, constants or {}, python_mode, assigned, functions, skim_accessors)
    source = t.visit(tree)
    return Translated(formula=formula, source=source, columns=t.columns,
                      constants=t.used_constants, skims=t.skims, names=t.names)


# --- helpers used by generated code -------------------------------------------


def _clip(x, lower=None, upper=None):
    """pandas Series.clip semantics for arrays (NaN stays NaN)."""
    if lower is None and upper is None:
        return x
    return np.clip(x, lower, upper)


def _fillna(x, value):
    x = np.asarray(x)
    if x.dtype.kind != "f":
        return x
    return np.where(np.isnan(x), value, x)


RUNTIME_NAMESPACE = {"np": np, "_clip": _clip, "_fillna": _fillna}


def build_function(name: str, items: list[tuple[str, Translated]], doc: str = "") -> str:
    """
    Python source for a function ``name(c)`` that yields one array (or
    scalar) per formula, in order, so callers can use each term and discard it
    before the next is computed. ``items`` are (label, translated formula)
    pairs. The source is plain NumPy code meant to be read by people.
    """
    lines = [f"def {name}(c):"]
    if doc:
        lines.append(f'    """{doc}"""')
    for label, tr in items:
        lines.append(f"    # {label}: {tr.formula}")
        lines.append(f"    yield {tr.source}")
    if not items:
        lines.append("    yield from ()")
    return "\n".join(lines) + "\n"


def build_assignment_function(name: str, items: list[tuple[str, str, Translated]], doc: str = "") -> str:
    """
    Python source for ``name(c, s)``, which evaluates assignments in order and
    returns ``v``, a dict of every assigned variable (temporary ones, starting
    with ``_``, included). ``items`` are (target, description, translated
    formula). ``c`` maps column names to arrays; ``s(accessor, skim_name)``
    returns skim values.
    """
    lines = [f"def {name}(c, s):"]
    if doc:
        lines.append(f'    """{doc}"""')
    lines.append("    v = {}")
    for target, description, tr in items:
        lines.append(f"    # {description + ': ' if description else ''}{tr.formula}")
        if target == "_":
            lines.append(f"    {tr.source}")
        else:
            lines.append(f"    v[{target!r}] = {tr.source}")
    lines.append("    return v")
    return "\n".join(lines) + "\n"


def compile_function(source: str, name: str, filename: str = "<asim_core-generated>"):
    namespace = dict(RUNTIME_NAMESPACE)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]
