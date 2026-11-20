"""
asim_core's formula language: one language for every model, with stated rules.

Formulas are written in Python syntax, parsed with Python's ``ast`` module
(nothing is executed), checked, and compiled to NumPy code, the same kind
of code the translator of today's formulas produces (translate.py), so the
rest of asim_core runs them the same way.

**Names**

* ``household.income``, ``person.age``, ``tour.start``, ``home_zone.TOTEMP``:
  a column of one of the tables the model sees ("roles"; each model kind
  lists its roles). Unknown names get a "did you mean ...?".
* A bare name is one of the model's own values: an earlier derived value,
  a constant (constants.yaml and the model's), a coefficient (where the
  model allows it), or a parameter set by the step using the model.
* ``o`` and ``d``: the zones the model is evaluated between (origin and
  destination), set by the step.
* ``land_use.TOPOLOGY[d]``: a column of another table, looked up by ID.
* ``skim.SOV_TIME[o, d]``, ``skim.SOV_TIME[d, o]``,
  ``skim.SOV_TIME[o, d, out_period]``, ``skim.SOV_TIME[o, d, 'AM']``:
  skims from one zone to another, optionally at a time period (a derived
  value holding each row's period, or a fixed period in quotes).
* ``tt.previous_tour_ends(...)``: functions of objects the model provides
  (the timetable).

**Operators and functions**

``+ - * / // % **``, comparisons (chains such as ``18 <= person.age < 65``
too), ``x in [1, 2]``, ``x not in [1, 2]``, ``and``, ``or``, ``not``,
``a if CONDITION else b`` (the condition decided when the model loads, from
constants and parameters only; use ``where`` for row by row choices).

Functions: ``where(condition, a, b)``, ``min(a, b, ...)``, ``max(a, b, ...)``
(element by element), ``clip(x, lower, upper)``, ``between(x, lower, upper)``
(both ends included), ``log``, ``log1p``, ``log10``, ``log2``, ``exp``,
``sqrt``, ``abs``, ``floor``, ``ceil``, ``fillna(x, value)``,
``map(x, {...})``, ``cut(x, bins=[...], labels=[...])``, ``shift(x, rows)``,
``count_by(table.column, keys)`` (rows of a table with each key),
``int(x)``, ``float(x)``, ``bool(x)``, ``int8(x)`` ... ``float64(x)``,
``setting('name')`` (from settings.yaml), and the random draw
``random_lognormal(mean=..., sd=...)`` (or ``mu=..., sigma=...`` for the
underlying normal).

**Number rules: NumPy's.** Arithmetic on columns follows NumPy's type rules
(e.g. a 32-bit float column times a 64-bit integer column is computed in
64 bits), whatever the number of rows. Today's plain-style formulas follow
pandas.eval's rules instead, which can differ in the last bits (asim1
findings, #3).
"""

from __future__ import annotations

import ast
import difflib
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .translate import _NOT_CONSTANT, FormulaError, NeedsColumns, Translated, _literal


class NeedsParameter(NeedsColumns):
    """The formula uses a parameter that only the step using the model sets."""

ZONE_NAMES = ("o", "d")
UFUNCS = {"log", "log1p", "log10", "log2", "exp", "sqrt", "floor", "ceil"}
TYPES = {"int": "int", "float": "float", "bool": "bool"}
TYPES.update({t: repr(t) for t in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                                    "uint64", "float32", "float64")})
FUNCTIONS = (UFUNCS | set(TYPES) | {
    "where", "min", "max", "clip", "between", "abs", "fillna", "map", "cut", "shift", "count_by",
    "setting", "random_lognormal",
})
RANDOM_CALL = "random_lognormal("

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
           ast.Mod: "%", ast.Pow: "**"}
_CMPOPS = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}


@dataclass
class SkimScope:
    """
    How skims are looked up in this model. ``accessors`` maps a pattern
    (from, to, period) to the name of the lookup the step provides:
    from/to are 'o' or 'd'; period is None, 'fixed' (a period in quotes),
    or the name of a derived value holding each row's period
    ('out_period', 'in_period'). ``lookup`` names the lookup by any zones
    (``skim.NAME[a, b, 'P']`` with other zones than o and d), if the step has one.
    """

    accessors: dict[tuple[str, str, str | None], str] = field(default_factory=dict)
    lookup: str | None = None


@dataclass
class Scope:
    """Everything a formula can refer to, in one model and one step."""

    # role -> its columns (None: not known yet; then any column is accepted)
    roles: dict[str, set[str] | None] = field(default_factory=dict)
    # model values the step puts into the rows (c[name]): earlier derived values
    derived: set[str] = field(default_factory=set)
    # values assigned by earlier rows of the same list (v[name])
    assigned: set[str] = field(default_factory=set)
    constants: Mapping[str, object] = field(default_factory=dict)
    coefficients: set[str] = field(default_factory=set)
    parameters: Mapping[str, object] = field(default_factory=dict)
    # tables formulas can look up by ID: table -> columns (None: not known yet)
    tables: dict[str, set[str] | None] = field(default_factory=dict)
    skims: SkimScope | None = None
    objects: dict[str, set[str]] = field(default_factory=dict)
    # 'o' / 'd' -> the column holding those zones
    zones: dict[str, str] = field(default_factory=dict)
    settings: Mapping[str, object] = field(default_factory=dict)
    random_index: int | None = None


def _suggest(name: str, options) -> str:
    close = difflib.get_close_matches(name, sorted(options), n=3, cutoff=0.6)
    return f"; did you mean {' or '.join(close)}?" if close else ""


class _Compiler:
    def __init__(self, formula: str, scope: Scope):
        self.formula = formula
        self.scope = scope
        self.columns: set[str] = set()
        self.qualified: set[tuple[str, str]] = set()
        self.used_constants: set[str] = set()
        self.skims: set[str] = set()
        self.skim_uses: set[tuple[str, str, str]] = set()
        self.names: set[str] = set()
        self.coefficients: set[str] = set()
        self.tables: set[tuple[str, str]] = set()
        self.random_draws = False

    def fail(self, node, why: str):
        part = ast.unparse(node) if isinstance(node, ast.AST) else str(node)
        raise FormulaError(f"formula {self.formula!r}: {why} (in '{part}')")

    # --- values known when the model loads ------------------------------------------

    def fold(self, node):
        s = self.scope
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in s.assigned or node.id in s.derived or node.id in s.coefficients:
                return _NOT_CONSTANT
            if node.id in s.parameters:
                if s.parameters[node.id] is None:
                    raise NeedsParameter(f"formula {self.formula!r}: parameter '{node.id}' is set by "
                                         "the step using the model, and wasn't")
                self.used_constants.add(node.id)
                return s.parameters[node.id]
            if node.id in s.constants:
                self.used_constants.add(node.id)
                return s.constants[node.id]
            if node.id in ("True", "False"):
                return node.id == "True"
            return _NOT_CONSTANT
        if isinstance(node, (ast.List, ast.Tuple)):
            vals = [self.fold(e) for e in node.elts]
            return _NOT_CONSTANT if any(v is _NOT_CONSTANT for v in vals) else vals
        if isinstance(node, ast.Dict):
            keys = [self.fold(k) for k in node.keys]
            vals = [self.fold(v) for v in node.values]
            if any(x is _NOT_CONSTANT for x in keys + vals):
                return _NOT_CONSTANT
            return dict(zip(keys, vals))
        if isinstance(node, ast.UnaryOp):
            v = self.fold(node.operand)
            if v is _NOT_CONSTANT:
                return v
            if isinstance(node.op, ast.Not):
                return not v
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return +v
            return _NOT_CONSTANT
        if isinstance(node, ast.BoolOp):
            vals = [self.fold(v) for v in node.values]
            if any(v is _NOT_CONSTANT for v in vals):
                return _NOT_CONSTANT
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.Compare):
            vals = [self.fold(node.left)] + [self.fold(c) for c in node.comparators]
            if any(v is _NOT_CONSTANT for v in vals):
                return _NOT_CONSTANT
            ops = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
                   ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
                   ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
                   ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
            return all(ops[type(op)](a, b) for a, op, b in zip(vals, node.ops, vals[1:]))
        if isinstance(node, ast.IfExp):
            test = self.fold(node.test)
            if test is _NOT_CONSTANT:
                return test
            return self.fold(node.body if test else node.orelse)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setting":
            return self.setting(node)
        return _NOT_CONSTANT

    def setting(self, node: ast.Call):
        if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Constant):
            self.fail(node, "setting() takes one setting name in quotes")
        name = node.args[0].value
        if name not in self.scope.settings:
            self.fail(node, f"setting '{name}' not found in settings.yaml")
        return self.scope.settings[name]

    # --- compiling --------------------------------------------------------------------

    def visit(self, node) -> str:
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float, str)):
                return repr(node.value)
            self.fail(node, "unsupported literal")
        if isinstance(node, (ast.List, ast.Tuple)):
            return "[" + ", ".join(self.visit(e) for e in node.elts) + "]"
        if isinstance(node, ast.Name):
            return self.name(node)
        if isinstance(node, ast.Attribute):
            return self.attribute(node)
        if isinstance(node, ast.Subscript):
            return self.subscript(node)
        if isinstance(node, ast.IfExp):
            test = self.fold(node.test)
            if test is _NOT_CONSTANT:
                self.fail(node, "the condition of 'a if condition else b' must be known when the model "
                                "loads (constants and parameters); use where() for row-by-row choices")
            return self.visit(node.body if test else node.orelse)
        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                self.fail(node, "unsupported operator (use and, or, not for logic)")
            return f"({self.visit(node.left)} {op} {self.visit(node.right)})"
        if isinstance(node, ast.UnaryOp):
            value = self.fold(node)
            if value is not _NOT_CONSTANT:
                return _literal(value)
            if isinstance(node.op, ast.Not):
                return f"np.logical_not({self.visit(node.operand)})"
            if isinstance(node.op, ast.USub):
                return f"(-{self.visit(node.operand)})"
            if isinstance(node.op, ast.UAdd):
                return f"(+{self.visit(node.operand)})"
            self.fail(node, "unsupported operator (use not for logic)")
        if isinstance(node, ast.BoolOp):
            value = self.fold(node)
            if value is not _NOT_CONSTANT:
                return _literal(value)
            # constants decided now; the rest element by element
            parts = []
            for v in node.values:
                fv = self.fold(v)
                if fv is _NOT_CONSTANT:
                    parts.append(self.visit(v))
                elif isinstance(node.op, ast.And) and not fv:
                    return "False"
                elif isinstance(node.op, ast.Or) and fv:
                    return "True"
            op = " & " if isinstance(node.op, ast.And) else " | "
            return parts[0] if len(parts) == 1 else "(" + op.join(parts) + ")"
        if isinstance(node, ast.Compare):
            return self.compare(node)
        if isinstance(node, ast.Call):
            return self.call(node)
        self.fail(node, f"unsupported syntax ({type(node).__name__})")

    def name(self, node: ast.Name) -> str:
        s, n = self.scope, node.id
        if n in s.assigned:
            self.names.add(n)
            return f"v[{n!r}]"
        if n in s.derived:
            self.columns.add(n)
            return f"c[{n!r}]"
        if n in s.coefficients:
            self.coefficients.add(n)
            return f"k[{n!r}]"
        if n in ZONE_NAMES:
            if n not in s.zones:
                self.fail(node, f"'{n}' (the {'origin' if n == 'o' else 'destination'} zone) "
                                "is not available in this model")
            self.columns.add(s.zones[n])
            return f"c[{s.zones[n]!r}]"
        value = self.fold(node)
        if value is not _NOT_CONSTANT:
            return _literal(value)
        if n in ("inf", "nan"):
            return f"np.{n}"
        if n in s.roles:
            self.fail(node, f"'{n}' is a table; write {n}.<column>")
        known = set(s.assigned) | set(s.derived) | set(s.coefficients) | set(s.constants) | set(s.parameters)
        self.fail(node, f"unknown name '{n}' (columns are written table.column, e.g. "
                        f"{next(iter(s.roles), 'person')}.{n}){_suggest(n, known | set(s.roles))}")

    def role_column(self, node, role: str, column: str) -> str:
        cols = self.scope.roles[role]
        if cols is not None and column not in cols:
            self.fail(node, f"{role} has no column '{column}'{_suggest(column, cols)}")
        if column in self.scope.derived:
            self.fail(node, f"{role}.{column} has the same name as the derived value '{column}', "
                            "which takes its place in these rows; rename the derived value")
        self.columns.add(column)
        self.qualified.add((role, column))
        return f"c[{column!r}]"

    def attribute(self, node: ast.Attribute) -> str:
        s = self.scope
        if isinstance(node.value, ast.Name):
            base = node.value.id
            if base in s.roles:
                return self.role_column(node, base, node.attr)
            if base in s.tables:
                self.fail(node, f"{base} is looked up by ID: write {base}.{node.attr}[ids]")
            if base == "skim":
                self.fail(node, f"a skim needs zones: skim.{node.attr}[o, d]")
            options = set(s.roles) | set(s.tables) | ({"skim"} if s.skims else set())
            self.fail(node, f"unknown table '{base}' (tables here: {sorted(options)}){_suggest(base, options)}")
        self.fail(node, "unsupported attribute")

    def subscript(self, node: ast.Subscript) -> str:
        s = self.scope
        value = node.value
        # role['column name'] for names that aren't identifiers
        if isinstance(value, ast.Name) and value.id in s.roles:
            col = self.fold(node.slice)
            if not isinstance(col, str):
                self.fail(node, f"{value.id}[...] needs a column name in quotes")
            return self.role_column(node, value.id, col)
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            base, col = value.value.id, value.attr
            if base == "skim":
                return self.skim(node, col)
            if base in s.tables:
                cols = s.tables[base]
                if cols is not None and col not in cols:
                    self.fail(node, f"table '{base}' has no column '{col}'{_suggest(col, cols)}")
                self.tables.add((base, col))
                return f"t({base!r}, {col!r}, {self.visit(node.slice)})"
        self.fail(node, "unsupported subscript (table.column[ids] or skim.NAME[from, to(, period)])")

    def skim(self, node: ast.Subscript, name: str) -> str:
        if self.scope.skims is None:
            self.fail(node, "this model has no skims")
        keys = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(keys) not in (2, 3):
            self.fail(node, "skims take a from zone, a to zone and optionally a time period")
        a, b = keys[0], keys[1]
        period_node = keys[2] if len(keys) == 3 else None
        period, key = None, name
        if period_node is not None:
            fixed = self.fold(period_node)
            if isinstance(fixed, str):
                period, key = "fixed", f"{name}__{fixed}"
            elif isinstance(period_node, ast.Name) and period_node.id in self.scope.derived:
                period = period_node.id
            else:
                self.fail(node, "the time period must be a period in quotes or a derived value "
                                "holding each row's period (e.g. out_period)")
        acc = self.scope.skims.accessors
        za = a.id if isinstance(a, ast.Name) and a.id in ZONE_NAMES else None
        zb = b.id if isinstance(b, ast.Name) and b.id in ZONE_NAMES else None
        if za and zb and za != zb:
            pattern = (za, zb, None if period == "fixed" else period)
            if pattern in acc:
                self.skims.add(key)
                self.skim_uses.add((acc[pattern], key, "get"))
                return f"s({acc[pattern]!r}, {key!r})"
            reverse = (zb, za, None)
            if period in (None, "fixed") and reverse in acc:  # the other direction's lookup
                self.skims.add(key)
                self.skim_uses.add((acc[reverse], key, "reverse"))
                return f"s({acc[reverse]!r}, {key!r}, 'reverse')"
        if period not in (None, "fixed"):
            self.fail(node, f"skims at '{period}' are only available from o to d or d to o")
        lookup = self.scope.skims.lookup
        if lookup is None:
            self.fail(node, "this model only looks skims up between o and d")
        self.skims.add(key)
        self.skim_uses.add((lookup, key, "lookup"))
        return f"s.lookup({key!r}, {self.visit(a)}, {self.visit(b)})"

    def compare(self, node: ast.Compare) -> str:
        value = self.fold(node)
        if value is not _NOT_CONSTANT:
            return _literal(value)
        parts, left = [], node.left
        for op, right in zip(node.ops, node.comparators):
            lhs, rhs = self.visit(left), self.visit(right)
            if isinstance(op, ast.In):
                parts.append(f"np.isin({lhs}, {rhs})")
            elif isinstance(op, ast.NotIn):
                parts.append(f"(~np.isin({lhs}, {rhs}))")
            elif type(op) in _CMPOPS:
                parts.append(f"({lhs} {_CMPOPS[type(op)]} {rhs})")
            else:
                self.fail(node, "unsupported comparison")
            left = right
        return parts[0] if len(parts) == 1 else "(" + " & ".join(parts) + ")"

    def args(self, node: ast.Call, names: list[str], required: int) -> dict[str, ast.AST]:
        """Positional and keyword arguments by name, checked."""
        f = node.func.id
        if len(node.args) > len(names):
            self.fail(node, f"{f}() takes at most {len(names)} arguments ({', '.join(names)})")
        out = dict(zip(names, node.args))
        for k in node.keywords:
            if k.arg not in names or k.arg in out:
                self.fail(node, f"{f}() has no argument '{k.arg}' here (arguments: {', '.join(names)})")
            out[k.arg] = k.value
        missing = [n for n in names[:required] if n not in out]
        if missing:
            self.fail(node, f"{f}() needs {', '.join(missing)}")
        return out

    def call(self, node: ast.Call) -> str:
        s = self.scope
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in s.objects:
            obj, method = func.value.id, func.attr
            if method not in s.objects[obj]:
                self.fail(node, f"{obj}.{method}() is not available{_suggest(method, s.objects[obj])}")
            if node.keywords:
                self.fail(node, f"{obj}.{method}() takes no keyword arguments")
            return f"{obj}.{method}({', '.join(self.visit(a) for a in node.args)})"
        if not isinstance(func, ast.Name):
            self.fail(node, "only the listed functions can be called (no methods)")
        f = func.id
        if f not in FUNCTIONS:
            self.fail(node, f"unknown function '{f}'{_suggest(f, FUNCTIONS)}")
        if f == "setting":
            return _literal(self.setting(node))
        if f in UFUNCS or f == "abs":
            a = self.args(node, ["x"], 1)
            return f"np.{'abs' if f == 'abs' else f}({self.visit(a['x'])})"
        if f in TYPES:
            a = self.args(node, ["x"], 1)
            return f"_astype({self.visit(a['x'])}, {TYPES[f]})"
        if f == "where":
            a = self.args(node, ["condition", "a", "b"], 3)
            return f"np.where({self.visit(a['condition'])}, {self.visit(a['a'])}, {self.visit(a['b'])})"
        if f in ("min", "max"):
            if node.keywords or len(node.args) < 2:
                self.fail(node, f"{f}() takes two or more values")
            ufunc = "np.minimum" if f == "min" else "np.maximum"
            out = self.visit(node.args[0])
            for arg in node.args[1:]:
                out = f"{ufunc}({out}, {self.visit(arg)})"
            return out
        if f == "clip":
            a = self.args(node, ["x", "lower", "upper"], 1)
            lower = self.visit(a["lower"]) if "lower" in a else "None"
            upper = self.visit(a["upper"]) if "upper" in a else "None"
            return f"_clip({self.visit(a['x'])}, {lower}, {upper})"
        if f == "between":
            a = self.args(node, ["x", "lower", "upper"], 3)
            x = self.visit(a["x"])
            return f"(({x} >= {self.visit(a['lower'])}) & ({x} <= {self.visit(a['upper'])}))"
        if f == "fillna":
            a = self.args(node, ["x", "value"], 2)
            return f"_fillna({self.visit(a['x'])}, {self.visit(a['value'])})"
        if f == "map":
            a = self.args(node, ["x", "mapping"], 2)
            mapping = self.fold(a["mapping"])
            if not isinstance(mapping, dict):
                self.fail(node, "map() needs a dictionary of constants")
            return f"_map({self.visit(a['x'])}, {_literal(mapping)})"
        if f == "cut":
            a = self.args(node, ["x", "bins", "labels"], 3)
            return f"_cut({self.visit(a['x'])}, {self.visit(a['bins'])}, {self.visit(a['labels'])})"
        if f == "shift":
            a = self.args(node, ["x", "rows"], 2)
            rows = self.fold(a["rows"])
            if not isinstance(rows, int):
                self.fail(node, "shift() takes a whole number of rows")
            return f"_shift({self.visit(a['x'])}, {rows})"
        if f == "count_by":
            a = self.args(node, ["column", "keys"], 2)
            col = a["column"]
            if not (isinstance(col, ast.Attribute) and isinstance(col.value, ast.Name)
                    and col.value.id in s.tables):
                self.fail(node, f"count_by() takes a table's column (tables: {sorted(s.tables)}) and keys")
            table, column = col.value.id, col.attr
            cols = s.tables[table]
            if cols is not None and column not in cols:
                self.fail(node, f"table '{table}' has no column '{column}'{_suggest(column, cols)}")
            self.tables.add((table, column))
            return f"t.count({table!r}, {column!r}, {self.visit(a['keys'])})"
        if f == "random_lognormal":
            if s.random_index is None:
                self.fail(node, "random draws are only available in derived values")
            if node.args:
                self.fail(node, "random_lognormal() takes mean= and sd=, or mu= and sigma=")
            kw = {k.arg: k.value for k in node.keywords}
            if set(kw) == {"mean", "sd"}:
                mu, sigma, scale = kw["mean"], kw["sd"], True
            elif set(kw) == {"mu", "sigma"}:
                mu, sigma, scale = kw["mu"], kw["sigma"], False
            else:
                self.fail(node, "random_lognormal() takes mean= and sd= (of the lognormal values), "
                                "or mu= and sigma= (of the underlying normal)")
            self.random_draws = True
            return (f"_lognormal(r[:, {s.random_index}], {self.visit(mu)}, {self.visit(sigma)}, "
                    f"scale={scale})")
        self.fail(node, f"function '{f}' is not supported here")


def parse(formula: str) -> ast.AST:
    try:
        return ast.parse(formula.strip(), mode="eval")
    except SyntaxError as err:
        raise FormulaError(f"formula {formula!r} is not valid syntax: {err.msg}") from None


def compile_formula(formula: str, scope: Scope) -> Translated:
    """Compile one formula of asim_core's language to a NumPy expression (see the module notes)."""
    c = _Compiler(formula, scope)
    source = c.visit(parse(formula))
    return Translated(formula=formula, source=source, columns=c.columns, constants=c.used_constants,
                      skims=c.skims, names=c.names, skim_uses=c.skim_uses, coefficients=c.coefficients,
                      tables=c.tables, random_draws=c.random_draws, qualified=c.qualified)


def fold_formula(formula: str, scope: Scope):
    """The formula's value if it is known when the model loads, else raises FormulaError."""
    c = _Compiler(formula, scope)
    value = c.fold(parse(formula).body)
    if value is _NOT_CONSTANT:
        raise FormulaError(f"formula {formula!r} is not known when the model loads")
    return value, c.used_constants


def split_assignment(text: str) -> tuple[str | None, str]:
    """'name = formula' -> ('name', 'formula'); a plain formula -> (None, formula)."""
    try:
        tree = ast.parse(text.strip(), mode="exec")
    except SyntaxError:
        return None, text
    if (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign) and len(tree.body[0].targets) == 1
            and isinstance(tree.body[0].targets[0], ast.Name)):
        node = tree.body[0]
        lines = text.strip().splitlines()
        start = sum(len(x) + 1 for x in lines[:node.value.lineno - 1]) + node.value.col_offset
        return node.targets[0].id, text.strip()[start:].strip()
    return None, text


__all__ = ["Scope", "SkimScope", "compile_formula", "fold_formula", "split_assignment", "RANDOM_CALL",
           "FUNCTIONS", "np"]
