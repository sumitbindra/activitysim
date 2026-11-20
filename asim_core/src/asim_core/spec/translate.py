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

**Decided when the model loads.** Some formulas test what data exists, e.g.
``df.number_of_participants if 'tour_type' in df.columns else 1``. Given the
columns that will be present (``df_columns``, ``tables``), such tests are
worked out during translation and only the branch that applies is written
into the code. So the generated code is simpler, and the columns a step reads
are known exactly before it runs. A test that can't be worked out from
constants and column names is reported as unsupported.

Other names the generated code can use, supplied by the component:

* ``k['name']``: a coefficient used inside a formula (its value can differ by
  segment, e.g. tour purpose);
* ``t('land_use', 'TOPOLOGY', keys)``: a column of another table looked up by
  ID (1.x's ``reindex(land_use.TOPOLOGY, keys)``);
* ``r[:, i]``: the step's i-th set of random draws (``rng.lognormal_for_df``).
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class FormulaError(ValueError):
    pass


INDEX = "@index"  # the key under which generated code finds the rows' IDs (1.x: df.index)


class NeedsColumns(FormulaError):
    """The formula tests or reads columns that are only known once the run's tables are."""


class NotConstant(FormulaError):
    """The formula's value can't be worked out when the model loads."""


# numpy functions a formula may call as np.<name>(...)
NP_FUNCTIONS = {
    "where", "minimum", "maximum", "clip", "isin",
    "log", "log1p", "log10", "log2", "exp", "sqrt", "abs", "floor", "ceil",
}
NP_VALUES = {"inf", "nan"}  # np.inf, np.nan
ASTYPE_NAMES = {"int", "float", "bool"}  # .astype(int) etc.
ASTYPE_STRINGS = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                  "float32", "float64", "bool"}

_BINOPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
}
_CMPOPS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
}
_UNARY = {ast.USub: "-", ast.UAdd: "+", ast.Invert: "~"}
_NOT_CONSTANT = object()


@dataclass
class Translated:
    formula: str
    source: str
    columns: set[str] = field(default_factory=set)
    constants: set[str] = field(default_factory=set)
    skims: set[str] = field(default_factory=set)  # skim names, e.g. 'SOVTOLL_TIME__AM'
    names: set[str] = field(default_factory=set)  # earlier assigned variables used
    # (accessor, skim key, how): how is 'get', 'reverse' or 'max'
    skim_uses: set[tuple[str, str, str]] = field(default_factory=set)
    coefficients: set[str] = field(default_factory=set)  # used as k['name']
    tables: set[tuple[str, str]] = field(default_factory=set)  # (table, column) looked up by ID
    random_draws: bool = False
    # asim_core's language: (table role, column) for each column read, e.g. ('person', 'age')
    qualified: set[tuple[str, str]] = field(default_factory=set)


def _literal(value) -> str:
    if isinstance(value, (bool, int, float, str)) or value is None:
        if isinstance(value, float) and not np.isfinite(value):
            return "np.nan" if np.isnan(value) else ("np.inf" if value > 0 else "(-np.inf)")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items()) + "}"
    raise FormulaError(f"constant value {value!r} is not a number, string, list or dict")


class _Translator:
    def __init__(
        self,
        formula: str,
        constants: Mapping[str, object],
        python_mode: bool,
        assigned: set[str] = frozenset(),
        functions: Mapping[str, str] | None = None,
        skim_accessors: set[str] = frozenset(),
        coefficients: set[str] = frozenset(),
        df_columns: set[str] | None = None,
        tables: Mapping[str, set[str]] | None = None,
        settings: Mapping[str, object] | None = None,
        random_index: int | None = None,
        pandas_locals: bool = False,
        df_names: set[str] = frozenset({"df"}),
        objects: Mapping[str, set[str]] | None = None,
    ):
        self.formula = formula
        self.constants = constants
        self.python_mode = python_mode
        self.assigned = assigned
        self.functions = functions or {}
        self.skim_accessors = skim_accessors
        self.coefficient_names = coefficients
        self.df_columns = df_columns
        self.table_columns = dict(tables or {})
        self.settings = settings or {}
        self.random_index = random_index
        self.pandas_locals = pandas_locals
        self.df_names = set(df_names) | {"df"}
        self.objects = dict(objects or {})
        self.columns: set[str] = set()
        self.used_constants: set[str] = set()
        self.skims: set[str] = set()
        self.skim_uses: set[tuple[str, str, str]] = set()
        self.names: set[str] = set()
        self.coefficients: set[str] = set()
        self.tables: set[tuple[str, str]] = set()
        self.random_draws = False
        self._lookup: tuple[str, str] | None = None  # (table, keys) inside reindex()

    def fail(self, node: ast.AST, why: str):
        part = ast.unparse(node) if isinstance(node, ast.AST) else str(node)
        raise FormulaError(f"formula {self.formula!r}: {why} (in '{part}')")

    def column(self, name: str) -> str:
        self.columns.add(name)
        return f"c[{name!r}]"

    # --- values known when the model loads ------------------------------------

    def fold(self, node: ast.AST):
        """The node's value if it can be worked out now, else _NOT_CONSTANT."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and self.python_mode:
            if node.id in self.assigned or node.id in self.coefficient_names:
                return _NOT_CONSTANT
            if node.id in self.constants:
                self.used_constants.add(node.id)
                return self.constants[node.id]
            return _NOT_CONSTANT
        if isinstance(node, (ast.List, ast.Tuple)):
            vals = [self.fold(e) for e in node.elts]
            return _NOT_CONSTANT if any(v is _NOT_CONSTANT for v in vals) else vals
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
            if isinstance(node.op, ast.And):
                out = True
                for v in vals:
                    out = out and v
                return out
            out = False
            for v in vals:
                out = out or v
            return out
        if isinstance(node, ast.Compare):
            member = self.membership(node)
            if member is not None:
                return member
            vals = [self.fold(node.left)] + [self.fold(c) for c in node.comparators]
            if any(v is _NOT_CONSTANT for v in vals):
                return _NOT_CONSTANT
            ops = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
                   ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
                   ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
                   ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
            for a, op, b in zip(vals, node.ops, vals[1:]):
                if type(op) not in ops or not ops[type(op)](a, b):
                    return False if type(op) in ops else _NOT_CONSTANT
            return True
        if isinstance(node, ast.IfExp):
            test = self.fold(node.test)
            if test is _NOT_CONSTANT:
                return test
            return self.fold(node.body if test else node.orelse)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "setting":
                return self.setting(node)
        return _NOT_CONSTANT

    def membership(self, node: ast.Compare):
        """``'x' in df.columns``, ``'x' in df``, ``'x' in land_use``: True/False, or None."""
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.In, ast.NotIn)):
            return None
        container = node.comparators[0]
        if isinstance(container, ast.Attribute) and container.attr == "columns":
            container = container.value
        if not isinstance(container, ast.Name):
            return None
        if container.id in self.df_names:
            if self.df_columns is None:
                raise NeedsColumns(f"formula {self.formula!r}: testing which columns exist "
                                   f"needs the chooser's columns")
            columns = self.df_columns
        elif container.id in self.table_columns:
            columns = self.table_columns[container.id]
            if columns is None:
                raise NeedsColumns(f"formula {self.formula!r}: needs the columns of "
                                   f"table '{container.id}'")
        else:
            return None
        name = self.fold(node.left)
        if not isinstance(name, str):
            self.fail(node, "column test needs a column name in quotes")
        found = name in columns
        return found if isinstance(node.ops[0], ast.In) else not found

    def setting(self, node: ast.Call):
        if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Constant):
            self.fail(node, "setting() takes one setting name in quotes")
        name = node.args[0].value
        if name not in self.settings:
            self.fail(node, f"setting '{name}' not found in settings.yaml")
        return self.settings[name]

    # --- translation ------------------------------------------------------------

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
                if self.pandas_locals:  # names the model defines come before columns
                    if node.id in self.assigned:
                        self.names.add(node.id)
                        return f"v[{node.id!r}]"
                    if node.id in self.coefficient_names:
                        self.coefficients.add(node.id)
                        return f"k[{node.id!r}]"
                    if node.id in self.constants:
                        self.used_constants.add(node.id)
                        return _literal(self.constants[node.id])
                return self.column(node.id)  # pandas style: bare names are columns
            if node.id in self.assigned:  # a variable assigned by an earlier row
                self.names.add(node.id)
                return f"v[{node.id!r}]"
            if node.id in self.coefficient_names:
                self.coefficients.add(node.id)
                return f"k[{node.id!r}]"
            if node.id in self.constants:
                self.used_constants.add(node.id)
                return _literal(self.constants[node.id])
            self.fail(node, f"unknown name '{node.id}' (not a constant or an earlier "
                            f"variable; columns are written df.name)")

        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in self.df_names:
                if node.attr == "index":  # the table's IDs
                    return self.column(INDEX)
                return self.column(node.attr)
            if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr in NP_VALUES:
                return f"np.{node.attr}"
            if isinstance(node.value, ast.Name) and node.value.id in self.table_columns:
                return self.table_column(node, node.value.id, node.attr)
            self.fail(node, "unsupported attribute")

        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in self.df_names:
                name = self.fold(node.slice)
                if isinstance(name, str):
                    return self.column(name)
                self.fail(node, "df[...] needs a column name")
            if isinstance(node.value, ast.Name) and node.value.id in self.skim_accessors:
                return self.skim(node)
            if isinstance(node.value, ast.Name) and node.value.id in self.table_columns:
                name = self.fold(node.slice)
                if isinstance(name, str):
                    return self.table_column(node, node.value.id, name)
            self.fail(node, "unsupported subscript (only df['column'] and skim lookups are allowed)")

        if isinstance(node, ast.IfExp):
            test = self.fold(node.test)
            if test is _NOT_CONSTANT:
                self.fail(node, "the condition of 'x if condition else y' must be decided "
                                "when the model loads (constants and column names only)")
            return self.visit(node.body if test else node.orelse)

        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                self.fail(node, "unsupported operator")
            return f"({self.visit(node.left)} {op} {self.visit(node.right)})"

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                value = self.fold(node)
                if value is not _NOT_CONSTANT:
                    return repr(value)
                return f"np.logical_not({self.visit(node.operand)})"
            return f"({_UNARY[type(node.op)]}{self.visit(node.operand)})"

        if isinstance(node, ast.BoolOp):
            if self.python_mode:
                value = self.fold(node)
                if value is not _NOT_CONSTANT:
                    return _literal(value)
                self.fail(node, "'and'/'or' do not work on columns; use & and |")
            op = " & " if isinstance(node.op, ast.And) else " | "
            return "(" + op.join(self.visit(v) for v in node.values) + ")"

        if isinstance(node, ast.Compare):
            return self.compare(node)

        if isinstance(node, ast.Call):
            return self.call(node)

        self.fail(node, f"unsupported syntax ({type(node).__name__})")

    def compare(self, node: ast.Compare) -> str:
        member = self.membership(node)
        if member is not None:
            return repr(member)
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
        self.skim_uses.add((node.value.id, name, "get"))
        return f"s({node.value.id!r}, {name!r})"

    def skim_method(self, node: ast.Call, accessor: str, how: str) -> str:
        """od_skims.reverse('DIST') (destination to origin) and od_skims.max('DIST') (larger of both)."""
        if len(node.args) != 1 or node.keywords or not (
            isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
        ):
            self.fail(node, f".{how}() takes one skim name in quotes")
        name = node.args[0].value
        self.skims.add(name)
        self.skim_uses.add((accessor, name, how))
        return f"s({accessor!r}, {name!r}, {how!r})"

    def call(self, node: ast.Call) -> str:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "setting":
            return _literal(self.setting(node))
        if isinstance(func, ast.Name) and func.id == "reindex":
            return self.reindex(node)
        if isinstance(func, ast.Name) and func.id == "reindex_i":
            return self.reindex_i(node)
        # methods of objects the model provides, e.g. tt.previous_tour_ends(...)
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id in self.objects):
            if func.attr not in self.objects[func.value.id]:
                self.fail(node, f"{func.value.id}.{func.attr}() is not supported "
                                f"(supported: {sorted(self.objects[func.value.id])})")
            if node.keywords:
                self.fail(node, f"{func.value.id}.{func.attr}() takes no keyword arguments")
            return f"{func.value.id}.{func.attr}({', '.join(self.visit(a) for a in node.args)})"
        # skim_dict.lookup(origins, destinations, ('NAME', 'PERIOD'))
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id in self.skim_accessors and func.attr == "lookup"):
            return self.skim_lookup(node, func.value.id)
        # Python's abs() on a column is NumPy's absolute value (pandas Series.__abs__)
        if isinstance(func, ast.Name) and func.id == "abs" and func.id not in self.functions:
            if len(node.args) != 1 or node.keywords:
                self.fail(node, "abs() takes one value")
            return f"np.abs({self.visit(node.args[0])})"
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
        # pd.cut(x, bins=[...], labels=[...])
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "pd":
            if func.attr == "cut":
                return self.cut(node)
            self.fail(node, f"pd.{func.attr} is not supported (only pd.cut)")
        # rng.lognormal_for_df(df, mu=..., sigma=..., broadcast=True, scale=...)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "rng":
            return self.lognormal(node)
        # skim_accessor.reverse('NAME') / .max('NAME')
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id in self.skim_accessors and func.attr in ("reverse", "max")):
            return self.skim_method(node, func.value.id, func.attr)
        # T.column.groupby(T.key).sum().reindex(ids): a sum over another table's rows per ID
        if isinstance(func, ast.Attribute) and func.attr == "reindex":
            grouped = self.group_sum(func.value)
            if grouped is not None:
                if len(node.args) != 1 or node.keywords:
                    self.fail(node, "reindex() takes the IDs to look up")
                table, col, key = grouped
                return f"t.sum_by({table!r}, {col!r}, {key!r}, {self.visit(node.args[0])})"
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
            if method == "apply":
                # Series.apply(np.<ufunc>) applies the ufunc to the whole column
                f = node.args[0] if len(node.args) == 1 and not kw else None
                if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                        and f.value.id == "np" and f.attr in NP_FUNCTIONS):
                    self.fail(node, f"apply() supports np.<function> from {sorted(NP_FUNCTIONS)}")
                return f"np.{f.attr}({target})"
            if method == "shift":
                periods = self.fold(node.args[0]) if node.args else kw.get("periods") and self.fold(kw["periods"])
                if not isinstance(periods, int) or (node.args and kw) or len(node.args) > 1:
                    self.fail(node, "shift() takes a whole number of rows")
                return f"_shift({target}, {periods})"
            if method == "map":
                if len(node.args) != 1 or kw:
                    self.fail(node, "map() takes one dictionary")
                return f"_map({target}, {_literal(self.mapping(node.args[0]))})"
            if method == "astype":
                if len(node.args) != 1 or kw:
                    self.fail(node, "astype() takes one type")
                a = node.args[0]
                if isinstance(a, ast.Name) and a.id in ASTYPE_NAMES:
                    return f"_astype({target}, {a.id})"
                if isinstance(a, ast.Constant) and a.value in ASTYPE_STRINGS:
                    return f"_astype({target}, {a.value!r})"
                if (isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name)
                        and a.value.id == "np" and a.attr in ASTYPE_STRINGS):
                    return f"_astype({target}, np.{a.attr})"
                self.fail(node, f"astype() supports {sorted(ASTYPE_NAMES)} and {sorted(ASTYPE_STRINGS)}")
            self.fail(node, f"method .{method}() is not supported")
        self.fail(node, "unsupported function call")

    def group_sum(self, node) -> tuple[str, str, str] | None:
        """T.column.groupby(T.key).sum() -> (T, column, key), else None."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sum" and not node.args and not node.keywords):
            return None
        g = node.func.value
        if not (isinstance(g, ast.Call) and isinstance(g.func, ast.Attribute) and g.func.attr == "groupby"
                and len(g.args) == 1 and not g.keywords):
            return None
        col, key = g.func.value, g.args[0]
        if not (isinstance(col, ast.Attribute) and isinstance(col.value, ast.Name)
                and isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name)
                and col.value.id == key.value.id and col.value.id in self.table_columns):
            return None
        table = col.value.id
        cols = self.table_columns[table]
        if cols is None:
            raise NeedsColumns(f"formula {self.formula!r}: needs the columns of table '{table}'")
        for c in (col.attr, key.attr):
            if c not in cols:
                self.fail(node, f"table '{table}' has no column '{c}'")
            self.tables.add((table, c))
        return table, col.attr, key.attr

    def reindex(self, node: ast.Call) -> str:
        """
        reindex(land_use.TOPOLOGY, df.destination) -> t('land_use', 'TOPOLOGY', c['destination']).
        The first argument can also be element-by-element arithmetic on one
        table's columns, e.g. reindex(land_use.LANDAREA * 640, ...): the
        columns are looked up first, then the arithmetic is done, which gives
        the same values.
        """
        if len(node.args) != 2 or node.keywords:
            self.fail(node, "reindex() takes a table column and the IDs to look up")
        if self._lookup is not None:
            self.fail(node, "reindex() inside reindex() is not supported")
        tables = {n.id for n in ast.walk(node.args[0])
                  if isinstance(n, ast.Name) and n.id in self.table_columns}
        if len(tables) != 1:
            known = sorted(self.table_columns) or "none"
            self.fail(node, f"reindex() needs columns of one table the model lists (tables: {known})")
        table = tables.pop()
        if self.table_columns[table] is None:
            raise NeedsColumns(f"formula {self.formula!r}: needs the columns of table '{table}'")
        self._lookup = (table, self.visit(node.args[1]))
        try:
            return self.visit(node.args[0])
        finally:
            self._lookup = None

    def reindex_i(self, node: ast.Call) -> str:
        """
        reindex_i(joint_tour_participants.groupby('person_id').size(), df.person_id):
        how many rows of a table each ID has (0 if none), as 8-bit integers,
        like 1.x's reindex_i. Also reindex_i(table.column, ids), with 0 for
        missing IDs.
        """
        if len(node.args) != 2 or node.keywords:
            self.fail(node, "reindex_i() takes a table column (or group size) and the IDs to look up")
        arg = node.args[0]
        keys = self.visit(node.args[1])
        # T.groupby('col').size()
        if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr == "size"
                and not arg.args and isinstance(arg.func.value, ast.Call)):
            g = arg.func.value
            if (isinstance(g.func, ast.Attribute) and g.func.attr == "groupby"
                    and isinstance(g.func.value, ast.Name) and g.func.value.id in self.table_columns
                    and len(g.args) == 1 and isinstance(self.fold(g.args[0]), str)):
                table, col = g.func.value.id, self.fold(g.args[0])
                if self.table_columns[table] is None:
                    raise NeedsColumns(f"formula {self.formula!r}: needs the columns of table '{table}'")
                if col not in self.table_columns[table]:
                    self.fail(node, f"table '{table}' has no column '{col}'")
                self.tables.add((table, col))
                return f"_as_int8(t.count({table!r}, {col!r}, {keys}))"
        if (isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                and arg.value.id in self.table_columns):
            table, col = arg.value.id, arg.attr
            if self.table_columns[table] is None:
                raise NeedsColumns(f"formula {self.formula!r}: needs the columns of table '{table}'")
            if col not in self.table_columns[table]:
                self.fail(node, f"table '{table}' has no column '{col}'")
            self.tables.add((table, col))
            return f"_as_int8(t.fill({table!r}, {col!r}, {keys}, 0))"
        self.fail(node, "reindex_i() supports table.column or table.groupby('column').size()")

    def skim_lookup(self, node: ast.Call, accessor: str) -> str:
        """skim_dict.lookup(o, d, ('SOV_DIST', 'MD')) -> s.lookup('SOV_DIST__MD', o, d)"""
        if len(node.args) != 3 or node.keywords:
            self.fail(node, "lookup() takes origins, destinations and a skim name")
        key = self.fold(node.args[2])
        if isinstance(key, (list, tuple)) and len(key) == 2 and all(isinstance(x, str) for x in key):
            name = f"{key[0]}__{key[1]}"
        elif isinstance(key, str):
            name = key
        else:
            self.fail(node, "skim name must be 'NAME' or ('NAME', 'PERIOD')")
        self.skims.add(name)
        self.skim_uses.add((accessor, name, "lookup"))
        return f"s.lookup({name!r}, {self.visit(node.args[0])}, {self.visit(node.args[1])})"

    def table_column(self, node: ast.AST, table: str, name: str) -> str:
        """A table's column inside reindex(): looked up for the reindex keys."""
        lookup = self._lookup
        if lookup is None or lookup[0] != table:
            self.fail(node, f"columns of '{table}' can only be used inside reindex()")
        if name not in self.table_columns[table]:
            self.fail(node, f"table '{table}' has no column '{name}'")
        self.tables.add((table, name))
        return f"t({table!r}, {name!r}, {lookup[1]})"

    def mapping(self, node: ast.AST) -> dict:
        """A dictionary for .map(): a constant, a literal, or {k: v for k, v in CONSTANT.items()}."""
        if isinstance(node, ast.DictComp) and len(node.generators) == 1:
            gen = node.generators[0]
            it = gen.iter
            if (isinstance(gen.target, ast.Tuple) and len(gen.target.elts) == 2
                    and all(isinstance(e, ast.Name) for e in gen.target.elts)
                    and isinstance(node.key, ast.Name) and isinstance(node.value, ast.Name)
                    and node.key.id == gen.target.elts[0].id and node.value.id == gen.target.elts[1].id
                    and not gen.ifs and isinstance(it, ast.Call) and not it.args
                    and isinstance(it.func, ast.Attribute) and it.func.attr == "items"):
                node = it.func.value  # {k: v for k, v in X.items()} is X
            else:
                self.fail(node, "only {k: v for k, v in NAME.items()} is supported")
        if isinstance(node, ast.Dict):
            keys = [self.fold(k) for k in node.keys]
            vals = [self.fold(v) for v in node.values]
            if any(x is _NOT_CONSTANT for x in keys + vals):
                self.fail(node, "map() dictionary must hold constants")
            return dict(zip(keys, vals))
        value = self.fold(node)
        if not isinstance(value, dict):
            self.fail(node, "map() needs a dictionary constant")
        return value

    def cut(self, node: ast.Call) -> str:
        kw = {k.arg: k.value for k in node.keywords}
        if len(node.args) != 1 or set(kw) != {"bins", "labels"}:
            self.fail(node, "pd.cut() is supported as pd.cut(x, bins=[...], labels=[...])")
        return f"_cut({self.visit(node.args[0])}, {self.visit(kw['bins'])}, {self.visit(kw['labels'])})"

    def lognormal(self, node: ast.Call) -> str:
        func = node.func
        if func.attr != "lognormal_for_df":
            self.fail(node, f"rng.{func.attr}() is not supported (only rng.lognormal_for_df)")
        if self.random_index is None:
            self.fail(node, "random draws are only available in assignment specs")
        if not (len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == "df"):
            self.fail(node, "lognormal_for_df() must draw for df")
        kw = {k.arg: k.value for k in node.keywords}
        if not {"mu", "sigma"} <= set(kw) or set(kw) - {"mu", "sigma", "broadcast", "scale"}:
            self.fail(node, "lognormal_for_df() takes mu, sigma, broadcast and scale")
        broadcast = self.fold(kw["broadcast"]) if "broadcast" in kw else False
        if broadcast is not True:
            # 1.x draws all of a spec's lognormals in one call, which needs broadcast=True
            self.fail(node, "lognormal_for_df() needs broadcast=True")
        scale = self.fold(kw["scale"]) if "scale" in kw else False
        if not isinstance(scale, bool):
            self.fail(node, "scale must be True or False")
        self.random_draws = True
        return (f"_lognormal(r[:, {self.random_index}], {self.visit(kw['mu'])}, "
                f"{self.visit(kw['sigma'])}, scale={scale})")


def translate(
    formula: str,
    constants: Mapping[str, object] | None = None,
    *,
    python_mode: bool | None = None,
    assigned: set[str] = frozenset(),
    functions: Mapping[str, str] | None = None,
    skim_accessors: set[str] = frozenset(),
    coefficients: set[str] = frozenset(),
    df_columns: set[str] | None = None,
    tables: Mapping[str, set[str]] | None = None,
    settings: Mapping[str, object] | None = None,
    random_index: int | None = None,
    pandas_locals: bool = False,
    df_names: set[str] = frozenset({"df"}),
    objects: Mapping[str, set[str]] | None = None,
) -> Translated:
    """
    Translate one spec formula to a NumPy expression over column arrays ``c``.

    ``python_mode`` defaults to "starts with @" (utility specs). Assignment
    specs are always Python style; for those, ``assigned`` names earlier
    variables (referenced as ``v[name]``), ``functions`` maps bare function
    names to NumPy (e.g. ``{'exp': 'np.exp'}``) and ``skim_accessors`` names
    the skim lookups available (referenced as ``s(accessor, skim_name)``).

    ``coefficients`` are names whose values are supplied at run time
    (``k[name]``); they take precedence over constants of the same name.
    ``df_columns`` and ``tables`` (table name -> its columns) let formulas
    test which columns exist and look up other tables' columns; ``settings``
    answers ``setting('name')``; ``random_index`` is the position of this
    formula's random draws, for ``rng.lognormal_for_df``.

    Plain-style (pandas) formulas: with ``pandas_locals``, bare names the
    model defines (earlier temporaries, coefficients, constants) come before
    columns, as in 1.x's interaction models. A plain-style formula that does
    arithmetic is evaluated with ``pandas.eval``, exactly as 1.x evaluates it:
    pandas.eval uses numexpr, whose type rules differ from NumPy's (e.g. a
    float32 column times an int64 column is computed in 32 bits), so NumPy
    code would not give the same last bits. Comparisons and logic give the
    same values either way and are written as NumPy code.

    ``df_names`` are other names for the chooser table (1.x's DF setting);
    ``objects`` maps names of objects the model provides to the methods
    formulas may call (e.g. ``{'tt': {'previous_tour_ends', ...}}``).
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
    t = _Translator(formula, constants or {}, python_mode, assigned, functions, skim_accessors,
                    coefficients, df_columns, tables, settings, random_index,
                    pandas_locals, df_names, objects)
    source = t.visit(tree)
    if not python_mode and _has_arithmetic(tree):
        names = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})
        mapping = ", ".join(f"{n!r}: {t.visit(ast.Name(id=n, ctx=ast.Load()))}" for n in names)
        source = f"_pandas_eval({text.strip()!r}, {{{mapping}}})"
    return Translated(formula=formula, source=source, columns=t.columns,
                      constants=t.used_constants, skims=t.skims, names=t.names,
                      skim_uses=t.skim_uses, coefficients=t.coefficients, tables=t.tables,
                      random_draws=t.random_draws)


_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)


def _has_arithmetic(tree: ast.AST) -> bool:
    return any(isinstance(n, ast.BinOp) and isinstance(n.op, _ARITHMETIC) for n in ast.walk(tree))


def fold_constant(formula: str, constants: Mapping[str, object] | None = None, **kw):
    """
    The value of a Python-style formula if it can be worked out now (from
    constants, settings and which columns exist), else raises FormulaError.
    """
    text = formula.strip().lstrip("@")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as err:
        raise FormulaError(f"formula {formula!r} is not valid syntax: {err.msg}") from None
    t = _Translator(formula, constants or {}, True, **kw)
    value = t.fold(tree.body)
    if value is _NOT_CONSTANT:
        raise NotConstant(f"formula {formula!r} can't be worked out when the model loads")
    return value, t.used_constants


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


def _map(x, mapping):
    """pandas Series.map(dict): values not in the dictionary become NaN."""
    return pd.Series(np.asarray(x)).map(mapping).to_numpy()


def _cut(x, bins, labels):
    """pandas pd.cut (right-closed bins); returns labels as a pandas Categorical."""
    return pd.cut(np.asarray(x), bins=bins, labels=labels)


def _astype(x, dtype):
    return np.asarray(x.astype(dtype))


def _lognormal(draws, mu, sigma, scale=False):
    """
    Lognormal values from standard normal draws, as 1.x's assign_variables
    does for rng.lognormal_for_df (scale=True: mu and sigma are the mean and
    standard deviation of the lognormal distribution itself).
    """
    if scale:
        x = 1 + ((sigma * sigma) / (mu * mu))
        mu = np.log(mu / (np.sqrt(x)))
        sigma = np.sqrt(np.log(x))
    return np.exp(draws * sigma + mu)


def _pandas_eval(expr, names):
    """A plain-style formula evaluated with pandas.eval, as 1.x's fast_eval does."""
    n = next((len(v) for v in names.values() if np.ndim(v)), None)
    index = pd.RangeIndex(n) if n is not None else None
    resolvers = {k: (pd.Series(v, index=index, copy=False) if np.ndim(v) else v)
                 for k, v in names.items()}
    out = pd.eval(expr, resolvers=[resolvers])
    return out.to_numpy() if isinstance(out, pd.Series) else out


def _shift(x, periods):
    """pandas Series.shift: move values down (or up, if negative) by rows, filling with missing."""
    return pd.Series(x).shift(periods).to_numpy()


def _as_int8(x):
    """1.x's reindex_i: missing values become 0, then 8-bit integers."""
    x = np.asarray(x)
    if x.dtype.kind == "f":
        x = np.where(np.isnan(x), 0, x)
    return x.astype(np.int8)


def _full(x, n):
    """A scalar result becomes a column of n equal values (1.x promotes scalars to Series)."""
    if np.ndim(x) == 0:
        if isinstance(x, np.ndarray):
            return np.full(n, x[()], dtype=x.dtype)
        return np.full(n, x, dtype=object if isinstance(x, str) else None)
    return x


RUNTIME_NAMESPACE = {"np": np, "_clip": _clip, "_fillna": _fillna, "_map": _map, "_cut": _cut,
                     "_astype": _astype, "_lognormal": _lognormal, "_full": _full,
                     "_pandas_eval": _pandas_eval, "_as_int8": _as_int8, "_shift": _shift}


def build_function(name: str, items: list[tuple[str, Translated]], doc: str = "") -> str:
    """
    Python source for a function ``name(c, s=None, k=None)`` that yields one
    array (or scalar) per formula, in order, so callers can use each term and
    discard it before the next is computed. ``items`` are (label, translated
    formula) pairs. ``s`` looks up skims and ``k`` holds coefficient values
    used inside formulas. The source is plain NumPy code meant to be read by
    people.
    """
    lines = [f"def {name}(c, s=None, k=None):"]
    if doc:
        lines.append(f'    """{doc}"""')
    for label, tr in items:
        lines.append(f"    # {label}: {tr.formula}")
        lines.append(f"    yield {tr.source}")
    if not items:
        lines.append("    yield from ()")
    return "\n".join(lines) + "\n"


def build_assignment_function(name: str, items: list[tuple[str, str, Translated | object]],
                              doc: str = "") -> str:
    """
    Python source for ``name(c, s, n, t=None, r=None, k=None)``, which
    evaluates assignments in order and returns ``v``, a dict of every
    assigned variable (temporary ones, starting with ``_``, included).

    ``items`` are (target, description, translated formula); a non-Translated
    third item is a value worked out when the model loaded (a scalar
    temporary such as ``_DF_IS_TOUR``), recorded as a comment. ``c`` maps
    column names to arrays, ``s(accessor, skim_name)`` returns skim values,
    ``n`` is the number of rows, ``t(table, column, ids)`` looks up another
    table, ``r`` holds random draws and ``k`` coefficient values.

    As in 1.x, a scalar result is made into a column of ``n`` values, except
    for targets in capitals starting with ``_``, which stay scalars.
    """
    lines = [f"def {name}(c, s, n, t=None, r=None, k=None):"]
    if doc:
        lines.append(f'    """{doc}"""')
    lines.append("    v = {}")
    for target, description, tr in items:
        if not isinstance(tr, Translated):
            lines.append(f"    # {description + ': ' if description else ''}{target} = {tr!r} "
                         f"(worked out when the model loaded)")
            continue
        lines.append(f"    # {description + ': ' if description else ''}{tr.formula}")
        if target == "_":
            lines.append(f"    {tr.source}")
        elif target.startswith("_") and target.isupper():
            lines.append(f"    v[{target!r}] = {tr.source}")
        else:
            lines.append(f"    v[{target!r}] = _full({tr.source}, n)")
    lines.append("    return v")
    return "\n".join(lines) + "\n"


def compile_function(source: str, name: str, filename: str = "<asim_core-generated>"):
    namespace = dict(RUNTIME_NAMESPACE)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]
