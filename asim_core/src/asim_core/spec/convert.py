"""
Converting today's formulas (ActivitySim 1.x's two dialects) into asim_core's
formula language (language.py).

A formula is parsed as the translator parses it (translate.py), and each
part is rewritten:

* a column (``df.age``, ``df['age']``, or a bare name in plain style) gets
  the name of the table it comes from in this step (``person.age``), or
  stays a bare name if it is one of the model's derived values;
* ``reindex(land_use.TOPOLOGY, df[dest_col_name])`` becomes
  ``land_use.TOPOLOGY[d]``; ``df[orig_col_name]`` / ``df[dest_col_name]``
  become ``o`` / ``d``;
* skim lookups become ``skim.NAME[o, d]``, ``skim.NAME[d, o, in_period]``,
  ``skim.NAME[o, d, 'AM']`` ...;
* methods become functions (``x.clip(0, 5)`` -> ``clip(x, 0, 5)``,
  ``np.where`` -> ``where``, ``x.isin([1, 2])`` -> ``x in [1, 2]``);
* ``&``, ``|``, ``~`` become ``and``, ``or``, ``not``;
* a test of which columns exist (``'tour_type' in df.columns``) becomes a
  parameter (``has_tour_type``) that the step using the model sets;
* ``rng.lognormal_for_df(df, mu=..., sigma=..., broadcast=True, scale=True)``
  becomes ``random_lognormal(mean=..., sd=...)``.

Anything the translator doesn't support is reported, as there.
"""

from __future__ import annotations

import ast
import keyword
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .translate import ASTYPE_NAMES, ASTYPE_STRINGS, NP_FUNCTIONS, FormulaError

NP_RENAMES = {"minimum": "min", "maximum": "max", "abs": "abs"}
ZONE_CONSTANTS = {"orig_col_name": "o", "dest_col_name": "d"}


@dataclass
class LegacyScope:
    """What a 1.x formula could refer to (as the translator's arguments)."""

    constants: Mapping[str, object] = field(default_factory=dict)
    python_mode: bool | None = None  # None: decided by a leading '@'
    assigned: set[str] = field(default_factory=set)
    pandas_locals: bool = False
    df_names: set[str] = field(default_factory=lambda: {"df"})
    tables: set[str] = field(default_factory=set)  # tables reindex() may read
    # accessor -> (from, to, period): e.g. odt_skims -> ('o', 'd', 'out_period')
    skims: dict[str, tuple[str, str, str | None]] = field(default_factory=dict)
    skim_lookup: set[str] = field(default_factory=set)  # accessors with .lookup(o, d, key)
    functions: set[str] = field(default_factory=set)  # bare functions, e.g. exp
    objects: set[str] = field(default_factory=set)  # e.g. tt
    coefficients: set[str] = field(default_factory=set)
    settings: Mapping[str, object] = field(default_factory=dict)
    df_columns: set[str] | None = None  # for folding tests of which columns exist
    index_name: str | None = None  # the table's ID column (1.x: df.index), e.g. household_id
    # columns of the other tables (e.g. land_use), for tests such as 'x' in land_use: a
    # region's input tables don't change from step to step, so those tests are settled now
    table_columns: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class NewNames:
    """Where names go in asim_core's language, in this step."""

    role_of: Callable[[str], str | None]  # column -> table role ('person'), or None if unknown
    derived: set[str] = field(default_factory=set)  # model values written as bare names


@dataclass
class Converted:
    text: str
    parameters: dict[str, str] = field(default_factory=dict)  # has_x -> column 'x'


def _call(name: str, args: list, keywords: list | None = None) -> ast.Call:
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=keywords or [])


def _name(n: str) -> ast.Name:
    return ast.Name(id=n, ctx=ast.Load())


class _Converter:
    def __init__(self, formula: str, legacy: LegacyScope, names: NewNames, python_mode: bool):
        self.formula = formula
        self.legacy = legacy
        self.names = names
        self.python_mode = python_mode
        self.parameters: dict[str, str] = {}
        self._lookup: tuple[str, ast.AST] | None = None

    def fail(self, node, why: str):
        part = ast.unparse(node) if isinstance(node, ast.AST) else str(node)
        raise FormulaError(f"formula {self.formula!r}: {why} (in '{part}')")

    # --- names ------------------------------------------------------------------------

    def column(self, node, col: str) -> ast.AST:
        if col in self.names.derived:
            return _name(col)
        role = self.names.role_of(col)
        if role is None:
            self.fail(node, f"column '{col}' is not in any table of this step")
        if col.isidentifier() and not keyword.iskeyword(col):
            return ast.Attribute(value=_name(role), attr=col, ctx=ast.Load())
        return ast.Subscript(value=_name(role), slice=ast.Constant(col), ctx=ast.Load())

    def constant_value(self, node):
        """A constant's value (for df[NAME] and folding), or None."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.legacy.constants:
            return self.legacy.constants[node.id]
        return None

    def visit_name(self, node: ast.Name) -> ast.AST:
        lg = self.legacy
        n = node.id
        if self.python_mode or lg.pandas_locals:
            if n in lg.assigned or n in lg.coefficients or n in lg.constants:
                return _name(n)
        if self.python_mode:
            self.fail(node, f"unknown name '{n}'")
        return self.column(node, n)

    # --- the walk ------------------------------------------------------------------------

    def visit(self, node) -> ast.AST:
        lg = self.legacy
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            return node
        if isinstance(node, (ast.List, ast.Tuple)):
            return ast.List(elts=[self.visit(e) for e in node.elts], ctx=ast.Load())
        if isinstance(node, ast.Name):
            return self.visit_name(node)
        if isinstance(node, ast.Attribute):
            v = node.value
            if isinstance(v, ast.Name) and v.id in lg.df_names:
                if node.attr == "index":
                    if not lg.index_name:
                        self.fail(node, "the table's ID column is not known")
                    return self.column(node, lg.index_name)
                return self.column(node, node.attr)
            if isinstance(v, ast.Name) and v.id == "np" and node.attr in ("inf", "nan"):
                return _name(node.attr)
            if isinstance(v, ast.Name) and v.id in lg.tables:
                return self.table_column(node, v.id, node.attr)
            self.fail(node, "unsupported attribute")
        if isinstance(node, ast.Subscript):
            v = node.value
            if isinstance(v, ast.Name) and v.id in lg.df_names:
                key = node.slice
                if isinstance(key, ast.Name) and key.id in ZONE_CONSTANTS:
                    return _name(ZONE_CONSTANTS[key.id])
                col = self.constant_value(key)
                if isinstance(col, str):
                    return self.column(node, col)
                self.fail(node, "df[...] needs a column name")
            if isinstance(v, ast.Name) and v.id in lg.skims:
                return self.skim_get(node, v.id)
            if isinstance(v, ast.Name) and v.id in lg.tables:
                col = self.constant_value(node.slice)
                if isinstance(col, str):
                    return self.table_column(node, v.id, col)
            self.fail(node, "unsupported subscript")
        if isinstance(node, ast.IfExp):
            test = self.visit(node.test)
            if isinstance(test, ast.Constant):  # settled now: keep the branch that applies
                return self.visit(node.body if test.value else node.orelse)
            return ast.IfExp(test=test, body=self.visit(node.body), orelse=self.visit(node.orelse))
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                op = ast.And() if isinstance(node.op, ast.BitAnd) else ast.Or()
                return ast.BoolOp(op=op, values=[self.visit(node.left), self.visit(node.right)])
            if isinstance(node.op, ast.BitXor):
                self.fail(node, "^ is not supported")
            return ast.BinOp(left=self.visit(node.left), op=node.op, right=self.visit(node.right))
        if isinstance(node, ast.UnaryOp):
            op = ast.Not() if isinstance(node.op, (ast.Invert, ast.Not)) else node.op
            return ast.UnaryOp(op=op, operand=self.visit(node.operand))
        if isinstance(node, ast.BoolOp):
            return ast.BoolOp(op=node.op, values=[self.visit(v) for v in node.values])
        if isinstance(node, ast.Compare):
            return self.compare(node)
        if isinstance(node, ast.Call):
            return self.call(node)
        self.fail(node, f"unsupported syntax ({type(node).__name__})")

    def compare(self, node: ast.Compare) -> ast.AST:
        member = self.membership(node)
        if member is not None:
            return member
        return ast.Compare(left=self.visit(node.left), ops=node.ops,
                           comparators=[self.visit(c) for c in node.comparators])

    def membership(self, node: ast.Compare):
        """'x' in df.columns -> parameter has_x; 'x' in TABLE -> its value now."""
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.In, ast.NotIn)):
            return None
        container = node.comparators[0]
        if isinstance(container, ast.Attribute) and container.attr == "columns":
            container = container.value
        if not isinstance(container, ast.Name):
            return None
        col = self.constant_value(node.left)
        if container.id in self.legacy.df_names:
            if not isinstance(col, str):
                self.fail(node, "column test needs a column name in quotes")
            param = f"has_{col}"
            self.parameters[param] = col
            test = _name(param)
        elif container.id in self.legacy.tables:
            if container.id not in self.legacy.table_columns:
                self.fail(node, f"tests of {container.id}'s columns need its columns")
            found = col in self.legacy.table_columns[container.id]
            return ast.Constant(found if isinstance(node.ops[0], ast.In) else not found)
        else:
            return None
        return ast.UnaryOp(op=ast.Not(), operand=test) if isinstance(node.ops[0], ast.NotIn) else test

    def table_column(self, node, table: str, col: str) -> ast.AST:
        if self._lookup is None or self._lookup[0] != table:
            self.fail(node, f"columns of '{table}' can only be used inside reindex()")
        return ast.Subscript(value=ast.Attribute(value=_name(table), attr=col, ctx=ast.Load()),
                             slice=self._lookup[1], ctx=ast.Load())

    # --- skims -----------------------------------------------------------------------------

    def _skim(self, name: str, a: str | ast.AST, b: str | ast.AST, period) -> ast.AST:
        keys = [_name(a) if isinstance(a, str) else a, _name(b) if isinstance(b, str) else b]
        if period is not None:
            keys.append(ast.Constant(period) if not isinstance(period, ast.AST) else period)
        return ast.Subscript(value=ast.Attribute(value=_name("skim"), attr=name, ctx=ast.Load()),
                             slice=ast.Tuple(elts=keys, ctx=ast.Load()), ctx=ast.Load())

    def skim_key(self, node, key) -> tuple[str, str | None]:
        value = self.constant_value(key)
        if isinstance(key, ast.Tuple) and len(key.elts) == 2:
            parts = [self.constant_value(e) for e in key.elts]
            if all(isinstance(p, str) for p in parts):
                return parts[0], parts[1]
        if isinstance(value, str):
            return value, None
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(p, str) for p in value):
            return value[0], value[1]
        self.fail(node, "skim keys must be 'NAME' or ('NAME', 'PERIOD')")

    def skim_get(self, node: ast.Subscript, accessor: str) -> ast.AST:
        a, b, period = self.legacy.skims[accessor]
        name, fixed = self.skim_key(node, node.slice)
        if fixed is not None:
            if period is not None:
                self.fail(node, "a fixed period with a time-of-day skim lookup")
            return self._skim(name, a, b, fixed)
        return self._skim(name, a, b, _name(period) if period else None)

    # --- calls ----------------------------------------------------------------------------

    def call(self, node: ast.Call) -> ast.AST:
        lg = self.legacy
        func = node.func
        if isinstance(func, ast.Name) and func.id == "setting":
            return node
        if isinstance(func, ast.Name) and func.id == "reindex":
            return self.reindex(node)
        if isinstance(func, ast.Name) and func.id == "reindex_i":
            return self.reindex_i(node)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            base = func.value.id
            if base in lg.objects:
                return ast.Call(func=func, args=[self.visit(a) for a in node.args], keywords=[])
            if base in lg.skim_lookup and func.attr == "lookup":
                if len(node.args) != 3:
                    self.fail(node, "lookup() takes origins, destinations and a skim name")
                name, period = self.skim_key(node, node.args[2])
                return self._skim(name, self.visit(node.args[0]), self.visit(node.args[1]), period)
            if base in lg.skims and func.attr in ("reverse", "max"):
                a, b, period = lg.skims[base]
                if period is not None or len(node.args) != 1:
                    self.fail(node, f".{func.attr}() is only supported on skims without a time period")
                name, fixed = self.skim_key(node, node.args[0])
                rev = self._skim(name, b, a, fixed)
                if func.attr == "reverse":
                    return rev
                return _call("max", [rev, self._skim(name, a, b, fixed)])
            if base == "np":
                if func.attr not in NP_FUNCTIONS:
                    self.fail(node, f"np.{func.attr} is not supported")
                args = [self.visit(a) for a in node.args]
                kws = [ast.keyword(arg=k.arg, value=self.visit(k.value)) for k in node.keywords]
                if func.attr == "isin":
                    if len(args) != 2 or kws:
                        self.fail(node, "np.isin() takes values and a list")
                    return ast.Compare(left=args[0], ops=[ast.In()], comparators=[args[1]])
                if func.attr == "clip":
                    return _call("clip", args, kws)
                return _call(NP_RENAMES.get(func.attr, func.attr), args, kws)
            if base == "pd":
                if func.attr != "cut":
                    self.fail(node, f"pd.{func.attr} is not supported")
                return _call("cut", [self.visit(a) for a in node.args],
                             [ast.keyword(arg=k.arg, value=self.visit(k.value)) for k in node.keywords])
            if base == "rng":
                return self.lognormal(node)
        if isinstance(func, ast.Name) and func.id == "abs":
            return _call("abs", [self.visit(a) for a in node.args])
        if isinstance(func, ast.Name) and func.id in lg.functions:
            return _call(func.id, [self.visit(a) for a in node.args])
        if isinstance(func, ast.Attribute) and func.attr == "reindex":
            grouped = self.group_sum(func.value)
            if grouped is not None:
                col, key = grouped
                return _call("sum_by", [col, key, self.visit(node.args[0])])
        if isinstance(func, ast.Attribute):
            return self.method(node, func)
        self.fail(node, "unsupported function call")

    def group_sum(self, node):
        """T.column.groupby(T.key).sum() -> (T.column, T.key) as written in asim_core's language."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sum"):
            return None
        g = node.func.value
        if not (isinstance(g, ast.Call) and isinstance(g.func, ast.Attribute) and g.func.attr == "groupby"):
            return None
        col, key = g.func.value, g.args[0]
        if (isinstance(col, ast.Attribute) and isinstance(col.value, ast.Name) and col.value.id in self.legacy.tables
                and isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name)
                and key.value.id == col.value.id):
            return (ast.Attribute(value=_name(col.value.id), attr=col.attr, ctx=ast.Load()),
                    ast.Attribute(value=_name(key.value.id), attr=key.attr, ctx=ast.Load()))
        return None

    def method(self, node: ast.Call, func: ast.Attribute) -> ast.AST:
        x = self.visit(func.value)
        m = func.attr
        kw = {k.arg: k.value for k in node.keywords}
        args = [] if m in ("astype", "apply", "map") else [self.visit(a) for a in node.args]
        if m == "clip":
            return _call("clip", [x] + args, [ast.keyword(arg=k, value=self.visit(v)) for k, v in kw.items()])
        if m == "isin":
            return ast.Compare(left=x, ops=[ast.In()], comparators=[args[0]])
        if m == "between":
            inclusive = kw.get("inclusive")
            inclusive = inclusive.value if isinstance(inclusive, ast.Constant) else "both"
            lo, hi = args
            if inclusive == "both":
                return _call("between", [x, lo, hi])
            lop = ast.LtE() if inclusive == "left" else ast.Lt()
            hop = ast.LtE() if inclusive == "right" else ast.Lt()
            return ast.Compare(left=lo, ops=[lop, hop], comparators=[x, hi])
        if m == "fillna":
            return _call("fillna", [x] + args)
        if m == "apply":
            f = node.args[0] if len(node.args) == 1 else None
            if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "np"
                    and f.attr in NP_FUNCTIONS):
                self.fail(node, "apply() supports np.<function>")
            return _call(NP_RENAMES.get(f.attr, f.attr), [x])
        if m == "shift":
            return _call("shift", [x] + (args or [self.visit(kw["periods"])]))
        if m == "map":
            mapping = node.args[0]
            if isinstance(mapping, ast.DictComp):
                gen = mapping.generators[0]
                mapping = gen.iter.func.value  # {k: v for k, v in X.items()} is X
            return _call("map", [x, mapping if isinstance(mapping, ast.Name) else self.visit(mapping)])
        if m == "astype":
            a = node.args[0]
            if isinstance(a, ast.Name) and a.id in ASTYPE_NAMES:
                return _call(a.id, [x])
            if isinstance(a, ast.Constant) and a.value in ASTYPE_STRINGS:
                return _call(a.value, [x])
            if isinstance(a, ast.Attribute) and a.attr in ASTYPE_STRINGS:
                return _call(a.attr, [x])
            self.fail(node, "unsupported astype()")
        self.fail(node, f"method .{m}() is not supported")

    def reindex(self, node: ast.Call) -> ast.AST:
        tables = {n.id for n in ast.walk(node.args[0]) if isinstance(n, ast.Name) and n.id in self.legacy.tables}
        if len(tables) != 1:
            self.fail(node, "reindex() needs columns of one table")
        self._lookup = (tables.pop(), self.visit(node.args[1]))
        try:
            return self.visit(node.args[0])
        finally:
            self._lookup = None

    def reindex_i(self, node: ast.Call) -> ast.AST:
        arg, keys = node.args[0], self.visit(node.args[1])
        if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr == "size"
                and isinstance(arg.func.value, ast.Call)):
            g = arg.func.value
            table = g.func.value.id
            col = self.constant_value(g.args[0])
            column = ast.Attribute(value=_name(table), attr=col, ctx=ast.Load())
            # 1.x's reindex_i gives 8-bit integers
            return _call("int8", [_call("count_by", [column, keys])])
        self.fail(node, "reindex_i() is supported as reindex_i(table.groupby('column').size(), ids)")

    def lognormal(self, node: ast.Call) -> ast.AST:
        if node.func.attr != "lognormal_for_df":
            self.fail(node, "only rng.lognormal_for_df is supported")
        kw = {k.arg: k.value for k in node.keywords}
        scale = self.constant_value(kw.get("scale", ast.Constant(False)))
        mu, sigma = self.visit(kw["mu"]), self.visit(kw["sigma"])
        names = ("mean", "sd") if scale else ("mu", "sigma")
        return _call("random_lognormal", [], [ast.keyword(arg=names[0], value=mu),
                                              ast.keyword(arg=names[1], value=sigma)])


def convert_formula(formula: str, legacy: LegacyScope, names: NewNames) -> Converted:
    """One 1.x formula in asim_core's language."""
    text = formula.strip()
    python_mode = legacy.python_mode if legacy.python_mode is not None else text.startswith("@")
    if text.startswith("@"):
        text = text[1:]
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as err:
        raise FormulaError(f"formula {formula!r} is not valid syntax: {err.msg}") from None
    c = _Converter(formula, legacy, names, python_mode)
    new = ast.fix_missing_locations(c.visit(tree))
    return Converted(ast.unparse(new), c.parameters)
