"""
The model definition: one checked, self-contained description of a choice model.

Whatever file format a model is written in (today's YAML + spec CSV +
coefficients CSV, or a future format), it is loaded into this Pydantic class.
Loading runs the checks below, so mistakes surface when the model loads:

* every alternative a term mentions is one of the model's alternatives
* every coefficient a term uses exists in the coefficients
* every formula is valid and uses only supported operations
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator

from .language import RANDOM_CALL, Scope, SkimScope, compile_formula, fold_formula
from .translate import (
    FormulaError,
    NeedsColumns,
    NotConstant,
    Translated,
    build_assignment_function,
    build_function,
    compile_function,
    fold_constant,
    translate,
)


class Term(BaseModel):
    """One utility term: a formula, and a coefficient for each alternative it applies to."""

    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    formula: str
    # alternative -> coefficient cell as written (a coefficient name, a number,
    # or simple arithmetic of those). Alternatives the term does not apply to
    # are left out.
    coefficients: dict[str, str]


class Coefficient(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    constrain: bool = False  # held fixed during estimation


class Nest(BaseModel):
    """
    One nest of a nested logit model: a name, a nesting coefficient (a number,
    or the name of a coefficient), and the alternatives or nests inside it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    coefficient: float | str
    alternatives: list[Union[str, "Nest"]]

    @field_validator("coefficient", mode="before")
    @classmethod
    def _numbers_as_floats(cls, v):
        # as 1.x: a coefficient written as text that reads as a number is a number
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return v.strip()
        return v

    def walk(self):
        """Every nest and alternative name below and including this nest."""
        yield self
        for a in self.alternatives:
            if isinstance(a, Nest):
                yield from a.walk()
            else:
                yield a


Nest.model_rebuild()


@dataclass
class NestTree:
    """A nest with its coefficient resolved to a number (used by the arithmetic)."""

    name: str
    coefficient: float
    alternatives: list = field(default_factory=list)  # str (alternative) or NestTree


Language = Literal["asim1", "asim_core"]


class LanguageScope(BaseModel):
    """
    The names a model written in asim_core's language can use, besides its
    constants and coefficients (see language.py). Set by the model's kind.
    """

    model_config = ConfigDict(extra="forbid")

    roles: list[str] = []          # tables formulas read as role.column
    derived: list[str] = []        # model values in the rows (c[name])
    tables: list[str] = []         # tables looked up by ID: table.column[ids]
    # skim lookups: "o d out_period" (from, to, period or None) -> the step's accessor name
    skims: dict[str, str] = {}
    skim_lookup: str | None = None
    objects: dict[str, list[str]] = {}
    # parameters set by the step using the model (None: must be set)
    parameters: dict[str, Any] = {}

    def make(self, constants: dict, coefficients=(), assigned=(), random_index=None, settings=None,
             roles: dict | None = None, tables: dict | None = None, extra=None) -> Scope:
        extra = dict(extra or {})
        params = {k: extra.pop(k) if k in extra else v for k, v in self.parameters.items()}
        consts = {**constants, **extra}
        zones = {z: consts[c] for z, c in (("o", "orig_col_name"), ("d", "dest_col_name")) if c in consts}
        skims = None
        if self.skims or self.skim_lookup:
            pats = {}
            for key, acc in self.skims.items():
                a, b, p = (key.split() + ["None"])[:3]
                pats[(a, b, None if p == "None" else p)] = acc
            skims = SkimScope(pats, self.skim_lookup)
        role_cols = {r: None for r in self.roles}
        role_cols.update({r: set(c) for r, c in (roles or {}).items() if r in role_cols})
        table_cols = {t: None for t in self.tables}
        table_cols.update({t: set(c) for t, c in (tables or {}).items() if t in table_cols})
        return Scope(roles=role_cols, derived=set(self.derived), assigned=set(assigned), constants=consts,
                     coefficients=set(coefficients), parameters=params, tables=table_cols, skims=skims,
                     objects={k: set(v) for k, v in self.objects.items()}, zones=zones,
                     settings=dict(settings or {}), random_index=random_index)


def check_columns(translated, roles: dict[str, set[str]]) -> list[str]:
    """
    For asim_core's language: every role.column read exists in the run's
    tables (checked once the step knows its columns). Returns problems.
    """
    import difflib

    problems = []
    for tr in translated:
        for role, col in sorted(getattr(tr, "qualified", ()) or ()):
            if role not in roles:
                problems.append(f"{tr.formula!r}: table '{role}' is not available in this step "
                                f"(available: {sorted(roles)})")
            elif col not in roles[role]:
                close = difflib.get_close_matches(col, sorted(roles[role]), n=3, cutoff=0.6)
                hint = f"; did you mean {' or '.join(close)}?" if close else ""
                problems.append(f"{tr.formula!r}: {role} has no column '{col}'{hint}")
    return problems


class ChoiceModelDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    logit_type: Literal["MNL", "NL"] = "MNL"
    alternatives: list[str]
    nests: Nest | None = None
    constants: dict[str, Any] = {}
    terms: list[Term]
    coefficients: dict[str, Coefficient]
    # Optional coefficient template (segments, e.g. tour purposes): generic
    # coefficient name -> {segment: the coefficient used for that segment}.
    # Terms and nests then use the generic names.
    coefficient_template: dict[str, dict[str, str]] | None = None
    # whether formulas can use coefficient names as values (1.x's mode choice
    # models allow it; other simple models don't)
    formulas_see_coefficients: bool = False
    # names that look up skims in formulas, e.g. odt_skims['SOV_TIME']
    skim_accessors: list[str] = []
    # the formulas' language: 1.x's (two dialects) or asim_core's
    language: Language = "asim1"
    scope: LanguageScope | None = None  # for asim_core's language
    source_files: list[str] = []

    _translated: list[Translated] = PrivateAttr(default_factory=list)

    # --- checks ---------------------------------------------------------------

    @model_validator(mode="after")
    def _check(self) -> ChoiceModelDefinition:
        problems = self._check_template()
        alts = set(self.alternatives)
        names = self.coefficient_names
        translated = []
        for i, term in enumerate(self.terms):
            where = f"term {i + 1} ({term.label or term.formula})"
            for alt, cell in term.coefficients.items():
                if alt not in alts:
                    problems.append(f"{where}: unknown alternative '{alt}'")
                try:
                    for name in _names_in_cell(cell):
                        if name not in names:
                            problems.append(f"{where}, {alt}: unknown coefficient '{name}'")
                except FormulaError as err:
                    problems.append(f"{where}, {alt}: {err}")
            try:
                coefs = names if self.formulas_see_coefficients else frozenset()
                if self.language == "asim_core":
                    translated.append(compile_formula(
                        term.formula, (self.scope or LanguageScope()).make(self.constants, coefs)))
                else:
                    translated.append(translate(term.formula, self.constants, coefficients=coefs,
                                                skim_accessors=set(self.skim_accessors)))
            except FormulaError as err:
                problems.append(f"{where}: {err}")
                translated.append(None)
        problems += self._check_nests()
        if problems:
            raise ValueError(
                f"model '{self.name}' has {len(problems)} problem(s):\n  " + "\n  ".join(problems)
            )
        self._translated = translated
        return self

    def _check_template(self) -> list[str]:
        if self.coefficient_template is None:
            return []
        problems = []
        for generic, by_segment in self.coefficient_template.items():
            for segment, name in by_segment.items():
                if name not in self.coefficients:
                    problems.append(f"coefficient template: '{generic}' for segment '{segment}' "
                                    f"is '{name}', which is not in the coefficients")
        if not self.segments:
            problems.append("coefficient template has no segments")
        return problems

    @property
    def segments(self) -> list[str]:
        """Segment names of the coefficient template (empty without one)."""
        if not self.coefficient_template:
            return []
        first = next(iter(self.coefficient_template.values()))
        return list(first)

    @property
    def coefficient_names(self) -> set[str]:
        """Names terms and nests use: generic names with a template, else the coefficients."""
        if self.coefficient_template is not None:
            return set(self.coefficient_template)
        return set(self.coefficients)

    def coefficient_values(self, segment: str | None = None) -> dict[str, float]:
        """Name -> value, for one segment when the model has a coefficient template."""
        if self.coefficient_template is None:
            if segment is not None:
                raise ValueError(f"model '{self.name}' has no segments")
            return {k: c.value for k, c in self.coefficients.items()}
        if segment not in self.segments:
            raise KeyError(f"model '{self.name}' has no segment '{segment}' "
                           f"(segments: {self.segments})")
        return {g: self.coefficients[by_seg[segment]].value
                for g, by_seg in self.coefficient_template.items()}

    def _check_nests(self) -> list[str]:
        if self.logit_type == "MNL":
            return []
        if self.nests is None:
            return ["nested logit needs NESTS"]
        problems = []
        names, leaves = [], []
        for node in self.nests.walk():
            if isinstance(node, Nest):
                names.append(node.name)
                if isinstance(node.coefficient, str) and node.coefficient not in self.coefficient_names:
                    problems.append(f"nest '{node.name}': unknown coefficient '{node.coefficient}'")
                if not node.alternatives:
                    problems.append(f"nest '{node.name}' has no alternatives")
            else:
                names.append(node)
                leaves.append(node)
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            problems.append(f"names used more than once in NESTS: {dupes}")
        missing = [a for a in self.alternatives if a not in leaves]
        extra = [a for a in leaves if a not in self.alternatives]
        if missing:
            problems.append(f"alternatives not in any nest: {missing}")
        if extra:
            problems.append(f"NESTS lists unknown alternatives: {extra}")
        return problems

    # --- derived values -------------------------------------------------------

    def nest_tree(self, segment: str | None = None) -> NestTree | None:
        """The nests with coefficient names replaced by values (64-bit, as 1.x does)."""
        if self.logit_type != "NL":
            return None
        values = self.coefficient_values(segment)

        def resolve(nest: Nest) -> NestTree:
            coef = nest.coefficient
            coef = float(values[coef]) if isinstance(coef, str) else float(coef)
            kids = [resolve(a) if isinstance(a, Nest) else a for a in nest.alternatives]
            return NestTree(nest.name, coef, kids)

        return resolve(self.nests)

    @property
    def translated(self) -> list[Translated]:
        return self._translated

    def columns_used(self, term_mask: np.ndarray | None = None) -> set[str]:
        cols = set()
        for i, tr in enumerate(self._translated):
            if term_mask is None or term_mask[i]:
                cols |= tr.columns
        return cols

    def skim_uses(self, term_mask: np.ndarray | None = None) -> set[tuple[str, str, str]]:
        """(accessor, skim key, how) for every skim lookup in the formulas."""
        uses = set()
        for i, tr in enumerate(self._translated):
            if term_mask is None or term_mask[i]:
                uses |= tr.skim_uses
        return uses

    def coefficient_matrix(self, float32_rounding: bool = True, segment: str | None = None) -> np.ndarray:
        """
        Coefficients as a (terms x alternatives) array; 0 where a term does not
        apply. ActivitySim 1.x rounds coefficients to 32-bit floats before using
        them, so that is the default here too, for exact matching.
        """
        values = self.coefficient_values(segment)
        m = np.zeros((len(self.terms), len(self.alternatives)), dtype=np.float64)
        col = {a: j for j, a in enumerate(self.alternatives)}
        for i, term in enumerate(self.terms):
            for alt, cell in term.coefficients.items():
                m[i, col[alt]] = _eval_cell(cell, values)
        if float32_rounding:
            m = m.astype(np.float32).astype(np.float64)
        return m

    def utility_function(self, term_mask: np.ndarray | None = None):
        """
        Compile the formulas into one NumPy function ``f(c)`` that yields one
        array per term. Returns (function, source code). Terms where
        ``term_mask`` is False are listed as comments but not computed.
        """
        func_name = f"{_identifier(self.name)}_terms"
        items = []
        skipped = []
        for i, (term, tr) in enumerate(zip(self.terms, self._translated)):
            label = term.label or f"term_{i + 1}"
            if term_mask is None or term_mask[i]:
                items.append((label, tr))
            else:
                skipped.append(f"{label}: {term.formula}")
        doc = f"Utility terms for '{self.name}', generated by asim_core from {', '.join(self.source_files) or 'the model definition'}."
        source = build_function(func_name, items, doc)
        if skipped:
            source += "\n# Not computed (all coefficients are zero, as in ActivitySim 1.x):\n"
            source += "".join(f"#   {s}\n" for s in skipped)
        return compile_function(source, func_name, f"<{func_name}>"), source


class Assignment(BaseModel):
    """One row of an assignment spec: ``target = formula``."""

    model_config = ConfigDict(extra="forbid")

    target: str
    formula: str
    description: str = ""


class InteractionTerm(BaseModel):
    """
    One row of an interaction spec: a formula and its coefficient. A row
    written ``_name@formula`` assigns a temporary instead (``target``), which
    later rows can use; it adds nothing to the utility.
    """

    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    formula: str
    coefficient: str  # a coefficient name, a number, or simple arithmetic of those
    target: str | None = None


class InteractionModelDefinition(BaseModel):
    """
    A choice model whose alternatives are made at run time and joined to
    each chooser, e.g. a tour's possible departure and return times. One
    coefficient per term; the utility of a chooser-alternative row is the
    sum of term value times coefficient. As in 1.x's interaction models,
    plain-style formulas see the model's own names (temporaries, constants)
    before columns, and formulas can call the methods listed in ``objects``
    (e.g. the timetable's ``tt.previous_tour_ends``).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    terms: list[InteractionTerm]
    coefficients: dict[str, Coefficient]
    constants: dict[str, Any] = {}
    objects: dict[str, list[str]] = {}
    # names that look up skims, e.g. skims['DIST']
    skim_accessors: list[str] = []
    language: Language = "asim1"
    scope: LanguageScope | None = None  # for asim_core's language
    source_files: list[str] = []

    _translated: list[Translated] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> InteractionModelDefinition:
        problems = []
        translated = []
        assigned: set[str] = set()
        objects = {k: set(v) for k, v in self.objects.items()}
        for i, term in enumerate(self.terms):
            where = f"row {i + 1} ({term.label or term.formula})"
            try:
                for name in _names_in_cell(term.coefficient):
                    if name not in self.coefficients:
                        problems.append(f"{where}: unknown coefficient '{name}'")
            except FormulaError as err:
                problems.append(f"{where}: {err}")
            try:
                if self.language == "asim_core":
                    translated.append(compile_formula(
                        term.formula, (self.scope or LanguageScope()).make(self.constants, assigned=assigned)))
                else:
                    translated.append(translate(
                        term.formula, self.constants, assigned=assigned, pandas_locals=True,
                        objects=objects, python_mode=True if term.target else None,
                        skim_accessors=set(self.skim_accessors),
                    ))
            except FormulaError as err:
                problems.append(f"{where}: {err}")
                translated.append(None)
            if term.target:
                assigned.add(term.target)
        if problems:
            raise ValueError(
                f"model '{self.name}' has {len(problems)} problem(s):\n  " + "\n  ".join(problems)
            )
        self._translated = translated
        return self

    @property
    def translated(self) -> list[Translated]:
        return self._translated

    def coefficient_values(self) -> np.ndarray:
        """Each row's coefficient, rounded to 32-bit as 1.x does (temporaries included)."""
        values = {k: c.value for k, c in self.coefficients.items()}
        return np.array([_eval_cell(t.coefficient, values) for t in self.terms],
                        dtype=np.float32)

    def kept(self) -> np.ndarray:
        """Rows that are computed: 1.x drops rows whose coefficient is zero."""
        return self.coefficient_values() != 0

    def columns_used(self, mask: np.ndarray | None = None) -> set[str]:
        mask = self.kept() if mask is None else mask
        return set().union(set(), *(tr.columns for tr, m in zip(self._translated, mask) if m))

    def skim_uses(self, mask: np.ndarray | None = None) -> set[tuple[str, str, str]]:
        mask = self.kept() if mask is None else mask
        return set().union(set(), *(tr.skim_uses for tr, m in zip(self._translated, mask) if m))

    def utility_terms(self, mask: np.ndarray | None = None):
        """
        Compile into ``f(c, n, tt=None, s=None)``, which yields the value of each
        computed term in order (temporaries are assigned as it goes), and
        return (function, coefficients of the yielded terms as Python floats,
        source). Python floats, because 1.x multiplies pandas Series by the
        coefficients, and pandas treats a NumPy float32 scalar like a Python
        float.
        """
        mask = self.kept() if mask is None else mask
        coefs = self.coefficient_values()
        func_name = f"{_identifier(self.name)}_terms"
        lines = [f"def {func_name}(c, n, tt=None, s=None):",
                 f'    """Utility terms for \'{self.name}\', generated by asim_core from '
                 f'{", ".join(self.source_files) or "the model definition"}."""',
                 "    v = {}"]
        term_coefs = []
        for i, (term, tr) in enumerate(zip(self.terms, self._translated)):
            label = term.label or f"row_{i + 1}"
            if not mask[i]:
                lines.append(f"    # {label}: not computed (coefficient is zero): {term.formula}")
                continue
            lines.append(f"    # {label}: {term.formula}")
            if term.target:
                lines.append(f"    v[{term.target!r}] = _full({tr.source}, n)")
            else:
                lines.append(f"    yield {tr.source}")
                term_coefs.append(float(coefs[i]))
        if not term_coefs:
            lines.append("    yield from ()")
        source = "\n".join(lines) + "\n"
        return compile_function(source, func_name, f"<{func_name}>"), term_coefs, source


RANDOM_MARKER = "rng.lognormal_for_df(df,"  # as 1.x's assign_variables looks for it


def counting_lambda(formula: str) -> str | None:
    """
    The key column if the formula is 1.x's counting lambda, e.g.
    ``lambda query, persons, households: persons.query(query).groupby('household_id')
    .size().reindex(households.index).fillna(0).astype(np.int8)``: for each row
    of the second table, how many rows of the first meet a query. Else None.
    """
    import ast as _ast

    text = formula.strip().lstrip("@").strip()
    if not text.startswith("lambda"):
        return None
    try:
        node = _ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, _ast.Lambda) or len(node.args.args) != 3:
        return None
    q, table, other = (a.arg for a in node.args.args)

    def call(n, attr, nargs):
        ok = (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute) and n.func.attr == attr
              and len(n.args) == nargs and not n.keywords)
        return n.func.value if ok else None

    n = call(node.body, "astype", 1)
    if n is None or _ast.unparse(node.body.args[0]) not in ("np.int8", "'int8'"):
        return None
    fill = n
    n = call(fill, "fillna", 1)
    if n is None or _ast.unparse(fill.args[0]) != "0":
        return None
    reindex = n
    n = call(reindex, "reindex", 1)
    if n is None or _ast.unparse(reindex.args[0]) != f"{other}.index":
        return None
    n = call(n, "size", 0)
    grouped = n
    n = call(grouped, "groupby", 1)
    if n is None or not isinstance(grouped.args[0], _ast.Constant) or not isinstance(grouped.args[0].value, str):
        return None
    key = grouped.args[0].value
    queried = n
    n = call(queried, "query", 1)
    if n is None or _ast.unparse(queried.args[0]) != q or not (isinstance(n, _ast.Name) and n.id == table):
        return None
    return key


@dataclass
class BoundAssignments:
    """
    An assignment spec translated for a known set of columns: the generated
    function and everything it reads. Made by ``AssignmentSpec.bind``.
    """

    spec_name: str
    # (target, description, Translated) or (target, description, value) for
    # scalar temporaries worked out when the model loaded
    items: list[tuple[str, str, object]]
    function: Any
    source: str
    targets: list[str]
    n_random: int  # random draws per row (rng.lognormal_for_df)

    @property
    def translated(self) -> list[Translated]:
        return [tr for _, _, tr in self.items if isinstance(tr, Translated)]

    @property
    def draws_per_call(self) -> bool:
        """True if the random draws are made one call at a time (rng.lognormal_for_df with
        broadcast=False: each call takes the next draw of each row's stream), False if all
        of them at once (broadcast=True)."""
        modes = {t.random_broadcast for t in self.translated if t.random_draws}
        if len(modes) > 1:
            raise ValueError(f"'{self.spec_name}' mixes lognormal draws with and without broadcast")
        return modes == {False}

    def columns_used(self) -> set[str]:
        return set().union(set(), *(t.columns for t in self.translated))

    def skims_used(self) -> set[str]:
        return set().union(set(), *(t.skims for t in self.translated))

    def skim_uses(self) -> set[tuple[str, str, str]]:
        return set().union(set(), *(t.skim_uses for t in self.translated))

    def tables_used(self) -> set[tuple[str, str]]:
        return set().union(set(), *(t.tables for t in self.translated))

    def coefficients_used(self) -> set[str]:
        return set().union(set(), *(t.coefficients for t in self.translated))


class AssignmentSpec(BaseModel):
    """
    A list of assignments evaluated in order, as in ActivitySim 1.x's
    preprocessor, annotation and accessibility files. Targets starting with
    ``_`` are temporary; ``_`` alone is evaluated and thrown away. Any target,
    once assigned, can be used by later rows by name. Temporaries in capitals
    (``_DF_IS_TOUR``) hold a single value; when it can be worked out from
    constants and which columns exist, that is done when the model loads.

    Loading checks every formula, with each row seeing only the variables
    assigned before it. Rows that test which columns exist (``'x' in
    df.columns``) or read other tables can only be checked once the run's
    columns are known: ``bind`` does that, and the components call it in the
    startup check, before anything runs.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    assignments: list[Assignment]
    constants: dict[str, Any] = {}
    # bare function names formulas may call, mapped to NumPy
    functions: dict[str, str] = {"exp": "np.exp", "log": "np.log"}
    # names that look up skims, e.g. skim_od[('SOVTOLL_TIME', 'AM')]
    skim_accessors: list[str] = []
    # other tables formulas may read with reindex(), e.g. ['land_use', 'tours']
    tables: list[str] = []
    # names whose values are supplied at run time (k['name']), e.g. coefficients
    coefficients: list[str] = []
    # values for setting('name')
    settings: dict[str, Any] = {}
    # other names for the table being annotated (1.x's DF setting), besides df
    df_names: list[str] = []
    language: Language = "asim1"
    scope: LanguageScope | None = None  # for asim_core's language
    source_files: list[str] = []

    _bound: BoundAssignments | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check(self) -> AssignmentSpec:
        problems = []
        try:
            self._bound = self._bind(None, None, problems)
        except NeedsColumns:
            self._bound = None  # checked by bind() once the columns are known
        if not self.targets:
            problems.append("no non-temporary targets: nothing would be produced")
        if problems:
            raise ValueError(
                f"assignments '{self.name}' have {len(problems)} problem(s):\n  " + "\n  ".join(problems)
            )
        return self

    def bind(self, df_columns: set[str] | None = None,
             tables: dict[str, set[str]] | None = None,
             constants: dict[str, Any] | None = None) -> BoundAssignments:
        """
        Translate for the given chooser columns and other tables' columns
        (table name -> columns). ``constants`` add to or replace the spec's
        constants for this binding (e.g. a different destination column).
        Raises ValueError listing every problem.
        """
        problems: list[str] = []
        bound = self._bind(df_columns, tables, problems, constants)
        if problems:
            raise ValueError(
                f"assignments '{self.name}' have {len(problems)} problem(s):\n  " + "\n  ".join(problems)
            )
        return bound

    def _bind(self, df_columns, tables, problems, extra_constants=None) -> BoundAssignments:
        if self.language == "asim_core":
            return self._bind_new(tables, problems, extra_constants)
        table_columns = {t: None for t in self.tables}
        table_columns.update({t: set(c) for t, c in (tables or {}).items() if t in table_columns})
        constants = dict(self.constants)
        constants.update(extra_constants or {})
        kw = dict(functions=self.functions, skim_accessors=set(self.skim_accessors),
                  coefficients=set(self.coefficients), df_columns=df_columns,
                  tables=table_columns, settings=self.settings, df_names=set(self.df_names))
        assigned: set[str] = set()
        items: list[tuple[str, str, object]] = []
        n_random = 0
        count_functions: dict[str, str] = {}
        kw["count_functions"] = count_functions  # filled as counting lambdas are found
        for i, a in enumerate(self.assignments):
            random_index = None
            if RANDOM_MARKER in a.formula:
                random_index = n_random
                n_random += 1
            key = counting_lambda(a.formula)
            if key is not None:  # e.g. _PERSON_COUNT: rows of a table per key, used by later rows
                count_functions[a.target] = key
                items.append((a.target, a.description, f"counts rows of a table per '{key}'"))
                continue
            if a.target.startswith("_") and a.target.isupper():
                try:
                    value, _ = fold_constant(a.formula, constants, assigned=assigned, **kw)
                except NotConstant:
                    pass  # computed when the step runs
                except NeedsColumns:
                    raise
                except FormulaError as err:
                    problems.append(f"row {i + 1} ({a.target}): {err}")
                    continue
                else:
                    constants[a.target] = value
                    assigned.discard(a.target)
                    items.append((a.target, a.description, value))
                    continue
            try:
                tr = translate(a.formula, constants, python_mode=True, assigned=assigned,
                               random_index=random_index, **kw)
                items.append((a.target, a.description, tr))
            except NeedsColumns:
                raise
            except FormulaError as err:
                problems.append(f"row {i + 1} ({a.target}): {err}")
            if a.target != "_":
                assigned.add(a.target)
        func_name = f"{_identifier(self.name)}_assign"
        doc = (f"Assignments for '{self.name}', generated by asim_core from "
               f"{', '.join(self.source_files) or 'the definition'}.")
        if problems:
            return BoundAssignments(self.name, items, None, "", self.targets, n_random)
        source = build_assignment_function(func_name, items, doc)
        func = compile_function(source, func_name, f"<{func_name}>")
        return BoundAssignments(self.name, items, func, source, self.targets, n_random)

    def _bind_new(self, tables, problems, extra) -> BoundAssignments:
        """asim_core's language: parameters and the zones' columns come in ``extra``."""
        scope_def = self.scope or LanguageScope()
        constants = dict(self.constants)
        assigned: set[str] = set()
        items: list[tuple[str, str, object]] = []
        n_random = 0
        for i, a in enumerate(self.assignments):
            random_index = None
            if RANDOM_CALL in a.formula:
                random_index = n_random
                n_random += 1
            scope = scope_def.make(constants, self.coefficients, assigned, random_index, self.settings,
                                   tables=tables, extra=extra)
            if a.target.startswith("_") and a.target.isupper():
                try:
                    value, _ = fold_formula(a.formula, scope)
                except NeedsColumns:
                    raise
                except FormulaError:
                    pass  # computed when the step runs
                else:
                    constants[a.target] = value
                    assigned.discard(a.target)
                    items.append((a.target, a.description, value))
                    continue
            try:
                items.append((a.target, a.description, compile_formula(a.formula, scope)))
            except NeedsColumns:
                raise
            except FormulaError as err:
                problems.append(f"row {i + 1} ({a.target}): {err}")
            if a.target != "_":
                assigned.add(a.target)
        func_name = f"{_identifier(self.name)}_assign"
        doc = (f"Assignments for '{self.name}', generated by asim_core from "
               f"{', '.join(self.source_files) or 'the definition'}.")
        if problems:
            return BoundAssignments(self.name, items, None, "", self.targets, n_random)
        source = build_assignment_function(func_name, items, doc)
        func = compile_function(source, func_name, f"<{func_name}>")
        return BoundAssignments(self.name, items, func, source, self.targets, n_random)

    @property
    def targets(self) -> list[str]:
        """Non-temporary targets, in order of first assignment (the outputs)."""
        out = []
        for a in self.assignments:
            if not a.target.startswith("_") and a.target not in out:
                out.append(a.target)
        return out

    def _loaded(self) -> BoundAssignments:
        if self._bound is None:
            raise ValueError(f"assignments '{self.name}' depend on the run's columns; use bind()")
        return self._bound

    @property
    def translated(self) -> list[Translated]:
        return self._loaded().translated

    def columns_used(self) -> set[str]:
        return self._loaded().columns_used()

    def skims_used(self) -> set[str]:
        return self._loaded().skims_used()

    def function(self):
        """The compiled ``f(c, s, n, ...) -> dict of assigned variables``, and its source."""
        b = self._loaded()
        return b.function, b.source


# --- coefficient cells ----------------------------------------------------------

_CELL_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)


def _parse_cell(cell: str) -> ast.Expression:
    try:
        tree = ast.parse(str(cell).strip(), mode="eval")
    except SyntaxError:
        raise FormulaError(f"coefficient cell {cell!r} is not valid") from None
    for node in ast.walk(tree):
        if not isinstance(node, _CELL_NODES):
            raise FormulaError(f"coefficient cell {cell!r} may only use names, numbers and + - * /")
    return tree


def _names_in_cell(cell: str) -> set[str]:
    return {n.id for n in ast.walk(_parse_cell(cell)) if isinstance(n, ast.Name)}


def _eval_cell(cell: str, values: dict[str, float]) -> float:
    tree = _parse_cell(cell)
    code = compile(tree, "<coefficient>", "eval")
    return float(eval(code, {"__builtins__": {}}, values))


def _identifier(name: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in name)
    return out if out and not out[0].isdigit() else f"m_{out}"
