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

from .translate import (
    FormulaError,
    Translated,
    build_assignment_function,
    build_function,
    compile_function,
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


class ChoiceModelDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    logit_type: Literal["MNL", "NL"] = "MNL"
    alternatives: list[str]
    nests: Nest | None = None
    constants: dict[str, Any] = {}
    terms: list[Term]
    coefficients: dict[str, Coefficient]
    source_files: list[str] = []

    _translated: list[Translated] = PrivateAttr(default_factory=list)

    # --- checks ---------------------------------------------------------------

    @model_validator(mode="after")
    def _check(self) -> ChoiceModelDefinition:
        problems = []
        alts = set(self.alternatives)
        translated = []
        for i, term in enumerate(self.terms):
            where = f"term {i + 1} ({term.label or term.formula})"
            for alt, cell in term.coefficients.items():
                if alt not in alts:
                    problems.append(f"{where}: unknown alternative '{alt}'")
                try:
                    for name in _names_in_cell(cell):
                        if name not in self.coefficients:
                            problems.append(f"{where}, {alt}: unknown coefficient '{name}'")
                except FormulaError as err:
                    problems.append(f"{where}, {alt}: {err}")
            try:
                translated.append(translate(term.formula, self.constants))
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
                if isinstance(node.coefficient, str) and node.coefficient not in self.coefficients:
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

    def nest_tree(self) -> NestTree | None:
        """The nests with coefficient names replaced by values (64-bit, as 1.x does)."""
        if self.logit_type != "NL":
            return None
        values = {k: c.value for k, c in self.coefficients.items()}

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

    def coefficient_matrix(self, float32_rounding: bool = True) -> np.ndarray:
        """
        Coefficients as a (terms x alternatives) array; 0 where a term does not
        apply. ActivitySim 1.x rounds coefficients to 32-bit floats before using
        them, so that is the default here too, for exact matching.
        """
        values = {k: c.value for k, c in self.coefficients.items()}
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


class AssignmentSpec(BaseModel):
    """
    A list of assignments evaluated in order, as in ActivitySim 1.x's
    preprocessor, annotation and accessibility files. Targets starting with
    ``_`` are temporary; ``_`` alone is evaluated and thrown away. Any target,
    once assigned, can be used by later rows by name.

    Loading checks every formula, with each row seeing only the variables
    assigned before it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    assignments: list[Assignment]
    constants: dict[str, Any] = {}
    # bare function names formulas may call, mapped to NumPy
    functions: dict[str, str] = {"exp": "np.exp", "log": "np.log"}
    # names that look up skims, e.g. skim_od[('SOVTOLL_TIME', 'AM')]
    skim_accessors: list[str] = []
    source_files: list[str] = []

    _translated: list[Translated] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> AssignmentSpec:
        problems = []
        translated = []
        assigned: set[str] = set()
        for i, a in enumerate(self.assignments):
            try:
                translated.append(translate(
                    a.formula, self.constants, python_mode=True, assigned=assigned,
                    functions=self.functions, skim_accessors=set(self.skim_accessors),
                ))
            except FormulaError as err:
                problems.append(f"row {i + 1} ({a.target}): {err}")
                translated.append(None)
            if a.target != "_":
                assigned.add(a.target)
        if not self.targets:
            problems.append("no non-temporary targets: nothing would be produced")
        if problems:
            raise ValueError(
                f"assignments '{self.name}' have {len(problems)} problem(s):\n  " + "\n  ".join(problems)
            )
        self._translated = translated
        return self

    @property
    def targets(self) -> list[str]:
        """Non-temporary targets, in order of first assignment (the outputs)."""
        out = []
        for a in self.assignments:
            if not a.target.startswith("_") and a.target not in out:
                out.append(a.target)
        return out

    @property
    def translated(self) -> list[Translated]:
        return self._translated

    def columns_used(self) -> set[str]:
        return set().union(*(t.columns for t in self._translated)) if self._translated else set()

    def skims_used(self) -> set[str]:
        return set().union(*(t.skims for t in self._translated)) if self._translated else set()

    def function(self):
        """Compile into ``f(c, s) -> dict of assigned variables``. Returns (function, source)."""
        func_name = f"{_identifier(self.name)}_assign"
        items = [(a.target, a.description, tr) for a, tr in zip(self.assignments, self._translated)]
        doc = f"Assignments for '{self.name}', generated by asim_core from {', '.join(self.source_files) or 'the definition'}."
        source = build_assignment_function(func_name, items, doc)
        return compile_function(source, func_name, f"<{func_name}>"), source


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
