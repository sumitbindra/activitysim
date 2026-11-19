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
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from .translate import FormulaError, Translated, build_function, compile_function, translate


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


class ChoiceModelDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    logit_type: Literal["MNL"] = "MNL"
    alternatives: list[str]
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
        if problems:
            raise ValueError(
                f"model '{self.name}' has {len(problems)} problem(s):\n  " + "\n  ".join(problems)
            )
        self._translated = translated
        return self

    # --- derived values -------------------------------------------------------

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
        Compile the formulas into one NumPy function ``f(c) -> list of arrays``.
        Returns (function, source code). Terms where ``term_mask`` is False are
        listed as comments but not computed.
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
