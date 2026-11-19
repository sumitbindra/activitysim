"""
A component that runs a simple multinomial logit choice model from its
``ChoiceModelDefinition``: one choice per chooser, among a fixed list of
alternatives.

Choosers are seen through a ``ChooserView``: the chooser table plus other
tables joined by a key column (e.g. households joined to land_use by
home_zone_id). Joined columns are looked up only when a formula uses them;
no merged copy of the tables is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .. import logit
from ..component import Component, MissingInputError, Needs, StepContext
from ..seams import Engine
from ..spec.definition import ChoiceModelDefinition


@dataclass
class ChooserView:
    table: str
    # (other table, key column in the chooser table), in order of precedence:
    # a column found in the chooser table wins, then the first joined table
    joins: list[tuple[str, str]] = field(default_factory=list)

    def resolve(self, columns: set[str], available: Mapping[str, set[str]]) -> Needs:
        needs: dict[str, set[str]] = {self.table: set()}
        missing = []
        for col in sorted(columns):
            if col in available.get(self.table, set()):
                needs[self.table].add(col)
                continue
            for other, key in self.joins:
                if col in available.get(other, set()):
                    needs.setdefault(other, set()).add(col)
                    needs[self.table].add(key)
                    break
            else:
                missing.append(col)
        if missing:
            where = [self.table] + [t for t, _ in self.joins]
            raise MissingInputError(f"columns {missing} not found in {where}")
        return {t: sorted(c) for t, c in needs.items()}

    def gather(self, ctx: StepContext, needs: Needs) -> dict[str, object]:
        cols = {c: ctx.column(self.table, c) for c in needs.get(self.table, [])}
        for other, key in self.joins:
            if other in needs:
                keys = ctx.column(self.table, key)
                for c in needs[other]:
                    if c not in cols:
                        cols[c] = ctx.column_for(other, c, keys)
        return cols

    def gather_from_store(self, store, columns: set[str]) -> dict[str, object]:
        """Same as ``gather``, straight from a Store (for checks and notebooks)."""
        needs = self.resolve(columns, store.columns())
        cols = {c: store[self.table][c] for c in needs.get(self.table, [])}
        for other, key in self.joins:
            if other in needs:
                pos = store[other].positions(store[self.table][key])
                for c in needs[other]:
                    if c not in cols:
                        cols[c] = store[other][c][pos]
        return cols


def simple_choice_component(
    definition: ChoiceModelDefinition,
    step: str,
    view: ChooserView,
    output_column: str,
    random_channel: str | None = None,
    shift_utilities: bool = False,
    drop_zero_terms: bool = True,
) -> Component:
    """
    Build a component that writes the chosen alternative's position
    (0 = first alternative) to ``view.table``.``output_column``.

    ``drop_zero_terms`` skips terms whose coefficients are all zero, as
    ActivitySim 1.x does with its legacy engine.
    """
    coef = definition.coefficient_matrix()
    keep = ~(coef == 0).all(axis=1) if drop_zero_terms else np.ones(len(coef), dtype=bool)
    coef = coef[keep]
    terms_fn, source = definition.utility_function(keep)
    labels = [t.label or f"term_{i + 1}" for i, t in enumerate(definition.terms) if keep[i]]
    formulas = [t.formula for i, t in enumerate(definition.terms) if keep[i]]
    columns = definition.columns_used(keep)
    channel = random_channel or view.table
    resolved: dict[str, Needs] = {}

    def resolve_needs(available: Mapping[str, set[str]]) -> Needs:
        resolved["needs"] = view.resolve(columns, available)
        return resolved["needs"]

    def compute(cols: Mapping[str, object], n: int, engine: Engine):
        """Utilities and probabilities for ``n`` choosers, using ``engine``'s pieces."""
        with np.errstate(divide="ignore", invalid="ignore"):
            utils = engine.utilities(terms_fn(cols), coef, n)
        probs, bad = engine.probabilities(utils, shift=shift_utilities)
        return utils, probs, bad

    def run(ctx: StepContext):
        ids = ctx.ids(view.table)
        cols = view.gather(ctx, resolved["needs"])
        utils, probs, bad = compute(cols, len(ids), ctx.engine)
        if bad.any():
            raise ValueError(
                f"{step}: {int(bad.sum())} choosers have no usable probabilities, "
                f"e.g. IDs {ids[bad][:5].tolist()}"
            )
        rands = ctx.random.uniform(channel, ids)
        choices = ctx.engine.choose(probs, rands)
        rows = ctx.tracer.targets(view.table, ids)
        if len(rows):
            _trace(ctx, ids, rows, cols, terms_fn, labels, formulas, coef, utils, probs,
                   rands, choices, definition.alternatives)
        return {view.table: {output_column: choices}}

    return Component(
        name=step,
        run=run,
        provides={view.table: [output_column]},
        resolve_needs=resolve_needs,
        description=f"{definition.name}: multinomial logit over {definition.alternatives}",
        info={
            "definition": definition,
            "generated_source": source,
            "chooser_table": view.table,
            "view": view,
            "columns": columns,
            "compute": compute,  # for utility- and probability-level checks
        },
    )


def _trace(ctx, ids, rows, cols, terms_fn, labels, formulas, coef, utils, probs, rands,
           choices, alts):
    # re-evaluate the formulas for the traced rows only, so tracing works the
    # same whichever implementation computed the utilities
    subset = {c: (v[rows] if np.ndim(v) else v) for c, v in cols.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        values = logit.term_values(terms_fn(subset), len(rows), n_terms=len(labels))
    term_rows = []
    for j, r in enumerate(rows):
        for k, (label, formula) in enumerate(zip(labels, formulas)):
            term_rows.append([ids[r], label, formula, values[k, j]] + list(values[k, j] * coef[k]))
    ctx.tracer.write(ctx.step, "terms", ["id", "label", "formula", "value"] + alts, term_rows)
    choice_rows = []
    for r in rows:
        choice_rows.append(
            [ids[r]] + list(utils[r]) + list(probs[r]) + [rands[r, 0], choices[r], alts[choices[r]]]
        )
    header = (["id"] + [f"utility_{a}" for a in alts] + [f"probability_{a}" for a in alts]
              + ["random_draw", "choice_position", "choice"])
    ctx.tracer.write(ctx.step, "choices", header, choice_rows)
