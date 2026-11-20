"""
A component that runs a simple choice model (multinomial or nested logit)
from its ``ChoiceModelDefinition``: one choice per chooser, among a fixed
list of alternatives.

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
    """
    The chooser table plus other tables joined by a key column, e.g. tours
    joined to persons by person_id, persons to households by household_id.
    A join's key can come from the chooser table or an earlier join.

    When a column name is in several tables, the chooser table wins, then
    the earlier join; this is the order 1.x's merged tables give.
    """

    table: str
    # (other table, key column), in order of precedence
    joins: list[tuple[str, str]] = field(default_factory=list)

    def _join_index(self, table: str) -> int:
        return [t for t, _ in self.joins].index(table)

    def _source(self, col: str, available: Mapping[str, set[str]], upto: int | None = None):
        if col in available.get(self.table, set()):
            return self.table
        for other, _ in self.joins[:upto]:
            if col in available.get(other, set()):
                return other
        return None

    def columns(self, available: Mapping[str, set[str]]) -> set[str]:
        """Every column the view offers."""
        out = set(available.get(self.table, set()))
        for other, _ in self.joins:
            out |= available.get(other, set())
        return out

    def resolve(self, columns: set[str], available: Mapping[str, set[str]]) -> Needs:
        needs: dict[str, set[str]] = {self.table: set()}

        def add(col: str, upto: int | None = None) -> bool:
            src = self._source(col, available, upto)
            if src is None:
                return False
            if col in needs.get(src, set()):
                return True
            needs.setdefault(src, set()).add(col)
            if src != self.table:
                i = self._join_index(src)
                key = self.joins[i][1]
                if not add(key, i):
                    raise MissingInputError(
                        f"key '{key}' to join '{src}' not found in {self.table} or an earlier join"
                    )
            return True

        missing = [col for col in sorted(columns) if not add(col)]
        if missing:
            where = [self.table] + [t for t, _ in self.joins]
            raise MissingInputError(f"columns {missing} not found in {where}")
        return {t: sorted(c) for t, c in needs.items()}

    def _collect(self, needs: Needs, column, positions, column_at) -> dict[str, object]:
        sources: dict[str, str] = {}
        for t in [self.table] + [o for o, _ in self.joins]:
            for c in needs.get(t, []):
                sources.setdefault(c, t)
        cols: dict[str, object] = {}
        pos: dict[str, object] = {}

        def value(c):
            if c not in cols:
                t = sources[c]
                if t == self.table:
                    cols[c] = column(t, c)
                else:
                    if t not in pos:
                        pos[t] = positions(t, value(self.joins[self._join_index(t)][1]))
                    cols[c] = column_at(t, c, pos[t])
            return cols[c]

        for c in sources:
            value(c)
        return cols

    def gather(self, ctx: StepContext, needs: Needs) -> dict[str, object]:
        """The needed columns, in chooser row order (joined columns looked up, not merged)."""
        return self._collect(needs, ctx.column, ctx.positions, ctx.column_at)

    def gather_from_store(self, store, columns: set[str]) -> dict[str, object]:
        """Same as ``gather``, straight from a Store (for checks and notebooks)."""
        needs = self.resolve(columns, store.columns())
        return self._collect(
            needs,
            lambda t, c: store[t][c],
            lambda t, ids: store[t].positions(ids),
            lambda t, c, p: store[t][c][p],
        )


def simple_choice_component(
    definition: ChoiceModelDefinition,
    step: str,
    view: ChooserView,
    output_column: str,
    random_channel: str | None = None,
    shift_utilities: bool = False,
    drop_zero_terms: bool = True,
    roles: Mapping[str, list[str]] | None = None,
) -> Component:
    """
    Build a component that writes the chosen alternative's position
    (0 = first alternative) to ``view.table``.``output_column``.

    ``drop_zero_terms`` skips terms whose coefficients are all zero, as
    ActivitySim 1.x does with its legacy engine.
    """
    nests = definition.nest_tree()  # None for multinomial logit
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
        if roles is not None:  # asim_core's language: role.column names checked
            from .roles import check_roles, role_columns

            translated = [t for i, t in enumerate(definition.translated) if keep[i]]
            check_roles(step, translated, role_columns(view, roles, available))
        resolved["needs"] = view.resolve(columns, available)
        return resolved["needs"]

    def compute(cols: Mapping[str, object], n: int, engine: Engine):
        """Utilities and probabilities for ``n`` choosers, using ``engine``'s pieces."""
        with np.errstate(divide="ignore", invalid="ignore"):
            utils = engine.utilities(terms_fn(cols), coef, n)
        if nests is None:
            probs, bad = engine.probabilities(utils, shift=shift_utilities)
        else:  # 1.x uses no overflow protection for nested logit
            probs, bad, _ = engine.nested_probabilities(utils, definition.alternatives, nests)
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
        description=f"{definition.name}: {'nested' if nests else 'multinomial'} logit over "
                    f"{definition.alternatives}",
        info={
            "definition": definition,
            "generated_source": source,
            "chooser_table": view.table,
            "trace_table": view.table,
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
