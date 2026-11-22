"""
CDAP, the coordinated daily activity pattern: each person's pattern for the
day, M (mandatory), N (non-mandatory) or H (home), chosen jointly within the
household (1.x: ``abm/models/cdap.py``, ``abm/models/util/cdap.py``).

As 1.x does it:

1. **Rank** household members (``cdap_rank``): up to two workers (full-time
   before part-time), up to three children (by person type), then others
   in a random order (one uniform draw per person, persons' streams), to
   fill five places; then by rank group and age. Only the first five
   (``MAX_HHSIZE``) are in the joint choice.
2. **Individual utilities** of M, N and H for the ranked members
   (``INDIV_AND_HHSIZE1_SPEC`` with its coefficients), the same matrix
   product as 1.x.
3. **Household choice**, one model per household size 1 to 5 (5 = five or
   more): the alternatives are all patterns (``'MNH'`` ...), utilities are
   the members' individual utilities plus interaction terms built from
   ``INTERACTION_COEFFICIENTS`` (by activity and the members' person types,
   for 2 and 3 members, or all members). One draw per household
   (households' streams).
4. **Extra members** (rank above 5): drawn from fixed relative proportions
   by person type (``FIXED_RELATIVE_PROPORTIONS_SPEC``), a second draw on
   persons' streams.
5. ``persons.cdap_activity`` (a category with '', M, N, H), then the
   settings' ``annotate_persons`` and ``annotate_households``.

The interaction spec, the chooser table per household size and the
interaction codes (``p1_p3 == 13``: members 1 and 3 have person types 1
and 3, in either order) follow 1.x's code, with pandas where 1.x's results
depend on pandas' sorting and merging.

Not supported yet: ``ADD_JOINT_TOUR_UTILITY`` (none of the examples uses it).
"""

from __future__ import annotations

import ast
import itertools
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import Component, MissingInputError, StepContext
from ..logit import FormulaValues
from ..spec.definition import ChoiceModelDefinition, Term
from ..spec.legacy import find_config_file, load_choice_model, load_top_settings, read_settings, read_yaml
from ..spec.translate import INDEX
from . import annotate
from .runtime import TableLookup
from .simple_choice import ChooserView

STEP = "cdap_simulate"
OUTPUT = "cdap_activity"
CATEGORIES = ["", "M", "N", "H"]  # 1.x's category for cdap_activity
ACTIVITIES = ["M", "N", "H"]
MAX_HHSIZE = 5
MAX_INTERACTION_CARDINALITY = 3
# 1.x's persons_merged
VIEW = ChooserView("persons", [
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])
BASE_COLUMNS = {"household_id", "ptype", "age", "hhsize"}
HANDLED = ["preprocessor", "annotate_persons", "annotate_households"]


SPEC_BUILDERS = {"numpy": None, "pandas": None}  # filled below: name -> builder


def cdap_simulate(configs_dirs: Sequence[str | Path], settings_file: str = "cdap.yaml",
                  interaction_specs: str = "numpy") -> Component:
    """
    ``interaction_specs``: how the interaction tables are built, ``"numpy"``
    (default) or ``"pandas"`` (1.x's algorithm, row by row, 2.4 s more per
    process); the same tables either way.
    """
    if interaction_specs not in SPEC_BUILDERS:
        raise ValueError(f"interaction_specs must be one of {sorted(SPEC_BUILDERS)}")
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_settings(configs_dirs, settings_file)
    top = load_top_settings(configs_dirs)
    if settings.get("ADD_JOINT_TOUR_UTILITY"):
        raise NotImplementedError(f"{settings_file}: ADD_JOINT_TOUR_UTILITY")
    person_type_map = settings.get("PERSON_TYPE_MAP") or {}
    if not {"WORKER", "CHILD"} <= set(person_type_map):
        raise ValueError(f"{settings_file}: PERSON_TYPE_MAP needs WORKER and CHILD")

    indiv = load_choice_model(configs_dirs, settings_file, name="cdap_indiv", handled=HANDLED,
                              spec_key="INDIV_AND_HHSIZE1_SPEC")
    if indiv.alternatives != ACTIVITIES:
        raise ValueError(f"{settings_file}: the individual spec's alternatives must be {ACTIVITIES}, "
                         f"not {indiv.alternatives}")
    coef = indiv.coefficient_matrix()
    keep = ~(coef == 0).all(axis=1)  # 1.x drops all-zero rows (eval_coefficients)
    indiv_coef = coef[keep]
    indiv_fn, indiv_source = indiv.utility_function(keep)
    indiv_columns = indiv.columns_used(keep)

    interaction = read_interaction_coefficients(
        find_config_file(configs_dirs, str(settings.get("INTERACTION_COEFFICIENTS",
                                                        "cdap_interaction_coefficients.csv"))),
        {k: c.value for k, c in indiv.coefficients.items()})
    build = SPEC_BUILDERS[interaction_specs]
    specs = {k: build(interaction, k) for k in range(2, MAX_HHSIZE + 1)}
    hh_models = {k: (list(spec.index), spec.astype(np.float64).values, list(spec.columns))
                 for k, spec in specs.items()}

    fixed_path = find_config_file(configs_dirs, str(settings.get("FIXED_RELATIVE_PROPORTIONS_SPEC",
                                                                 "cdap_fixed_relative_proportions.csv")))
    fixed_spec = read_model_spec(fixed_path)
    if list(fixed_spec.columns) != ACTIVITIES:
        raise ValueError(f"{fixed_path.name}: columns must be {ACTIVITIES}")
    fixed = ChoiceModelDefinition(
        name="cdap_fixed_relative_proportions", alternatives=ACTIVITIES, constants=indiv.constants,
        coefficients={},
        terms=[Term(formula=e.split(" # (")[0], coefficients={a: "1" for a in ACTIVITIES})
               for e in fixed_spec.index],
        source_files=[fixed_path.name])
    fixed_fn, fixed_source = fixed.utility_function(np.ones(len(fixed_spec), dtype=bool))
    fixed_coef = fixed_spec.values  # relative proportions: not rounded to 32 bits (no coefficients file)
    fixed_columns = fixed.columns_used(np.ones(len(fixed_spec), dtype=bool))

    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, settings.get("preprocessor"),
                                                      dict(indiv.constants), top, STEP, "preprocessor")
    annotations = annotate.legacy_annotations(configs_dirs, settings, top, STEP)
    shift = not top.get("skip_failed_choices", True)  # 1.x's overflow protection
    resolved: dict = {}

    def resolve_needs(available):
        wanted = set(BASE_COLUMNS) | indiv_columns | fixed_columns
        extra: dict[str, set[str]] = {}
        if pre_spec is not None:
            bound = pre_spec.bind(VIEW.columns(available), {t: available.get(t, set()) for t in pre_tables})
            if bound.n_random or bound.skim_uses():
                raise NotImplementedError(f"{STEP}: preprocessor with random draws or skims")
            resolved["pre"] = bound
            wanted = (wanted - set(bound.targets)) | (bound.columns_used() - {INDEX})
            for t, c in bound.tables_used():
                extra.setdefault(t, set()).add(c)
        missing = sorted(c for c in BASE_COLUMNS if c not in VIEW.columns(available))
        if missing:
            raise MissingInputError(f"persons (with households) lack {missing}")
        resolved["view_needs"] = view_needs = VIEW.resolve(wanted, available)
        bound_ann, ann_needs, ann_provides = annotate.bind(annotations, available, {"persons": {OUTPUT}})
        resolved["annotations"] = bound_ann
        needs = {t: set(c) for t, c in view_needs.items()}
        for src in (extra, ann_needs):
            for t, c in src.items():
                needs[t] = needs.get(t, set()) | set(c)
        needs.setdefault("persons", set()).add("household_id")
        return {t: sorted(c) for t, c in needs.items()}

    def run(ctx: StepContext):
        ids = ctx.ids("persons")
        cols = VIEW.gather(ctx, resolved["view_needs"])
        pre = resolved.get("pre")
        if pre is not None:
            from .annotate import _update

            c = {k: cols[k] for k in pre.columns_used() - {INDEX}}
            c[INDEX] = ids
            with np.errstate(all="ignore"):
                v = pre.function(c, None, len(ids), TableLookup(ctx), None, {})
            cols = dict(cols)
            for target in pre.targets:
                cols[target] = _update(cols.get(target), np.asarray(v[target]))

        hh = np.asarray(cols["household_id"])
        hhsize = np.asarray(cols["hhsize"])
        # 1. ranks: one uniform per person (1.x: random_for_df(persons))
        order_draws = ctx.random.uniform("persons", ids)[:, 0]
        rank = assign_cdap_rank(ids, hh, np.asarray(cols["ptype"]), np.asarray(cols["age"]), person_type_map,
                                order_draws)
        ranked = np.flatnonzero(rank <= MAX_HHSIZE)

        # 2. individual utilities of the ranked members
        sub = {k: (v[ranked] if np.ndim(v) else v) for k, v in cols.items()}
        with np.errstate(divide="ignore", invalid="ignore"):
            indiv_utils = ctx.engine.utilities(FormulaValues(indiv_fn, sub), indiv_coef, len(ranked))
        indiv_df = pd.DataFrame(indiv_utils, index=pd.Index(ids[ranked], name="person_id"), columns=ACTIVITIES)
        indiv_df["household_id"] = hh[ranked]
        indiv_df["ptype"] = np.asarray(cols["ptype"])[ranked]
        indiv_df["cdap_rank"] = rank[ranked]
        indiv_df["hhsize"] = hhsize[ranked]

        # 3. one joint choice per household, by household size
        patterns = []
        for k in range(1, MAX_HHSIZE + 1):
            if k == 1:
                one = indiv_df[indiv_df.hhsize == 1]
                hh_ids = one.household_id.to_numpy()
                utils, alts = one[ACTIVITIES].to_numpy(), ACTIVITIES
            else:
                choosers = hh_choosers(indiv_df, k)
                hh_ids = choosers.index.to_numpy()
                if len(hh_ids) == 0:
                    continue
                exprs, matrix, alts = hh_models[k]
                terms = [_spec_term(choosers, e) for e in exprs]
                utils = ctx.engine.utilities(terms, matrix, len(choosers))
            if len(hh_ids) == 0:
                continue
            probs, bad = ctx.engine.probabilities(utils, shift=shift)
            if bad.any():
                raise ValueError(f"{STEP}: {int(bad.sum())} households of size {k} have no usable "
                                 f"probabilities, e.g. {hh_ids[bad][:5].tolist()}")
            rands = ctx.random.uniform("households", hh_ids)
            choice = ctx.engine.choose(probs, rands)
            patterns.append(pd.Series(np.asarray(alts, dtype=object)[choice], index=hh_ids))
            _trace(ctx, k, hh_ids, utils, probs, rands, choice, alts)
        pattern = pd.concat(patterns) if patterns else pd.Series(dtype=object)

        # unpack: member with rank r takes the household pattern's r-th letter
        activity = np.full(len(ids), np.nan, dtype=object)
        p = pattern.reindex(hh[ranked]).to_numpy()
        r = rank[ranked]
        activity[ranked] = [s[i - 1] if isinstance(s, str) and i <= len(s) else np.nan for s, i in zip(p, r)]

        # 4. extra members: fixed relative proportions
        extra_rows = np.flatnonzero(rank > MAX_HHSIZE)
        if len(extra_rows):
            sub = {k: (v[extra_rows] if np.ndim(v) else v) for k, v in cols.items()}
            values = ctx.engine.utilities(FormulaValues(fixed_fn, sub), fixed_coef, len(extra_rows))
            proportions = pd.DataFrame(values, columns=ACTIVITIES)
            probs = proportions.div(proportions.sum(axis=1), axis=0).to_numpy()
            rands = ctx.random.uniform("persons", ids[extra_rows])
            activity[extra_rows] = np.asarray(ACTIVITIES, dtype=object)[ctx.engine.choose(probs, rands)]

        missing = pd.isna(activity)
        if missing.any():
            ctx.warn(f"{int(missing.sum())} persons got no activity pattern: their household's hhsize "
                     f"differs from its number of persons (as in 1.x, they are left missing)")
        result = pd.Categorical(activity, categories=CATEGORIES)

        # 5. annotations
        made = {"persons": {OUTPUT: result}}
        annotate.run(ctx, resolved["annotations"], made)
        return {t: dict(v) for t, v in made.items()}

    provides = {"persons": [OUTPUT]}
    for a in annotations:
        for target in a.spec.targets:
            if target not in provides.setdefault(a.table, []):
                provides[a.table].append(target)
    return Component(
        STEP, run, provides=provides, resolve_needs=resolve_needs,
        description="coordinated daily activity pattern (M, N, H) per person, chosen per household",
        info={"trace_table": "households",
              "generated_source": "# individual utilities\n" + indiv_source
                                  + "\n# fixed relative proportions\n" + fixed_source,
              "interaction_specs": specs},
    )


# --- 1.x's pieces, kept as 1.x computes them --------------------------------------------


def assign_cdap_rank(person_ids, household_id, ptype, age, person_type_map, random_order) -> np.ndarray:
    """1.x's assign_cdap_rank (same pandas sorts, so ties go the same way)."""
    persons = pd.DataFrame({"household_id": household_id, "ptype": ptype, "age": age},
                           index=pd.Index(person_ids, name="person_id"))
    RANK_WORKER, RANK_CHILD, RANK_BACKFILL, RANK_UNASSIGNED = 1, 2, 3, 9
    persons["cdap_rank"] = RANK_UNASSIGNED
    workers = (persons.loc[persons.ptype.isin(person_type_map["WORKER"]), ["household_id", "ptype"]]
               .sort_values(by=["household_id", "ptype"], ascending=[True, True])
               .groupby("household_id").head(2))
    persons.loc[workers.index, "cdap_rank"] = RANK_WORKER
    children = (persons.loc[persons.ptype.isin(person_type_map["CHILD"]), ["household_id", "ptype", "age"]]
                .sort_values(by=["household_id", "ptype"], ascending=[True, True])
                .groupby("household_id").head(3))
    persons.loc[children.index, "cdap_rank"] = RANK_CHILD
    others = persons[["household_id", "cdap_rank"]].copy()
    others["random_order"] = random_order
    others = (others.sort_values(by=["household_id", "random_order"], ascending=[True, True])
              .groupby("household_id").head(MAX_HHSIZE))
    persons.loc[others[others.cdap_rank == RANK_UNASSIGNED].index, "cdap_rank"] = RANK_BACKFILL
    p = persons[["household_id", "cdap_rank", "age"]].sort_values(
        by=["household_id", "cdap_rank", "age"], ascending=[True, True, True])
    sizes = p.groupby("household_id").size().to_numpy()
    p["cdap_rank"] = np.concatenate([np.arange(1, n + 1) for n in sizes]) if len(sizes) else []
    persons["cdap_rank"] = p["cdap_rank"]
    return persons["cdap_rank"].to_numpy()


def read_model_spec(path) -> pd.DataFrame:
    """1.x's read_model_spec: expressions as the index (duplicates numbered), blanks as 0."""
    spec = pd.read_csv(path, comment="#")
    spec = spec.dropna(subset=["Expression"])
    if "Description" in spec.columns:
        spec = spec.drop("Description", axis=1)
    spec = spec.set_index("Expression").fillna(0)
    _uniquify_index(spec)
    if "Label" in spec.columns:
        spec = spec.drop("Label", axis=1)
    return spec


def _uniquify_index(spec: pd.DataFrame) -> None:
    """1.x's uniquify_spec_index: a repeated expression gets ' # (n)' appended."""
    seen: dict = {}
    out = []
    for expr in spec.index:
        key, n = expr, 1
        while key in seen:
            n += 1
            key = f"{expr} # ({n})"
        seen[key] = expr
        out.append(key)
    name = spec.index.name
    spec.index = out
    spec.index.name = name


def read_interaction_coefficients(path, coefficient_values: dict) -> pd.DataFrame:
    """The interaction coefficients file, coefficient names replaced by values, then 1.x's
    preprocess_interaction_coefficients (cardinality, wildcards, slugs)."""
    df = pd.read_csv(path, comment="#")
    mapped = df.coefficient.map(coefficient_values)
    values = pd.to_numeric(df.coefficient.where(mapped.isnull(), mapped), errors="coerce").astype(float)
    if values.isnull().any():
        raise ValueError(f"{Path(path).name}: coefficients that are neither numbers nor names in the "
                         f"coefficients file: {df.coefficient[values.isnull()].tolist()}")
    df.coefficient = values
    if not df.activity.isin(ACTIVITIES).all():
        raise ValueError(f"{Path(path).name}: activity must be M, N or H "
                         f"(rows {df.index[~df.activity.isin(ACTIVITIES)].tolist()})")
    df["cardinality"] = df.interaction_ptypes.astype(str).str.len()
    wildcards = df.interaction_ptypes == df.cardinality.map(lambda x: x * "*")
    df.loc[wildcards, "interaction_ptypes"] = ""
    df["slug"] = df.activity * df.cardinality + df.interaction_ptypes.astype(str)
    return df


def build_cdap_spec(interaction: pd.DataFrame, hhsize: int) -> pd.DataFrame:
    """1.x's build_cdap_spec (without joint tours): rows carrying each member's individual
    utilities, then the interaction rows; the coefficients put in for their slugs."""
    hhsize = min(hhsize, MAX_HHSIZE)
    alternatives = ["".join(t) for t in itertools.product("HMN", repeat=hhsize)]
    spec = pd.DataFrame(columns=["Expression"] + alternatives)
    for pnum in range(1, hhsize + 1):
        for activity in ACTIVITIES:
            i = len(spec)
            spec.loc[i, "Expression"] = f"{activity}_p{pnum}"
            spec.loc[i, [a for a in alternatives if a[pnum - 1] == activity]] = 1
    for row in interaction[interaction.cardinality <= hhsize].itertuples():
        if not row.interaction_ptypes:  # wildcard: applies when every member is in the interaction
            if row.slug in alternatives:
                spec.loc[len(spec), ["Expression", row.slug]] = ["1", row.slug]
            continue
        if not 0 <= row.cardinality <= MAX_INTERACTION_CARDINALITY:
            raise ValueError(f"bad interaction cardinality {row.cardinality} for {row.slug}: "
                             f"interactions involve at most {MAX_INTERACTION_CARDINALITY} person types")
        for tup in itertools.combinations(range(1, hhsize + 1), row.cardinality):
            column = f"ptype_p{tup[0]}" if row.cardinality == 1 else "_".join(f"p{p}" for p in tup)
            expression = f"{column}=={row.interaction_ptypes}"
            columns = [a for a in alternatives if all(a[p - 1] == row.activity for p in tup)]
            existing = spec["Expression"] == expression
            if existing.any():
                spec.loc[existing, columns] = row.slug
                spec.loc[existing, "Expression"] = expression
            else:
                i = len(spec)
                spec.loc[i, columns] = row.slug
                spec.loc[i, "Expression"] = expression
    spec = spec.set_index("Expression")
    _uniquify_index(spec)
    d = interaction.set_index("slug")["coefficient"].to_dict()
    for c in spec.columns:
        spec[c] = spec[c].map(lambda x: d.get(x, x or 0.0)).fillna(0)
    return spec


def build_cdap_spec_numpy(interaction: pd.DataFrame, hhsize: int) -> pd.DataFrame:
    """
    The same table as ``build_cdap_spec`` (1.x's algorithm, kept as the
    reference), built on NumPy arrays instead of one pandas row at a time:
    about 2.4 s less per process for MTC's household sizes 2 to 5.
    ``tests/test_cdap.py`` checks the two give the same expressions, values
    and memory layout for every example's coefficients.
    """
    hhsize = min(hhsize, MAX_HHSIZE)
    alternatives = ["".join(t) for t in itertools.product("HMN", repeat=hhsize)]
    letters = np.array([list(a) for a in alternatives])  # (alternatives, members)
    expressions: list[str] = []
    cells: list[np.ndarray] = []  # per row: 1, a slug, or None (empty), per alternative
    where: dict[str, int] = {}
    for pnum in range(1, hhsize + 1):
        for activity in ACTIVITIES:
            row = np.full(len(alternatives), None, dtype=object)
            row[letters[:, pnum - 1] == activity] = 1
            expressions.append(f"{activity}_p{pnum}")
            cells.append(row)
    for r in interaction[interaction.cardinality <= hhsize].itertuples():
        if not r.interaction_ptypes:  # wildcard: applies when every member is in the interaction
            if r.slug in alternatives:
                row = np.full(len(alternatives), None, dtype=object)
                row[alternatives.index(r.slug)] = r.slug
                expressions.append("1")
                cells.append(row)
            continue
        if not 0 <= r.cardinality <= MAX_INTERACTION_CARDINALITY:
            raise ValueError(f"bad interaction cardinality {r.cardinality} for {r.slug}: "
                             f"interactions involve at most {MAX_INTERACTION_CARDINALITY} person types")
        for tup in itertools.combinations(range(1, hhsize + 1), r.cardinality):
            column = f"ptype_p{tup[0]}" if r.cardinality == 1 else "_".join(f"p{p}" for p in tup)
            expression = f"{column}=={r.interaction_ptypes}"
            match = np.all(letters[:, [p - 1 for p in tup]] == r.activity, axis=1)
            i = where.get(expression)
            if i is None:  # 1.x adds a row; an expression seen before gets these alternatives too
                where[expression] = i = len(cells)
                expressions.append(expression)
                cells.append(np.full(len(alternatives), None, dtype=object))
            cells[i][match] = r.slug
    d = interaction.set_index("slug")["coefficient"].to_dict()  # a repeated slug: the last one, as 1.x
    values = np.zeros((len(cells), len(alternatives)), dtype=np.float64, order="F")
    for i, row in enumerate(cells):
        for j in np.flatnonzero(row != None):  # noqa: E711 (elementwise test)
            x = row[j]
            values[i, j] = d[x] if isinstance(x, str) else x
    spec = pd.DataFrame(values, columns=alternatives)
    spec.index = pd.Index(expressions, name="Expression")
    _uniquify_index(spec)
    return spec


def hh_choosers(indiv: pd.DataFrame, hhsize: int) -> pd.DataFrame:
    """1.x's hh_choosers: one row per household of this size (5: five or more), each ranked
    member's utilities and person type, and the interaction codes (p1_p3 == 13)."""
    merge_cols = ["household_id", "ptype", "M", "N", "H"]
    include = indiv.hhsize == hhsize if hhsize < MAX_HHSIZE else indiv.hhsize >= MAX_HHSIZE

    def add_pn(cols, pnum):
        return [c if c == "household_id" else f"{c}_p{pnum}" for c in cols]

    choosers = indiv.loc[include & (indiv.cdap_rank == 1), merge_cols]
    choosers.columns = add_pn(merge_cols, 1)
    for pnum in range(2, hhsize + 1):
        rhs = indiv.loc[include & (indiv.cdap_rank == pnum), merge_cols]
        rhs.columns = add_pn(merge_cols, pnum)
        choosers = pd.merge(left=choosers, right=rhs, on="household_id")
    choosers = choosers.set_index("household_id")
    for pnum in range(1, hhsize + 1):
        pn = add_pn(ACTIVITIES, pnum)
        choosers[pn] = choosers[pn].astype(float)
    for i in range(2, min(hhsize, MAX_INTERACTION_CARDINALITY) + 1):
        for tup in itertools.combinations(range(1, hhsize + 1), i):
            choosers["_".join(f"p{p}" for p in tup)] = _interaction_code(
                [choosers[f"ptype_p{p}"].to_numpy() for p in tup])
    choosers["hhsize"] = hhsize
    return choosers


def _interaction_code(ptypes: list[np.ndarray]) -> np.ndarray:
    """1.x's add_interaction_column: the members' person types as text, digits sorted, as an
    integer (types 1 and 3 in either order: 13). Done with numbers when types are single digits."""
    stacked = np.stack([np.asarray(p) for p in ptypes])
    if stacked.size == 0:
        return np.zeros(stacked.shape[1], dtype=np.int64)
    if stacked.min() >= 0 and stacked.max() <= 9:
        digits = np.sort(stacked.astype(np.int64), axis=0)
        code = np.zeros(stacked.shape[1], dtype=np.int64)
        for row in digits:
            code = code * 10 + row
        return code
    text = pd.Series(ptypes[0]).astype(str)
    for p in ptypes[1:]:
        text = text + pd.Series(p).astype(str)
    return text.apply(lambda x: "".join(sorted(x))).astype(int).to_numpy()


_CODE = re.compile(r"^(\w+)==(.+)$")


def _spec_term(choosers: pd.DataFrame, expression: str):
    """The values of one generated spec expression (1.x evaluates them with pandas.eval):
    a member's utility (M_p1), an interaction test (p1_p3==13, ptype_p1==1) or 1."""
    expr = expression.split(" # (")[0].strip()
    if expr in choosers.columns:
        return choosers[expr].to_numpy()
    if expr == "1":
        return 1
    m = _CODE.match(expr)
    if m and m.group(1) in choosers.columns:
        return choosers[m.group(1)].to_numpy() == ast.literal_eval(m.group(2).strip())
    raise ValueError(f"unexpected CDAP spec expression {expression!r}")


def _trace(ctx, hhsize, hh_ids, utils, probs, rands, choice, alts):
    rows = ctx.tracer.targets("households", hh_ids)
    if not len(rows):
        return
    out = [[hh_ids[r], hhsize] + list(utils[r]) + list(probs[r]) + [rands[r, 0], alts[choice[r]]] for r in rows]
    header = (["household_id", "hhsize"] + [f"utility_{a}" for a in alts] + [f"probability_{a}" for a in alts]
              + ["random_draw", "pattern"])
    ctx.tracer.write(ctx.step, f"hhsize{hhsize}", header, out)


SPEC_BUILDERS.update(numpy=build_cdap_spec_numpy, pandas=build_cdap_spec)
