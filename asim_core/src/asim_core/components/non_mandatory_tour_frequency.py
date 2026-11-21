"""
Non-mandatory tour frequency: how many escort, shopping, maintenance,
discretionary, eating-out and social tours each active person makes (1.x:
``abm/models/non_mandatory_tour_frequency.py``).

As 1.x:

1. Choosers: persons with a mandatory or non-mandatory day
   (``cdap_activity`` M or N), annotated by the preprocessor (with their
   longest free stretch, ``person_max_window``, and their household's joint
   tours).
2. One model per person type (``SPEC_SEGMENTS``), each over **all**
   alternatives of ``non_mandatory_tour_frequency_alternatives.csv`` (1.x's
   ``interaction_simulate``: every chooser with every alternative, the
   alternatives' columns and ``tot_tours`` next to the chooser's).
3. ``persons.non_mandatory_tour_frequency``: the chosen alternative's row
   number (16-bit; the first alternative for everyone else).
4. **Extension** (1.x's ``extend_tour_counts``): a person with 1 to 4 tours
   whose count of a type is that type's most, may get one or two more, drawn
   from ``non_mandatory_tour_frequency_extension_probs.csv`` by person type
   and whether they have mandatory and joint tours: one draw per type.
5. The tours (1.x's ``process_non_mandatory_tours``), with 1.x's IDs, added
   after the existing tours; then ``annotate_persons``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import Component, MissingInputError, StepContext
from ..logit import FormulaValues
from ..spec.legacy import (
    find_config_file,
    load_global_constants,
    load_interaction_model,
    load_top_settings,
    read_settings,
)
from ..spec.translate import INDEX
from ..store import Table
from . import annotate
from .runtime import TableLookup, plain
from .simple_choice import ChooserView
from .tours import canonical_tour_channels, create_tours, extend_table, set_tour_index

STEP = "non_mandatory_tour_frequency"
OUTPUT = "non_mandatory_tour_frequency"
VIEW = ChooserView("persons", [  # 1.x's persons_merged
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])
TOUR_COLUMNS = ["person_id", "tour_type", "tour_type_count", "tour_type_num", "tour_num", "tour_count",
                "tour_category", "number_of_participants", "household_id", "origin"]
JOIN_COLUMNS = ["ptype", "has_mandatory_tour", "has_joint_tour"]  # 1.x's extend_tour_counts
PROBABILITY_COLUMNS = ["0_tours", "1_tours", "2_tours"]


def non_mandatory_tour_frequency(configs_dirs: Sequence[str | Path],
                                 settings_file: str = "non_mandatory_tour_frequency.yaml",
                                 max_rows: int = 2_000_000) -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_settings(configs_dirs, settings_file)
    top = load_top_settings(configs_dirs)
    if settings.get("alts_preprocessor"):
        raise NotImplementedError(f"{settings_file}: alts_preprocessor")
    segment_col = settings.get("SEGMENT_COL", "ptype")
    segments = settings.get("SPEC_SEGMENTS") or []
    if not segments:
        raise ValueError(f"{settings_file}: SPEC_SEGMENTS is required")
    shift = not top.get("skip_failed_choices", True)

    alts = pd.read_csv(find_config_file(configs_dirs, "non_mandatory_tour_frequency_alternatives.csv"),
                       comment="#")
    if "tot_tours" not in alts.columns:
        alts["tot_tours"] = alts.sum(axis=1)
    tour_types = [c for c in alts.columns if c != "tot_tours"]
    alt_values = {c: alts[c].to_numpy() for c in alts.columns}  # what the formulas see of each alternative
    counts = alts[tour_types]  # 1.x deletes tot_tours before using the alternatives' counts
    no_tours = int(counts.index[0])  # 1.x: (alternatives.sum(axis=1) == 0).index[0]
    ext = pd.read_csv(find_config_file(configs_dirs, "non_mandatory_tour_frequency_extension_probs.csv"),
                      comment="#")
    ext["2_tours"] = ext["2_tours"] - ext["1_tours"]  # cumulative -> each count's probability (1.x)
    ext["1_tours"] = ext["1_tours"] - ext["0_tours"]
    channels = canonical_tour_channels(configs_dirs, top.get("models"))
    tour_type = pd.CategoricalDtype(tour_types, ordered=False)

    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    models = []
    for seg in segments:
        d = load_interaction_model(configs_dirs, settings["SPEC"], seg["COEFFICIENTS"], constants,
                                   name=f"{STEP}_{seg['NAME']}", value_column=seg["NAME"])
        terms_fn, coefs, source = d.utility_terms()
        models.append((seg["NAME"], int(seg["PTYPE"]), d, terms_fn, coefs, source))
    pre = settings.get("preprocessor")
    if pre:  # 1.x: the preprocessor can ask for the persons' time windows
        pre = dict(pre)
        pre["TABLES"] = list(pre.get("TABLES") or []) + ["person_windows"]
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, pre, constants, top, STEP, "preprocessor")
    annotations = annotate.legacy_annotations(configs_dirs, settings, top, STEP)
    resolved: dict = {}

    def resolve_needs(available):
        view_cols = VIEW.columns(available)
        needed = {"cdap_activity", segment_col, "household_id", "home_zone_id"}
        missing = sorted(needed - view_cols)
        if missing:
            raise MissingInputError(f"persons lack {missing}")
        wanted = set(needed) | {c for c in JOIN_COLUMNS if c in view_cols}
        extra: dict[str, set[str]] = {}
        bound = None
        targets: set[str] = set()
        if pre_spec is not None:
            bound = pre_spec.bind(view_cols, {t: available.get(t, set()) for t in pre_tables})
            if bound.n_random or bound.skim_uses():
                raise NotImplementedError(f"{STEP}: preprocessor with random draws or skims")
            targets = set(bound.targets)
            wanted |= bound.columns_used() - {INDEX}
            for t, c in bound.tables_used():
                extra.setdefault(t, set()).add(c)
        resolved["pre"] = bound
        for _, _, d, _, _, _ in models:
            for c in d.columns_used():
                if c in alt_values or c in targets:
                    continue
                if c.endswith("_chooser") and c[:-len("_chooser")] in alt_values:
                    wanted.add(c[:-len("_chooser")])
                    continue
                wanted.add(c)
        missing = sorted(c for c in JOIN_COLUMNS if c not in view_cols and c not in targets)
        if missing:
            raise MissingInputError(f"the tour count extension needs persons' {missing}")
        resolved["view_needs"] = VIEW.resolve(wanted - targets, available)
        made = {"persons": {OUTPUT}, "tours": set(TOUR_COLUMNS) | set(available.get("tours", set()))}
        bound_ann, ann_needs, _ = annotate.bind(annotations, available, made)
        resolved["annotations"] = bound_ann
        needs = {t: set(c) for t, c in resolved["view_needs"].items()}
        if "tours" in available:
            needs["tours"] = set(available["tours"])  # added to, as 1.x
        for src in (extra, ann_needs):
            for t, c in src.items():
                if t != "tours":
                    needs[t] = needs.get(t, set()) | set(c)
                else:
                    needs.setdefault("tours", set()).update(c)
        return {t: sorted(c) for t, c in needs.items()}

    def resolve_provides(available):
        tours = list(available.get("tours", []))
        provides = {"persons": [OUTPUT], "tours": tours + [c for c in TOUR_COLUMNS if c not in tours]}
        for a in annotations:
            for target in a.spec.targets:
                if target not in provides.setdefault(a.table, []):
                    provides[a.table].append(target)
        return provides

    def run(ctx: StepContext):
        person_ids = ctx.ids("persons")
        cols = VIEW.gather(ctx, resolved["view_needs"])
        active = np.flatnonzero(np.isin(np.asarray(plain(cols["cdap_activity"]), dtype=object), ["M", "N"]))
        ids = person_ids[active]
        ch = {c: (np.asarray(v)[active] if np.ndim(v) else v) for c, v in cols.items()}
        bound = resolved["pre"]
        if bound is not None and len(ids):
            from .annotate import _update

            c = {k: ch[k] for k in bound.columns_used() - {INDEX}}
            c[INDEX] = ids
            with np.errstate(all="ignore"):  # 1.x logs numpy warnings and carries on
                v = bound.function(c, None, len(ids), TableLookup(ctx), None, {})
            for target in bound.targets:
                ch[target] = _update(ch.get(target), np.asarray(v[target]))

        # one interaction model per person type, over all alternatives
        chosen = np.full(len(ids), -1, dtype=np.int64)
        seg_values = np.asarray(plain(ch[segment_col]))
        n_alts = len(alts)
        for name, ptype, d, terms_fn, coefs, _ in models:
            sel = np.flatnonzero(seg_values == ptype)
            per_batch = max(1, max_rows // n_alts)
            for b in range(0, len(sel), per_batch):
                batch = sel[b:b + per_batch]
                rows = _interaction_rows(ch, batch, alt_values, d.columns_used())
                n = len(batch) * n_alts
                with np.errstate(divide="ignore", invalid="ignore"):
                    utils = ctx.engine.interaction_utilities(FormulaValues(terms_fn, rows, n), coefs, n)
                utils = np.asarray(utils, dtype=np.float64).reshape(len(batch), n_alts)
                probs, bad = ctx.engine.probabilities(utils, shift=shift)
                if bad.any():
                    raise ValueError(f"{STEP}: {int(bad.sum())} persons ({name}) have no usable "
                                     f"probabilities, e.g. {ids[batch][bad][:5].tolist()}")
                rands = ctx.random.uniform("persons", ids[batch])
                chosen[batch] = ctx.engine.choose(probs, rands)
        if (chosen < 0).any():
            ctx.warn(f"{int((chosen < 0).sum())} active persons have a person type with no spec segment: "
                     f"no non-mandatory tours for them (1.x leaves them out too)")
        made_choice = chosen >= 0
        choices = pd.Series(alts.index.to_numpy()[chosen[made_choice]], index=pd.Index(ids[made_choice],
                                                                                         name="person_id"))
        choices = choices.sort_index()
        frequency = choices.reindex(person_ids).fillna(no_tours).astype(np.int16).to_numpy()

        # tour counts, extended
        tour_counts = counts.loc[choices.to_numpy()].copy()
        tour_counts.index = choices.index
        join = pd.DataFrame({c: np.asarray(plain(ch[c])) for c in JOIN_COLUMNS}, index=pd.Index(ids,
                                                                                                name="person_id"))
        tour_counts = extend_tour_counts(ctx, join, tour_counts, counts, ext)

        persons = pd.DataFrame({"household_id": np.asarray(cols["household_id"]),
                                "home_zone_id": np.asarray(cols["home_zone_id"])},
                               index=pd.Index(person_ids, name="person_id"))
        tours = create_tours(tour_counts, "non_mandatory")
        tours["household_id"] = persons.household_id.reindex(tours.person_id).to_numpy()
        tours["origin"] = persons.home_zone_id.reindex(tours.person_id).to_numpy()
        tours = set_tour_index(tours, channels)
        tours["tour_type"] = tours["tour_type"].astype(tour_type)
        old = ctx.table_frame("tours") if ctx.has("tours") else None
        tours_table = Table.from_pandas("tours", extend_table(old, tours[TOUR_COLUMNS]))
        ctx.create_table(tours_table)

        made = {"persons": {OUTPUT: frequency}}
        annotate.run(ctx, resolved["annotations"], made)
        out = {t: dict(v) for t, v in made.items()}
        out["tours"] = tours_table
        return out

    return Component(
        STEP, run, provides={"persons": [OUTPUT], "tours": list(TOUR_COLUMNS)},
        resolve_needs=resolve_needs, resolve_provides=resolve_provides,
        description=f"non-mandatory tour frequency: {len(alts)} alternatives, {len(models)} person types; "
                    f"extension; the tours",
        info={"generated_source": "\n\n".join(m[5] for m in models), "trace_table": "persons",
              "tour_channels": channels},
    )


def _interaction_rows(ch: dict, batch: np.ndarray, alt_values: dict, used: set) -> dict:
    """1.x's interaction_dataset: every chooser with every alternative, in that order; a chooser
    column named like an alternative's becomes '<name>_chooser'."""
    n_alts = len(next(iter(alt_values.values())))
    rows = {c: np.tile(v, len(batch)) for c, v in alt_values.items()}
    for c, v in ch.items():
        name = f"{c}_chooser" if c in alt_values else c
        if name in used or c in used:
            rows[name] = np.repeat(np.asarray(v)[batch], n_alts) if np.ndim(v) else v
    return rows


def extend_tour_counts(ctx: StepContext, persons: pd.DataFrame, tour_counts: pd.DataFrame,
                       alternatives: pd.DataFrame, probs_spec: pd.DataFrame) -> pd.DataFrame:
    """1.x's extend_tour_counts: one draw per tour type for persons with 1 to 4 tours whose
    count of that type is the type's most, giving them 0, 1 or 2 more."""
    extend = tour_counts.sum(axis=1).between(1, 4)
    if not extend.any():
        return tour_counts
    for i, tt in enumerate(alternatives.columns):
        maxed = extend & (tour_counts[tt] == alternatives[tt].max())
        idx = tour_counts.index[maxed]
        if len(idx) == 0:
            continue
        choosers = pd.merge(persons.loc[idx], probs_spec[probs_spec["nonmandatory_tour_type"] == i + 1],
                            on=JOIN_COLUMNS, how="left").set_index(idx)
        probs = choosers[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
        rands = ctx.random.uniform("persons", idx.to_numpy())
        more = ctx.engine.choose(probs, rands)
        if more.any():
            tour_counts.loc[idx, tt] += more
    return tour_counts
