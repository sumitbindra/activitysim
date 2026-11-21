"""
Joint tour participation: who of each household goes on its joint tours
(1.x: ``abm/models/joint_tour_participation.py``).

As 1.x:

1. **Candidates**: each joint tour with each member of its household
   (1.x's merge of the tours with ``persons_merged``), except adults on
   children-only tours and children on adults-only tours. A candidate's ID
   is ``tour_id * 100 + PNUM``; it names their random number stream.
2. The preprocessor annotates the candidates (with the persons' shared free
   time, ``person_time_window_overlap``); a logit gives each candidate's
   probability to participate.
3. **Choosing, in rounds** (1.x's ``participants_chooser``): every candidate
   still choosing draws; a tour is done when its party is valid (two or more
   people; for a mixed tour, an adult and a child); the candidates of other
   tours draw again (the next number on their stream), up to
   ``max_participation_choice_iterations`` rounds.
4. ``joint_tour_participants`` (tour, household, person and the
   participant's number within the tour, by person ID); each joint tour's
   ``person_id`` becomes its first participant and ``number_of_participants``
   its party size (1.x's ``assign_in_place``); then ``annotate_persons``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import Component, MissingInputError, StepContext
from ..logit import FormulaValues
from ..spec.legacy import find_config_file, load_choice_model, load_global_constants, load_top_settings, read_yaml
from ..spec.translate import INDEX
from ..store import Table
from . import annotate
from .runtime import TableLookup, plain
from .simple_choice import ChooserView
from .tours import assign_in_place

STEP = "joint_tour_participation"
CHANNEL = "joint_tour_participants"
MAX_PARTICIPANT_PNUM = 100  # 1.x's canonical_ids.MAX_PARTICIPANT_PNUM
PARTICIPANT_COLUMNS = ["tour_id", "household_id", "person_id", "participant_num"]
# 1.x's persons_merged
PERSONS_VIEW = ChooserView("persons", [
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])
TOUR_BASE = {"tour_category", "household_id", "person_id", "composition"}
PERSON_BASE = {"household_id", "num_hh_joint_tours", "adult"}


class ParticipationError(ValueError):
    pass


def joint_tour_participation(configs_dirs: Sequence[str | Path],
                             settings_file: str = "joint_tour_participation.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    definition = load_choice_model(configs_dirs, settings_file, name=STEP,
                                   handled=["preprocessor", "annotate_persons", "annotate_households",
                                            "annotate_tours"])
    choice_col = settings.get("participation_choice", "participate")
    if choice_col not in definition.alternatives:
        raise ValueError(f"{settings_file}: participation choice '{choice_col}' is not an alternative "
                         f"{definition.alternatives}")
    participate_alt = definition.alternatives.index(choice_col)
    max_iterations = int(settings.get("max_participation_choice_iterations", 5000))
    force = bool(settings.get("FORCE_PARTICIPATION", False))
    if len(definition.alternatives) != 2 and force:
        raise NotImplementedError(f"{settings_file}: FORCE_PARTICIPATION with more than two alternatives")
    shift = not top.get("skip_failed_choices", True)
    if definition.nest_tree() is not None:
        raise NotImplementedError(f"{settings_file}: nested logit")
    coef = definition.coefficient_matrix()
    keep = ~(coef == 0).all(axis=1)
    coef = coef[keep]
    terms_fn, source = definition.utility_function(keep)
    term_columns = definition.columns_used(keep)

    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    pre = settings.get("preprocessor")
    if pre:  # 1.x: persons (persons_merged) and their time windows are there for the preprocessor
        pre = dict(pre)
        pre["TABLES"] = list(pre.get("TABLES") or []) + ["persons", "person_windows"]
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, pre, constants, top, STEP, "preprocessor")
    annotations = annotate.legacy_annotations(configs_dirs, settings, top, STEP)
    resolved: dict = {}

    def resolve_needs(available):
        tour_cols = set(available.get("tours", set()))
        person_cols = PERSONS_VIEW.columns(available)
        missing = sorted((TOUR_BASE - tour_cols)) + sorted(PERSON_BASE - person_cols)
        if missing:
            raise MissingInputError(f"tours or persons lack {missing}")
        # the candidates' columns, as 1.x's merge gives them
        cand_cols = tour_cols - {"person_id"} | {"tour_id", "point_person_id", "person_id"} | person_cols
        cand_cols.add("PNUM")
        wanted = set(term_columns)
        extra: dict[str, set[str]] = {}
        bound = None
        if pre_spec is not None:
            bound = pre_spec.bind(cand_cols, {t: available.get(t, set()) for t in pre_tables})
            if bound.n_random or bound.skim_uses():
                raise NotImplementedError(f"{STEP}: preprocessor with random draws or skims")
            wanted = (wanted - set(bound.targets)) | (bound.columns_used() - {INDEX})
            for t, c in bound.tables_used():
                extra.setdefault(t, set()).add(c)
        resolved["pre"] = bound
        from_tours, from_persons = set(TOUR_BASE), set(PERSON_BASE)
        if "PNUM" in person_cols:
            from_persons.add("PNUM")
        for c in sorted(wanted - {"tour_id", "point_person_id", "person_id", "PNUM"}):
            in_tours, in_persons = c in tour_cols, c in person_cols
            if in_tours and in_persons and c != "household_id":
                raise MissingInputError(f"'{c}' is a column of both tours and persons: 1.x's merge renames it")
            if in_tours:
                from_tours.add(c)
            elif in_persons:
                from_persons.add(c)
            else:
                raise MissingInputError(f"candidates have no column '{c}'")
        resolved["from_tours"], resolved["from_persons"] = sorted(from_tours), sorted(from_persons)
        resolved["person_view"] = PERSONS_VIEW.resolve(from_persons, available)
        made = {"persons": set(), CHANNEL: set(PARTICIPANT_COLUMNS)}
        bound_ann, ann_needs, _ = annotate.bind(annotations, available, made)
        resolved["annotations"] = bound_ann
        needs = {"tours": set(tour_cols)}  # the whole table: it is rewritten, as 1.x
        for src in (resolved["person_view"], extra, ann_needs):
            for t, c in src.items():
                if t != CHANNEL:
                    needs[t] = needs.get(t, set()) | set(c)
        return {t: sorted(c) for t, c in needs.items()}

    def resolve_provides(available):
        provides = {CHANNEL: list(PARTICIPANT_COLUMNS), "tours": list(available.get("tours", []))}
        for a in annotations:
            for target in a.spec.targets:
                if target not in provides.setdefault(a.table, []):
                    provides[a.table].append(target)
        return provides

    def run(ctx: StepContext):
        tours_all = ctx.table_frame("tours")
        joint = tours_all[np.asarray(tours_all.tour_category) == "joint"]
        candidates = make_candidates(ctx, joint, resolved)
        n = len(candidates)
        pid = candidates.index.to_numpy()
        cols = {c: candidates[c].to_numpy() if not isinstance(candidates[c].dtype, pd.CategoricalDtype)
                else np.asarray(candidates[c].astype(object)) for c in candidates.columns}
        bound = resolved["pre"]
        if bound is not None and n:
            from .annotate import _update

            c = {k: cols[k] for k in bound.columns_used() - {INDEX}}
            c[INDEX] = pid
            with np.errstate(all="ignore"):  # 1.x logs numpy warnings and carries on
                v = bound.function(c, None, n, TableLookup(ctx), None, {})
            for target in bound.targets:
                cols[target] = _update(cols.get(target), np.asarray(v[target]))
        with np.errstate(divide="ignore", invalid="ignore"):
            utils = ctx.engine.utilities(FormulaValues(terms_fn, cols), coef, n)
        probs, bad = ctx.engine.probabilities(utils, shift=shift)
        if bad.any():
            raise ValueError(f"{STEP}: {int(bad.sum())} candidates have no usable probabilities, "
                             f"e.g. {pid[bad][:5].tolist()}")
        sat_frame = pd.DataFrame({"tour_id": candidates.tour_id.to_numpy(),
                                  "composition": candidates.composition.to_numpy(),
                                  "adult": candidates.adult.to_numpy()}, index=candidates.index)
        choices = choose_participants(ctx, probs, pid, sat_frame, participate_alt, max_iterations, force)
        participate = choices == participate_alt
        if not tour_satisfaction(sat_frame, participate).all():
            raise ParticipationError(f"{STEP}: tours without a valid party")

        participants = candidates.loc[participate, ["tour_id", "household_id", "person_id"]].copy()
        participants["participant_num"] = (participants.sort_values(by=["tour_id", "person_id"])
                                           .groupby("tour_id").cumcount() + 1)
        participants_table = Table.from_pandas(CHANNEL, participants)
        ctx.create_table(participants_table)

        # each joint tour's point person and party size (1.x: assign_in_place)
        joint = joint.copy()
        point = participants[participants.participant_num == 1]
        joint["person_id"] = point.set_index("tour_id").person_id
        joint["number_of_participants"] = participants.groupby("tour_id").size()
        assign_in_place(tours_all, joint[["person_id", "number_of_participants"]])

        made = {"persons": {}}
        annotate.run(ctx, resolved["annotations"], made)
        out = {t: dict(v) for t, v in made.items() if v}
        out[CHANNEL] = participants_table
        out["tours"] = Table.from_pandas("tours", tours_all)
        return out

    return Component(
        STEP, run, provides={CHANNEL: list(PARTICIPANT_COLUMNS), "tours": []},
        resolve_needs=resolve_needs, resolve_provides=resolve_provides,
        description="who of each household goes on its joint tours (chosen in rounds until each party is valid)",
        info={"generated_source": source, "trace_table": "tours"},
    )


def make_candidates(ctx: StepContext, joint: pd.DataFrame, resolved) -> pd.DataFrame:
    """1.x's joint_tour_participation_candidates, with the columns the step uses."""
    person_ids = ctx.ids("persons")
    cols = PERSONS_VIEW.gather(ctx, resolved["person_view"])
    persons = pd.DataFrame({c: plain(cols[c]) for c in resolved["from_persons"]},
                           index=pd.Index(person_ids, name="person_id"))
    persons = persons[persons.num_hh_joint_tours > 0]
    left = joint[[c for c in resolved["from_tours"] if c in joint.columns]].reset_index()
    left = left.rename(columns={"person_id": "point_person_id"})
    candidates = pd.merge(left, persons.reset_index(), left_on=["household_id"], right_on=["household_id"])
    if candidates.tour_id.nunique() != len(joint):
        raise ParticipationError(f"{STEP}: joint tours whose household has no persons with joint tours")
    eligible = ~(((candidates.composition == "adults") & ~candidates.adult)
                 | ((candidates.composition == "children") & candidates.adult))
    candidates = candidates[eligible]
    if "PNUM" not in candidates.columns:
        candidates["PNUM"] = candidates.groupby("household_id").cumcount() + 1
    if len(candidates) and candidates.PNUM.max() >= MAX_PARTICIPANT_PNUM:
        raise ParticipationError(f"PNUM above {MAX_PARTICIPANT_PNUM - 1}: participant IDs would collide")
    candidates["participant_id"] = (candidates.tour_id * MAX_PARTICIPANT_PNUM + candidates.PNUM).astype(np.int64)
    candidates.set_index("participant_id", drop=True, inplace=True, verify_integrity=True)
    return candidates


def tour_satisfaction(candidates: pd.DataFrame, participate: np.ndarray) -> pd.Series:
    """1.x's get_tour_satisfaction: True for tours whose chosen party is valid."""
    tour_ids = candidates.tour_id.unique()
    participate = np.asarray(participate, dtype=bool)
    if participate.any():
        chosen = candidates[participate]
        x = (chosen[["tour_id", "composition", "adult"]]
             .groupby(["tour_id", "composition"], observed=True)
             .agg(participants=("adult", "size"), adults=("adult", "sum"))
             .reset_index("composition"))
        sat = (x.composition != "mixed") & (x.participants > 1) | \
              (x.composition == "mixed") & (x.adults > 0) & (x.participants > x.adults)
        sat = sat.reindex(tour_ids, fill_value=False).astype(bool)  # 1.x: .reindex().fillna(False)
    else:
        sat = pd.Series(dtype=bool)
    return sat.reindex(tour_ids, fill_value=False).astype(bool)


def choose_participants(ctx: StepContext, probs: np.ndarray, pid: np.ndarray, candidates: pd.DataFrame,
                        participate_alt: int, max_iterations: int, force: bool) -> np.ndarray:
    """1.x's participants_chooser: rounds of draws until every tour's party is valid."""
    n = len(pid)
    out = np.full(n, -1, dtype=np.int64)
    remaining = np.arange(n)
    probs = np.array(probs, dtype=np.float64)
    tours = candidates.tour_id.to_numpy()
    left = len(np.unique(tours))
    rounds = 0
    while len(remaining):
        rounds += 1
        if rounds > max_iterations:
            if not force:
                raise ParticipationError(f"{STEP}: {left} tours could not be satisfied after {rounds} rounds")
            if rounds > max_iterations + 1:
                raise ParticipationError(f"{STEP}: {left} tours could not be satisfied even with forcing")
            p = probs[remaining]
            p[:, participate_alt] = np.where(p[:, participate_alt] > 0, 1, 0)
            p[:, 1 - participate_alt] = 1 - p[:, participate_alt]
            probs[remaining] = p
        rands = ctx.random.uniform(CHANNEL, pid[remaining])
        chosen = ctx.engine.choose(probs[remaining], rands)
        sat = tour_satisfaction(candidates.iloc[remaining], chosen == participate_alt)
        done = int(sat.sum())
        if done:
            left -= done
            ok = sat.reindex(tours[remaining]).to_numpy()
            out[remaining[ok]] = chosen[ok]
            remaining = remaining[~ok]
    return out
