"""
Joint tour frequency: how many tours, of which purposes, each household's
members make together (1.x: ``abm/models/joint_tour_frequency.py``).

As 1.x: households with ``participates_in_jtf_model`` choose among the
alternatives of ``joint_tour_frequency_alternatives.csv`` (MTC: none, one of
five purposes, or two), with a preprocessor that sees the households'
persons and their free time together (``hh_time_window_overlap``). Each
chosen tour becomes a row of ``tours`` (category 'joint'), given for now to
the household's first person (``PNUM == 1``) and starting from home, with
1.x's tour IDs; the tours are added after the existing ones as 1.x's
``extend_table`` adds them. ``households.joint_tour_frequency`` (the others
get the first alternative) and ``households.num_hh_joint_tours``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import Component, MissingInputError, StepContext
from ..spec.legacy import find_config_file, load_choice_model, load_global_constants, load_top_settings, read_yaml
from ..store import Table
from . import annotate
from .simple_choice import ChoiceOutput, ChooserView, Choosers, simple_choice_component
from .tours import canonical_tour_channels, create_tours, extend_table, set_tour_index

STEP = "joint_tour_frequency"
OUTPUT = "joint_tour_frequency"
VIEW = ChooserView("households", [])  # 1.x's choosers: the households table itself
JOINT_TOUR_COLUMNS = ["household_id", "tour_type", "tour_type_count", "tour_type_num", "tour_num", "tour_count",
                      "tour_category", "number_of_participants", "person_id", "origin"]
PERSON_COLUMNS = ["household_id", "home_zone_id"]


def joint_tour_frequency(configs_dirs: Sequence[str | Path], settings_file: str = "joint_tour_frequency.yaml"
                         ) -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    definition = load_choice_model(configs_dirs, settings_file, name=STEP,
                                   handled=["preprocessor", "annotate_persons", "annotate_households",
                                            "annotate_tours"])
    alts = pd.read_csv(find_config_file(configs_dirs, "joint_tour_frequency_alternatives.csv"), comment="#")
    alts = alts.rename(columns={"Alt": "alt"}).set_index("alt")
    missing = [a for a in definition.alternatives if a not in alts.index]
    if missing:
        raise ValueError(f"joint_tour_frequency_alternatives.csv lacks the spec's alternatives {missing}")
    names = np.asarray(definition.alternatives, dtype=object)
    choice_type = pd.CategoricalDtype(list(definition.alternatives), ordered=False)
    tour_type = pd.CategoricalDtype(list(alts.columns), ordered=False)
    # 1.x: (alternatives.sum(axis=1) == 0).index[0], i.e. the first alternative
    no_tours = alts.index[0]
    channels = canonical_tour_channels(configs_dirs, top.get("models"))

    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    pre = settings.get("preprocessor")
    if pre:  # 1.x also gives the preprocessor the households' persons and their time windows
        pre = dict(pre)
        pre["TABLES"] = list(pre.get("TABLES") or []) + ["persons", "person_windows"]
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, pre, constants, top, STEP, "preprocessor")
    choice = simple_choice_component(
        definition, STEP, VIEW, OUTPUT, random_channel="households",
        shift_utilities=not top.get("skip_failed_choices", True),
        choosers=Choosers("participates_in_jtf_model", lambda x: x.astype(bool), "households in the model"),
        preprocessor=None if pre_spec is None else (pre_spec, pre_tables),
        output=ChoiceOutput(lambda c: names[c], fill=no_tours, dtype=object),
    )
    annotations = annotate.legacy_annotations(configs_dirs, settings, top, STEP)
    resolved: dict = {}

    def resolve_needs(available):
        missing = [c for c in PERSON_COLUMNS if c not in available.get("persons", set())]
        if missing:
            raise MissingInputError(f"persons lack {missing}")
        needs = {t: set(c) for t, c in choice.resolve_needs(available).items()}
        needs.setdefault("persons", set()).update(PERSON_COLUMNS)
        point = "PNUM" if "PNUM" in available.get("persons", set()) else None
        if point:
            needs["persons"].add(point)
        resolved["point"] = point
        if "tours" in available:
            needs["tours"] = set(available["tours"])  # added to, as 1.x
        made = {"households": {OUTPUT, "num_hh_joint_tours"}, "tours": set(JOINT_TOUR_COLUMNS)}
        bound, ann_needs, _ = annotate.bind(annotations, available, made)
        resolved["annotations"] = bound
        for t, c in ann_needs.items():
            if t != "tours":
                needs[t] = needs.get(t, set()) | set(c)
        return {t: sorted(c) for t, c in needs.items()}

    def resolve_provides(available):
        tours = list(available.get("tours", []))
        provides = {"households": [OUTPUT, "num_hh_joint_tours"],
                    "tours": tours + [c for c in JOINT_TOUR_COLUMNS if c not in tours]}
        for a in annotations:
            for target in a.spec.targets:
                if target not in provides.setdefault(a.table, []):
                    provides[a.table].append(target)
        return provides

    def run(ctx: StepContext):
        chosen = np.asarray(choice.run(ctx)["households"][OUTPUT], dtype=object)
        hh_ids = ctx.ids("households")
        choosers = np.asarray(ctx.column("households", "participates_in_jtf_model")).astype(bool)
        frequency = pd.Series(pd.Categorical(chosen[choosers], dtype=choice_type), index=hh_ids[choosers])
        frequency.index.name = "household_id"

        # the point person: PNUM 1, or the lowest person ID, of each household that chose
        person_ids = ctx.ids("persons")
        persons = pd.DataFrame({c: np.asarray(ctx.column("persons", c)) for c in PERSON_COLUMNS
                                + ([resolved["point"]] if resolved["point"] else [])},
                               index=pd.Index(person_ids, name="person_id"))
        persons = persons[persons.household_id.isin(frequency.index)]
        if resolved["point"]:
            point = persons.loc[persons.PNUM == 1]
        else:
            point = persons.sort_index().groupby("household_id", as_index=False).first()
        point = point.assign(person_id=point.index).set_index("household_id")[["person_id", "home_zone_id"]]

        tours = joint_tours(frequency, alts, point, channels)
        tours["tour_type"] = tours["tour_type"].astype(tour_type)
        old = ctx.table_frame("tours") if ctx.has("tours") else None
        all_tours = extend_table(old, tours[JOINT_TOUR_COLUMNS])
        tours_table = Table.from_pandas("tours", all_tours)
        ctx.create_table(tours_table)

        counts = tours.groupby("household_id").size().reindex(hh_ids).fillna(0).astype(np.int8).to_numpy()
        made = {"households": {OUTPUT: pd.Categorical(chosen, dtype=choice_type), "num_hh_joint_tours": counts}}
        annotate.run(ctx, resolved["annotations"], made)
        out = {t: dict(v) for t, v in made.items()}
        out["tours"] = tours_table
        return out

    return Component(
        STEP, run, provides={"households": [OUTPUT, "num_hh_joint_tours"], "tours": list(JOINT_TOUR_COLUMNS)},
        resolve_needs=resolve_needs, resolve_provides=resolve_provides,
        description=f"joint tour frequency ({len(definition.alternatives)} alternatives) and the joint tours",
        info={**{k: v for k, v in choice.info.items() if k not in ("compute", "view", "columns")},
              "trace_table": "households", "tour_channels": channels},
    )


def joint_tours(frequency: pd.Series, alts: pd.DataFrame, point: pd.DataFrame, channels) -> pd.DataFrame:
    """1.x's process_joint_tours: the households' tours, their point person and origin, 1.x's IDs."""
    counts = alts.loc[frequency.to_numpy()]
    counts.index = frequency.index
    tours = create_tours(counts, "joint", parent_col="household_id")
    tours["person_id"] = point.person_id.reindex(tours.household_id).to_numpy()
    tours["origin"] = point.home_zone_id.reindex(tours.household_id).to_numpy()
    return set_tour_index(tours, channels, is_joint=True)
