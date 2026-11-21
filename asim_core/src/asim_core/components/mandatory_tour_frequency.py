"""
Mandatory tour frequency: how many work and school tours each person with a
mandatory day (``cdap_activity == 'M'``) makes, and those tours (1.x:
``abm/models/mandatory_tour_frequency.py``, ``process_mandatory_tours``).

The choice is a simple logit over the spec's alternatives (MTC: work1,
work2, school1, school2, work_and_school), written to
``persons.mandatory_tour_frequency`` ('' for everyone else). Each choice
becomes tours (``mandatory_tour_frequency_alternatives.csv``: how many of
each type), numbered as 1.x numbers them (for work_and_school, the school
tour comes first for non-workers), with destination (workplace or school
zone), origin (home zone), household, and 1.x's tour IDs (see
``tours.py``). Then the settings' ``annotate_persons`` (and
``annotate_households`` / ``annotate_tours`` if given).

1.x appends to a tours table if one exists (e.g. after school escorting);
asim_core doesn't support that yet.
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
from .tours import canonical_tour_channels, create_tours, set_tour_index

STEP = "mandatory_tour_frequency"
OUTPUT = "mandatory_tour_frequency"
VIEW = ChooserView("persons", [
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])
PERSON_COLUMNS = ["is_worker", "school_zone_id", "workplace_zone_id", "home_zone_id", "household_id"]
TOUR_COLUMNS = ["person_id", "tour_type", "tour_type_count", "tour_type_num", "tour_num", "tour_count",
                "tour_category", "number_of_participants", "destination", "origin", "household_id"]


def mandatory_tour_frequency(configs_dirs: Sequence[str | Path],
                             settings_file: str = "mandatory_tour_frequency.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    definition = load_choice_model(configs_dirs, settings_file, name=STEP,
                                   handled=["preprocessor", "annotate_persons", "annotate_households",
                                            "annotate_tours"])
    alts = pd.read_csv(find_config_file(configs_dirs, "mandatory_tour_frequency_alternatives.csv"), comment="#")
    if "Alt" in alts.columns:
        alts = alts.rename(columns={"Alt": "alt"})
    alts = alts.set_index("alt")
    missing = [a for a in definition.alternatives if a not in alts.index]
    if missing:
        raise ValueError(f"mandatory_tour_frequency_alternatives.csv lacks the spec's alternatives {missing}")
    names = np.asarray(definition.alternatives, dtype=object)
    choice_type = pd.CategoricalDtype(list(definition.alternatives) + [""], ordered=False)
    tour_type = pd.CategoricalDtype(list(alts.columns) + ["univ", "home", "escort"], ordered=False)
    channels = canonical_tour_channels(configs_dirs, top.get("models"))

    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, settings.get("preprocessor"), constants, top,
                                                      STEP, "preprocessor")
    choice = simple_choice_component(
        definition, STEP, VIEW, OUTPUT, random_channel="persons",
        shift_utilities=not top.get("skip_failed_choices", True),
        choosers=Choosers("cdap_activity", lambda a: a == "M", "persons with a mandatory day"),
        preprocessor=None if pre_spec is None else (pre_spec, pre_tables),
        output=ChoiceOutput(lambda c: names[c], fill="", dtype=object),
    )
    annotations = annotate.legacy_annotations(configs_dirs, settings, top, STEP)
    resolved: dict = {}

    def resolve_needs(available):
        if "tours" in available:
            raise NotImplementedError(f"{STEP}: a tours table exists already (1.x would add to it)")
        missing = [c for c in PERSON_COLUMNS if c not in VIEW.columns(available)]
        if missing:
            raise MissingInputError(f"persons (with households and land use) lack {missing}")
        needs = {t: set(c) for t, c in choice.resolve_needs(available).items()}
        resolved["view_needs"] = VIEW.resolve(set(PERSON_COLUMNS), available)
        bound, ann_needs, _ = annotate.bind(annotations, available,
                                            {"persons": {OUTPUT}, "tours": set(TOUR_COLUMNS)})
        resolved["annotations"] = bound
        for src in (resolved["view_needs"], ann_needs):
            for t, c in src.items():
                if t != "tours":
                    needs[t] = needs.get(t, set()) | set(c)
        return {t: sorted(c) for t, c in needs.items()}

    def run(ctx: StepContext):
        chosen = np.asarray(choice.run(ctx)["persons"][OUTPUT], dtype=object)
        ids = ctx.ids("persons")
        rows = np.flatnonzero(chosen != "")
        cols = VIEW.gather(ctx, resolved["view_needs"])
        persons = pd.DataFrame({c: np.asarray(cols[c])[rows] for c in PERSON_COLUMNS},
                               index=pd.Index(ids[rows], name="person_id"))
        persons[OUTPUT] = chosen[rows]
        tours = mandatory_tours(persons, alts, channels)
        tours["tour_type"] = tours["tour_type"].astype(tour_type)
        tours_table = Table.from_pandas("tours", tours[TOUR_COLUMNS])
        ctx.create_table(tours_table)  # the annotations read it
        made = {"persons": {OUTPUT: pd.Categorical(chosen, dtype=choice_type)}}
        annotate.run(ctx, resolved["annotations"], made)
        out = {t: dict(v) for t, v in made.items()}
        out["tours"] = tours_table
        return out

    provides = {"persons": [OUTPUT], "tours": list(TOUR_COLUMNS)}
    for a in annotations:
        for target in a.spec.targets:
            if target not in provides.setdefault(a.table, []):
                provides[a.table].append(target)
    return Component(
        STEP, run, provides=provides, resolve_needs=resolve_needs,
        description=f"mandatory tour frequency ({', '.join(definition.alternatives)}) and the mandatory tours",
        info={**{k: v for k, v in choice.info.items() if k not in ("compute", "view", "columns")},
              "trace_table": "persons", "tour_channels": channels},
    )


def mandatory_tours(persons: pd.DataFrame, alts: pd.DataFrame, channels: Sequence[str]) -> pd.DataFrame:
    """1.x's process_mandatory_tours: the tours of persons' choices (``mandatory_tour_frequency``),
    with destination, origin, household and 1.x's tour IDs."""
    frequency = persons[OUTPUT]
    counts = alts.loc[frequency.to_numpy()]
    counts.index = frequency.index
    tours = create_tours(counts, "mandatory")
    merged = pd.merge(tours[["person_id", "tour_type"]], persons, left_on="person_id", right_index=True)
    # work_and_school: workers go to work first, others to school first
    student_first = (merged[OUTPUT] == "work_and_school") & ~merged.is_worker
    tours.tour_num = tours.tour_num.where(~student_first, 3 - tours.tour_num)
    tours["destination"] = merged.workplace_zone_id.where(merged.tour_type == "work", merged.school_zone_id)
    tours["origin"] = merged.home_zone_id
    tours["household_id"] = merged.household_id
    return set_tour_index(tours, channels)
