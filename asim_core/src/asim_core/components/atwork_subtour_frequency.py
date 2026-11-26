"""
At-work subtour frequency: how many tours each work tour has from the
workplace (eat, business, maintenance), and those subtours (1.x:
``abm/models/atwork_subtour_frequency.py``, ``process_atwork_subtours``).

As 1.x: each work tour, merged with its person (``persons_merged``) and
annotated by the preprocessor, chooses among
``atwork_subtour_frequency_alternatives.csv``'s alternatives;
``tours.atwork_subtour_frequency`` is that choice (missing for the other
tours). Each subtour gets its work tour as ``parent_tour_id``, its person
and household, starts from the workplace (the work tour's destination), and
1.x's tour ID (``eat1_1``: the eat subtour of work tour 1); they are added
after the existing tours.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import Component, MissingInputError, StepContext
from ..spec.legacy import find_config_file, load_choice_model, load_global_constants, load_top_settings, read_settings
from ..store import Table
from . import annotate
from .simple_choice import ChoiceOutput, ChooserView, Choosers, simple_choice_component
from .tours import canonical_tour_channels, create_tours, extend_table, set_tour_index

STEP = "atwork_subtour_frequency"
OUTPUT = "atwork_subtour_frequency"
VIEW = ChooserView("tours", [  # 1.x: the work tours merged with persons_merged
    ("persons", "person_id"),
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])
SUBTOUR_COLUMNS = ["parent_tour_id", "tour_type", "tour_type_count", "tour_type_num", "tour_num", "tour_count",
                   "tour_category", "number_of_participants", "person_id", "household_id", "origin"]
WORK_TOUR_COLUMNS = ["person_id", "household_id", "tour_num", "destination", "tour_type"]


def atwork_subtour_frequency(configs_dirs: Sequence[str | Path],
                             settings_file: str = "atwork_subtour_frequency.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_settings(configs_dirs, settings_file)
    top = load_top_settings(configs_dirs)
    definition = load_choice_model(configs_dirs, settings_file, name=STEP,
                                   handled=["preprocessor", "annotate_persons", "annotate_households",
                                            "annotate_tours"])
    alts = pd.read_csv(find_config_file(configs_dirs, "atwork_subtour_frequency_alternatives.csv"), comment="#")
    alts = alts.rename(columns={"Alt": "alt"}).set_index("alt")
    missing = [a for a in definition.alternatives if a not in alts.index]
    if missing:
        raise ValueError(f"atwork_subtour_frequency_alternatives.csv lacks the spec's alternatives {missing}")
    names = np.asarray(definition.alternatives, dtype=object)
    categories = list(alts.index) + [""]  # 1.x: the alternatives file's names, then ''
    tour_type = pd.CategoricalDtype(list(alts.columns) + ["atwork"], ordered=False)
    channels = canonical_tour_channels(configs_dirs, top.get("models"))

    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, settings.get("preprocessor"), constants, top,
                                                      STEP, "preprocessor")
    choice = simple_choice_component(
        definition, STEP, VIEW, OUTPUT, random_channel="tours",
        shift_utilities=not top.get("skip_failed_choices", True),
        choosers=Choosers("tour_type", lambda t: t == "work", "work tours"),
        preprocessor=None if pre_spec is None else (pre_spec, pre_tables),
        output=ChoiceOutput(lambda c: names[c], fill=None, dtype=object, categories=categories),
    )
    annotations = annotate.legacy_annotations(configs_dirs, settings, top, STEP)
    resolved: dict = {}

    def resolve_needs(available):
        missing = [c for c in WORK_TOUR_COLUMNS if c not in available.get("tours", set())]
        if missing:
            raise MissingInputError(f"tours lack {missing}")
        needs = {t: set(c) for t, c in choice.resolve_needs(available).items()}
        needs["tours"] = needs.get("tours", set()) | set(available["tours"])  # added to, as 1.x
        made = {"tours": set(SUBTOUR_COLUMNS) | {OUTPUT} | set(available["tours"])}
        bound, ann_needs, _ = annotate.bind(annotations, available, made)
        resolved["annotations"] = bound
        for t, c in ann_needs.items():
            needs[t] = needs.get(t, set()) | set(c)
        return {t: sorted(c) for t, c in needs.items()}

    def resolve_provides(available):
        tours = list(available.get("tours", []))
        provides = {"tours": tours + [c for c in [OUTPUT] + SUBTOUR_COLUMNS if c not in tours]}
        for a in annotations:
            for target in a.spec.targets:
                if target not in provides.setdefault(a.table, []):
                    provides[a.table].append(target)
        return provides

    def run(ctx: StepContext):
        frequency = choice.run(ctx)["tours"][OUTPUT]  # a category, missing for tours that aren't work
        tours = ctx.table_frame("tours")
        tours[OUTPUT] = pd.Categorical(frequency, categories=categories)
        work = tours[tours.tour_type == "work"]
        subtours = atwork_subtours(work, alts, channels)
        subtours["tour_type"] = subtours["tour_type"].astype(tour_type)
        tours_table = Table.from_pandas("tours", extend_table(tours, subtours[SUBTOUR_COLUMNS]))
        ctx.create_table(tours_table)
        made: dict = {}
        annotate.run(ctx, resolved["annotations"], made)
        out = {t: dict(v) for t, v in made.items() if t != "tours"}
        out["tours"] = tours_table
        return out

    return Component(
        STEP, run, provides={"tours": [OUTPUT] + SUBTOUR_COLUMNS}, resolve_needs=resolve_needs,
        resolve_provides=resolve_provides,
        description=f"at-work subtour frequency ({len(definition.alternatives)} alternatives) and the subtours",
        info={**{k: v for k, v in choice.info.items() if k not in ("compute", "view", "columns")},
              "trace_table": "tours", "tour_channels": channels},
    )


def atwork_subtours(work_tours: pd.DataFrame, alts: pd.DataFrame, channels) -> pd.DataFrame:
    """1.x's process_atwork_subtours: the work tours' subtours, from the workplace, with 1.x's IDs."""
    frequency = work_tours[OUTPUT].dropna()
    counts = alts.loc[frequency.astype(str).to_numpy()]
    counts.index = frequency.index
    tours = create_tours(counts, "atwork", parent_col="parent_tour_id")
    parents = work_tours[["person_id", "household_id", "tour_num", "destination"]].rename(
        columns={"tour_num": "parent_tour_num", "destination": "origin"})
    tours = pd.merge(tours, parents, left_on="parent_tour_id", right_index=True)
    tours = set_tour_index(tours, channels, parent_tour_num_col="parent_tour_num")
    del tours["parent_tour_num"]
    return tours
