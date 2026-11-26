"""
Work from home: whether each worker works from home (1.x:
``abm/models/work_from_home.py``; MWCOG and SEMCOG run it).

Reads today's work_from_home.yaml / .csv / coefficients. Choosers are the
persons whose ``CHOOSER_FILTER_COLUMN_NAME`` (default ``is_worker``) is
true, seen as 1.x's ``persons_merged``; an optional preprocessor adds
columns first (MWCOG: whether others in the household are preschool
children or non-working adults). Writes, as 1.x:

* ``persons.work_from_home``: True if the chosen alternative is
  ``WORK_FROM_HOME_ALT``, False for everyone else;
* ``persons.is_out_of_home_worker``: the filter column and not working
  from home;
* ``DEST_CHOICE_COLUMN_NAME`` (default ``workplace_zone_id``), if the
  persons have it: -1 for those working from home (1.x: so the telecommute
  model leaves them out).

Not supported yet: ``WORK_FROM_HOME_ITERATIONS`` above 1 (1.x runs the
model again with a constant adjusted towards a target share).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..component import Component, MissingInputError, StepContext
from .person_choice import person_choice
from .runtime import plain
from .simple_choice import ChoiceOutput, Choosers, simple_choice_component

STEP = "work_from_home"
OUTPUT = "work_from_home"
OUT_OF_HOME = "is_out_of_home_worker"


def work_from_home(configs_dirs: Sequence[str | Path], settings_file: str = "work_from_home.yaml") -> Component:
    settings, definition, common = person_choice(configs_dirs, settings_file, STEP)
    if "WORK_FROM_HOME_ALT" not in settings:
        raise ValueError(f"{settings_file}: WORK_FROM_HOME_ALT is required")
    home_alt = int(settings["WORK_FROM_HOME_ALT"])
    if not 0 <= home_alt < len(definition.alternatives):
        raise ValueError(f"{settings_file}: WORK_FROM_HOME_ALT {home_alt} is not an alternative's position "
                         f"(alternatives: {definition.alternatives})")
    if int(settings.get("WORK_FROM_HOME_ITERATIONS") or 1) != 1:
        raise NotImplementedError(f"{settings_file}: not supported yet: WORK_FROM_HOME_ITERATIONS above 1 (the "
                                  f"model run again with its constant adjusted towards a target share)")
    filter_col = settings.get("CHOOSER_FILTER_COLUMN_NAME") or "is_worker"
    dest_col = settings.get("DEST_CHOICE_COLUMN_NAME") or "workplace_zone_id"
    base = simple_choice_component(
        **common, output_column=OUTPUT,
        choosers=Choosers(filter_col, lambda v: np.asarray(v, dtype=bool), f"persons with {filter_col}"),
        output=ChoiceOutput(lambda choices: choices == home_alt, fill=False, dtype=np.bool_),
    )

    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]):
        persons = available.get("persons", set())
        resolved["set_dest"] = dest_col in persons
        if filter_col not in persons:  # 1.x reads it from the persons for is_out_of_home_worker
            raise MissingInputError(f"persons lack '{filter_col}' (CHOOSER_FILTER_COLUMN_NAME)")
        needs = {t: set(c) for t, c in base.needs_given(available).items()}
        needs.setdefault("persons", set()).add(filter_col)
        if dest_col in persons:
            needs["persons"].add(dest_col)
        return needs

    def resolve_provides(available: Mapping[str, set[str]]):
        return {"persons": [OUTPUT, OUT_OF_HOME] + ([dest_col] if dest_col in available.get("persons", set())
                                                     else [])}

    def run(ctx: StepContext):
        at_home = np.asarray(base.run(ctx)["persons"][OUTPUT], dtype=bool)
        out = {OUTPUT: at_home,
               OUT_OF_HOME: np.asarray(plain(ctx.column("persons", filter_col))) & ~at_home}
        if resolved["set_dest"]:
            out[dest_col] = np.where(at_home, -1, np.asarray(plain(ctx.column("persons", dest_col))))
        return {"persons": out}

    return Component(STEP, run, provides={"persons": [OUTPUT, OUT_OF_HOME]}, resolve_needs=resolve_needs,
                     resolve_provides=resolve_provides, description=base.description, info=base.info)
