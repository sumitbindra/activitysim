"""
Transit pass subsidy and transit pass ownership (1.x:
``abm/models/transit_pass_subsidy.py`` and ``transit_pass_ownership.py``;
MWCOG and SEMCOG run them).

Each reads today's yaml / .csv / coefficients and chooses for every person
(subsidy: those whose ``CHOOSER_FILTER_COLUMN_NAME`` is true, if given),
seen as 1.x's ``persons_merged``, with an optional preprocessor. The
chosen alternative's position is written as 1.x writes it:

* ``persons.transit_pass_subsidy``: 0 for persons not choosing, as int64
  (1.x: ``fillna(0).astype(int)``); its preprocessor runs on every person
  before the choosers are picked, so the filter column may be one it makes;
* ``persons.transit_pass_ownership``: as the choice comes (int32).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..component import Component
from .person_choice import person_choice
from .simple_choice import ChoiceOutput, Choosers, simple_choice_component


def transit_pass_subsidy(configs_dirs: Sequence[str | Path],
                         settings_file: str = "transit_pass_subsidy.yaml") -> Component:
    step = "transit_pass_subsidy"
    settings, _, common = person_choice(configs_dirs, settings_file, step)
    filter_col = settings.get("CHOOSER_FILTER_COLUMN_NAME")
    choosers = None if filter_col is None else \
        Choosers(filter_col, lambda v: np.asarray(v, dtype=bool), f"persons with {filter_col}")
    return simple_choice_component(
        **common, output_column=step, choosers=choosers, filter_after_preprocessor=True,
        output=ChoiceOutput(lambda choices: choices.astype(np.int64), fill=0, dtype=np.int64),
    )


def transit_pass_ownership(configs_dirs: Sequence[str | Path],
                           settings_file: str = "transit_pass_ownership.yaml") -> Component:
    step = "transit_pass_ownership"
    _, _, common = person_choice(configs_dirs, settings_file, step)
    return simple_choice_component(**common, output_column=step)
