"""
Telecommute frequency: how many days a week each worker with a workplace
works from home (1.x: ``abm/models/telecommute_frequency.py``; MWCOG and
SEMCOG run it).

Reads today's telecommute_frequency.yaml / .csv / coefficients. Choosers
are the persons with a workplace (``workplace_zone_id > -1``, which leaves
out those working from home: work_from_home sets theirs to -1), seen as
1.x's ``persons_merged``, with an optional preprocessor. Writes the chosen
alternative's name to ``persons.telecommute_frequency``, "" for everyone
else, as a category of the alternatives and "" (as 1.x).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..component import Component
from .person_choice import person_choice
from .simple_choice import ChoiceOutput, Choosers, simple_choice_component

STEP = "telecommute_frequency"


def telecommute_frequency(configs_dirs: Sequence[str | Path],
                          settings_file: str = "telecommute_frequency.yaml") -> Component:
    _, definition, common = person_choice(configs_dirs, settings_file, STEP)
    names = np.asarray(definition.alternatives, dtype=object)
    return simple_choice_component(
        **common, output_column=STEP,
        choosers=Choosers("workplace_zone_id", lambda z: z > -1, "persons with a workplace"),
        output=ChoiceOutput(lambda choices: names[choices], fill="", dtype=object,
                            categories=list(definition.alternatives) + [""]),
    )
