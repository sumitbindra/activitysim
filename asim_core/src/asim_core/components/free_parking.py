"""
Free parking: whether each worker can park for free at work (1.x:
``abm/models/free_parking.py``).

Reads today's free_parking.yaml / .csv / coefficients. Choosers are the
persons with a workplace (``workplace_zone_id > -1``), seeing their
household, home zone land use and accessibility, as in 1.x's
``persons_merged``; an optional preprocessor adds columns first (MTC: the
workplace's county). The choice is written to
``persons.free_parking_at_work``: True if the chosen alternative is
``FREE_PARKING_ALT``, False for everyone else, as 1.x.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..component import Component
from ..spec.legacy import find_config_file, load_choice_model, load_global_constants, load_top_settings, read_yaml
from .simple_choice import ChoiceOutput, ChooserView, Choosers, simple_choice_component
from .annotate import load_preprocessor

STEP = "free_parking"
# 1.x's persons_merged
VIEW = ChooserView("persons", [
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])
OUTPUT = "free_parking_at_work"


def free_parking(configs_dirs: Sequence[str | Path], settings_file: str = "free_parking.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    definition = load_choice_model(configs_dirs, settings_file, name=STEP, handled=["preprocessor"])
    if "FREE_PARKING_ALT" not in settings:
        raise ValueError(f"{settings_file}: FREE_PARKING_ALT is required")
    free_alt = int(settings["FREE_PARKING_ALT"])
    if not 0 <= free_alt < len(definition.alternatives):
        raise ValueError(f"{settings_file}: FREE_PARKING_ALT {free_alt} is not an alternative's position "
                         f"(alternatives: {definition.alternatives})")
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    pre_spec, pre_tables = load_preprocessor(configs_dirs, settings.get("preprocessor"), constants, top, STEP,
                                              "preprocessor")
    return simple_choice_component(
        definition, STEP, VIEW, OUTPUT, random_channel="persons",
        # 1.x subtracts each row's largest utility only with skip_failed_choices off (on by default)
        shift_utilities=not top.get("skip_failed_choices", True),
        choosers=Choosers("workplace_zone_id", lambda z: z > -1, "persons with a workplace"),
        preprocessor=None if pre_spec is None else (pre_spec, pre_tables),
        output=ChoiceOutput(lambda choices: choices == free_alt, fill=False, dtype=np.bool_),
    )
