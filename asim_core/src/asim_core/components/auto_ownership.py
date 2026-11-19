"""
Auto ownership: how many cars each household owns.

Reads today's auto_ownership.yaml / .csv / _coefficients.csv unchanged.
Choosers are households, seeing land_use and accessibility columns through
their home zone, as in ActivitySim 1.x's ``households_merged``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..component import Component
from ..spec.legacy import load_choice_model, load_top_settings
from .simple_choice import ChooserView, simple_choice_component

STEP = "auto_ownership_simulate"  # 1.x step name: random streams are seeded from it


def auto_ownership(
    configs_dirs: Sequence[str | Path],
    settings_file: str = "auto_ownership.yaml",
) -> Component:
    definition = load_choice_model(configs_dirs, settings_file, name="auto_ownership")
    top = load_top_settings(configs_dirs)
    # 1.x skips overflow protection when skip_failed_choices is on (its default)
    shift = not top.get("skip_failed_choices", True)
    return simple_choice_component(
        definition,
        step=STEP,
        view=ChooserView(
            "households",
            joins=[("land_use", "home_zone_id"), ("accessibility", "home_zone_id")],
        ),
        output_column="auto_ownership",
        random_channel="households",
        shift_utilities=shift,
    )
