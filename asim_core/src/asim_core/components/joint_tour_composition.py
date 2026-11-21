"""
Joint tour composition: whether each joint tour's party is adults, children
or mixed (1.x: ``abm/models/joint_tour_composition.py``).

As 1.x: the preprocessor annotates the households with joint tours (with
their persons and shared free time, ``hh_time_window_overlap``); each joint
tour, merged with its household, chooses; ``tours.composition`` is a
category of the spec's alternatives and '' (the other tours).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..component import Component
from ..spec.legacy import find_config_file, load_choice_model, load_global_constants, load_top_settings, read_yaml
from . import annotate
from .simple_choice import ChoiceOutput, ChooserView, Choosers, Preprocessor, simple_choice_component

STEP = "joint_tour_composition"
OUTPUT = "composition"
# 1.x: the joint tours merged with their (annotated) households
VIEW = ChooserView("tours", [("households", "household_id")])


def joint_tour_composition(configs_dirs: Sequence[str | Path], settings_file: str = "joint_tour_composition.yaml"
                           ) -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    if settings.get("annotate_persons") or settings.get("annotate_households") or settings.get("annotate_tours"):
        raise NotImplementedError(f"{settings_file}: annotations after the model")
    top = load_top_settings(configs_dirs)
    definition = load_choice_model(configs_dirs, settings_file, name=STEP, handled=["preprocessor"])
    names = np.asarray(definition.alternatives, dtype=object)
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    pre = settings.get("preprocessor")
    if pre:  # 1.x also gives the preprocessor the households' persons and their time windows
        pre = dict(pre)
        pre["TABLES"] = list(pre.get("TABLES") or []) + ["persons", "person_windows"]
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, pre, constants, top, STEP, "preprocessor")
    return simple_choice_component(
        definition, STEP, VIEW, OUTPUT, random_channel="tours",
        shift_utilities=not top.get("skip_failed_choices", True),
        choosers=Choosers("tour_category", lambda c: c == "joint", "joint tours"),
        preprocessor=None if pre_spec is None else Preprocessor(
            pre_spec, pre_tables, on="households",
            rows=Choosers("num_hh_joint_tours", lambda n: n > 0, "households with joint tours")),
        output=ChoiceOutput(lambda c: names[c], fill="", dtype=object,
                            categories=list(definition.alternatives) + [""]),
    )
