"""
What the simple choices by persons share (1.x's work_from_home,
transit_pass_subsidy, transit_pass_ownership, telecommute_frequency): today's
settings, the model, and how 1.x runs it (its ``simple_simulate`` on
``persons_merged``, the persons' random stream, an optional preprocessor).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..spec.legacy import find_config_file, load_choice_model, load_global_constants, load_top_settings, read_yaml
from .annotate import load_preprocessor
from .simple_choice import PERSONS_MERGED


def person_choice(configs_dirs: Sequence[str | Path], settings_file: str, step: str):
    """
    (settings, model definition, arguments for ``simple_choice_component``):
    the persons as 1.x's ``persons_merged``, their random stream, 1.x's
    utility shift and the preprocessor, if any.
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    definition = load_choice_model(configs_dirs, settings_file, name=step, handled=["preprocessor"])
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    pre_spec, pre_tables = load_preprocessor(configs_dirs, settings.get("preprocessor"), constants, top, step,
                                              "preprocessor")
    common = dict(
        definition=definition, step=step, view=PERSONS_MERGED, random_channel="persons",
        # 1.x subtracts each row's largest utility only with skip_failed_choices off (on by default)
        shift_utilities=not top.get("skip_failed_choices", True),
        preprocessor=None if pre_spec is None else (pre_spec, pre_tables),
    )
    return settings, definition, common
