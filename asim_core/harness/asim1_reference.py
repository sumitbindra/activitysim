"""
Utilities and probabilities computed by ActivitySim 1.x itself, for checking
asim_core below the level of final choices.

Restores 1.x's saved state just before a step, then calls 1.x's own functions
(``eval_utilities`` and ``utils_to_probs``, legacy engine) on the same
choosers. Harness code: imports ActivitySim 1.x.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def simple_model_internals(
    example_dir: Path,
    configs: list[str],
    pipeline: Path,
    resume_after: str,
    settings_cls,
    settings_file: str,
    choosers_table: str,
    scratch_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from activitysim import abm  # noqa: F401  (registers tables and steps)
    from activitysim.core import chunk, config, simulate, workflow
    from activitysim.core import logit as asim_logit

    scratch_dir.mkdir(parents=True, exist_ok=True)
    state = workflow.State.make_default(
        working_dir=example_dir,
        configs_dir=tuple(configs),
        data_dir=("data",),
        output_dir=scratch_dir,
    )
    state.checkpoint.restore_from(Path(pipeline), resume_after)
    settings = settings_cls.read_settings_file(state.filesystem, settings_file)
    spec = state.filesystem.read_model_spec(file_name=settings.SPEC)
    coefficients = state.filesystem.read_model_coefficients(settings)
    spec = simulate.eval_coefficients(state, spec, coefficients, None)
    constants = config.get_model_constants(settings)
    choosers = state.get_dataframe(choosers_table)

    parts = []
    for _, chunk_choosers, label, sizer in chunk.adaptive_chunked_choosers(
        state, choosers, "asim1_reference"
    ):
        parts.append(
            simulate.eval_utilities(
                state, spec, chunk_choosers, constants, trace_label=label, chunk_sizer=sizer
            )
        )
    utils = pd.concat(parts)
    probs = asim_logit.utils_to_probs(state, utils.copy(), trace_label="asim1_reference")
    return utils, probs
