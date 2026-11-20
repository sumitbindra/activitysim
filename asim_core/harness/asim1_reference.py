"""
Utilities and probabilities computed by ActivitySim 1.x itself, for checking
asim_core below the level of final choices.

Restores 1.x's saved state just before a step, then calls 1.x's own functions
on the same choosers (legacy engine): ``eval_utilities``, then
``utils_to_probs`` for multinomial logit, or ``compute_nested_exp_utilities``,
``compute_nested_probabilities`` and ``compute_base_probabilities`` for nested
logit. Harness code: imports ActivitySim 1.x.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    nest_spec = config.get_logit_model_settings(settings)
    if nest_spec is None:
        probs = asim_logit.utils_to_probs(state, utils.copy(), trace_label="asim1_reference")
    else:
        nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients, "asim1_reference")
        exp_u = simulate.compute_nested_exp_utilities(utils, nest_spec)
        nested = simulate.compute_nested_probabilities(state, exp_u, nest_spec, "asim1_reference")
        probs = simulate.compute_base_probabilities(nested, nest_spec, spec)
    return utils, probs


def scheduling_round_utilities(
    example_dir: Path,
    configs: list[str],
    pipeline: Path,
    resume_after: str,
    scratch_dir: Path,
    tour_num: int = 1,
    settings_file: str = "non_mandatory_tour_scheduling.yaml",
    category: str = "non_mandatory",
    rows_per_call: int | None = None,
) -> pd.Series:
    """
    The utility of every (tour, departure-return option) row of one
    scheduling round, as 1.x computes it: its preprocessor, its
    tdd_interaction_dataset (options that fit each person's day), the
    previous-tour columns, then eval_interaction_utilities on the whole
    round at once (1.x's default, without chunking). Round 1 only needs no
    earlier choices, so ``tour_num`` is normally 1. Unsegmented specs only.
    ``rows_per_call`` evaluates the rows in slices of that size instead (as
    1.x's chunking would). Returns a Series indexed by (tour_id, tdd).
    """
    from activitysim import abm  # noqa: F401
    from activitysim.abm.models.util import vectorize_tour_scheduling as vts
    from activitysim.core import config, expressions, interaction_simulate, workflow

    scratch_dir.mkdir(parents=True, exist_ok=True)
    state = workflow.State.make_default(
        working_dir=example_dir, configs_dir=tuple(configs), data_dir=("data",), output_dir=scratch_dir,
    )
    state.checkpoint.restore_from(Path(pipeline), resume_after)
    settings = vts.TourSchedulingSettings.read_settings_file(state.filesystem, settings_file, mandatory=False)
    if settings.TOUR_SPEC_SEGMENTS:
        raise NotImplementedError("segmented specs are not supported by this check")
    tours = state.get_dataframe("tours")
    persons_merged = state.get_dataframe("persons_merged")
    tdd_alts = state.get_injectable("tdd_alts")
    timetable = state.get_injectable("timetable")
    chosen = tours[tours.tour_category == category].copy()
    constants = config.get_model_constants(settings) or {}
    locals_d = {"tt": timetable.attach_state(state), **constants}
    expressions.annotate_preprocessors(state, df=chosen, locals_dict=locals_d, skims=None,
                                       model_settings=settings, trace_label="asim1_reference")
    spec = state.filesystem.read_model_spec(file_name=settings.SPEC)
    coefficients = state.filesystem.read_model_coefficients(settings)
    from activitysim.core import simulate

    spec = simulate.eval_coefficients(state, spec, coefficients, None)
    nth = chosen[chosen.tour_num == tour_num].sort_index()
    merged = pd.merge(nth, persons_merged, left_on="person_id", right_index=True, suffixes=("", "_y"))
    alt_tdd = vts.tdd_interaction_dataset(state, merged, tdd_alts, timetable, "tdd", "person_id",
                                          "asim1_reference")
    alt_tdd["mode_choice_logsum"] = 0
    previous = pd.Series(tdd_alts.index[0], index=chosen.person_id.unique())
    merged = merged.join(vts.get_previous_tour_by_tourid(merged["person_id"], previous, tdd_alts))
    interaction_df = alt_tdd.join(merged, how="left", rsuffix="_chooser")
    step = rows_per_call or len(interaction_df)
    values = []
    for start in range(0, len(interaction_df), step):
        utils, _ = interaction_simulate.eval_interaction_utilities(
            state, spec, interaction_df.iloc[start:start + step],
            {"tt": timetable.attach_state(state), **constants}, "asim1_reference", None,
        )
        values.append(utils.utility.to_numpy())
    return pd.Series(np.concatenate(values),
                     index=pd.MultiIndex.from_arrays([interaction_df.index.to_numpy(),
                                                      interaction_df["tdd"].to_numpy()],
                                                     names=["tour_id", "tdd"]))


def capture_scheduling_logsums(
    example_dir: Path,
    configs: list[str],
    pipeline: Path,
    step: str,
    scratch_dir: Path,
) -> pd.DataFrame:
    """
    Run one scheduling step of 1.x from its saved state, recording every mode
    choice logsum it computes for a (tour, option) row. Returns a DataFrame
    with tour_id, tdd and logsum (one row per tour and option, all rounds).
    """
    import shutil

    from activitysim import abm  # noqa: F401
    from activitysim.abm.models.util import vectorize_tour_scheduling as vts
    from activitysim.core import workflow

    from asim_core.asim1 import Asim1State

    prev = Asim1State(pipeline).previous_step(step)
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)
    shutil.copytree(pipeline, scratch_dir / "pipeline.parquetpipeline")
    captured = []
    original = vts.compute_tour_scheduling_logsums

    def recording(state, alt_tdd, tours_merged, tour_purpose, model_settings, trace_label, **kw):
        logsums = original(state, alt_tdd, tours_merged, tour_purpose, model_settings, trace_label, **kw)
        captured.append(pd.DataFrame({"tour_id": alt_tdd.index.to_numpy(),
                                      "tdd": alt_tdd["tdd"].to_numpy(),
                                      "logsum": np.asarray(logsums, dtype=np.float64)}))
        return logsums

    vts.compute_tour_scheduling_logsums = recording
    try:
        state = workflow.State.make_default(working_dir=example_dir, configs_dir=tuple(configs),
                                            data_dir=("data",), output_dir=scratch_dir)
        state.checkpoint.restore(prev)
        state.run(models=[step], resume_after=prev)
        state.checkpoint.close_store()
    finally:
        vts.compute_tour_scheduling_logsums = original
    return pd.concat(captured, ignore_index=True)
