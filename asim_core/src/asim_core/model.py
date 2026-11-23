"""
A model's steps, built from its settings files: which of ActivitySim 1.x's
steps asim_core runs, and how each is made from an example's configs.

``build_steps`` is what the harness and the multiprocess runner both use, so a
worker process (started fresh, as Windows starts processes) builds exactly
the steps the main process would.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

from .component import Component
from .components import (
    atwork_subtour_destination,
    atwork_subtour_frequency,
    atwork_subtour_mode_choice,
    atwork_subtour_scheduling,
    auto_ownership,
    cdap_simulate,
    compute_accessibility,
    free_parking,
    initialize_households,
    initialize_landuse,
    joint_tour_composition,
    joint_tour_destination,
    joint_tour_frequency,
    joint_tour_participation,
    joint_tour_scheduling,
    mandatory_tour_frequency,
    mandatory_tour_scheduling,
    non_mandatory_tour_destination,
    non_mandatory_tour_frequency,
    non_mandatory_tour_scheduling,
    school_location,
    stop_frequency,
    telecommute_frequency,
    tour_mode_choice,
    transit_pass_ownership,
    transit_pass_subsidy,
    trip_departure_choice,
    trip_destination,
    trip_mode_choice,
    trip_purpose,
    trip_purpose_and_destination,
    trip_scheduling,
    trip_scheduling_choice,
    work_from_home,
    workplace_location,
)
from .components.output import track_skim_usage, write_data_dictionary, write_tables
from .components.summarize import summarize
from .components.trip_matrices import write_trip_matrices
from .spec.translate import FormulaError

# every step asim_core can run, as factories taking the example's configs folders
STEPS = [initialize_landuse, initialize_households, compute_accessibility, school_location,
         workplace_location, work_from_home, transit_pass_subsidy, transit_pass_ownership, auto_ownership,
         free_parking, telecommute_frequency, cdap_simulate, mandatory_tour_frequency,
         mandatory_tour_scheduling, joint_tour_frequency, joint_tour_composition,
         joint_tour_participation, joint_tour_destination, joint_tour_scheduling,
         non_mandatory_tour_frequency, non_mandatory_tour_destination,
         non_mandatory_tour_scheduling, tour_mode_choice, atwork_subtour_frequency,
         atwork_subtour_destination, atwork_subtour_scheduling, atwork_subtour_mode_choice,
         stop_frequency, trip_purpose, trip_destination, trip_purpose_and_destination, trip_scheduling,
         trip_scheduling_choice, trip_departure_choice, trip_mode_choice,
         write_data_dictionary, track_skim_usage, write_trip_matrices, write_tables, summarize]
# the step each factory makes, where the names differ
STEP_NAMES = {auto_ownership: "auto_ownership_simulate", tour_mode_choice: "tour_mode_choice_simulate"}
# steps that write files instead of tables
OUTPUT_STEPS = {"write_data_dictionary", "track_skim_usage", "write_trip_matrices", "write_tables", "summarize"}
# steps that read the input files: they also take the data folders (and the households sample size)
READS_INPUTS = {initialize_landuse, initialize_households}


def step_name(make) -> str:
    return STEP_NAMES.get(make, make.__name__)


def build_steps(configs: Sequence[str | Path], data_dirs: Sequence[str | Path],
                households_sample_size: int | None = None,
                not_built: dict[str, str] | None = None, only: Collection[str] | None = None) -> list[Component]:
    """
    The steps asim_core can run for these configs (``only``: just these
    steps, e.g. a worker process's). A step whose configs use something
    asim_core doesn't support yet is left out, with the reason in
    ``not_built`` (factory name -> first line of the error).
    """
    configs = [Path(c) for c in configs]
    data_dirs = [Path(d) for d in data_dirs]
    out = []
    for make in STEPS:
        if only is not None and step_name(make) not in only:
            continue
        try:
            if make is initialize_households:
                out.append(make(configs, data_dirs, households_sample_size=households_sample_size))
            elif make in READS_INPUTS:
                out.append(make(configs, data_dirs))
            else:
                out.append(make(configs))
        except (NotImplementedError, FormulaError, ValueError, FileNotFoundError) as err:
            if not_built is not None:
                not_built[make.__name__] = str(err).splitlines()[0][:300]
    return out
