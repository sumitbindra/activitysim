"""
Converting a region's ported steps from today's files to asim_core's model
files:

    python -m asim_core.converter -c configs --asim1-run output/pipeline.parquetpipeline --out new_configs

Converting needs to know which table each column comes from when a step
runs, so it reads the column names a 1.x run saved (only their names).
The converted files go to ``--out``; list that folder first among the
configs folders to run with them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .asim1 import Asim1State
import importlib

_accessibility = importlib.import_module(".components.accessibility", __package__)
_tmc = importlib.import_module(".components.tour_mode_choice", __package__)
from .components.auto_ownership import auto_ownership, convert as convert_auto_ownership
from .components.roles import PERSON_ROLES, TOUR_ROLES, role_columns
from .components.tour_destination import PERSONS_VIEW, non_mandatory_tour_destination
from .components.tour_scheduling import mandatory_tour_scheduling, non_mandatory_tour_scheduling


def _columns(state: Asim1State, step: str):
    return state.columns_at(state.previous_step(step))


def mode_choice_contexts(configs_dirs, state: Asim1State) -> list[dict]:
    """
    Where tour mode choice runs: its own step, then the steps using it for
    logsums, each with the chooser columns 1.x's preprocessor sees there and
    the table each column comes from (the step's view; anything else the step
    puts in the rows is the tour's, e.g. a scheduling option's times or a
    candidate destination).
    """
    out = []
    users = [
        ("tour_mode_choice_simulate", _tmc.tour_mode_choice, "model", _tmc.VIEW, TOUR_ROLES),
        ("mandatory_tour_scheduling", mandatory_tour_scheduling, "logsum_model", _tmc.VIEW, TOUR_ROLES),
        ("non_mandatory_tour_scheduling", non_mandatory_tour_scheduling, "logsum_model", _tmc.VIEW, TOUR_ROLES),
        ("non_mandatory_tour_destination", non_mandatory_tour_destination, "logsum_model", PERSONS_VIEW,
         PERSON_ROLES),
        ("school_location", _loc.school_location, "logsum_model", PERSONS_VIEW, PERSON_ROLES),
        ("workplace_location", _loc.workplace_location, "logsum_model", PERSONS_VIEW, PERSON_ROLES),
    ]
    for step, make, key, view, roles in users:
        if step not in state.steps:
            continue
        comp = make(configs_dirs)
        model = comp.info.get(key)
        if model is None:
            continue
        available = _columns(state, step)
        model.bind_log.clear()
        comp.resolve_needs(available)
        by_role = role_columns(view, roles, available)
        for df_columns, dest, orig in model.bind_log:
            lookup = {c: r for r, cs in by_role.items() for c in cs if c in df_columns}
            rest = {c: "tour" for c in df_columns if c not in lookup and c not in _tmc.ADDED}
            out.append({"step": step, "df_columns": df_columns, "dest": dest, "orig": orig,
                        "role_of": {**rest, **lookup}.get, "tables": available})
    return out


def _convert_tour_mode_choice(configs_dirs, state, out_dir):
    return _tmc.convert(configs_dirs, mode_choice_contexts(configs_dirs, state), out_dir)


def _convert_with_columns(convert):
    def run(configs_dirs, state, out_dir, step):
        return convert(configs_dirs, _columns(state, step), out_dir)
    return run


_sched = importlib.import_module(".components.tour_scheduling", __package__)


def _convert_scheduling(category, settings_file):
    def run(configs_dirs, state, out_dir, step):
        return _sched.convert(configs_dirs, _columns(state, step), out_dir, step, category, settings_file)
    return run


_dest = importlib.import_module(".components.tour_destination", __package__)
_loc = importlib.import_module(".components.location_choice", __package__)
_loc_files = importlib.import_module(".components.location_choice_files", __package__)


def _convert_location(settings_file):
    def run(configs_dirs, state, out_dir, step):
        return _loc_files.convert(configs_dirs, _columns(state, step), out_dir, step, settings_file)
    return run


STEPS = {
    "compute_accessibility": {"component": _accessibility.compute_accessibility,
                              "convert": _convert_with_columns(_accessibility.convert)},
    "school_location": {"component": _loc.school_location, "convert": _convert_location("school_location.yaml")},
    "workplace_location": {"component": _loc.workplace_location,
                           "convert": _convert_location("workplace_location.yaml")},
    "auto_ownership_simulate": {"component": auto_ownership,
                                "convert": _convert_with_columns(convert_auto_ownership)},
    "tour_mode_choice_simulate": {"component": _tmc.tour_mode_choice,
                                  "convert": lambda c, st, out, step: _convert_tour_mode_choice(c, st, out)},
    "mandatory_tour_scheduling": {"component": mandatory_tour_scheduling,
                                  "convert": _convert_scheduling("mandatory", "mandatory_tour_scheduling.yaml")},
    "non_mandatory_tour_destination": {
        "component": non_mandatory_tour_destination,
        "convert": _convert_with_columns(_dest.convert)},
    "non_mandatory_tour_scheduling": {
        "component": non_mandatory_tour_scheduling,
        "convert": _convert_scheduling("non_mandatory", "non_mandatory_tour_scheduling.yaml")},
}


def convert_step(step: str, configs_dirs: Sequence[str | Path], state: Asim1State, out_dir: str | Path
                 ) -> list[Path]:
    """Convert one step's files; ``state`` gives the tables' columns when steps run."""
    return STEPS[step]["convert"]([Path(d) for d in configs_dirs], state, out_dir, step)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m asim_core.converter")
    ap.add_argument("-c", "--configs", action="append", required=True)
    ap.add_argument("--asim1-run", required=True, help="a saved 1.x run (pipeline.parquetpipeline)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", nargs="*", default=list(STEPS))
    args = ap.parse_args(argv)
    st = Asim1State(args.asim1_run)
    for step in args.steps:
        written = convert_step(step, args.configs, st, args.out)
        print(f"{step}: {', '.join(str(p) for p in written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
