"""
Trip purpose and destination, again for failed trips (1.x:
``abm/models/trip_purpose_and_destination.py``).

1.x runs this step after trip destination to retry the trips that failed
(with their direction's other trips), choosing their purpose and
destination again, up to ``MAX_ITERATIONS`` times. When no trip failed, as
in every run checked so far, it only drops the trips' ``failed`` column;
that is what this component does. Retrying failed trips, and running the
step without trip destination first, are not supported yet (the step says
so if it meets them).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..component import Component, MissingInputError, StepContext
from ..spec.legacy import read_settings
from ..store import Table

STEP = "trip_purpose_and_destination"


def trip_purpose_and_destination(configs_dirs: Sequence[str | Path],
                                 settings_file: str = "trip_purpose_and_destination.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_settings(configs_dirs, settings_file)
    if int(settings.get("MAX_ITERATIONS", 5)) < 1:
        raise ValueError(f"{settings_file}: MAX_ITERATIONS must be at least 1")

    def resolve_needs(available):
        cols = set(available.get("trips", set()))
        if "destination" not in cols:
            raise NotImplementedError(f"{STEP} without trip_destination first is not supported yet")
        return {"trips": sorted(cols)}  # the table is rewritten whole (one column fewer)

    def resolve_provides(available):
        return {"trips": [c for c in available.get("trips", []) if c != "failed"]}

    def run(ctx: StepContext):
        trips = ctx.table_frame("trips")
        if "failed" not in trips.columns:
            return {}  # 1.x: nothing to do
        if trips.failed.any():
            raise NotImplementedError(f"{STEP}: {int(trips.failed.sum())} trips failed in trip_destination; "
                                      f"choosing them again is not supported yet")
        table = Table.from_pandas("trips", trips.drop(columns="failed"))
        ctx.create_table(table)
        return {"trips": table}

    return Component(STEP, run, provides={"trips": []}, resolve_needs=resolve_needs,
                     resolve_provides=resolve_provides,
                     description="drops the trips' failed column (retrying failed trips: not supported yet)",
                     info={"trace_table": "trips"})
