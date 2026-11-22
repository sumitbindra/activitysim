"""
Trip mode choice: a mode for every trip, and the trip's mode choice logsum
(1.x: ``abm/models/trip_mode_choice.py``).

The model is the tour mode choice model (``TourModeChoiceModel``) read from
``trip_mode_choice.yaml``, with the trips' skim lookups: ``odt_skims`` /
``dot_skims`` at the trip's period (``trip_period``, from its departure
time), ``od_skims`` without one; and the names 1.x gives the trip formulas
(``ORIGIN``, ``DESTINATION``: the columns with the trip's zones;
``MIN_PER_PERIOD``, ``PERIODS_PER_HOUR``). Trip destination uses the same
model for its logsums, with other origin and destination columns.

As 1.x: choosers are the trips with the columns of their tour (and the
tour's person, household, home zone) listed in
``TOURS_MERGED_CHOOSER_COLUMNS``; per primary purpose (in the category's
order) the preprocessor runs (random waiting times included), then the
nested logit chooses; ``trips.trip_mode`` is a category of '' and the
modes, in the spec's order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, MissingInputError, Needs, StepContext
from ..spec.legacy import find_config_file, read_settings, read_yaml
from .runtime import plain
from .simple_choice import ChooserView
from .tour_mode_choice import TourModeChoiceModel

STEP = "trip_mode_choice"
CHANNEL = "trips"
ORIG, DEST = "origin", "destination"
PERIOD = "trip_period"
# 1.x's trip skim wrappers: (from, to, period column)
TRIP_SKIM_WRAPPERS = {
    "odt_skims": ("orig", "dest", PERIOD),
    "dot_skims": ("dest", "orig", PERIOD),
    "od_skims": ("orig", "dest", None),
}
TOURS_VIEW = ChooserView("tours", [  # 1.x's tours_merged
    ("persons", "person_id"),
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])


def trip_names(dest: str, orig: str) -> dict:
    """The names 1.x gives trip mode choice formulas for the origin and destination columns."""
    return {"ORIGIN": orig, "DESTINATION": dest}


def trip_mode_choice_model(configs_dirs: Sequence[str | Path],
                           settings_file: str = "trip_mode_choice.yaml") -> TourModeChoiceModel:
    return TourModeChoiceModel(configs_dirs, settings_file, skim_wrappers=TRIP_SKIM_WRAPPERS, names=trip_names,
                               orig=ORIG, dest=DEST, name="trip_mode_choice", choice_column="trip_mode")


def period_values(configs_dirs) -> dict:
    """MIN_PER_PERIOD and PERIODS_PER_HOUR, as 1.x's trip mode choice gives them."""
    los = read_settings([Path(d) for d in configs_dirs], "network_los.yaml")
    minutes = los.get("skim_time_periods", {}).get("period_minutes", 60)
    return {"MIN_PER_PERIOD": minutes, "PERIODS_PER_HOUR": 60 / minutes}


def tours_merged_columns(settings: Mapping, trip_columns: set[str]) -> list[str]:
    """The tours_merged columns trip models merge onto the trips (those the trips don't have)."""
    cols = list(settings.get("TOURS_MERGED_CHOOSER_COLUMNS") or [])
    return [c for c in cols if c not in trip_columns]


def trip_mode_choice(configs_dirs: Sequence[str | Path], settings_file: str = "trip_mode_choice.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_settings(configs_dirs, settings_file)
    for key in ("CHOOSER_COLS_TO_KEEP",):
        if settings.get(key):
            raise NotImplementedError(f"{settings_file}: {key} is not supported yet")
    model = trip_mode_choice_model(configs_dirs, settings_file)
    alternatives = model.alternatives
    logsum_column = settings.get("MODE_CHOICE_LOGSUM_COLUMN_NAME", "mode_choice_logsum")
    names = {"ORIGIN": ORIG, "DESTINATION": DEST, **period_values(configs_dirs)}
    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]) -> Needs:
        trip_cols = set(available.get("trips", set()))
        missing = [c for c in ("tour_id", "primary_purpose", "depart", ORIG, DEST) if c not in trip_cols]
        if missing:
            raise MissingInputError(f"trips lack {missing}")
        tour_cols = tours_merged_columns(settings, trip_cols)
        offered = TOURS_VIEW.columns(available)
        absent = [c for c in tour_cols if c not in offered]
        if absent:
            raise MissingInputError(f"TOURS_MERGED_CHOOSER_COLUMNS {absent} are not columns of tours_merged")
        if PERIOD in trip_cols:
            raise ValueError(f"trips already have {PERIOD} (1.x expects to add it)")
        df_cols = trip_cols | set(tour_cols) | {PERIOD}
        bound = model.bind(df_cols, available, dest=DEST, orig=ORIG, names=names)
        columns = bound.columns - {PERIOD}
        needs: dict[str, set[str]] = {"trips": ({c for c in columns if c in trip_cols}
                                                | {"tour_id", "primary_purpose", "depart", ORIG, DEST})}
        from_tours = sorted(c for c in columns if c not in trip_cols)
        tours_view = TOURS_VIEW.resolve(set(from_tours), available) if from_tours else {}
        for t, c in tours_view.items():
            needs.setdefault(t, set()).update(c)
        if from_tours:
            needs.setdefault("tours", set())
        for t, c in bound.tables:
            needs.setdefault(t, set()).add(c)
        needs[SKIMS] = set(bound.skims)
        resolved.update(bound=bound, tours_view=tours_view, from_tours=from_tours,
                        needs={t: sorted(c) for t, c in needs.items()})
        return resolved["needs"]

    def run(ctx: StepContext):
        bound = resolved["bound"]
        ids = ctx.ids("trips")
        cols = {c: np.asarray(plain(ctx.column("trips", c))) for c in resolved["needs"]["trips"]}
        cols.update(merged_tour_columns(ctx, resolved["tours_view"], resolved["from_tours"], cols["tour_id"]))
        cols[PERIOD] = model.periods.labels_for(cols["depart"])
        primary = ctx.column("trips", "primary_purpose")
        order = (list(primary.categories) if isinstance(primary, pd.Categorical)
                 else sorted(set(np.asarray(primary, dtype=object))))
        purposes = np.asarray(plain(primary), dtype=object)

        choices = np.full(len(ids), -1, dtype=np.int64)
        logsums = np.full(len(ids), np.nan)
        for purpose in order:  # 1.x: groupby on the category (observed values, category order)
            rows = np.flatnonzero(purposes == purpose)
            if not len(rows):
                continue
            seg_ids = ids[rows]
            c = {k: v[rows] for k, v in cols.items()}
            utils, probs, bad, seg_logsums, _ = model.evaluate(ctx, bound, c, len(rows), purpose, seg_ids,
                                                                channel=CHANNEL)
            if bad.any():
                raise ValueError(f"{STEP}: {int(bad.sum())} '{purpose}' trips have no usable probabilities, "
                                 f"e.g. {seg_ids[bad][:5].tolist()}")
            rands = ctx.random.uniform(CHANNEL, seg_ids)
            choices[rows] = ctx.engine.choose(probs, rands)
            logsums[rows] = seg_logsums
        if (choices < 0).any():
            raise ValueError(f"{STEP}: trips without a mode (primary purposes with no coefficients?)")
        modes = pd.Categorical(np.asarray(alternatives, dtype=object)[choices], categories=[""] + list(alternatives))
        out = {model.choice_column: modes}
        if logsum_column:
            out[logsum_column] = logsums
        return {"trips": out}

    return Component(
        name=STEP, run=run,
        provides={"trips": [model.choice_column] + ([logsum_column] if logsum_column else [])},
        resolve_needs=resolve_needs,
        description=f"trip mode choice: nested logit over {len(alternatives)} modes, "
                    f"{len(model.segments)} purpose segments",
        info={"model": model, "definition": model.definition, "preprocessor": model.preprocessor,
              "generated_source": model.generated_source, "trace_table": "trips", "resolved": resolved},
    )


def merged_tour_columns(ctx: StepContext, tours_view, columns, tour_ids) -> dict[str, np.ndarray]:
    """tours_merged's columns for each trip's tour (1.x: merged on tour_id; missing if the tour isn't there)."""
    if not columns:
        return {}
    gathered = TOURS_VIEW.gather(ctx, tours_view)
    pos = pd.Index(ctx.ids("tours")).get_indexer(np.asarray(tour_ids))
    if (pos < 0).any():
        raise ValueError(f"trips of tours that aren't in the tours table: {np.asarray(tour_ids)[pos < 0][:5].tolist()}")
    return {c: np.asarray(plain(gathered[c]))[pos] for c in columns}
