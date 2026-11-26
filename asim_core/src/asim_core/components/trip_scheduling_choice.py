"""
Trip scheduling choice: how each tour's time is split between its outbound
stops, the main leg and its inbound stops (1.x:
``abm/models/trip_scheduling_choice.py``; ARC uses it, with trip departure
choice, instead of trip scheduling).

As 1.x:

1. Each tour gets its number of stops each way, the origin of its last
   outbound trip and the destination of its first inbound one; the
   preprocessor adds the legs' distances and times (``od_skims``,
   ``do_skims``, ``obib_skims``: from the last outbound stop to the first
   inbound one), the person type and the stops by purpose.
2. A tour without stops spends its whole duration on the main leg. For a
   tour with stops, every way of splitting its duration (whole periods)
   between the main leg and the leg(s) with stops is an alternative
   (``schedule_id``); one is chosen with the spec's ``stage_one`` column
   (1.x's interaction_sample_simulate: options padded to the tour with the
   most), one draw per tour.
3. ``tours`` gains ``main_leg_duration``, ``outbound_duration``,
   ``inbound_duration``, ``has_outbound_stops``, ``has_inbound_stops`` and
   the preprocessor's columns; ``duration`` becomes 8-bit, as 1.x leaves it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, MissingInputError, StepContext
from ..logit import FormulaValues
from ..spec.legacy import load_global_constants, load_interaction_model, load_top_settings, read_settings
from ..spec.translate import INDEX
from ..store import Table
from . import annotate
from .runtime import TableLookup, WrappedSkims, plain

STEP = "trip_scheduling_choice"
CHANNEL = "tours"
PAD_UTILITY = -999.0
SEGMENT = "stage_one"
MAIN, OB, IB = "main_leg_duration", "outbound_duration", "inbound_duration"
NUM_OB, NUM_IB = "num_outbound_stops", "num_inbound_stops"
HAS_OB, HAS_IB = "has_outbound_stops", "has_inbound_stops"
LAST_OB, FIRST_IB = "last_outbound_stop", "last_inbound_stop"  # 1.x's names
TEMP_COLS = [NUM_OB, LAST_OB, NUM_IB, FIRST_IB]
SKIM_ROLES = {"od_skims": ("origin", "destination"), "do_skims": ("destination", "origin"),
              "obib_skims": (LAST_OB, FIRST_IB)}
TOUR_COLUMNS = ["origin", "destination", "duration", "start", "tour_type", "person_id"]
TRIP_COLUMNS = ["tour_id", "outbound", "trip_num", "origin", "destination"]


def trip_scheduling_choice(configs_dirs: Sequence[str | Path],
                           settings_file: str = "trip_scheduling_choice.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_settings(configs_dirs, settings_file)
    top = load_top_settings(configs_dirs)
    for key in ("alts_preprocessor", "annotate_tours", "annotate_trips", "annotate_persons", "annotate_households"):
        if settings.get(key):
            raise NotImplementedError(f"{settings_file}: {key} is not supported yet")
    if settings.get("LOGIT_TYPE", "MNL") != "MNL":
        raise NotImplementedError(f"{settings_file}: only a multinomial logit is supported")
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    constants.update({"orig_col_name": "origin", "dest_col_name": "destination",
                      "timeframe": "timeless_directional"})
    definition = load_interaction_model(configs_dirs, settings["SPEC"], settings.get("COEFFICIENTS"), constants,
                                        name=STEP, value_column=SEGMENT, skim_accessors=list(SKIM_ROLES))
    terms = definition.utility_terms()
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, settings.get("preprocessor"), constants, top,
                                                      STEP, "preprocessor", skim_accessors=list(SKIM_ROLES))
    shift = not top.get("skip_failed_choices", True)  # as 1.x's utils_to_probs
    resolved: dict = {}

    def resolve_needs(available):
        tour_cols = set(available.get("tours", set()))
        missing = [c for c in TOUR_COLUMNS if c not in tour_cols]
        if missing:
            raise MissingInputError(f"tours lack {missing}")
        missing = [c for c in TRIP_COLUMNS if c not in available.get("trips", set())]
        if missing:
            raise MissingInputError(f"trips lack {missing}")
        df_cols = tour_cols | set(TEMP_COLS)
        needs: dict[str, set[str]] = {"tours": set(tour_cols), "trips": set(TRIP_COLUMNS)}
        skims: set[str] = set()
        bound = None
        targets: set[str] = set()
        if pre_spec is not None:
            bound = pre_spec.bind(df_cols, {t: available.get(t, set()) for t in pre_tables})
            if bound.n_random:
                raise NotImplementedError(f"{STEP}: preprocessor with random draws")
            targets = set(bound.targets)
            for t, c in bound.tables_used():
                needs.setdefault(t, set()).add(c)
            skims |= {name for _, name, _ in bound.skim_uses()}
        alt_cols = {MAIN, OB, IB, "schedule_id"}
        unknown = definition.columns_used() - alt_cols - df_cols - targets - {HAS_OB, HAS_IB}
        if unknown:
            raise MissingInputError(f"{STEP}: the spec reads {sorted(unknown)}, which the tours lack")
        skims |= {name for _, name, _ in definition.skim_uses()}
        if skims:
            needs[SKIMS] = skims
        resolved.update(bound=bound)
        return {t: sorted(c) for t, c in needs.items()}

    def resolve_provides(available):
        tours = list(available.get("tours", []))
        new = (list(pre_spec.targets) if pre_spec is not None else []) + [HAS_OB, HAS_IB, MAIN, IB, OB]
        return {"tours": tours + [c for c in new if c not in tours]}

    def run(ctx: StepContext):
        tours = ctx.table_frame("tours")
        trips = ctx.table_frame("trips")
        out_trips = trips[trips.outbound]
        in_trips = trips[~trips.outbound]
        last_out = trips.loc[out_trips.groupby("tour_id")["trip_num"].idxmax()]
        first_in = trips.loc[in_trips.groupby("tour_id")["trip_num"].idxmin()]
        tours[NUM_OB] = out_trips.groupby("tour_id").size().reindex(tours.index) - 1
        tours[NUM_IB] = in_trips.groupby("tour_id").size().reindex(tours.index) - 1
        tours[LAST_OB] = last_out[["tour_id", "origin"]].set_index("tour_id").reindex(tours.index)["origin"]
        tours[FIRST_IB] = first_in[["tour_id", "destination"]].set_index("tour_id").reindex(tours.index)["destination"]

        bound = resolved["bound"]
        if bound is not None and len(tours):
            c = {k: np.asarray(plain(tours[k].to_numpy())) for k in bound.columns_used() - {INDEX}}
            c[INDEX] = tours.index.to_numpy()
            s = WrappedSkims(ctx, {a: (tours[o].to_numpy(), tours[d].to_numpy()) for a, (o, d) in SKIM_ROLES.items()})
            with np.errstate(all="ignore"):
                v = bound.function(c, s, len(tours), TableLookup(ctx), None, {})
            for target in bound.targets:
                tours[target] = annotate._update(tours[target].to_numpy() if target in tours else None,
                                                 np.asarray(v[target]))

        tours = schedule_legs(ctx, tours, terms, shift)
        table = Table.from_pandas("tours", tours.drop(columns=TEMP_COLS))
        ctx.create_table(table)
        return {"tours": table}

    return Component(STEP, run, provides={"tours": [MAIN, OB, IB, HAS_OB, HAS_IB]}, resolve_needs=resolve_needs,
                     resolve_provides=resolve_provides,
                     description="how each tour's duration splits between its legs with stops and the main leg",
                     info={"definition": definition, "generated_source": terms[2], "trace_table": "tours"})


def schedule_legs(ctx, tours: pd.DataFrame, terms, shift) -> pd.DataFrame:
    """1.x's run_trip_scheduling_choice."""
    tours["duration"] = tours["duration"].astype(np.int8)
    tours[HAS_OB] = tours[NUM_OB] >= 1
    tours[HAS_IB] = tours[NUM_IB] >= 1
    direct = ~tours[HAS_OB] & ~tours[HAS_IB]
    tours.loc[direct, MAIN] = tours["duration"]
    tours.loc[direct, [IB, OB]] = 0
    indirect = tours.loc[tours[HAS_OB] | tours[HAS_IB]].sort_index()
    if len(indirect):
        schedules = schedule_alternatives(indirect).sort_index()
        terms_fn, coefs, _ = terms
        rows = indirect.index.get_indexer(schedules.index)
        n = len(schedules)
        c = {col: schedules[col].to_numpy() for col in (MAIN, OB, IB, "schedule_id")}
        for col in indirect.columns:
            if col not in c:
                c[col] = np.asarray(plain(indirect[col].to_numpy()))[rows]
        s = WrappedSkims(ctx, {a: (c[o], c[d]) for a, (o, d) in SKIM_ROLES.items()})
        with np.errstate(all="ignore"):
            utils = ctx.engine.interaction_utilities(FormulaValues(terms_fn, c, n, None, s), coefs, n)
        counts = np.bincount(rows, minlength=len(indirect))
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        padded = np.full((len(indirect), int(counts.max())), PAD_UTILITY)
        padded[rows, np.arange(n) - offsets[rows]] = utils
        probs, bad = ctx.engine.probabilities(padded, shift=shift)
        if bad.any():
            raise ValueError(f"{STEP}: tours with no usable probabilities, e.g. {indirect.index[bad][:5].tolist()}")
        positions = ctx.engine.choose(probs, ctx.random.uniform(CHANNEL, indirect.index.to_numpy()))
        chosen = schedules.iloc[offsets + positions]
        tours.update(chosen[[MAIN, OB, IB]])
    tours[[MAIN, OB, IB]] = tours[[MAIN, OB, IB]].astype(np.int8)
    return tours


def get_time_windows(residual, level):
    """1.x's util/trip.get_time_windows: every way to split ``residual`` periods into ``level`` parts."""
    ranges = []
    for a in np.arange(residual + 1):
        if level > 1:
            windows = get_time_windows(residual - a, level - 1)
            width_dim = len(windows.shape) - 1
            ranges.append(np.vstack([np.repeat(a, windows.shape[width_dim]), windows]))
        else:
            return np.arange(residual + 1)
    return np.concatenate(ranges, axis=1)


def _patterns(tour_index, durations, one_way: bool):
    """1.x's get_pattern_index_and_arrays."""
    max_columns = 2 if one_way else 3
    windows = get_time_windows(np.max(durations), max_columns)
    patterns, sizes = [], []
    for duration in durations:
        possible = windows[:max_columns, np.where(windows.sum(axis=0) == duration)[0]]
        possible = np.unique(possible, axis=1).transpose()
        patterns.append(possible)
        sizes.append(possible.shape[0])
    return np.repeat(tour_index, sizes), np.concatenate(patterns), sizes


def schedule_alternatives(tours: pd.DataFrame) -> pd.DataFrame:
    """1.x's generate_schedule_alternatives: each tour's possible (main, outbound, inbound) durations."""
    pattern = tours[HAS_OB].astype(int) + tours[HAS_IB].astype(int)
    parts = []
    none = tours[pattern == 0]
    if len(none):
        alts = none[["duration"]].rename(columns={"duration": MAIN})
        alts[[IB, OB]] = 0
        parts.append(alts.astype(int))
    one = tours[pattern == 1]
    if len(one):
        idx, pats, sizes = _patterns(one.index, one["duration"], one_way=True)
        direction = np.repeat(one[HAS_OB], sizes)
        inbound = np.where(direction == 0, pats[:, 1], 0)
        outbound = np.where(direction == 1, pats[:, 1], 0)
        df = pd.DataFrame(index=idx, data=np.column_stack((pats[:, 0], outbound, inbound)), columns=[MAIN, OB, IB])
        df.index.name = tours.index.name
        parts.append(df)
    two = tours[pattern > 1]
    if len(two):
        idx, pats, _ = _patterns(two.index, two["duration"], one_way=False)
        df = pd.DataFrame(index=idx, data=pats, columns=[MAIN, OB, IB])
        df.index.name = tours.index.name
        parts.append(df)
    schedules = pd.concat(parts, sort=True)
    schedules["schedule_id"] = np.arange(1, schedules.shape[0] + 1)
    schedules = schedules.reset_index().sort_values(by=[tours.index.name, "schedule_id"]).set_index(tours.index.name)
    return schedules
