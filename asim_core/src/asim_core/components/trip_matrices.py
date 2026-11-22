"""
Trip matrices: the trips added up by origin and destination into OMX
files for assignment (1.x: ``abm/models/trip_matrices.py``,
``write_trip_matrices``).

Reads today's write_trip_matrices.yaml. The trips get the preprocessor's
columns first (MTC: a count per mode and period, by tour participants and
occupancy), with origin-destination skims (``od_skims``), the trip's period's
skims (``odt_skims``; the period from ``depart`` if the trips have none, as
1.x) and the tables it lists. Each trip is weighted by its household's ``HH_EXPANSION_WEIGHT_COL``
(default ``sample_rate``): a cell is the sum of the trips' values divided
by the mean sample rate of its trips, as 1.x (the same pandas calls, so the
same floating-point results). Tables with the same name in one file are
added up. Each file gets one zone mapping: the land use's original zone
IDs if it has them (``_original_zone_id``), else its zone IDs.

Not supported yet (reported before the run): ``SAVE_TRIPS_TABLE`` (the
annotated trips written back), parking location choice's parking zones,
two- and three-zone systems.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import Component, MissingInputError, StepContext
from ..omx import write_omx
from ..spec.legacy import find_config_file, load_global_constants, load_top_settings, read_yaml
from ..spec.translate import INDEX
from ..skims import time_periods_from_settings
from .annotate import _update, load_preprocessor
from .runtime import TableLookup, plain
from .tour_mode_choice import SkimLookup

PERIOD = "trip_period"
# 1.x's annotate_trips: skim_dict.wrap("origin", "destination") and wrap_3d(..., dim3_key="trip_period")
SKIMS = {"od_skims": ("orig", "dest", None), "odt_skims": ("orig", "dest", PERIOD)}

STEP = "write_trip_matrices"


def write_trip_matrices(configs_dirs: Sequence[str | Path],
                        settings_file: str = "write_trip_matrices.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    unsupported = []
    if settings.get("SAVE_TRIPS_TABLE"):
        unsupported.append("SAVE_TRIPS_TABLE (the annotated trips written back)")
    if "parking_location" in (top.get("models") or []):
        unsupported.append("parking location choice's parking zones")
    los = read_yaml(find_config_file(configs_dirs, "network_los.yaml"))
    if int(los.get("zone_system", 1)) != 1:
        unsupported.append("two- and three-zone systems")
    if unsupported:
        raise NotImplementedError(f"{settings_file}: not supported yet: {', '.join(unsupported)}")
    weight_col = settings.get("HH_EXPANSION_WEIGHT_COL", "sample_rate")
    matrices = []  # (file name, [(name, data field, origin column, destination column)])
    for m in settings.get("MATRICES") or []:
        tables = [(t["name"], t["data_field"], t.get("origin") or "origin", t.get("destination") or "destination")
                  for t in m.get("tables") or []]
        matrices.append((str(m["file_name"]), tables))
    if not matrices:
        raise ValueError(f"{settings_file}: no MATRICES (1.x writes nothing then)")
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    pre_spec, pre_tables = load_preprocessor(configs_dirs, settings.get("preprocessor"), constants, top, STEP,
                                              "preprocessor", skim_accessors=list(SKIMS))
    periods = time_periods_from_settings(configs_dirs)
    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]):
        trip_cols = set(available.get("trips", set()))
        for t in ("trips", "households", "land_use"):
            if t not in available:
                raise MissingInputError(f"needs the {t} table")
        needs: dict[str, set[str]] = {"trips": set(), "households": {weight_col}, "land_use": set()}
        made: set[str] = set()
        bound = None
        add_period = PERIOD not in trip_cols  # 1.x labels each trip's departure with its skim period
        if pre_spec is not None:
            bound = pre_spec.bind(trip_cols | {PERIOD}, {t: available.get(t, set()) for t in pre_tables})
            if bound.n_random:
                raise NotImplementedError(f"{STEP}: a preprocessor with random draws")
            made = set(bound.targets)
            needs["trips"] |= bound.columns_used() - {INDEX}
            uses = bound.skim_uses()
            if any(how != "get" for _, _, how in uses):
                raise NotImplementedError(f"{STEP}: only skim lookups by [key] in the preprocessor")
            if uses:
                needs["trips"] |= {"origin", "destination"}
                needs["skims"] = set()
                for accessor, key, _ in uses:
                    if SKIMS[accessor][2] is None:
                        needs["skims"].add(key)
                    else:
                        needs["skims"] |= {f"{key}__{label}" for label in periods.unique_labels}
                        needs["trips"].add(PERIOD)
            for t, c in bound.tables_used():
                needs.setdefault(t, set()).add(c)
        if add_period and PERIOD in needs["trips"]:
            needs["trips"].discard(PERIOD)
            needs["trips"].add("depart")
        missing = []
        for _, tables in matrices:
            for name, field, o, d in tables:
                for c in (field, o, d):
                    if c in made:
                        continue
                    if c in trip_cols:
                        needs["trips"].add(c)
                    else:  # 1.x skips the table with a warning
                        missing.append(f"{name}: '{c}'")
        if weight_col not in available["households"]:
            raise MissingInputError(f"households lack '{weight_col}' (HH_EXPANSION_WEIGHT_COL)")
        if weight_col in trip_cols:
            needs["trips"].add(weight_col)
        elif weight_col not in made:
            needs["trips"].add("household_id")  # the household's weight, as 1.x maps it
        # 1.x adjusts the weights for households it skipped (skip_failed_choices)
        skipped = "households_skipped" in available and weight_col in available["households_skipped"]
        if skipped:
            needs["households_skipped"] = {weight_col}
        # the zone mapping: the original zone IDs if the land use has them
        needs["land_use"] |= {c for c in available["land_use"] if c.startswith("_original_")}
        resolved.update(bound=bound, missing=missing, made=made, skipped=skipped,
                        add_period=add_period and "depart" in needs["trips"])
        return {t: sorted(c) for t, c in needs.items()}

    def run(ctx: StepContext):
        ids = ctx.ids("trips")
        n = len(ids)
        cols = {c: np.asarray(plain(ctx.column("trips", c))) for c in ctx.declared("trips")}
        if resolved["add_period"]:
            cols[PERIOD] = periods.labels_for(cols["depart"])
        bound = resolved["bound"]
        if bound is not None and n:
            c = {k: cols[k] for k in bound.columns_used() - {INDEX}}
            c[INDEX] = ids
            skims = None
            if "origin" in cols and "destination" in cols:
                skims = SkimLookup(ctx, periods, ctx.skim_positions(cols["origin"]),
                                   ctx.skim_positions(cols["destination"]),
                                   {PERIOD: cols[PERIOD]} if PERIOD in cols else {}, wrappers=SKIMS)
            with np.errstate(all="ignore"):  # 1.x logs numpy warnings and carries on
                v = bound.function(c, skims, n, TableLookup(ctx), None, {})
            for target in bound.targets:
                cols[target] = _update(cols.get(target), np.asarray(v[target]))
        for warning in resolved["missing"]:
            ctx.warn(f"skipped table {warning} is not a column of the trips")
        trips = pd.DataFrame({k: v for k, v in cols.items() if np.ndim(v)}, index=pd.Index(ids, name="trip_id"))
        for k, v in cols.items():
            if not np.ndim(v):  # a constant from the preprocessor
                trips[k] = v

        if weight_col not in trips:  # 1.x: trips_df.household_id.map(households[weight])
            weights = pd.Series(np.asarray(plain(ctx.column("households", weight_col))), index=ctx.ids("households"))
            trips[weight_col] = trips.household_id.map(weights)
        rates = pd.Series(np.asarray(plain(ctx.column("households", weight_col))), index=ctx.ids("households"))
        zero = rates == 0
        if zero.all():
            ctx.warn(f"every household has {weight_col} 0: trips counted unweighted")
        elif zero.any():
            ctx.warn(f"some households have {weight_col} 0: their trips are left out")
            trips = trips[trips[weight_col] != 0]
            if trips.empty:
                return {}
        if resolved["skipped"]:
            skipped = pd.Series(np.asarray(plain(ctx.column("households_skipped", weight_col))))
            if len(skipped):
                if zero.all():
                    factor = len(rates) / (len(skipped) + len(rates))
                else:
                    kept_w = (1 / rates[rates != 0]).sum()
                    factor = kept_w / (kept_w + (1 / skipped[skipped != 0]).sum())
                trips[weight_col] = trips[weight_col] * factor

        zones = pd.Index(ctx.ids("land_use"))
        index_name = ctx.index_name("land_use")
        original = f"_original_{index_name}"
        if original in ctx.declared("land_use"):
            labels = pd.Series(np.asarray(plain(ctx.column("land_use", original))), index=zones, name=original)
        else:
            labels = pd.Index(zones, name=index_name)
        for file_name, tables in matrices:
            data = _matrices(trips, tables, labels, weight_col)
            mapping_name = labels.name or "ZONE"
            write_omx(ctx.output_path(file_name), data, {mapping_name: pd.Index(labels).to_numpy()})
        return {}

    return Component(STEP, run, provides={}, resolve_needs=resolve_needs,
                     description=f"trip matrices: {len(matrices)} OMX files, "
                                 f"{sum(len(t) for _, t in matrices)} tables")


def _matrices(trips: pd.DataFrame, tables, labels, weight_col: str) -> dict[str, np.ndarray]:
    """1.x's write_matrices for one file: each table summed by OD, weighted, added into its matrix."""
    pos_index = labels.index if isinstance(labels, pd.Series) else labels
    n = len(pos_index)
    pos_index = pd.Index(pos_index, name=pos_index.name or "ZONE")
    out: dict[str, np.ndarray] = {}
    for name, field, ocol, dcol in tables:
        if field not in trips.columns or ocol not in trips.columns or dcol not in trips.columns:
            continue  # 1.x skips it (reported before the run)
        work = trips[[ocol, dcol, field]].copy()
        if weight_col and weight_col in trips.columns:
            work[weight_col] = trips[weight_col]
        work["_o"] = work[ocol]  # one zone system: trips' zones are the land use's zones
        work["_d"] = work[dcol]
        work = work.dropna(subset=["_o", "_d"])
        if weight_col and weight_col in work.columns:
            grouped_sum = work.groupby(["_o", "_d"], sort=False)[field].sum()
            mean_w = work.groupby(["_o", "_d"], sort=False)[weight_col].mean()
            vals = (grouped_sum / mean_w.replace(0, np.nan)).fillna(0.0)
        else:
            vals = work.groupby(["_o", "_d"], sort=False)[field].sum()
        if vals.empty:
            continue
        oi = pos_index.get_indexer(vals.index.get_level_values(0))
        di = pos_index.get_indexer(vals.index.get_level_values(1))
        mask = (oi != -1) & (di != -1)
        if not mask.any():
            continue
        data = out.get(name)
        if data is None:
            data = out[name] = np.zeros((n, n), dtype=float)
        data[oi[mask], di[mask]] += vals.to_numpy()[mask]
    return out
