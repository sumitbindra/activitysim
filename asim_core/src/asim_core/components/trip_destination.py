"""
Trip destination: where each intermediate stop is (1.x:
``abm/models/trip_destination.py``, one-zone systems).

The last trip of each direction already goes to the tour's destination or
back to its origin; the stops in between are chosen, as 1.x does:

1. **Rounds by trip number**: every trip 1 that has a stop after it, then
   trips 2, ...; each round's trips start where the round before put them
   (a stop's zone becomes the next trip's origin). The preprocessor adds the
   tour's mode, the trip's skim time period (``trip_period``, from the
   tour's start or end) and the tour's leg destination (``tour_leg_dest``:
   its destination outbound, its origin inbound).
2. **Per primary purpose** (category order), with that purpose's
   coefficients:

   * **sample**: every zone gets a utility from the sample spec (distances
     origin -> stop -> leg destination, the stop's size for the trip's
     purpose from the ``trip`` size terms); ``SAMPLE_SIZE`` draws per trip;
     a trip whose zones all have zero probability fails;
   * **logsums**: for each sampled zone, two trip mode choice logsums
     (``LOGSUM_SETTINGS``): origin -> stop (``od_logsum``) and stop -> leg
     destination (``dp_logsum``), each with the trip mode choice
     preprocessor run on the same rows (1.x runs the second on the rows the
     first annotated);
   * **choice**: among the sampled zones (spec, with the sampling correction
     and the two logsums); the chosen zone's logsum is
     ``destination_logsum``. A trip with no usable zone fails.

3. A failed trip gets destination -1 and ``failed``; its next trip starts
   where it started. With ``CLEANUP``, failed trips and their direction's
   other trips are handled as 1.x's ``cleanup_failed_trips`` does.

The trips table is sorted by trip ID, as 1.x leaves it.

Random draws, per trip and in 1.x's order: the sample's draws, the logsum
preprocessors' (origin -> stop, then stop -> destination), the choice's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, MissingInputError, StepContext
from ..logit import FormulaValues
from ..skims import time_periods_from_settings
from ..spec.legacy import (
    find_config_file,
    load_global_constants,
    load_interaction_model,
    load_top_settings,
    read_settings,
    read_yaml,
)
from ..spec.translate import INDEX
from ..store import Table
from . import annotate
from .runtime import TableLookup, plain
from .tour_destination import LOGSUM_ROWS, PAD_UTILITY, _dedupe, _size_terms, _size_weights
from .tour_mode_choice import SkimLookup
from .trip_mode_choice import TOURS_VIEW, merged_tour_columns, tours_merged_columns, trip_mode_choice_model

STEP = "trip_destination"
CHANNEL = "trips"
NO_DESTINATION = -1
PERIOD = "trip_period"
# 1.x's SkimHotel.sample_skims: roles o (trip origin), d (the zone considered), n (primary origin),
# p (primary destination); (from, to, period column)
SAMPLE_SKIMS = {
    "od_skims": ("o", "d", None), "dp_skims": ("d", "p", None), "op_skims": ("o", "p", None),
    "nd_skims": ("n", "d", None),
    "odt_skims": ("o", "d", PERIOD), "dot_skims": ("d", "o", PERIOD), "dpt_skims": ("d", "p", PERIOD),
    "pdt_skims": ("p", "d", PERIOD), "opt_skims": ("o", "p", PERIOD), "pot_skims": ("p", "o", PERIOD),
    "ndt_skims": ("n", "d", PERIOD), "dnt_skims": ("d", "n", PERIOD),
}
TRIP_COLUMNS = ["tour_id", "trip_num", "trip_count", "outbound", "primary_purpose", "purpose", "origin",
                "destination"]
DEFAULTS = {"TRIP_ORIGIN": "origin", "ALT_DEST_COL_NAME": "dest_taz", "PRIMARY_ORIGIN": "origin",
            "PRIMARY_DEST": "tour_leg_dest"}
_UNSUPPORTED = {
    "alts_preprocessor_sample": "a preprocessor for the sampled zones",
    "alts_preprocessor_simulate": "a preprocessor for the sampled zones",
    "fail_some_trips_for_testing": "failing trips for testing",
}


class SizeTerms:
    """1.x's DataFrameMatrix of trip size terms: ``get(zones, purposes)``, ``get_cols(purposes)``."""

    def __init__(self, zones: np.ndarray, columns: list[str], values: np.ndarray):
        self.zones = np.asarray(zones)
        self.columns = list(columns)
        self.values = np.asarray(values, dtype=np.float64)  # (zones, columns)
        self._index = pd.Index(self.zones)
        self._cols = pd.Index(self.columns)

    def get(self, zones, purposes) -> np.ndarray:
        rows = self._index.get_indexer(np.asarray(zones))
        if (rows < 0).any():
            raise ValueError(f"size terms: zones not in the land use: {np.asarray(zones)[rows < 0][:5].tolist()}")
        return self.values[rows, self.column_positions(purposes)]

    def column_positions(self, purposes) -> np.ndarray:
        """Each purpose's column position (an error for purposes without size terms)."""
        cols = self._cols.get_indexer(np.asarray(purposes, dtype=object))
        if (cols < 0).any():
            bad = sorted(set(np.asarray(purposes, dtype=object)[cols < 0].tolist()), key=str)
            raise ValueError(f"size terms: no trip size terms for purposes {bad}")
        return cols


def trip_destination(configs_dirs: Sequence[str | Path], settings_file: str = "trip_destination.yaml",
                     max_rows: int = 1_000_000) -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = dict(read_settings(configs_dirs, settings_file))
    for old, new in (("DESTINATION_SAMPLE_SPEC", "SAMPLE_SPEC"), ("DESTINATION_SPEC", "SPEC")):
        if old in settings:  # 1.x's deprecated names: the same value if both are given
            if new in settings and settings[new] != settings[old]:
                raise ValueError(f"{settings_file}: {old} and {new} differ")
            settings.setdefault(new, settings.pop(old))
    for key, value in DEFAULTS.items():
        settings.setdefault(key, value)
    top = load_top_settings(configs_dirs)
    unsupported = [f"{k} ({why})" for k, why in _UNSUPPORTED.items() if settings.get(k)]
    for key in ("annotate_trips", "annotate_tours", "annotate_persons", "annotate_households"):
        if settings.get(key):
            unsupported.append(f"{key} (annotation after the model)")
    method = (settings.get("compute_settings") or {}).get("sample_method") or top.get("sample_method")
    if method not in (None, "inverse_cdf"):
        unsupported.append(f"sample_method: {method}")
    for key in ("use_explicit_error_terms", "disable_destination_sampling", "testing_fail_trip_destination"):
        if top.get(key):
            unsupported.append(key)
    if top.get("want_dest_choice_sample_tables") and settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME"):
        unsupported.append("saving the destination sample table")
    los = read_yaml(find_config_file(configs_dirs, "network_los.yaml"))
    if int(los.get("zone_system", 1)) != 1:
        unsupported.append("two- and three-zone systems")
    if unsupported:
        raise NotImplementedError(f"{settings_file}: not supported yet: {', '.join(unsupported)}")
    if "CLEANUP" not in settings:
        raise ValueError(f"{settings_file}: needs CLEANUP (1.x requires it)")
    cleanup = bool(settings["CLEANUP"])
    orig_col, alt_col = settings["TRIP_ORIGIN"], settings["ALT_DEST_COL_NAME"]
    primary_orig, primary_dest = settings["PRIMARY_ORIGIN"], settings["PRIMARY_DEST"]
    sample_size = int(settings["SAMPLE_SIZE"])
    logsum_column = settings.get("DEST_CHOICE_LOGSUM_COLUMN_NAME")

    # size terms: one column per 'trip' segment (purpose), in the file's order
    st_path = find_config_file(configs_dirs, "destination_choice_size_terms.csv")
    st = pd.read_csv(st_path, comment="#", index_col="segment")
    purposes = [s for s in st[st.model_selector == "trip"].index]
    size_weights, size_columns = _size_weights(st_path, "trip", purposes)

    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    periods = time_periods_from_settings(configs_dirs)
    pre_constants = dict(constants, network_los=periods, size_terms=SimpleNamespace(columns=purposes))
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, settings.get("preprocessor"), pre_constants,
                                                      top, STEP, "preprocessor")

    # specs, one coefficient column per primary purpose
    spec_purposes = [c for c in pd.read_csv(find_config_file(configs_dirs, settings["SPEC"]), comment="#").columns
                     if c not in ("Label", "Description", "Expression")]
    objects = {"size_terms": ["get"]}

    def load(spec_key, seg):
        d = load_interaction_model(configs_dirs, settings[spec_key], settings.get("COEFFICIENTS"), constants,
                                   objects=objects, name=f"{STEP}_{spec_key.lower()}_{seg}", value_column=seg,
                                   skim_accessors=list(SAMPLE_SKIMS))
        return d, d.utility_terms()

    sample_defs = {seg: load("SAMPLE_SPEC", seg) for seg in spec_purposes}
    # the stops' columns each sample spec reads: only these are copied out to every zone
    sample_reads = {seg: (set(d.columns_used()), any(SAMPLE_SKIMS[a][2] for a, _, _ in d.skim_uses()))
                    for seg, (d, _) in sample_defs.items()}
    choice_defs = {seg: load("SPEC", seg) for seg in spec_purposes}

    model = trip_mode_choice_model(configs_dirs, settings["LOGSUM_SETTINGS"]) if settings.get("LOGSUM_SETTINGS") \
        else None
    logsum_settings = read_yaml(find_config_file(configs_dirs, settings["LOGSUM_SETTINGS"])) if model else {}
    redundant = set(settings.get("REDUNDANT_TOURS_MERGED_CHOOSER_COLUMNS") or [])
    sample_cols = {alt_col, "prob", "pick_count"}
    resolved: dict = {}

    def check_spec_columns(chooser_cols: set[str]):
        """
        Before the run: every column the specs read is one they are given, as
        in 1.x: the trips' columns (with the preprocessor's and next_trip_id)
        and the zone column; the choice spec also each sampled zone's prob,
        pick_count, od_logsum and dp_logsum. Tour columns are not among them
        (1.x merges those only for the logsums).
        """
        problems, unsupported = [], []
        for key, defs, given in (("SAMPLE_SPEC", sample_defs, chooser_cols | {alt_col}),
                                 ("SPEC", choice_defs, chooser_cols | sample_cols | {"od_logsum", "dp_logsum"})):
            by_column: dict[str, list[str]] = {}
            for seg, (d, _) in defs.items():
                for c in d.columns_used() - given:
                    by_column.setdefault(c, []).append(seg)
            if INDEX in by_column:
                unsupported.append(f"{settings[key]} reads df.index")
                del by_column[INDEX]
            if key == "SAMPLE_SPEC":  # 1.x's sample also joins each zone's size per purpose (df.<purpose>)
                sizes = sorted(c for c in by_column if c in purposes)
                if sizes:
                    unsupported.append(f"{settings[key]} reads the zones' size columns {sizes} (use "
                                       f"size_terms.get(df.{alt_col}, df.purpose))")
                    for c in sizes:
                        del by_column[c]
            for c, segs in sorted(by_column.items()):
                where = "" if len(segs) == len(defs) else f" (purposes {', '.join(segs)})"
                problems.append(f"{settings[key]} reads '{c}'{where}")
        if unsupported:
            raise NotImplementedError(f"{STEP}: not supported yet: {'; '.join(unsupported)}")
        if problems:
            raise MissingInputError(f"{'; '.join(problems)}: not a column of the trips, nor made by the "
                                    f"preprocessor")

    def resolve_needs(available: Mapping[str, set[str]]):
        trip_cols = set(available.get("trips", set()))
        missing = [c for c in TRIP_COLUMNS + ["failed"] if c not in trip_cols]
        if missing:
            raise MissingInputError(f"trips lack {missing}")
        needs: dict[str, set[str]] = {"trips": set(trip_cols)}  # the table is rewritten whole (sorted)
        targets: set[str] = set()
        bound_pre = None
        if pre_spec is not None:
            bound_pre = pre_spec.bind(trip_cols | {"next_trip_id"}, {t: available.get(t, set()) for t in pre_tables})
            if bound_pre.n_random or bound_pre.skim_uses():
                raise NotImplementedError(f"{STEP}: preprocessor with random draws or skims")
            targets = set(bound_pre.targets)
            for t, c in bound_pre.tables_used():
                needs.setdefault(t, set()).add(c)
        chooser_cols = trip_cols | {"next_trip_id"} | targets
        for c in (PERIOD, primary_dest):
            if c not in chooser_cols:
                raise MissingInputError(f"trips need '{c}' (1.x: made by the preprocessor)")
        check_spec_columns(chooser_cols)
        skims: set[str] = set()
        for d, _ in [*sample_defs.values(), *choice_defs.values()]:
            for accessor, key, how in d.skim_uses():
                if how != "get":
                    raise NotImplementedError(f"{STEP}: {accessor}.{how}() is not supported")
                if SAMPLE_SKIMS[accessor][2] is None:
                    skims.add(key)
                else:
                    skims.update(f"{key}__{p}" for p in periods.unique_labels)
        bound = {}
        from_tours: list[str] = []
        if model is not None:
            tour_cols = [c for c in tours_merged_columns(logsum_settings, trip_cols) if c not in redundant]
            if primary_dest in tour_cols:
                raise ValueError(f"{STEP}: PRIMARY_DEST must not be a tours_merged column")
            offered = TOURS_VIEW.columns(available)
            absent = [c for c in tour_cols if c not in offered]
            if absent:
                raise MissingInputError(f"TOURS_MERGED_CHOOSER_COLUMNS {absent} are not columns of tours_merged")
            df_cols = sample_cols | chooser_cols | set(tour_cols)
            od = model.bind(df_cols, available, dest=alt_col, orig=orig_col,
                            names={"ORIGIN": orig_col, "DESTINATION": alt_col, "timeframe": "trip"})
            od_targets = set(od.preprocessor.targets) if od.preprocessor is not None else set()
            dp = model.bind(df_cols | od_targets, available, dest=primary_dest, orig=alt_col,
                            names={"ORIGIN": alt_col, "DESTINATION": primary_dest, "timeframe": "trip"})
            bound = {"od": od, "dp": dp}
            for b in (od, dp):
                for t, c in b.tables:
                    needs.setdefault(t, set()).add(c)
                skims |= b.skims
            read = (od.columns | dp.columns) - sample_cols - chooser_cols
            from_tours = sorted(c for c in read if c in tour_cols)
            unknown = sorted(read - set(from_tours) - od_targets)
            if unknown:
                raise MissingInputError(f"{STEP}: the logsum model reads {unknown}, which the trips lack")
        tours_view = TOURS_VIEW.resolve(set(from_tours), available) if from_tours else {}
        for t, c in tours_view.items():
            needs.setdefault(t, set()).update(c)
        needs.setdefault("land_use", set()).update(size_columns)
        if skims:
            needs[SKIMS] = skims
        resolved.update(bound_pre=bound_pre, bound=bound, tours_view=tours_view, from_tours=from_tours)
        return {t: sorted(c) for t, c in needs.items()}

    def resolve_provides(available):
        cols = list(available.get("trips", []))
        if logsum_column and logsum_column not in cols:
            cols.append(logsum_column)
        if cleanup and "failed" in cols:
            cols.remove("failed")
        return {"trips": cols}

    def run(ctx: StepContext):
        trips = ctx.table_frame("trips").sort_index()  # 1.x sorts the trips by ID
        n = len(trips)
        ids = trips.index.to_numpy()
        trip_num = trips.trip_num.to_numpy()
        trip_count = trips.trip_count.to_numpy()
        next_trip_id = np.where(trip_num < trip_count, np.roll(ids, -1), 0)
        destination = trips.destination.to_numpy().astype(np.int64).copy()
        origin = trips.origin.to_numpy().astype(np.int64).copy()
        failed = trips.failed.to_numpy().astype(bool).copy()
        logsums = np.full(n, np.nan)
        tour_cols = merged_tour_columns(ctx, resolved["tours_view"], resolved["from_tours"], trips.tour_id.to_numpy())

        zones, sizes = _size_terms(ctx, size_weights, size_columns)
        size_terms = SizeTerms(zones, purposes, np.column_stack([sizes[p] for p in purposes]))
        objs = SimpleNamespace(size_terms=size_terms)
        stats = info["stats"] = {"sample_rows": 0, "failed": 0}
        primary = trips.primary_purpose
        order = list(primary.cat.categories) if isinstance(primary.dtype, pd.CategoricalDtype) \
            else sorted(set(primary.astype(object)))
        primary = np.asarray(primary.astype(object), dtype=object)

        intermediate = trip_num < trip_count
        if intermediate.any():
            for tnum in range(int(trip_num[intermediate].min()), int(trip_num[intermediate].max()) + 1):
                rows = np.flatnonzero(intermediate & (trip_num == tnum))
                if not len(rows):
                    continue
                cols = {c: np.asarray(plain(trips[c].to_numpy()))[rows] for c in trips.columns}
                cols["origin"], cols["destination"], cols["failed"] = origin[rows], destination[rows], failed[rows]
                cols["next_trip_id"] = next_trip_id[rows]
                cols["primary_purpose"] = primary[rows]
                bound_pre = resolved["bound_pre"]
                if bound_pre is not None:
                    c = {k: cols[k] for k in bound_pre.columns_used() - {INDEX}}
                    c[INDEX] = ids[rows]
                    with np.errstate(all="ignore"):
                        v = bound_pre.function(c, None, len(rows), TableLookup(ctx), None, {})
                    for target in bound_pre.targets:
                        cols[target] = annotate._update(cols.get(target), np.asarray(v[target]))
                for name, values in tour_cols.items():
                    cols[name] = values[rows]
                chosen = np.full(len(rows), NO_DESTINATION, dtype=np.int64)
                seg_logsums = np.full(len(rows), np.nan)
                for purpose in order:  # 1.x: groupby on the category (observed values, category order)
                    sel = np.flatnonzero(cols["primary_purpose"] == purpose)
                    if not len(sel):
                        continue
                    if purpose not in sample_defs:
                        raise ValueError(f"{STEP}: no coefficients for primary purpose '{purpose}'")
                    z, ls = choose_destinations(ctx, purpose, sel, ids[rows], cols, zones, size_terms, objs,
                                                sample_defs[purpose][1], choice_defs[purpose][1], model,
                                                resolved["bound"], sample_size, max_rows, alt_col, orig_col,
                                                primary_orig, primary_dest, periods, stats,
                                                sample_reads=sample_reads[purpose])
                    chosen[sel] = z
                    seg_logsums[sel] = ls
                bad = chosen == NO_DESTINATION
                stats["failed"] += int(bad.sum())
                ok = ~bad
                destination[rows[ok]] = chosen[ok]
                if logsum_column:
                    logsums[rows[ok]] = seg_logsums[ok]
                nxt = pd.Index(ids).get_indexer(next_trip_id[rows])
                if (nxt < 0).any():
                    raise ValueError(f"{STEP}: intermediate trips without a next trip")
                origin[nxt[ok]] = chosen[ok]
                if bad.any():  # 1.x: failed, no destination, the next trip starts where this one did
                    failed[rows[bad]] = True
                    destination[rows[bad]] = NO_DESTINATION
                    origin[nxt[bad]] = origin[rows[bad]]

        out = trips.copy()
        out["destination"] = destination
        out["origin"] = origin
        out["failed"] = failed
        if logsum_column:
            out[logsum_column] = logsums
        if cleanup:
            if out.failed.any():
                out = cleanup_failed_trips(out)
            out = out.drop(columns="failed", errors="ignore")
        table = Table.from_pandas("trips", out)
        ctx.create_table(table)
        return {"trips": table}

    info: dict = {}
    component = Component(
        STEP, run, provides={"trips": ["destination", "origin"] + ([logsum_column] if logsum_column else [])},
        resolve_needs=resolve_needs, resolve_provides=resolve_provides,
        description=f"trip destinations: {sample_size} sampled zones per stop, {len(spec_purposes)} purposes"
                    f"{', two mode choice logsums per sampled zone' if model else ''}",
    )
    info.update(trace_table="trips", sample_specs={k: v[0] for k, v in sample_defs.items()},
                specs={k: v[0] for k, v in choice_defs.items()}, logsum_model=model, resolved=resolved)
    component.info = info
    return component


def choose_destinations(ctx, purpose, sel, trip_ids, cols, zones, size_terms, objs, sample_terms, choice_terms,
                        model, bound, sample_size, max_rows, alt_col, orig_col, primary_orig, primary_dest,
                        periods, stats, sample_reads=None):
    """1.x's choose_trip_destination for one round's trips of one primary purpose."""
    seg_ids = trip_ids[sel]
    c_seg = {k: v[sel] for k, v in cols.items()}
    pos = {"o": ctx.skim_positions(np.asarray(c_seg[orig_col], dtype=np.int64)),
           "n": ctx.skim_positions(np.asarray(c_seg[primary_orig], dtype=np.int64)),
           "p": ctx.skim_positions(np.asarray(c_seg[primary_dest], dtype=np.int64))}
    chosen = np.full(len(sel), NO_DESTINATION, dtype=np.int64)
    logsums = np.full(len(sel), np.nan)
    # 1. sample
    sample = sample_zones(ctx, seg_ids, c_seg, pos, zones, objs, sample_terms, sample_size, max_rows, alt_col,
                          periods, reads=sample_reads)
    stats["sample_rows"] += len(sample["tour"])
    if not len(sample["tour"]):
        return chosen, logsums
    # 2. logsums for each sampled zone, both ways
    rows_of = sample["tour"]  # positions among this segment's trips
    if model is not None:
        c_ls = {alt_col: sample["zone"], "prob": sample["prob"], "pick_count": sample["pick_count"]}
        for name, values in c_seg.items():
            if name not in c_ls:
                c_ls[name] = values[rows_of]
        od, dp = np.empty(len(rows_of)), np.empty(len(rows_of))
        start = 0
        while start < len(rows_of):  # pieces never splitting a trip: its draws happen once
            end = min(start + LOGSUM_ROWS, len(rows_of))
            while end < len(rows_of) and rows_of[end] == rows_of[end - 1]:
                end += 1
            part = {k: v[start:end] for k, v in c_ls.items()}
            draw_ids = seg_ids[rows_of[start:end]]
            # 1.x runs the second preprocessor on the rows the first annotated (the same frame)
            od[start:end] = model.evaluate(ctx, bound["od"], part, end - start, purpose, draw_ids,
                                           logsums_only=True, channel=CHANNEL)
            dp[start:end] = model.evaluate(ctx, bound["dp"], part, end - start, purpose, draw_ids,
                                           logsums_only=True, channel=CHANNEL)
            start = end
    else:
        od = dp = np.zeros(len(rows_of))
    sample["od_logsum"], sample["dp_logsum"] = od, dp
    # 3. choose among the sampled zones
    has = np.unique(rows_of)
    z, ls = choose_among_sample(ctx, has, seg_ids, c_seg, pos, sample, objs, choice_terms, max_rows, alt_col,
                                periods)
    chosen[has] = z
    logsums[has] = ls
    return chosen, logsums


def _skims(ctx, pos_rows, dest_pos, period_labels, periods):
    zones = {"o": pos_rows["o"], "n": pos_rows["n"], "p": pos_rows["p"], "d": dest_pos}
    labels = {PERIOD: period_labels} if period_labels is not None else {}
    return SkimLookup(ctx, periods, zones["o"], zones["d"], labels, wrappers=SAMPLE_SKIMS, zones=zones)


def sample_zones(ctx, seg_ids, c_seg, pos, zones, objs, terms, sample_size, max_rows, alt_col, periods,
                 reads=None):
    """
    1.x's interaction_sample for trips: every zone, zero-probability trips
    dropped, then the draws. ``reads``: (the stops' columns the sample spec
    reads, whether it uses time-of-day skims). Which columns are copied out to
    every zone, and how size terms are looked up, are seams (``sample_columns``,
    ``size_lookup``).
    """
    terms_fn, coefs, _ = terms
    n_alts = len(zones)
    sample_size = min(sample_size, n_alts)  # 1.x caps the draws at the number of zones
    dest_pos_alts = ctx.skim_positions(zones)
    per_batch = max(1, max_rows // n_alts)
    columns, uses_period = reads if reads is not None else (set(c_seg), True)
    copied = ctx.engine.sample_columns(list(c_seg), columns, alt_col, PERIOD, uses_period)
    parts = []
    for b in range(0, len(seg_ids), per_batch):
        batch = np.arange(b, min(b + per_batch, len(seg_ids)))
        n_b = len(batch)
        rows = np.repeat(batch, n_alts)
        c = {alt_col: np.tile(zones, n_b)}
        for name in copied:
            c[name] = c_seg[name][rows]
        s = _skims(ctx, {k: v[rows] for k, v in pos.items()}, np.tile(dest_pos_alts, n_b),
                   c.get(PERIOD), periods)
        batch_objs = objs
        if isinstance(getattr(objs, "size_terms", None), SizeTerms):  # the size_lookup seam
            batch_objs = SimpleNamespace(**vars(objs))
            batch_objs.size_terms = ctx.engine.size_lookup(
                objs.size_terms, c[alt_col], {name: (c[name], c_seg[name][batch]) for name in copied}, zones)
        n = n_b * n_alts
        with np.errstate(all="ignore"):
            utils = ctx.engine.interaction_utilities(FormulaValues(terms_fn, c, n, batch_objs, s), coefs, n)
        probs, bad = ctx.engine.probabilities(utils.reshape(n_b, n_alts), shift=False)  # allow_zero_probs
        good = np.flatnonzero(~bad)
        if not len(good):
            continue
        rands = ctx.random.uniform(CHANNEL, seg_ids[batch[good]], sample_size)
        picks, picked_probs = ctx.engine.sample(probs[good], rands)
        parts.append(_dedupe(batch[good], picks, picked_probs, zones))
    if not parts:
        return {"tour": np.empty(0, dtype=np.int64), "zone": np.empty(0, dtype=zones.dtype),
                "prob": np.empty(0, dtype=np.float32), "pick_count": np.empty(0, dtype=np.uint32)}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def choose_among_sample(ctx, sel, seg_ids, c_seg, pos, sample, objs, terms, max_rows, alt_col, periods):
    """1.x's interaction_sample_simulate for trips (zero probabilities allowed: no choice)."""
    terms_fn, coefs, _ = terms
    of = sample["tour"]
    lo, hi = np.searchsorted(of, sel, side="left"), np.searchsorted(of, sel, side="right")
    width = int((hi - lo).max())  # padded to the widest trip of the call, as 1.x
    chosen = np.full(len(sel), NO_DESTINATION, dtype=np.int64)
    logsums = np.full(len(sel), np.nan)
    dest_pos = ctx.skim_positions(sample["zone"])
    per_batch = max(1, max_rows // max(1, width))
    for b in range(0, len(sel), per_batch):
        bt = np.arange(b, min(b + per_batch, len(sel)))
        rows = np.arange(lo[bt[0]], hi[bt[-1]])
        n = len(rows)
        trips_r = of[rows]
        c = {alt_col: sample["zone"][rows], "prob": sample["prob"][rows], "pick_count": sample["pick_count"][rows],
             "od_logsum": sample["od_logsum"][rows], "dp_logsum": sample["dp_logsum"][rows]}
        for name, values in c_seg.items():
            if name not in c:
                c[name] = values[trips_r]
        s = _skims(ctx, {k: v[trips_r] for k, v in pos.items()}, dest_pos[rows], c.get(PERIOD), periods)
        with np.errstate(all="ignore"):
            utils = ctx.engine.interaction_utilities(FormulaValues(terms_fn, c, n, objs, s), coefs, n)
        local = np.searchsorted(sel[bt], trips_r)
        counts = np.bincount(local, minlength=len(bt))
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        padded = np.full((len(bt), width), PAD_UTILITY)
        padded[local, np.arange(n) - offsets[local]] = utils
        probs, bad, ls = ctx.engine.probabilities(padded, shift=False, with_logsums=True)
        rands = ctx.random.uniform(CHANNEL, seg_ids[sel[bt]])  # 1.x draws for every chooser
        positions = ctx.engine.choose(np.where(bad[:, None], 0.0, probs), rands)
        picked = sample["zone"][rows][offsets + np.minimum(positions, counts - 1)]
        chosen[bt] = np.where(bad, NO_DESTINATION, picked)
        logsums[bt] = np.where(bad, np.nan, ls)
    return chosen, logsums


def cleanup_failed_trips(trips: pd.DataFrame) -> pd.DataFrame:
    """1.x's cleanup_failed_trips: drop failed trips (and renumber their direction's other trips)."""
    from .tours import assign_in_place

    trips = trips.copy()
    trips["patch"] = False
    leg_mates = failed_trip_cohorts(trips, trips.failed) & ~trips.failed
    trips.loc[leg_mates, "patch"] = True
    trips = trips[~trips.failed]
    patch = trips[trips.patch].sort_index()
    grouped = patch.groupby(["tour_id", "outbound"])
    patch["trip_num"] = grouped.cumcount() + 1
    patch["trip_count"] = patch["trip_num"] + grouped.cumcount(ascending=False)
    assign_in_place(trips, patch[["trip_num", "trip_count"]])
    trips["origin"] = np.where((trips["trip_num"] == 1) & (trips["outbound"] == True),  # noqa: E712
                               trips["origin"], trips.groupby("tour_id")["destination"].shift()).astype(int)
    del trips["patch"]
    return trips


def failed_trip_cohorts(trips: pd.DataFrame, failed) -> pd.Series:
    bad_out = trips.outbound & trips.tour_id.isin(trips.tour_id[failed & trips.outbound])
    bad_in = ~trips.outbound & trips.tour_id.isin(trips.tour_id[failed & ~trips.outbound])
    return bad_out | bad_in
