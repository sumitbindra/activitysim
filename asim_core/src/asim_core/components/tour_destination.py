"""
Non-mandatory tour destination choice: a destination zone for every
non-mandatory tour, and the destination choice logsum.

Reads today's non_mandatory_tour_destination.yaml, its sample spec, spec,
coefficients and destination_choice_size_terms.csv unchanged, and follows
ActivitySim 1.x (activitysim/abm/models/non_mandatory_destination.py and
util/tour_destination.py), segment by segment (tour type):

1. **Size terms.** Each zone's size for the segment is a weighted sum of
   land use columns (e.g. retail employment). Zones with size zero can't be
   chosen.
2. **Sample.** The sample spec gives each tour × zone a utility; from the
   resulting probabilities, ``SAMPLE_SIZE`` zones are drawn per tour, with
   replacement. A zone drawn several times is kept once, with its
   ``pick_count``. (Swappable piece: ``sample``.)
3. **Logsums.** The tour mode choice model gives each sampled zone a
   logsum, at the segment's fixed departure and return periods
   (``OUT_PERIOD``, ``IN_PERIOD``). Its choosers are the sampled rows and
   the person's columns, as in 1.x.
4. **Choice.** The spec gives each sampled zone a utility (including the
   logsum and a correction for sampling, ``log(pick_count / prob)``); one
   zone is chosen per tour, and the logsum over the sampled zones is kept.

Tours are processed in batches of at most ``max_rows`` tour × zone rows;
a tour's sample, logsums and choice depend only on its own rows and random
stream, so the batch size doesn't change any result.

Not supported yet (the loader says so): preprocessors, two- and three-zone
systems (presampling), sampling methods other than 1.x's default, saving
the sample table, estimation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, MissingInputError, Needs, StepContext
from ..spec.legacy import (
    find_config_file,
    load_global_constants,
    load_interaction_model,
    load_top_settings,
    read_yaml,
)
from .runtime import plain
from .simple_choice import ChooserView
from .tour_mode_choice import TourModeChoiceModel

STEP = "non_mandatory_tour_destination"
CHANNEL = "tours"
CATEGORY = "non_mandatory"
PAD_UTILITY = -999.0
LOGSUM_ROWS = 100_000
SKIM_ACCESSOR = "skims"

TOURS_VIEW = ChooserView("tours", [
    ("persons", "person_id"),
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])
# 1.x's persons_merged: what the mode choice logsums see besides the sample
PERSONS_VIEW = ChooserView("persons", [
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])

_UNSUPPORTED = {
    "preprocessor": "preprocessors",
    "alts_preprocessor_sample": "preprocessors",
    "alts_preprocessor_simulate": "preprocessors",
    "annotate_tours": "annotation after the model",
    "annotate_persons": "annotation after the model",
}


def non_mandatory_tour_destination(
    configs_dirs: Sequence[str | Path],
    settings_file: str = "non_mandatory_tour_destination.yaml",
    max_rows: int = 1_000_000,
) -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    unsupported = [f"{k} ({why})" for k, why in _UNSUPPORTED.items() if settings.get(k)]
    method = (settings.get("compute_settings") or {}).get("sample_method") or top.get("sample_method")
    if method not in (None, "inverse_cdf"):
        unsupported.append(f"sample_method: {method}")
    for key in ("use_explicit_error_terms", "disable_destination_sampling"):
        if top.get(key):
            unsupported.append(key)
    if top.get("want_dest_choice_sample_tables") and settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME"):
        unsupported.append("saving the destination sample table")
    if unsupported:
        raise NotImplementedError(f"{settings_file}: not supported yet: {', '.join(unsupported)}")
    shift = not top.get("skip_failed_choices", True)  # as 1.x's utils_to_probs

    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    segments = list(settings["SEGMENTS"])
    segment_col = settings.get("CHOOSER_SEGMENT_COLUMN_NAME")
    if segment_col is None and len(segments) != 1:
        raise ValueError(f"{settings_file}: several SEGMENTS need CHOOSER_SEGMENT_COLUMN_NAME")
    orig_col = settings["CHOOSER_ORIG_COL_NAME"]
    alt_col = settings["ALT_DEST_COL_NAME"]
    chooser_id = settings.get("CHOOSER_ID_COLUMN", "person_id")
    sample_size = int(settings["SAMPLE_SIZE"])
    logsum_column = settings.get("DEST_CHOICE_LOGSUM_COLUMN_NAME")

    # size terms: segment -> {land use column: weight}
    st = pd.read_csv(find_config_file(configs_dirs, "destination_choice_size_terms.csv"),
                     comment="#", index_col="segment")
    st = st[st.model_selector == settings["SIZE_TERM_SELECTOR"]].drop(columns="model_selector")
    missing_segments = [s for s in segments if s not in st.index]
    if missing_segments:
        raise ValueError(f"no size terms for segments {missing_segments}")
    size_weights = {seg: st.loc[seg] for seg in segments}
    size_columns = sorted(st.columns)

    def load(spec_key, seg):
        return load_interaction_model(configs_dirs, settings[spec_key], settings["COEFFICIENTS"],
                                      constants, name=f"{STEP}_{spec_key.lower()}_{seg}",
                                      value_column=seg, skim_accessors=[SKIM_ACCESSOR])

    sample_defs = {seg: load("SAMPLE_SPEC", seg) for seg in segments}
    choice_defs = {seg: load("SPEC", seg) for seg in segments}
    sample_fns = {seg: d.utility_terms() for seg, d in sample_defs.items()}
    choice_fns = {seg: d.utility_terms() for seg, d in choice_defs.items()}

    model = TourModeChoiceModel(configs_dirs, str(settings["LOGSUM_SETTINGS"])) \
        if settings.get("LOGSUM_SETTINGS") else None

    def periods_for(seg):
        out_p, in_p = settings.get("OUT_PERIOD"), settings.get("IN_PERIOD")
        if isinstance(out_p, dict):
            out_p, in_p = out_p[seg], in_p[seg]
        return int(out_p), int(in_p)

    sample_cols = {alt_col, "prob", "pick_count", chooser_id}
    ls_added = {"in_period", "out_period", "duration"}
    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]) -> Needs:
        if "land_use" not in available:
            raise MissingInputError("needs the land_use table")
        missing = [c for c in size_columns if c not in available["land_use"]]
        if missing:
            raise MissingInputError(f"size terms use land_use columns {missing}, which don't exist")
        zone_key = "zone_id"  # 1.x's name for the sampled zone (land_use's index)
        needs: dict[str, set[str]] = {
            "tours": {"tour_category", chooser_id} | ({segment_col} if segment_col else set()),
            "land_use": set(size_columns),
        }
        tour_cols = available.get("tours", set())
        needs["tours"] |= {c for c in ("destination", logsum_column) if c and c in tour_cols}
        skims: set[str] = set()
        temps = {t.target for d in [*sample_defs.values(), *choice_defs.values()] for t in d.terms if t.target}
        columns = {orig_col}
        for d in sample_defs.values():
            columns |= d.columns_used() - {"size_term", zone_key}
            skims |= {k for _, k, _ in d.skim_uses()}
        for d in choice_defs.values():
            columns |= d.columns_used() - sample_cols - {"mode_choice_logsum", "size_term"}
            skims |= {k for _, k, _ in d.skim_uses()}
        columns -= temps
        tours_view = TOURS_VIEW.resolve(columns, available)
        bound = {}
        persons_columns: set[str] = set()
        if model is not None:
            df_cols = sample_cols | ls_added | PERSONS_VIEW.columns(available)
            for seg in segments:
                b = model.bind(df_cols, available, dest=alt_col, orig=orig_col)
                bound[seg] = b
                persons_columns |= b.columns - sample_cols - ls_added
                for t, c in b.tables:
                    needs.setdefault(t, set()).add(c)
                skims |= b.skims
        persons_view = PERSONS_VIEW.resolve(persons_columns, available) if persons_columns else {}
        for view in (tours_view, persons_view):
            for t, c in view.items():
                needs.setdefault(t, set()).update(c)
        if skims:
            needs[SKIMS] = skims
        resolved.update(needs={t: sorted(c) for t, c in needs.items()}, bound=bound,
                        tours_view=tours_view, persons_view=persons_view)
        return resolved["needs"]

    def run(ctx: StepContext):
        ids = ctx.ids("tours")
        cat = np.asarray(plain(ctx.column("tours", "tour_category")))
        rows_all = np.flatnonzero(cat == CATEGORY)
        rows_all = rows_all[np.argsort(ids[rows_all], kind="stable")]  # 1.x sorts by tour ID
        cols = {c: v[rows_all] for c, v in TOURS_VIEW.gather(ctx, resolved["tours_view"]).items()}
        for c in ("tour_category", chooser_id) + ((segment_col,) if segment_col else ()):
            cols[c] = ctx.column("tours", c)[rows_all]
        tour_ids = ids[rows_all]
        seg_values = np.asarray(plain(cols[segment_col])) if segment_col else None

        zones, sizes = _size_terms(ctx, size_weights, size_columns)
        debug = info["samples"] = {}
        chosen = np.full(len(rows_all), np.nan)
        logsums = np.full(len(rows_all), np.nan)
        orig_pos = ctx.skim_positions(np.asarray(cols[orig_col]))
        for seg in segments:
            sel = np.flatnonzero(seg_values == seg) if segment_col else np.arange(len(rows_all))
            if not len(sel):
                continue
            size = sizes[seg]
            alt_zones = zones[size > 0]
            alt_size = size[size > 0]
            if not len(alt_zones):
                raise ValueError(f"{STEP}: no zone has a size for segment '{seg}'")
            # 2. sample
            sample = _sample(ctx, sel, tour_ids, cols, orig_pos, alt_zones, alt_size,
                             sample_fns[seg], sample_size, shift, max_rows)
            # 3. logsums
            if model is not None:
                sample["mode_choice_logsum"] = _logsums(
                    ctx, model, resolved["bound"][seg], resolved["persons_view"], seg, periods_for(seg),
                    sample, tour_ids, cols, alt_col, chooser_id)
            else:
                sample["mode_choice_logsum"] = np.zeros(len(sample["tour"]), dtype=np.int64)
            # 4. choice
            zone_size = pd.Series(alt_size, index=alt_zones)
            seg_chosen, seg_logsums = _choose(ctx, sel, tour_ids, cols, orig_pos, sample, zone_size,
                                              choice_fns[seg], shift, max_rows, alt_col, chooser_id)
            chosen[sel] = seg_chosen
            logsums[sel] = seg_logsums
            debug[seg] = {"tour_id": tour_ids[sample["tour"]], **{k: v for k, v in sample.items() if k != "tour"}}

        out = {}
        for name, values in (("destination", chosen), (logsum_column, logsums)):
            if not name:
                continue
            current = (np.asarray(ctx.column("tours", name), dtype=np.float64).copy()
                       if name in resolved["needs"]["tours"] else np.full(len(ids), np.nan))
            done = ~np.isnan(values)
            current[rows_all[done]] = values[done]
            out[name] = current
        return {"tours": out}

    provides = ["destination"] + ([logsum_column] if logsum_column else [])
    info: dict = {}
    component = Component(
        name=STEP,
        run=run,
        provides={"tours": provides},
        resolve_needs=resolve_needs,
        description=f"non-mandatory tour destination: {sample_size} sampled zones per tour, "
                    f"{len(segments)} segments{', mode choice logsums' if model else ''}",
    )
    info.update(sample_specs=sample_defs, specs=choice_defs, logsum_model=model, trace_table="tours",
                resolved=resolved, generated_source="\n\n".join(src for _, _, src in choice_fns.values()))
    component.info = info
    return component


def _size_terms(ctx, size_weights, size_columns):
    """Each zone's size per segment: land use columns times weights, as 1.x's pandas .dot."""
    zones = ctx.ids("land_use")
    order = np.argsort(zones, kind="stable")  # 1.x sorts land use by zone first
    land_use = pd.DataFrame({c: ctx.column("land_use", c)[order] for c in size_columns},
                            index=zones[order])
    sizes = {seg: land_use[w.index].dot(w).to_numpy() for seg, w in size_weights.items()}
    return zones[order], sizes


class _ZoneSkims:
    """``s('skims', key)``: skims from each row's origin to its destination zone positions."""

    def __init__(self, ctx, orig_pos, dest_pos):
        self.ctx, self.orig, self.dest = ctx, orig_pos, dest_pos

    def __call__(self, accessor, key, how="get"):
        if how == "reverse":
            return self.ctx.skim(key, self.dest, self.orig)
        if how == "max":
            return np.maximum(self.ctx.skim(key, self.dest, self.orig), self.ctx.skim(key, self.orig, self.dest))
        return self.ctx.skim(key, self.orig, self.dest)


def _sample(ctx, sel, tour_ids, cols, orig_pos, alt_zones, alt_size, terms, sample_size, shift, max_rows):
    """Sampled zones per tour: rows (tour position, zone, prob as float32, pick_count)."""
    terms_fn, coefs, _ = terms
    n_alts = len(alt_zones)
    # 1.x caps the number of draws at the number of zones that can be chosen
    sample_size = min(sample_size, n_alts)
    dest_pos_alts = ctx.skim_positions(alt_zones)
    per_batch = max(1, max_rows // n_alts)
    parts = []
    for b in range(0, len(sel), per_batch):
        batch = sel[b:b + per_batch]
        n_b = len(batch)
        rows = np.repeat(batch, n_alts)
        c = {"zone_id": np.tile(alt_zones, n_b), "size_term": np.tile(alt_size, n_b)}
        for name, values in cols.items():
            if name not in c:
                c[name] = values[rows]
        s = _ZoneSkims(ctx, orig_pos[rows], np.tile(dest_pos_alts, n_b))
        n = n_b * n_alts
        with np.errstate(all="ignore"):
            utils = ctx.engine.interaction_utilities(terms_fn(c, n, None, s), coefs, n)
        probs, bad = ctx.engine.probabilities(utils.reshape(n_b, n_alts), shift=shift)
        if bad.any():
            raise ValueError(f"{STEP}: tours with no usable sampling probabilities: "
                             f"{tour_ids[batch][bad][:5].tolist()}")
        rands = ctx.random.uniform(CHANNEL, tour_ids[batch], sample_size)
        picks, picked_probs = ctx.engine.sample(probs, rands)
        parts.append(_dedupe(batch, picks, picked_probs, alt_zones))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _dedupe(batch, picks, picked_probs, alt_zones):
    """
    One row per (tour, zone) drawn, with how many times it was drawn, sorted
    by tour and then zone (1.x: interaction_sample sorts its result so).
    """
    n_b, n_draws = picks.shape
    key = (np.arange(n_b)[:, None] * len(alt_zones) + picks).ravel()  # row-major: tour, then draw
    # np.unique sorts by key: by tour, then by zone position (zones are in
    # increasing ID order), which is the order 1.x returns the sample in
    _, first, counts = np.unique(key, return_index=True, return_counts=True)
    tour_local = first // n_draws
    return {"tour": batch[tour_local], "zone": alt_zones[picks.ravel()[first]],
            "prob": picked_probs.ravel()[first].astype(np.float32),
            "pick_count": counts.astype(np.uint32)}


def _logsums(ctx, model, bound, persons_view, seg, periods, sample, tour_ids, cols, alt_col, chooser_id):
    """Mode choice logsums for the sampled rows, at the segment's fixed periods."""
    out_p, in_p = periods
    labels = model.periods.labels_for(np.array([out_p, in_p]))
    n = len(sample["tour"])
    person_ids = np.asarray(cols[chooser_id])[sample["tour"]]
    gathered = persons_view_gather(ctx, persons_view, person_ids)
    c_all = {alt_col: sample["zone"], "prob": sample["prob"], "pick_count": sample["pick_count"],
             chooser_id: person_ids,
             "out_period": np.full(n, labels[0], dtype=object),
             "in_period": np.full(n, labels[1], dtype=object),
             "duration": np.full(n, in_p - out_p, dtype=np.int64)}
    for name in bound.columns:
        if name not in c_all and name in gathered:
            c_all[name] = gathered[name]
    tours_of_rows = sample["tour"]
    result = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:  # pieces of about LOGSUM_ROWS rows, never splitting a tour
        end = min(start + LOGSUM_ROWS, n)
        while end < n and tours_of_rows[end] == tours_of_rows[end - 1]:
            end += 1
        part = {k: v[start:end] for k, v in c_all.items()}
        result[start:end] = model.evaluate(ctx, bound, part, end - start, seg,
                                           tour_ids[tours_of_rows[start:end]], logsums_only=True)
        start = end
    return result


def persons_view_gather(ctx, persons_view, person_ids):
    """The persons view's columns for the given person IDs (1.x: merged with persons_merged)."""
    if not persons_view:
        return {}
    pos = ctx.positions("persons", person_ids)
    view = PERSONS_VIEW
    full = view.gather(ctx, persons_view)
    return {k: np.asarray(plain(v))[pos] for k, v in full.items()}


def _choose(ctx, sel, tour_ids, cols, orig_pos, sample, zone_size, terms, shift, max_rows, alt_col, chooser_id):
    """Choose one sampled zone per tour; returns (zones, logsums) for the tours in ``sel``."""
    terms_fn, coefs, _ = terms
    tour_of_row = sample["tour"]
    # rows of each tour are contiguous and tours are in ``sel`` order
    bounds = np.searchsorted(tour_of_row, sel, side="left"), np.searchsorted(tour_of_row, sel, side="right")
    position_of = {t: i for i, t in enumerate(sel)}
    if not np.array_equal(np.unique(tour_of_row), np.sort(sel)):
        raise ValueError(f"{STEP}: sample rows don't cover every tour")
    chosen = np.empty(len(sel), dtype=np.float64)
    logsums = np.empty(len(sel), dtype=np.float64)
    sample_dest_pos = ctx.skim_positions(sample["zone"])
    size_of = zone_size.reindex(sample["zone"]).to_numpy()
    per_batch = max(1, max_rows // max(1, int((bounds[1] - bounds[0]).max())))
    for b in range(0, len(sel), per_batch):
        batch_tours = sel[b:b + per_batch]
        lo, hi = bounds[0][b], bounds[1][b + len(batch_tours) - 1]
        rows = np.arange(lo, hi)
        n = len(rows)
        tours_r = tour_of_row[rows]
        c = {alt_col: sample["zone"][rows], "prob": sample["prob"][rows],
             "pick_count": sample["pick_count"][rows],
             chooser_id: np.asarray(cols[chooser_id])[tours_r],
             "mode_choice_logsum": sample["mode_choice_logsum"][rows],
             "size_term": size_of[rows]}
        for name, values in cols.items():
            if name not in c:
                c[name] = values[tours_r]
        s = _ZoneSkims(ctx, orig_pos[tours_r], sample_dest_pos[rows])
        with np.errstate(all="ignore"):
            utils = ctx.engine.interaction_utilities(terms_fn(c, n, None, s), coefs, n)
        local = np.array([position_of[t] for t in tours_r]) - b
        counts = np.bincount(local, minlength=len(batch_tours))
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        padded = np.full((len(batch_tours), int(counts.max())), PAD_UTILITY)
        padded[local, np.arange(n) - offsets[local]] = utils
        probs, bad, ls = ctx.engine.probabilities(padded, shift=shift, with_logsums=True)
        if bad.any():
            raise ValueError(f"{STEP}: tours with no usable probabilities: "
                             f"{tour_ids[batch_tours][bad][:5].tolist()}")
        rands = ctx.random.uniform(CHANNEL, tour_ids[batch_tours])
        positions = ctx.engine.choose(probs, rands)
        chosen[b:b + len(batch_tours)] = sample["zone"][rows][offsets + positions]
        logsums[b:b + len(batch_tours)] = ls
    return chosen, logsums
