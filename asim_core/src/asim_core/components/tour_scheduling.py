"""
Tour scheduling: a departure and return time for every tour of a category
(mandatory or non-mandatory), fitted into each person's day.

Reads today's settings (``mandatory_tour_scheduling.yaml``,
``non_mandatory_tour_scheduling.yaml``), specs, coefficients, preprocessors,
``tour_departure_and_duration_alternatives.csv`` and, if present,
``tour_departure_and_duration_segments.csv`` unchanged, and follows
ActivitySim 1.x (activitysim/abm/models/mandatory_scheduling.py,
non_mandatory_scheduling.py and util/vectorize_tour_scheduling.py):

* the chooser preprocessor adds columns to the tours being scheduled;
* tours are scheduled in rounds: every person's first tour of the
  category, then their second, and so on. Each round sees the timetable as
  earlier rounds and earlier steps left it. With spec segments (work,
  school, ...), each segment has its own spec;
* in a round, each tour gets one row per departure-return option that still
  fits its person's day. With ``LOGSUM_SETTINGS``, each row gets a tour mode
  choice logsum for its departure and return periods (computed by the tour
  mode choice model, ``TourModeChoiceModel``). The alternatives' preprocessor
  adds columns; the spec gives each row a utility; one option is chosen per
  tour (multinomial logit); the choice is booked into the person's
  timetable and becomes the person's "previous tour" for the next round.

**Memory.** 1.x builds a round's whole tour-by-option table at once (with
its default settings). Here a round is processed in batches of at most
``max_rows`` rows. A tour's options, logsums, utilities and choice depend
only on its own person's timetable and random stream, so the batch size
doesn't change any result.

**Logsums: computed once per distinct input.** Many options of a tour share
the same logsum: it depends on the option only through the columns the
mode choice model reads. Which rows get their own logsum is a swappable
piece (``logsum_keys``): the reference does what 1.x does (one per
departure period, return period and, without a segments file, duration);
``distinct_inputs`` uses only the option columns the mode choice formulas
actually read, which the formula translator knows exactly.

Not supported yet (the loader says so): park-and-ride in the logsums,
annotation after the model, a LOGSUM_PREPROCESSOR other than
``preprocessor``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, MissingInputError, Needs, StepContext
from ..spec.legacy import (
    find_config_file,
    load_assignment_spec,
    load_global_constants,
    load_interaction_model,
    load_top_settings,
    read_yaml,
)
from ..timetable import FORMULA_FUNCTIONS, TimeTable, read_tdd_alternatives
from .runtime import TableLookup, ZoneSkims, plain
from .simple_choice import ChooserView
from .tour_mode_choice import TourModeChoiceModel

CHANNEL = "tours"
PAD_UTILITY = -999.0  # 1.x pads tours with fewer options to a rectangle with this utility
LOGSUM_ROWS = 100_000  # logsum rows evaluated at once
PREVIOUS_COLUMNS = ("start_previous", "end_previous")
OUTPUT_COLUMNS = ["start", "end", "duration", "tdd"]
MANDATORY_SEGMENT_COLUMN = "mandatory_tour_seg"  # 1.x's name

VIEW = ChooserView("tours", [
    ("persons", "person_id"),
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])

_UNSUPPORTED = {
    "annotate_tours": "annotation after the model",
    "annotate_persons": "annotation after the model",
    "annotate_households": "annotation after the model",
}


# --- the components ----------------------------------------------------------------


def non_mandatory_tour_scheduling(configs_dirs: Sequence[str | Path],
                                  settings_file: str = "non_mandatory_tour_scheduling.yaml",
                                  max_rows: int = 1_000_000, keep_utilities: bool = False) -> Component:
    return tour_scheduling_component(configs_dirs, "non_mandatory_tour_scheduling", settings_file,
                                     "non_mandatory", max_rows, keep_utilities)


def mandatory_tour_scheduling(configs_dirs: Sequence[str | Path],
                              settings_file: str = "mandatory_tour_scheduling.yaml",
                              max_rows: int = 1_000_000, keep_utilities: bool = False) -> Component:
    return tour_scheduling_component(configs_dirs, "mandatory_tour_scheduling", settings_file,
                                     "mandatory", max_rows, keep_utilities)


def tour_scheduling_component(
    configs_dirs: Sequence[str | Path],
    step: str,
    settings_file: str,
    category: str,
    max_rows: int = 1_000_000,
    keep_utilities: bool = False,
) -> Component:
    """
    ``step`` is the 1.x step name (random streams depend on it); ``category``
    the tours it schedules. ``keep_utilities`` keeps each round's (tour ID,
    option, utility, logsum) rows in ``info["utilities"]``, for checks.
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    unsupported = [f"{k} ({why})" for k, why in _UNSUPPORTED.items() if settings.get(k)]
    if settings.get("LOGIT_TYPE", "MNL") != "MNL":
        unsupported.append(f"LOGIT_TYPE: {settings['LOGIT_TYPE']}")
    if settings.get("LOGSUM_PREPROCESSOR", "preprocessor") != "preprocessor":
        unsupported.append("LOGSUM_PREPROCESSOR other than 'preprocessor'")
    if unsupported:
        raise NotImplementedError(f"{settings_file}: not supported yet: {', '.join(unsupported)}")
    top = load_top_settings(configs_dirs)
    shift = not top.get("skip_failed_choices", True)  # as 1.x's utils_to_probs
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    objects = {"tt": FORMULA_FUNCTIONS}

    # spec(s): one, or one per segment (TOUR_SPEC_SEGMENTS -> SPEC_SEGMENTS)
    if settings.get("TOUR_SPEC_SEGMENTS"):
        by_spec = {
            name: load_interaction_model(configs_dirs, s["SPEC"], s["COEFFICIENTS"], constants,
                                         objects, name=f"{step}_{name}")
            for name, s in settings["SPEC_SEGMENTS"].items()
        }
        segments = {seg: by_spec[spec] for seg, spec in settings["TOUR_SPEC_SEGMENTS"].items()}
        segment_column = MANDATORY_SEGMENT_COLUMN if category == "mandatory" else "tour_type"
    else:
        if category == "mandatory":
            raise NotImplementedError(f"{settings_file}: mandatory scheduling needs TOUR_SPEC_SEGMENTS")
        segments = {None: load_interaction_model(configs_dirs, settings["SPEC"], settings["COEFFICIENTS"],
                                                 constants, objects, name=step)}
        segment_column = None
    compiled = {seg: d.utility_terms() for seg, d in segments.items()}

    alts = read_tdd_alternatives(find_config_file(configs_dirs, "tour_departure_and_duration_alternatives.csv"))
    n_alts = len(alts["start"])
    periods = [str(p) for p in range(int(alts["start"].min()) - 1, int(alts["end"].max()) + 2)]

    # chooser preprocessor
    pre = settings.get("preprocessor")
    pre_spec, pre_tables = _load_preprocessor(configs_dirs, pre, constants, top, step, "preprocessor")

    # alternatives' preprocessor: one, or one per segment
    alts_pre = settings.get("ALTS_PREPROCESSOR") or {}
    alts_specs = {}
    if alts_pre:
        if "SPEC" in alts_pre:  # one for every segment
            alts_specs = {seg: _load_preprocessor(configs_dirs, alts_pre, constants, top, step, "alts")[0]
                          for seg in segments}
        elif settings.get("LOGSUM_SETTINGS"):
            # 1.x looks per-segment preprocessors up by the logsum purpose, so
            # without logsums it skips them (asim1_findings, smaller items)
            alts_specs = {seg: _load_preprocessor(configs_dirs, s, constants, top, step, f"alts_{seg}")[0]
                          for seg, s in alts_pre.items()}

    # logsums from tour mode choice, for each option's departure and return periods
    model = None
    tdd_segments = None
    if settings.get("LOGSUM_SETTINGS"):
        model = TourModeChoiceModel(configs_dirs, str(settings["LOGSUM_SETTINGS"]))
        dest_setting = settings.get("DESTINATION_FOR_TOUR_PURPOSE")
        if dest_setting is None:
            raise ValueError(f"{settings_file}: LOGSUM_SETTINGS needs DESTINATION_FOR_TOUR_PURPOSE")
        try:
            tdd_segments = _read_tdd_segments(
                find_config_file(configs_dirs, "tour_departure_and_duration_segments.csv"))
        except FileNotFoundError:
            tdd_segments = None

    def destination(seg):
        d = settings.get("DESTINATION_FOR_TOUR_PURPOSE")
        return d if isinstance(d, str) else d.get(seg)

    temps = {t.target for d in segments.values() for t in d.terms if t.target}
    spec_columns = set().union(*(d.columns_used() for d in segments.values()))
    resolved: dict = {}
    info: dict = {}

    def option_columns(with_logsums: bool) -> set[str]:
        cols = {"start", "end", "duration", "tdd", "mode_choice_logsum"}
        return cols | ({"out_period", "in_period"} if with_logsums else set())

    def resolve_needs(available: Mapping[str, set[str]]) -> Needs:
        if "person_windows" not in available:
            raise MissingInputError("needs the person_windows table (each person's timetable)")
        missing = [p for p in periods if p not in available["person_windows"]]
        if missing:
            raise MissingInputError(f"person_windows lacks periods {missing}")
        tour_cols = set(available.get("tours", set()))
        view_cols = VIEW.columns(available)
        needs: dict[str, set[str]] = {"tours": {"tour_category", "tour_num", "tour_type", "person_id"}}
        base = {"tour_type"} | ({"is_university"} if category == "mandatory" else set())
        skims: set[str] = set()

        # chooser preprocessor: sees the tours being scheduled (1.x: DF)
        bound_pre = None
        pre_targets: set[str] = set()
        chooser_df_cols = tour_cols | ({segment_column} if segment_column else set())
        if pre_spec is not None:
            bound_pre = pre_spec.bind(chooser_df_cols, {t: available.get(t, set()) for t in pre_tables})
            pre_targets = set(bound_pre.targets)
            needs["tours"] |= bound_pre.columns_used() - {segment_column}
            for t, c in bound_pre.tables_used():
                needs.setdefault(t, set()).add(c)
            skims |= {name for _, name, _ in bound_pre.skim_uses()}
        chooser_cols = view_cols | pre_targets | ({segment_column} if segment_column else set())
        made_here = pre_targets | ({segment_column} if segment_column else set())

        # logsums, per segment (each has its own destination column)
        bound_ls = {}
        ls_columns: set[str] = set()
        if model is not None:
            ls_option_cols = {"out_period", "in_period", "duration"}
            if tdd_segments is not None:
                ls_option_cols |= {"start", "end"}
            renamed = {f"{c}_chooser" for c in chooser_cols & ls_option_cols}
            ls_df_cols = ls_option_cols | (chooser_cols - ls_option_cols) | renamed
            for seg in segments:
                b = model.bind(ls_df_cols, available, destination(seg))
                bound_ls[seg] = b
                ls_columns |= b.columns - ls_option_cols - made_here
                for t, c in b.tables:
                    needs.setdefault(t, set()).add(c)
                skims |= b.skims

        # alternatives' preprocessors: see the option rows only (1.x: alt_tdd)
        bound_alts = {}
        alts_targets: set[str] = set()
        for seg, spec in alts_specs.items():
            bound_alts[seg] = spec.bind(option_columns(model is not None), {})
            alts_targets |= set(bound_alts[seg].targets)

        columns = (spec_columns - option_columns(model is not None) - set(PREVIOUS_COLUMNS)
                   - alts_targets - temps - made_here) | ls_columns | base | {"person_id"}
        view_needs = VIEW.resolve(columns, available)
        for t, c in view_needs.items():
            needs.setdefault(t, set()).update(c)
        needs["tours"] |= {c for c in OUTPUT_COLUMNS if c in tour_cols}  # values of other tours
        needs["person_windows"] = set(periods)
        if skims:
            needs[SKIMS] = skims
        resolved.update(needs={t: sorted(c) for t, c in needs.items()}, bound_pre=bound_pre,
                        bound_ls=bound_ls, bound_alts=bound_alts, view_needs=view_needs)
        return resolved["needs"]

    def run(ctx: StepContext):
        ids = ctx.ids("tours")
        cat = np.asarray(plain(ctx.column("tours", "tour_category")))
        table_rows = np.flatnonzero(cat == category)          # 1.x's chooser order (table order)
        order = np.argsort(ids[table_rows], kind="stable")    # 1.x schedules in tour ID order
        n = len(table_rows)

        # chooser columns, in table order. Category columns stay categorical
        # (as in 1.x): comparing them with a string compares integer codes.
        cols = {c: v[table_rows] for c, v in VIEW.gather(ctx, resolved["view_needs"]).items()}
        for c in ("tour_num", "tour_type", "person_id", "tour_category"):
            cols[c] = ctx.column("tours", c)[table_rows]
        if segment_column == MANDATORY_SEGMENT_COLUMN:
            university = (np.asarray(plain(cols["tour_type"])) == "school") & \
                np.asarray(cols["is_university"]).astype(bool)
            cols[segment_column] = np.where(university, "univ", np.asarray(plain(cols["tour_type"]))).astype(object)
        bound_pre = resolved["bound_pre"]
        if bound_pre is not None and n:
            pre_cols = {c: (cols[c] if c in cols else ctx.column("tours", c)[table_rows])
                        for c in bound_pre.columns_used()}
            with np.errstate(all="ignore"):
                v = bound_pre.function(pre_cols, ZoneSkims(ctx), n, TableLookup(ctx), None, {})
            for name in bound_pre.targets:
                cols[name] = np.asarray(v[name])
        cols = {c: v[order] for c, v in cols.items()}         # now in tour ID order
        rows_in_table = table_rows[order]
        tour_ids = ids[rows_in_table]

        tt = TimeTable.from_columns(ctx.ids("person_windows"),
                                    {p: ctx.column("person_windows", p) for p in periods}, alts)
        # each person's previous tour of this category in this step; 1.x starts with option 0
        previous = np.zeros(len(tt.ids), dtype=np.int64)
        person_rows = tt.rows(cols["person_id"])
        chosen = np.full(n, -1, dtype=np.int64)
        traced = set(ctx.tracer.targets("tours", tour_ids).tolist())
        kept = info["utilities"] = {}
        stats = info["stats"] = {"option_rows": 0, "logsum_rows": 0}
        per_batch = max(1, max_rows // n_alts)
        seg_values = None if segment_column is None else np.asarray(plain(cols[segment_column]))

        for tour_num in np.unique(cols["tour_num"]):
            in_round = cols["tour_num"] == tour_num
            for seg, (terms_fn, coefs, _) in compiled.items():
                sel = np.flatnonzero(in_round if seg is None else in_round & (seg_values == seg))
                logsum_plan = None
                if model is not None:
                    logsum_plan = (model, resolved["bound_ls"][seg], seg,
                                   _segment_times(tdd_segments, seg) if tdd_segments is not None else None,
                                   stats)
                for b in range(0, len(sel), per_batch):
                    batch = sel[b:b + per_batch]
                    chosen[batch] = _schedule_batch(
                        ctx, batch, tour_ids, cols, person_rows, previous, tt, alts,
                        terms_fn, coefs, shift, traced, logsum_plan, resolved["bound_alts"].get(seg),
                        kept.setdefault(int(tour_num), []) if keep_utilities else None,
                    )
                    tt.assign(person_rows[batch], chosen[batch])
                    previous[person_rows[batch]] = chosen[batch]
            unscheduled = in_round & (chosen < 0)
            if unscheduled.any():
                values = np.asarray(plain(cols[segment_column or "tour_type"]))[unscheduled]
                raise ValueError(f"{step}: no spec segment for {sorted(set(values.tolist()))}")

        out = {}
        for c in OUTPUT_COLUMNS:
            current = ctx.column("tours", c) if c in resolved["needs"]["tours"] else None
            values = (np.asarray(current, dtype=np.float64).copy() if current is not None
                      else np.full(len(ids), np.nan))
            values[rows_in_table] = chosen if c == "tdd" else alts[c][chosen]
            out[c] = values
        return {"tours": out, "person_windows": tt.to_columns()}

    component = Component(
        name=step,
        run=run,
        provides={"tours": OUTPUT_COLUMNS, "person_windows": periods},
        resolve_needs=resolve_needs,
        description=f"{category.replace('_', '-')} tour scheduling: {n_alts} departure-return options, "
                    f"{len(segments)} spec(s){', mode choice logsums' if model else ''}, "
                    f"batches of at most {max_rows:,} rows",
    )
    info.update(
        segments=segments, preprocessor=pre_spec, alts_preprocessors=alts_specs, logsum_model=model,
        generated_source="\n\n".join(src for _, _, src in compiled.values()),
        trace_table="tours", resolved=resolved, alts=alts,
    )
    component.info = info
    return component


def _load_preprocessor(configs_dirs, settings, constants, top, step, label):
    if not settings:
        return None, []
    if isinstance(settings, list):
        raise NotImplementedError("more than one preprocessor is not supported yet")
    tables = list(settings.get("TABLES") or [])
    spec_file = str(settings["SPEC"])
    spec = load_assignment_spec(
        configs_dirs, spec_file if spec_file.endswith(".csv") else spec_file + ".csv",
        constants, skim_accessors=["skim_dict"], name=f"{step}_{label}",
        tables=tables, settings=top, df_names=[settings.get("DF", "df")],
    )
    return spec, tables


def _read_tdd_segments(path) -> pd.DataFrame:
    """tour_departure_and_duration_segments.csv, as 1.x reads it (start and end as 8-bit)."""
    df = pd.read_csv(path, comment="#")
    df["start"] = df["start"].astype(np.int8)
    df["end"] = df["end"].astype(np.int8)
    return df


def _segment_times(tdd_segments: pd.DataFrame, purpose) -> dict[str, tuple]:
    """Representative (start, end) per time period label for a purpose, as 1.x's dedupe_alt_tdd picks them."""
    df = tdd_segments
    if "tour_purpose" in df:
        mine = df.tour_purpose == purpose
        if not mine.any():
            mine = df.tour_purpose.isnull()
        if not mine.any():
            raise ValueError(f"no segments for tour purpose {purpose} in tour_departure_and_duration_segments")
        df = df[mine]
    return {"start": dict(zip(df.time_period.astype(str), df.start)),
            "end": dict(zip(df.time_period.astype(str), df.end))}


def _schedule_batch(ctx, batch, tour_ids, cols, person_rows, previous, tt, alts, terms_fn, coefs,
                    shift, traced, logsum_plan, bound_alts, keep):
    """Choose an option for each tour in ``batch`` (positions among the tours being scheduled)."""
    rows_p = person_rows[batch]
    fits = tt.available(rows_p)                        # (tours, options)
    counts = fits.sum(axis=1)
    if (counts == 0).any():
        raise ValueError(f"{ctx.step}: tours with no option that fits the person's day: "
                         f"{tour_ids[batch][counts == 0][:5].tolist()}")
    tour_ix, alt_ix = np.nonzero(fits)                 # per tour, options in order (as 1.x)
    n = len(tour_ix)
    rows = batch[tour_ix]

    # one row per (tour, option): option columns, then logsums
    c = {"start": alts["start"][alt_ix], "end": alts["end"][alt_ix],
         "duration": alts["duration"][alt_ix], "tdd": alt_ix.astype(np.int64)}
    if logsum_plan is None:
        c["mode_choice_logsum"] = np.zeros(n, dtype=np.int64)  # 1.x: 0 without LOGSUM_SETTINGS
    else:
        c.update(_option_logsums(ctx, batch, tour_ids, cols, tour_ix, alt_ix, alts, logsum_plan))
    if bound_alts is not None:
        with np.errstate(all="ignore"):
            v = bound_alts.function(dict(c), None, n, None, None, {})
        for name in bound_alts.targets:
            c[name] = np.asarray(v[name])
    prev = previous[rows_p][tour_ix]
    c["start_previous"] = alts["start"][prev]
    c["end_previous"] = alts["end"][prev]
    for name, values in cols.items():
        if name not in c:
            c[name] = values[rows]

    with np.errstate(all="ignore"):
        utils = ctx.engine.interaction_utilities(terms_fn(c, n, tt), coefs, n)
    if keep is not None:
        keep.append((tour_ids[rows], alt_ix.copy(), utils.copy(),
                     np.asarray(c["mode_choice_logsum"], dtype=np.float64).copy()))

    # one row per tour, options left-aligned, padded with -999 (as 1.x)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    width = int(counts.max())
    padded = np.full((len(batch), width), PAD_UTILITY)
    padded[tour_ix, np.arange(n) - offsets[tour_ix]] = utils
    probs, bad = ctx.engine.probabilities(padded, shift=shift)
    if bad.any():
        raise ValueError(f"{ctx.step}: tours with no usable probabilities: "
                         f"{tour_ids[batch][bad][:5].tolist()}")
    rands = ctx.random.uniform(CHANNEL, tour_ids[batch])
    positions = ctx.engine.choose(probs, rands)
    chosen = alt_ix[offsets + positions]
    if traced:
        _trace(ctx, batch, tour_ids, tour_ix, alt_ix, utils, c["mode_choice_logsum"], probs,
               offsets, rands, chosen, traced)
    return chosen


def _option_logsums(ctx, batch, tour_ids, cols, tour_ix, alt_ix, alts, logsum_plan):
    """
    Mode choice logsums for each (tour, option) row, computed once per tour
    and distinct value of the keys the ``logsum_keys`` piece picks, then
    copied to every row that shares them. Returns the option columns 1.x adds
    (out_period, in_period, duration) and mode_choice_logsum.
    """
    model, bound, segment, seg_times, stats = logsum_plan
    periods = model.periods
    labels = np.asarray(periods.unique_labels, dtype=object)
    out_ix = periods.label_index(alts["start"])[alt_ix]
    in_ix = periods.label_index(alts["end"])[alt_ix]
    duration = (alts["end"] - alts["start"]).astype(np.int8)[alt_ix]  # 1.x recomputes it

    read = bound.columns | bound.period_columns
    keys = ctx.engine.logsum_keys(read, seg_times is not None)
    code = tour_ix.astype(np.int64)
    for k in keys:
        values = {"out_period": out_ix, "in_period": in_ix, "duration": duration}[k].astype(np.int64)
        code = code * (int(values.max()) + 1) + values
    _, first, inverse = np.unique(code, return_index=True, return_inverse=True)  # sorted by tour
    stats["option_rows"] += len(code)
    stats["logsum_rows"] += len(first)

    u_tour = tour_ix[first]
    u_rows = batch[u_tour]
    c = {"out_period": labels[out_ix[first]], "in_period": labels[in_ix[first]]}
    if seg_times is not None:  # representative times for the periods (1.x: dedupe_alt_tdd)
        try:
            c["start"] = np.array([seg_times["start"][p] for p in c["out_period"]], dtype=np.int8)
            c["end"] = np.array([seg_times["end"][p] for p in c["in_period"]], dtype=np.int8)
        except KeyError as err:
            raise ValueError(f"time period {err} missing from tour_departure_and_duration_segments") from None
        c["duration"] = (c["end"] - c["start"]).astype(np.int8)
    else:
        c["duration"] = duration[first]
    for name in bound.columns:
        if name not in c and name in cols:
            c[name] = np.asarray(plain(cols[name]))[u_rows]
    # evaluate in pieces of about LOGSUM_ROWS rows (the model holds one value
    # per term and row), never splitting a tour: its random draws happen once
    logsums = np.empty(len(first), dtype=np.float64)
    start = 0
    while start < len(first):
        end = min(start + LOGSUM_ROWS, len(first))
        while end < len(first) and u_tour[end] == u_tour[end - 1]:
            end += 1
        part = {k: v[start:end] for k, v in c.items()}
        logsums[start:end] = model.evaluate(ctx, bound, part, end - start, segment,
                                            tour_ids[u_rows[start:end]], logsums_only=True)
        start = end
    return {"out_period": labels[out_ix], "in_period": labels[in_ix], "duration": duration,
            "mode_choice_logsum": np.asarray(logsums, dtype=np.float64)[inverse]}


def _trace(ctx, batch, tour_ids, tour_ix, alt_ix, utils, logsums, probs, offsets, rands, chosen, traced):
    for k, pos in enumerate(batch):
        tid = int(tour_ids[pos])
        if tid not in traced:
            continue
        mine = np.flatnonzero(tour_ix == k)
        rows = [[tid, int(alt_ix[r]), utils[r], logsums[r], probs[k, r - offsets[k]], rands[k, 0],
                 int(alt_ix[r] == chosen[k])] for r in mine]
        ctx.tracer.write(ctx.step, f"tour_{tid}", ["tour_id", "tdd", "utility", "mode_choice_logsum",
                                                   "probability", "random_draw", "chosen"], rows)
