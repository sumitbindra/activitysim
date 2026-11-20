"""
Non-mandatory tour scheduling: a departure and return time for every
non-mandatory tour, fitted into each person's day.

Reads today's non_mandatory_tour_scheduling.yaml, spec(s), coefficients,
preprocessor and tour_departure_and_duration_alternatives.csv unchanged, and
follows ActivitySim 1.x (activitysim/abm/models/non_mandatory_scheduling.py
and util/vectorize_tour_scheduling.py):

* the preprocessor adds columns to the non-mandatory tours;
* tours are scheduled in rounds: every person's first non-mandatory tour,
  then their second, and so on. Each round sees the timetable as the
  earlier rounds (and earlier steps) left it;
* in a round, each tour gets one row per departure-return option that
  still fits its person's day; the spec gives each row a utility; one
  option is chosen per tour (multinomial logit); the choice is booked into
  the person's timetable and becomes the person's "previous tour"
  (``start_previous``, ``end_previous``) for the next round;
* with spec segments (MWCOG), each tour type has its own spec.

**Memory.** 1.x builds a round's whole tour-by-option table at once (with
its default settings): on ARC that's tens of millions of rows, and the step
needs gigabytes. Here a round is processed in batches of at most
``max_rows`` rows. A tour's options, utilities and choice depend only on
its own person's timetable and random stream, so the batch size doesn't
change any result.

Not supported yet (the loader says so): mode choice logsums
(``LOGSUM_SETTINGS``), alternative preprocessors, annotation after the
model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

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

STEP = "non_mandatory_tour_scheduling"  # 1.x step name (random streams depend on it)
CHANNEL = "tours"
CATEGORY = "non_mandatory"
PAD_UTILITY = -999.0  # 1.x pads tours with fewer options to a rectangle with this utility
# columns each row gets from its option, and from the person's previous tour
OPTION_COLUMNS = ("start", "end", "duration", "tdd", "mode_choice_logsum")
PREVIOUS_COLUMNS = ("start_previous", "end_previous")
OUTPUT_COLUMNS = ["start", "end", "duration", "tdd"]

VIEW = ChooserView("tours", [
    ("persons", "person_id"),
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])

_UNSUPPORTED = {
    "LOGSUM_SETTINGS": "mode choice logsums",
    "ALTS_PREPROCESSOR": "preprocessors for the alternatives",
    "annotate_tours": "annotation after the model",
    "annotate_persons": "annotation after the model",
    "annotate_households": "annotation after the model",
}


def non_mandatory_tour_scheduling(
    configs_dirs: Sequence[str | Path],
    settings_file: str = "non_mandatory_tour_scheduling.yaml",
    max_rows: int = 1_000_000,
    keep_utilities: bool = False,
) -> Component:
    """
    ``keep_utilities`` keeps every round's (tour ID, option, utility) rows in
    ``info["utilities"]``, for checking against 1.x below the level of choices.
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    unsupported = [f"{k} ({why})" for k, why in _UNSUPPORTED.items() if settings.get(k)]
    if settings.get("LOGIT_TYPE", "MNL") != "MNL":
        unsupported.append(f"LOGIT_TYPE: {settings['LOGIT_TYPE']}")
    if unsupported:
        raise NotImplementedError(f"{settings_file}: not supported yet: {', '.join(unsupported)}")
    top = load_top_settings(configs_dirs)
    shift = not top.get("skip_failed_choices", True)  # as 1.x's utils_to_probs
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    objects = {"tt": FORMULA_FUNCTIONS}

    # spec(s): one, or one per tour type (TOUR_SPEC_SEGMENTS -> SPEC_SEGMENTS)
    if settings.get("TOUR_SPEC_SEGMENTS"):
        spec_settings = settings["SPEC_SEGMENTS"]
        by_spec = {
            name: load_interaction_model(configs_dirs, s["SPEC"], s["COEFFICIENTS"], constants,
                                         objects, name=f"{STEP}_{name}")
            for name, s in spec_settings.items()
        }
        segments = {tour_type: by_spec[spec] for tour_type, spec in settings["TOUR_SPEC_SEGMENTS"].items()}
    else:
        segments = {None: load_interaction_model(configs_dirs, settings["SPEC"], settings["COEFFICIENTS"],
                                                 constants, objects, name=STEP)}
    compiled = {seg: d.utility_terms() for seg, d in segments.items()}

    alts = read_tdd_alternatives(find_config_file(configs_dirs, "tour_departure_and_duration_alternatives.csv"))
    n_alts = len(alts["start"])
    periods = [str(p) for p in range(int(alts["start"].min()) - 1, int(alts["end"].max()) + 2)]

    pre = settings.get("preprocessor")
    pre_spec = None
    pre_tables: list[str] = []
    if pre:
        if isinstance(pre, list):
            raise NotImplementedError("more than one preprocessor is not supported yet")
        pre_tables = list(pre.get("TABLES") or [])
        spec_file = str(pre["SPEC"])
        pre_spec = load_assignment_spec(
            configs_dirs, spec_file if spec_file.endswith(".csv") else spec_file + ".csv",
            constants, skim_accessors=["skim_dict"], name=f"{STEP}_preprocessor",
            tables=pre_tables, settings=top, df_names=[pre.get("DF", "df")],
        )

    temps = {t.target for d in segments.values() for t in d.terms if t.target}
    spec_columns = set().union(*(d.columns_used() for d in segments.values()))
    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]) -> Needs:
        if "person_windows" not in available:
            raise MissingInputError("needs the person_windows table (each person's timetable)")
        missing = [p for p in periods if p not in available["person_windows"]]
        if missing:
            raise MissingInputError(f"person_windows lacks periods {missing}")
        tour_cols = available.get("tours", set())
        bound = None
        pre_targets: set[str] = set()
        needs: dict[str, set[str]] = {"tours": {"tour_category", "tour_num", "tour_type", "person_id"}}
        if pre_spec is not None:
            # the preprocessor sees the non-mandatory tours only (1.x: DF)
            bound = pre_spec.bind(tour_cols, {t: available.get(t, set()) for t in pre_tables})
            pre_targets = set(bound.targets)
            needs["tours"] |= bound.columns_used()
            for t, c in bound.tables_used():
                needs.setdefault(t, set()).add(c)
            skim_names = {name for _, name, _ in bound.skim_uses()}
            if skim_names:
                needs[SKIMS] = skim_names
                needs["tours"] |= bound.columns_used()
        columns = spec_columns - set(OPTION_COLUMNS) - set(PREVIOUS_COLUMNS) - pre_targets - temps
        for t, c in VIEW.resolve(columns, available).items():
            needs.setdefault(t, set()).update(c)
        needs["tours"] |= {c for c in OUTPUT_COLUMNS if c in tour_cols}  # values of other tours
        needs["person_windows"] = set(periods)
        resolved["needs"] = {t: sorted(c) for t, c in needs.items()}
        resolved["bound"] = bound
        resolved["view_needs"] = {t: c for t, c in VIEW.resolve(columns | {"person_id"}, available).items()}
        return resolved["needs"]

    def run(ctx: StepContext):
        bound = resolved["bound"]
        ids = ctx.ids("tours")
        category = np.asarray(plain(ctx.column("tours", "tour_category")))
        chosen_rows = np.flatnonzero(category == CATEGORY)
        chosen_rows = chosen_rows[np.argsort(ids[chosen_rows], kind="stable")]  # 1.x sorts by tour ID
        tour_ids = ids[chosen_rows]
        n = len(chosen_rows)

        # chooser columns for the non-mandatory tours. Category columns stay
        # categorical (as in 1.x): comparing them with a string compares
        # integer codes, much faster than comparing strings row by row.
        cols = {c: v[chosen_rows] for c, v in VIEW.gather(ctx, resolved["view_needs"]).items()}
        for c in ("tour_num", "tour_type", "person_id"):
            cols[c] = ctx.column("tours", c)[chosen_rows]
        if bound is not None and n:
            pre_cols = {c: np.asarray(plain(ctx.column("tours", c)))[chosen_rows]
                        for c in bound.columns_used()}
            with np.errstate(all="ignore"):
                v = bound.function(pre_cols, ZoneSkims(ctx), n, TableLookup(ctx), None, {})
            for name in bound.targets:
                cols[name] = np.asarray(v[name])

        tt = TimeTable.from_columns(ctx.ids("person_windows"),
                                    {p: ctx.column("person_windows", p) for p in periods}, alts)
        # each person's previous non-mandatory tour in this step; 1.x starts with option 0
        previous = np.zeros(len(tt.ids), dtype=np.int64)
        person_rows = tt.rows(cols["person_id"])
        chosen = np.full(n, -1, dtype=np.int64)
        traced = set(ctx.tracer.targets("tours", tour_ids).tolist())
        kept = info["utilities"] = {}

        for tour_num in np.unique(cols["tour_num"]):
            in_round = cols["tour_num"] == tour_num
            for seg, (terms_fn, coefs, _) in compiled.items():
                sel = np.flatnonzero(in_round if seg is None else in_round & (cols["tour_type"] == seg))
                per_batch = max(1, max_rows // n_alts)
                for b in range(0, len(sel), per_batch):
                    batch = sel[b:b + per_batch]
                    chosen[batch] = _schedule_batch(
                        ctx, batch, tour_ids, cols, person_rows, previous, tt, alts,
                        terms_fn, coefs, shift, traced,
                        kept.setdefault(int(tour_num), []) if keep_utilities else None,
                    )
                    tt.assign(person_rows[batch], chosen[batch])
                    previous[person_rows[batch]] = chosen[batch]
            unscheduled = in_round & (chosen < 0)
            if unscheduled.any():
                types = sorted(set(np.asarray(cols["tour_type"])[unscheduled].tolist()))
                raise ValueError(f"{STEP}: no spec segment for tour types {types}")

        out = {}
        for c in OUTPUT_COLUMNS:
            current = ctx.column("tours", c) if c in resolved["needs"]["tours"] else None
            values = (np.asarray(current, dtype=np.float64).copy() if current is not None
                      else np.full(len(ids), np.nan))
            values[chosen_rows] = chosen if c == "tdd" else alts[c][chosen]
            out[c] = values
        return {"tours": out, "person_windows": tt.to_columns()}

    info: dict = {}
    component = Component(
        name=STEP,
        run=run,
        provides={"tours": OUTPUT_COLUMNS, "person_windows": periods},
        resolve_needs=resolve_needs,
        description=f"non-mandatory tour scheduling: {n_alts} departure-return options, "
                    f"{len(segments)} spec(s), batches of at most {max_rows:,} rows",
        info={
            "segments": segments,
            "preprocessor": pre_spec,
            "generated_source": "\n\n".join(src for _, _, src in compiled.values()),
            "trace_table": "tours",
            "resolved": resolved,
            "alts": alts,
        },
    )
    info.update(component.info)
    component.info = info
    return component


def _schedule_batch(ctx, batch, tour_ids, cols, person_rows, previous, tt, alts, terms_fn, coefs,
                    shift, traced, keep=None):
    """Choose an option for each tour in ``batch`` (positions among the tours being scheduled)."""
    rows_p = person_rows[batch]
    fits = tt.available(rows_p)                        # (tours, options)
    counts = fits.sum(axis=1)
    if (counts == 0).any():
        raise ValueError(f"{STEP}: tours with no option that fits the person's day: "
                         f"{tour_ids[batch][counts == 0][:5].tolist()}")
    tour_ix, alt_ix = np.nonzero(fits)                 # per tour, options in order (as 1.x)
    n = len(tour_ix)

    # one row per (tour, option): option columns, the person's previous tour, tour columns
    c = {"start": alts["start"][alt_ix], "end": alts["end"][alt_ix],
         "duration": alts["duration"][alt_ix], "tdd": alt_ix.astype(np.int64),
         "mode_choice_logsum": np.zeros(n, dtype=np.int64)}  # 1.x: 0 without LOGSUM_SETTINGS
    prev = previous[rows_p][tour_ix]
    c["start_previous"] = alts["start"][prev]
    c["end_previous"] = alts["end"][prev]
    rows = batch[tour_ix]
    for name, values in cols.items():
        if name not in c:
            c[name] = values[rows]

    with np.errstate(all="ignore"):
        utils = ctx.engine.interaction_utilities(terms_fn(c, n, tt), coefs, n)
    if keep is not None:
        keep.append((tour_ids[rows], alt_ix.copy(), utils.copy()))

    # one row per tour, options left-aligned, padded with -999 (as 1.x)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    width = int(counts.max())
    padded = np.full((len(batch), width), PAD_UTILITY)
    padded[tour_ix, np.arange(n) - offsets[tour_ix]] = utils
    probs, bad = ctx.engine.probabilities(padded, shift=shift)
    if bad.any():
        raise ValueError(f"{STEP}: tours with no usable probabilities: "
                         f"{tour_ids[batch][bad][:5].tolist()}")
    rands = ctx.random.uniform(CHANNEL, tour_ids[batch])
    positions = ctx.engine.choose(probs, rands)
    chosen = alt_ix[offsets + positions]
    if traced:
        _trace(ctx, batch, tour_ids, tour_ix, alt_ix, utils, probs, offsets, rands, chosen, traced)
    return chosen


def _trace(ctx, batch, tour_ids, tour_ix, alt_ix, utils, probs, offsets, rands, chosen, traced):
    for k, pos in enumerate(batch):
        tid = int(tour_ids[pos])
        if tid not in traced:
            continue
        mine = np.flatnonzero(tour_ix == k)
        rows = [[tid, int(alt_ix[r]), utils[r], probs[k, r - offsets[k]], rands[k, 0],
                 int(alt_ix[r] == chosen[k])] for r in mine]
        ctx.tracer.write(ctx.step, f"tour_{tid}", ["tour_id", "tdd", "utility", "probability",
                                                   "random_draw", "chosen"], rows)
