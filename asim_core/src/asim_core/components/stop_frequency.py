"""
Stop frequency: how many stops each tour makes on its way out and on its
way back, and the trips that follow (1.x: ``abm/models/stop_frequency.py``
and ``util/trip.py``'s ``initialize_from_tours``).

As 1.x:

1. **Choosers**: every tour with its person (1.x's ``tours_merged``: the
   tours joined with ``persons_merged``, the tour's own column winning),
   annotated by the preprocessor. MTC's works out ``primary_purpose`` (the
   tour type; ``univ`` for a university student's school tour; ``atwork``
   for at-work subtours), the tour's distance (``od_skims``, origin to
   destination), household counts by person type, and the tour's and
   household's counts by purpose.
2. **One model per segment** (``SPEC_SEGMENTS``, by ``SEGMENT_COL``), each
   with its own spec and coefficients; one draw per tour.
3. ``tours.stop_frequency`` (``'1out_2in'``, ...) and ``tours.primary_purpose``.
4. **The trips table**: for each tour, ``out + 1`` outbound trips and
   ``in + 1`` inbound ones (``stop_frequency_alternatives.csv``), numbered
   within each direction; outbound trips go from the tour's origin to its
   destination, inbound ones back. 1.x's trip ID: ``tour_id × 2 × (most
   trips per leg) + trip number`` (inbound trips numbered after the
   outbound ones).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, MissingInputError, StepContext
from ..logit import FormulaValues
from ..skims import time_periods_from_settings
from ..spec.legacy import find_config_file, load_choice_model, load_global_constants, load_top_settings, read_settings
from ..spec.translate import INDEX
from ..store import Table
from . import annotate
from .runtime import TableLookup, WrappedSkims, plain
from .simple_choice import ChooserView

STEP = "stop_frequency"
CHANNEL = "tours"
VIEW = ChooserView("tours", [  # 1.x's tours_merged: tours joined with persons_merged
    ("persons", "person_id"),
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])
TOUR_COLUMNS = ["person_id", "household_id", "origin", "destination"]
TRIP_COLUMNS = ["person_id", "household_id", "primary_purpose", "trip_num", "outbound", "trip_count",
                "destination", "origin", "failed", "tour_id"]
DEFAULT_MAX_TRIPS_PER_LEG = 4  # 1.x's default


def stop_frequency(configs_dirs: Sequence[str | Path], settings_file: str = "stop_frequency.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_settings(configs_dirs, settings_file)
    top = load_top_settings(configs_dirs)
    if settings.get("LOGIT_TYPE", "MNL") != "MNL":
        raise NotImplementedError(f"{settings_file}: only multinomial stop frequency models are supported")
    segment_col = settings.get("SEGMENT_COL", "primary_purpose")
    if not settings.get("SPEC_SEGMENTS"):
        raise NotImplementedError(f"{settings_file}: needs SPEC_SEGMENTS (as 1.x does)")
    handled = ["preprocessor", "annotate_tours", "annotate_persons", "annotate_households", "annotate_trips"]
    segments = []
    for seg in settings["SPEC_SEGMENTS"]:
        if segment_col not in seg:
            raise ValueError(f"{settings_file}: a SPEC_SEGMENTS entry lacks {segment_col}")
        d = load_choice_model(configs_dirs, settings_file, name=f"{STEP}_{seg[segment_col]}", handled=handled,
                              overrides={"SPEC": seg["SPEC"], "COEFFICIENTS": seg["COEFFICIENTS"]})
        coef = d.coefficient_matrix()
        keep = ~(coef == 0).all(axis=1)  # 1.x skips terms whose coefficients are all zero
        terms_fn, source = d.utility_function(keep)
        segments.append((seg[segment_col], d, coef[keep], terms_fn, source, d.columns_used(keep)))
    alternatives = segments[0][1].alternatives
    for value, d, *_ in segments:
        if d.alternatives != alternatives:
            raise ValueError(f"{settings_file}: segment {value} has other alternatives than {segments[0][0]}")
    shift = not top.get("skip_failed_choices", True)

    alts = pd.read_csv(find_config_file(configs_dirs, "stop_frequency_alternatives.csv"), comment="#")
    alts = alts.set_index("alt")
    if "out" not in alts.columns or "in" not in alts.columns:
        raise ValueError("stop_frequency_alternatives.csv needs 'out' and 'in' columns")
    missing = [a for a in alternatives if a not in alts.index]
    if missing:
        raise ValueError(f"stop_frequency_alternatives.csv lacks the spec's alternatives {missing}")
    max_trips = max_trips_per_leg(settings, alts)

    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    constants["network_los"] = time_periods_from_settings(configs_dirs)  # skim_time_period_label
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, settings.get("preprocessor"), constants, top,
                                                      STEP, "preprocessor", skim_accessors=["od_skims"])
    annotations = annotate.legacy_annotations(configs_dirs, settings, top, STEP)
    resolved: dict = {}

    def resolve_needs(available):
        tour_cols = set(available.get("tours", set()))
        missing = [c for c in TOUR_COLUMNS + ["tour_type"] if c not in tour_cols]
        if missing:
            raise MissingInputError(f"tours lack {missing}")
        df_cols = VIEW.columns(available)
        wanted = set(TOUR_COLUMNS)
        extra: dict[str, set[str]] = {}
        skims: set[str] = set()
        targets: set[str] = set()
        bound = None
        if pre_spec is not None:
            bound = pre_spec.bind(df_cols, {t: available.get(t, set()) for t in pre_tables})
            if bound.n_random:
                raise NotImplementedError(f"{STEP}: preprocessor with random draws")
            targets = set(bound.targets)
            wanted |= bound.columns_used() - {INDEX}
            for t, c in bound.tables_used():
                extra.setdefault(t, set()).add(c)
            for accessor, name, _ in bound.skim_uses():
                if accessor not in ("od_skims", "skim_dict"):
                    raise NotImplementedError(f"{STEP}: preprocessor skims '{accessor}'")
                skims.add(name)
        if segment_col not in targets:
            wanted.add(segment_col)
        for *_, columns in segments:
            wanted |= set(columns) - targets
        view_needs = VIEW.resolve(wanted, available)
        needs = {t: set(c) for t, c in view_needs.items()}
        for t, c in extra.items():
            needs[t] = needs.get(t, set()) | c
        needs["tours"] |= set(available["tours"])  # tours are copied whole into the new tours table
        made = {"tours": {"stop_frequency", "primary_purpose"} | tour_cols, "trips": set(TRIP_COLUMNS)}
        bound_ann, ann_needs, _ = annotate.bind(annotations, available, made)
        for t, c in ann_needs.items():
            if t != "trips":
                needs[t] = needs.get(t, set()) | set(c)
        if skims:
            needs[SKIMS] = skims
        resolved.update(view_needs=view_needs, bound=bound, annotations=bound_ann)
        return {t: sorted(c) for t, c in needs.items()}

    def resolve_provides(available):
        tours = list(available.get("tours", []))
        provides = {"tours": tours + [c for c in ("stop_frequency", "primary_purpose") if c not in tours],
                    "trips": list(TRIP_COLUMNS)}
        for a in annotations:
            for target in a.spec.targets:
                if target not in provides.setdefault(a.table, []):
                    provides[a.table].append(target)
        return provides

    def run(ctx: StepContext):
        ids = ctx.ids("tours")
        n = len(ids)
        cols = dict(VIEW.gather(ctx, resolved["view_needs"]))
        bound = resolved["bound"]
        if bound is not None and n:
            c = {k: cols[k] for k in bound.columns_used() - {INDEX}}
            c[INDEX] = ids
            skims = WrappedSkims(ctx, {"od_skims": (np.asarray(plain(cols["origin"])),
                                                    np.asarray(plain(cols["destination"])))})
            with np.errstate(all="ignore"):  # 1.x logs numpy warnings and carries on
                v = bound.function(c, skims, n, TableLookup(ctx), None, {})
            for target in bound.targets:
                cols[target] = annotate._update(cols.get(target), np.asarray(v[target]))
        purpose = np.asarray(plain(cols[segment_col]), dtype=object)

        chosen = np.full(n, None, dtype=object)
        for value, d, coef, terms_fn, _, columns in segments:
            rows = np.flatnonzero(purpose == value)
            if not len(rows):
                continue  # 1.x skips empty segments
            c = {k: (np.asarray(plain(cols[k]))[rows] if np.ndim(cols[k]) else cols[k]) for k in columns}
            with np.errstate(divide="ignore", invalid="ignore"):
                utils = ctx.engine.utilities(FormulaValues(terms_fn, c), coef, len(rows))
            probs, bad = ctx.engine.probabilities(utils, shift=shift)
            if bad.any():
                raise ValueError(f"{STEP}: {int(bad.sum())} '{value}' tours have no usable probabilities, "
                                 f"e.g. {ids[rows][bad][:5].tolist()}")
            rands = ctx.random.uniform(CHANNEL, ids[rows])
            chosen[rows] = np.asarray(alternatives, dtype=object)[ctx.engine.choose(probs, rands)]

        tours = ctx.table_frame("tours")
        tours["stop_frequency"] = pd.Categorical(chosen, categories=list(alternatives))
        tour_type = tours["tour_type"].dtype
        # 1.x: the tour type's categories (where() keeps them), so 'univ' and 'atwork' must be among them
        if isinstance(tour_type, pd.CategoricalDtype):
            unknown = sorted(set(purpose[pd.notna(purpose)]) - set(tour_type.categories))
            if unknown:
                raise ValueError(f"{STEP}: primary purposes {unknown} are not tour type categories")
            tours["primary_purpose"] = pd.Categorical(purpose, categories=tour_type.categories)
        else:
            tours["primary_purpose"] = purpose
        if tours["stop_frequency"].isna().any():
            unset = sorted(set(purpose[tours["stop_frequency"].isna().to_numpy()].tolist()), key=str)
            raise ValueError(f"{STEP}: tours whose {segment_col} has no spec segment: {unset}")
        trips = initialize_from_tours(tours, alts, max_trips)
        tours_table = Table.from_pandas("tours", tours)
        trips_table = Table.from_pandas("trips", trips)
        ctx.create_table(tours_table)
        ctx.create_table(trips_table)
        made: dict = {}
        annotate.run(ctx, resolved["annotations"], made)
        out = {t: dict(v) for t, v in made.items() if t not in ("tours", "trips") and v}
        out["tours"] = tours_table
        out["trips"] = trips_table
        return out

    return Component(
        STEP, run, provides={"tours": ["stop_frequency", "primary_purpose"], "trips": list(TRIP_COLUMNS)},
        resolve_needs=resolve_needs, resolve_provides=resolve_provides,
        description=f"stops on the way out and back ({len(alternatives)} alternatives, {len(segments)} segments), "
                    f"and the trips",
        info={"generated_source": "\n\n".join(s[4] for s in segments), "trace_table": "tours",
              "segments": {s[0]: s[1] for s in segments}, "max_trips_per_leg": max_trips},
    )


def max_trips_per_leg(settings, alts: pd.DataFrame) -> int:
    """1.x's determine_max_trips_per_leg: MAX_TRIPS_PER_LEG, or the alternatives' most stops + 1."""
    given = settings.get("MAX_TRIPS_PER_LEG")
    numeric = [int(alts[c].max()) for c in alts.columns if alts[c].astype(str).str.isnumeric().all()]
    found = max(numeric) + 1 if numeric else None
    if given is not None:
        return int(given)
    if found is not None and found > 1:
        return found
    return DEFAULT_MAX_TRIPS_PER_LEG


def initialize_from_tours(tours: pd.DataFrame, alts: pd.DataFrame, max_trips: int) -> pd.DataFrame:
    """1.x's trip.initialize_from_tours: the trips of each tour, with 1.x's trip IDs."""
    trips = alts.loc[tours.stop_frequency]
    unique_tours = tours.copy().reset_index()
    trips.index = unique_tours.index
    trips = trips[["out", "in"]].stack().reset_index()  # per tour: out, then in
    trips.columns = ["tour_temp_index", "direction", "trip_count"]
    trips.trip_count += 1
    trips["outbound"] = trips.direction == "out"
    trips = trips.take(np.repeat(trips.index.values, trips.trip_count.values))
    trips = trips.reset_index(drop=True)
    grouped = trips.groupby(["tour_temp_index", "outbound"])
    trips["trip_num"] = grouped.cumcount() + 1
    for c in ("person_id", "household_id", "primary_purpose"):
        trips[c] = unique_tours[c].reindex(trips.tour_temp_index).to_numpy()
    if isinstance(unique_tours["primary_purpose"].dtype, pd.CategoricalDtype):
        trips["primary_purpose"] = pd.Categorical(trips["primary_purpose"],
                                                  categories=unique_tours["primary_purpose"].cat.categories)
    trips = trips[["person_id", "household_id", "tour_temp_index", "primary_purpose", "trip_num", "outbound",
                   "trip_count"]]
    tour_destination = unique_tours.destination.reindex(trips.tour_temp_index).to_numpy().astype(np.int64)
    tour_origin = unique_tours.origin.reindex(trips.tour_temp_index).to_numpy().astype(np.int64)
    trips["destination"] = np.where(trips.outbound, tour_destination, tour_origin)
    trips["origin"] = np.where(trips.outbound, tour_origin, tour_destination)
    trips["failed"] = False
    trips["tour_id"] = unique_tours[tours.index.name or "tour_id"].reindex(trips.tour_temp_index).to_numpy()
    if trips.groupby(["tour_id", "outbound", "trip_num"])["person_id"].count().max() > 1:
        raise ValueError("tours with the same tour ID: trips would get the same trip IDs")
    canonical_trip_num = (~trips.outbound * max_trips) + trips.trip_num
    trips["trip_id"] = trips["tour_id"] * (2 * max_trips) + canonical_trip_num
    trips = trips.set_index("trip_id", verify_integrity=True)
    del trips["tour_temp_index"]
    return trips
