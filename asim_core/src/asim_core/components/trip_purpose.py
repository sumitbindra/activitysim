"""
Trip purpose: the purpose of each trip's destination (1.x:
``abm/models/trip_purpose.py``).

As 1.x:

* the last outbound trip of a tour goes to the tour's primary purpose; the
  last inbound trip goes home (``work`` for at-work subtours, back to work);
* every other trip (an intermediate stop) draws its purpose from
  ``trip_purpose_probs.csv``: the row matching the trip on
  ``probs_join_cols`` (by default the tour's primary purpose, the direction
  and the person type) and, with ``use_depart_time``, whose departure range
  holds the tour's start (MTC's preprocessor gives the trips the tour's
  ``start`` and the person's ``person_type``). Each row's probabilities are
  divided by their sum; one draw per trip;
* ``trips.purpose`` is a category with the primary purposes' categories,
  'home' added if it isn't one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import Component, MissingInputError, StepContext
from ..spec.legacy import find_config_file, load_global_constants, load_top_settings, read_settings
from ..spec.translate import INDEX
from . import annotate
from .runtime import TableLookup, plain

STEP = "trip_purpose"
CHANNEL = "trips"
JOIN_COLUMNS = ["primary_purpose", "outbound", "person_type"]  # 1.x's default probs_join_cols
TRIP_COLUMNS = ["primary_purpose", "trip_num", "trip_count", "outbound"]


def trip_purpose(configs_dirs: Sequence[str | Path], settings_file: str = "trip_purpose.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_settings(configs_dirs, settings_file)
    top = load_top_settings(configs_dirs)
    join_cols = list(settings.get("probs_join_cols") or JOIN_COLUMNS)
    use_depart_time = settings.get("use_depart_time", True)
    probs_spec = pd.read_csv(find_config_file(configs_dirs, settings.get("PROBS_SPEC", "trip_purpose_probs.csv")),
                             comment="#")
    non_purpose = join_cols + (["depart_range_start", "depart_range_end"] if use_depart_time else [])
    missing = [c for c in non_purpose if c not in probs_spec.columns]
    if missing:
        raise ValueError(f"{settings_file}: the probabilities table lacks {missing}")
    purpose_cols = [c for c in probs_spec.columns if c not in non_purpose]
    # 1.x divides each row by its sum first
    probs_spec[purpose_cols] = probs_spec[purpose_cols].div(probs_spec[purpose_cols].sum(axis=1), axis=0)

    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    pre_spec, pre_tables = annotate.load_preprocessor(configs_dirs, settings.get("preprocessor"), constants, top,
                                                      STEP, "preprocessor")
    annotations = annotate.legacy_annotations(configs_dirs, settings, top, STEP)
    resolved: dict = {}

    def resolve_needs(available):
        trip_cols = set(available.get("trips", set()))
        if not trip_cols:
            raise MissingInputError("needs the trips table")
        needs: dict[str, set[str]] = {"trips": set(TRIP_COLUMNS)}
        targets: set[str] = set()
        bound = None
        if pre_spec is not None:
            bound = pre_spec.bind(trip_cols, {t: available.get(t, set()) for t in pre_tables})
            if bound.n_random or bound.skim_uses():
                raise NotImplementedError(f"{STEP}: preprocessor with random draws or skims")
            targets = set(bound.targets)
            needs["trips"] |= bound.columns_used() - {INDEX}
            for t, c in bound.tables_used():
                needs.setdefault(t, set()).add(c)
        wanted = (set(join_cols) | ({"start"} if use_depart_time else set())) - targets
        missing = sorted(wanted - trip_cols)
        if missing:
            raise MissingInputError(f"trips lack {missing} (made by neither the trips table nor the preprocessor)")
        needs["trips"] |= wanted
        bound_ann, ann_needs, _ = annotate.bind(annotations, available, {"trips": {"purpose"}})
        for t, c in ann_needs.items():
            needs[t] = needs.get(t, set()) | set(c)
        resolved.update(bound=bound, annotations=bound_ann)
        return {t: sorted(c) for t, c in needs.items()}

    def run(ctx: StepContext):
        ids = ctx.ids("trips")
        primary = ctx.column("trips", "primary_purpose")
        primary = primary if isinstance(primary, pd.Categorical) else pd.Categorical(np.asarray(primary, dtype=object))
        if "home" not in primary.categories:
            primary = primary.add_categories(["home"])
        trip_num = np.asarray(ctx.column("trips", "trip_num"))
        trip_count = np.asarray(ctx.column("trips", "trip_count"))
        outbound = np.asarray(ctx.column("trips", "outbound")).astype(bool)
        last = trip_num == trip_count
        purpose = np.asarray(primary.astype(object), dtype=object).copy()
        back = last & ~outbound
        purpose[back] = np.where(purpose[back] == "atwork", "work", "home")

        rows = np.flatnonzero(~last)  # intermediate trips: drawn
        if len(rows):
            cols = {c: np.asarray(plain(ctx.column("trips", c)))[rows] for c in resolved_columns(ctx)}
            cols["primary_purpose"] = np.asarray(primary.astype(object), dtype=object)[rows]
            bound = resolved["bound"]
            if bound is not None:
                c = {k: cols[k] for k in bound.columns_used() - {INDEX}}
                c[INDEX] = ids[rows]
                with np.errstate(all="ignore"):
                    v = bound.function(c, None, len(rows), TableLookup(ctx), None, {})
                for target in bound.targets:
                    cols[target] = annotate._update(cols.get(target), np.asarray(v[target]))
            probs = chooser_probs(ids[rows], cols, probs_spec, join_cols, purpose_cols, use_depart_time)
            rands = ctx.random.uniform(CHANNEL, ids[rows])
            choices = ctx.engine.choose(probs, rands)
            purpose[rows] = np.asarray(purpose_cols, dtype=object)[choices]

        categories = list(primary.categories) + [p for p in purpose_cols if p not in primary.categories]
        # 1.x adds 'home' to the trips' primary purpose categories (in place), if it isn't one
        made = {"trips": {"purpose": pd.Categorical(purpose, categories=categories), "primary_purpose": primary}}
        annotate.run(ctx, resolved["annotations"], made)
        return made

    def resolved_columns(ctx):
        return [c for c in ctx.declared("trips") if c != "primary_purpose"]

    return Component(
        STEP, run, provides={"trips": ["purpose", "primary_purpose"]}, resolve_needs=resolve_needs,
        description=f"trip purposes: last trips by rule, intermediate stops drawn from {len(probs_spec)} "
                    f"probability rows ({len(purpose_cols)} purposes)",
        info={"trace_table": "trips", "probs_spec": probs_spec, "purpose_columns": purpose_cols},
    )


def chooser_probs(trip_ids, cols, probs_spec, join_cols, purpose_cols, use_depart_time) -> np.ndarray:
    """
    Each trip's probabilities: its row of the table, joined as 1.x joins them
    (a left merge on the join columns, then the row whose departure range
    holds the tour's start). Every trip must match exactly one row.
    """
    trips = pd.DataFrame({c: cols[c] for c in join_cols}, index=pd.Index(trip_ids, name="trip_id"))
    if use_depart_time:
        trips["start"] = cols["start"]
    choosers = pd.merge(trips.reset_index(), probs_spec, on=join_cols, how="left").set_index("trip_id")
    if use_depart_time:
        match = (choosers.start >= choosers["depart_range_start"]) & (choosers.start <= choosers["depart_range_end"])
        choosers = choosers[match]
    if not choosers.index.equals(trips.index):
        unmatched = trips.index[~trips.index.isin(choosers.index)]
        raise ValueError(f"{STEP}: {len(unmatched)} trips match no row (or several) of the probabilities on "
                         f"{join_cols}{' and the departure range' if use_depart_time else ''}, "
                         f"e.g. {unmatched[:5].tolist()}")
    return choosers[purpose_cols].to_numpy(dtype=np.float64)
