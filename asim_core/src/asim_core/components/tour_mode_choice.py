"""
Tour mode choice: a mode for every tour, and the tour's mode choice logsum.

Reads today's tour_mode_choice.yaml, its spec, coefficients, coefficient
template and preprocessor unchanged, and follows ActivitySim 1.x's
``tour_mode_choice_simulate`` (activitysim/abm/models/tour_mode_choice.py and
util/mode.py):

* choosers are tours, with their person's, household's, home zone's and
  accessibility columns (looked up by key, not merged into a copy);
* each tour gets a purpose: its tour type, or ``univ`` for a university
  student's school tour. Each purpose (segment) has its own coefficients,
  from the coefficient template;
* each tour's departure and return times get skim time period labels
  (``out_period``, ``in_period``), used by the time-of-day skim lookups;
* per segment: the preprocessor adds columns (including random wait times
  for taxis and ride-hailing), then the nested logit model chooses.

Skims are looked up as in 1.x's ``setup_skims``: ``od_skims`` from the home
zone to the destination, ``do_skims`` back; ``odt_skims`` / ``dot_skims`` at
the departure / return period, ``odr_skims`` / ``dor_skims`` the other way
round.

Not supported yet (the loader says so): park-and-ride, trip mode choice
logsums inside tour mode choice, annotation after the model, multinomial
mode choice models, and at-work subtours (a separate 1.x step).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, MissingInputError, Needs, StepContext
from ..skims import time_periods_from_settings
from ..spec.legacy import (
    find_config_file,
    load_assignment_spec,
    load_choice_model,
    load_global_constants,
    load_top_settings,
    read_yaml,
)
from .simple_choice import ChooserView

STEP = "tour_mode_choice_simulate"  # 1.x step name (random streams depend on it)
CHANNEL = "tours"
ORIG, DEST = "home_zone_id", "destination"  # 1.x setup_skims defaults
OUT_TIME, IN_TIME = "start", "end"

# skim lookups formulas can use: (origin column, destination column, period column or None)
SKIM_WRAPPERS = {
    "odt_skims": (ORIG, DEST, "out_period"),
    "dot_skims": (DEST, ORIG, "in_period"),
    "odr_skims": (ORIG, DEST, "in_period"),
    "dor_skims": (DEST, ORIG, "out_period"),
    "od_skims": (ORIG, DEST, None),
    "do_skims": (DEST, ORIG, None),
}
# names 1.x's skims dict also makes available to formulas
SKIM_LOCALS = {"orig_col_name": ORIG, "dest_col_name": DEST,
               "out_time_col_name": OUT_TIME, "in_time_col_name": IN_TIME}
# columns this step adds to the choosers before the preprocessor runs
ADDED = ("tour_purpose", "out_period", "in_period")

VIEW = ChooserView("tours", [
    ("persons", "person_id"),
    ("households", "household_id"),
    ("land_use", "home_zone_id"),
    ("accessibility", "home_zone_id"),
])

_UNSUPPORTED = {
    "COMPUTE_TRIP_MODE_CHOICE_LOGSUMS": "trip mode choice logsums inside tour mode choice",
    "annotate_tours": "annotation after the model",
    "annotate_persons": "annotation after the model",
    "annotate_households": "annotation after the model",
    "annotate_trips": "annotation after the model",
}


def tour_mode_choice(configs_dirs: Sequence[str | Path],
                     settings_file: str = "tour_mode_choice.yaml") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    unsupported = [f"{k} ({why})" for k, why in _UNSUPPORTED.items() if settings.get(k)]
    if unsupported:
        raise NotImplementedError(f"{settings_file}: not supported yet: {', '.join(unsupported)}")

    definition = load_choice_model(
        configs_dirs, settings_file, name="tour_mode_choice",
        handled=["preprocessor", "nontour_preprocessor"],
        formulas_see_coefficients=True, skim_accessors=list(SKIM_WRAPPERS),
        extra_constants=SKIM_LOCALS,
    )
    if definition.logit_type != "NL":
        raise NotImplementedError("multinomial tour mode choice is not supported yet")
    logsum_column = settings.get("MODE_CHOICE_LOGSUM_COLUMN_NAME")
    alternatives = definition.alternatives
    categories = sorted([""] + alternatives)  # as the tours table holds them after 1.x's step
    periods = time_periods_from_settings(configs_dirs)

    # preprocessor (1.x runs it for each segment, before the model)
    pre = settings.get("preprocessor")
    if isinstance(pre, list):
        if len(pre) != 1:
            raise NotImplementedError("more than one preprocessor is not supported yet")
        pre = pre[0]
    pre_spec = None
    pre_tables: list[str] = []
    if pre:
        pre_tables = list(pre.get("TABLES") or [])
        constants = dict(load_global_constants(configs_dirs))
        constants.update(settings.get("CONSTANTS") or {})
        constants.update(SKIM_LOCALS)
        spec_file = str(pre["SPEC"])
        pre_spec = load_assignment_spec(
            configs_dirs, spec_file if spec_file.endswith(".csv") else spec_file + ".csv",
            constants, skim_accessors=list(SKIM_WRAPPERS), name=f"{STEP}_preprocessor",
            tables=pre_tables, coefficients=sorted(definition.coefficient_names),
            settings=load_top_settings(configs_dirs),
        )

    # per segment: coefficients, which terms are computed (1.x skips terms whose
    # coefficients are all zero), and the compiled terms
    segments = {}
    for seg in definition.segments:
        coef = definition.coefficient_matrix(segment=seg)
        keep = ~(coef == 0).all(axis=1)
        segments[seg] = {"coef": coef[keep], "keep": keep, "values": definition.coefficient_values(seg),
                         "nests": definition.nest_tree(seg)}
    compiled: dict[bytes, tuple] = {}
    for seg in segments.values():
        key = seg["keep"].tobytes()
        if key not in compiled:
            compiled[key] = definition.utility_function(seg["keep"])
        seg["terms"], seg["source"] = compiled[key]
    any_term = np.any([s["keep"] for s in segments.values()], axis=0)
    spec_columns = definition.columns_used(any_term)
    spec_skims = definition.skim_uses(any_term)

    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]) -> Needs:
        view_cols = VIEW.columns(available)
        if "pnr_zone_id" in view_cols:
            raise MissingInputError("park-and-ride (pnr_zone_id) is not supported yet")
        df_cols = view_cols | set(ADDED)
        bound = None
        pre_targets: set[str] = set()
        pre_columns: set[str] = set()
        skim_uses = set(spec_skims)
        tables: set[tuple[str, str]] = set()
        if pre_spec is not None:
            bound = pre_spec.bind(df_cols, {t: available.get(t, set()) for t in pre_tables})
            pre_targets = set(bound.targets)
            pre_columns = bound.columns_used()  # read from the choosers as they are
            skim_uses |= bound.skim_uses()
            tables = bound.tables_used()
        # the model's formulas also see the preprocessor's results
        columns = (set(spec_columns) - pre_targets) | pre_columns
        columns = columns - set(ADDED) - {"tour_id"}
        columns |= {"tour_type", OUT_TIME, IN_TIME, ORIG, DEST}
        if "is_university" in view_cols:
            columns.add("is_university")
        needs = {t: set(c) for t, c in VIEW.resolve(columns, available).items()}
        for t, c in tables:
            needs.setdefault(t, set()).add(c)
        needs[SKIMS] = set()
        for accessor, key, how in skim_uses:
            if SKIM_WRAPPERS[accessor][2] is None:
                needs[SKIMS].add(key)
            elif how != "get":
                raise MissingInputError(f"{accessor}.{how}() is not available for time-of-day skims")
            else:
                needs[SKIMS].update(f"{key}__{p}" for p in periods.unique_labels)
        resolved["needs"] = {t: sorted(c) for t, c in needs.items()}
        resolved["bound"] = bound
        return resolved["needs"]

    def run(ctx: StepContext):
        needs, bound = resolved["needs"], resolved["bound"]
        ids = ctx.ids("tours")
        view_needs = {t: c for t, c in needs.items() if t != SKIMS}
        cols = {c: _plain(v) for c, v in VIEW.gather(ctx, view_needs).items()}
        cols.update(chooser_columns(cols, periods))
        orig = ctx.skim_positions(cols[ORIG])
        dest = ctx.skim_positions(cols[DEST])
        out_idx = periods.label_index(cols[OUT_TIME])
        in_idx = periods.label_index(cols[IN_TIME])

        choices = np.empty(len(ids), dtype=np.int64)
        logsums = np.empty(len(ids), dtype=np.float64)
        purposes = cols["tour_purpose"]
        traced = ctx.tracer.targets("tours", ids)
        for purpose in sorted(set(purposes)):  # 1.x's groupby order
            if purpose not in segments:
                raise ValueError(f"{STEP}: no coefficients for tour purpose '{purpose}' "
                                 f"(template segments: {list(segments)})")
            seg = segments[purpose]
            rows = np.flatnonzero(purposes == purpose)
            seg_ids = ids[rows]
            c = {k: v[rows] for k, v in cols.items()}
            s = SkimLookup(ctx, periods.unique_labels, orig[rows], dest[rows],
                           {"out_period": out_idx[rows], "in_period": in_idx[rows]})
            k = seg["values"]
            pre_values = {}
            if bound is not None:
                draws = ctx.random.normal(CHANNEL, seg_ids, bound.n_random) if bound.n_random else None
                with np.errstate(all="ignore"):  # 1.x logs numpy warnings and carries on
                    pre_values = bound.function(c, s, len(rows), _table_lookup(ctx), draws, k)
                for name in bound.targets:
                    c[name] = _assign(c.get(name), pre_values[name])
            with np.errstate(all="ignore"):
                utils = ctx.engine.utilities(seg["terms"](c, s, k), seg["coef"], len(rows))
            probs, bad, seg_logsums = ctx.engine.nested_probabilities(utils, alternatives, seg["nests"])
            if bad.any():
                raise ValueError(f"{STEP}: {int(bad.sum())} '{purpose}' tours have no usable "
                                 f"probabilities, e.g. tour IDs {seg_ids[bad][:5].tolist()}")
            rands = ctx.random.uniform(CHANNEL, seg_ids)
            seg_choices = ctx.engine.choose(probs, rands)
            choices[rows] = seg_choices
            logsums[rows] = seg_logsums
            local = np.flatnonzero(np.isin(rows, traced))
            if len(local):
                _trace(ctx, purpose, seg_ids, local, c, pre_values, bound, utils, probs,
                       seg_logsums, rands, seg_choices, alternatives)

        modes = pd.Categorical(np.asarray(alternatives, dtype=object)[choices], categories=categories)
        out = {"tour_mode": modes}
        if logsum_column:
            out[logsum_column] = logsums
        return {"tours": out}

    return Component(
        name=STEP,
        run=run,
        provides={"tours": ["tour_mode"] + ([logsum_column] if logsum_column else [])},
        resolve_needs=resolve_needs,
        description=f"tour mode choice: nested logit over {len(alternatives)} modes, "
                    f"{len(segments)} purpose segments",
        info={
            "definition": definition,
            "preprocessor": pre_spec,
            "segments": segments,
            "generated_source": next(iter(compiled.values()))[1] if compiled else "",
            "chooser_table": "tours",
            "trace_table": "tours",
            "resolved": resolved,
            "periods": periods,
        },
    )


def chooser_columns(cols: Mapping[str, np.ndarray], periods) -> dict[str, np.ndarray]:
    """tour_purpose, out_period and in_period, as 1.x adds them to the choosers."""
    tour_type = cols["tour_type"]
    university = cols["is_university"].astype(bool) if "is_university" in cols else False
    not_university = (tour_type != "school") | ~np.asarray(university)
    purpose = np.where(not_university, tour_type, "univ").astype(object)
    return {
        "tour_purpose": purpose,
        "out_period": periods.labels_for(cols[OUT_TIME]),
        "in_period": periods.labels_for(cols[IN_TIME]),
    }


class SkimLookup:
    """
    ``s(accessor, key, how)`` for one segment's tours: the skim lookups of
    1.x's setup_skims. Time-of-day lookups pick each tour's period skim
    (``KEY__AM``, ...), period by period.
    """

    def __init__(self, ctx: StepContext, labels: list[str], orig: np.ndarray, dest: np.ndarray,
                 period_index: Mapping[str, np.ndarray]):
        self.ctx = ctx
        self.labels = labels
        self.zones = {ORIG: orig, DEST: dest}
        # rows in each period, per period column
        self.groups = {
            col: [(lab, np.flatnonzero(idx == j)) for j, lab in enumerate(labels)]
            for col, idx in period_index.items()
        }
        self.n = len(orig)

    def __call__(self, accessor: str, key: str, how: str = "get"):
        o_col, d_col, period_col = SKIM_WRAPPERS[accessor]
        o, d = self.zones[o_col], self.zones[d_col]
        if period_col is None:
            if how == "reverse":
                return self.ctx.skim(key, d, o)
            if how == "max":
                return np.maximum(self.ctx.skim(key, d, o), self.ctx.skim(key, o, d))
            return self.ctx.skim(key, o, d)
        out = None
        for label, rows in self.groups[period_col]:
            if not len(rows):
                continue
            values = self.ctx.skim(f"{key}__{label}", o[rows], d[rows])
            if out is None:
                out = np.empty(self.n, dtype=values.dtype)
            out[rows] = values
        return out if out is not None else np.empty(0, dtype=np.float32)


def _table_lookup(ctx: StepContext):
    def t(table: str, column: str, keys):
        return _plain(ctx.column_at(table, column, ctx.positions(table, keys)))
    return t


def _plain(values):
    """Categoricals as plain arrays of their values, as formulas see them."""
    if isinstance(values, pd.Categorical):
        return np.asarray(values, dtype=object if values.categories.dtype == object else None)
    return values


def _assign(old, new):
    """
    A preprocessor result for a column the choosers already have replaces it,
    except where the new value is missing (1.x uses pandas ``update``).
    """
    if old is None:
        return new
    new = np.asarray(new)
    if new.dtype.kind == "f":
        missing = np.isnan(new)
        if missing.any():
            new = np.where(missing, old, new)
    return new


def _trace(ctx, purpose, seg_ids, local, c, pre_values, bound, utils, probs, logsums, rands,
           choices, alts):
    if bound is not None:
        names = [n for n in pre_values]
        rows = []
        for r in local:
            for n in names:
                v = pre_values[n]
                rows.append([seg_ids[r], n, v[r] if np.ndim(v) else v])
        ctx.tracer.write(ctx.step, f"{purpose}.preprocessor", ["tour_id", "variable", "value"], rows)
    header = (["tour_id"] + [f"utility_{a}" for a in alts] + [f"probability_{a}" for a in alts]
              + ["logsum", "random_draw", "choice"])
    rows = [[seg_ids[r]] + list(utils[r]) + list(probs[r]) + [logsums[r], rands[r, 0], alts[choices[r]]]
            for r in local]
    ctx.tracer.write(ctx.step, f"{purpose}.choices", header, rows)
