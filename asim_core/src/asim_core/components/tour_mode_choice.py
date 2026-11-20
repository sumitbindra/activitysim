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

The model itself (``TourModeChoiceModel``) is separate from the step, so
other components can use it: tour scheduling asks it for the logsums of a
tour's possible departure and return times, with a destination column of
its choosing. It reads the same files.

Skims are looked up as in 1.x's ``setup_skims``: ``od_skims`` from the
origin (home zone) to the destination, ``do_skims`` back; ``odt_skims`` /
``dot_skims`` at the departure / return period, ``odr_skims`` /
``dor_skims`` the other way round.

Not supported yet (the loader says so): park-and-ride, trip mode choice
logsums inside tour mode choice, annotation after the model, multinomial
mode choice models, and at-work subtours (a separate 1.x step).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import SKIMS, Component, MissingInputError, Needs, StepContext
from ..skims import time_periods_from_settings
from ..spec.definition import BoundAssignments
from ..spec.legacy import (
    find_config_file,
    load_assignment_spec,
    load_choice_model,
    load_global_constants,
    load_top_settings,
    read_yaml,
)
from .runtime import TableLookup, plain
from .simple_choice import ChooserView

STEP = "tour_mode_choice_simulate"  # 1.x step name (random streams depend on it)
CHANNEL = "tours"
ORIG, DEST = "home_zone_id", "destination"  # 1.x setup_skims defaults
OUT_TIME, IN_TIME = "start", "end"

# skim lookups formulas can use: (from, to, period column or None)
SKIM_WRAPPERS = {
    "odt_skims": ("orig", "dest", "out_period"),
    "dot_skims": ("dest", "orig", "in_period"),
    "odr_skims": ("orig", "dest", "in_period"),
    "dor_skims": ("dest", "orig", "out_period"),
    "od_skims": ("orig", "dest", None),
    "do_skims": ("dest", "orig", None),
}
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
    "include_pnr_for_logsums": "park-and-ride",
}


def skim_locals(dest: str) -> dict[str, str]:
    """Names 1.x's skims dict makes available to formulas."""
    return {"orig_col_name": ORIG, "dest_col_name": dest,
            "out_time_col_name": OUT_TIME, "in_time_col_name": IN_TIME}


@dataclass
class BoundModel:
    """The model bound to a set of chooser columns and a destination column."""

    dest: str
    preprocessor: BoundAssignments | None
    columns: set[str]                      # chooser columns read (before the preprocessor)
    tables: set[tuple[str, str]]           # other tables' columns looked up
    skims: set[str]                        # skim names needed
    period_columns: set[str] = field(default_factory=set)  # out_period / in_period, if read


class TourModeChoiceModel:
    def __init__(self, configs_dirs: Sequence[str | Path], settings_file: str = "tour_mode_choice.yaml"):
        configs_dirs = [Path(d) for d in configs_dirs]
        settings = read_yaml(find_config_file(configs_dirs, settings_file))
        unsupported = [f"{k} ({why})" for k, why in _UNSUPPORTED.items() if settings.get(k)]
        if unsupported:
            raise NotImplementedError(f"{settings_file}: not supported yet: {', '.join(unsupported)}")
        self.settings_file = settings_file
        self.definition = load_choice_model(
            configs_dirs, settings_file, name="tour_mode_choice",
            handled=["preprocessor", "nontour_preprocessor"],
            formulas_see_coefficients=True, skim_accessors=list(SKIM_WRAPPERS),
            extra_constants=skim_locals(DEST),
        )
        if self.definition.logit_type != "NL":
            raise NotImplementedError("multinomial tour mode choice is not supported yet")
        uses_names = set().union(*(t.constants for t in self.definition.translated))
        self._spec_uses_dest_name = bool(uses_names & {"dest_col_name", "orig_col_name"})
        self.logsum_column = settings.get("MODE_CHOICE_LOGSUM_COLUMN_NAME")
        self.alternatives = self.definition.alternatives
        self.periods = time_periods_from_settings(configs_dirs)

        # preprocessor (1.x runs it for each segment, before the model)
        pre = settings.get("preprocessor")
        if isinstance(pre, list):
            if len(pre) != 1:
                raise NotImplementedError("more than one preprocessor is not supported yet")
            pre = pre[0]
        self.preprocessor = None
        self.preprocessor_tables: list[str] = []
        if pre:
            self.preprocessor_tables = list(pre.get("TABLES") or [])
            constants = dict(load_global_constants(configs_dirs))
            constants.update(settings.get("CONSTANTS") or {})
            constants.update(skim_locals(DEST))
            spec_file = str(pre["SPEC"])
            self.preprocessor = load_assignment_spec(
                configs_dirs, spec_file if spec_file.endswith(".csv") else spec_file + ".csv",
                constants, skim_accessors=list(SKIM_WRAPPERS), name="tour_mode_choice_preprocessor",
                tables=self.preprocessor_tables,
                coefficients=sorted(self.definition.coefficient_names),
                settings=load_top_settings(configs_dirs),
            )

        # per segment: coefficients, which terms are computed (1.x skips terms
        # whose coefficients are all zero), and the compiled terms
        self.segments: dict[str, dict] = {}
        for seg in self.definition.segments:
            coef = self.definition.coefficient_matrix(segment=seg)
            keep = ~(coef == 0).all(axis=1)
            self.segments[seg] = {"coef": coef[keep], "keep": keep,
                                  "values": self.definition.coefficient_values(seg),
                                  "nests": self.definition.nest_tree(seg)}
        compiled: dict[bytes, tuple] = {}
        for seg in self.segments.values():
            key = seg["keep"].tobytes()
            if key not in compiled:
                compiled[key] = self.definition.utility_function(seg["keep"])
            seg["terms"], seg["source"] = compiled[key]
        self.generated_source = next(iter(compiled.values()))[1] if compiled else ""
        any_term = np.any([s["keep"] for s in self.segments.values()], axis=0)
        self._spec_columns = self.definition.columns_used(any_term)
        self._spec_skims = self.definition.skim_uses(any_term)

    # --- binding ------------------------------------------------------------------

    def bind(self, df_columns: set[str], tables: Mapping[str, set[str]], dest: str = DEST) -> BoundModel:
        """
        Work out, for choosers with ``df_columns`` and skims to ``dest``, what
        the model reads: chooser columns, other tables' columns and skims.
        Raises if something is missing or unsupported.
        """
        if dest != DEST and self._spec_uses_dest_name:
            raise NotImplementedError("the mode choice spec uses dest_col_name; only "
                                      "'destination' is supported as its destination")
        bound = None
        pre_targets: set[str] = set()
        pre_columns: set[str] = set()
        skim_uses = set(self._spec_skims)
        table_cols: set[tuple[str, str]] = set()
        if self.preprocessor is not None:
            bound = self.preprocessor.bind(
                df_columns, {t: tables.get(t, set()) for t in self.preprocessor_tables},
                constants=skim_locals(dest),
            )
            pre_targets = set(bound.targets)
            pre_columns = bound.columns_used()  # read from the choosers as they are
            skim_uses |= bound.skim_uses()
            table_cols = bound.tables_used()
        # the model's formulas also see the preprocessor's results
        columns = (set(self._spec_columns) - pre_targets) | pre_columns | {ORIG, dest}
        skims: set[str] = set()
        period_columns: set[str] = set()
        for accessor, key, how in skim_uses:
            period_col = SKIM_WRAPPERS[accessor][2]
            if period_col is None:
                skims.add(key)
            elif how != "get":
                raise MissingInputError(f"{accessor}.{how}() is not available for time-of-day skims")
            else:
                skims.update(f"{key}__{p}" for p in self.periods.unique_labels)
                period_columns.add(period_col)
        return BoundModel(dest, bound, columns, table_cols, skims, period_columns)

    # --- evaluating -----------------------------------------------------------------

    def evaluate(self, ctx: StepContext, bound: BoundModel, c: dict, n: int, segment: str,
                 draw_ids: np.ndarray, logsums_only: bool = False):
        """
        Run the preprocessor and the model for ``n`` rows of chooser columns
        ``c`` (changed in place: the preprocessor's results are added), with
        segment ``segment``'s coefficients. Random draws (the preprocessor's)
        come from each row's ``draw_ids`` stream; rows with the same ID share
        a draw, as 1.x broadcasts them.

        Returns logsums, or (utilities, probabilities, unusable rows,
        logsums, preprocessor values).
        """
        if segment not in self.segments:
            raise ValueError(f"tour mode choice: no coefficients for segment '{segment}' "
                             f"(template segments: {list(self.segments)})")
        seg = self.segments[segment]
        s = SkimLookup(ctx, self.periods, ctx.skim_positions(c[ORIG]), ctx.skim_positions(c[bound.dest]),
                       {p: c[p] for p in ("out_period", "in_period") if p in c})
        k = seg["values"]
        pre_values: dict = {}
        if bound.preprocessor is not None:
            draws = None
            if bound.preprocessor.n_random:
                uniq, inv = np.unique(np.asarray(draw_ids), return_inverse=True)
                draws = ctx.random.normal(CHANNEL, uniq, bound.preprocessor.n_random)[inv]
            with np.errstate(all="ignore"):  # 1.x logs numpy warnings and carries on
                pre_values = bound.preprocessor.function(c, s, n, TableLookup(ctx), draws, k)
            for name in bound.preprocessor.targets:
                c[name] = _assign(c.get(name), pre_values[name])
        with np.errstate(all="ignore"):
            utils = ctx.engine.utilities(seg["terms"](c, s, k), seg["coef"], n)
        if logsums_only:
            return ctx.engine.nested_logsums(utils, self.alternatives, seg["nests"])
        probs, bad, logsums = ctx.engine.nested_probabilities(utils, self.alternatives, seg["nests"])
        return utils, probs, bad, logsums, pre_values


def tour_mode_choice(configs_dirs: Sequence[str | Path],
                     settings_file: str = "tour_mode_choice.yaml") -> Component:
    model = TourModeChoiceModel(configs_dirs, settings_file)
    alternatives = model.alternatives
    categories = sorted([""] + alternatives)  # as the tours table holds them after 1.x's step
    periods = model.periods
    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]) -> Needs:
        view_cols = VIEW.columns(available)
        if "pnr_zone_id" in view_cols:
            raise MissingInputError("park-and-ride (pnr_zone_id) is not supported yet")
        bound = model.bind(view_cols | set(ADDED), available, DEST)
        columns = bound.columns - set(ADDED) - {"tour_id"}
        columns |= {"tour_type", OUT_TIME, IN_TIME}
        if "is_university" in view_cols:
            columns.add("is_university")
        needs = {t: set(c) for t, c in VIEW.resolve(columns, available).items()}
        for t, c in bound.tables:
            needs.setdefault(t, set()).add(c)
        needs[SKIMS] = set(bound.skims)
        resolved["needs"] = {t: sorted(c) for t, c in needs.items()}
        resolved["bound"] = bound
        return resolved["needs"]

    def run(ctx: StepContext):
        needs, bound = resolved["needs"], resolved["bound"]
        ids = ctx.ids("tours")
        view_needs = {t: c for t, c in needs.items() if t != SKIMS}
        cols = {c: plain(v) for c, v in VIEW.gather(ctx, view_needs).items()}
        cols.update(chooser_columns(cols, periods))

        choices = np.empty(len(ids), dtype=np.int64)
        logsums = np.empty(len(ids), dtype=np.float64)
        purposes = cols["tour_purpose"]
        traced = ctx.tracer.targets("tours", ids)
        for purpose in sorted(set(purposes)):  # 1.x's groupby order
            rows = np.flatnonzero(purposes == purpose)
            seg_ids = ids[rows]
            c = {k: v[rows] for k, v in cols.items()}
            utils, probs, bad, seg_logsums, pre_values = model.evaluate(
                ctx, bound, c, len(rows), purpose, seg_ids)
            if bad.any():
                raise ValueError(f"{STEP}: {int(bad.sum())} '{purpose}' tours have no usable "
                                 f"probabilities, e.g. tour IDs {seg_ids[bad][:5].tolist()}")
            rands = ctx.random.uniform(CHANNEL, seg_ids)
            seg_choices = ctx.engine.choose(probs, rands)
            choices[rows] = seg_choices
            logsums[rows] = seg_logsums
            local = np.flatnonzero(np.isin(rows, traced))
            if len(local):
                _trace(ctx, purpose, seg_ids, local, pre_values, bound.preprocessor, utils, probs,
                       seg_logsums, rands, seg_choices, alternatives)

        modes = pd.Categorical(np.asarray(alternatives, dtype=object)[choices], categories=categories)
        out = {"tour_mode": modes}
        if model.logsum_column:
            out[model.logsum_column] = logsums
        return {"tours": out}

    return Component(
        name=STEP,
        run=run,
        provides={"tours": ["tour_mode"] + ([model.logsum_column] if model.logsum_column else [])},
        resolve_needs=resolve_needs,
        description=f"tour mode choice: nested logit over {len(alternatives)} modes, "
                    f"{len(model.segments)} purpose segments",
        info={
            "model": model,
            "definition": model.definition,
            "preprocessor": model.preprocessor,
            "segments": model.segments,
            "generated_source": model.generated_source,
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
    ``s(accessor, key, how)``: the skim lookups of 1.x's setup_skims, for
    rows with origin and destination zone positions. Time-of-day lookups
    pick each row's period skim (``KEY__AM``, ...), period by period.
    ``period_labels`` maps out_period / in_period to each row's label.
    """

    def __init__(self, ctx: StepContext, periods, orig: np.ndarray, dest: np.ndarray,
                 period_labels: Mapping[str, np.ndarray]):
        self.ctx = ctx
        self.zones = {"orig": orig, "dest": dest}
        labels = periods.unique_labels
        self.groups = {}
        for col, values in period_labels.items():
            values = np.asarray(values, dtype=object)
            self.groups[col] = [(lab, np.flatnonzero(values == lab)) for lab in labels]
            covered = sum(len(r) for _, r in self.groups[col])
            if covered != len(values):
                raise ValueError(f"{col}: labels not in the skim time periods {labels}")
        self.n = len(orig)

    def __call__(self, accessor: str, key: str, how: str = "get"):
        o_role, d_role, period_col = SKIM_WRAPPERS[accessor]
        o, d = self.zones[o_role], self.zones[d_role]
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


def _trace(ctx, purpose, seg_ids, local, pre_values, preprocessor, utils, probs, logsums, rands,
           choices, alts):
    if preprocessor is not None:
        rows = []
        for r in local:
            for name, v in pre_values.items():
                rows.append([seg_ids[r], name, v[r] if np.ndim(v) else v])
        ctx.tracer.write(ctx.step, f"{purpose}.preprocessor", ["tour_id", "variable", "value"], rows)
    header = (["tour_id"] + [f"utility_{a}" for a in alts] + [f"probability_{a}" for a in alts]
              + ["logsum", "random_draw", "choice"])
    rows = [[seg_ids[r]] + list(utils[r]) + list(probs[r]) + [logsums[r], rands[r, 0], alts[choices[r]]]
            for r in local]
    ctx.tracer.write(ctx.step, f"{purpose}.choices", header, rows)
