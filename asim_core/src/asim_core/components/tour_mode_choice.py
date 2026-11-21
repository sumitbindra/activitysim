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

At-work subtours (1.x's atwork_subtour_mode_choice) use the same model:
``atwork_subtour_mode_choice`` runs it for those tours only, from the
workplace, with the ``atwork`` coefficients.

Not supported yet (the loader says so): park-and-ride, trip mode choice
logsums inside tour mode choice, annotation after the model and
multinomial mode choice models.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import ast
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..component import SKIMS, Component, MissingInputError, Needs, StepContext
from ..logit import FormulaValues
from ..skims import time_periods_from_settings
from ..spec.definition import (
    AssignmentSpec,
    BoundAssignments,
    ChoiceModelDefinition,
    LanguageScope,
    Nest,
    Term,
)
from ..spec.language import compile_formula
from ..spec.legacy import (
    find_config_file,
    load_assignment_spec,
    load_choice_model,
    load_global_constants,
    load_top_settings,
    read_yaml,
)
from ..spec.model_file import (
    ModelFile,
    is_model_file,
    read_coefficients_table,
    read_derived,
    read_model_yaml,
    read_terms,
    validate,
)
from ..spec.translate import build_function, compile_function
from .roles import TOUR_ROLES, check_roles
from .runtime import TableLookup, plain
from .simple_choice import ChooserView

STEP = "tour_mode_choice_simulate"  # 1.x step name (random streams depend on it)
ATWORK_STEP = "atwork_subtour_mode_choice"
ATWORK_ORIG = "workplace_zone_id"  # 1.x: at-work subtours start from the workplace
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


# asim_core's language: skim.NAME[from, to(, period)] -> the lookups above
SKIM_PATTERNS = {"o d out_period": "odt_skims", "d o in_period": "dot_skims",
                 "o d in_period": "odr_skims", "d o out_period": "dor_skims",
                 "o d": "od_skims", "d o": "do_skims"}
PURPOSE_1X = "where(tour.tour_type == 'school' and person.is_university, 'univ', tour.tour_type)"


class Outputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: str = "tour_mode"
    logsum: str | None = None


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 1.x skips terms whose coefficients are all zero (per segment)
    drop_zero_terms: Literal[True] = True
    # random draws in derived values come from each chooser's stream, in the
    # order the rows are listed, before the choice's draw (1.x)
    random_draws: Literal["chooser_stream"] = "chooser_stream"


class TourModeChoiceFile(ModelFile):
    kind: Literal["tour_mode_choice"]
    alternatives: list[str]
    nests: Nest
    terms: str
    coefficients: str  # one column per segment (tour purpose)
    derived: str | None = None
    # tables derived values may look up by ID (table.column[ids])
    derived_tables: list[str] = []
    # formulas may use coefficient values by name (1.x allows it in mode choice)
    formulas_see_coefficients: bool = True
    # values when this model runs as its own step; steps using it for
    # logsums set their own (in their logsums: parameters:)
    parameters: dict[str, Any] = {}
    # each tour's purpose, which picks its coefficients (segment)
    tour_purpose: str = PURPOSE_1X
    # the times whose skim time periods out_period and in_period hold
    time_periods: dict[Literal["out_period", "in_period"], str] = {"out_period": "tour.start",
                                                                   "in_period": "tour.end"}
    outputs: Outputs = Outputs()
    options: Options = Options()


def _uses_zone_values(formula: str) -> bool:
    """True if o or d are used as values (not only to look skims up, which follow the step's zones)."""
    def walk(node):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name) and node.value.value.id == "skim"):
            keys = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return any(walk(k) for k in keys if not (isinstance(k, ast.Name) and k.id in ("o", "d")))
        if isinstance(node, ast.Name):
            return node.id in ("o", "d")
        return any(walk(child) for child in ast.iter_child_nodes(node))
    return walk(ast.parse(formula, mode="eval"))


def skim_locals(dest: str, orig: str = ORIG) -> dict[str, str]:
    """Names 1.x's skims dict makes available to formulas."""
    return {"orig_col_name": orig, "dest_col_name": dest,
            "out_time_col_name": OUT_TIME, "in_time_col_name": IN_TIME}


@dataclass
class BoundModel:
    """The model bound to a set of chooser columns, an origin and a destination column."""

    dest: str
    preprocessor: BoundAssignments | None
    columns: set[str]                      # chooser columns read (before the preprocessor)
    tables: set[tuple[str, str]]           # other tables' columns looked up
    skims: set[str]                        # skim names needed
    period_columns: set[str] = field(default_factory=set)  # out_period / in_period, if read
    orig: str = ORIG


class TourModeChoiceModel:
    def __init__(self, configs_dirs: Sequence[str | Path], settings_file: str = "tour_mode_choice.yaml"):
        configs_dirs = [Path(d) for d in configs_dirs]
        path = find_config_file(configs_dirs, settings_file)
        self.language = "asim1"
        self.parameters: dict = {}  # this step's parameter values (model files)
        self.purpose = None         # (function, columns) for model files
        self.time_columns = {"out_period": OUT_TIME, "in_period": IN_TIME}
        self.bind_log: list[tuple] = []  # (df_columns, dest, orig) of each bind, for the converter
        if is_model_file(path):
            self._load_model_file(path, configs_dirs)
        else:
            self._load_legacy(configs_dirs, settings_file)
        self._prepare_segments()

    def _load_model_file(self, path, configs_dirs):
        f = validate(TourModeChoiceFile, read_model_yaml(path), path)
        self.language = "asim_core"
        self.settings_file = Path(path).name
        self.file = f
        self.parameters = dict(f.parameters)
        constants = dict(load_global_constants(configs_dirs))
        constants.update(f.constants)
        self.constants = constants
        rows, value_cols = read_terms(f.terms)
        extra = [c for c in value_cols if c not in f.alternatives]
        if extra:
            raise ValueError(f"{f.terms}: columns {extra} are not alternatives")
        _, by_seg = read_coefficients_table(f.coefficients, f.coefficient_values)
        if by_seg is None:
            raise ValueError(f"{f.coefficients}: tour mode choice needs one column per tour purpose")
        generic = list(next(iter(by_seg.values())))
        template = {g: {seg: f"{g}@{seg}" for seg in by_seg} for g in generic}
        coefficients = {f"{g}@{seg}": c for seg, cs in by_seg.items() for g, c in cs.items()}
        self.preprocessor = None
        self.preprocessor_tables = list(f.derived_tables)
        targets = []
        if f.derived:
            scope = LanguageScope(roles=list(TOUR_ROLES), derived=list(ADDED), tables=f.derived_tables,
                                  skims=SKIM_PATTERNS, parameters={k: None for k in f.parameters})
            self.preprocessor = AssignmentSpec(
                name="tour_mode_choice_derived", assignments=read_derived(f.derived),
                constants={**constants, **skim_locals(DEST)}, coefficients=sorted(generic),
                settings=load_top_settings(configs_dirs), language="asim_core", scope=scope,
                source_files=[Path(f.derived).name])
            targets = self.preprocessor.targets
        terms = []
        for label, _, formula, target, cells in rows:
            if target:
                raise ValueError(f"{f.terms}: '{target} = ...' rows are not supported in this model")
            terms.append(Term(label=label, formula=formula, coefficients=cells))
        term_scope = LanguageScope(roles=list(TOUR_ROLES), derived=list(ADDED) + list(targets),
                                   skims=SKIM_PATTERNS, parameters=self.parameters)
        self.definition = ChoiceModelDefinition(
            name="tour_mode_choice", logit_type="NL", nests=f.nests, alternatives=f.alternatives,
            constants={**constants, **skim_locals(DEST)}, terms=terms, coefficients=coefficients,
            coefficient_template=template, formulas_see_coefficients=f.formulas_see_coefficients,
            language="asim_core", scope=term_scope,
            source_files=[Path(p).name for p in f.source_files] + [Path(f.terms).name, Path(f.coefficients).name],
        )
        self._spec_uses_dest_name = any(_uses_zone_values(t.formula) for t in terms)
        self.logsum_column = f.outputs.logsum
        self.choice_column = f.outputs.choice
        self.alternatives = self.definition.alternatives
        self.periods = time_periods_from_settings(configs_dirs)
        role_scope = LanguageScope(roles=list(TOUR_ROLES)).make(constants)
        purpose = compile_formula(f.tour_purpose, role_scope)
        src = build_function("tour_purpose", [("tour_purpose", purpose)])
        self.purpose = (compile_function(src, "tour_purpose"), purpose)
        for name, text in f.time_periods.items():
            node = ast.parse(text, mode="eval").body
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in TOUR_ROLES):
                raise ValueError(f"{path}: time_periods.{name} must be a column, e.g. tour.start")
            self.time_columns[name] = node.attr

    def _load_legacy(self, configs_dirs, settings_file):
        settings = read_yaml(find_config_file(configs_dirs, settings_file))
        self.choice_column = "tour_mode"
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

    def _prepare_segments(self):
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

    def bind(self, df_columns: set[str], tables: Mapping[str, set[str]], dest: str = DEST,
             orig: str = ORIG, parameters: Mapping | None = None,
             roles: Mapping[str, set[str]] | None = None,
             period_labels: Mapping[str, Iterable[str]] | None = None) -> BoundModel:
        """
        Work out, for choosers with ``df_columns`` and skims from ``orig`` to
        ``dest`` (column names), what the model reads: chooser columns, other
        tables' columns and skims. Raises if something is missing or
        unsupported. For model files, ``parameters`` are the using step's
        values (default: this model's own step's) and ``roles`` the columns
        each table name offers in that step (checked if given).
        ``period_labels``: the time periods the choosers can have, per period
        column (``out_period``, ``in_period``), when the using step fixes them
        (see ``fixed_period_labels``); time-of-day skims are then declared for
        those periods only. By default, for every period.
        """
        self.bind_log.append((set(df_columns), dest, orig))
        if (dest, orig) != (DEST, ORIG) and self._spec_uses_dest_name:
            raise NotImplementedError("the mode choice spec uses dest_col_name or orig_col_name; "
                                      "only the default origin and destination are supported")
        bound = None
        pre_targets: set[str] = set()
        pre_columns: set[str] = set()
        skim_uses = set(self._spec_skims)
        table_cols: set[tuple[str, str]] = set()
        if self.preprocessor is not None:
            extra = dict(skim_locals(dest, orig))
            if self.language == "asim_core":
                values = dict(self.parameters if parameters is None else parameters)
                missing = sorted(set(self.parameters) - set(values))
                if missing:
                    raise ValueError(f"tour mode choice: the step using it must set parameters {missing}")
                extra.update(values)
            bound = self.preprocessor.bind(
                df_columns, {t: tables.get(t, set()) for t in self.preprocessor_tables},
                constants=extra,
            )
            if roles is not None and self.language == "asim_core":
                check_roles("tour mode choice derived values", bound.translated, roles)
            pre_targets = set(bound.targets)
            pre_columns = bound.columns_used()  # read from the choosers as they are
            skim_uses |= bound.skim_uses()
            table_cols = bound.tables_used()
        if roles is not None and self.language == "asim_core":
            check_roles("tour mode choice terms", self.definition.translated, roles)
        # the model's formulas also see the preprocessor's results
        columns = (set(self._spec_columns) - pre_targets) | pre_columns | {orig, dest}
        skims: set[str] = set()
        period_columns: set[str] = set()
        for accessor, key, how in skim_uses:
            period_col = SKIM_WRAPPERS[accessor][2]
            if period_col is None:
                skims.add(key)
            elif how != "get":
                raise MissingInputError(f"{accessor}.{how}() is not available for time-of-day skims")
            else:
                labels = (self.periods.unique_labels if period_labels is None
                          else period_labels.get(period_col, self.periods.unique_labels))
                skims.update(f"{key}__{p}" for p in labels)
                period_columns.add(period_col)
        return BoundModel(dest, bound, columns, table_cols, skims, period_columns, orig)

    def fixed_period_labels(self, out_time, in_time) -> dict[str, set[str]]:
        """The period labels of choosers whose times are fixed (logsums at set times)."""
        return {"out_period": {self.periods.labels_for([out_time])[0]},
                "in_period": {self.periods.labels_for([in_time])[0]}}

    # --- the tour purpose and time periods ---------------------------------------------

    def chooser_inputs(self, view_cols: set[str]) -> set[str]:
        """Columns the tour purpose and time periods are worked out from."""
        if self.purpose is None:  # 1.x's rule
            return {"tour_type", OUT_TIME, IN_TIME} | ({"is_university"} & set(view_cols))
        return set(self.purpose[1].columns) | set(self.time_columns.values())

    def chooser_columns(self, cols: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """tour_purpose, out_period and in_period for the choosers (1.x adds them)."""
        if self.purpose is None:
            return chooser_columns(cols, self.periods)
        fn = self.purpose[0]
        with np.errstate(all="ignore"):
            purpose = next(iter(fn(cols)))
        n = len(next(iter(cols.values())))
        purpose = np.broadcast_to(np.asarray(purpose, dtype=object), (n,)).copy()
        return {"tour_purpose": purpose,
                **{k: self.periods.labels_for(cols[c]) for k, c in self.time_columns.items()}}

    # --- evaluating -----------------------------------------------------------------

    def evaluate(self, ctx: StepContext, bound: BoundModel, c: dict, n: int, segment: str,
                 draw_ids: np.ndarray, logsums_only: bool = False, channel: str = CHANNEL):
        """
        Run the preprocessor and the model for ``n`` rows of chooser columns
        ``c`` (changed in place: the preprocessor's results are added), with
        segment ``segment``'s coefficients. Random draws (the preprocessor's)
        come from each row's ``draw_ids`` stream; rows with the same ID share
        a draw, as 1.x broadcasts them. ``channel``: the random stream (tours,
        or persons in location choice).

        Returns logsums, or (utilities, probabilities, unusable rows,
        logsums, preprocessor values).
        """
        if segment not in self.segments:
            raise ValueError(f"tour mode choice: no coefficients for segment '{segment}' "
                             f"(template segments: {list(self.segments)})")
        seg = self.segments[segment]
        s = SkimLookup(ctx, self.periods, ctx.skim_positions(c[bound.orig]), ctx.skim_positions(c[bound.dest]),
                       {p: c[p] for p in ("out_period", "in_period") if p in c})
        # the preprocessor and the formulas may share the skim values they read
        pre_fn = bound.preprocessor.function if bound.preprocessor is not None else None
        s = ctx.engine.skim_reuse(s, [getattr(pre_fn, "_asim_source", None),
                                      getattr(seg["terms"], "_asim_source", None)])
        k = seg["values"]
        pre_values: dict = {}
        if bound.preprocessor is not None:
            draws = None
            if bound.preprocessor.n_random:
                uniq, inv = np.unique(np.asarray(draw_ids), return_inverse=True)
                draws = ctx.random.normal(channel, uniq, bound.preprocessor.n_random)[inv]
            with np.errstate(all="ignore"):  # 1.x logs numpy warnings and carries on
                pre_values = bound.preprocessor.function(c, s, n, TableLookup(ctx), draws, k)
            for name in bound.preprocessor.targets:
                c[name] = _assign(c.get(name), pre_values[name])
        with np.errstate(all="ignore"):
            utils = ctx.engine.utilities(FormulaValues(seg["terms"], c, s, k), seg["coef"], n)
        if logsums_only:
            return ctx.engine.nested_logsums(utils, self.alternatives, seg["nests"])
        probs, bad, logsums = ctx.engine.nested_probabilities(utils, self.alternatives, seg["nests"])
        return utils, probs, bad, logsums, pre_values


def tour_mode_choice(configs_dirs: Sequence[str | Path],
                     settings_file: str = "tour_mode_choice.yaml") -> Component:
    return tour_mode_choice_component(configs_dirs, settings_file)


def atwork_subtour_mode_choice(configs_dirs: Sequence[str | Path],
                               settings_file: str = "tour_mode_choice.yaml") -> Component:
    """
    1.x's atwork_subtour_mode_choice: the tour mode choice model for the
    at-work subtours only, from the workplace (the person's
    ``workplace_zone_id``) to the subtour's destination, all with the
    ``atwork`` coefficients. As 1.x, the subtours don't get a
    ``tour_purpose`` column, and columns both the tours and persons_merged
    have (``household_id``) are not columns of the choosers (1.x's merge
    renames both).
    """
    return tour_mode_choice_component(configs_dirs, settings_file, step=ATWORK_STEP, category="atwork",
                                      orig=ATWORK_ORIG, purpose="atwork")


def tour_mode_choice_component(configs_dirs: Sequence[str | Path], settings_file: str = "tour_mode_choice.yaml",
                               step: str = STEP, category: str | None = None, orig: str = ORIG,
                               purpose: str | None = None) -> Component:
    """
    ``category``: only the tours of this category choose (others keep their
    mode); None: every tour. ``purpose``: every chooser's segment (none: each
    tour's purpose, as 1.x's tour_mode_choice_simulate works it out).
    """
    model = TourModeChoiceModel(configs_dirs, settings_file)
    alternatives = model.alternatives
    categories = sorted([""] + alternatives)  # as the tours table holds them after 1.x's step
    periods = model.periods
    added = set(ADDED) if purpose is None else {"out_period", "in_period"}
    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]) -> Needs:
        view_cols = VIEW.columns(available)
        if "pnr_zone_id" in view_cols:
            raise MissingInputError("park-and-ride (pnr_zone_id) is not supported yet")
        df_cols = view_cols
        if category is not None:
            # 1.x merges the subtours with persons_merged with pandas' default suffixes, so a
            # column both have is neither's
            tour_cols = set(available.get("tours", set()))
            merged = set().union(*(available.get(t, set()) for t, _ in VIEW.joins))
            df_cols = view_cols - (tour_cols & merged)
            for c in [model.choice_column] + ([model.logsum_column] if model.logsum_column else []):
                if c not in tour_cols and c == model.choice_column:
                    raise MissingInputError(f"tours lack {c} (the other tours' modes)")
        roles = None
        if model.language == "asim_core":
            from .roles import role_columns

            roles = role_columns(VIEW, TOUR_ROLES, available)
            if purpose is None:
                check_roles("tour mode choice: tour_purpose", [model.purpose[1]], roles)
        bound = model.bind(df_cols | added, available, DEST, orig=orig, roles=roles)
        columns = bound.columns - added - {"tour_id"}
        if purpose is None:
            columns |= model.chooser_inputs(view_cols)
        else:
            columns |= set(model.time_columns.values())
        needs = {t: set(c) for t, c in VIEW.resolve(columns, available).items()}
        if category is not None:
            needs["tours"] |= {"tour_category"} | {c for c in (model.choice_column, model.logsum_column)
                                                   if c and c in available.get("tours", set())}
        for t, c in bound.tables:
            needs.setdefault(t, set()).add(c)
        needs[SKIMS] = set(bound.skims)
        resolved["needs"] = {t: sorted(c) for t, c in needs.items()}
        resolved["bound"] = bound
        resolved["view_needs"] = VIEW.resolve(columns, available)
        return resolved["needs"]

    def run(ctx: StepContext):
        needs, bound = resolved["needs"], resolved["bound"]
        all_ids = ctx.ids("tours")
        if category is None:
            view_needs = {t: c for t, c in needs.items() if t != SKIMS}
            cols = {c: plain(v) for c, v in VIEW.gather(ctx, view_needs).items()}
            cols.update(model.chooser_columns(cols))
            table_rows = np.arange(len(all_ids))
        else:
            cat = np.asarray(plain(ctx.column("tours", "tour_category")))
            table_rows = np.flatnonzero(cat == category)   # 1.x's chooser order (table order)
            cols = {c: np.asarray(plain(v))[table_rows] for c, v in VIEW.gather(ctx, resolved["view_needs"]).items()}
            for k, c in model.time_columns.items():
                cols[k] = periods.labels_for(cols[c])
            cols["tour_purpose"] = np.full(len(table_rows), purpose, dtype=object)
        ids = all_ids[table_rows]

        choices = np.empty(len(ids), dtype=np.int64)
        logsums = np.empty(len(ids), dtype=np.float64)
        purposes = cols["tour_purpose"]
        traced = ctx.tracer.targets("tours", ids)
        for seg_purpose in sorted(set(purposes)):  # 1.x's groupby order
            rows = np.flatnonzero(purposes == seg_purpose)
            seg_ids = ids[rows]
            c = {k: v[rows] for k, v in cols.items() if not (purpose is not None and k == "tour_purpose")}
            utils, probs, bad, seg_logsums, pre_values = model.evaluate(
                ctx, bound, c, len(rows), seg_purpose, seg_ids)
            if bad.any():
                raise ValueError(f"{step}: {int(bad.sum())} '{seg_purpose}' tours have no usable "
                                 f"probabilities, e.g. tour IDs {seg_ids[bad][:5].tolist()}")
            rands = ctx.random.uniform(CHANNEL, seg_ids)
            seg_choices = ctx.engine.choose(probs, rands)
            choices[rows] = seg_choices
            logsums[rows] = seg_logsums
            local = np.flatnonzero(np.isin(rows, traced))
            if len(local):
                _trace(ctx, seg_purpose, seg_ids, local, pre_values, bound.preprocessor, utils, probs,
                       seg_logsums, rands, seg_choices, alternatives)

        chosen = np.asarray(alternatives, dtype=object)[choices]
        if category is None:
            out = {model.choice_column: pd.Categorical(chosen, categories=categories)}
            if model.logsum_column:
                out[model.logsum_column] = logsums
            return {"tours": out}
        # the other tours keep theirs; categories: both sets, sorted (1.x's assign_in_place)
        current = ctx.column("tours", model.choice_column)
        current = current if isinstance(current, pd.Categorical) else pd.Categorical(np.asarray(current, dtype=object))
        cats = sorted(set(current.categories) | set(chosen))
        values = np.asarray(current.astype(object), dtype=object).copy()
        values[table_rows] = chosen
        out = {model.choice_column: pd.Categorical(values, categories=cats)}
        if model.logsum_column:
            if model.logsum_column in needs["tours"]:
                lv = np.asarray(ctx.column("tours", model.logsum_column), dtype=np.float64).copy()
            else:
                lv = np.full(len(all_ids), np.nan)
            lv[table_rows] = logsums
            out[model.logsum_column] = lv
        return {"tours": out}

    return Component(
        name=step,
        run=run,
        provides={"tours": [model.choice_column] + ([model.logsum_column] if model.logsum_column else [])},
        resolve_needs=resolve_needs,
        description=(f"tour mode choice: nested logit over {len(alternatives)} modes, "
                     + (f"{len(model.segments)} purpose segments" if purpose is None
                        else f"{category} tours, '{purpose}' coefficients, from {orig}")),
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
        # positions shared by every skim looked up for the same zone pairs (and periods)
        self._pairs: dict = {}
        labels = periods.unique_labels
        self.groups = {}
        for col, values in period_labels.items():
            values = np.asarray(values, dtype=object)
            self.groups[col] = [(lab, np.flatnonzero(values == lab)) for lab in labels]
            covered = sum(len(r) for _, r in self.groups[col])
            if covered != len(values):
                raise ValueError(f"{col}: labels not in the skim time periods {labels}")
        self.n = len(orig)

    def pairs(self, o_role: str, d_role: str):
        if (o_role, d_role) not in self._pairs:
            self._pairs[(o_role, d_role)] = self.ctx.zone_pairs(self.zones[o_role], self.zones[d_role])
        return self._pairs[(o_role, d_role)]

    def __call__(self, accessor: str, key: str, how: str = "get"):
        o_role, d_role, period_col = SKIM_WRAPPERS[accessor]
        pairs = self.pairs(o_role, d_role)
        if period_col is None:
            if how == "reverse":
                return self.ctx.skim(key, pairs.reversed())
            if how == "max":
                return np.maximum(self.ctx.skim(key, pairs.reversed()), self.ctx.skim(key, pairs))
            return self.ctx.skim(key, pairs)
        out = None
        for label, rows in self.groups[period_col]:
            if not len(rows):
                continue
            values = self.ctx.skim(f"{key}__{label}", pairs.subset((period_col, label), rows))
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


# --- converting today's files ------------------------------------------------------------

LEGACY_SKIMS = {acc: (("o" if a == "orig" else "d"), ("o" if b == "orig" else "d"), p)
                for acc, (a, b, p) in SKIM_WRAPPERS.items()}


def legacy_parameters(configs_dirs, settings_file: str = "tour_mode_choice.yaml") -> dict[str, str]:
    """The parameters converting today's preprocessor creates: has_<column> -> column."""
    from ..spec.convert import LegacyScope, NewNames, convert_formula

    model = TourModeChoiceModel(configs_dirs, settings_file)
    params: dict[str, str] = {}
    if model.preprocessor is None:
        return params
    names = NewNames(role_of=lambda c: "tour")
    assigned: set[str] = set()
    for a in model.preprocessor.assignments:
        legacy = LegacyScope(constants=model.preprocessor.constants, python_mode=True, assigned=set(assigned),
                             tables=set(model.preprocessor_tables), skims=LEGACY_SKIMS,
                             functions=set(model.preprocessor.functions),
                             coefficients=set(model.definition.coefficient_names))
        try:
            params.update(convert_formula(a.formula, legacy, names).parameters)
        except Exception:  # noqa: BLE001 - only the column tests matter here
            pass
        assigned.add(a.target)
    return params


def convert(configs_dirs: Sequence[str | Path], contexts: list[dict], out_dir: str | Path,
            settings_file: str = "tour_mode_choice.yaml") -> list[Path]:
    """
    asim_core's model files for today's tour mode choice files. ``contexts``
    are the places the model runs (this step first, then steps using it for
    logsums): each with ``role_of`` (column -> table name) and
    ``df_columns`` (the chooser columns 1.x's tests of which columns exist see).
    """
    from ..spec.convert import LegacyScope, NewNames, convert_formula
    from ..spec.model_file import write_coefficients, write_derived, write_terms, write_yaml

    configs_dirs = [Path(d) for d in configs_dirs]
    out_dir = Path(out_dir)
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    model = TourModeChoiceModel(configs_dirs, settings_file)
    d = model.definition

    def role_of(col):
        for ctx in contexts:
            r = ctx["role_of"](col)
            if r is not None:
                return r
        # not in any step that uses the model (e.g. in a branch none of them
        # takes): the chooser rows are tours. Checked when a step uses it.
        return "tour"

    params: dict[str, str] = {}
    derived_rows, targets = [], []
    if model.preprocessor is not None:
        names = NewNames(role_of=role_of, derived=set(ADDED))
        assigned: set[str] = set()
        for a in model.preprocessor.assignments:
            legacy = LegacyScope(constants=model.preprocessor.constants, python_mode=True,
                                 assigned=set(assigned), tables=set(model.preprocessor_tables),
                                 skims=LEGACY_SKIMS, functions=set(model.preprocessor.functions),
                                 coefficients=set(d.coefficient_names),
                                 table_columns=contexts[0].get("tables", {}))
            conv = convert_formula(a.formula, legacy, names)
            params.update(conv.parameters)
            derived_rows.append((a.description, a.target, conv.text))
            assigned.add(a.target)
        targets = model.preprocessor.targets

    names = NewNames(role_of=role_of, derived=set(ADDED) | set(targets))
    legacy = LegacyScope(constants=d.constants, skims=LEGACY_SKIMS, coefficients=set(d.coefficient_names))
    spec_file = str(settings["SPEC"])
    spec = pd.read_csv(find_config_file(configs_dirs, spec_file if spec_file.endswith(".csv") else spec_file + ".csv"),
                       comment="#").dropna(subset=["Expression"])
    descriptions = (["" if pd.isna(x) else str(x).strip() for x in spec["Description"]]
                    if "Description" in spec.columns else [""] * len(spec))
    rows = []
    for i, term in enumerate(d.terms):
        conv = convert_formula(term.formula, legacy, names)
        if conv.parameters:
            raise NotImplementedError(f"term {term.label}: tests of which columns exist in a term")
        rows.append({"label": term.label, "description": descriptions[i], "formula": conv.text,
                     "cells": term.coefficients})

    stem = Path(settings_file).stem
    paths = {k: out_dir / f"{stem}_{k}.csv" for k in ("terms", "coefficients", "derived")}
    write_terms(paths["terms"], rows, d.alternatives)
    write_coefficients(paths["coefficients"], by_segment={seg: d.coefficient_values(seg) for seg in d.segments})
    if derived_rows:
        write_derived(paths["derived"], derived_rows)

    own = contexts[0]["df_columns"]
    has_university = contexts[0]["role_of"]("is_university") is not None
    data = {
        "kind": "tour_mode_choice",
        "description": "A mode for every tour, and the tour's mode choice logsum (converted from "
                       f"{settings_file}, {', '.join(d.source_files[1:])}"
                       f"{', ' + model.preprocessor.source_files[0] if model.preprocessor else ''}).",
        "alternatives": d.alternatives,
        "nests": d.nests.model_dump(),
        "terms": paths["terms"].name,
        "coefficients": paths["coefficients"].name,
    }
    if derived_rows:
        data["derived"] = paths["derived"].name
        data["derived_tables"] = list(model.preprocessor_tables)
    data["formulas_see_coefficients"] = True
    if params:
        data["parameters"] = {p: col in own for p, col in sorted(params.items())}
    data["tour_purpose"] = PURPOSE_1X if has_university else "tour.tour_type"
    data["time_periods"] = {"out_period": f"tour.{OUT_TIME}", "in_period": f"tour.{IN_TIME}"}
    data["outputs"] = {"choice": "tour_mode", "logsum": model.logsum_column}
    if settings.get("CONSTANTS"):
        data["constants"] = settings["CONSTANTS"]
    data["options"] = {"drop_zero_terms": True, "random_draws": "chooser_stream"}
    yaml_path = out_dir / Path(settings_file).name
    write_yaml(yaml_path, data, "# asim_core model file (formulas in asim_core's language; see docs/model_files.md)\n"
                                "# parameters: 1.x tested which columns the choosers have; each step using\n"
                                "# this model now says so (these values are for this model's own step).")
    return [p for p in [yaml_path] + list(paths.values()) if p.exists()]
