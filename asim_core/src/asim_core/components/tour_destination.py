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

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..component import SKIMS, Component, MissingInputError, Needs, StepContext
from ..logit import FormulaValues
from ..spec.definition import InteractionModelDefinition, InteractionTerm, LanguageScope
from ..spec.model_file import (
    ModelFile,
    is_model_file,
    locate,
    read_coefficients_table,
    read_model_yaml,
    read_terms,
    validate,
)
from .roles import TOUR_ROLES, PERSON_ROLES, check_roles, context_roles, role_columns
from ..spec.legacy import (
    find_config_file,
    load_global_constants,
    load_interaction_model,
    load_top_settings,
    read_settings,
    read_yaml,
)
from .runtime import plain
from .segments import segment_problem
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
    return tour_destination_component(configs_dirs, settings_file, STEP, CATEGORY, max_rows)


def joint_tour_destination(
    configs_dirs: Sequence[str | Path],
    settings_file: str = "joint_tour_destination.yaml",
    max_rows: int = 1_000_000,
) -> Component:
    """1.x's joint_tour_destination: the same model as non-mandatory destination
    (run_tour_destination), for the joint tours, from their point person."""
    return tour_destination_component(configs_dirs, settings_file, "joint_tour_destination", "joint", max_rows)


def atwork_subtour_destination(
    configs_dirs: Sequence[str | Path],
    settings_file: str = "atwork_subtour_destination.yaml",
    max_rows: int = 1_000_000,
) -> Component:
    """1.x's atwork_subtour_destination: the same model for the at-work subtours, from the
    workplace, with 1.x's defaults for settings a region may leave out ("future settings")."""
    return tour_destination_component(
        configs_dirs, settings_file, "atwork_subtour_destination", "atwork", max_rows,
        defaults={"SIZE_TERM_SELECTOR": "atwork", "SEGMENTS": ["atwork"], "ORIG_ZONE_ID": "workplace_zone_id"})


def tour_destination_component(
    configs_dirs: Sequence[str | Path],
    settings_file: str,
    step: str,
    category: str,
    max_rows: int = 1_000_000,
    defaults: dict | None = None,
) -> Component:
    """A destination for each tour of ``category`` (1.x's run_tour_destination). ``defaults``:
    values for settings the file leaves out (1.x's future_component_settings)."""
    STEP, CATEGORY = step, category  # noqa: N806 (the module's names, for this step)
    configs_dirs = [Path(d) for d in configs_dirs]
    path = find_config_file(configs_dirs, settings_file)
    cfg = _load_model_file(path, configs_dirs) if is_model_file(path) else \
        _load_legacy(configs_dirs, settings_file, step, defaults)
    segments, segment_col, orig_col, alt_col = cfg.segments, cfg.segment_col, cfg.orig_col, cfg.alt_col
    chooser_id, sample_size, logsum_column = cfg.chooser_id, cfg.sample_size, cfg.logsum_column
    size_weights, size_columns, shift = cfg.size_weights, cfg.size_columns, cfg.shift
    sample_defs, choice_defs, sample_fns, choice_fns = cfg.sample_defs, cfg.choice_defs, cfg.sample_fns, cfg.choice_fns
    model, periods_for = cfg.model, cfg.periods_for
    new_format = cfg.language == "asim_core"
    max_rows = cfg.max_rows or max_rows

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
        if new_format:
            for d in sample_defs.values():
                check_roles(f"{STEP} sample terms", d.translated,
                            role_columns(TOURS_VIEW, TOUR_ROLES, available, extra={"alt": {zone_key}}))
            for d in choice_defs.values():
                check_roles(f"{STEP} terms", d.translated,
                            role_columns(TOURS_VIEW, TOUR_ROLES, available, extra={"alt": {alt_col}}))
        tours_view = TOURS_VIEW.resolve(columns, available)
        bound = {}
        persons_columns: set[str] = set()
        if model is not None:
            df_cols = sample_cols | ls_added | PERSONS_VIEW.columns(available)
            for seg in segments:
                # logsums at the segment's fixed periods: only those periods' skims are read
                labels = model.fixed_period_labels(*periods_for(seg))
                if new_format:
                    roles = context_roles(PERSONS_VIEW, PERSON_ROLES, available, df_cols)
                    b = model.bind(df_cols, available, dest=alt_col, orig=orig_col,
                                   parameters=cfg.logsum_parameters, roles=roles, period_labels=labels)
                else:
                    b = model.bind(df_cols, available, dest=alt_col, orig=orig_col, period_labels=labels)
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
        problem = segment_problem("tours", segment_col, seg_values, segments) if segment_col else None
        if problem:
            ctx.warn(problem)

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
                             sample_fns[seg], sample_size, shift, max_rows, step=STEP)
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
                                              choice_fns[seg], shift, max_rows, alt_col, chooser_id, step=STEP)
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
        self.ctx = ctx
        self.pairs = ctx.zone_pairs(orig_pos, dest_pos)  # shared by every skim looked up

    def __call__(self, accessor, key, how="get"):
        if how == "reverse":
            return self.ctx.skim(key, self.pairs.reversed())
        if how == "max":
            return np.maximum(self.ctx.skim(key, self.pairs.reversed()), self.ctx.skim(key, self.pairs))
        return self.ctx.skim(key, self.pairs)


@dataclass
class _Config:
    """What the step runs, from either today's files or a model file."""

    segments: list
    segment_col: str | None
    orig_col: str
    alt_col: str
    chooser_id: str
    sample_size: int
    logsum_column: str | None
    size_weights: dict
    size_columns: list
    shift: bool
    sample_defs: dict
    choice_defs: dict
    sample_fns: dict
    choice_fns: dict
    model: object
    periods_for: object
    language: str = "asim1"
    logsum_parameters: dict | None = None
    max_rows: int | None = None


def _load_legacy(configs_dirs, settings_file, step=STEP, defaults=None) -> _Config:
    """Today's settings, specs and coefficients."""
    settings = dict(read_settings(configs_dirs, settings_file))
    for key, value in (defaults or {}).items():
        if settings.get(key) is None:
            settings[key] = value
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
                                      constants, name=f"{step}_{spec_key.lower()}_{seg}",
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
    return _Config(segments, segment_col, orig_col, alt_col, chooser_id, sample_size, logsum_column,
                   size_weights, size_columns, shift, sample_defs, choice_defs, sample_fns, choice_fns, model,
                   periods_for)


def _size_weights(path, selector, segments):
    st = pd.read_csv(path, comment="#", index_col="segment")
    st = st[st.model_selector == selector].drop(columns="model_selector")
    missing_segments = [s for s in segments if s not in st.index]
    if missing_segments:
        raise ValueError(f"no size terms for segments {missing_segments}")
    return {seg: st.loc[seg] for seg in segments}, sorted(st.columns)


# --- model files ------------------------------------------------------------------------

SAMPLE_VALUES = ["size_term"]
CHOICE_VALUES = ["size_term", "prob", "pick_count", "mode_choice_logsum"]


class SizeTerms(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str = "destination_choice_size_terms.csv"
    selector: str  # the rows of the file (model_selector) for this model


class Logsums(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str  # a tour mode choice model file
    # the time periods (departure, return) logsums are worked out for: one pair, or one per segment
    periods: tuple[int, int] | dict[str, tuple[int, int]]
    parameters: dict[str, Any] = {}


class Outputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination: str = "destination"
    logsum: str | None = None


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utility_shift: bool = False  # 1.x: only when skip_failed_choices is off
    max_rows: int | None = None  # tour x zone rows evaluated at once (results don't depend on it)
    # how 1.x samples: draws by inverse CDF; at most as many draws as zones with a size;
    # the sample sorted by zone within each tour; its probabilities kept as 32-bit floats
    sample_method: Literal["inverse_cdf"] = "inverse_cdf"
    sample_size_cap: Literal["zones_with_size"] = "zones_with_size"
    sample_order: Literal["by_zone"] = "by_zone"
    sample_probabilities: Literal["float32"] = "float32"


class TourDestinationFile(ModelFile):
    kind: Literal["tour_destination"]
    tours: Literal["non_mandatory"]
    segment: str | None = None  # the tours' column picking the segment, e.g. tour.tour_type
    segments: list[str]
    size_terms: SizeTerms
    sample_size: int
    sample_terms: str  # one coefficient column per segment
    terms: str         # one coefficient column per segment
    coefficients: str
    origin: str        # the column holding each tour's origin zone, e.g. household.home_zone_id
    alternative_zone: str = "alt_dest"  # name of the candidate zone in the choice rows (alt.<name>)
    chooser_id: str = "person_id"
    logsums: Logsums | None = None
    outputs: Outputs = Outputs()
    options: Options = Options()


def _column_name(text: str, path) -> str:
    import ast as _ast

    node = _ast.parse(text, mode="eval").body
    if isinstance(node, _ast.Attribute) and isinstance(node.value, _ast.Name):
        return node.attr
    raise ValueError(f"{path}: '{text}' must be a column, e.g. household.home_zone_id")


def _load_model_file(path, configs_dirs) -> _Config:
    f = validate(TourDestinationFile, read_model_yaml(path), path)
    constants = dict(load_global_constants(configs_dirs))
    constants.update(f.constants)
    coefficients, by_seg = read_coefficients_table(f.coefficients, f.coefficient_values)
    if by_seg is not None:
        raise ValueError(f"{f.coefficients}: one value per coefficient (segments are columns of the term tables)")
    roles = list(TOUR_ROLES) + ["alt"]
    skims = {"o d": SKIM_ACCESSOR}

    def load(terms_path, values, kind):
        rows, value_cols = read_terms(terms_path)
        missing = [seg for seg in f.segments if seg not in value_cols]
        if missing:
            raise ValueError(f"{terms_path}: no coefficient column for segments {missing}")
        scope = LanguageScope(roles=roles, derived=values, skims=skims)
        out = {}
        for seg in f.segments:
            terms = [InteractionTerm(label=label, formula=formula, coefficient=cells.get(seg, "0"), target=target)
                     for label, _, formula, target, cells in rows]
            out[seg] = InteractionModelDefinition(
                name=f"{STEP}_{kind}_{seg}", terms=terms, coefficients=coefficients, constants=constants,
                language="asim_core", scope=scope,
                source_files=[Path(terms_path).name, Path(f.coefficients).name])
        return out

    sample_defs = load(f.sample_terms, SAMPLE_VALUES, "sample")
    choice_defs = load(f.terms, CHOICE_VALUES, "spec")
    size_weights, size_columns = _size_weights(locate(f.size_terms.file, configs_dirs), f.size_terms.selector,
                                               f.segments)
    model = None
    periods = None
    if f.logsums is not None:
        model_path = Path(locate(f.logsums.model, configs_dirs))
        model = TourModeChoiceModel([model_path.parent] + list(configs_dirs), model_path.name)
        periods = f.logsums.periods

    def periods_for(seg):
        p = periods[seg] if isinstance(periods, dict) else periods
        return int(p[0]), int(p[1])

    return _Config(
        list(f.segments), _column_name(f.segment, path) if f.segment else None, _column_name(f.origin, path),
        f.alternative_zone, f.chooser_id, f.sample_size, f.outputs.logsum, size_weights, size_columns,
        f.options.utility_shift, sample_defs, choice_defs,
        {seg: d.utility_terms() for seg, d in sample_defs.items()},
        {seg: d.utility_terms() for seg, d in choice_defs.items()},
        model, periods_for, language="asim_core",
        logsum_parameters=dict(f.logsums.parameters) if f.logsums else None, max_rows=f.options.max_rows)


def _sample(ctx, sel, tour_ids, cols, orig_pos, alt_zones, alt_size, terms, sample_size, shift, max_rows,
            channel=CHANNEL, step=STEP, alt_values=None):
    """
    Sampled zones per chooser (tour, or person in location choice): rows
    (chooser position, zone, prob as float32, pick_count). ``alt_values``:
    more columns of the alternatives, the same for every zone (location
    choice's shadow price adjustments).
    """
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
        for name, value in (alt_values or {}).items():
            c[name] = np.full(n_b * n_alts, value)
        for name, values in cols.items():
            if name not in c:
                c[name] = values[rows]
        s = _ZoneSkims(ctx, orig_pos[rows], np.tile(dest_pos_alts, n_b))
        s = ctx.engine.skim_reuse(s, [getattr(terms_fn, "_asim_source", None)])
        n = n_b * n_alts
        with np.errstate(all="ignore"):
            utils = ctx.engine.interaction_utilities(FormulaValues(terms_fn, c, n, None, s), coefs, n)
        probs, bad = ctx.engine.probabilities(utils.reshape(n_b, n_alts), shift=shift)
        if bad.any():
            raise ValueError(f"{step}: choosers with no usable sampling probabilities: "
                             f"{tour_ids[batch][bad][:5].tolist()}")
        rands = ctx.random.uniform(channel, tour_ids[batch], sample_size)
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


def _logsums(ctx, model, bound, persons_view, seg, periods, sample, tour_ids, cols, alt_col, chooser_id,
             channel=CHANNEL):
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
                                           tour_ids[tours_of_rows[start:end]], logsums_only=True,
                                           channel=channel)
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


def _choose(ctx, sel, tour_ids, cols, orig_pos, sample, zone_size, terms, shift, max_rows, alt_col, chooser_id,
            channel=CHANNEL, step=STEP, alt_values=None):
    """Choose one sampled zone per chooser; returns (zones, logsums) for the choosers in ``sel``."""
    terms_fn, coefs, _ = terms
    tour_of_row = sample["tour"]
    # rows of each tour are contiguous and tours are in ``sel`` order
    bounds = np.searchsorted(tour_of_row, sel, side="left"), np.searchsorted(tour_of_row, sel, side="right")
    position_of = {t: i for i, t in enumerate(sel)}
    if not np.array_equal(np.unique(tour_of_row), np.sort(sel)):
        raise ValueError(f"{step}: sample rows don't cover every chooser")
    chosen = np.empty(len(sel), dtype=np.float64)
    logsums = np.empty(len(sel), dtype=np.float64)
    sample_dest_pos = ctx.skim_positions(sample["zone"])
    size_of = zone_size.reindex(sample["zone"]).to_numpy()
    width = int((bounds[1] - bounds[0]).max())
    per_batch = max(1, max_rows // max(1, width))
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
        for name, value in (alt_values or {}).items():
            c[name] = np.full(n, value)
        for name, values in cols.items():
            if name not in c:
                c[name] = values[tours_r]
        s = _ZoneSkims(ctx, orig_pos[tours_r], sample_dest_pos[rows])
        s = ctx.engine.skim_reuse(s, [getattr(terms_fn, "_asim_source", None)])
        with np.errstate(all="ignore"):
            utils = ctx.engine.interaction_utilities(FormulaValues(terms_fn, c, n, None, s), coefs, n)
        local = np.array([position_of[t] for t in tours_r]) - b
        counts = np.bincount(local, minlength=len(batch_tours))
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        # padded to the widest chooser of the whole segment, as 1.x pads its
        # (unchunked) call: the width changes how NumPy adds each row up, so
        # padding per batch would change the last bits of logsums
        padded = np.full((len(batch_tours), width), PAD_UTILITY)
        padded[local, np.arange(n) - offsets[local]] = utils
        probs, bad, ls = ctx.engine.probabilities(padded, shift=shift, with_logsums=True)
        if bad.any():
            raise ValueError(f"{step}: choosers with no usable probabilities: "
                             f"{tour_ids[batch_tours][bad][:5].tolist()}")
        rands = ctx.random.uniform(channel, tour_ids[batch_tours])
        positions = ctx.engine.choose(probs, rands)
        chosen[b:b + len(batch_tours)] = sample["zone"][rows][offsets + positions]
        logsums[b:b + len(batch_tours)] = ls
    return chosen, logsums


# --- converting today's files ------------------------------------------------------------


def convert(configs_dirs: Sequence[str | Path], available: Mapping[str, set[str]], out_dir: str | Path,
            settings_file: str = "non_mandatory_tour_destination.yaml") -> list[Path]:
    """asim_core's model files for today's destination files; ``available``: columns when the step runs."""
    from ..spec.convert import LegacyScope, NewNames, convert_formula
    from ..spec.model_file import write_coefficients, write_terms, write_yaml
    from .roles import role_finder
    from .tour_mode_choice import legacy_parameters

    configs_dirs = [Path(d) for d in configs_dirs]
    out_dir = Path(out_dir)
    settings = read_settings(configs_dirs, settings_file)
    cfg = _load_legacy(configs_dirs, settings_file)
    comp = non_mandatory_tour_destination(configs_dirs, settings_file)
    if cfg.model is not None:
        comp.info["logsum_model"].bind_log.clear()
    comp.resolve_needs(available)
    written = []

    def convert_spec(defs, spec_key, alt_cols, values, name):
        first = defs[cfg.segments[0]]
        names = NewNames(role_of=role_finder(role_columns(TOURS_VIEW, TOUR_ROLES, available,
                                                          extra={"alt": set(alt_cols)})),
                         derived=set(values))
        spec = pd.read_csv(find_config_file(configs_dirs, settings[spec_key]), comment="#").dropna(
            subset=["Expression"])
        descriptions = ["" if pd.isna(x) else str(x).strip() for x in spec.get("Description", [""] * len(spec))]
        rows, assigned = [], set()
        for i, t in enumerate(first.terms):
            legacy = LegacyScope(constants=first.constants, pandas_locals=True, assigned=set(assigned),
                                 python_mode=True if t.target else None, skims={SKIM_ACCESSOR: ("o", "d", None)})
            cells = {seg: defs[seg].terms[i].coefficient for seg in cfg.segments}
            rows.append({"label": t.label, "description": descriptions[i] if i < len(descriptions) else "",
                         "target": t.target, "formula": convert_formula(t.formula, legacy, names).text,
                         "cells": cells})
            if t.target:
                assigned.add(t.target)
        path = out_dir / f"{name}_terms.csv"
        write_terms(path, rows, list(cfg.segments))
        written.append(path)
        return path

    sample_path = convert_spec(cfg.sample_defs, "SAMPLE_SPEC", ["zone_id"], SAMPLE_VALUES,
                               Path(settings["SAMPLE_SPEC"]).stem)
    terms_path = convert_spec(cfg.choice_defs, "SPEC", [cfg.alt_col], CHOICE_VALUES, Path(settings["SPEC"]).stem)
    coef_path = out_dir / f"{Path(settings['COEFFICIENTS']).stem}.csv"
    first = cfg.choice_defs[cfg.segments[0]]
    write_coefficients(coef_path, {n: c.value for n, c in first.coefficients.items()})
    written.append(coef_path)

    find_role = role_finder(role_columns(TOURS_VIEW, TOUR_ROLES, available))
    data = {"kind": "tour_destination",
            "description": f"A destination for every non-mandatory tour (converted from {settings_file} "
                           "and the files it names).",
            "tours": "non_mandatory"}
    if cfg.segment_col:
        data["segment"] = f"{find_role(cfg.segment_col) or 'tour'}.{cfg.segment_col}"
    data.update({
        "segments": list(cfg.segments),
        "size_terms": {"file": "destination_choice_size_terms.csv", "selector": settings["SIZE_TERM_SELECTOR"]},
        "sample_size": cfg.sample_size,
        "sample_terms": sample_path.name,
        "terms": terms_path.name,
        "coefficients": coef_path.name,
        "origin": f"{find_role(cfg.orig_col) or 'tour'}.{cfg.orig_col}",
        "alternative_zone": cfg.alt_col,
        "chooser_id": cfg.chooser_id,
    })
    if cfg.model is not None:
        model = comp.info["logsum_model"]
        params = legacy_parameters(configs_dirs, str(settings["LOGSUM_SETTINGS"]))
        df_cols = model.bind_log[0][0]
        pairs = {seg: list(cfg.periods_for(seg)) for seg in cfg.segments}
        periods = next(iter(pairs.values())) if len({tuple(p) for p in pairs.values()}) == 1 else pairs
        data["logsums"] = {"model": str(settings["LOGSUM_SETTINGS"]), "periods": periods}
        if params:
            data["logsums"]["parameters"] = {p: col in df_cols for p, col in sorted(params.items())}
    data["outputs"] = {"destination": "destination", "logsum": cfg.logsum_column}
    if settings.get("CONSTANTS"):
        data["constants"] = settings["CONSTANTS"]
    data["options"] = {"utility_shift": cfg.shift, "max_rows": None, "sample_method": "inverse_cdf",
                       "sample_size_cap": "zones_with_size", "sample_order": "by_zone",
                       "sample_probabilities": "float32"}
    yaml_path = out_dir / Path(settings_file).name
    write_yaml(yaml_path, data, "# asim_core model file (formulas in asim_core's language; see docs/model_files.md)\n"
                                "# alt.<column>: the candidate zone; size_term, prob, pick_count and\n"
                                "# mode_choice_logsum: worked out by the step")
    return [yaml_path] + written
