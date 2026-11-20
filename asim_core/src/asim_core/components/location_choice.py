"""
School and workplace location: a usual school zone for every student and a
usual work zone for every worker.

Reads today's school_location.yaml / workplace_location.yaml, their sample
spec, spec, coefficients, destination_choice_size_terms.csv and annotation
files unchanged, and follows ActivitySim 1.x (activitysim/abm/models/
location_choice.py), without shadow pricing (none of the examples turn it
on). Choosers are persons with their household, home zone and
accessibility (1.x's ``persons_merged``), filtered by
``CHOOSER_FILTER_COLUMN_NAME`` (e.g. ``is_worker``) and segmented by
``CHOOSER_SEGMENT_COLUMN_NAME`` (e.g. income segment). For each segment, in
the order of ``SEGMENT_IDS``, as non-mandatory tour destination does:

1. **Size terms**: each zone's size (land use columns times weights, for
   the segment); zones with size zero can't be chosen.
2. **Sample** zones per person (swappable piece: ``sample``).
3. **Logsums** from tour mode choice for each sampled zone, at the fixed
   periods ``OUT_PERIOD`` / ``IN_PERIOD``, with the segment's tour purpose
   (``LOGSUM_TOUR_PURPOSE``).
4. **Choice** of one sampled zone; its logsum over the sample is kept.

Then, as 1.x: the chosen zone goes to persons (``-1`` for everyone else),
the logsum too (missing for everyone else); the number of persons choosing
each zone, per segment, becomes a table (``MODELED_SIZE_TABLE``); and the
annotations run (``annotate_persons``, then ``annotate_households``).

Random numbers come from each person's stream: the sample's draws, the
mode choice preprocessor's (once per person), then the choice.

The alternatives carry 1.x's shadow price columns at their values without
shadow pricing: ``shadow_price_size_term_adjustment`` 1 and
``shadow_price_utility_adjustment`` 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
from ..store import Table
from . import annotate
from .runtime import plain
from .segments import check_segment_values
from .tour_destination import (
    PERSONS_VIEW,
    SKIM_ACCESSOR,
    _choose,
    _logsums,
    _sample,
    _size_terms,
)
from .tour_mode_choice import TourModeChoiceModel

CHANNEL = "persons"
NO_ZONE = -1  # 1.x: persons without a location
ALT_VALUES = {"shadow_price_size_term_adjustment": 1, "shadow_price_utility_adjustment": 0}
ZONE_KEY = "zone_id"  # 1.x's name for the sampled zone (land_use's index)
MODE_CHOICE_LOGSUM = "mode_choice_logsum"

_UNSUPPORTED = {
    "preprocessor": "preprocessors",
    "alts_preprocessor_sample": "preprocessors",
    "alts_preprocessor_simulate": "preprocessors",
    "annotate_tours": "annotating tours",
    "annotate_trips": "annotating trips",
}


@dataclass
class _Config:
    """What the step runs, from either today's files or a model file."""

    segments: dict                 # segment name -> segment id
    segment_col: str
    filter_col: str
    orig_col: str
    alt_col: str
    sample_size: int
    size_selector: str
    sample_defs: dict
    choice_defs: dict
    sample_fns: dict
    choice_fns: dict
    model: object                  # TourModeChoiceModel or None
    logsum_purpose: object         # segment -> tour purpose
    periods_for: object            # tour purpose -> (out, in)
    dest_column: str
    logsum_column: str | None
    mode_logsum_column: str | None
    modeled_size_table: str | None
    annotations: list = field(default_factory=list)
    shift: bool = False
    language: str = "asim1"
    logsum_parameters: dict | None = None
    max_rows: int | None = None
    size_terms_file: str = "destination_choice_size_terms.csv"


def _load_legacy(configs_dirs, settings_file, step) -> _Config:
    """Today's settings, specs, coefficients and annotations."""
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    unsupported = [f"{k} ({why})" for k, why in _UNSUPPORTED.items() if settings.get(k)]
    if top.get("use_shadow_pricing"):
        unsupported.append("shadow pricing (use_shadow_pricing)")
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

    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    segments = dict(settings["SEGMENT_IDS"])

    def load(spec_key, seg):
        return load_interaction_model(configs_dirs, settings[spec_key], settings["COEFFICIENTS"], constants,
                                      name=f"{step}_{spec_key.lower()}_{seg}", value_column=seg,
                                      skim_accessors=[SKIM_ACCESSOR])

    sample_defs = {seg: load("SAMPLE_SPEC", seg) for seg in segments}
    choice_defs = {seg: load("SPEC", seg) for seg in segments}

    model = None
    if settings.get("LOGSUM_SETTINGS"):
        model = TourModeChoiceModel(configs_dirs, str(settings["LOGSUM_SETTINGS"]))
        pre = settings.get("LOGSUM_PREPROCESSOR", "preprocessor")
        if pre != "preprocessor":
            ls_settings = read_yaml(find_config_file(configs_dirs, str(settings["LOGSUM_SETTINGS"])))
            if (ls_settings.get(pre) or {}).get("SPEC") != (ls_settings.get("preprocessor") or {}).get("SPEC"):
                raise NotImplementedError(f"{settings_file}: LOGSUM_PREPROCESSOR {pre} with its own spec")
            # same spec as 'preprocessor'; only its TABLES differ, which the
            # model looks up only in branches these choosers don't take

    purpose = settings.get("LOGSUM_TOUR_PURPOSE")

    def periods_for(tour_purpose):
        out_p, in_p = settings.get("OUT_PERIOD"), settings.get("IN_PERIOD")
        if isinstance(out_p, dict):  # 1.x looks these up by tour purpose
            out_p, in_p = out_p[tour_purpose], in_p[tour_purpose]
        return int(out_p), int(in_p)

    return _Config(
        segments=segments, segment_col=settings["CHOOSER_SEGMENT_COLUMN_NAME"],
        filter_col=settings["CHOOSER_FILTER_COLUMN_NAME"], orig_col=settings["CHOOSER_ORIG_COL_NAME"],
        alt_col=settings["ALT_DEST_COL_NAME"], sample_size=int(settings["SAMPLE_SIZE"]),
        size_selector=settings["MODEL_SELECTOR"], sample_defs=sample_defs, choice_defs=choice_defs,
        sample_fns={seg: d.utility_terms() for seg, d in sample_defs.items()},
        choice_fns={seg: d.utility_terms() for seg, d in choice_defs.items()},
        model=model, logsum_purpose=(lambda seg: purpose[seg] if isinstance(purpose, dict) else purpose),
        periods_for=periods_for, dest_column=settings["DEST_CHOICE_COLUMN_NAME"],
        logsum_column=settings.get("DEST_CHOICE_LOGSUM_COLUMN_NAME"),
        mode_logsum_column=settings.get("MODE_CHOICE_LOGSUM_COLUMN_NAME"),
        modeled_size_table=settings.get("MODELED_SIZE_TABLE"),
        annotations=annotate.legacy_annotations(configs_dirs, settings, top, step),
        shift=not top.get("skip_failed_choices", True),  # as 1.x's utils_to_probs
    )


def school_location(configs_dirs: Sequence[str | Path], settings_file: str = "school_location.yaml",
                    max_rows: int = 1_000_000) -> Component:
    return location_choice_component(configs_dirs, "school_location", settings_file, max_rows)


def workplace_location(configs_dirs: Sequence[str | Path], settings_file: str = "workplace_location.yaml",
                       max_rows: int = 1_000_000) -> Component:
    return location_choice_component(configs_dirs, "workplace_location", settings_file, max_rows)


def location_choice_component(configs_dirs: Sequence[str | Path], step: str, settings_file: str,
                              max_rows: int = 1_000_000) -> Component:
    """``step``: the 1.x step name (random streams depend on it)."""
    configs_dirs = [Path(d) for d in configs_dirs]
    path = find_config_file(configs_dirs, settings_file)
    from ..spec.model_file import is_model_file

    if is_model_file(path):
        from .location_choice_files import load_model_file

        cfg = load_model_file(path, configs_dirs, step)
    else:
        cfg = _load_legacy(configs_dirs, settings_file, step)
    max_rows = cfg.max_rows or max_rows
    new_format = cfg.language == "asim_core"

    size_path = find_config_file(configs_dirs, cfg.size_terms_file)
    st = pd.read_csv(size_path, comment="#", index_col="segment")
    st = st[st.model_selector == cfg.size_selector].drop(columns="model_selector")
    missing = [s for s in cfg.segments if s not in st.index]
    if missing:
        raise ValueError(f"no size terms for segments {missing}")
    size_weights = {seg: st.loc[seg] for seg in st.index}  # all of the selector's segments, file order
    size_columns = sorted(st.columns)

    alt_values = set(ALT_VALUES)
    sample_cols = {cfg.alt_col, "prob", "pick_count", "person_id"}
    ls_added = {"in_period", "out_period", "duration"}
    made = {"persons": {cfg.dest_column} | {c for c in (cfg.logsum_column, cfg.mode_logsum_column) if c}}
    resolved: dict = {}

    def resolve_needs(available: Mapping[str, set[str]]) -> Needs:
        for t in ("land_use", "persons"):
            if t not in available:
                raise MissingInputError(f"needs the {t} table")
        missing = [c for c in size_columns if c not in available["land_use"]]
        if missing:
            raise MissingInputError(f"size terms use land_use columns {missing}, which don't exist")
        needs: dict[str, set[str]] = {"land_use": set(size_columns)}
        skims: set[str] = set()
        temps = {t.target for d in [*cfg.sample_defs.values(), *cfg.choice_defs.values()] for t in d.terms
                 if t.target}
        columns = {cfg.orig_col, cfg.filter_col, cfg.segment_col}
        for d in cfg.sample_defs.values():
            columns |= d.columns_used() - {"size_term", ZONE_KEY} - alt_values
            skims |= {k for _, k, _ in d.skim_uses()}
        for d in cfg.choice_defs.values():
            columns |= d.columns_used() - sample_cols - {MODE_CHOICE_LOGSUM, "size_term"} - alt_values
            skims |= {k for _, k, _ in d.skim_uses()}
        columns -= temps
        if new_format:
            from .location_choice_files import check_terms

            check_terms(cfg, available)
        bound = {}
        if cfg.model is not None:
            df_cols = sample_cols | ls_added | PERSONS_VIEW.columns(available)
            for seg in cfg.segments:
                if new_format:
                    from .roles import PERSON_ROLES, context_roles

                    roles = context_roles(PERSONS_VIEW, PERSON_ROLES, available, df_cols)
                    b = cfg.model.bind(df_cols, available, dest=cfg.alt_col, orig=cfg.orig_col,
                                       parameters=cfg.logsum_parameters, roles=roles)
                else:
                    b = cfg.model.bind(df_cols, available, dest=cfg.alt_col, orig=cfg.orig_col)
                bound[seg] = b
                columns |= b.columns - sample_cols - ls_added
                for t, c in b.tables:
                    needs.setdefault(t, set()).add(c)
                skims |= b.skims
        view = PERSONS_VIEW.resolve(columns, available)
        for t, c in view.items():
            needs.setdefault(t, set()).update(c)
        bound_ann, ann_needs, ann_provides = annotate.bind(cfg.annotations, available, made)
        for t, c in ann_needs.items():
            needs.setdefault(t, set()).update(c)
        needs.setdefault(SKIMS, set()).update(skims)
        if not needs[SKIMS]:
            del needs[SKIMS]
        resolved.update(needs={t: sorted(c) for t, c in needs.items()}, view=view, bound=bound,
                        annotations=bound_ann, annotation_provides=ann_provides)
        return resolved["needs"]

    def run(ctx: StepContext):
        ids = ctx.ids("persons")
        cols_all = {c: plain(v) for c, v in PERSONS_VIEW.gather(ctx, resolved["view"]).items()}
        chooser_rows = np.flatnonzero(np.asarray(cols_all[cfg.filter_col]).astype(bool))
        chooser_rows = chooser_rows[np.argsort(ids[chooser_rows], kind="stable")]  # 1.x sorts by person ID
        cols = {c: np.asarray(v)[chooser_rows] for c, v in cols_all.items()}
        person_ids = ids[chooser_rows]
        cols["person_id"] = person_ids
        seg_values = np.asarray(cols[cfg.segment_col])
        check_segment_values(step, "choosers", cfg.segment_col, seg_values, cfg.segments.values())
        zones, sizes = _size_terms(ctx, size_weights, size_columns)
        orig_pos = ctx.skim_positions(np.asarray(cols[cfg.orig_col]))
        chosen = np.full(len(person_ids), np.nan)
        dc_logsums = np.full(len(person_ids), np.nan)
        mc_logsums = np.full(len(person_ids), np.nan)
        debug = info["samples"] = {}
        for seg, seg_id in cfg.segments.items():
            sel = np.flatnonzero(seg_values == seg_id)
            if not len(sel):
                continue  # 1.x skips segments without choosers
            size = sizes[seg]
            alt_zones, alt_size = zones[size > 0], size[size > 0]
            if not len(alt_zones):
                raise ValueError(f"{step}: no zone has a size for segment '{seg}'")
            sample = _sample(ctx, sel, person_ids, cols, orig_pos, alt_zones, alt_size, cfg.sample_fns[seg],
                             cfg.sample_size, cfg.shift, max_rows, channel=CHANNEL, step=step,
                             alt_values=ALT_VALUES)
            if cfg.model is not None:
                purpose = cfg.logsum_purpose(seg)
                sample[MODE_CHOICE_LOGSUM] = _logsums(
                    ctx, cfg.model, resolved["bound"][seg], resolved["view"], purpose, cfg.periods_for(purpose),
                    sample, person_ids, cols, cfg.alt_col, "person_id", channel=CHANNEL)
            else:
                sample[MODE_CHOICE_LOGSUM] = np.zeros(len(sample["tour"]))  # 1.x: 0.0 without logsums
            zone_size = pd.Series(alt_size, index=alt_zones)
            seg_chosen, seg_logsums = _choose(ctx, sel, person_ids, cols, orig_pos, sample, zone_size,
                                              cfg.choice_fns[seg], cfg.shift, max_rows, cfg.alt_col, "person_id",
                                              channel=CHANNEL, step=step, alt_values=ALT_VALUES)
            chosen[sel] = seg_chosen
            dc_logsums[sel] = seg_logsums
            if cfg.mode_logsum_column:
                key = pd.Series(sample[MODE_CHOICE_LOGSUM],
                                index=pd.MultiIndex.from_arrays([sample["tour"], sample["zone"]]))
                mc_logsums[sel] = key.reindex(pd.MultiIndex.from_arrays([sel, seg_chosen])).to_numpy()
            debug[seg] = {"person_id": person_ids[sample["tour"]], **{k: v for k, v in sample.items() if k != "tour"}}

        persons: dict[str, np.ndarray] = {}
        dest = np.full(len(ids), NO_ZONE, dtype=np.int64)
        dest[chooser_rows] = np.where(np.isnan(chosen), NO_ZONE, chosen).astype(np.int64)
        persons[cfg.dest_column] = dest
        for name, values in ((cfg.logsum_column, dc_logsums), (cfg.mode_logsum_column, mc_logsums)):
            if name:
                full = np.full(len(ids), np.nan)
                full[chooser_rows] = values
                persons[name] = full
        out: dict = {}
        if cfg.modeled_size_table:
            columns = {}
            for seg in size_weights:  # 1.x: every segment of the size terms, counts per zone
                seg_id = cfg.segments.get(seg)
                picks = chosen[seg_values == seg_id].astype(np.int64) if seg_id is not None else np.array([], int)
                counts = pd.Series(picks).value_counts().reindex(zones, fill_value=0)
                columns[seg] = counts.to_numpy().astype(int)
            out[cfg.modeled_size_table] = Table(cfg.modeled_size_table, zones, "zone_id", columns)
        made_values = {"persons": persons}
        annotate.run(ctx, resolved["annotations"], made_values)
        out.update(made_values)
        return out

    provides: dict[str, list[str]] = {"persons": sorted(made["persons"])}
    info: dict = {}
    # annotations add columns; worked out from the files (their targets)
    for a in cfg.annotations:
        targets = a.spec.targets
        provides.setdefault(a.table, [])
        provides[a.table] += [t for t in targets if t not in provides[a.table]]
    if cfg.modeled_size_table:
        provides[cfg.modeled_size_table] = list(size_weights)
    component = Component(
        name=step,
        run=run,
        provides=provides,
        resolve_needs=resolve_needs,
        description=f"{step.replace('_', ' ')}: {cfg.sample_size} sampled zones per person, "
                    f"{len(cfg.segments)} segments{', mode choice logsums' if cfg.model else ''}",
    )
    info.update(sample_specs=cfg.sample_defs, specs=cfg.choice_defs, logsum_model=cfg.model, trace_table="persons",
                resolved=resolved, annotations=cfg.annotations,
                generated_source="\n\n".join(src for _, _, src in cfg.choice_fns.values()))
    component.info = info
    return component
