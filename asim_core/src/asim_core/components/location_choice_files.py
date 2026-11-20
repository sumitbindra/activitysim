"""
School and workplace location in asim_core's model files (``kind:
location_choice``), and the converter from today's files.

Formulas see ``person``, ``household``, ``home_zone`` (the chooser) and
``alt`` (the candidate zone: ``alt.zone_id`` in the sample,
``alt.<alternative_zone>`` in the choice), and the step's values
``size_term``, ``prob``, ``pick_count``, ``mode_choice_logsum``,
``shadow_price_size_term_adjustment`` and ``shadow_price_utility_adjustment``.
Skims go from the person's origin (``o``) to the candidate (``d``).
Annotations see their table (``person`` or ``household``), other tables by
ID, and skims between any zones (``skim.DIST[person.home_zone_id,
person.workplace_zone_id]``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ..spec.definition import InteractionModelDefinition, InteractionTerm, LanguageScope
from ..spec.legacy import find_config_file, load_global_constants, load_top_settings, read_yaml
from ..spec.model_file import (
    ModelFile,
    locate,
    read_coefficients_table,
    read_derived,
    read_model_yaml,
    read_terms,
    validate,
)
from . import annotate
from .roles import PERSON_ROLES, check_roles, role_columns
from .tour_destination import PERSONS_VIEW, SKIM_ACCESSOR
from .tour_mode_choice import TourModeChoiceModel

SAMPLE_VALUES = ["size_term", "shadow_price_size_term_adjustment", "shadow_price_utility_adjustment"]
CHOICE_VALUES = SAMPLE_VALUES + ["prob", "pick_count", "mode_choice_logsum"]


class SizeTerms(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str = "destination_choice_size_terms.csv"
    selector: str


class Logsums(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str  # a tour mode choice model file
    tour_purpose: str | dict[str, str]  # the mode choice segment: one, or one per segment
    # (departure, return) periods: one pair, or one per tour purpose (as 1.x looks them up)
    periods: tuple[int, int] | dict[str, tuple[int, int]]
    parameters: dict[str, Any] = {}


class Outputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zone: str                       # e.g. workplace_zone_id
    zone_when_none: Literal[-1] = -1  # persons who don't choose (1.x)
    logsum: str | None = None
    mode_choice_logsum: str | None = None
    modeled_size: str | None = None  # a table: persons choosing each zone, per segment


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    derived: str
    tables: list[str] = []  # other tables looked up by ID


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utility_shift: bool = False
    max_rows: int | None = None
    shadow_pricing: Literal[False] = False  # not implemented yet
    sample_method: Literal["inverse_cdf"] = "inverse_cdf"
    sample_size_cap: Literal["zones_with_size"] = "zones_with_size"
    sample_order: Literal["by_zone"] = "by_zone"
    sample_probabilities: Literal["float32"] = "float32"


class LocationChoiceFile(ModelFile):
    kind: Literal["location_choice"]
    filter: str      # who chooses, e.g. person.is_worker
    segment: str     # the column picking the segment, e.g. household.income_segment
    segments: dict[str, Any]  # segment name -> its value in that column, in the order they run
    size_terms: SizeTerms
    sample_size: int
    sample_terms: str  # one coefficient column per segment, or a single 'coefficient' column
    terms: str
    coefficients: str
    origin: str      # e.g. household.home_zone_id
    alternative_zone: str = "alt_dest"
    logsums: Logsums | None = None
    outputs: Outputs
    # annotations after the choice, in 1.x's order (persons, then households)
    annotate: dict[Literal["persons", "households"], Annotation] = {}
    options: Options = Options()


INDEX_NAMES = {"persons": "person_id", "households": "household_id"}


def _column(text: str, path) -> str:
    import ast

    node = ast.parse(text, mode="eval").body
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr
    raise ValueError(f"{path}: '{text}' must be a column, e.g. household.home_zone_id")


def load_model_file(path, configs_dirs: Sequence[Path], step: str):
    from .location_choice import _Config

    f = validate(LocationChoiceFile, read_model_yaml(path), path)
    constants = dict(load_global_constants(configs_dirs))
    top = load_top_settings(configs_dirs)
    constants_model = {**constants, **f.constants}
    coefficients, by_seg = read_coefficients_table(f.coefficients, f.coefficient_values)
    if by_seg is not None:
        raise ValueError(f"{f.coefficients}: one value per coefficient (segments are columns of the term tables)")
    roles = list(PERSON_ROLES) + ["alt"]

    def load(terms_path, values, kind):
        rows, value_cols = read_terms(terms_path)
        single = value_cols == ["coefficient"]
        missing = [seg for seg in f.segments if not single and seg not in value_cols]
        if missing:
            raise ValueError(f"{terms_path}: no coefficient column for segments {missing}")
        scope = LanguageScope(roles=roles, derived=values, skims={"o d": SKIM_ACCESSOR})
        out = {}
        for seg in f.segments:
            col = "coefficient" if single else seg
            terms = [InteractionTerm(label=label, formula=formula, coefficient=cells.get(col, "0"), target=target)
                     for label, _, formula, target, cells in rows]
            out[seg] = InteractionModelDefinition(
                name=f"{step}_{kind}_{seg}", terms=terms, coefficients=coefficients, constants=constants_model,
                language="asim_core", scope=scope, source_files=[Path(terms_path).name, Path(f.coefficients).name])
        return out

    sample_defs = load(f.sample_terms, SAMPLE_VALUES, "sample")
    choice_defs = load(f.terms, CHOICE_VALUES, "spec")
    model = None
    if f.logsums is not None:
        model_path = Path(locate(f.logsums.model, configs_dirs))
        model = TourModeChoiceModel([model_path.parent] + list(configs_dirs), model_path.name)
    ls = f.logsums

    def purpose(seg):
        return ls.tour_purpose[seg] if isinstance(ls.tour_purpose, dict) else ls.tour_purpose

    def periods_for(tour_purpose):
        p = ls.periods[tour_purpose] if isinstance(ls.periods, dict) else ls.periods
        return int(p[0]), int(p[1])

    annotations = []
    for table, a in f.annotate.items():
        annotations.append(annotate.model_file_annotation(
            table, read_derived(a.derived), list(a.tables), constants, top, step, Path(a.derived).name,
            INDEX_NAMES[table]))
    return _Config(
        segments=dict(f.segments), segment_col=_column(f.segment, path), filter_col=_column(f.filter, path),
        orig_col=_column(f.origin, path), alt_col=f.alternative_zone, sample_size=f.sample_size,
        size_selector=f.size_terms.selector, sample_defs=sample_defs, choice_defs=choice_defs,
        sample_fns={seg: d.utility_terms() for seg, d in sample_defs.items()},
        choice_fns={seg: d.utility_terms() for seg, d in choice_defs.items()},
        model=model, logsum_purpose=purpose if ls else None, periods_for=periods_for if ls else None,
        dest_column=f.outputs.zone, logsum_column=f.outputs.logsum,
        mode_logsum_column=f.outputs.mode_choice_logsum, modeled_size_table=f.outputs.modeled_size,
        annotations=annotations, shift=f.options.utility_shift, language="asim_core",
        logsum_parameters=dict(ls.parameters) if ls else None, max_rows=f.options.max_rows,
        size_terms_file=locate(f.size_terms.file, configs_dirs),
    )


def check_terms(cfg, available: Mapping[str, set[str]]) -> None:
    """Every role.column the sample and choice terms read exists in the run's tables."""
    for kind, defs, alt in (("sample terms", cfg.sample_defs, "zone_id"), ("terms", cfg.choice_defs, cfg.alt_col)):
        roles = role_columns(PERSONS_VIEW, PERSON_ROLES, available, extra={"alt": {alt}})
        for d in defs.values():
            check_roles(f"{d.name} {kind}", d.translated, roles)


# --- converting today's files ------------------------------------------------------------


def convert(configs_dirs: Sequence[str | Path], available: Mapping[str, set[str]], out_dir: str | Path,
            step: str, settings_file: str) -> list[Path]:
    """asim_core's model files for today's location choice files; ``available``: columns when the step runs."""
    from ..spec.convert import LegacyScope, NewNames, convert_formula
    from ..spec.model_file import write_coefficients, write_derived, write_terms, write_yaml
    from .location_choice import _load_legacy, location_choice_component
    from .roles import role_finder
    from .tour_mode_choice import legacy_parameters

    configs_dirs = [Path(d) for d in configs_dirs]
    out_dir = Path(out_dir)
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    cfg = _load_legacy(configs_dirs, settings_file, step)
    comp = location_choice_component(configs_dirs, step, settings_file)
    if cfg.model is not None:
        comp.info["logsum_model"].bind_log.clear()
    comp.resolve_needs(available)
    written = []
    find_role = role_finder(role_columns(PERSONS_VIEW, PERSON_ROLES, available))

    def convert_spec(defs, spec_key, alt_col, values):
        first = defs[next(iter(cfg.segments))]
        names = NewNames(role_of=role_finder(role_columns(PERSONS_VIEW, PERSON_ROLES, available,
                                                          extra={"alt": {alt_col}})), derived=set(values))
        spec = pd.read_csv(find_config_file(configs_dirs, settings[spec_key]), comment="#").dropna(
            subset=["Expression"])
        value_cols = [c for c in spec.columns if c not in ("Label", "Description", "Expression")]
        single = len(value_cols) == 1 and value_cols[0] in ("coefficient",)
        descriptions = ["" if pd.isna(x) else str(x).strip() for x in spec.get("Description", [""] * len(spec))]
        rows, assigned = [], set()
        for i, t in enumerate(first.terms):
            legacy = LegacyScope(constants=first.constants, pandas_locals=True, assigned=set(assigned),
                                 python_mode=True if t.target else None, skims={SKIM_ACCESSOR: ("o", "d", None)})
            cells = ({"coefficient": t.coefficient} if single
                     else {seg: defs[seg].terms[i].coefficient for seg in cfg.segments})
            rows.append({"label": t.label, "description": descriptions[i] if i < len(descriptions) else "",
                         "target": t.target, "formula": convert_formula(t.formula, legacy, names).text,
                         "cells": cells})
            if t.target:
                assigned.add(t.target)
        path = out_dir / f"{Path(settings[spec_key]).stem}_terms.csv"
        write_terms(path, rows, ["coefficient"] if single else list(cfg.segments))
        written.append(path)
        return path

    sample_path = convert_spec(cfg.sample_defs, "SAMPLE_SPEC", "zone_id", SAMPLE_VALUES)
    terms_path = convert_spec(cfg.choice_defs, "SPEC", cfg.alt_col, CHOICE_VALUES)
    coef_path = out_dir / f"{Path(settings['COEFFICIENTS']).stem}.csv"
    first = cfg.choice_defs[next(iter(cfg.segments))]
    write_coefficients(coef_path, {n: c.value for n, c in first.coefficients.items()})
    written.append(coef_path)

    # annotations
    made = {"persons": {cfg.dest_column} | {c for c in (cfg.logsum_column, cfg.mode_logsum_column) if c}}
    annotations = {}
    for a in cfg.annotations:
        table_cols = set(available.get(a.table, set())) | made.get(a.table, set())
        own = annotate.ROLE[a.table]
        names = NewNames(role_of=lambda c, cols=table_cols, own=own, idx=INDEX_NAMES[a.table]:
                         own if c in cols or c == idx else None)
        rows, assigned = [], set()
        df_name = settings[f"annotate_{a.table}"].get("DF", a.table)
        for x in a.spec.assignments:
            legacy = LegacyScope(constants=a.spec.constants, python_mode=True, assigned=set(assigned),
                                 df_names={"df", df_name}, tables=set(a.tables), skim_lookup={annotate.SKIM_ACCESSOR},
                                 settings=top, index_name=INDEX_NAMES[a.table])
            rows.append((x.description, x.target, convert_formula(x.formula, legacy, names).text))
            assigned.add(x.target)
        path = out_dir / f"{Path(a.spec.source_files[0]).stem}.csv"
        write_derived(path, rows)
        written.append(path)
        annotations[a.table] = {"derived": path.name, **({"tables": list(a.tables)} if a.tables else {})}
        made.setdefault(a.table, set()).update(a.spec.targets)

    data = {"kind": "location_choice",
            "description": f"{step.replace('_', ' ').capitalize()} (converted from {settings_file} and the files "
                           "it names).",
            "filter": f"{find_role(cfg.filter_col) or 'person'}.{cfg.filter_col}",
            "segment": f"{find_role(cfg.segment_col) or 'person'}.{cfg.segment_col}",
            "segments": dict(cfg.segments),
            "size_terms": {"file": "destination_choice_size_terms.csv", "selector": cfg.size_selector},
            "sample_size": cfg.sample_size,
            "sample_terms": sample_path.name,
            "terms": terms_path.name,
            "coefficients": coef_path.name,
            "origin": f"{find_role(cfg.orig_col) or 'person'}.{cfg.orig_col}",
            "alternative_zone": cfg.alt_col}
    if cfg.model is not None:
        model = comp.info["logsum_model"]
        params = legacy_parameters(configs_dirs, str(settings["LOGSUM_SETTINGS"]))
        df_cols = model.bind_log[0][0]
        purpose = settings.get("LOGSUM_TOUR_PURPOSE")
        purposes = set(purpose.values()) if isinstance(purpose, dict) else {purpose}
        pairs = {p: list(cfg.periods_for(p)) for p in sorted(purposes)}
        periods = next(iter(pairs.values())) if len({tuple(v) for v in pairs.values()}) == 1 else pairs
        data["logsums"] = {"model": str(settings["LOGSUM_SETTINGS"]), "tour_purpose": purpose, "periods": periods}
        if params:
            data["logsums"]["parameters"] = {p: col in df_cols for p, col in sorted(params.items())}
    data["outputs"] = {"zone": cfg.dest_column, "zone_when_none": -1, "logsum": cfg.logsum_column,
                       "mode_choice_logsum": cfg.mode_logsum_column, "modeled_size": cfg.modeled_size_table}
    if annotations:
        data["annotate"] = annotations
    if settings.get("CONSTANTS"):
        data["constants"] = settings["CONSTANTS"]
    data["options"] = {"utility_shift": cfg.shift, "max_rows": None, "shadow_pricing": False,
                       "sample_method": "inverse_cdf", "sample_size_cap": "zones_with_size",
                       "sample_order": "by_zone", "sample_probabilities": "float32"}
    yaml_path = out_dir / Path(settings_file).name
    write_yaml(yaml_path, data, "# asim_core model file (formulas in asim_core's language; see docs/model_files.md)\n"
                                "# alt.<column>: the candidate zone; size_term, prob, pick_count, mode_choice_logsum\n"
                                "# and the shadow price adjustments: worked out by the step")
    return [yaml_path] + written
