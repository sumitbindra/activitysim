"""
Auto ownership: how many cars each household owns.

Reads either today's auto_ownership.yaml / .csv / _coefficients.csv, or
asim_core's model file (``kind: auto_ownership``). Choosers are households,
seeing land_use and accessibility columns through their home zone, as in
ActivitySim 1.x's ``households_merged``.

In asim_core's language, formulas read ``household.<column>`` and
``home_zone.<column>`` (the home zone's land use and accessibility).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ..component import Component
from ..spec.convert import LegacyScope, NewNames, convert_formula
from ..spec.definition import ChoiceModelDefinition, LanguageScope, Nest, Term
from ..spec.legacy import (
    find_config_file,
    load_choice_model,
    load_global_constants,
    load_top_settings,
    read_yaml,
)
from ..spec.model_file import (
    ModelFile,
    is_model_file,
    read_coefficients_table,
    read_model_yaml,
    read_terms,
    validate,
    write_coefficients,
    write_terms,
    write_yaml,
)
from .roles import HOUSEHOLD_ROLES, role_columns, role_finder
from .simple_choice import ChooserView, simple_choice_component

STEP = "auto_ownership_simulate"  # 1.x step name: random streams are seeded from it
VIEW = ChooserView("households", joins=[("land_use", "home_zone_id"), ("accessibility", "home_zone_id")])
SCOPE = LanguageScope(roles=list(HOUSEHOLD_ROLES))


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 1.x subtracts each row's largest utility before exponentiating only when
    # skip_failed_choices is off; it is on by default
    utility_shift: bool = False
    # 1.x skips terms whose coefficients are all zero
    drop_zero_terms: bool = True


class AutoOwnershipFile(ModelFile):
    kind: Literal["auto_ownership"]
    alternatives: list[str]
    logit: Literal["multinomial", "nested"] = "multinomial"
    nests: Nest | None = None
    terms: str
    coefficients: str
    output: str = "auto_ownership"
    random_stream: str = "households"
    options: Options = Options()


def load_model_file(path: str | Path, configs_dirs: Sequence[str | Path]) -> tuple[ChoiceModelDefinition,
                                                                                    AutoOwnershipFile]:
    data = read_model_yaml(path)
    f = validate(AutoOwnershipFile, data, path)
    rows, value_cols = read_terms(f.terms)
    extra = [c for c in value_cols if c not in f.alternatives]
    if extra:
        raise ValueError(f"{f.terms}: columns {extra} are not alternatives {f.alternatives}")
    coefs, by_seg = read_coefficients_table(f.coefficients, f.coefficient_values)
    if by_seg is not None:
        raise ValueError(f"{f.coefficients}: auto ownership has one value per coefficient")
    constants = dict(load_global_constants([Path(d) for d in configs_dirs]))
    constants.update(f.constants)
    terms = []
    for label, _, formula, target, cells in rows:
        if target:
            raise ValueError(f"{f.terms}: '{target} = ...' rows are not supported in this model")
        terms.append(Term(label=label, formula=formula, coefficients=cells))
    definition = ChoiceModelDefinition(
        name="auto_ownership",
        logit_type="NL" if f.logit == "nested" else "MNL",
        nests=f.nests,
        alternatives=f.alternatives,
        constants=constants,
        terms=terms,
        coefficients=coefs,
        language="asim_core",
        scope=SCOPE,
        source_files=[Path(p).name for p in f.source_files] + [Path(f.terms).name, Path(f.coefficients).name],
    )
    return definition, f


def auto_ownership(
    configs_dirs: Sequence[str | Path],
    settings_file: str = "auto_ownership.yaml",
) -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    path = find_config_file(configs_dirs, settings_file)
    if is_model_file(path):
        definition, f = load_model_file(path, configs_dirs)
        return simple_choice_component(
            definition, step=STEP, view=VIEW, output_column=f.output, random_channel=f.random_stream,
            shift_utilities=f.options.utility_shift, drop_zero_terms=f.options.drop_zero_terms,
            roles=HOUSEHOLD_ROLES,
        )
    definition = load_choice_model(configs_dirs, settings_file, name="auto_ownership")
    top = load_top_settings(configs_dirs)
    # 1.x skips overflow protection when skip_failed_choices is on (its default)
    shift = not top.get("skip_failed_choices", True)
    return simple_choice_component(
        definition, step=STEP, view=VIEW, output_column="auto_ownership",
        random_channel="households", shift_utilities=shift,
    )


# --- converting today's files ------------------------------------------------------------


def convert(configs_dirs: Sequence[str | Path], columns: Mapping[str, set[str]], out_dir: str | Path,
            settings_file: str = "auto_ownership.yaml") -> list[Path]:
    """
    Write asim_core's model file for today's auto ownership files.
    ``columns``: each table's columns when the step runs (from a 1.x run).
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    out_dir = Path(out_dir)
    settings_path = find_config_file(configs_dirs, settings_file)
    settings = read_yaml(settings_path)
    definition = load_choice_model(configs_dirs, settings_file, name="auto_ownership")
    top = load_top_settings(configs_dirs)
    names = NewNames(role_of=role_finder(role_columns(VIEW, HOUSEHOLD_ROLES, columns)))
    legacy = LegacyScope(constants=definition.constants)
    descriptions = _spec_descriptions(configs_dirs, settings)
    rows = []
    for i, term in enumerate(definition.terms):
        rows.append({"label": term.label, "description": descriptions[i],
                     "formula": convert_formula(term.formula, legacy, names).text,
                     "cells": term.coefficients})
    stem = Path(settings_file).stem
    terms_path = out_dir / f"{stem}_terms.csv"
    coef_path = out_dir / f"{stem}_coefficients.csv"
    write_terms(terms_path, rows, definition.alternatives)
    write_coefficients(coef_path, {n: c.value for n, c in definition.coefficients.items()})
    data = {
        "kind": "auto_ownership",
        "description": "How many cars each household owns (converted from "
                       f"{', '.join(definition.source_files)}).",
        "alternatives": definition.alternatives,
        "logit": "nested" if definition.logit_type == "NL" else "multinomial",
    }
    if definition.nests is not None:
        data["nests"] = definition.nests.model_dump()
    data.update({
        "terms": terms_path.name,
        "coefficients": coef_path.name,
        "constants": settings.get("CONSTANTS") or {},
        "output": "auto_ownership",
        "random_stream": "households",
        "options": {"utility_shift": not top.get("skip_failed_choices", True), "drop_zero_terms": True},
    })
    if not data["constants"]:
        del data["constants"]
    yaml_path = out_dir / Path(settings_file).name
    write_yaml(yaml_path, data, "# asim_core model file (formulas in asim_core's language; see docs/model_files.md)")
    return [yaml_path, terms_path, coef_path]


def _spec_descriptions(configs_dirs, settings) -> list[str]:
    spec_file = str(settings["SPEC"])
    spec_file = spec_file if spec_file.lower().endswith(".csv") else spec_file + ".csv"
    spec = pd.read_csv(find_config_file(configs_dirs, spec_file), comment="#").dropna(subset=["Expression"])
    if "Description" not in spec.columns:
        return [""] * len(spec)
    return ["" if pd.isna(d) else str(d).strip() for d in spec["Description"]]
