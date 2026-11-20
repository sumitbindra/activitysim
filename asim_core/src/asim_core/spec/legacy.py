"""
Load today's ActivitySim 1.x model files into a ``ChoiceModelDefinition``.

Reads the model's settings YAML, its spec CSV and its coefficients CSV,
following the same rules as ActivitySim 1.x (activitysim/core/simulate.py:
``read_model_spec``, ``read_model_coefficients``, ``eval_coefficients``), so
existing models load unchanged. Config folders are searched in order and the
first match wins, as in 1.x.

Supported so far: simple multinomial and nested logit models, with or
without coefficient templates (segments). Preprocessors are handled by the
components that use them (e.g. tour mode choice). Anything else raises
``NotImplementedError`` naming the setting, rather than running a partial
model.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yaml

from .definition import (
    Assignment,
    AssignmentSpec,
    ChoiceModelDefinition,
    Coefficient,
    InteractionModelDefinition,
    InteractionTerm,
    Term,
)

# values read as missing in assignment specs: pandas 1.5's defaults, without
# 'None' (used in spec files as the object None), as in ActivitySim 1.x
_ASSIGN_NA = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
              "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]

_SPEC_META_COLUMNS = {"Label", "Description", "Expression"}

# settings that change how a model runs and are not supported yet
_NOT_YET_SUPPORTED = {
    "preprocessor": "preprocessors",
    "annotate_households": "annotation after the model",
    "annotate_persons": "annotation after the model",
    "annotate_tours": "annotation after the model",
}


def find_config_file(configs_dirs: Sequence[Path], file_name: str) -> Path:
    for d in configs_dirs:
        p = Path(d) / file_name
        if p.exists():
            return p
    raise FileNotFoundError(f"'{file_name}' not found in {[str(d) for d in configs_dirs]}")


def read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_top_settings(configs_dirs: Sequence[str | Path]) -> dict:
    """
    The run's settings.yaml. As in 1.x, a settings.yaml with
    ``inherit_settings: True`` is layered on top of the next one found.
    """
    layers = []
    for d in configs_dirs:
        p = Path(d) / "settings.yaml"
        if p.exists():
            s = read_yaml(p)
            layers.append(s)
            if not s.get("inherit_settings"):
                break
    merged: dict = {}
    for s in reversed(layers):
        merged.update(s)
    merged.pop("inherit_settings", None)
    return merged


def load_global_constants(configs_dirs: Sequence[Path]) -> dict:
    try:
        return read_yaml(find_config_file(configs_dirs, "constants.yaml"))
    except FileNotFoundError:
        return {}


def load_choice_model(
    configs_dirs: Sequence[str | Path],
    settings_file: str,
    name: str | None = None,
    handled: Sequence[str] = (),
    formulas_see_coefficients: bool = False,
    skim_accessors: Sequence[str] = (),
    extra_constants: dict | None = None,
) -> ChoiceModelDefinition:
    """
    ``handled`` lists settings the calling component deals with itself (e.g.
    ``preprocessor``), so they aren't reported as unsupported.
    ``formulas_see_coefficients``: formulas may use coefficient names as values
    (1.x allows this in mode choice models). ``skim_accessors`` are the skim
    lookups formulas may use (e.g. ``odt_skims``). ``extra_constants`` are
    added last (they win over constants.yaml and the model's CONSTANTS).
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    settings_path = find_config_file(configs_dirs, settings_file)
    settings = read_yaml(settings_path)

    unsupported = [f"{k} ({why})" for k, why in _NOT_YET_SUPPORTED.items()
                   if settings.get(k) and k not in handled]
    logit_type = settings.get("LOGIT_TYPE", "MNL")
    if logit_type not in ("MNL", "NL"):
        unsupported.append(f"LOGIT_TYPE: {logit_type}")
    if unsupported:
        raise NotImplementedError(
            f"{settings_file} uses settings asim_core does not support yet: {', '.join(unsupported)}"
        )

    spec_file = str(settings["SPEC"])
    if not spec_file.lower().endswith(".csv"):
        spec_file += ".csv"
    spec_path = find_config_file(configs_dirs, spec_file)
    coef_path = find_config_file(configs_dirs, str(settings["COEFFICIENTS"]))

    # spec: same reading rules as ActivitySim 1.x read_model_spec
    spec = pd.read_csv(spec_path, comment="#")
    spec = spec.dropna(subset=["Expression"])
    alternatives = [c for c in spec.columns if c not in _SPEC_META_COLUMNS]
    terms = []
    for _, row in spec.iterrows():
        coefs = {}
        for alt in alternatives:
            cell = row[alt]
            if pd.isna(cell) or str(cell).strip() == "":
                continue  # blank cell: the term does not apply to this alternative
            coefs[alt] = str(cell).strip()
        label = row["Label"] if "Label" in spec.columns and not pd.isna(row["Label"]) else None
        terms.append(Term(label=label, formula=str(row["Expression"]).strip(), coefficients=coefs))

    # coefficients: same checks as ActivitySim 1.x read_model_coefficients
    coef_df = pd.read_csv(coef_path, comment="#", index_col="coefficient_name")
    dupes = coef_df.index[coef_df.index.duplicated()].unique().tolist()
    if dupes:
        raise ValueError(f"duplicate coefficients in {coef_path}: {dupes}")
    nulls = coef_df.index[coef_df["value"].isnull()].tolist()
    if nulls:
        raise ValueError(f"coefficients without a value in {coef_path}: {nulls}")
    coefficients = {}
    for coef_name, row in coef_df.iterrows():
        constrain = str(row.get("constrain", "F")).strip().upper() in ("T", "TRUE", "1")
        coefficients[str(coef_name)] = Coefficient(value=float(row["value"]), constrain=constrain)

    # coefficient template: same rules as 1.x read_model_coefficient_template
    template = None
    source_files = [settings_path.name, spec_path.name, coef_path.name]
    if settings.get("COEFFICIENT_TEMPLATE"):
        template_path = find_config_file(configs_dirs, str(settings["COEFFICIENT_TEMPLATE"]))
        source_files.append(template_path.name)
        template = read_coefficient_template(template_path)

    # constants: global constants.yaml, overridden by the model's own CONSTANTS
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    constants.update(extra_constants or {})

    return ChoiceModelDefinition(
        name=name or Path(settings_file).stem,
        logit_type=logit_type,
        # as 1.x: NESTS is only used for nested logit
        nests=settings.get("NESTS") if logit_type == "NL" else None,
        alternatives=alternatives,
        constants=constants,
        terms=terms,
        coefficients=coefficients,
        coefficient_template=template,
        formulas_see_coefficients=formulas_see_coefficients,
        skim_accessors=list(skim_accessors),
        source_files=source_files,
    )


def read_coefficient_template(path: Path) -> dict[str, dict[str, str]]:
    """
    Generic coefficient name -> {segment: coefficient name}. As in 1.x, an
    empty cell means the generic name itself is used for that segment.
    """
    df = pd.read_csv(path, comment="#", index_col="coefficient_name")
    dupes = df.index[df.index.duplicated()].unique().tolist()
    if dupes:
        raise ValueError(f"duplicate coefficient names in {path}: {dupes}")
    template = {}
    for generic, row in df.iterrows():
        template[str(generic)] = {
            str(seg): (str(generic) if pd.isna(cell) else str(cell).strip()) for seg, cell in row.items()
        }
    return template


def load_assignment_spec(
    configs_dirs: Sequence[str | Path],
    spec_file: str,
    constants: dict | None = None,
    skim_accessors: Sequence[str] = (),
    name: str | None = None,
    tables: Sequence[str] = (),
    coefficients: Sequence[str] = (),
    settings: dict | None = None,
    df_names: Sequence[str] = (),
) -> AssignmentSpec:
    """
    Read an assignment spec CSV (Description, Target, Expression), following
    ActivitySim 1.x's ``read_assignment_spec``. ``constants`` should already
    combine the global constants.yaml with the model's own CONSTANTS.
    ``tables`` are the other tables formulas may read (1.x's ``TABLES``),
    ``coefficients`` names supplied at run time, ``settings`` the run's
    settings.yaml (for ``setting('name')``).
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    path = find_config_file(configs_dirs, spec_file)
    df = pd.read_csv(path, comment="#", na_values=_ASSIGN_NA, keep_default_na=False)
    assignments = []
    for _, row in df.iterrows():
        target, formula = row.get("Target"), row.get("Expression")
        if pd.isna(target) or pd.isna(formula):
            continue
        description = row.get("Description", "")
        assignments.append(Assignment(
            target=str(target).strip(),
            formula=str(formula).strip(),
            description="" if pd.isna(description) else str(description).strip(),
        ))
    return AssignmentSpec(
        name=name or Path(spec_file).stem,
        assignments=assignments,
        constants=constants or {},
        skim_accessors=list(skim_accessors),
        tables=list(tables),
        coefficients=list(coefficients),
        settings=dict(settings or {}),
        df_names=list(df_names),
        source_files=[path.name],
    )


def read_coefficients(path: Path) -> dict[str, Coefficient]:
    """A coefficients file, with 1.x's read_model_coefficients checks."""
    coef_df = pd.read_csv(path, comment="#", index_col="coefficient_name")
    dupes = coef_df.index[coef_df.index.duplicated()].unique().tolist()
    if dupes:
        raise ValueError(f"duplicate coefficients in {path}: {dupes}")
    nulls = coef_df.index[coef_df["value"].isnull()].tolist()
    if nulls:
        raise ValueError(f"coefficients without a value in {path}: {nulls}")
    out = {}
    for name, row in coef_df.iterrows():
        constrain = str(row.get("constrain", "F")).strip().upper() in ("T", "TRUE", "1")
        out[str(name)] = Coefficient(value=float(row["value"]), constrain=constrain)
    return out


def load_interaction_model(
    configs_dirs: Sequence[str | Path],
    spec_file: str,
    coefficients_file: str,
    constants: dict | None = None,
    objects: dict[str, Sequence[str]] | None = None,
    name: str | None = None,
    value_column: str | None = None,
    skim_accessors: Sequence[str] = (),
) -> InteractionModelDefinition:
    """
    An interaction spec (Label, Description, Expression, coefficient
    column(s)), read as 1.x's read_model_spec reads it. Rows written
    ``_name@formula`` assign a temporary. A spec with one coefficient column
    per segment needs ``value_column`` (1.x's spec_for_segment).
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    if not spec_file.lower().endswith(".csv"):
        spec_file += ".csv"
    spec_path = find_config_file(configs_dirs, spec_file)
    coef_path = find_config_file(configs_dirs, coefficients_file)
    spec = pd.read_csv(spec_path, comment="#").dropna(subset=["Expression"])
    value_cols = [c for c in spec.columns if c not in _SPEC_META_COLUMNS]
    if value_column is not None and len(value_cols) == 1 and value_cols[0] in ("coefficient", value_column):
        pass  # 1.x's spec_for_segment: a single coefficient column serves every segment
    elif value_column is not None:
        if value_column not in value_cols:
            raise ValueError(f"{spec_path.name}: no coefficient column '{value_column}' "
                             f"(columns: {value_cols})")
        value_cols = [value_column]
    if len(value_cols) != 1:
        raise ValueError(f"{spec_path.name}: an interaction spec has one coefficient column, "
                         f"found {value_cols}")
    terms = []
    for _, row in spec.iterrows():
        expr = str(row["Expression"]).strip()
        cell = row[value_cols[0]]
        cell = "0" if pd.isna(cell) or str(cell).strip() == "" else str(cell).strip()
        label = row["Label"] if "Label" in spec.columns and not pd.isna(row["Label"]) else None
        target = None
        if expr.startswith("_") and "@" in expr:  # _name@formula: a temporary (1.x)
            target, expr = expr[:expr.index("@")], expr[expr.index("@") + 1:]
        terms.append(InteractionTerm(label=label, formula=expr, coefficient=cell, target=target))
    return InteractionModelDefinition(
        name=name or Path(spec_file).stem,
        terms=terms,
        coefficients=read_coefficients(coef_path),
        constants=constants or {},
        objects={k: list(v) for k, v in (objects or {}).items()},
        skim_accessors=list(skim_accessors),
        source_files=[spec_path.name, coef_path.name],
    )
