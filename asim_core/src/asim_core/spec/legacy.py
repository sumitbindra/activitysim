"""
Load today's ActivitySim 1.x model files into a ``ChoiceModelDefinition``.

Reads the model's settings YAML, its spec CSV and its coefficients CSV,
following the same rules as ActivitySim 1.x (activitysim/core/simulate.py:
``read_model_spec``, ``read_model_coefficients``, ``eval_coefficients``), so
existing models load unchanged. Config folders are searched in order and the
first match wins, as in 1.x.

Supported so far: simple multinomial logit models without preprocessors or
coefficient templates. Anything else raises ``NotImplementedError`` naming
the setting, rather than running a partial model.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yaml

from .definition import ChoiceModelDefinition, Coefficient, Term

_SPEC_META_COLUMNS = {"Label", "Description", "Expression"}

# settings that change how a model runs and are not supported yet
_NOT_YET_SUPPORTED = {
    "NESTS": "nested logit",
    "COEFFICIENT_TEMPLATE": "coefficient templates (purpose segments)",
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
) -> ChoiceModelDefinition:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings_path = find_config_file(configs_dirs, settings_file)
    settings = read_yaml(settings_path)

    unsupported = [f"{k} ({why})" for k, why in _NOT_YET_SUPPORTED.items() if settings.get(k)]
    logit_type = settings.get("LOGIT_TYPE", "MNL")
    if logit_type != "MNL":
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

    # constants: global constants.yaml, overridden by the model's own CONSTANTS
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})

    return ChoiceModelDefinition(
        name=name or Path(settings_file).stem,
        logit_type="MNL",
        alternatives=alternatives,
        constants=constants,
        terms=terms,
        coefficients=coefficients,
        source_files=[settings_path.name, spec_path.name, coef_path.name],
    )
