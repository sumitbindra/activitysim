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

import locale

from collections.abc import Mapping, Sequence
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


def read_text(path: str | Path) -> str:
    """
    A settings or spec file's text: UTF-8 (YAML's own encoding, and what
    pandas reads CSV files as), or, for a file that isn't valid UTF-8 (say,
    saved as "ANSI" by Windows Notepad), the system's default encoding, which
    is how 1.x opens settings files. On Linux and macOS that default is UTF-8
    already; on Windows it is usually cp1252.
    """
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False))


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(read_text(path)) or {}


def read_settings(configs_dirs: Sequence[str | Path], file_name: str, mandatory: bool = True,
                  _stack: tuple = ()) -> dict:
    """
    A settings file, as 1.x's read_settings_file reads it, for every settings
    file (settings.yaml, constants.yaml, network_los.yaml, each model's): the
    first copy found in the configs folders; with ``inherit_settings: True``,
    its missing settings filled in from the next copy found (and so on);
    ``inherit_settings: <file>`` fills them in from that file instead;
    ``include_settings: <file>`` (alone in the file) is that other file. The
    two directives are not part of the result. Not found: FileNotFoundError,
    or {} if not ``mandatory``.
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    if not file_name.lower().endswith(".yaml"):
        file_name = f"{file_name}.yaml"
    settings: dict = {}
    found = False
    for d in configs_dirs:
        path = d / file_name
        if not path.exists():
            continue
        if str(path) in _stack:
            raise ValueError(f"{path}: settings files include or inherit each other: {' -> '.join(_stack)}")
        found = True
        _stack = _stack + (str(path),)
        s = read_yaml(path) or {}
        settings = {**s, **settings}  # what is already there wins (1.x: backfill)
        include = s.get("include_settings")
        if include:
            if len(s) > 1:
                raise ValueError(f"{path}: 'include_settings' must appear alone in the file (as 1.x requires)")
            s = read_settings(configs_dirs, include, True, _stack)
            settings = {**s, **settings}
        inherit = s.get("inherit_settings")
        if not inherit:
            break
        if isinstance(inherit, str):  # switch the chain to that file
            settings = {**read_settings(configs_dirs, inherit, True, _stack), **settings}
            break
    if not found:
        if mandatory:
            raise FileNotFoundError(f"'{file_name}' not found in {[str(d) for d in configs_dirs]}")
        return {}
    settings.pop("inherit_settings", None)
    settings.pop("include_settings", None)
    return settings


def load_top_settings(configs_dirs: Sequence[str | Path]) -> dict:
    """The run's settings.yaml, with 1.x's inheritance (``read_settings``)."""
    return read_settings(configs_dirs, "settings.yaml", mandatory=False)


def load_global_constants(configs_dirs: Sequence[Path]) -> dict:
    """constants.yaml, with 1.x's inheritance (SANDAG's inherits MTC's and changes one value)."""
    return read_settings(configs_dirs, "constants.yaml", mandatory=False)


def load_choice_model(
    configs_dirs: Sequence[str | Path],
    settings_file: str,
    name: str | None = None,
    handled: Sequence[str] = (),
    formulas_see_coefficients: bool = False,
    skim_accessors: Sequence[str] = (),
    extra_constants: dict | None = None,
    spec_key: str = "SPEC",
    overrides: Mapping | None = None,
) -> ChoiceModelDefinition:
    """
    ``handled`` lists settings the calling component deals with itself (e.g.
    ``preprocessor``), so they aren't reported as unsupported. ``spec_key``:
    the setting naming the spec file (CDAP: INDIV_AND_HHSIZE1_SPEC).
    ``overrides``: settings replacing the file's (a segment's spec and coefficients).
    ``formulas_see_coefficients``: formulas may use coefficient names as values
    (1.x allows this in mode choice models). ``skim_accessors`` are the skim
    lookups formulas may use (e.g. ``odt_skims``). ``extra_constants`` are
    added last (they win over constants.yaml and the model's CONSTANTS).
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    settings_path = find_config_file(configs_dirs, settings_file)
    settings = read_settings(configs_dirs, settings_file)
    settings.update(overrides or {})  # e.g. one segment's SPEC and COEFFICIENTS (1.x: SPEC_SEGMENTS)

    unsupported = [f"{k} ({why})" for k, why in _NOT_YET_SUPPORTED.items()
                   if settings.get(k) and k not in handled]
    logit_type = settings.get("LOGIT_TYPE", "MNL")
    if logit_type not in ("MNL", "NL"):
        unsupported.append(f"LOGIT_TYPE: {logit_type}")
    if unsupported:
        raise NotImplementedError(
            f"{settings_file} uses settings asim_core does not support yet: {', '.join(unsupported)}"
        )

    spec_file = str(settings[spec_key])
    if not spec_file.lower().endswith(".csv"):
        spec_file += ".csv"
    spec_path = find_config_file(configs_dirs, spec_file)
    legacy_path = None
    if settings.get("COEFFICIENTS") is None and settings.get("LEGACY_COEFFICIENTS"):
        legacy_path = find_config_file(configs_dirs, str(settings["LEGACY_COEFFICIENTS"]))
        coef_path = legacy_path
    else:
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

    if legacy_path is not None:
        coefficients, template = read_legacy_coefficients(legacy_path, settings.get("CONSTANTS") or {})
        source_files = [settings_path.name, spec_path.name, legacy_path.name]
        constants = dict(load_global_constants(configs_dirs))
        constants.update(settings.get("CONSTANTS") or {})
        constants.update(extra_constants or {})
        return ChoiceModelDefinition(
            name=name or Path(settings_file).stem, logit_type=logit_type,
            nests=settings.get("NESTS") if logit_type == "NL" else None, alternatives=alternatives,
            constants=constants, terms=terms, coefficients=coefficients, coefficient_template=template,
            formulas_see_coefficients=formulas_see_coefficients, skim_accessors=list(skim_accessors),
            source_files=source_files,
        )

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


def read_legacy_coefficients(path: Path, constants: Mapping) -> tuple[dict, dict]:
    """
    1.x's LEGACY_COEFFICIENTS: one table of coefficient names by segment, each
    value evaluated in order (1.x's assign.evaluate_constants: Python eval, with
    the segment's earlier values and the model's CONSTANTS). Returned as
    coefficients named ``name@segment`` and a template mapping each name to them.
    """
    df = pd.read_csv(path, comment="#", index_col="coefficient_name")
    coefficients: dict[str, Coefficient] = {}
    template: dict[str, dict[str, str]] = {}
    for seg in df.columns:
        values: dict = {}
        for k, v in df[seg].items():
            values[k] = eval(str(v), values.copy(), dict(constants))  # noqa: S307 (1.x evaluates them so)
        for k, v in values.items():
            coefficients[f"{k}@{seg}"] = Coefficient(value=float(v), constrain=False)
            template.setdefault(str(k), {})[str(seg)] = f"{k}@{seg}"
    return coefficients, template


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
    # without a coefficients file (1.x allows it), every cell must be a number
    coef_path = find_config_file(configs_dirs, coefficients_file) if coefficients_file else None
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
        if coef_path is None:
            try:
                float(cell)
            except ValueError:
                raise ValueError(f"{spec_path.name}: no coefficients file, but the cell '{cell}' isn't a number "
                                 f"(1.x needs every cell to be one then)") from None
        terms.append(InteractionTerm(label=label, formula=expr, coefficient=cell, target=target))
    return InteractionModelDefinition(
        name=name or Path(spec_file).stem,
        terms=terms,
        coefficients=read_coefficients(coef_path) if coef_path is not None else {},
        float32_coefficients=coef_path is not None,
        constants=constants or {},
        objects={k: list(v) for k, v in (objects or {}).items()},
        skim_accessors=list(skim_accessors),
        source_files=[spec_path.name] + ([coef_path.name] if coef_path is not None else []),
    )
