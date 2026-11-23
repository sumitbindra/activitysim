"""
asim_core's model files: one YAML file per model, plus CSV tables.

    kind: auto_ownership          # what the model is; sets the tables formulas see
    terms: auto_ownership_terms.csv
    coefficients: auto_ownership_coefficients.csv
    ...every option written out, with 1.x's behaviour as the value...

The YAML is checked by the Pydantic class of its kind when it loads (a
misspelled option is an error, not ignored). Formulas are in asim_core's
language (language.py). The CSV tables:

* **term tables**: ``Label, Description, Formula`` and one column per
  alternative (or segment, or a single ``coefficient`` column), holding a
  coefficient name or a number; blank where a term doesn't apply. A row
  whose formula is ``name = formula`` defines a value for later rows.
* **coefficients**: ``coefficient_name`` and either ``value`` or one column
  per segment (e.g. tour purpose), so one table replaces 1.x's coefficients
  file and coefficient template;
* **derived values**: ``Description, Name, Formula``, computed in order
  (1.x's preprocessors). Names starting with ``_`` are only used by later rows.

**Inheriting.** ``extends: ../base/auto_ownership.yaml`` starts from another
model file and changes what this one lists (options are merged; anything
else listed replaces the base's). ``coefficient_values: {name: value}``
changes single coefficients. File names are relative to the YAML file that
names them.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from .definition import Assignment, Coefficient
from .language import split_assignment
from .legacy import read_text

FILE_SUFFIXES = (".csv", ".yaml", ".yml")
TERM_META = ("Label", "Description", "Formula")


class ModelFileError(ValueError):
    pass


# --- reading YAML with inheritance -------------------------------------------------------


def _absolute_paths(data, base: Path):
    """File names (strings ending .csv / .yaml) made relative to ``base``'s folder."""
    if isinstance(data, dict):
        return {k: _absolute_paths(v, base) for k, v in data.items()}
    if isinstance(data, list):
        return [_absolute_paths(v, base) for v in data]
    if isinstance(data, str) and data.lower().endswith(FILE_SUFFIXES) and "\n" not in data:
        p = Path(data)
        if p.is_absolute():
            return str(p)
        here = (base.parent / p).resolve()
        # a name not next to the model file is looked up in the configs folders (locate)
        return str(here) if here.exists() else data
    return data


def locate(name: str, configs_dirs) -> str:
    """A file a model file names: next to it, or else in the first configs folder that has it."""
    p = Path(name)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"file '{name}' not found")
        return str(p)
    for d in configs_dirs:
        if (Path(d) / p).exists():
            return str((Path(d) / p).resolve())
    raise FileNotFoundError(f"file '{name}' not found next to the model file or in the configs folders "
                            f"{[str(d) for d in configs_dirs]}")


def _merge(base: dict, top: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in top.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k in ("options", "constants",
                                                                           "coefficient_values", "logsums"):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def read_model_yaml(path: str | Path, _seen: tuple = ()) -> dict:
    """A model file's settings, with ``extends`` resolved and file names made absolute."""
    path = Path(path).resolve()
    if path in _seen:
        raise ModelFileError(f"{path}: extends itself (through {' -> '.join(str(p) for p in _seen)})")
    data = yaml.safe_load(read_text(path)) or {}
    if not isinstance(data, dict):
        raise ModelFileError(f"{path}: expected settings (name: value) at the top level")
    data = _absolute_paths(data, path)
    parent = data.pop("extends", None)
    if parent:
        data = _merge(read_model_yaml(parent, _seen + (path,)), data)
    data.setdefault("source_files", [])
    data["source_files"] = list(data["source_files"]) + [str(path)]
    return data


def is_model_file(path: str | Path) -> bool:
    """True for asim_core's model files (they say their ``kind``, or extend one); False for 1.x settings files."""
    try:
        data = yaml.safe_load(read_text(path)) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return False
    return isinstance(data, dict) and ("kind" in data or "extends" in data)


def validate(cls: type[BaseModel], data: dict, path) -> BaseModel:
    try:
        return cls(**data)
    except ValidationError as e:
        lines = [f"{path}:"]
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            msg = err["msg"]
            if err["type"] == "extra_forbidden":
                msg = "not a setting of this kind of model"
            lines.append(f"  {loc}: {msg}")
        raise ModelFileError("\n".join(lines)) from None


class ModelFile(BaseModel):
    """Settings every model file has."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    description: str = ""
    constants: dict[str, Any] = {}
    # single coefficients changed (e.g. by a region's file that extends a base model)
    coefficient_values: dict[str, float] = {}
    source_files: list[str] = []


# --- CSV tables -----------------------------------------------------------------------------


def _read_csv(path: str | Path) -> pd.DataFrame:
    # numbers read exactly (asim1 findings, #11); text kept as written
    return pd.read_csv(path, comment="#", dtype=str, keep_default_na=False, skipinitialspace=False)


def read_terms(path: str | Path):
    """
    A term table: [(label, description, formula, target, {column: cell})].
    ``target`` is set for rows written ``name = formula``.
    """
    df = _read_csv(path)
    missing = [c for c in ("Formula",) if c not in df.columns]
    if missing:
        raise ModelFileError(f"{path}: a term table needs a Formula column")
    value_cols = [c for c in df.columns if c not in TERM_META]
    rows = []
    for _, r in df.iterrows():
        text = r["Formula"].strip()
        if not text:
            continue
        target, formula = split_assignment(text)
        cells = {c: r[c].strip() for c in value_cols if r[c].strip() != ""}
        rows.append((r.get("Label", "").strip() or None, r.get("Description", "").strip(), formula, target, cells))
    return rows, value_cols


def read_coefficients_table(path: str | Path, overrides: dict[str, float] | None = None):
    """
    Coefficients: {name: Coefficient} from a ``value`` column, or
    {segment: {name: Coefficient}} from one column per segment.
    """
    df = _read_csv(path)
    if "coefficient_name" not in df.columns:
        raise ModelFileError(f"{path}: needs a coefficient_name column")
    names = df["coefficient_name"].str.strip()
    dupes = sorted(set(names[names.duplicated()]))
    if dupes:
        raise ModelFileError(f"{path}: coefficients listed twice: {dupes}")
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise ModelFileError(f"{path}: coefficient_values names unknown coefficients {unknown}")
    value_cols = [c for c in df.columns if c not in ("coefficient_name", "constrain", "Description")]

    def number(cell, name, col):
        try:
            return float(cell)
        except ValueError:
            raise ModelFileError(f"{path}: {name}, {col}: '{cell}' is not a number") from None

    if value_cols == ["value"]:
        return {n: Coefficient(value=overrides.get(n, number(v, n, "value")))
                for n, v in zip(names, df["value"])}, None
    by_segment = {}
    for col in value_cols:
        by_segment[col] = {n: Coefficient(value=overrides.get(n, number(v, n, col)))
                           for n, v in zip(names, df[col])}
    return None, by_segment


def read_derived(path: str | Path) -> list[Assignment]:
    df = _read_csv(path)
    for c in ("Name", "Formula"):
        if c not in df.columns:
            raise ModelFileError(f"{path}: a derived values table needs Name and Formula columns")
    out = []
    for _, r in df.iterrows():
        name, formula = r["Name"].strip(), r["Formula"].strip()
        if not name and not formula:
            continue
        if not name or not formula:
            raise ModelFileError(f"{path}: a row needs both a Name and a Formula ({name or formula})")
        out.append(Assignment(target=name, formula=formula, description=r.get("Description", "").strip()))
    return out


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])


def write_terms(path: Path, rows: list[dict], value_cols: list[str]) -> None:
    """rows: {label, description, formula, target, cells}."""
    out = []
    for r in rows:
        formula = f"{r['target']} = {r['formula']}" if r.get("target") else r["formula"]
        out.append([r.get("label") or "", r.get("description") or "", formula]
                   + [r["cells"].get(c, "") for c in value_cols])
    _write_csv(path, list(TERM_META) + value_cols, out)


def write_coefficients(path: Path, values: dict[str, float] | None = None,
                       by_segment: dict[str, dict[str, float]] | None = None) -> None:
    if values is not None:
        _write_csv(path, ["coefficient_name", "value"], [[n, repr(float(v))] for n, v in values.items()])
        return
    segs = list(by_segment)
    names = list(dict.fromkeys(n for s in segs for n in by_segment[s]))
    _write_csv(path, ["coefficient_name"] + segs,
               [[n] + [repr(float(by_segment[s][n])) for s in segs] for n in names])


def write_derived(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """rows: (description, name, formula)."""
    _write_csv(path, ["Description", "Name", "Formula"], [list(r) for r in rows])


class _Dumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _str_presenter)


def write_yaml(path: Path, data: dict, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(data, Dumper=_Dumper, sort_keys=False, default_flow_style=False, width=100)
    path.write_text((header.rstrip() + "\n\n" if header else "") + text, encoding="utf-8")


def relative(path: str | Path, to: Path) -> str:
    """A file name as written in a model file next to ``to``."""
    p = Path(path)
    try:
        return str(p.relative_to(to.parent))
    except ValueError:
        return str(p)
