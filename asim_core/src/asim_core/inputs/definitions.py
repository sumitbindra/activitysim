"""
Table definitions: what each input table and column means, and the rules
its values must follow.

Definitions are written in YAML and checked by the Pydantic classes below
when they load, so a misspelled rule or a wrong value type is reported with
the file and the place in it. They are never applied row by row: the checks
(checks.py) work a column at a time with NumPy.

asim_core ships the standard definitions (``standard_tables.yaml``): the
columns every region has and the core relies on. A region adds its own
columns, or changes a rule, in an ``input_tables.yaml`` in its configs
folders. A column given in both keeps the standard rules the region file
doesn't mention.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

STANDARD_FILE = Path(__file__).with_name("standard_tables.yaml")
REGION_FILE = "input_tables.yaml"

ColumnType = Literal[
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float32", "float64", "bool", "str", "category",
]


class Column(BaseModel):
    """One column: its meaning, and the rules its values must follow."""

    model_config = ConfigDict(extra="forbid")

    type: ColumnType | None = None
    description: str = ""
    unit: str | None = None
    # missing values (empty cells) are errors unless allowed
    missing_allowed: bool = False
    # errors: these stop the run
    min: float | None = None
    max: float | None = None
    values: list[int | float | str] | None = None
    unique: bool = False
    references: str | None = Field(
        None, description="'table.column': every value must be one of that column's values")
    count_of: str | None = Field(
        None, description="'table.column': equals the number of rows of that table with this row's ID")
    # warnings: reported, the run goes on
    warn_below: float | None = None
    warn_above: float | None = None
    sum_of: list[str] | None = Field(
        None, description="columns of the same table whose sum this column should equal")
    sum_tolerance: float = 0.0

    @model_validator(mode="after")
    def _consistent(self):
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) is above max ({self.max})")
        for name in ("references", "count_of"):
            target = getattr(self, name)
            if target is not None and target.count(".") != 1:
                raise ValueError(f"{name} must be written 'table.column', not '{target}'")
        if self.type in ("str", "category") and (self.min is not None or self.max is not None):
            raise ValueError("min and max need a numeric type")
        return self


class Table(BaseModel):
    """One input table."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    id: str | None = Field(None, description="the column that identifies each row")
    columns: dict[str, Column] = {}

    @model_validator(mode="after")
    def _id_defined(self):
        if self.id is not None and self.id not in self.columns:
            raise ValueError(f"the id column '{self.id}' has no definition")
        return self


class Definitions(BaseModel):
    """All input tables, standard and regional, as one checked set."""

    model_config = ConfigDict(extra="forbid")

    tables: dict[str, Table] = {}
    source_files: list[str] = []

    @model_validator(mode="after")
    def _targets_exist(self):
        for tname, table in self.tables.items():
            for cname, col in table.columns.items():
                for rule in ("references", "count_of"):
                    target = getattr(col, rule)
                    if target is None:
                        continue
                    t, c = target.split(".")
                    if t not in self.tables or c not in self.tables[t].columns:
                        raise ValueError(f"{tname}.{cname}: {rule} '{target}' is not a defined column")
                for other in col.sum_of or []:
                    if other not in table.columns:
                        raise ValueError(f"{tname}.{cname}: sum_of column '{other}' has no definition")
        return self

    def column(self, table: str, column: str) -> Column | None:
        t = self.tables.get(table)
        return None if t is None else t.columns.get(column)


class DefinitionError(ValueError):
    pass


def _merge(base: dict, region: dict) -> dict:
    """Region tables and columns added to (or changing) the base ones."""
    out = copy.deepcopy(base)
    for tname, tdef in (region.get("tables") or {}).items():
        tdef = tdef or {}
        target = out.setdefault("tables", {}).setdefault(tname, {})
        for key, value in tdef.items():
            if key != "columns":
                target[key] = value
        cols = target.setdefault("columns", {})
        for cname, cdef in (tdef.get("columns") or {}).items():
            cols[cname] = {**(cols.get(cname) or {}), **(cdef or {})}
    return out


def _read(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise DefinitionError(f"{path}: expected 'tables:' at the top")
    return data


def load_definitions(configs_dirs: Sequence[str | Path] = (), standard: bool = True) -> Definitions:
    """
    The standard definitions with the region's ``input_tables.yaml`` on top.
    As with every config file, the first configs folder that has one wins.
    """
    files = [STANDARD_FILE] if standard else []
    for d in configs_dirs:
        p = Path(d) / REGION_FILE
        if p.exists():
            files.append(p)
            break
    merged: dict = {}
    for p in files:
        merged = _merge(merged, _read(p))
    try:
        return Definitions(**merged, source_files=[str(p) for p in files])
    except ValidationError as e:
        lines = [f"table definitions ({', '.join(str(p) for p in files)}):"]
        for err in e.errors():
            loc = [str(x) for x in err["loc"]]
            msg = err["msg"]
            if err["type"] == "extra_forbidden":
                known = Column.model_fields if "columns" in loc[:-1] else Table.model_fields
                msg = f"'{loc[-1]}' is not a setting (settings: {', '.join(k for k in known)})"
                if " " in loc[-1]:
                    msg += "; text with a comma inside { } needs quotes"
                loc = loc[:-1]
            lines.append(f"  {'.'.join(loc)}: {msg}")
        raise DefinitionError("\n".join(lines)) from None
