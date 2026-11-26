"""
Table roles: the names formulas in asim_core's language use for tables
(``person.age``, ``home_zone.TOTEMP``).

A step sees its choosers through a ChooserView (the chooser table plus
joined tables). Each role names one or more of those tables; a column
belongs to the role of the table the view takes it from (when a name is in
several tables, the first in the view's order wins, as in 1.x's merged
tables).
"""

from __future__ import annotations

from collections.abc import Mapping

from ..spec.definition import check_columns
from .simple_choice import ChooserView

# the roles of the views the tour steps use
TOUR_ROLES = {
    "tour": ["tours"],
    "person": ["persons"],
    "household": ["households"],
    "home_zone": ["land_use", "accessibility"],
}
HOUSEHOLD_ROLES = {
    "household": ["households"],
    "home_zone": ["land_use", "accessibility"],
}
PERSON_ROLES = {
    "person": ["persons"],
    "household": ["households"],
    "home_zone": ["land_use", "accessibility"],
}


def role_columns(view: ChooserView, roles: Mapping[str, list[str]],
                 available: Mapping[str, set[str]], extra: Mapping[str, set[str]] | None = None
                 ) -> dict[str, set[str]]:
    """
    Role -> the columns formulas can read under it. ``extra`` adds columns
    the step makes itself (e.g. the alternatives' columns, role ``alt``);
    they come before the view's columns of the same name, as the step's rows
    hold them.
    """
    table_role = {t: r for r, ts in roles.items() for t in ts}
    out: dict[str, set[str]] = {r: set() for r in roles}
    taken: set[str] = set()
    for role, cols in (extra or {}).items():
        out.setdefault(role, set()).update(cols)
        taken |= set(cols)
    for col in view.columns(available):
        if col in taken:
            continue
        src = view._source(col, available)
        if src in table_role:
            out[table_role[src]].add(col)
    return out


def role_finder(columns_by_role: Mapping[str, set[str]]):
    """column -> its role (for converting 1.x formulas)."""
    lookup = {}
    for role, cols in columns_by_role.items():
        for c in cols:
            lookup.setdefault(c, role)
    return lookup.get


def check_roles(name: str, translated, columns_by_role: Mapping[str, set[str]]) -> None:
    problems = check_columns([t for t in translated if t is not None], columns_by_role)
    if problems:
        raise ValueError(f"{name}: {len(problems)} problem(s) with table columns:\n  " + "\n  ".join(problems))


def context_roles(view: ChooserView, roles: Mapping[str, list[str]], available: Mapping[str, set[str]],
                  df_columns: set[str], derived=("tour_purpose", "out_period", "in_period")
                  ) -> dict[str, set[str]]:
    """
    The roles of the chooser rows a step hands to another model (e.g. tour
    mode choice for logsums): the view's columns under their tables' roles;
    anything else in the rows (the step's own values, an option's times, a
    candidate destination) is the tour's.
    """
    by_role = role_columns(view, roles, available)
    out = {r: {c for c in cs if c in df_columns} for r, cs in by_role.items()}
    placed = set().union(*out.values()) if out else set()
    out.setdefault("tour", set()).update(c for c in df_columns if c not in placed and c not in derived)
    return out
