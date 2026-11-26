"""
Annotations after a model: columns worked out for whole tables once a step
has made its choices (1.x's ``annotate_persons``, ``annotate_households``, ...
settings, run by ``expressions.annotate_tables``).

Each annotation is an assignment list evaluated on one table (persons,
households, ...), in 1.x's order (trips, tours, vehicles, persons,
households), so a households annotation sees the persons annotation's
results. Formulas see the table's own columns (including the step's new
ones), other tables looked up by ID (1.x's ``TABLES``), skims by zone
(``skim_dict.lookup``), the global constants and ``setting()``.

A result for a column the table already has replaces it, except where the
new value is missing (1.x uses pandas ``update``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..component import SKIMS, StepContext
from ..spec.definition import AssignmentSpec, BoundAssignments, LanguageScope
from ..spec.legacy import find_config_file, load_assignment_spec, load_global_constants
from ..spec.translate import INDEX
from .runtime import TableLookup, ZoneSkims, plain

ORDER = ["trips", "tours", "vehicles", "persons", "households"]  # 1.x: least to most aggregated
ROLE = {"trips": "trip", "tours": "tour", "vehicles": "vehicle", "persons": "person", "households": "household"}
SKIM_ACCESSOR = "skim_dict"


@dataclass
class Annotation:
    table: str
    spec: AssignmentSpec
    tables: list[str] = field(default_factory=list)  # other tables read by ID
    index_name: str | None = None                    # the table's ID column (model files)


def legacy_annotations(configs_dirs: Sequence[str | Path], settings: Mapping, top: Mapping,
                       step: str) -> list[Annotation]:
    """The ``annotate_<table>`` settings of a 1.x model settings file, in 1.x's order."""
    constants = dict(load_global_constants([Path(d) for d in configs_dirs]))
    out = []
    for table in ORDER:
        a = settings.get(f"annotate_{table}")
        if not a:
            continue
        spec_file = str(a["SPEC"])
        spec_file = spec_file if spec_file.endswith(".csv") else spec_file + ".csv"
        tables = list(a.get("TABLES") or [])
        spec = load_assignment_spec(configs_dirs, spec_file, constants, skim_accessors=[SKIM_ACCESSOR],
                                    name=f"{step}_annotate_{table}", tables=tables, settings=top,
                                    df_names=[a.get("DF", table)])
        out.append(Annotation(table, spec, tables))
    return out


def model_file_annotation(table: str, assignments, tables: list[str], constants: dict, top: Mapping,
                          step: str, source: str, index_name: str) -> Annotation:
    """An annotation in asim_core's language: ``<role>.<column>`` for the table, ``T.column[ids]`` for others."""
    scope = LanguageScope(roles=[ROLE[table]], tables=tables, skim_lookup=SKIM_ACCESSOR)
    spec = AssignmentSpec(name=f"{step}_annotate_{table}", assignments=assignments, constants=constants,
                          tables=tables, settings=dict(top), language="asim_core", scope=scope,
                          source_files=[source])
    return Annotation(table, spec, tables, index_name)


@dataclass
class BoundAnnotation:
    annotation: Annotation
    bound: BoundAssignments
    columns: list[str]   # the table's columns read (from the store or the step's results)
    replaced: list[str] = field(default_factory=list)  # targets the table already has (updated)


def bind(annotations: Sequence[Annotation], available: Mapping[str, set[str]],
         made: Mapping[str, set[str]]) -> tuple[list[BoundAnnotation], dict[str, set[str]], dict[str, list[str]]]:
    """
    Bind each annotation to the run's columns, in order. ``made``: columns
    the step itself writes before annotating. Returns the bound annotations,
    what they read from the store (needs) and what they add (provides).
    """
    made = {t: set(c) for t, c in made.items()}
    needs: dict[str, set[str]] = {}
    provides: dict[str, list[str]] = {}
    out = []
    for a in annotations:
        cols = set(available.get(a.table, set())) | made.get(a.table, set())
        if a.index_name:
            cols.add(a.index_name)
        tables = {t: set(available.get(t, set())) | made.get(t, set()) for t in a.tables}
        if a.spec.language == "asim_core":
            from .roles import check_roles

            b = a.spec.bind(cols, tables)
            check_roles(a.spec.name, b.translated, {ROLE[a.table]: cols})
        else:
            b = a.spec.bind(cols, tables)
        b.draws_per_call  # raises now, not mid-run, if the spec mixes how it draws
        read = sorted(b.columns_used() - {INDEX})
        needs.setdefault(a.table, set()).update(c for c in read if c not in made.get(a.table, set())
                                                and c != a.index_name)
        for t, c in b.tables_used():
            if c not in made.get(t, set()):
                needs.setdefault(t, set()).add(c)
        skims = {name for _, name, _ in b.skim_uses()}
        if skims:
            needs.setdefault(SKIMS, set()).update(skims)
        replaced = [t for t in b.targets if t in available.get(a.table, set()) and t not in made.get(a.table, set())]
        needs.setdefault(a.table, set()).update(replaced)
        for target in b.targets:
            if target not in provides.setdefault(a.table, []):
                provides[a.table].append(target)
            made.setdefault(a.table, set()).add(target)
        out.append(BoundAnnotation(a, b, read, replaced))
    return out, needs, provides


def run(ctx: StepContext, bound: Sequence[BoundAnnotation], made: dict[str, dict[str, np.ndarray]]) -> None:
    """Evaluate the annotations in order, adding their results to ``made`` (table -> column -> values)."""
    for ba in bound:
        a = ba.annotation
        ids = ctx.ids(a.table)
        own = made.setdefault(a.table, {})
        c = {}
        for col in ba.columns:
            if col in own:
                c[col] = own[col]
            elif col == a.index_name:
                c[col] = ids
            else:
                c[col] = plain(ctx.column(a.table, col))
        c[INDEX] = ids
        if a.index_name:
            c.setdefault(a.index_name, ids)
        draws = None
        if ba.bound.n_random:
            # rng.lognormal_for_df on the table's own stream (its index names the channel)
            if ba.bound.draws_per_call:  # one draw per row per call, as 1.x's broadcast=False
                draws = np.column_stack([np.asarray(ctx.random.normal(a.table, ids, 1)).reshape(len(ids))
                                         for _ in range(ba.bound.n_random)])
            else:
                draws = np.asarray(ctx.random.normal(a.table, ids, ba.bound.n_random)).reshape(len(ids), -1)
        with np.errstate(all="ignore"):  # 1.x logs numpy warnings and carries on
            v = ba.bound.function(c, ZoneSkims(ctx), len(ids), TableLookup(ctx, made), draws, {})
        for target in ba.bound.targets:
            old = own.get(target)
            if old is None and target in ba.replaced:
                old = plain(ctx.column(a.table, target))
            own[target] = _update(old, np.asarray(v[target]))


def _update(old, new):
    """pandas DataFrame.update: the new value, except where it is missing."""
    if old is None:
        return new
    if new.dtype.kind == "f":
        missing = np.isnan(new)
        if missing.any():
            new = np.where(missing, old, new)
    return new


def load_preprocessor(configs_dirs, settings, constants, top, step, label, skim_accessors=(), df_aliases=()):
    """A model's ``preprocessor`` setting (1.x: SPEC, DF, TABLES) as an assignment spec and the
    tables it reads; (None, []) if there is none. ``skim_accessors``: skim wrappers the step
    gives the preprocessor besides ``skim_dict`` (e.g. ``od_skims``); ``df_aliases``: other names
    the rows go by in the formulas (summarize: ``trips_merged``)."""
    if not settings:
        return None, []
    if isinstance(settings, list):
        raise NotImplementedError("more than one preprocessor is not supported yet")
    tables = list(settings.get("TABLES") or [])
    spec_file = str(settings["SPEC"])
    spec = load_assignment_spec(
        configs_dirs, spec_file if spec_file.endswith(".csv") else spec_file + ".csv",
        constants, skim_accessors=[SKIM_ACCESSOR, *skim_accessors], name=f"{step}_{label}",
        tables=tables, settings=top, df_names=[settings.get("DF", "df"), *df_aliases],
    )
    return spec, tables


def find_spec(configs_dirs, name: str) -> Path:
    return find_config_file([Path(d) for d in configs_dirs], name if name.endswith(".csv") else name + ".csv")
