"""
The component contract.

A component is one model step. It declares:

* ``name``      the step name. Random streams are seeded from it, so use the
                ActivitySim 1.x step name when matching 1.x results.
* ``needs``     the tables and columns it reads
* ``provides``  the tables and columns it writes
* ``run``       a function that receives a ``StepContext`` and returns its
                outputs as ``{table: {column: values}}``

The context only hands out columns the component declared, and the runner
only accepts outputs the component declared. That keeps every dependency
between steps visible, so a region can add or drop components and get a clear
error at startup if something a later step needs is missing.

Components can be written in anything that can be called from Python; the
contract is only about data in and data out.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .seams import Engine
from .store import Store

Needs = dict[str, list[str]]
Outputs = dict[str, dict[str, object]]


class ContractError(Exception):
    """A component read or wrote something it did not declare."""


class MissingInputError(Exception):
    """A component needs a column that no earlier step or input provides."""


@dataclass
class Component:
    name: str
    run: Callable[[StepContext], Outputs]
    needs: Needs = field(default_factory=dict)
    provides: Needs = field(default_factory=dict)
    # Optional: work out ``needs`` from the columns available at this point in
    # the run (used by components whose inputs come from a joined view of
    # several tables). Receives {table: set(columns)}; returns ``Needs``.
    resolve_needs: Callable[[Mapping[str, set[str]]], Needs] | None = None
    description: str = ""
    # anything component-specific worth inspecting (e.g. generated code)
    info: dict = field(default_factory=dict)

    def needs_given(self, available: Mapping[str, set[str]]) -> Needs:
        if self.resolve_needs is not None:
            return self.resolve_needs(available)
        return self.needs


class Tracer:
    """Writes detailed values for a few chosen rows (e.g. one household)."""

    def __init__(self, out_dir: Path | None, ids: Mapping[str, list] | None):
        self.out_dir = Path(out_dir) if out_dir else None
        self.ids = {k: list(v) for k, v in (ids or {}).items()}

    def targets(self, table: str, ids: np.ndarray) -> np.ndarray:
        """Positions in ``ids`` of the rows being traced for ``table``."""
        wanted = self.ids.get(table)
        if not wanted or self.out_dir is None:
            return np.array([], dtype=np.int64)
        return np.flatnonzero(np.isin(ids, wanted))

    def write(self, step: str, label: str, header: list[str], rows: list[list]) -> Path | None:
        if self.out_dir is None:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{step}.{label}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return path


class StepContext:
    """
    What a component sees while it runs: its declared columns, the random
    streams for this step, the tracer, and ``engine``, the implementations
    chosen for each swappable piece (see ``seams.py``).
    """

    def __init__(
        self,
        component: Component,
        store: Store,
        needs: Needs,
        engine: Engine,
        tracer: Tracer,
        base_seed: int = 0,
    ):
        self.component = component
        self.step = component.name
        self._store = store
        self._needs = {t: set(cols) for t, cols in needs.items()}
        self.engine = engine
        self.random = engine.random(component.name, base_seed)
        self.tracer = tracer

    def _check(self, table: str, column: str | None = None) -> None:
        if table not in self._needs:
            raise ContractError(
                f"step '{self.step}' read table '{table}' but did not declare it in needs"
            )
        if column is not None and column not in self._needs[table]:
            raise ContractError(
                f"step '{self.step}' read '{table}.{column}' but did not declare it in needs"
            )

    def ids(self, table: str) -> np.ndarray:
        """The row IDs of a table the component declared."""
        self._check(table)
        return self._store[table].index

    def column(self, table: str, column: str):
        """A declared column, in the table's row order."""
        self._check(table, column)
        return self._store[table][column]

    def column_for(self, table: str, column: str, ids) -> object:
        """A declared column of ``table``, looked up for the given IDs (a join without a merged copy)."""
        self._check(table, column)
        t = self._store[table]
        return self.engine.lookup(t[column], t.positions(ids))
