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

Skims are declared like a table named ``skims``: ``needs={"skims": ["DIST",
"SOVTOLL_TIME__AM"]}``.
"""

from __future__ import annotations

import logging

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .seams import Engine
from .store import Store

Needs = dict[str, list[str]]
Outputs = dict[str, dict[str, object]]
SKIMS = "skims"  # the name skims are declared under in ``needs``


class ContractError(Exception):
    """A component read or wrote something it did not declare."""


@dataclass(frozen=True)
class StepRecord:
    """
    What a step left behind, as 1.x's checkpoint records it: when it
    finished, and each table it wrote with the table's columns then.
    """

    step: str
    finished: datetime
    tables: dict[str, list[str]]
    skims: tuple[str, ...] = ()  # the skims the step looked up, as 1.x records them (computed steps)


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
    # Optional: work out ``provides`` from the columns available at this point
    # (for a step that returns a whole table whose columns depend on earlier
    # steps, e.g. tours with rows added). The runner sets ``provides`` to it.
    resolve_provides: Callable[[Mapping[str, set[str]]], Needs] | None = None
    description: str = ""
    # anything component-specific worth inspecting (e.g. generated code)
    info: dict = field(default_factory=dict)

    def needs_given(self, available: Mapping[str, set[str]]) -> Needs:
        if self.resolve_needs is not None:
            return self.resolve_needs(available)
        return self.needs

    def provides_given(self, available: Mapping[str, set[str]]) -> Needs:
        if self.resolve_provides is not None:
            return self.resolve_provides(available)
        return self.provides


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
        skims=None,
        warnings: list | None = None,
        output_dir: Path | None = None,
        history: list | None = None,
        skim_names: tuple = (),
    ):
        # where steps write their files (output steps: tables, trip matrices, ...)
        self.output_dir = Path(output_dir) if output_dir else None
        # the steps run before this one, in order (``StepRecord``s; read only)
        self.history = tuple(history or ())
        self._skim_names = tuple(skim_names)  # every skim the run has (skim usage)
        self.skims_read: set[str] = set()  # the skims looked up, as 1.x records them (see skim)
        self.component = component
        self.step = component.name
        # problems worth a look that don't stop the step (see ``warn``)
        self.warnings = warnings if warnings is not None else []
        self._store = store
        self._needs = {t: set(cols) for t, cols in needs.items()}
        self.engine = engine
        self.random = engine.random(component.name, base_seed)
        self.tracer = tracer
        self._skims = skims

    def _check(self, table: str, column: str | None = None) -> None:
        if table not in self._needs:
            raise ContractError(
                f"step '{self.step}' read table '{table}' but did not declare it in needs"
            )
        if column is not None and column not in self._needs[table]:
            raise ContractError(
                f"step '{self.step}' read '{table}.{column}' but did not declare it in needs"
            )

    def create_table(self, table) -> None:
        """
        Add a table the step makes (e.g. reading its input), so the rest of the
        step can read it. It replaces any table of that name; the step still
        returns it (whole) among its outputs.
        """
        self._store.add(table)
        self._needs[table.name] = set(table.column_names)

    def warn(self, message: str) -> None:
        """Report something worth a look that doesn't stop the step: collected in the
        run's report and its run_manifest.json, and logged."""
        self.warnings.append({"step": self.step, "message": message})
        logging.getLogger("asim_core").warning(f"{self.step}: {message}")

    def has(self, table: str) -> bool:
        """True if the step declared it reads this table."""
        return table in self._needs

    def table_frame(self, table: str):
        """A declared table's declared columns, in the table's order, as a pandas DataFrame
        (for 1.x code that works on whole tables, e.g. adding rows)."""
        self._check(table)
        t = self._store[table]
        return t.to_pandas([c for c in t.column_names if c in self._needs[table]])

    def declared(self, table: str) -> list[str]:
        """The columns of a table the step declared it reads."""
        self._check(table)
        return sorted(self._needs[table])

    def ids(self, table: str) -> np.ndarray:
        """The row IDs of a table the component declared."""
        self._check(table)
        return self._store[table].index

    def output_path(self, name: str) -> Path:
        """A file in the run's output folder (the folders made if needed)."""
        if self.output_dir is None:
            raise ContractError(f"step '{self.step}' writes files, but the runner has no output folder")
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)  # also a subfolder (summarize/...)
        return path

    def skim_names(self) -> tuple[str, ...]:
        """Every skim the run has, read or not (skim usage reports)."""
        return self._skim_names

    def index_name(self, table: str) -> str | None:
        """The name of a declared table's row IDs (e.g. zone_id)."""
        self._check(table)
        return self._store[table].index_name

    def table_names(self) -> list[str]:
        """The tables there are now, in the order they were made (output steps write them all)."""
        return list(self._store.tables)

    def column(self, table: str, column: str):
        """A declared column, in the table's row order."""
        self._check(table, column)
        return self._store[table][column]

    def column_for(self, table: str, column: str, ids) -> object:
        """A declared column of ``table``, looked up for the given IDs (a join without a merged copy)."""
        self._check(table, column)
        t = self._store[table]
        return self.engine.lookup(t[column], t.positions(ids))

    def positions(self, table: str, ids) -> np.ndarray:
        """Row positions in a declared table for the given IDs (raises if an ID is missing)."""
        self._check(table)
        return self._store[table].positions(ids)

    def column_at(self, table: str, column: str, positions) -> object:
        """A declared column's values at the given row positions (see ``positions``)."""
        self._check(table, column)
        return self.engine.lookup(self._store[table][column], positions)

    def skim_positions(self, zone_ids) -> np.ndarray:
        """Matrix row/column positions for zone IDs (use with ``skim``)."""
        self._check(SKIMS)
        return self._skims.positions(zone_ids)

    def zone_pairs(self, orig_positions, dest_positions):
        """Origin/destination positions to look up several skims for (see ``skim``):
        what the lookups work out from them is shared (``skims.ZonePairs``)."""
        from .skims import ZonePairs

        self._check(SKIMS)
        return ZonePairs(orig_positions, dest_positions, self._skims.shape[1])

    def skim(self, name: str, orig_positions, dest_positions=None, by_period: str | None = None) -> np.ndarray:
        """Values of a declared skim for origin/destination positions, or for
        ``zone_pairs`` (then ``dest_positions`` is not given). ``by_period``:
        the skim's name without its period, when each row has its own period
        (1.x's 3-d lookup), for the skim usage record."""
        from .skims import ZonePairs

        self._check(SKIMS, name)
        # skim usage as 1.x's skim_dict records it: the name for a lookup by each row's period,
        # (name, period) for one period's skim named in full, the name for a skim without periods
        self.skims_read.add(by_period if by_period is not None else name)
        if isinstance(orig_positions, ZonePairs):
            pairs = orig_positions
            return self.engine.skim_lookup(self._skims[name], pairs.orig, pairs.dest, pairs=pairs)
        return self.engine.skim_lookup(self._skims[name], orig_positions, dest_positions)
