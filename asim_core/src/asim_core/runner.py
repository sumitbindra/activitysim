"""
The runner: runs a list of steps in order.

Each step is either

* **computed** by a registered component, or
* **replayed**: its outputs are copied from an ActivitySim 1.x run, so a model
  can run end to end before every step has been ported.

Before anything runs, ``check`` walks the step list and confirms that every
column each component needs is provided by the starting data or an earlier
step. Problems are reported all at once, before any work is done.

After each computed step, the columns it produced are saved as Parquet under
``<out_dir>/state/<step>/<table>.parquet``. Every step is profiled.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .asim1 import Asim1State
from .compare import ColumnComparison, compare_column
from .component import SKIMS, Component, ContractError, MissingInputError, StepContext, Tracer
from .profiler import Profiler
from .seams import Engine
from .store import Store, Table

COMPUTED = "computed"
REPLAYED = "replayed"


class StartupCheckError(Exception):
    pass


@dataclass
class RunReport:
    plan: list[tuple[str, str]]
    profiler: Profiler
    # comparisons of our computed columns against 1.x's copies of them,
    # found when later 1.x tables are replayed
    carried_over: list[ColumnComparison] = field(default_factory=list)


class Runner:
    def __init__(
        self,
        components: Iterable[Component],
        out_dir: str | Path,
        replay: Asim1State | None = None,
        trace_ids: dict[str, list] | None = None,
        base_seed: int = 0,
        save_outputs: bool = True,
        engine: Engine | None = None,
        step_engines: dict[str, Engine] | None = None,
        skims=None,
    ):
        """
        ``engine`` picks the implementation of each swappable piece for all
        steps (default: the reference implementations); ``step_engines``
        overrides it for individual steps. ``skims`` is a ``SkimStore`` for
        components that declare skims.
        """
        self.skims = skims
        self.components = {c.name: c for c in components}
        self.out_dir = Path(out_dir)
        self.replay = replay
        self.tracer = Tracer(self.out_dir / "trace", trace_ids)
        self.base_seed = base_seed
        self.save_outputs = save_outputs
        self.engine = engine or Engine()
        self.step_engines = dict(step_engines or {})
        self._zones_checked = False

    # --- planning and startup checks ---------------------------------------------

    def plan(self, steps: Iterable[str]) -> list[tuple[str, str]]:
        plan = []
        for s in steps:
            if s in self.components:
                plan.append((s, COMPUTED))
            elif self.replay is not None and s in self.replay.steps:
                plan.append((s, REPLAYED))
            else:
                raise StartupCheckError(
                    f"step '{s}' has no component and no ActivitySim 1.x output to replay"
                )
        return plan

    def check(self, plan: list[tuple[str, str]], store: Store) -> dict[str, dict]:
        """
        Confirm every component's inputs are available when it runs.
        Returns the resolved needs per computed step; raises listing all problems.
        """
        available = {t: set(cols) for t, cols in store.columns().items()}
        if self.skims is not None:
            available[SKIMS] = set(self.skims.names)
        computed_cols: dict[str, set[str]] = defaultdict(set)
        resolved = {}
        problems = []
        for step, mode in plan:
            if mode == COMPUTED:
                comp = self.components[step]
                try:
                    needs = comp.needs_given(available)
                except MissingInputError as err:
                    problems.append(f"{step}: {err}")
                    needs = {}
                for table, cols in needs.items():
                    missing = [c for c in cols if c not in available.get(table, set())]
                    if table == SKIMS and self.skims is None:
                        problems.append(f"{step}: needs skims, but the runner was given no skim store")
                    elif table not in available:
                        problems.append(f"{step}: needs table '{table}', which does not exist yet")
                    elif missing:
                        problems.append(f"{step}: needs {table}.{missing}, not provided by any earlier step")
                resolved[step] = needs
                for table, cols in comp.provides.items():
                    available.setdefault(table, set()).update(cols)
                    computed_cols[table].update(cols)
            else:
                cols_after = self.replay.columns_at(step)
                for table in self.replay.written_at(step):
                    available[table] = set(cols_after[table]) | computed_cols.get(table, set())
                for table in list(available):
                    if table not in cols_after and table not in computed_cols and table != SKIMS:
                        del available[table]
        if problems:
            raise StartupCheckError(
                f"{len(problems)} problem(s) found before running:\n  " + "\n  ".join(problems)
            )
        return resolved

    # --- running -----------------------------------------------------------------

    def run(self, steps: Iterable[str], store: Store, profiler: Profiler | None = None,
            start_note: str = "") -> RunReport:
        plan = self.plan(steps)
        resolved = self.check(plan, store)
        profiler = profiler or Profiler()
        report = RunReport(plan=plan, profiler=profiler)
        # table -> column -> IDs of the rows asim_core computed (a later
        # replayed 1.x step may add rows, whose values then come from 1.x)
        computed_cols: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        for step, mode in plan:
            with profiler.step(step, mode):
                if mode == COMPUTED:
                    self._run_component(step, store, resolved[step], computed_cols)
                else:
                    report.carried_over += self._replay_step(step, store, computed_cols)

        manifest = {
            "start": start_note,
            "steps": [
                {"step": s, "mode": m,
                 **({"engine": self.step_engines.get(s, self.engine).changed or "reference"}
                    if m == COMPUTED else {})}
                for s, m in plan
            ],
            "computed_columns": {t: sorted(c) for t, c in computed_cols.items()},
        }
        (self.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        profiler.write_csv(self.out_dir / "profile.csv")
        return report

    def _run_component(self, step, store: Store, needs, computed_cols) -> None:
        comp = self.components[step]
        if SKIMS in needs and not self._zones_checked and "land_use" in store:
            from .skims import check_zone_order

            problem = check_zone_order(self.skims, store["land_use"])
            if problem:
                raise StartupCheckError(f"{step}: {problem}")
            self._zones_checked = True
        engine = self.step_engines.get(step, self.engine)
        ctx = StepContext(comp, store, needs, engine, self.tracer, self.base_seed, skims=self.skims)
        outputs = comp.run(ctx) or {}
        declared = {(t, c) for t, cols in comp.provides.items() for c in cols}
        produced = {(t, c) for t, cols in outputs.items() for c in cols}
        if produced != declared:
            raise ContractError(
                f"step '{step}' produced {sorted(produced)} but declared {sorted(declared)}"
            )
        for table, cols in outputs.items():
            for column, values in cols.items():
                store[table].set_column(column, values)
                computed_cols[table][column] = store[table].index.copy()
            if self.save_outputs:
                path = self.out_dir / "state" / step / f"{table}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                store[table].to_pandas(list(cols)).to_parquet(path)

    def _replay_step(self, step, store: Store, computed_cols) -> list[ColumnComparison]:
        comparisons = []
        for table in self.replay.written_at(step):
            df = self.replay.read(table, step)
            new = Table.from_pandas(table, df)
            old = store.tables.get(table)
            for column, ids in sorted(computed_cols.get(table, {}).items()):
                if old is None or column not in old:
                    continue
                # rows we computed that the replayed table still has
                ids = ids[np.isin(ids, new.index)]
                ours = old[column][old.positions(ids)]
                if column in df.columns:
                    comparisons.append(
                        compare_column(step, Table(table, ids, old.index_name, {column: ours}),
                                       column, df[column].loc[ids])
                    )
                    values = new[column]
                elif len(ids) == len(new):
                    values = ours
                else:
                    raise ValueError(
                        f"replaying '{step}': table '{table}' gained rows, and 1.x has no "
                        f"'{column}' for them"
                    )
                # keep our computed values, aligned to the replayed rows
                values = _with_values(values, new.positions(ids), ours)
                new.set_column(column, values)
            store.add(new)
        still_there = set(self.replay.tables_at(step))
        for table in list(store.tables):
            if table not in still_there and table not in computed_cols:
                del store.tables[table]
        return comparisons


def _with_values(values, positions, ours):
    """``values`` with ``ours`` written at ``positions`` (categoricals keep their categories)."""
    import pandas as pd

    if isinstance(values, pd.Categorical) or isinstance(ours, pd.Categorical):
        cats = values.categories if isinstance(values, pd.Categorical) else ours.categories
        if isinstance(ours, pd.Categorical) and isinstance(values, pd.Categorical):
            cats = values.categories.union(ours.categories)
        out = pd.Categorical(np.asarray(values, dtype=object), categories=cats)
        out[positions] = np.asarray(ours, dtype=object)
        return out
    out = np.array(values, copy=True)
    out[positions] = ours
    return out
