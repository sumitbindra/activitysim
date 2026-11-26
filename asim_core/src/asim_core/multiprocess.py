"""
Running a model in several processes, as ActivitySim 1.x does.

The plan is 1.x's own settings (``multiprocess``, ``num_processes`` and
``multiprocess_steps`` in settings.yaml), so a region's multiprocess configs
work unchanged. Each ``multiprocess_steps`` entry is a group of steps:

* a group without ``slice`` runs in the main process (1.x's
  ``mp_initialize``, ``mp_summarize``);
* a group sliced by households runs in ``num_processes`` worker processes.
  As 1.x: process k gets every ``num_processes``-th household starting at
  the k-th, in table order; each other table follows if its index, or one of
  its columns, is the index of a table already split (persons by
  ``household_id``, tours by ``household_id``, ...); tables that don't (land
  use, accessibility) go whole to every process. Afterwards the split tables
  are put back together in process order, and the others are taken from
  process 0, so the tables come out in 1.x's order;
* a group sliced by another table (1.x's ``mp_accessibility``, by zone) runs
  once in the main process: its values don't depend on the split. Its rows
  are then put in the order 1.x's split leaves them.

The checkpoints a run leaves (``final_checkpoints.csv``, the data dictionary)
are named as 1.x names them in multiprocess runs: the steps of a group run
in the main process by step, a group run in processes by the group's name.

Worker processes are started fresh on every system (``spawn``, the only way
Windows starts them): each rebuilds the steps from the configs
(``model.build_steps``), opens the skim file itself (memory-mapped: the
operating system keeps one copy for all processes), reads its tables from
Parquet files the main process wrote and writes its tables back the same
way. Nothing relies on memory inherited from the main process, so a run
behaves the same on Windows, Linux and macOS.

Every row's random numbers come from its own stream, seeded by its ID, so a
household's results don't depend on which process ran it, or with which
other households.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .component import StepRecord
from .profiler import Profiler, StepProfile
from .store import Store, Table

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class ModelFiles:
    """Everything a worker process needs to rebuild the model: plain values, so it pickles."""

    configs: tuple[str, ...]
    data_dirs: tuple[str, ...]
    skims: str | None = None  # the skim cache folder
    seams: tuple[tuple[str, str], ...] = ()
    base_seed: int = 0
    households_sample_size: int | None = None


@dataclass
class Group:
    """One of 1.x's multiprocess_steps: a name, its steps, and how its tables are split (None: not split)."""

    name: str
    steps: list[str]
    slice_tables: list[str] | None = None
    exclude: list[str] | bool | None = None
    num_processes: int = 1

    @property
    def sliced(self) -> bool:
        return bool(self.slice_tables)


@dataclass
class GroupResult:
    """What a group run in processes left: per process, its step profiles and wall time."""

    group: str
    processes: int
    wall_s: float
    process_wall_s: list[float] = field(default_factory=list)
    profiles: list[list[StepProfile]] = field(default_factory=list)


def plan_groups(settings: dict, models: Sequence[str]) -> list[Group]:
    """1.x's multiprocess_steps, each with its steps from the models list (a group runs until the next begins)."""
    mp_steps = settings.get("multiprocess_steps") or []
    if not mp_steps:
        raise ValueError("settings.yaml has multiprocess: True but no multiprocess_steps")
    num = int(settings.get("num_processes") or 1)
    starts = []
    for entry in mp_steps:
        begin = entry["begin"]
        if begin not in models:
            raise ValueError(f"multiprocess step {entry['name']} begins at {begin}, which is not in models")
        starts.append(models.index(begin))
    if starts != sorted(starts) or starts[0] != 0:
        raise ValueError("multiprocess_steps must start at the first model and be in the models' order")
    groups = []
    for i, entry in enumerate(mp_steps):
        end = starts[i + 1] if i + 1 < len(starts) else len(models)
        sl = entry.get("slice") or {}
        groups.append(Group(entry["name"], list(models[starts[i]:end]), sl.get("tables"),
                            sl.get("exclude", sl.get("except")), int(entry.get("num_processes") or num)))
    return groups


# --- splitting and putting tables back together, as 1.x ------------------------------


def slice_rules(store: Store, slice_tables: Sequence[str], exclude=None) -> dict[str, tuple]:
    """
    1.x's build_slice_rules: for each table (the slice tables first, then the
    store's tables in order) how it is split: ("primary",), ("index", source),
    ("column", column, source) or None (whole to every process).
    """
    exclude = exclude or []
    order = list(slice_tables) + [t for t in store.tables if t not in slice_tables]
    if exclude is True or exclude == "*":
        exclude = [t for t in order if t not in slice_tables]
    primary = slice_tables[0]
    ref_cols: dict[str, str] = {}  # split table -> its index name
    rules: dict[str, tuple | None] = {}
    for name in order:
        if name not in store:
            continue
        t = store[name]
        rule = None
        if name == primary:
            rule = ("primary",)
        elif name not in exclude:
            for source, index_name in ref_cols.items():  # as 1.x: the last source decides
                if t.index_name == index_name:
                    rule = ("index", source)
                else:
                    hit = next(((s, c) for s, c in ref_cols.items() if c in t.column_names), None)
                    rule = ("column", hit[1], hit[0]) if hit else None
        if rule:
            ref_cols[name] = t.index_name
        rules[name] = rule
    return rules


def split_store(store: Store, rules: dict[str, tuple | None], n: int) -> list[dict[str, np.ndarray | None]]:
    """For each process, the rows (positions) of each split table it gets; None for a table it gets whole."""
    parts: list[dict[str, np.ndarray | None]] = [{} for _ in range(n)]
    for name, rule in rules.items():
        t = store[name]
        for k in range(n):
            if rule is None:
                parts[k][name] = None
            elif rule[0] == "primary":
                if n > len(t):
                    raise ValueError(f"{name} has fewer rows ({len(t)}) than processes ({n})")
                parts[k][name] = np.arange(k, len(t), n)
            elif rule[0] == "index":
                source = store[rule[1]]
                parts[k][name] = t.positions(source.index[parts[k][rule[1]]])
            else:
                _, column, source_name = rule
                source = store[source_name]
                keep = source.index[parts[k][source_name]]
                parts[k][name] = np.flatnonzero(np.isin(np.asarray(t[column]), keep))
    return parts


def take_rows(table: Table, rows: np.ndarray | None) -> Table:
    if rows is None:
        return table
    cols = {}
    for c in table.column_names:
        v = table[c]
        cols[c] = v.take(rows) if isinstance(v, pd.Categorical) else np.asarray(v)[rows]
    return Table(table.name, table.index[rows], table.index_name, cols)


def write_tables(store: Store, folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    for name, t in store.tables.items():
        t.to_pandas().to_parquet(folder / f"{name}.parquet")
    return list(store.tables)


def read_tables(folder: Path, names: Sequence[str]) -> Store:
    store = Store()
    for name in names:
        store.add(Table.from_pandas(name, pd.read_parquet(folder / f"{name}.parquet")))
    return store


def coalesce(parts: list[Store], rules_after: dict[str, tuple | None]) -> Store:
    """1.x's coalesce_pipelines: split tables concatenated in process order, the others taken from process 0."""
    out = Store()
    first = parts[0]
    mirrored = [t for t, r in rules_after.items() if r is None]
    omnibus = [t for t, r in rules_after.items() if r is not None]
    for name in mirrored:
        out.add(first[name])
    for name in omnibus:
        frames = [p[name].to_pandas() for p in parts]
        out.add(Table.from_pandas(name, pd.concat(frames, sort=False)))
    return out


# --- the worker ------------------------------------------------------------------------


def _worker(k: int, model: ModelFiles, steps: list[str], in_dir: str, out_dir: str, names: list[str]):
    """One process: rebuild the steps, read its tables, run the steps, write its tables."""
    t0 = time.perf_counter()
    from .model import build_steps
    from .runner import Runner
    from .seams import Engine
    from .skims import SkimStore

    components = {c.name: c for c in build_steps(model.configs, model.data_dirs, model.households_sample_size,
                                                 only=set(steps))}
    missing = [s for s in steps if s not in components]
    if missing:
        raise ValueError(f"process {k}: steps asim_core can't run for these configs: {missing}")
    skims = SkimStore(model.skims) if model.skims else None
    store = read_tables(Path(in_dir), names)
    profiler = Profiler(split_memory=True)
    runner = Runner([components[s] for s in steps], Path(out_dir) / "run", base_seed=model.base_seed,
                    skims=skims, engine=Engine(dict(model.seams)), save_outputs=False)
    runner.run(steps, store, profiler, start_note=f"process {k}")
    written = write_tables(store, Path(out_dir) / "tables")
    return {"tables": written, "history": runner.history, "profile": profiler.records,
            "wall_s": time.perf_counter() - t0}


def run_group_in_processes(group: Group, model: ModelFiles, store: Store, work_dir: Path,
                           threads_per_process: int | None = None) -> tuple[Store, GroupResult]:
    """
    Split the store by 1.x's rules, run the group's steps in worker processes,
    and put the tables back together. Workers keep the main process's thread
    settings for the numeric libraries, as 1.x's do (the BLAS library's last
    digits depend on its thread count, so this keeps results 1.x's);
    ``threads_per_process`` sets them instead (e.g. CPUs / processes, to
    share the CPUs), which can change the last digits of some results.
    """
    n = group.num_processes
    t0 = time.perf_counter()
    rules = slice_rules(store, group.slice_tables, group.exclude)
    parts = split_store(store, rules, n)
    in_dirs = []
    for k in range(n):
        folder = work_dir / group.name / f"process_{k}" / "in"
        write_tables(Store([take_rows(store[t], parts[k][t]) for t in rules]), folder)
        in_dirs.append(folder)
    names = list(rules)
    ctx = multiprocessing.get_context("spawn")  # as Windows must; the same everywhere
    saved = {v: os.environ.get(v) for v in THREAD_VARIABLES}
    if threads_per_process:
        for v in THREAD_VARIABLES:
            os.environ[v] = str(threads_per_process)
    try:
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
            futures = [pool.submit(_worker, k, model, group.steps, str(in_dirs[k]),
                                   str(work_dir / group.name / f"process_{k}"), names) for k in range(n)]
            results = [f.result() for f in futures]
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old
    stores = [read_tables(work_dir / group.name / f"process_{k}" / "tables", r["tables"]) for k, r in enumerate(results)]
    # 1.x decides how to put the tables together from process 0's tables, with the group's slice rules
    rules_after = slice_rules(stores[0], group.slice_tables, group.exclude)
    merged = coalesce(stores, rules_after)
    result = GroupResult(group.name, n, time.perf_counter() - t0, [r["wall_s"] for r in results],
                         [r["profile"] for r in results])
    return merged, result


def reorder_as_split(store: Store, group: Group) -> Store:
    """
    A group run once in the main process (its values don't depend on the
    split), its split tables put in the order 1.x's split and coalescing
    leave them: process 0's rows, then process 1's, ...
    """
    n = group.num_processes
    rules = slice_rules(store, group.slice_tables, group.exclude)
    parts = split_store(store, rules, n)
    out = Store()
    for name, t in store.tables.items():
        rule = rules.get(name)
        if rule is None:
            out.add(t)
        else:
            out.add(take_rows(t, np.concatenate([parts[k][name] for k in range(n)])))
    return out


def group_record(name: str, store: Store) -> StepRecord:
    """The checkpoint 1.x adds after putting a group's tables back together: every table, re-saved."""
    return StepRecord(name, datetime.now(), {t: list(store[t].column_names) for t in store.tables})


def run_in_groups(groups: Sequence[Group], components: dict, model: ModelFiles, store: Store, out_dir: Path,
                  skims=None, engine=None, profiler: Profiler | None = None,
                  threads_per_process: int | None = None) -> tuple[Store, list[StepRecord], list[dict]]:
    """
    Run a model's groups in order, as 1.x's multiprocess run does; returns
    the final store, the checkpoints (as 1.x names them in multiprocess runs:
    ``init`` first, the steps of a group run in the main process by step, a
    group run in processes, or split by zones, by its name) and, per group,
    where it ran and how long it took.
    """
    from .runner import Runner

    profiler = profiler or Profiler(split_memory=True)
    history: list[StepRecord] = [StepRecord("init", datetime.now(), {})]  # 1.x's first multiprocess checkpoint
    timings = []
    for g in groups:
        t0 = time.perf_counter()
        own_peaks: list = []
        runner = Runner([components[s] for s in g.steps], out_dir / "run", base_seed=model.base_seed,
                        skims=skims, engine=engine)
        if not g.sliced:  # in the main process, step by step
            # 1.x runs each group in a fresh process that reloads every table: the first checkpoint of
            # any group but the first saves them all again
            runner.run(g.steps, store, profiler, start_note=g.name, history=history, reloaded=bool(store.tables))
            history = runner.history + [StepRecord(g.name, datetime.now(), {})]
            where, per_process = "main process", []
        elif g.slice_tables[0] == "households" and g.num_processes > 1:  # in worker processes
            store, res = run_group_in_processes(g, model, store, out_dir / "processes", threads_per_process)
            history = history + [group_record(g.name, store)]
            where, per_process = f"{g.num_processes} processes", res.process_wall_s
            own_peaks = [max((r.anon_peak_mb or 0.0) for r in prof) if prof else None for prof in res.profiles]
        else:  # split by zones, or one process: once here, the rows put in the split's order (none for one)
            runner.run(g.steps, store, profiler, start_note=g.name, history=history)
            store = reorder_as_split(store, g)
            history = history + [group_record(g.name, store)]
            where, per_process = "main process (rows in the split's order)", []
        timings.append({"group": g.name, "first_step": g.steps[0], "last_step": g.steps[-1], "where": where,
                        "wall_s": time.perf_counter() - t0, "process_wall_s": per_process,
                        "process_own_peak_mb": own_peaks})
    return store, history, timings
