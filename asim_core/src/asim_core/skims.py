"""
Skims: zone-to-zone matrices (travel times, distances, costs, ...).

OMX stays the format for exchanging skims with assignment software. asim_core
imports the OMX files into a cache folder holding:

* ``skims.npy``: every matrix stacked into one array (matrices x zones x zones)
* ``skims.json``: matrix names, zone IDs for the rows and columns, the number
  type, and a fingerprint of the OMX files it came from

At run time the array is **memory-mapped** read-only: nothing is copied into
the process, pages are read from disk only when used, and every process that
opens the same folder shares one copy in memory. This is the same idea as
ActivitySim 1.x's own skim cache (``write_skim_cache`` / ``read_skim_cache``,
which writes ``cache/cached_taz.mmap``).

Settings follow 1.x and come from ``network_los.yaml``:

* ``taz_skims``: one OMX file, a list, or a ``{omx: ...}`` mapping; names may
  be glob patterns, found in the data folders as 1.x does. If a skim name is in
  more than one file, the later file wins (with a warning), as in 1.x.
* ``skim_time_periods.labels``: time-period skims (``NAME__PERIOD``) for other
  periods are skipped, as in 1.x.
* ``read_skim_cache``: False (1.x's default) imports the OMX files on every
  run, as 1.x reads them on every run. True reuses the cache, but unlike 1.x
  only if the OMX files haven't changed since it was made (size and
  modification time), so feedback loops can't pick up stale skims.
* ``network_cache_dir``: where the cache goes (default: ``<output>/cache``).

Each import goes into a new folder, named by fingerprint, and is renamed into
place only once complete. Existing folders are never overwritten (Windows
can't overwrite a file another process has memory-mapped); folders no longer
needed are removed when possible.

Importing needs an HDF5 reader (PyTables); opening a cache doesn't.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
import warnings
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MANIFEST = "skims.json"
DATA = "skims.npy"
FORMAT_VERSION = 1


def skim_zone_ids(zone_ids, lookup_values, rows: int) -> np.ndarray:
    """
    The zone ID of each skim row, by 1.x's rules (used by the import and by the
    input checks, so both find a zone's row the same way):

    * ``zone_ids=None``: zones renumbered from 0 (``recode_columns``): row i is
      zone i, the i-th smallest original zone number;
    * ``zone_ids="omx"``: original zone numbers: the OMX file's zone lookup, in
      matrix order (any order); without one, 1.x takes the skims as numbered
      from 1, zone z is row z - 1 (``SkimDict._offset_mapper``, "assume this is
      a one-based skim map");
    * an array: those IDs.
    """
    if isinstance(zone_ids, str) and zone_ids == "omx":
        ids = np.arange(1, rows + 1) if lookup_values is None else np.asarray(lookup_values)
    else:
        ids = np.arange(rows) if zone_ids is None else np.asarray(zone_ids)
    if len(ids) != rows:
        raise ValueError(f"{len(ids)} zone IDs given for {rows} zones")
    return ids


class ZoneRows:
    """Finds the skim row of zone IDs; unknown zones are an error."""

    def __init__(self, zone_ids):
        self.zone_ids = np.asarray(zone_ids)
        self._identity = np.array_equal(self.zone_ids, np.arange(len(self.zone_ids)))
        self._lookup = None if self._identity else pd.Index(self.zone_ids)

    def __call__(self, zone_ids) -> np.ndarray:
        ids = np.asarray(zone_ids)
        if self._identity:
            if len(ids) and (ids.min() < 0 or ids.max() >= len(self.zone_ids)):
                raise KeyError(f"zone IDs outside 0..{len(self.zone_ids) - 1}")
            return ids.astype(np.intp, copy=False)
        pos = self._lookup.get_indexer(ids)
        if (pos < 0).any():
            raise KeyError(f"zone IDs not in skims: {ids[pos < 0][:5].tolist()}")
        return pos


class ZonePairs:
    """
    Origin and destination skim positions for a set of rows (zone pairs). A
    step looks up many skims for the same pairs; what lookups need beyond the
    positions (the flat positions ``orig * zones + dest``) is worked out once,
    when first needed, and shared by all of them, and so are the reversed
    pairs and subsets of the rows (e.g. the rows of one time period).
    """

    def __init__(self, orig: np.ndarray, dest: np.ndarray, zones: int):
        self.orig = np.asarray(orig, dtype=np.intp)
        self.dest = np.asarray(dest, dtype=np.intp)
        self.zones = int(zones)
        self._flat = None
        self._reversed = None
        self._subsets: dict = {}

    def __len__(self) -> int:
        return len(self.orig)

    @property
    def flat(self) -> np.ndarray:
        if self._flat is None:
            self._flat = self.orig * self.zones + self.dest
        return self._flat

    def reversed(self) -> ZonePairs:
        """The same rows from destination to origin."""
        if self._reversed is None:
            self._reversed = ZonePairs(self.dest, self.orig, self.zones)
            self._reversed._reversed = self
        return self._reversed

    def subset(self, key, rows: np.ndarray) -> ZonePairs:
        """The pairs of some rows, kept under ``key`` (the same key must mean the same rows)."""
        if key not in self._subsets:
            self._subsets[key] = ZonePairs(self.orig[rows], self.dest[rows], self.zones)
        return self._subsets[key]


class SkimStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        manifest = json.loads((self.path / MANIFEST).read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError(f"skim cache {self.path} is incomplete")
        self.manifest = manifest
        self.names: list[str] = manifest["names"]
        self.zone_ids = np.asarray(manifest["zone_ids"])
        self.fingerprint: str = manifest.get("fingerprint", "")
        lookup = manifest.get("omx_zone_lookup")
        # zone numbers from the OMX file's lookup, if it had one (row order)
        self.omx_zone_numbers = None if lookup is None else np.asarray(lookup["values"])
        self._index = {n: i for i, n in enumerate(self.names)}
        self._data = np.load(self.path / DATA, mmap_mode="r")
        self._rows = ZoneRows(self.zone_ids)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __getitem__(self, name: str) -> np.ndarray:
        """One matrix (zones x zones), memory-mapped."""
        try:
            return self._data[self._index[name]]
        except KeyError:
            raise KeyError(f"no skim named '{name}' in {self.path}") from None

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._data.shape

    def positions(self, zone_ids) -> np.ndarray:
        """Row/column positions in the matrices for the given zone IDs."""
        return self._rows(zone_ids)

    def __repr__(self) -> str:
        n, z, _ = self.shape
        return f"SkimStore({self.path}, {n} matrices, {z} zones, {self.dtype})"


# --- fingerprints -------------------------------------------------------------------


def _source_info(paths: Sequence[Path]) -> list[dict]:
    out = []
    for p in paths:
        st = os.stat(p)
        out.append({"path": str(Path(p).resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return out


def fingerprint(omx_paths: Sequence[str | Path], dtype: str = "float32", zone_ids=None,
                periods: Sequence[str] | None = None) -> str:
    """
    Identifies the OMX files (path, size, modification time) and the import
    options. Changes whenever a file is rewritten.
    """
    if isinstance(zone_ids, str) and zone_ids == "omx":
        # caches made before files without a zone lookup were numbered from 1 are not reused
        zone_ids = "omx; without a lookup, numbered from 1"
    payload = {
        "format": FORMAT_VERSION,
        "sources": _source_info([Path(p) for p in omx_paths]),
        "dtype": str(np.dtype(dtype)),
        "zone_ids": None if zone_ids is None else (
            zone_ids if isinstance(zone_ids, str) else np.asarray(zone_ids).tolist()),
        "periods": None if periods is None else sorted(set(periods)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# --- importing ----------------------------------------------------------------------


def import_omx(
    omx_paths: str | Path | Sequence[str | Path],
    out_dir: str | Path,
    dtype: str = "float32",
    zone_ids=None,
    periods: Sequence[str] | None = None,
) -> SkimStore:
    """
    Convert one or more OMX files into a skim cache folder at ``out_dir``
    (which must not exist yet). The folder only appears once complete.

    ``dtype``: ActivitySim 1.x converts skims to float32 when loading them,
    so that's the default here too.

    ``zone_ids``: the zone ID used in the tables for each matrix row, in
    order. Defaults to 0, 1, 2, ..., which is what ActivitySim 1.x uses after
    it recodes zone IDs (``recode_columns`` in settings.yaml). ``"omx"`` uses
    the OMX file's zone lookup, for models that keep their original zone
    numbers (as 1.x does with the lookup when zones aren't recoded); without
    a lookup, 1, 2, 3, ..., as 1.x.

    ``periods``: if given, time-period skims (``NAME__PERIOD``) for other
    periods are skipped with a warning, as in 1.x.
    """
    import tables  # only needed for importing

    if isinstance(omx_paths, (str, Path)):
        omx_paths = [omx_paths]
    omx_paths = [Path(p) for p in omx_paths]
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; skim caches are never overwritten")
    fp = fingerprint(omx_paths, dtype, zone_ids, periods)
    sources = _source_info(omx_paths)
    allowed = None if periods is None else set(periods)

    tmp = out_dir.parent / f".importing-{out_dir.name}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True)
    try:
        handles = [tables.open_file(str(p), mode="r") for p in omx_paths]
        try:
            # skim name -> (file index, node); later files win, as in 1.x
            chosen: dict[str, tuple[int, object]] = {}
            shape = None
            lookup = None  # first zone lookup found; all must agree, as in 1.x
            for path, f in zip(omx_paths, handles):
                if "/lookup" in f:
                    for name, node in f.get_node("/lookup")._v_children.items():
                        values = np.asarray(node.read())
                        if lookup is None:
                            lookup = {"name": name, "values": values}
                        elif not np.array_equal(lookup["values"], values):
                            raise ValueError(f"zone lookups differ: '{lookup['name']}' and '{name}' in {path}")
            for i, (path, f) in enumerate(zip(omx_paths, handles)):
                group = f.get_node("/data")  # keep a reference: PyTables children need it alive
                for name in sorted(group._v_children.keys()):
                    base, sep, period = name.partition("__")
                    if allowed is not None and sep and period not in allowed:
                        warnings.warn(f"skim '{base}' has unknown time period '{period}'; skipped")
                        continue
                    node = group._f_get_child(name)
                    node_shape = tuple(int(x) for x in node.shape)
                    if shape is None:
                        shape = node_shape
                    elif node_shape != shape:
                        raise ValueError(f"skim '{name}' in {path} has shape {node_shape}, expected {shape}")
                    if name in chosen:
                        warnings.warn(f"duplicate skim '{name}' in {omx_paths[chosen[name][0]]} "
                                      f"and {path}; using {path}")
                    chosen[name] = (i, node._v_pathname)
                    node._f_close()
            if not chosen:
                raise ValueError(f"no skims found in {[str(p) for p in omx_paths]}")
            rows, cols = shape
            if rows != cols:
                raise ValueError(f"skims must be square, got {rows} x {cols}")
            names = sorted(chosen)
            data = np.lib.format.open_memmap(
                tmp / DATA, mode="w+", dtype=np.dtype(dtype), shape=(len(names), rows, cols)
            )
            for k, name in enumerate(names):
                i, node_path = chosen[name]
                node = handles[i].get_node(node_path)
                data[k] = node.read().astype(dtype)  # same rounding as 1.x on load
                # an open matrix keeps up to 16 MB of decompressed data cached
                # (PyTables' chunk cache); closing it frees that, so memory stays
                # flat however many matrices there are
                node._f_close()
            data.flush()
            del data
        finally:
            for f in handles:
                f.close()

        ids = skim_zone_ids(zone_ids, None if lookup is None else lookup["values"], rows)
        manifest = {
            "format": FORMAT_VERSION,
            "names": names,
            "zone_ids": ids.tolist(),
            "dtype": str(np.dtype(dtype)),
            "periods": None if periods is None else sorted(allowed),
            "sources": sources,
            "omx_zone_lookup": None if lookup is None else
                {"name": lookup["name"], "values": lookup["values"].tolist()},
            "fingerprint": fp,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "complete": True,  # written last
        }
        (tmp / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        tmp.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return SkimStore(out_dir)


# --- preparing skims for a run ------------------------------------------------------


def _cache_folders(cache_dir: Path, tag: str) -> list[Path]:
    return sorted(p for p in cache_dir.glob(f"asim_core_{tag}_*") if p.is_dir())


def _valid(path: Path, fp: str) -> bool:
    try:
        manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(manifest.get("complete")) and manifest.get("fingerprint") == fp and (path / DATA).exists()


def _source_paths(path: Path) -> set[str] | None:
    try:
        manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {src["path"] for src in manifest.get("sources", [])}


def _remove_other_caches(cache_dir: Path, tag: str, keep: Path) -> None:
    """
    Remove older caches made from the same OMX files as ``keep`` (caches of
    other skim files sharing the folder are left alone). Ones still open
    elsewhere (Windows) are left for next time.
    """
    sources = _source_paths(keep)
    for p in _cache_folders(cache_dir, tag):
        if p.resolve() == keep.resolve() or _source_paths(p) != sources:
            continue
        try:
            shutil.rmtree(p)
        except OSError as err:
            logger.info(f"skim cache {p} not removed (probably in use): {err}")
    for p in cache_dir.glob(f".importing-asim_core_{tag}_*"):  # left by a crashed import
        try:
            if time.time() - p.stat().st_mtime > 24 * 3600:
                shutil.rmtree(p)
        except OSError:
            pass


def prepare_skims(
    omx_paths: Sequence[str | Path],
    cache_dir: str | Path,
    reuse: bool = False,
    dtype: str = "float32",
    zone_ids=None,
    periods: Sequence[str] | None = None,
    tag: str = "taz",
) -> SkimStore:
    """
    Skims ready for a run.

    ``reuse=False`` (``read_skim_cache: False``, 1.x's default) imports the
    OMX files every time, into a new cache folder. ``reuse=True`` opens an
    existing cache made from the same, unchanged OMX files, and imports only
    if there isn't one. Prepare once per run and give the folder to any worker
    processes, rather than preparing in each.
    """
    omx_paths = [Path(p) for p in omx_paths]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(omx_paths, dtype, zone_ids, periods)
    stable = cache_dir / f"asim_core_{tag}_{fp}"
    if reuse and _valid(stable, fp):
        store = SkimStore(stable)
    else:
        target = stable
        if target.exists():  # a fresh import was asked for, or a broken leftover: never overwrite
            # time, process and a random part: two imports in the same second can't share a folder (on
            # Windows the folder of a cache still open elsewhere can't be removed, so names pile up)
            target = (cache_dir / f"asim_core_{tag}_{fp}_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
                                  f"_{uuid.uuid4().hex[:6]}")
        store = import_omx(omx_paths, target, dtype, zone_ids, periods)
    _remove_other_caches(cache_dir, tag, keep=store.path)
    return store


def omx_files_from_settings(configs_dirs: Sequence[str | Path], data_dirs: Sequence[str | Path],
                            tag: str = "taz") -> list[Path]:
    """
    The OMX files named by ``<tag>_skims`` in network_los.yaml, found the way
    1.x finds them: each name (or glob pattern) in the first data folder that
    has it.
    """
    from .spec.legacy import find_config_file, read_settings, read_yaml

    los = read_settings(configs_dirs, "network_los.yaml")
    names = los.get(f"{tag}_skims")
    if isinstance(names, dict):
        for key in ("file", "files", "omx"):
            if key in names:
                names = names[key]
                break
        else:
            raise ValueError(f"network_los.yaml: {tag}_skims needs 'omx' (or 'file'/'files')")
    if isinstance(names, (str, Path)):
        names = [names]
    paths: list[Path] = []
    for name in names:
        if str(name).lower().endswith((".parquet", ".pq")):
            raise NotImplementedError(f"Parquet skim files ({name}) are not supported yet")
        for d in data_dirs:
            exact = Path(d) / name
            if exact.is_file():
                paths.append(exact)
                break
            matches = sorted(p for p in Path(d).glob(str(name)) if p.is_file())
            if matches:
                paths.extend(matches)
                break
        else:
            raise FileNotFoundError(f"skim file '{name}' not found in {[str(d) for d in data_dirs]}")
    return paths


def prepare_skims_from_settings(
    configs_dirs: Sequence[str | Path],
    data_dirs: Sequence[str | Path],
    output_dir: str | Path,
    read_skim_cache: bool | None = None,
    tag: str = "taz",
) -> SkimStore:
    """
    Skims for a run, using network_los.yaml as 1.x does. ``read_skim_cache``
    overrides the setting when given.
    """
    from .spec.legacy import find_config_file, read_settings, read_yaml

    los = read_settings(configs_dirs, "network_los.yaml")
    paths = omx_files_from_settings(configs_dirs, data_dirs, tag)
    periods = (los.get("skim_time_periods") or {}).get("labels")
    reuse = bool(los.get("read_skim_cache", False)) if read_skim_cache is None else read_skim_cache
    cache_dir = los.get("network_cache_dir")
    cache_dir = Path(cache_dir) if cache_dir else Path(output_dir) / "cache"
    # zones renumbered from 0 (recode_columns): zone IDs are matrix positions.
    # Otherwise the original zone numbers are used, matched through the OMX lookup.
    zone_ids = None if land_use_zones_recoded(configs_dirs) else "omx"
    return prepare_skims(paths, cache_dir, reuse=reuse, zone_ids=zone_ids, periods=periods, tag=tag)


def skim_period_labels(configs_dirs: Sequence[str | Path]) -> list[str] | None:
    """The time period labels of network_los.yaml (``skim_time_periods``), each once."""
    from .spec.legacy import find_config_file, read_settings, read_yaml

    try:
        los = read_settings(configs_dirs, "network_los.yaml")
    except FileNotFoundError:
        return None
    labels = (los.get("skim_time_periods") or {}).get("labels")
    return None if labels is None else list(dict.fromkeys(labels))


def land_use_zones_recoded(configs_dirs: Sequence[str | Path]) -> bool:
    """
    True if settings.yaml renumbers land_use zone IDs from 0. As in 1.x, that
    needs both ``recode_columns: {zone_id: zero-based}`` on the land_use input
    table and ``recode_pipeline_columns: True`` (off by default), which turns
    recoding on at all.
    """
    from .spec.legacy import load_top_settings

    settings = load_top_settings(configs_dirs)
    if not settings.get("recode_pipeline_columns", False):
        return False
    for table in settings.get("input_table_list") or []:
        if table.get("tablename") == "land_use":
            recode = table.get("recode_columns") or {}
            return str(recode.get("zone_id", "")).strip() == "zero-based"
    return False


def check_zone_order(skims: SkimStore, land_use) -> str | None:
    """
    For renumbered zones: the OMX zone lookup must list land_use's original
    zone numbers in matrix order. Returns a problem description, or None.
    """
    if skims.omx_zone_numbers is None or "_original_zone_id" not in land_use:
        return None
    pos = skims.positions(land_use.index)
    expected = np.asarray(land_use["_original_zone_id"])
    found = skims.omx_zone_numbers[pos]
    bad = np.flatnonzero(found != expected)
    if len(bad) == 0:
        return None
    i = bad[0]
    return (f"skim zone order doesn't match land_use: zone_id {land_use.index[i]} is original zone "
            f"{expected[i]}, but matrix row {pos[i]} is zone {found[i]} in the OMX lookup "
            f"({len(bad)} zones differ)")


# --- time periods ------------------------------------------------------------------


@dataclass
class TimePeriods:
    """
    Skim time periods from network_los.yaml (``skim_time_periods``): times in
    (periods[i], periods[i+1]] get labels[i], as 1.x's
    ``skim_time_period_label`` (pandas ``cut``) assigns them.
    """

    periods: list[float]
    labels: list[str]

    @property
    def unique_labels(self) -> list[str]:
        return list(dict.fromkeys(self.labels))

    def label_index(self, times) -> np.ndarray:
        """For each time, the position of its label in ``unique_labels``."""
        times = np.asarray(times, dtype=np.float64)
        bins = np.searchsorted(np.asarray(self.periods, dtype=np.float64), times, side="left") - 1
        bad = (bins < 0) | (bins >= len(self.labels)) | np.isnan(times)
        if bad.any():
            raise ValueError(
                f"times outside the skim time periods {self.periods}: {np.unique(times[bad])[:5].tolist()}"
            )
        unique = self.unique_labels
        to_unique = np.array([unique.index(lab) for lab in self.labels])
        return to_unique[bins]

    def labels_for(self, times) -> np.ndarray:
        """Each time's label, as an array of strings (object dtype, like 1.x's)."""
        return np.asarray(self.unique_labels, dtype=object)[self.label_index(times)]


def time_periods_from_settings(configs_dirs: Sequence[str | Path]) -> TimePeriods:
    from .spec.legacy import find_config_file, read_settings, read_yaml

    los = read_settings(configs_dirs, "network_los.yaml")
    tp = los.get("skim_time_periods")
    if not tp:
        raise ValueError("network_los.yaml has no skim_time_periods")
    if len(tp["labels"]) != len(tp["periods"]) - 1:
        raise ValueError("skim_time_periods: need one label per period interval")
    return TimePeriods(periods=list(tp["periods"]), labels=[str(x) for x in tp["labels"]])
