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
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MANIFEST = "skims.json"
DATA = "skims.npy"
FORMAT_VERSION = 1


class SkimStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        manifest = json.loads((self.path / MANIFEST).read_text())
        if not manifest.get("complete"):
            raise ValueError(f"skim cache {self.path} is incomplete")
        self.manifest = manifest
        self.names: list[str] = manifest["names"]
        self.zone_ids = np.asarray(manifest["zone_ids"])
        self.fingerprint: str = manifest.get("fingerprint", "")
        self._index = {n: i for i, n in enumerate(self.names)}
        self._data = np.load(self.path / DATA, mmap_mode="r")
        self._identity = np.array_equal(self.zone_ids, np.arange(len(self.zone_ids)))
        self._lookup = None if self._identity else pd.Index(self.zone_ids)

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
        ids = np.asarray(zone_ids)
        if self._identity:
            if len(ids) and (ids.min() < 0 or ids.max() >= len(self.zone_ids)):
                raise KeyError(f"zone IDs outside 0..{len(self.zone_ids) - 1}")
            return ids.astype(np.intp, copy=False)
        pos = self._lookup.get_indexer(ids)
        if (pos < 0).any():
            raise KeyError(f"zone IDs not in skims: {ids[pos < 0][:5].tolist()}")
        return pos

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
    payload = {
        "format": FORMAT_VERSION,
        "sources": _source_info([Path(p) for p in omx_paths]),
        "dtype": str(np.dtype(dtype)),
        "zone_ids": None if zone_ids is None else np.asarray(zone_ids).tolist(),
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
    it recodes zone IDs (``recode_columns`` in settings.yaml).

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
                    chosen[name] = (i, node)
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
                data[k] = chosen[name][1].read().astype(dtype)  # same rounding as 1.x on load
            data.flush()
            del data
        finally:
            for f in handles:
                f.close()

        ids = np.arange(rows) if zone_ids is None else np.asarray(zone_ids)
        if len(ids) != rows:
            raise ValueError(f"{len(ids)} zone IDs given for {rows} zones")
        manifest = {
            "format": FORMAT_VERSION,
            "names": names,
            "zone_ids": ids.tolist(),
            "dtype": str(np.dtype(dtype)),
            "periods": None if periods is None else sorted(allowed),
            "sources": sources,
            "fingerprint": fp,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "complete": True,  # written last
        }
        (tmp / MANIFEST).write_text(json.dumps(manifest))
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
        manifest = json.loads((path / MANIFEST).read_text())
    except (OSError, ValueError):
        return False
    return bool(manifest.get("complete")) and manifest.get("fingerprint") == fp and (path / DATA).exists()


def _remove_other_caches(cache_dir: Path, tag: str, keep: Path) -> None:
    """Remove caches other than ``keep``. Ones still open elsewhere (Windows) are left for next time."""
    for p in _cache_folders(cache_dir, tag):
        if p.resolve() == keep.resolve():
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
            target = cache_dir / f"asim_core_{tag}_{fp}_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
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
    from .spec.legacy import find_config_file, read_yaml

    los = read_yaml(find_config_file(configs_dirs, "network_los.yaml"))
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
    from .spec.legacy import find_config_file, read_yaml

    los = read_yaml(find_config_file(configs_dirs, "network_los.yaml"))
    paths = omx_files_from_settings(configs_dirs, data_dirs, tag)
    periods = (los.get("skim_time_periods") or {}).get("labels")
    reuse = bool(los.get("read_skim_cache", False)) if read_skim_cache is None else read_skim_cache
    cache_dir = los.get("network_cache_dir")
    cache_dir = Path(cache_dir) if cache_dir else Path(output_dir) / "cache"
    return prepare_skims(paths, cache_dir, reuse=reuse, periods=periods, tag=tag)
