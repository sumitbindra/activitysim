"""
Skims: zone-to-zone matrices (travel times, distances, costs, ...).

Skims are imported once from OMX into a folder holding:

* ``skims.npy``: every matrix stacked into one array (matrices x zones x zones)
* ``skims.json``: matrix names, zone IDs for the rows and columns, the number
  type, and where the data came from

At run time the array is **memory-mapped** read-only: nothing is copied into
the process, pages are read from disk only when used, and every process that
opens the same folder shares one copy in memory (the operating system handles
that). Opening a skim store is close to instant.

Time-period matrices follow the OMX naming convention ``NAME__PERIOD``
(e.g. ``SOVTOLL_TIME__AM``).

Importing needs an HDF5 reader. It uses PyTables (``tables``) and only the
import does, so running models doesn't need it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = "skims.json"
DATA = "skims.npy"


class SkimStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        manifest = json.loads((self.path / MANIFEST).read_text())
        self.names: list[str] = manifest["names"]
        self.zone_ids = np.asarray(manifest["zone_ids"])
        self.source = manifest.get("source", "")
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


def import_omx(
    omx_path: str | Path,
    out_dir: str | Path,
    dtype: str = "float32",
    zone_ids=None,
) -> SkimStore:
    """
    Convert an OMX file into a skim store folder.

    ``dtype``: number type to store. ActivitySim 1.x converts skims to
    float32 when loading them, so that's the default here too.

    ``zone_ids``: the zone ID used in the tables for each matrix row, in
    order. Defaults to 0, 1, 2, ..., which is what ActivitySim 1.x uses after
    it recodes zone IDs (``recode_columns`` in settings.yaml).
    """
    import tables  # only needed for importing

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tables.open_file(str(omx_path), mode="r") as f:
        group = f.get_node("/data")  # keep a reference: PyTables children need it alive
        names = sorted(group._v_children.keys())
        nodes = {name: group._f_get_child(name) for name in names}
        rows, cols = (int(x) for x in nodes[names[0]].shape)
        if rows != cols:
            raise ValueError(f"skims must be square, got {rows} x {cols}")
        data = np.lib.format.open_memmap(
            out_dir / DATA, mode="w+", dtype=np.dtype(dtype), shape=(len(names), rows, cols)
        )
        for i, name in enumerate(names):
            m = nodes[name].read()
            if m.shape != (rows, cols):
                raise ValueError(f"skim '{name}' has shape {m.shape}, expected {(rows, cols)}")
            data[i] = m.astype(dtype)  # same rounding as 1.x's conversion on load
        data.flush()
        del data
    ids = np.arange(rows) if zone_ids is None else np.asarray(zone_ids)
    if len(ids) != rows:
        raise ValueError(f"{len(ids)} zone IDs given for {rows} zones")
    manifest = {
        "names": names,
        "zone_ids": ids.tolist(),
        "dtype": str(np.dtype(dtype)),
        "source": str(omx_path),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest))
    return SkimStore(out_dir)
