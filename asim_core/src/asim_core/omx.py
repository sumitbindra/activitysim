"""
Writing and reading OMX matrix files (HDF5; https://github.com/osPID/omx),
laid out as the ``openmatrix`` package writes them, which 1.x uses: root
attributes ``OMX_VERSION`` (b"0.2") and ``SHAPE`` (int32 [rows, columns]),
the matrices under ``/data`` (zlib level 1 with shuffle), zone mappings
under ``/lookup`` (uint32). Needs PyTables, imported only here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

OMX_VERSION = b"0.2"
CREATED_WITH = "asim_core"


class OmxWriter:
    """
    A new OMX file written a matrix at a time (only the matrix being added is
    needed in memory): ``add_matrix(name, values)``, ``add_mapping(name,
    zone_ids)``, ``close()``; or as a context manager.
    """

    def __init__(self, path: str | Path):
        import tables

        self._tables = tables
        self.path = path
        filters = tables.Filters(complevel=1, shuffle=True, fletcher32=False, complib="zlib")
        self.f = tables.open_file(str(path), "w", filters=filters)
        self.f.root._v_attrs["OMX_VERSION"] = OMX_VERSION
        self.f.root._v_attrs["OMX_CREATED_WITH"] = CREATED_WITH
        self.f.create_group(self.f.root, "data")
        self.f.create_group(self.f.root, "lookup")
        self.shape = None

    def add_matrix(self, name: str, values: np.ndarray) -> None:
        values = np.asarray(values)
        if self.shape is None:
            self.shape = values.shape
            self.f.root._v_attrs["SHAPE"] = np.array(self.shape, dtype="int32")
        elif values.shape != self.shape:
            raise ValueError(f"{self.path}: matrix {name} is {values.shape}, the file's matrices are {self.shape}")
        self.f.create_carray(self.f.root.data, name, self._tables.Atom.from_dtype(values.dtype), values.shape,
                             obj=values)

    def add_mapping(self, name: str, entries) -> None:
        entries = np.asarray(entries)
        if self.shape is not None and len(entries) not in self.shape:
            raise ValueError(f"{self.path}: mapping {name} has {len(entries)} entries, the matrices are {self.shape}")
        m = self.f.create_array(self.f.root.lookup, name, atom=self._tables.UInt32Atom(), shape=(len(entries),))
        m[:] = entries

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_omx(path: str | Path, matrices: Mapping[str, np.ndarray], mappings: Mapping[str, np.ndarray]) -> None:
    """Write ``matrices`` (name -> square array, in this order) and ``mappings`` (name -> zone IDs) to a new file."""
    with OmxWriter(path) as w:
        for name, values in matrices.items():
            w.add_matrix(name, values)
        for name, entries in mappings.items():
            w.add_mapping(name, entries)


def read_omx(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """(matrices, mappings, root attributes) of an OMX file, as arrays."""
    import tables

    with tables.open_file(str(path), "r") as f:
        matrices = {node.name: node.read() for node in f.list_nodes(f.root.data)} if "data" in f.root else {}
        mappings = {node.name: node.read() for node in f.list_nodes(f.root.lookup)} if "lookup" in f.root else {}
        attrs = {k: f.root._v_attrs[k] for k in f.root._v_attrs._v_attrnamesuser}
    return matrices, mappings, attrs
