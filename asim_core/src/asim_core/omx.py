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


def write_omx(path: str | Path, matrices: Mapping[str, np.ndarray], mappings: Mapping[str, np.ndarray]) -> None:
    """Write ``matrices`` (name -> square array, in this order) and ``mappings`` (name -> zone IDs) to a new file."""
    import tables

    filters = tables.Filters(complevel=1, shuffle=True, fletcher32=False, complib="zlib")
    with tables.open_file(str(path), "w", filters=filters) as f:
        f.root._v_attrs["OMX_VERSION"] = OMX_VERSION
        f.root._v_attrs["OMX_CREATED_WITH"] = CREATED_WITH
        f.create_group(f.root, "data")
        f.create_group(f.root, "lookup")
        shape = None
        for name, values in matrices.items():
            values = np.asarray(values)
            if shape is None:
                shape = values.shape
                f.root._v_attrs["SHAPE"] = np.array(shape, dtype="int32")
            elif values.shape != shape:
                raise ValueError(f"{path}: matrix {name} is {values.shape}, the file's matrices are {shape}")
            f.create_carray(f.root.data, name, tables.Atom.from_dtype(values.dtype), values.shape, obj=values)
        for name, entries in mappings.items():
            entries = np.asarray(entries)
            if shape is not None and len(entries) not in shape:
                raise ValueError(f"{path}: mapping {name} has {len(entries)} entries, the matrices are {shape}")
            m = f.create_array(f.root.lookup, name, atom=tables.UInt32Atom(), shape=(len(entries),))
            m[:] = entries


def read_omx(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """(matrices, mappings, root attributes) of an OMX file, as arrays."""
    import tables

    with tables.open_file(str(path), "r") as f:
        matrices = {node.name: node.read() for node in f.list_nodes(f.root.data)} if "data" in f.root else {}
        mappings = {node.name: node.read() for node in f.list_nodes(f.root.lookup)} if "lookup" in f.root else {}
        attrs = {k: f.root._v_attrs[k] for k in f.root._v_attrs._v_attrnamesuser}
    return matrices, mappings, attrs
