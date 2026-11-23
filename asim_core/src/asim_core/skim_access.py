"""
The ``skim_access`` seam: how a step holds the skims it reads while it runs.

asim_core keeps skims in one memory-mapped file per region. The reference,
``memory_map``, reads straight from it: the operating system brings a page
into the process the first time a lookup touches it. With large zone
systems (MTC: 826 skims of 1,475 x 1,475, 7 GB) a step touches many pages,
and every fresh process pays for mapping them again; if the operating
system has dropped the file from its cache, the pages come from disk.

Alternatives, all giving exactly the same values:

* ``prefault``: before the step, ask the operating system to map every page
  of the skims the step declares, in one call per skim (``madvise`` with
  ``MADV_POPULATE_READ``), instead of one page fault at a time.
* ``in_memory``: before the step, read the skims it declares from the file
  into memory (as 1.x holds all of its skims), and look up from the copies.

The time this takes is part of the step's time.
"""

from __future__ import annotations

import mmap
from collections.abc import Iterable

import numpy as np

from .seams import register

MADV_POPULATE_READ = getattr(mmap, "MADV_POPULATE_READ", 22)  # Linux 5.14+


@register("skim_access", "memory_map", reference=True,
          description="read from the memory-mapped skims; pages come in on first use")
def memory_map(store, names: Iterable[str]):
    return store


def _skim_ranges(store, names):
    """(start, length) in the mapping of each skim's bytes, page-aligned."""
    data = store._data
    mm = getattr(data, "_mmap", None)
    if mm is None:
        return None, []
    # numpy maps from an allocation-aligned offset below the array's offset
    array_start = data.offset - (data.offset - data.offset % mmap.ALLOCATIONGRANULARITY)
    per_skim = data.shape[1] * data.shape[2] * data.dtype.itemsize
    page = mmap.PAGESIZE
    ranges = []
    for name in sorted(set(names)):
        if name not in store:
            continue
        start = array_start + store._index[name] * per_skim
        aligned = start - start % page
        ranges.append((aligned, per_skim + (start - aligned)))
    return mm, ranges


@register("skim_access", "prefault", agreement="exact",
          description="map the pages of the step's skims up front (madvise MADV_POPULATE_READ)")
def prefault(store, names: Iterable[str]):
    mm, ranges = _skim_ranges(store, names)
    if mm is None:
        return store
    if not hasattr(mm, "madvise"):  # Windows has no madvise: read one value per page instead
        data = store._data
        every = max(1, mmap.PAGESIZE // data.dtype.itemsize)
        for name in sorted(set(names)):
            if name in store:
                data[store._index[name]].reshape(-1)[::every].sum()
        return store
    for start, length in _merge(ranges):
        try:
            mm.madvise(MADV_POPULATE_READ, start, length)
        except OSError:  # older kernels: ask for read-ahead instead
            mm.madvise(mmap.MADV_WILLNEED, start, length)
    return store


def _merge(ranges):
    """Adjacent ranges as one (skims declared together often sit next to each other)."""
    out = []
    for start, length in sorted(ranges):
        if out and start <= out[-1][0] + out[-1][1]:
            s, n = out[-1]
            out[-1] = (s, max(n, start + length - s))
        else:
            out.append((start, length))
    return out


def _read_copy(store, name):
    """One skim read from the cache file into the process's own memory (not through the map,
    so the process doesn't also hold the mapped pages)."""
    from .skims import DATA

    data = store._data
    path = getattr(store, "path", None)
    if getattr(data, "_mmap", None) is None or path is None or not (path / DATA).exists():
        return np.array(store[name])
    per_skim = data.shape[1] * data.shape[2]
    offset = data.offset + store._index[name] * per_skim * data.dtype.itemsize
    out = np.fromfile(path / DATA, dtype=data.dtype, count=per_skim, offset=offset)
    return out.reshape(data.shape[1:])


class InMemorySkims:
    """The step's skims copied into memory; anything else is asked of the store."""

    def __init__(self, store, names: Iterable[str]):
        self._store = store
        wanted = [n for n in sorted(set(names)) if n in store]
        self._copies = {n: _read_copy(store, n) for n in wanted}

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self._copies:
            return self._copies[name]
        return self._store[name]

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __getattr__(self, attr):
        return getattr(self._store, attr)


@register("skim_access", "in_memory", agreement="exact",
          description="copy the step's skims into memory before it runs (as 1.x holds its skims)")
def in_memory(store, names: Iterable[str]):
    return InMemorySkims(store, names)
