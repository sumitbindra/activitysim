"""
Put a file in a known state in the operating system's page cache before a
timed run (Linux), and say how much of it is there.

Memory-mapped skims are read through the page cache: if the file's pages are
there, a first lookup costs a page mapping; if not, it reads the disk. So
timings of memory-mapped steps depend on what ran before. These helpers make
that explicit:

* ``evict(path)``: drop the file's pages from the cache (as after a reboot, or
  once other programs have pushed it out). Pages another process has mapped
  stay, so call it from a fresh process.
* ``load(path)``: read the whole file once, so it is all in the cache (as when
  the model ran on the same machine just before), if memory allows.
* ``resident_share(path)``: the share of the file's pages in the cache now.

``evict`` and ``resident_share`` need Linux; elsewhere (Windows, macOS) they
warn and do nothing (``evict``) or return ``None`` (``resident_share``), so a
timing run still works but doesn't control or report the cache.

Harness code: not used by asim_core itself.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import sys
import warnings
from pathlib import Path

import numpy as np


def evict(path: str | Path) -> None:
    if not hasattr(os, "posix_fadvise"):
        warnings.warn(f"can't drop {path} from the page cache on this system; the timing may start warm")
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)  # dirty pages can't be dropped
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def load(path: str | Path, chunk: int = 64 << 20) -> None:
    buf = bytearray(chunk)
    with open(path, "rb", buffering=0) as f:
        while f.readinto(buf):
            pass


def resident_share(path: str | Path) -> float | None:
    if not sys.platform.startswith("linux"):
        return None
    size = os.path.getsize(path)
    if size == 0:
        return 1.0
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vec = np.zeros(pages, dtype=np.uint8)
            if libc.mincore(ctypes.c_void_p(addr), size, vec.ctypes.data) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return float((vec & 1).mean())
        finally:
            libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


def set_state(path: str | Path, state: str) -> float:
    """Put ``path`` in ``state`` ('as_is', 'cold' or 'warm'); returns the resident share after."""
    if state == "cold":
        evict(path)
    elif state == "warm":
        load(path)
    elif state != "as_is":
        raise ValueError(f"unknown page cache state '{state}'")
    return resident_share(path)
