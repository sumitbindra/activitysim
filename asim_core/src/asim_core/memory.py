"""
Giving freed memory back to the operating system.

After a step frees large tables, the C allocator can keep the freed memory
for the process instead of returning it (glibc does when many medium-sized
blocks were freed): the process then holds it for the rest of the run. For
example, checking full MTC's input tables (7.6 million persons) left about
1 GB held; returning it brings that to under 0.1 GB. Arrow's default
allocator (mimalloc) keeps freed memory too, so code that frees large Arrow
tables should allocate them from Arrow's system allocator (see
``inputs/checks.py``).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import sys

_libc = None


def release_freed_memory() -> bool:
    """Return freed memory to the operating system where the C library allows it
    (glibc's malloc_trim). True if it was attempted."""
    global _libc
    gc.collect()
    if not sys.platform.startswith("linux"):
        return False
    try:
        if _libc is None:
            _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        trim = getattr(_libc, "malloc_trim", None)
        if trim is None:  # not glibc (e.g. musl)
            return False
        trim(0)
        return True
    except OSError:
        return False
