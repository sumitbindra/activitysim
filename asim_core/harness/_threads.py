"""
Threads for the numeric libraries (BLAS, numexpr, Numba), set before NumPy
loads, as ActivitySim 1.x's command line sets them: one thread unless asked
otherwise.

    --threads N   N threads (default 1, as 1.x's command line)
    --fast        no limit: the libraries' own default, every core (1.x's --fast)

Why one: most of a model's work runs on one core whatever the setting (more
threads gained nothing past 4 on a 24-core PC, and 4% over one); the BLAS
library's last digits depend on its thread count, so with one thread results
don't depend on the machine's number of cores (asim1 findings, 24); a
multiprocess run doesn't have every process start a thread per core; and
Windows doesn't reserve memory for threads that wait. Use more processes,
not threads, to use more cores. Worker processes inherit the setting.

Call ``limit_from_argv()`` at the top of a script, before anything imports
NumPy, and ``add_arguments(parser)`` so its parser accepts the options.
"""

from __future__ import annotations

import os
import sys

VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")


def limit_from_argv(argv=None) -> int | None:
    """Set the thread variables from --threads/--fast in ``argv`` (default 1). Returns the count, None: no limit."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "numpy" in sys.modules:
        print("warning: NumPy was loaded before the thread setting; it may not apply", file=sys.stderr)
    if "--fast" in argv:
        return None
    n = 1
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
        elif a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
    if n <= 0:
        return None
    for v in VARIABLES:
        os.environ[v] = str(n)
    return n


def add_arguments(parser) -> None:
    parser.add_argument("--threads", type=int, default=1,
                        help="threads for the numeric libraries (default 1, as 1.x's command line; "
                             "use more processes, not threads, to use more cores)")
    parser.add_argument("--fast", action="store_true",
                        help="no thread limit (every core), as 1.x's --fast; results' last digits can then "
                             "depend on the machine's number of cores")


def describe() -> str:
    value = os.environ.get("OPENBLAS_NUM_THREADS")
    return f"{value} thread(s)" if value else "no thread limit"
