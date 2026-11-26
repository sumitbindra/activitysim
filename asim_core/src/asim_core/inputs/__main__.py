"""
Check a region's inputs:

    python -m asim_core.inputs -c configs -d data [--cache output/cache] [--no-skims]

Exits with status 1 when there are errors.
"""

from __future__ import annotations

import argparse
import sys

from .checks import check_inputs
from .definitions import DefinitionError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m asim_core.inputs", description="Check input tables and skims.")
    ap.add_argument("-c", "--configs", action="append", required=True, help="configs folder (repeatable, first wins)")
    ap.add_argument("-d", "--data", action="append", required=True, help="data folder (repeatable)")
    ap.add_argument("--cache", help="folder to keep a report per version of the inputs")
    ap.add_argument("--no-skims", action="store_true", help="don't check the skims")
    args = ap.parse_args(argv)
    try:
        report = check_inputs(args.configs, args.data, cache_dir=args.cache, skims=not args.no_skims)
    except DefinitionError as e:
        print(e, file=sys.stderr)
        return 2
    print(report.text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
