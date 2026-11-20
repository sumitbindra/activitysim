"""
How long the input checks take on a large population.

Builds a population ``--copies`` times the ARC example's (households and
persons repeated with new IDs, same land use), then times, each in a fresh
process: asim_core's checks (reading included), and reading the same files
with pandas as 1.x does (no checks).

    python asim_core/harness/bench_input_checks.py --copies 100
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "activitysim/examples/prototype_arc"
OURS = REPO / "asim_core/configs/prototype_arc"


def build(copies: int, out: Path) -> Path:
    data = out / f"arc_x{copies}"
    if (data / "persons.csv").exists():
        return data
    data.mkdir(parents=True, exist_ok=True)
    hh = pd.read_csv(EXAMPLE / "data/households.csv")
    per = pd.read_csv(EXAMPLE / "data/persons.csv")
    hh_step = int(hh.household_id.max()) + 1
    per_step = int(per.person_id.max()) + 1
    hh_all = pd.concat([hh.assign(household_id=hh.household_id + k * hh_step) for k in range(copies)])
    per_all = pd.concat([per.assign(household_id=per.household_id + k * hh_step,
                                    person_id=per.person_id + k * per_step) for k in range(copies)])
    hh_all.to_csv(data / "households.csv", index=False)
    per_all.to_csv(data / "persons.csv", index=False)
    for p in (EXAMPLE / "data").iterdir():
        if p.is_file() and not (data / p.name).exists():
            (data / p.name).symlink_to(p)
    return data


CHILD = r"""
import json, resource, sys, time
sys.path.insert(0, {src!r})
t0 = time.perf_counter()
what = {what!r}
if what == "asim_core":
    from asim_core.inputs import check_inputs
    r = check_inputs({configs!r}, [{data!r}], skims=False)
    out = {{"errors": len(r.errors), "warnings": len(r.warnings), "rows": r.rows_checked}}
elif what == "asim_core_read":
    from asim_core.inputs import read_tables
    read_tables({configs!r}, [{data!r}])
    out = {{}}
else:
    import pandas as pd
    for f in ("households.csv", "persons.csv", "land_use.csv"):
        pd.read_csv({data!r} + "/" + f, comment="#")
    out = {{}}
out["seconds"] = time.perf_counter() - t0
out["peak_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(json.dumps(out))
"""


def run(what: str, data: Path) -> dict:
    code = CHILD.format(src=str(REPO / "asim_core/src"), what=what,
                        configs=[str(OURS), str(EXAMPLE / "configs")], data=str(data))
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=100)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--work", default=str(REPO / "asim_core/work/input_checks"))
    ap.add_argument("--keep", action="store_true", help="keep the large population files")
    args = ap.parse_args()
    t = time.perf_counter()
    data = build(args.copies, Path(args.work))
    print(f"population: {data} ({time.perf_counter() - t:.0f} s to build)")
    for what in ("asim_core", "asim_core_read", "pandas_read"):
        results = [run(what, data) for _ in range(args.runs)]
        secs = sorted(r["seconds"] for r in results)
        print(f"{what:15} median {secs[len(secs) // 2]:.2f} s (min {secs[0]:.2f}, max {secs[-1]:.2f}), "
              f"peak memory {np.median([r['peak_mb'] for r in results]):,.0f} MB"
              + (f", {results[0]['errors']} errors, {results[0]['warnings']} warnings, rows {results[0]['rows']}"
                 if "errors" in results[0] else ""))
    if not args.keep:
        shutil.rmtree(data)


if __name__ == "__main__":
    main()
