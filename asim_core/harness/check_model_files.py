"""
Convert an example's ported steps to asim_core's model files, then run each
step twice on the same saved 1.x state, from today's files and from the
converted ones, and compare every output with each other and with 1.x.

    python asim_core/harness/check_model_files.py --example prototype_mtc

The converted files are written to asim_core/configs/<example>/ (read before
the example's own configs folder). A step's own formulas, coefficients and
options come from the converted files; shared settings (settings.yaml,
network_los.yaml, constants.yaml, size terms) are still read from the
example.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "asim_core/src"))
sys.path.insert(0, str(REPO / "asim_core/harness"))

from asim_core.asim1 import Asim1State  # noqa: E402
from asim_core.compare import compare_column  # noqa: E402
from asim_core.converter import STEPS, convert_step  # noqa: E402
from asim_core.runner import Runner  # noqa: E402

RUNS = {
    "prototype_mtc": REPO / "asim_core/work/asim1_legacy",
    "prototype_mwcog": REPO / "asim_core/work/mwcog/asim1_legacy",
    "prototype_arc": REPO / "asim_core/work/arc/asim1_legacy",
}


def run_step(make, configs, st: Asim1State, skims, workdir: Path):
    comp = make(configs)
    store = st.load_store(st.previous_step(comp.name))
    t0 = time.perf_counter()
    Runner([comp], workdir, skims=skims).run([comp.name], store)
    return comp, store, time.perf_counter() - t0


def main(argv=None) -> int:
    from run_milestone1 import skim_store

    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="prototype_mtc", choices=sorted(RUNS))
    ap.add_argument("--asim1-run", help="saved 1.x run (default: the example's in asim_core/work)")
    ap.add_argument("--steps", nargs="*", default=list(STEPS))
    ap.add_argument("--out", help="folder for the converted files (default: asim_core/configs/<example>)")
    args = ap.parse_args(argv)

    example = REPO / "activitysim/examples" / args.example
    legacy_configs = [example / "configs"]
    out = Path(args.out) if args.out else REPO / "asim_core/configs" / args.example
    new_configs = [out] + legacy_configs
    st = Asim1State(args.asim1_run or RUNS[args.example])
    skims = skim_store(example, legacy_configs)
    ok = True
    for step in args.steps:
        make = STEPS[step]["component"]
        written = convert_step(step, legacy_configs, st, out)
        print(f"\n{step}: converted to {', '.join(p.name for p in written)}")
        with tempfile.TemporaryDirectory() as tmp:
            old, old_store, t_old = run_step(make, legacy_configs, st, skims, Path(tmp) / "old")
            new, new_store, t_new = run_step(make, new_configs, st, skims, Path(tmp) / "new")
        print(f"  ran in {t_old:.2f} s (today's files) and {t_new:.2f} s (model files)")
        for table, cols in old.provides.items():
            lines = []
            for col in cols:
                ref = st.read(table, old.name)[col] if col in st.columns_at(old.name).get(table, ()) else None
                same = compare_column(step, new_store[table], col, old_store[table].to_pandas([col])[col])
                vs1x = compare_column(step, new_store[table], col, ref) if ref is not None else None
                status = f"same as today's files: {same.status}"
                if same.status != "exact":
                    status += f" ({same.mismatched} of {same.rows} rows differ, max {same.max_abs_diff})"
                if vs1x is not None:
                    status += f"; same as 1.x: {vs1x.status}"
                    if vs1x.status != "exact":
                        status += f" ({vs1x.mismatched} of {vs1x.rows} rows differ)"
                lines.append(f"  {table}.{col}: {status}")
                ok &= same.status == "exact" and (vs1x is None or vs1x.status == "exact")
            if len(lines) > 6 and all(ln.endswith("exact; same as 1.x: exact") for ln in lines):
                print(f"  {table}: {len(lines)} columns, each the same as today's files and as 1.x")
            else:
                print("\n".join(lines))
    print("\nALL EXACT" if ok else "\nSOME DIFFERENCES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
