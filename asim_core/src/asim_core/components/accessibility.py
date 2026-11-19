"""
Accessibility: for every zone, how many jobs can be reached, by mode and time
of day. Later models (auto ownership, ...) use the results.

Reads today's accessibility.yaml and accessibility.csv unchanged. For each
origin zone it evaluates the formulas for every destination zone, adds the
results up over destinations (in destination order) and takes log(1 + total),
as ActivitySim 1.x does (activitysim/abm/models/accessibility.py).

Origins can be processed in batches (``origins_per_batch``) to limit memory;
results don't depend on the batch size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..component import SKIMS, Component, StepContext
from ..spec.legacy import find_config_file, load_assignment_spec, load_global_constants, read_yaml

STEP = "compute_accessibility"  # 1.x step name
SKIM_ACCESSORS = {"skim_od": "od", "skim_do": "do"}  # skim_do looks up destination -> origin


def compute_accessibility(
    configs_dirs: Sequence[str | Path],
    settings_file: str = "accessibility.yaml",
    origins_per_batch: int | None = None,
) -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    if settings.get("land_use_columns_orig"):
        raise NotImplementedError(f"{settings_file}: land_use_columns_orig is not supported yet")
    constants = dict(load_global_constants(configs_dirs))
    constants.update(settings.get("CONSTANTS") or {})
    spec = load_assignment_spec(
        configs_dirs, settings.get("SPEC", "accessibility.csv"), constants,
        skim_accessors=list(SKIM_ACCESSORS), name=STEP,
    )
    land_use_columns = list(settings.get("land_use_columns") or [])
    used = spec.columns_used()
    not_listed = sorted(used - set(land_use_columns))
    if not_listed:
        raise ValueError(
            f"{settings_file}: formulas use df.{not_listed}, which are not in land_use_columns"
        )
    columns = sorted(used)
    assign, source = spec.function()
    targets = spec.targets

    def run(ctx: StepContext):
        origins = ctx.ids("accessibility")
        dests = ctx.ids("land_use")
        o_pos_all = ctx.skim_positions(origins)
        d_pos = ctx.skim_positions(dests)
        land_use = {c: ctx.column("land_use", c) for c in columns}
        n_dest = len(dests)
        batch = origins_per_batch or len(origins)
        out: dict[str, np.ndarray] = {}
        traced = ctx.tracer.targets("accessibility", origins)

        for start in range(0, len(origins), batch):
            o_pos = o_pos_all[start:start + batch]
            n_orig = len(o_pos)
            orig = np.repeat(o_pos, n_dest)
            dest = np.tile(d_pos, n_orig)
            c = {col: np.tile(vals, n_orig) for col, vals in land_use.items()}

            def s(accessor, name, orig=orig, dest=dest):
                if accessor == "skim_od":
                    return ctx.skim(name, orig, dest)
                return ctx.skim(name, dest, orig)

            with np.errstate(all="ignore"):  # 1.x logs numpy warnings and carries on
                v = assign(c, s)
            for t in targets:
                values = np.broadcast_to(np.asarray(v[t]), (n_orig * n_dest,))
                per_origin = values.reshape(n_orig, n_dest)
                # add up in destination order, in 64-bit, like 1.x's loop
                totals = np.cumsum(per_origin, axis=1, dtype=np.float64)[:, -1]
                # log1p from the C library, as 1.x's Numba code uses. NumPy's
                # vectorized log1p can differ in the last digit on some CPUs
                # (seen with AVX-512). One call per zone, so the cost is small.
                result = np.fromiter((math.log1p(x) for x in totals), np.float64, len(totals))
                result = result.astype(values.dtype, copy=False)
                if t not in out:
                    out[t] = np.empty(len(origins), dtype=result.dtype)
                out[t][start:start + n_orig] = result

            rows = traced[(traced >= start) & (traced < start + n_orig)]
            if len(rows):
                _trace(ctx, origins, dests, rows - start, start, v, targets, out, n_dest, land_use)
        return {"accessibility": out}

    return Component(
        name=STEP,
        run=run,
        needs={"land_use": columns, "accessibility": [], SKIMS: sorted(spec.skims_used())},
        provides={"accessibility": targets},
        description="accessibility to jobs by mode and time of day, per zone",
        info={"definition": spec, "generated_source": source, "trace_table": "accessibility"},
    )


def _trace(ctx, origins, dests, local_rows, start, v, targets, out, n_dest, land_use):
    """One CSV per traced origin zone: every destination's inputs and assigned values."""
    names = [k for k in v if k != "_"]
    for r in local_rows:
        sl = slice(r * n_dest, (r + 1) * n_dest)
        cols = {k: (np.asarray(v[k])[sl] if np.ndim(v[k]) else np.full(n_dest, v[k])) for k in names}
        rows = []
        for j, d in enumerate(dests):
            row = [origins[start + r], d]
            row += [land_use[c][j] for c in land_use]
            row += [cols[k][j] for k in names]
            rows.append(row)
        header = ["origin", "destination"] + list(land_use) + names
        ctx.tracer.write(ctx.step, f"origin_{origins[start + r]}", header, rows)
        summary = [[t, out[t][start + r]] for t in targets]
        ctx.tracer.write(ctx.step, f"origin_{origins[start + r]}.result", ["column", "log1p_total"], summary)
