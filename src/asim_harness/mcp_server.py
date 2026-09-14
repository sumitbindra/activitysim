"""MCP tools for the harness (PLAN.md, Phase 4): everything the CLI does, sized for a context window.

Served over stdio by ``asim mcp``. Every tool returns compact JSON: never a
raw table, and the big artifacts (summary, scorecard, error) come back in
their compact forms. No tool writes anything under ``example/``.

The MCP SDK is 2.x: ``FastMCP`` was renamed ``MCPServer`` (same decorator API).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__, compare, errors, example, ledger, paths, runner, summarize, targets
from . import manifest as mf
from .jsonio import read_json_if_exists

LOG_TAIL_MAX = 500
CONFIG_MAX_CHARS = 60_000
INSTRUCTIONS = (
    "asim-harness runs ActivitySim's prototype_mtc example reproducibly. Every run gets a run_id and a "
    "directory runs/<run_id>/ with manifest.json, summary.json (metrics), scorecard.json (deltas vs targets) "
    "or error.json. Always pass a descriptive label to run_model. Use sample_size (e.g. 500) to iterate and "
    "confirm at full size (sample_size omitted). A full run takes about 2 minutes and a 500-household run "
    "about 1.5 minutes, so call run_model with wait=false and poll get_run. Use resume_from + resume_after "
    "when only downstream steps changed. Report scorecards as deltas against targets together with n, not as "
    "raw shares. Nothing here edits files under example/."
)

server = MCPServer(name="asim-harness", instructions=INSTRUCTIONS, version=__version__, log_level="WARNING")


def _resolve(run_id: str) -> str:
    try:
        return ledger.resolve_run_id(run_id)
    except KeyError as e:
        raise ToolError(str(e)) from e


def _run_view(run_id: str) -> dict[str, Any]:
    run_dir = paths.run_dir(run_id)
    manifest = mf.read_manifest(run_dir)
    view = mf.summary_of(manifest)
    view["run_dir"] = str(run_dir)
    view["step_timings"] = manifest.get("step_timings") or {}
    view["files"] = sorted(p.name for p in run_dir.iterdir() if p.is_file())
    scorecard = read_json_if_exists(run_dir / targets.SCORECARD_NAME)
    error = read_json_if_exists(run_dir / errors.ERROR_NAME)
    if manifest.get("status") == "failed" and error is None:
        error = errors.error_for_run(run_id)
    return {
        "run": view,
        "scorecard": targets.compact(scorecard) if scorecard else None,
        "error": errors.compact(error, log_tail_lines=20) if error else None,
        "explanation": runner.failure_explanation(run_dir, manifest) if manifest.get("status") == "failed" else None,
        "summary_available": (run_dir / summarize.SUMMARY_NAME).exists(),
    }


@server.tool()
def list_runs(limit: int = 20) -> dict[str, Any]:
    """Ledger rows, newest first: run_id, label, status, sample size, duration, scorecard pass/fail."""
    rows = ledger.list_runs(max(1, min(int(limit), 200)))
    return {
        "runs": rows,
        "runs_dir": str(paths.runs_dir()),
        "targets_file": str(paths.targets_file()) if paths.targets_file().is_file() else None,
    }


@server.tool()
def run_model(
    label: str,
    sample_size: int | None = None,
    resume_from: str | None = None,
    resume_after: str | None = None,
    models: str | None = None,
    wait: bool = True,
) -> dict[str, Any]:
    """Start an ActivitySim run. label is required (describe what the run is for).

    sample_size sets households_sample_size (omit for the full example). resume_from + resume_after
    reuse a previous run's pipeline and rerun only the steps after resume_after. models is a
    comma-separated reduced step list. With wait=true the call blocks (a full run takes ~2 min, a
    500-household run ~1.5 min) and returns the manifest summary plus scorecard or error. With
    wait=false it returns the run_id immediately; poll get_run(run_id) until status is not "running".
    """
    try:
        runner.validate_args(label, sample_size=sample_size, resume_from=resume_from,
                             resume_after=resume_after, models=models)
    except runner.HarnessError as e:
        raise ToolError(str(e)) from e
    if wait:
        manifest = runner.run(label, sample_size=sample_size, resume_from=resume_from,
                              resume_after=resume_after, models=models)
        return _run_view(manifest["run_id"])

    run_id = runner.new_run_id()
    cmd = [sys.executable, "-m", "asim_harness.cli", "run", "--label", label, "--run-id", run_id]
    if sample_size is not None:
        cmd += ["--sample-size", str(int(sample_size))]
    if resume_from:
        cmd += ["--resume-from", resume_from, "--resume-after", resume_after]
    if models:
        cmd += ["--models", models]
    launch_dir = paths.runs_dir() / ".launch"
    launch_dir.mkdir(parents=True, exist_ok=True)
    log = open(launch_dir / f"{run_id}.log", "wb")
    env = os.environ.copy()
    for key in ("ASIM_HARNESS_ROOT", "ASIM_EXAMPLE_DIR", "ASIM_RUNS_DIR"):
        if key in os.environ:
            env[key] = os.environ[key]
    subprocess.Popen(cmd, cwd=paths.root(), env=env, stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    return {
        "run_id": run_id,
        "status": "running",
        "label": label,
        "poll": f"get_run('{run_id}') until run.status is 'succeeded' or 'failed'",
        "launcher_log": str(launch_dir / f"{run_id}.log"),
    }


@server.tool()
def get_run(run_id: str) -> dict[str, Any]:
    """Manifest summary of a run (accepts a unique id prefix), with its scorecard and error if present."""
    return _run_view(_resolve(run_id))


@server.tool()
def summarize_run(run_id: str, detail: str = "full") -> dict[str, Any]:
    """Metrics of a run from summary.json: counts, auto ownership, CDAP, tour/trip mode shares, tour frequency, checks.

    detail="overall" drops the by-purpose mode share blocks to keep the result small.
    """
    run_id = _resolve(run_id)
    try:
        summary = summarize.summarize_run(run_id)
    except summarize.SummaryError as e:
        raise ToolError(str(e)) from e
    if detail == "overall":
        summary = dict(summary)
        for key in ("tour_mode_share", "trip_mode_share"):
            summary[key] = {"overall": summary[key]["overall"]}
        summary["counts"] = {k: v for k, v in summary["counts"].items() if not k.endswith("_by_purpose")}
    return summary


@server.tool()
def check_targets(run_id: str) -> dict[str, Any]:
    """Score a run against targets/<model>.yaml. Failed metrics come with per-category deltas (run minus target) and n."""
    run_id = _resolve(run_id)
    try:
        return targets.compact(targets.check_run(run_id))
    except (targets.TargetsError, summarize.SummaryError) as e:
        raise ToolError(str(e)) from e


@server.tool()
def compare_runs(run_a: str, run_b: str) -> dict[str, Any]:
    """Per-metric deltas (b minus a) between two runs' summaries, with unchanged/drifted/missing verdicts."""
    try:
        return compare.compact(compare.compare_runs(_resolve(run_a), _resolve(run_b)))
    except summarize.SummaryError as e:
        raise ToolError(str(e)) from e


@server.tool()
def get_error(run_id: str, log_tail_lines: int = 20) -> dict[str, Any]:
    """error.json of a failed run: failed_step, exception, message, expression context, traceback tail."""
    run_id = _resolve(run_id)
    record = errors.error_for_run(run_id)
    if record is None:
        manifest = mf.read_manifest(paths.run_dir(run_id))
        return {"run_id": run_id, "status": manifest.get("status"), "error": None,
                "note": "this run did not fail" if manifest.get("status") != "running" else "still running"}
    return errors.compact(record, log_tail_lines=max(0, min(int(log_tail_lines), LOG_TAIL_MAX)))


@server.tool()
def get_log_tail(run_id: str, lines: int = 100) -> dict[str, Any]:
    """Last N lines of a run's ActivitySim log (falls back to the captured stdout)."""
    run_id = _resolve(run_id)
    run_dir = paths.run_dir(run_id)
    log = paths.find_log_file(run_dir / "output") or run_dir / "stdout.log"
    n = max(1, min(int(lines), LOG_TAIL_MAX))
    text = log.read_text(errors="replace").splitlines() if log.is_file() else []
    return {"run_id": run_id, "log_path": str(log), "total_lines": len(text), "lines": text[-n:]}


@server.tool()
def read_config(relative_path: str, max_chars: int = 20000) -> dict[str, Any]:
    """Read-only view of one file under example/configs (path relative to it; anything else is rejected)."""
    try:
        target = example.resolve_config_path(relative_path)
    except (ValueError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e
    text = target.read_text(errors="replace")
    limit = max(200, min(int(max_chars), CONFIG_MAX_CHARS))
    truncated = len(text) > limit
    return {"path": relative_path, "size": target.stat().st_size, "truncated": truncated,
            "content": text[:limit] if truncated else text}


@server.tool()
def list_configs() -> dict[str, Any]:
    """File names under example/configs with sizes (read-only)."""
    return {"configs_dir": str(paths.configs_dir()), "files": example.list_configs()}


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
