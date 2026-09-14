"""`asim` command line entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import __version__, compare, errors, example, ledger, paths, runner, summarize, targets
from . import manifest as mf
from .jsonio import dumps, read_json_if_exists


def _resolve(run_id: str) -> str:
    try:
        return ledger.resolve_run_id(run_id)
    except KeyError as e:
        raise click.ClickException(str(e)) from e


def _fmt(value, width: int) -> str:
    text = "-" if value is None else str(value)
    return text[:width].ljust(width)


@click.group()
@click.version_option(__version__, prog_name="asim")
def main():
    """Run and inspect ActivitySim prototype_mtc runs (asim-harness)."""


@main.command()
def init():
    """Recreate missing pieces of example/ from the installed ActivitySim package."""
    created = example.init_example()
    if created:
        click.echo(f"created under {paths.example_dir()}: {', '.join(created)}")
    else:
        click.echo(f"example is complete at {paths.example_dir()}")


@main.command()
@click.option("--label", required=True, help="Free-text description of the run (required).")
@click.option("--sample-size", type=int, default=None, metavar="N",
              help="households_sample_size override (0 = all households).")
@click.option("--resume-from", default=None, metavar="RUN_ID",
              help="Reuse this run's pipeline; requires --resume-after.")
@click.option("--resume-after", default=None, metavar="STEP",
              help="Last step to keep from the parent run ('_' = its last checkpoint).")
@click.option("--models", default=None, metavar="A,B,C", help="Reduced, comma-separated models list.")
@click.option("--trace-hh-id", type=int, default=None, metavar="ID", help="trace_hh_id override.")
@click.option("--override-file", "override_files", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="File that shadows the same-named file in example/configs (repeatable).")
@click.option("--run-id", default=None, hidden=True, help="Use this run id (set by the MCP server for detached runs).")
def run(label, sample_size, resume_from, resume_after, models, trace_hh_id, override_files, run_id):
    """Run the example into a new runs/<run_id>/ directory."""

    def started(m):
        click.echo(f"run {m['run_id']} started ({m['label']}); progress: tail -f "
                   f"{paths.run_dir(m['run_id']) / 'stdout.log'}", err=True)

    try:
        m = runner.run(
            label, sample_size=sample_size, resume_from=resume_from, resume_after=resume_after,
            models=models, trace_hh_id=trace_hh_id, override_files=list(override_files) or None,
            run_id=run_id, on_start=started,
        )
    except runner.HarnessError as e:
        raise click.ClickException(str(e)) from e
    run_dir = paths.run_dir(m["run_id"])
    if m["status"] == "succeeded":
        click.echo(f"{m['run_id']} succeeded in {m['duration_s']} s "
                   f"({len(m.get('step_timings') or {})} steps); {run_dir}")
        scorecard = read_json_if_exists(run_dir / "scorecard.json")
        if scorecard is not None:
            click.echo(f"scorecard: {'PASS' if scorecard.get('passed') else 'FAIL'} "
                       f"({scorecard.get('n_failed', 0)} of {scorecard.get('n_metrics', 0)} metrics out of tolerance)")
        return
    click.echo(runner.failure_explanation(run_dir, m), err=True)
    sys.exit(1)


@main.command("list")
@click.option("--limit", default=20, show_default=True, help="Newest N runs (0 = all).")
@click.option("--json", "as_json", is_flag=True, help="Print JSON rows instead of a table.")
def list_cmd(limit, as_json):
    """Runs from the ledger, newest first."""
    rows = ledger.list_runs(limit or None)
    if as_json:
        click.echo(dumps(rows))
        return
    if not rows:
        click.echo(f"no runs yet (ledger: {paths.ledger_path()})")
        return
    header = (f"{_fmt('run_id', 22)} {_fmt('status', 9)} {_fmt('sample', 7)} "
              f"{_fmt('dur_s', 7)} {_fmt('pass', 5)} {_fmt('resume', 12)} label")
    click.echo(header)
    for r in rows:
        passed = r.get("scorecard_pass")
        click.echo(
            f"{_fmt(r.get('run_id'), 22)} {_fmt(r.get('status'), 9)} {_fmt(r.get('sample_size'), 7)} "
            f"{_fmt(r.get('duration_s'), 7)} "
            f"{_fmt('-' if passed is None else ('yes' if passed else 'no'), 5)} "
            f"{_fmt(r.get('resume_after'), 12)} {r.get('label') or ''}"
        )


@main.command()
@click.argument("run_id")
@click.option("--full", is_flag=True, help="Print the whole manifest, including fingerprints and env.")
def show(run_id, full):
    """Manifest of a run (accepts a unique run id prefix)."""
    run_id = _resolve(run_id)
    run_dir = paths.run_dir(run_id)
    manifest = mf.read_manifest(run_dir)
    if full:
        click.echo(dumps(manifest))
    else:
        view = mf.summary_of(manifest)
        view["step_timings"] = manifest.get("step_timings") or {}
        view["command"] = manifest.get("command")
        view["run_dir"] = str(run_dir)
        if manifest.get("finalize_errors"):
            view["finalize_errors"] = manifest["finalize_errors"]
        click.echo(dumps(view))
    present = [n for n in ("summary.json", "scorecard.json", "error.json") if (run_dir / n).exists()]
    if present:
        click.echo(f"also present: {', '.join(present)}", err=True)
    error = read_json_if_exists(run_dir / "error.json")
    if error:
        click.echo(f"failed in step {error.get('failed_step')}: {error.get('exception_type')}: "
                   f"{(error.get('message') or '').splitlines()[0] if error.get('message') else ''} "
                   f"(see `asim error {run_id}`)", err=True)


@main.command("summarize")
@click.argument("run_id")
@click.option("--json", "as_json", is_flag=True, help="Print the full summary as JSON.")
@click.option("--force", is_flag=True, help="Recompute from the output tables even if summary.json exists.")
def summarize_cmd(run_id, as_json, force):
    """Metrics of a run (counts, shares, sanity checks); writes runs/<id>/summary.json."""
    try:
        summary = summarize.summarize_run(_resolve(run_id), force=force)
    except summarize.SummaryError as e:
        raise click.ClickException(str(e)) from e
    click.echo(dumps(summary) if as_json else summarize.compact_text(summary))


@main.command()
@click.argument("run_id")
@click.option("--targets", "targets_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Targets file (default: targets/<model>.yaml).")
@click.option("--json", "as_json", is_flag=True, help="Print the scorecard as JSON.")
@click.option("--strict", is_flag=True, help="Exit 2 when the scorecard fails.")
def check(run_id, targets_path, as_json, strict):
    """Score a run's summary.json against the targets; writes runs/<id>/scorecard.json."""
    try:
        card = targets.check_run(_resolve(run_id), targets_path)
    except (targets.TargetsError, summarize.SummaryError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(dumps(card) if as_json else targets.text(card))
    if strict and not card["passed"]:
        sys.exit(2)


@main.command("compare")
@click.argument("run_a")
@click.argument("run_b")
@click.option("--threshold", default=compare.UNCHANGED_TOLERANCE, show_default=True,
              help="Max abs share delta still reported as unchanged.")
@click.option("--json", "as_json", is_flag=True)
def compare_cmd(run_a, run_b, threshold, as_json):
    """Per-metric deltas (b - a) between two runs' summaries."""
    try:
        result = compare.compare_runs(_resolve(run_a), _resolve(run_b), threshold)
    except summarize.SummaryError as e:
        raise click.ClickException(str(e)) from e
    click.echo(dumps(result) if as_json else compare.text(result))


@main.group("targets")
def targets_group():
    """Manage targets/<model>.yaml."""


@targets_group.command("bootstrap")
@click.argument("run_id")
@click.option("--force", is_flag=True, help="Overwrite an existing targets file.")
def targets_bootstrap(run_id, force):
    """Write the run's share metrics as targets with default tolerances."""
    try:
        doc = targets.bootstrap(_resolve(run_id), force=force)
    except (targets.TargetsError, summarize.SummaryError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"wrote {doc['_path']}: {len(doc['metrics'])} metrics from run {doc['source_run']}")


@targets_group.command("show")
def targets_show():
    """Print the targets file."""
    try:
        doc = targets.load_targets()
    except targets.TargetsError as e:
        raise click.ClickException(str(e)) from e
    click.echo(Path(doc["_path"]).read_text())


@main.command()
@click.argument("run_id")
@click.option("--json", "as_json", is_flag=True, help="Print error.json (without the log tail).")
@click.option("--tail", "tail_lines", default=15, show_default=True, help="Log lines to show.")
@click.option("--force", is_flag=True, help="Re-extract from the logs even if error.json exists.")
def error(run_id, as_json, tail_lines, force):
    """What failed in a run: step, exception, expression context, traceback tail."""
    run_id = _resolve(run_id)
    record = errors.error_for_run(run_id, force=force)
    if record is None:
        click.echo(f"run {run_id} did not fail; no error record")
        return
    click.echo(dumps(errors.compact(record, tail_lines)) if as_json else errors.text(record, tail_lines))


@main.command()
def mcp():
    """Serve the harness as MCP tools over stdio (registered in .mcp.json)."""
    from .mcp_server import main as serve  # slow import, keep local

    serve()


@main.command()
@click.argument("name", required=False)
@click.argument("arguments", required=False)
@click.option("--list", "list_only", is_flag=True, help="List the server's tools and exit.")
@click.option("--timeout", default=3600.0, show_default=True, help="Seconds to wait for the tool.")
def tool(name, arguments, list_only, timeout):
    """Call one MCP tool through a stdio client: asim tool get_run '{"run_id": "2026..."}'."""
    from . import mcp_client  # slow import, keep local

    if list_only or not name:
        for t in mcp_client.list_tools():
            first = t["description"].splitlines()[0] if t["description"] else ""
            click.echo(f"{t['name']:<15s} {first}")
        return
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        raise click.ClickException(f"arguments must be a JSON object: {e}") from e
    if not isinstance(args, dict):
        raise click.ClickException("arguments must be a JSON object")
    is_error, payload = mcp_client.call_tool(name, args, timeout=timeout)
    if is_error:
        click.echo(payload if isinstance(payload, str) else dumps(payload), err=True)
        sys.exit(1)
    click.echo(payload if isinstance(payload, str) else dumps(payload))


@main.command()
def reindex():
    """Rebuild runs/index.jsonl from every run's manifest."""
    rows = ledger.reindex()
    click.echo(f"indexed {len(rows)} runs into {paths.ledger_path()}")


if __name__ == "__main__":
    main()
