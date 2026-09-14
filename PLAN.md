# asim-harness — build plan

A minimal harness for running ActivitySim's shipped example (`prototype_mtc`) in a reproducible, agent-operable way. Every run gets an ID, a manifest, a scorecard, and a place in a ledger; a small CLI and an MCP server expose the same operations. No LLM code lives in this repo.

This file is the spec. Work through the phases in order. Do not start a phase until the previous phase's acceptance checks pass.

## How to work on this

* Keep `NOTES.md` at the repo root. After every phase (and any time you hit something surprising), append: what you did, what you verified, what drifted from this plan and why. Future sessions start by reading it.
* Commit at the end of each phase with the phase name in the message.
* ActivitySim's CLI flags, settings keys, and output file names drift between versions. Before writing code that depends on them, run `activitysim run --help` and read `example/configs/settings.yaml`. Where this plan names a flag or key, treat it as a starting point and correct it.
* Never edit files under `example/` after Phase 0. All variation is applied through override config directories (see Phase 1). Never modify ActivitySim itself.
* Prefer boring choices: plain Python, `pandas`, `pyyaml`, `click` or `typer`, `pytest`. No frameworks beyond that and the MCP SDK.
* If a step takes more than ~10 minutes of wall clock on the example data, stop and reconsider the sample size rather than waiting.

## Target layout

```
asim-harness/
  PLAN.md                 this file
  NOTES.md                running log (you create it)
  CLAUDE.md               playbook for agents operating the harness (Phase 5)
  README.md
  pyproject.toml
  src/asim_harness/
    __init__.py
    cli.py                `asim` entry point
    runner.py             build override dir, invoke activitysim, capture output
    manifest.py           run manifest + hashing
    ledger.py             runs index
    summarize.py          metrics from output tables
    compare.py            diff two runs' metrics
    targets.py            load targets.yaml, score a run
    errors.py             structured failure extraction from logs
    mcp_server.py         MCP tools (Phase 4)
  targets/prototype_mtc.yaml
  example/                created by `activitysim create`; data is gitignored
  runs/                   gitignored; one directory per run
  tests/
```

## Phase 0 — environment and baseline run

Goal: the example runs end to end from a clean environment, and you know how long it takes.

1. Create the repo skeleton above with `pyproject.toml` (package name `asim-harness`, module `asim_harness`, console script `asim`). Python 3.11 unless the installed ActivitySim version requires otherwise.
2. Create a virtual environment and install ActivitySim. Try `pip install activitysim` first. If native dependencies fail, follow the installation section of the ActivitySim docs (they document a conda/mamba route). Record the exact ActivitySim version and Python version in `NOTES.md`.
3. Create the example: `activitysim create -e prototype_mtc -d example` (check `activitysim create --help` for the current flags; `-l` lists examples). Confirm `example/configs`, `example/data`, and `example/output` (or whatever the created layout is) exist.
4. Run it once, unmodified, from inside `example/`: `activitysim run -c configs -o output -d data`. Record wall-clock time, the number of households in the input, and the list of output files produced.
5. Read `example/configs/settings.yaml` and record in `NOTES.md`: the `models` list (the ordered steps), whether `inherit_settings` is set, the household sampling key (expected: `households_sample_size`), the resume key (expected: `resume_after`), and the `output_tables` block.
6. Find the mode choice configs: the yaml for tour and trip mode choice, the spec CSVs they reference, and the coefficient CSVs they reference. Record the file names. Do not edit them.

Acceptance: a clean `activitysim run` completes; `NOTES.md` has the timing, version, step list, and the key names verified against the real files.

## Phase 1 — run wrapper with manifests and a ledger

Goal: `asim run` produces a self-describing run directory and never touches `example/`.

Design:

* Every run gets `run_id = <YYYYMMDD-HHMMSS>-<short random>` and a directory `runs/<run_id>/` containing:
   * `overrides/configs/` — an override config directory. It contains a `settings.yaml` with `inherit_settings: True` plus only the keys being overridden for this run (sample size, `resume_after`, trace household, a reduced `models` list if requested). Any other file placed here shadows the same-named file in `example/configs` because ActivitySim's `-c` flag accepts multiple config directories with earlier ones taking precedence. This is the whole edit mechanism for later phases: a changeset is a set of files in an override directory. Verify precedence order against `activitysim run --help` before relying on it.
   * `output/` — ActivitySim's output directory for this run.
   * `manifest.json` — see below.
   * `stdout.log`, `stderr.log` — captured subprocess streams.
* The command line assembled by `runner.py` is roughly `activitysim run -c runs/<id>/overrides/configs -c example/configs -o runs/<id>/output -d example/data`, run as a subprocess with the working directory and environment recorded.
* `manifest.json` fields: `run_id`, `label` (free text from the user, required), `created_at`, `finished_at`, `duration_s`, `status` (`running|succeeded|failed`), `exit_code`, `command`, `cwd`, `python_version`, `activitysim_version`, `harness_git_sha`, `settings_overrides` (the dict written to the override settings.yaml), `override_files` (names of any shadowing files), `config_hash` (sha256 over the sorted contents of every file in `example/configs` plus the override dir), `data_fingerprint` (file name, size, mtime for each file in `example/data`; hashing skims can be slow, so fingerprint rather than hash), `step_timings` (parsed from ActivitySim's timing log if present, else empty), `parent_run_id` (set when `--resume-from` is used).
* `ledger.py` maintains `runs/index.jsonl`, one line per run, rewritten from manifests by `asim reindex` and appended to by `asim run`.

CLI:

```
asim run --label "baseline full"                       # full example
asim run --label "smoke" --sample-size 500              # households_sample_size
asim run --label "mc only" --resume-from <run_id> --resume-after <step>
asim run --label "x" --models initialize_landuse,initialize_households,...
asim list                                               # table from the ledger
asim show <run_id>                                      # manifest, pretty
asim reindex
```

`--resume-from <run_id>` copies (or symlinks, if ActivitySim tolerates it) that run's pipeline file into the new run's output directory and sets `resume_after` in the override settings. Verify the pipeline file name and format from the Phase 0 output listing.

Acceptance:

* `asim run --label baseline` reproduces the Phase 0 run with identical output tables (compare row counts and a checksum of the sorted trips table).
* `asim run --label smoke --sample-size N` runs in under 3 minutes for some N you choose and record; note any steps that misbehave at small samples.
* `asim run --label mc --resume-from <baseline> --resume-after <the step before trip mode choice>` reruns only the tail of the model list and produces a trips table.
* `git status` shows nothing changed under `example/`.

## Phase 2 — summaries, targets, compare

Goal: a run can be reduced to a small dictionary of metrics, scored against targets, and diffed against another run.

`summarize.py` reads the output tables and returns a nested dict. Version 1 metrics, all computed from the final tables ActivitySim writes:

* `counts`: households, persons, tours, trips
* `auto_ownership_share`: distribution of the auto ownership column in the households table
* `cdap_share`: distribution of the CDAP activity pattern column in persons
* `tour_mode_share`: by tour mode, overall and by tour purpose
* `trip_mode_share`: by trip mode, overall and by trip purpose
* `tour_frequency`: mandatory and non-mandatory tour count distributions
* `checks`: boolean sanity checks — no null modes, no null destinations, every trip belongs to a tour, share vectors sum to 1 within 1e-6

Column names come from the actual output tables; discover them in Phase 0 and keep them in one place (`summarize.py` top-level constants) so an agency model can remap them later.

Write `runs/<id>/summary.json` on every successful run automatically.

`targets/prototype_mtc.yaml`: there is no observed data for the example, so bootstrap targets from the baseline run — `asim targets bootstrap <run_id>` writes the baseline's share metrics as targets with default tolerances (absolute share difference, 0.02 for mode shares, 0.03 for auto ownership and CDAP). Format:

```yaml
model: prototype_mtc
source_run: <run_id>
metrics:
  trip_mode_share.overall:
    tolerance_abs: 0.02
    targets: {<mode>: <share>, ...}
  auto_ownership_share:
    tolerance_abs: 0.03
    targets: {...}
```

`targets.py` scores a run: for each metric, per-category delta, max absolute delta, pass/fail against tolerance, and an overall pass/fail. Write `runs/<id>/scorecard.json`.

`compare.py` takes two run IDs and produces, for every metric present in both, the per-category deltas plus a one-line verdict per metric (unchanged within 0.005, drifted, or missing).

CLI:

```
asim summarize <run_id>            # prints the summary, compact
asim check <run_id>                 # scorecard against targets/<model>.yaml
asim compare <run_a> <run_b>
asim targets bootstrap <run_id>
```

Acceptance:

* Bootstrapped targets score the baseline run as a full pass.
* A 500-household sample run scores as a partial fail with sensible deltas (sampling noise), and `asim compare baseline smoke` shows those deltas.
* Round trip: `summarize` output is JSON-serialisable, and `check` reads only `summary.json`, never the raw tables.

## Phase 3 — structured failures

Goal: when a run fails, the harness says which step failed and why, in a form an agent can act on without reading a 2 MB log.

`errors.py`:

* Locate ActivitySim's log file for the run (find it under the run's output directory; the exact path varies by version) and the captured stderr.
* Extract: the last model step that started (from the log's step markers), the exception type and message, the traceback's last ~15 frames, and any expression-evaluation context ActivitySim printed (the spec file and the offending expression, when present).
* Write `runs/<id>/error.json` with `failed_step`, `exception_type`, `message`, `traceback_tail`, `expression_context`, `log_path`, `log_tail` (last 100 lines).

Manufacture a failure to develop against: an override `settings.yaml` that points a coefficient file name at a nonexistent file, or an override coefficients CSV with a deliberately malformed row. Keep this as a test fixture, not as an edit to `example/`.

CLI: `asim error <run_id>` prints `error.json` readably; `asim show` mentions it when present.

Acceptance: the manufactured failure yields an `error.json` whose `failed_step` and `message` are correct, and `asim run` exits nonzero with a one-paragraph explanation on stderr.

## Phase 4 — MCP server

Goal: an agent can do everything the CLI does through tools, with results sized for a context window.

Use the official Python MCP SDK (`mcp` on PyPI, the `FastMCP` server class). Entry point `asim mcp` runs it over stdio. Tools, each returning compact JSON (never raw tables, never more than a few KB):

* `list_runs(limit=20)` — ledger rows: run_id, label, status, sample size, duration, scorecard pass/fail if scored
* `run_model(label, sample_size=None, resume_from=None, resume_after=None, models=None, wait=True)` — starts a run. With `wait=True` it blocks and returns manifest summary + scorecard or error. With `wait=False` it returns the run_id immediately for polling; MCP clients enforce tool timeouts, so use `wait=False` for anything longer than a couple of minutes.
* `get_run(run_id)` — manifest summary, scorecard if present, error if present
* `summarize_run(run_id)`, `check_targets(run_id)`, `compare_runs(a, b)`
* `get_error(run_id)` and `get_log_tail(run_id, lines=100)`
* `read_config(relative_path)` — read-only, path must resolve inside `example/configs`; reject anything else
* `list_configs()` — file names under `example/configs` with sizes

No tool writes anything under `example/`. There is deliberately no edit tool in this phase.

Register the server for Claude Code in the repo's `.mcp.json` (project scope) so that opening Claude Code in this directory picks it up. Check the Claude Code documentation for the current `.mcp.json` shape.

Acceptance: from Claude Code, with the server registered, the tools appear, `list_runs` returns the ledger, and `run_model(label="mcp smoke", sample_size=<your N>)` completes and returns a scorecard.

## Phase 5 — playbook and agent smoke test

Goal: a fresh agent session can be told "run a smoke test and report the mode shares" and do it correctly without help.

Write `CLAUDE.md` containing, and nothing more:

* What this repo is, in three sentences.
* The model system map: the ordered step list from `settings.yaml`, one line each, with the mode choice steps marked.
* How to run: use the `asim` tools or CLI; always pass a descriptive `label`; use a sample size for iteration and confirm at full size; use `resume_from` + `resume_after` when only downstream steps changed.
* Rules: never edit files under `example/`; never edit `targets/*.yaml`; never delete run directories; report scorecards as deltas against targets, not as raw shares.
* Where things are: `runs/<id>/{manifest,summary,scorecard,error}.json`.

Then the smoke test. Open Claude Code in the repo and give it exactly:

> Run a smoke test at the sample size noted in NOTES.md, then tell me the trip mode shares and how they compare to targets.

Record in `NOTES.md` every point where the agent hesitated, called the wrong tool, read a raw file it didn't need, or misread a result. Each of those is the next item to fix in the harness or the playbook.

Acceptance: the agent completes the task using only the MCP tools, and its report matches `asim check` output.

## Tests

`pytest` throughout. Unit tests for manifest hashing, override generation, summarize (against a small fixture of output tables committed to `tests/`), targets scoring, compare, and error extraction (against a committed log fixture). One integration test, marked `slow`, that runs the example at the smoke sample size and asserts a scorecard is produced.

## Explicitly out of scope for this plan

* Editing coefficients or specs (the changeset/approval layer). The override directory design in Phase 1 is the seam it will plug into.
* Multiprocessing and chunking settings.
* Any model other than `prototype_mtc`.
* Hosting, web UI, authentication.
