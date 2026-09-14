# NOTES

Running log for the asim-harness build. Newest entries at the bottom. Future
sessions: read this first, then `PLAN.md` (the spec) and `CLAUDE.md` (the playbook).

## Phase 0 — environment and baseline run

### Environment

- Repo: this is `ActivitySim/activitysim-prototype-mtc`, the upstream of the
  example itself. The harness lives here on a branch; the repo's own root
  `configs/`, `data/`, `test/` are left untouched so its CI keeps working.
- Python 3.11.15, **ActivitySim 1.4.0**, installed with `uv sync --locked`
  (the repo already pinned `activitysim==1.4` in `uv.lock`; `pip install`
  was not needed). Wheels only, no conda. 4 CPUs, 15 GB RAM.
- `pyproject.toml` was rewritten for the harness (name `asim-harness`,
  module `asim_harness` under `src/`, console script `asim`, hatchling
  build) while keeping every dependency pin the example environment had.
  `uv.lock` regenerated with `uv lock`. Added deps: `click`, `pyyaml`, `mcp`.
- The `mcp` package resolved to **2.2.0**, where `FastMCP` was renamed
  `MCPServer` (`from mcp.server.mcpserver import MCPServer`). Phase 4 uses
  that name; the decorator API (`@server.tool()`, `server.run("stdio")`) is
  otherwise the same.
- No `/usr/bin/time` in the container; timing is done with `date` in the
  shell and `time.monotonic()` in the runner.

### Creating the example

- `activitysim create -l` lists `prototype_mtc` ("25-zone example extracted
  from the prototype MTC model"). It is bundled inside the wheel
  (`activitysim/examples/prototype_mtc`), so no network is needed.
- Drift: `activitysim create -e prototype_mtc -d example` nests the files at
  `example/prototype_mtc/...`. I created it into the scratch directory and
  moved `prototype_mtc/` to `example/` so the plan's `example/configs`,
  `example/data`, `example/output` layout holds. To recreate `example/data`
  on a fresh clone (data is gitignored):
  `activitysim create -e prototype_mtc -d /tmp/x && mv /tmp/x/prototype_mtc/data example/data`
  (Phase 1 wraps this as `asim init`).
- Created layout: `example/configs/` (plus a `configs/legacy-1.1.3/`
  subdir shipped by the package), `example/configs_mp/`, `example/data/`
  (`households.csv`, `persons.csv`, `land_use.csv`, `skims.omx`,
  `example_hwy_data.csv`, `override_hh_ids.csv`, `mtc_asim.h5`),
  `example/output/` (skeleton with `.gitignore`), `example/README.MD`.
- The packaged copy differs slightly from the repo root copy (root has
  parquet data and a handful of newer config edits). The harness uses the
  packaged copy in `example/`, as the plan says. Committed: `example/configs`,
  `example/configs_mp`, `example/README.MD`. Ignored: `example/data`,
  `example/output`.

### Baseline run

Command, run from inside `example/`:
`activitysim run -c configs -o output -d data`

- Exit 0. Wall clock **128.1 s** (ActivitySim reports 121.4 s for the 34
  models). Peak RSS 842 MB. Single process, `chunk_size: 0`, no sharrow.
- Input: **5000 households** (`households.csv`), 8212 persons, 25 zones.
  `households_sample_size: 100000` in `settings.yaml` exceeds the population,
  so the baseline is effectively a full run. Output: 9806 tours, 23583 trips.
- Slowest steps: `trip_destination` 37.7 s, `mandatory_tour_scheduling`
  16.3 s, `trip_scheduling` 10.4 s. Everything else is under 7 s.
- Output files produced in `example/output/`:
  - tables: `final_accessibility.csv`, `final_checkpoints.csv`,
    `final_households.csv`, `final_joint_tour_participants.csv`,
    `final_land_use.csv`, `final_persons.csv`, `final_tours.csv`,
    `final_trips.csv`
  - `cdap_spec_{2,3,4,5}.csv`, `data_dict.csv`, `data_dict.txt`,
    `skim_usage.txt`, `trips_{ea,am,md,pm,ev}.omx`
  - `pipeline.parquetpipeline/` (the checkpoint store: one subdir per table
    with `<checkpoint>.parquet` files, plus `checkpoints.parquet`)
  - `log/activitysim.log`, `log/timing_log.csv`, `log/mem.csv`
  - `summarize/*.csv` (the `summarize` step), `trace/` (shadow price files),
    `cache/cached_taz.mmap`
- ActivitySim creates `log/`, `trace/`, `summarize/` itself, so a run can
  point at an empty output directory.

### Facts verified against the real files (settings.yaml, CLI, source)

- `models` (ordered, 34 steps): initialize_landuse, initialize_households,
  compute_accessibility, school_location, workplace_location,
  auto_ownership_simulate, free_parking, cdap_simulate,
  mandatory_tour_frequency, mandatory_tour_scheduling, joint_tour_frequency,
  joint_tour_composition, joint_tour_participation, joint_tour_destination,
  joint_tour_scheduling, non_mandatory_tour_frequency,
  non_mandatory_tour_destination, non_mandatory_tour_scheduling,
  tour_mode_choice_simulate, atwork_subtour_frequency,
  atwork_subtour_destination, atwork_subtour_scheduling,
  atwork_subtour_mode_choice, stop_frequency, trip_purpose, trip_destination,
  trip_purpose_and_destination, trip_scheduling, trip_mode_choice,
  write_data_dictionary, track_skim_usage, write_trip_matrices, write_tables,
  summarize.
- `inherit_settings` is **not** set in `example/configs/settings.yaml`. It is
  the mechanism for override dirs: `read_settings_file` reads the first
  `settings.yaml` found in the config dir list and, only if that file has
  `inherit_settings: True`, backfills missing keys from the next dir's
  `settings.yaml`. Keys in the earlier file win. Block-valued keys
  (`output_tables`, `models`, ...) are replaced wholesale, not deep-merged,
  so an override of `output_tables` must be the complete block.
- `activitysim run --help` confirms: "Both '--config' and '--data' can be
  specified multiple times. Directories listed first take precedence."
- Household sampling key: `households_sample_size` (also the CLI flag
  `--households_sample_size N`). 0 means all households.
- Resume key: `resume_after` (also the CLI flag `-r/--resume STEPNAME`).
  `_` means "after the last checkpoint". A non-resume run first deletes
  csv/txt/yaml/prof/omx/h5 files in the output dir (`cleanup_output_files`);
  a resume run keeps them.
- Trace key: `trace_hh_id` (single id or empty).
- `checkpoint_format` defaults to `parquet`, so the pipeline is the directory
  `output/pipeline.parquetpipeline/` (name from `pipeline_file_name` =
  `pipeline`). `cleanup_pipeline_after_run` defaults to False, so it stays.
- `output_tables` block: `h5_store: False`, `action: include`,
  `prefix: final_`, tables = checkpoints, accessibility, land_use,
  households, persons, tours, trips (with `decode_columns` for zone ids),
  joint_tour_participants.
- Log: `output/log/activitysim.log` (from `configs/logging.yaml`, handler
  `logfile` with `get_log_file_path: activitysim.log`, mode `w`). Step start
  markers look like
  `... - INFO - activitysim.core.workflow.runner - #run_model running step <name>`.
- Timing: `output/log/timing_log.csv`, columns
  `process_name,model_name,seconds,minutes,notes`.

### Mode choice configs (not edited)

- Tour mode choice (`tour_mode_choice_simulate`, and reused by
  `atwork_subtour_mode_choice`): `tour_mode_choice.yaml` → `SPEC:
  tour_mode_choice.csv`, `COEFFICIENTS: tour_mode_choice_coefficients.csv`,
  `COEFFICIENT_TEMPLATE: tour_mode_choice_coefficients_template.csv`,
  preprocessor `tour_mode_choice_annotate_choosers_preprocessor.csv`.
  Nested logit, 21 alternatives (DRIVEALONEFREE ... TNC_SHARED).
- Trip mode choice (`trip_mode_choice`): `trip_mode_choice.yaml` → `SPEC:
  trip_mode_choice.csv`, `COEFFICIENTS: trip_mode_choice_coefficients.csv`,
  `COEFFICIENT_TEMPLATE: trip_mode_choice_coefficients_template.csv`,
  preprocessor `trip_mode_choice_annotate_trips_preprocessor.csv`.

### Output table columns that Phase 2 will use

- households: `auto_ownership` (0-4), `household_id`
- persons: `cdap_activity` (M/N/H), `mandatory_tour_frequency`
  (NaN, work1, work2, school1, school2, work_and_school), `num_mand`,
  `num_non_mand`, `non_mandatory_tour_frequency` (alternative index)
- tours: `tour_mode`, `primary_purpose`, `tour_type`, `tour_category`
  (mandatory / non_mandatory / atwork / joint), `destination`, `tour_id`
- trips: `trip_mode`, `purpose`, `primary_purpose`, `tour_id`, `destination`
- Baseline sanity: no null destinations, every trip's `tour_id` is in tours.
  Mode shares in this 25-zone extract are walk-heavy (WALK is 67% of trips).

## Phase 1 — run wrapper with manifests and a ledger

### What was built

- `src/asim_harness/`: `paths.py` (all locations, overridable with
  `ASIM_HARNESS_ROOT`, `ASIM_EXAMPLE_DIR`, `ASIM_RUNS_DIR`), `jsonio.py`
  (atomic JSON writes), `example.py` (read-only access to `example/`, plus
  `asim init`), `manifest.py` (manifest fields, config hash, data
  fingerprint, timing parse, git sha), `ledger.py` (`runs/index.jsonl`,
  file-locked writes, `reindex`, unique-prefix run id resolution),
  `runner.py` (override dir, subprocess, capture, finalize hooks), `cli.py`.
- `asim run` layout is exactly the plan's: `runs/<id>/overrides/configs/
  settings.yaml` (`inherit_settings: True` + overridden keys, plus any
  `--override-file` copies), `output/`, `manifest.json`, `stdout.log`,
  `stderr.log`. Command:
  `<venv python> -m activitysim run -c <overrides> -c example/configs -o <output> -d example/data`,
  cwd = the run directory, `PYTHONUNBUFFERED=1`. The subprocess uses the
  same interpreter as the harness (`python -m activitysim`), so there is no
  PATH dependency. The manifest records a filtered environment (only
  `ASIM_*`, `ACTIVITYSIM*`, `OMP_*`, `MKL_*`, `NUMBA_*`, `OPENBLAS_*`,
  `PYTHON*` variables) to avoid capturing secrets.
- A failed model run is a normal outcome (`status: failed`, nonzero
  `exit_code`), not an exception; the CLI exits 1 with an explanation.
- `asim init` recreates missing pieces of `example/` from the installed
  package (data is gitignored, so fresh clones need it).

### Verified

- `asim run --label "baseline full"`: exit 0, 123.4 s, 34 steps. Row counts
  and sha256 checksums of the sorted `final_households/persons/tours/trips`
  tables are identical to the Phase 0 run.
- **Smoke sample size: 500 households** (`--sample-size 500`): 90.7 s wall
  clock while a full run was executing concurrently (ActivitySim reported
  87.8 s for the models). That is under the 3-minute bar with room to
  spare. The cost at small samples is dominated by per-run fixed work:
  `trip_destination` 26 s, `mandatory_tour_scheduling` 10 s,
  `trip_scheduling` 9 s. Nothing misbehaves at 500 households: the log has
  147 warnings, all pandas `FutureWarning`s from ActivitySim itself (also
  present in the full run), and every step ran.
- Resume: `asim run --resume-from <baseline> --resume-after trip_scheduling`
  copied the parent's `pipeline.parquetpipeline/`, ran only the six tail
  steps (`trip_mode_choice` ... `summarize`) in 14.9 s, and its
  `final_trips.csv` is byte-for-byte identical to the parent's (ActivitySim
  restores the per-step random state from the checkpoint).
- `git status` shows nothing changed under `example/`.
- 29 unit tests (`pytest`), all in temp directories.

### Drift from the plan and why

- ActivitySim writes `activitysim.log`, `timing_log.csv` and `mem.csv` into
  `output/log/` only if that folder already exists (the packaged example
  ships it; a fresh run directory does not). The first harness runs had the
  logs in `output/` and no step timings. The runner now pre-creates
  `output/log/` and `output/trace/`, and `paths.find_log_file` /
  `find_timing_log` look in both places.
- The pipeline is copied, not symlinked: ActivitySim appends checkpoints to
  the store it resumes from, so a symlink would write into the parent run.
- `--resume-from` requires `--resume-after`: resuming after the parent's
  last checkpoint (`_`) of a completed run would run nothing. The step name
  is validated against the parent's `checkpoints.parquet` up front, so a
  typo fails in milliseconds instead of after loading the pipeline.
- Ledger writes take a file lock (`runs/index.jsonl.lock`) because two
  runs finishing at the same time (Phase 4 `wait=False`) would otherwise
  race on the rewrite.
- `sample_size: null` in a manifest means "the example's default", which
  is `households_sample_size: 100000`, i.e. all 5000 households.
- The two pre-fix runs from this phase were deleted before the commit; the
  ledger was rebuilt with `asim reindex`.

## Phase 2 — summaries, targets, compare

### What was built

- `summarize.py`: file and column names live in `TABLE_FILES` / `COLS` at
  the top. Metrics: `counts` (households, persons, tours, trips, plus
  `tours_by_purpose` and `trips_by_purpose` so every share vector has an
  `n`), `auto_ownership_share`, `cdap_share`, `tour_mode_share`
  (`overall`, `by_purpose` keyed by the tour's `primary_purpose`),
  `trip_mode_share` (`overall`, `by_purpose` keyed by the trip's own
  `purpose` column, so `home` is a purpose), `tour_frequency` (tours per
  person from the tours table, over all persons, binned 0/1/2/3+, for
  `tour_category` mandatory and non_mandatory) and `checks`. Shares are
  rounded to 6 decimals after the sum-to-one check. ~7.5 KB of JSON.
  Written automatically as `runs/<id>/summary.json` by a finalize hook.
- `targets.py`: `asim targets bootstrap <run_id>` writes every share
  vector of the run as a target (27 metrics: the 5 top-level vectors plus
  by-purpose tour and trip mode shares) with the plan's tolerances (0.02
  mode shares, 0.03 auto ownership and CDAP; tour_frequency 0.02) and the
  `n` behind each. Scoring reads only `summary.json` (a test monkeypatches
  the table reader to prove it). `scorecard.json` per run, and a scorecard
  is written automatically after a successful run when a targets file
  exists. `asim check --strict` exits 2 on failure for scripts.
- `compare.py`: deltas are `b - a` for every share vector present in
  either run; verdicts `unchanged` (max |delta| <= 0.005), `drifted`,
  `missing`.
- `asim summarize|check|compare|targets bootstrap|targets show`.

### Verified

- Baseline scored against its own bootstrapped targets: PASS, 0 of 27.
- The 500-household smoke run: FAIL, 19 of 27, all sampling noise:
  `trip_mode_share.overall` passes (max |delta| 0.014 on WALK, n=2279),
  `tour_mode_share.overall` just fails (0.021, n=961), auto ownership,
  CDAP and both tour-frequency vectors pass; the by-purpose cells with
  n around 30 (`univ`, `social`) miss by 0.11-0.14. `asim compare
  baseline smoke` shows the same deltas (26 drifted, 1 unchanged).
- The resumed mode-choice-only run scores PASS 0 of 27, as it must.
- Round trip: `summary.json` is plain JSON; `check` never opens the tables.
- 43 unit tests, including summarize against a committed 19-household
  fixture (`tests/fixtures/output/`, 32 KB, cut from the smoke run).

### Drift and observations

- The plan lists only the overall vectors in its target-file example; I
  also bootstrap the by-purpose mode shares because that is what a mode
  choice calibration needs. The price is that a 500-household run always
  fails those small cells. Every scorecard line carries `n`, and the
  playbook (Phase 5) must tell the agent to report deltas together with
  `n` and to confirm at full size. A tolerance that scales with `n` is a
  reasonable later improvement, but out of scope now.
- "Trip purpose" is the trip's `purpose` column (destination purpose,
  including `home`), not the tour's `primary_purpose`; that column is also
  in the trips table if a remap is wanted.
- `scorecard.json` (15 KB) keeps per-category deltas for every metric;
  the compact views used by the CLI text and the MCP tools drop passing
  metrics to one line and deltas below 0.0005.

## Phase 3 — structured failures

### What was built

- `errors.py` reads the run's `activitysim.log` (found via
  `paths.find_log_file`, so either `output/log/` or `output/`), plus
  `stdout.log` and `stderr.log`, and writes `runs/<id>/error.json` with
  `failed_step`, `exception_type`, `message`, `traceback_tail` (last 15
  frames, site-packages paths shortened, Python 3.11 caret lines dropped),
  `traceback_source` (log / stdout / stderr), `expression_context`,
  `exit_code`, `log_path`, `log_tail` (100 lines), `stderr_tail`, and
  `override_files` (which shadowing files this run carried, because a
  failure usually follows a config edit).
- Written automatically for failed runs by a finalize hook; `asim error
  <run_id>` prints it readably (`--json`, `--tail N`, `--force` to
  re-extract); `asim show` mentions it; the ledger row carries
  `failed_step`; `asim run` prints one paragraph on stderr and exits 1.
- Three manufactured failures, kept as fixtures under
  `tests/fixtures/failures/` and applied with `--override-file` on a
  resume after `trip_scheduling` (5 s each instead of 2 min):
  `missing_coefficients_file/trip_mode_choice.yaml` (COEFFICIENTS points
  at a file that does not exist), `malformed_coefficients/
  trip_mode_choice_coefficients.csv` (value `oops`), and
  `bad_preprocessor_expression/trip_mode_choice_annotate_trips_preprocessor.csv`
  (an expression using an undefined name). Trimmed real logs from the
  first and third are committed under `tests/fixtures/logs/` for the unit
  tests.

### What ActivitySim 1.4 actually does on failure (verified)

- The CLI catches the exception, prints the traceback to **stdout**, and
  exits **99**; `stderr` only has the `pkg_resources` deprecation warning.
- The log brackets the failure with `===== ERROR IN <step> =====`, the
  message, a traceback, `===== / =====`, then logs
  `activitysim run encountered an unrecoverable error` with the traceback
  again. `failed_step` comes from that marker, falling back to the last
  `#run_model running step` line.
- Expression failures are logged as `<trace_label> - <ExcType> (<msg>)
  evaluating: <expression>` (utilities, interaction, `assign_variables`),
  `Variable evaluation failed ...`, or `assign_variables expression:
  <target> = <expr>`; spec-file problems as `Error reading spec file:
  <path>`, `read_model_spec error reading <path>`, `Coefficient File
  Invalid: <path>`; a missing config file as `FileNotFoundError: ... file
  '<name>' not in [<config dirs>]`. All of these feed `expression_context`.
- A malformed coefficient value does not get an expression context: the
  error is a pandas `ValueError: could not convert string to float:
  'oops'` raised while reading the file, with no file name in the message.
  The `override_files` list in `error.json` is what points at the culprit.

### Verified

- `missing_coefficients_file`: `failed_step` = `trip_mode_choice`,
  `FileNotFoundError`, message names the missing file, `missing_file`
  extracted; `asim run` exit 1 with the paragraph above; `error.json`
  written by the hook; ledger shows `failed_step`.
- `bad_preprocessor_expression`: `NameError: name 'no_such_column_xyz' is
  not defined`, `expression_context.expression` = `no_such_column_xyz + 1`,
  `trace_label` = `assign_variables`.
- 50 unit tests pass (7 new for `errors.py`, against the committed logs,
  including the stdout fallback and the no-log case).
- The extra failure runs were deleted; one failed run
  (`bad_preprocessor_expression`) is kept in the ledger as a live example.

### Drift

- My first attempt at the expression fixture produced a malformed CSV
  instead (the example file has no trailing newline, so an appended row
  merged with the last line). Harmless, but a reminder that override CSVs
  need a newline check; the fixture build now writes one explicitly.

## Phase 4 — MCP server

### What was built

- `mcp_server.py` on the official `mcp` SDK, version 2.2.0. Drift: the plan
  names `FastMCP`; in 2.x that class is `MCPServer`
  (`from mcp.server.mcpserver import MCPServer`), same decorator API.
  Two 2.x details that matter: a tool must annotate its return as
  `dict[str, Any]` (a bare `dict` gives text-only results, no
  `structured_content`), and anticipated failures must raise
  `mcp.server.mcpserver.exceptions.ToolError`, otherwise the agent only sees
  "Error executing tool <name>" and the message stays on the server. Sync
  tool functions run in a worker thread, so a blocking `run_model` does not
  freeze the server.
- Tools, all returning compact JSON: `list_runs`, `run_model` (`wait=true`
  blocks; `wait=false` validates the arguments up front, then detaches
  `asim run --run-id <id>` with `start_new_session=True`, launcher output in
  `runs/.launch/<id>.log`, and returns the id to poll), `get_run` (manifest
  summary + compact scorecard + compact error + failure explanation),
  `summarize_run` (`detail="overall"` drops the by-purpose blocks),
  `check_targets`, `compare_runs`, `get_error`, `get_log_tail` (capped at
  500 lines), `read_config` (sandboxed to `example/configs`, rejects `..`,
  absolute and drive paths), `list_configs`. No tool writes under `example/`;
  there is no edit tool.
- `asim mcp` runs it over stdio. `.mcp.json` (project scope, checked against
  the Claude Code docs at code.claude.com/docs/en/mcp): `type: "stdio"`,
  `command: ".venv/bin/asim"` (relative commands resolve against the project
  directory, which is also the server's cwd), `args: ["mcp"]`, and
  `ASIM_HARNESS_ROOT=${CLAUDE_PROJECT_DIR}` in `env`. Claude Code asks for
  approval of a project `.mcp.json` on first use.
- The runner's argument validation was factored into `runner.validate_args`
  so the server can fail fast before detaching a run; `asim run` gained a
  hidden `--run-id`.

### Verified

- Through a real stdio client (`mcp.client.stdio`), against
  `.venv/bin/asim mcp` exactly as `.mcp.json` launches it: the 10 tools are
  listed; `list_runs` returns the ledger; `run_model(label="mcp smoke",
  sample_size=500, wait=false)` returned a run id at once and, after 10
  polls of `get_run` (100 s; the run itself took 91.5 s), the run was
  `succeeded` with the scorecard attached (FAIL 19 of 27, identical to the
  CLI smoke run since ActivitySim is deterministic for a given sample); the
  result was 10.5 KB. Escaping paths, unknown run ids, an empty label and a
  `resume_from` without `resume_after` all come back as `is_error` results
  with the harness's own message.
- `tests/test_mcp_server.py` drives the server over stdio against a temp
  runs dir and checks every tool, the sandbox, error surfacing, and result
  sizes (< 20 KB each). 51 unit tests pass in about 4 s.
- `tests/test_integration.py` (marked `slow`): a real 500-household run
  producing summary + scorecard, and a real failure-fixture run producing
  `error.json`.
- Addendum: the blocking path was verified too: `run_model(label="mcp resume
  check: mode choice only", resume_from=<baseline>,
  resume_after="trip_scheduling", wait=true)` returned after 15 s with
  `status: succeeded` and a passing scorecard (4.7 KB).

## Phase 5 — playbook and agent smoke test

### What was built

- `CLAUDE.md` with exactly the plan's five sections: what the repo is (three
  sentences), the 34-step model system map with the three mode choice steps
  marked (`tour_mode_choice_simulate`, `atwork_subtour_mode_choice`,
  `trip_mode_choice`), how to run, the four rules, and where things are.
- Drift: this session cannot open an interactive Claude Code window, and
  the branch could not be pushed (see below), so a new remote session could
  not be started either. The smoke test was run instead with a fresh
  general-purpose subagent given the exact prompt from the plan plus one
  framing paragraph: "the MCP tools are available through
  `.venv/bin/asim tool <name> '<json>'`". `asim tool` is a new stdio client
  shim (`mcp_client.py`) that talks to the real `asim mcp` server, returns
  exactly what an MCP client gets, and appends every call to
  `runs/.tool-calls.log`, which gave an objective record of the agent's
  tool use next to its own self-reported trace.

### Smoke test, first run (before fixes)

Prompt, verbatim: "Run a smoke test at the sample size noted in NOTES.md,
then tell me the trip mode shares and how they compare to targets."

- Outcome: correct. The agent read `CLAUDE.md` and `NOTES.md`, found the
  500-household smoke size, ran `run_model(label=..., sample_size=500,
  wait=true)` (91.1 s), then `summarize_run` and `check_targets`, and
  reported the overall trip mode shares with signed deltas per mode (WALK
  −0.014, n=2279, PASS at tolerance 0.02), the by-purpose cells that fail
  from sampling noise (n between 29 and 139), and the 19-of-27 overall
  verdict with the right interpretation. Every number matched
  `asim check` on the same run.
- Points to fix, from the agent's trace and the call log:
  1. It read `targets/prototype_mtc.yaml` and grepped
     `runs/<id>/scorecard.json` from disk because the compact scorecard
     returned by `run_model`, `get_run` and `check_targets` dropped
     per-category deltas for *passing* metrics, and the one metric the
     user asked about had passed. So the task was completed with two raw
     file reads, not with the MCP tools alone.
  2. `asim tool --list` showed tool names and descriptions but no
     parameter schemas, so it guessed `run_model`'s argument names from the
     playbook examples (correctly). A real MCP client shows schemas; the
     shim did not.
  3. It hesitated over `wait=true` versus `wait=false`: the playbook said
     "longer than a couple of minutes" and a 500-household run is about 90 s.
  4. It called `check_targets` after `run_model` had already attached the
     same scorecard (harmless, one redundant call).
- Fixes made:
  1. `scorecard.json` now stores per-category `actual` and `target` next to
     `deltas`; `check_targets(run_id, detail="failed"|"all", metric=<name
     or prefix>)` returns `{category: {share, target, delta}}` for every
     selected metric, passing ones included; the default view still keeps
     passing metrics to one line. Existing runs were re-scored.
  2. `asim tool --list` prints each tool's parameters with types and
     defaults from the input schema.
  3. `CLAUDE.md` now says a 500-household run or a resume is fine with
     `wait=true`, full runs should use `wait=false`, that `run_model` and
     `get_run` already carry the scorecard, and that `check_targets` with
     `metric=` or `detail="all"` replaces opening `scorecard.json` or the
     targets file.

### Smoke test, second run (after the fixes)

Same prompt, same framing, a fresh agent, the call log emptied first.

- Outcome: correct, and this time with the MCP tools alone. The call log
  shows five calls, all successful: `run_model(label=..., sample_size=500,
  wait=true)` (run `20260914-184658-f83646`, 89.2 s), `list_runs`,
  `check_targets(run_id, metric="trip_mode_share")`, `compare_runs` against
  the previous smoke run (27 of 27 unchanged: ActivitySim is deterministic
  for a given sample), and `summarize_run(detail="overall")`. No
  `scorecard.json` or targets file was opened. The report gave every
  overall trip mode with share, target and delta (WALK −1.4 pp, n=2279,
  PASS) and the by-purpose table with n and worst delta, and explained the
  19-of-27 verdict as sampling noise at n ≤ 139. Every number matches
  `asim check 20260914-184658-f83646`.
- Remaining friction from its trace:
  1. The `run_model` result was about 21 KB of JSON (the shell tool showed
     the agent a truncated preview, so it re-read the saved result twice):
     19 failed metrics with their categories, plus 34 step timings. Fixed:
     `run_model` / `get_run` now attach a category-free scorecard summary
     (`detail="summary"`, one line per metric) and only the eight slowest
     steps with a total; the view is 6 KB. `check_targets` keeps the
     per-category detail.
  2. It worried that a client timeout during `wait=true` might orphan the
     run. It could have: the run was a child of the server process. Fixed:
     `run_model` now always launches `asim run --run-id <id>` in its own
     session (`start_new_session=True`) and, with `wait=true`, polls the
     manifest every 2 s (up to an hour) before returning the run view. A
     run therefore survives a client timeout or a server exit, and
     `list_runs` / `get_run` show it. `CLAUDE.md` says so. Verified through
     the shim with a resume run (15 s, scorecard PASS, 5.9 KB result).
  3. It noticed `NOTES.md` change under it and spent two commands checking
     why: that was me committing the first-run notes during its session,
     not a harness problem.
  4. It judged, reasonably, that "confirm at full size" did not require a
     new full run because the targets are the full-size baseline of the
     identical configs and that run is in the ledger as PASS 0 of 27.

### Acceptance

- Both runs completed the task; the second used only the MCP tools and its
  report matches `asim check` output. 52 unit tests and the 2 slow
  integration tests pass.

## State at hand-off

- Branch `claude/eager-cerf-vf3yt8`, all phases committed locally. Pushing
  is refused with HTTP 403: "Claude doesn't have GitHub access to
  ActivitySim/activitysim-prototype-mtc for your organization". An org
  admin can install the Claude GitHub App
  (https://github.com/apps/claude/installations/select_target) or the user
  can reconnect GitHub under claude.ai settings; then
  `git push -u origin claude/eager-cerf-vf3yt8` from this checkout.
- Ledger (`runs/`, not committed): the full baseline
  `20260914-181919-339aa0` (targets source), 500-household smoke runs, a
  mode-choice-only resume, one kept failure example
  (`bad_preprocessor_expression`), and the two agent smoke-test runs.
- Fresh clone: `uv sync --locked && source .venv/bin/activate && asim init`
  (recreates `example/data`), then `asim run --label "baseline full"` and
  `asim targets bootstrap <run_id>` only if `targets/prototype_mtc.yaml`
  should be re-derived (it is committed, bootstrapped from the baseline
  above; deterministic, so a new baseline scores PASS against it).
- Next items, in order of value: a tolerance that scales with `n` (the
  by-purpose cells at 500 households can never pass at 0.02); an
  `asim tool`-style shim is not needed once a real Claude Code session
  loads `.mcp.json`, but keep it for CI; the changeset/approval layer the
  plan leaves out plugs into `--override-file` and `override_files`.
