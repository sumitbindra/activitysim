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
