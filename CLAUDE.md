# asim-harness playbook

## What this repo is

This repo is a harness for running ActivitySim's shipped `prototype_mtc`
example (25 zones, 5000 households) reproducibly: every run gets a run id, a
directory under `runs/`, a manifest, a metrics summary, and a scorecard
against targets or a structured error. The `asim` CLI and the `asim` MCP
server (registered in `.mcp.json`) expose the same operations; prefer the MCP
tools when they are available. The example itself under `example/` is
read-only; all variation goes through per-run override config directories.

## Model system map (the `models` list in `example/configs/settings.yaml`, in order)

1. `initialize_landuse` — load and annotate the land use table
2. `initialize_households` — load, sample and annotate households and persons
3. `compute_accessibility` — zone accessibility measures from the skims
4. `school_location` — school zone for students
5. `workplace_location` — workplace zone for workers
6. `auto_ownership_simulate` — household vehicles (0-4)
7. `free_parking` — free parking at work
8. `cdap_simulate` — coordinated daily activity pattern (M/N/H per person)
9. `mandatory_tour_frequency` — work/school tours per person
10. `mandatory_tour_scheduling` — departure and duration of mandatory tours
11. `joint_tour_frequency` — household joint tours
12. `joint_tour_composition` — adults/children on joint tours
13. `joint_tour_participation` — who joins each joint tour
14. `joint_tour_destination` — destination of joint tours
15. `joint_tour_scheduling` — timing of joint tours
16. `non_mandatory_tour_frequency` — escort/shop/maintenance/eat/social/discretionary tours
17. `non_mandatory_tour_destination` — destination of non-mandatory tours
18. `non_mandatory_tour_scheduling` — timing of non-mandatory tours
19. `tour_mode_choice_simulate` — **TOUR MODE CHOICE** (`tour_mode_choice.yaml`, `tour_mode_choice.csv`, `tour_mode_choice_coefficients.csv`)
20. `atwork_subtour_frequency` — subtours from work
21. `atwork_subtour_destination` — destination of at-work subtours
22. `atwork_subtour_scheduling` — timing of at-work subtours
23. `atwork_subtour_mode_choice` — **MODE CHOICE** for at-work subtours (reuses `tour_mode_choice.yaml`)
24. `stop_frequency` — intermediate stops per tour
25. `trip_purpose` — purpose of each intermediate stop
26. `trip_destination` — destination of intermediate stops (the slowest step)
27. `trip_purpose_and_destination` — retry for stops with no destination
28. `trip_scheduling` — departure time of each trip
29. `trip_mode_choice` — **TRIP MODE CHOICE** (`trip_mode_choice.yaml`, `trip_mode_choice.csv`, `trip_mode_choice_coefficients.csv`)
30. `write_data_dictionary` — documentation of the pipeline tables
31. `track_skim_usage` — which skims were used
32. `write_trip_matrices` — trip OMX matrices by time period
33. `write_tables` — `final_households/persons/tours/trips.csv` (what the summary reads)
34. `summarize` — ActivitySim's own summary CSVs

## How to run

- Use the `asim` MCP tools (`list_runs`, `run_model`, `get_run`,
  `summarize_run`, `check_targets`, `compare_runs`, `get_error`,
  `get_log_tail`, `read_config`, `list_configs`) or the CLI (`asim run`,
  `asim list`, `asim show`, `asim summarize`, `asim check`, `asim compare`,
  `asim error`).
- Always pass a descriptive `label` saying what the run is for.
- Iterate with a sample size and confirm at full size: the smoke sample size
  is 500 households (`sample_size=500`, about 1.5 minutes); the full example
  (`sample_size` omitted) is 5000 households, about 2 minutes. At 500
  households the by-purpose mode share metrics will miss their targets from
  sampling noise alone; read the `n` on each metric before drawing
  conclusions, and confirm at full size.
- A run longer than a couple of minutes should be started with
  `run_model(..., wait=false)` and polled with `get_run(run_id)` until
  `run.status` is `succeeded` or `failed`.
- When only downstream steps changed, pass `resume_from=<run_id>` and
  `resume_after=<last step to keep>`: for mode choice work,
  `resume_after="trip_scheduling"` reruns only `trip_mode_choice` and the
  writers (about 15 seconds).
- A successful run already has its summary and scorecard; `check_targets`
  and `summarize_run` return them without touching the output tables.

## Rules

- Never edit files under `example/`.
- Never edit `targets/*.yaml`.
- Never delete run directories.
- Report scorecards as deltas against targets (run minus target, with the
  tolerance and `n`), not as raw shares.

## Where things are

- `runs/<run_id>/manifest.json` — what ran (label, command, overrides, config
  hash, step timings, status, exit code)
- `runs/<run_id>/summary.json` — counts, share vectors, sanity checks
- `runs/<run_id>/scorecard.json` — per-metric deltas vs `targets/prototype_mtc.yaml`
- `runs/<run_id>/error.json` — failed step, exception, expression context,
  traceback tail (failed runs only)
- `runs/<run_id>/output/log/activitysim.log`, `runs/<run_id>/stdout.log`,
  `runs/<run_id>/stderr.log` — raw logs
- `runs/index.jsonl` — the ledger; `NOTES.md` — the build log, including the
  smoke sample size
