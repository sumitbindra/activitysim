# Block 4: the rest of the model

Blocks 1 to 3 ported every model step MTC and ARC run. This block ports
what was left: MWCOG's four person-level steps, and the output steps every
region runs (below). With both, **each region's whole model runs in
asim_core from the input files to its output files**, and every output file
matches 1.x's.

MWCOG's four person-level steps, which MWCOG and SEMCOG run and MTC and ARC
don't:

| Step | Chooses | Writes |
|---|---|---|
| `work_from_home` | workers (`CHOOSER_FILTER_COLUMN_NAME`, default `is_worker`) | `work_from_home` (bool), `is_out_of_home_worker`, and `workplace_zone_id` -1 for those working from home |
| `transit_pass_subsidy` | every person (or those passing `CHOOSER_FILTER_COLUMN_NAME`) | `transit_pass_subsidy` (the alternative's position, 0 for others, int64) |
| `transit_pass_ownership` | every person | `transit_pass_ownership` (the alternative's position, int32) |
| `telecommute_frequency` | persons with a workplace (`workplace_zone_id > -1`) | `telecommute_frequency` (the alternative's name, "" for others; a category) |

With them, **every model step of the three example regions runs in
asim_core**, and MWCOG's whole model runs from the input files alone, as
MTC's and ARC's do.

## Summary

- **All four match 1.x exactly**, each from 1.x's saved state and end to
  end (`run_milestone1.py`: ALL EXACT), on MWCOG's 100-household run (the
  whole model) and on all 29,972 households (40,137 persons; 1.x's run
  stops at `mandatory_tour_frequency` because of the 1.x bug in
  [mwcog.md](mwcog.md), so this one ends after `cdap_simulate`).
- **MWCOG on its own** (`run_standalone.py`, from the input files, nothing
  from 1.x): 310 of 310 columns exact after trip mode choice (100
  households); 245 of 245 after CDAP with all households (6.9 million
  values).
- **Time**: 0.07 to 0.13 s each with all households, against 0.16 to
  0.22 s in 1.x (its own log).
- **Built from the simple choice component** (as `free_parking`), with
  one addition: the preprocessor can run on every row before the choosers
  are picked, so the filter column may be one it makes (1.x's
  `transit_pass_subsidy` does this; the others filter first).

## Details

- The four share how 1.x runs them (`components/person_choice.py`): its
  `simple_simulate` on `persons_merged` (each person with their household,
  home zone and its accessibility), the persons' random stream, the
  utility shift only with `skip_failed_choices` off, and an optional
  preprocessor (MWCOG's work from home: whether others in the household
  are preschool children or non-working adults, with `other_than`).
- Each output keeps 1.x's type: work from home is True for the
  `WORK_FROM_HOME_ALT` alternative; the subsidy goes through
  `fillna(0).astype(int)` in 1.x (int64), ownership keeps the choice's
  int32; telecommute frequency is a category of the alternatives and "".
- Not supported yet (reported before the run): `WORK_FROM_HOME_ITERATIONS`
  above 1 (1.x runs the model again with a constant adjusted towards a
  target share), annotations after these models, and a preprocessor on
  another table before the subsidy's filter. No example uses them.
- Tests: the four against MWCOG's 1.x run, and the subsidy with a filter
  column its preprocessor makes (whether someone else in the household
  works, checked against pandas over every person).
- The harness reads a 1.x run's households sample size from 1.x's log when
  the run has no `run_info.json` (MWCOG's all-households run stopped early
  and has none; asim_core had read 100 households instead of all).

## Reproducing

```bash
python asim_core/harness/run_milestone1.py --profiles none --example-dir activitysim/examples/prototype_mwcog \
  --asim1-out asim_core/work/mwcog/asim1_legacy --asim1-sharrow-out none --out asim_core/work/mwcog/milestone1_b4
python asim_core/harness/run_milestone1.py --profiles none --example-dir activitysim/examples/prototype_mwcog \
  --asim1-out asim_core/work/mwcog/asim1_legacy_all --asim1-sharrow-out none --out asim_core/work/mwcog/milestone1_b4_all
python asim_core/harness/run_standalone.py --example-dir activitysim/examples/prototype_mwcog \
  --asim1-out asim_core/work/mwcog/asim1_legacy --through trip_mode_choice --out asim_core/work/mwcog/standalone_b4
python asim_core/harness/run_standalone.py --example-dir activitysim/examples/prototype_mwcog \
  --asim1-out asim_core/work/mwcog/asim1_legacy_all --through cdap_simulate --out asim_core/work/mwcog/standalone_b4_all
python -m pytest asim_core/tests/test_against_asim1.py -k "mwcog or subsidy"
```

The 1.x runs are made as in [mwcog.md](mwcog.md) (`--households 0` for
all households).

## The output steps

| Step | Writes | Compared with 1.x |
|---|---|---|
| `write_trip_matrices` | OMX files of trips by origin and destination (MTC: 5 files of 23 tables, by mode and period; MWCOG: 16 files) | every matrix, cell by cell; the zone mapping; the file's shape and version |
| `write_tables` | the final tables (`final_<table>.csv`), and `final_checkpoints.csv` | byte for byte (checkpoints: everything but the times) |
| `write_data_dictionary` | `data_dict.csv`, `data_dict.txt`: each table's columns, dtypes and the step that first wrote them | byte for byte |
| `track_skim_usage` | `skim_usage.txt`: the skims looked up, then those not | the same lines (1.x's order changes from run to run) |
| `summarize` (MTC) | 24 summaries in `summarize/` | byte for byte |

- **Every output file is exactly 1.x's**, with every step computed from the
  input files (`run_standalone.py`): MTC 40 of 40 files, MWCOG 27 of 27,
  ARC 16 of 16; and each output step from 1.x's saved state and end to end
  (`run_milestone1.py`: ALL EXACT; MTC 76 file comparisons, MWCOG 50, ARC
  28). Full MTC zones: 40 of 40 at 10,000 households, against a fresh 1.x
  run of the whole model (numexpr off, as exact matching needs); at 50,000
  households 38 of 40, the checkpoint table and skim usage differing only
  because that 1.x run was resumed (1.x re-records every table at the first
  checkpoint after a resume, and records skim lookups only after it).
- **Trip matrices**: the trips get the preprocessor's columns (counts per
  mode and period, by tour participants and occupancy), with origin to
  destination skims and the trip's period's skims (ARC's distance); a cell
  is the sum of its trips' values divided by their mean household sample
  rate, through the same pandas calls as 1.x, so the same floating-point
  sums. Written with PyTables in the layout the `openmatrix` package
  writes (`src/asim_core/omx.py`; `openmatrix` reads the files).
- **Tables**: through PyArrow as 1.x writes them (columns in order, row IDs
  last; with `recode_pipeline_columns`, zone positions turned back into
  the input's zone IDs, the decoded columns moved to the end). Byte for
  byte equal needs the same columns in the same order with the same types:
  two orders were changed to 1.x's for this (tour scheduling's `tdd` before
  `start`, `end`, `duration`; `initialize_households` making
  `households_skipped` before `persons` and `person_windows` after the
  size tables). The values were already 1.x's.
- **The run's history**: the checkpoint table and the data dictionary need
  each step's tables and their columns afterwards, and skim usage the skims
  each step looked up; the runner keeps one record per step
  (`StepRecord`), which steps see as `ctx.history`. Skim lookups are
  recorded as 1.x records them: the skim's name for a lookup by each row's
  period, (name, period) for one period's skim named in full.
- **Summaries**: 1.x's summaries are Python code over pandas tables, so
  they run as 1.x runs them: the same merged tables (persons, households,
  tours and trips with their joined tables), the same binning functions,
  each expression evaluated in order. The preprocessor (times, distances
  and costs per trip from skims) goes through asim_core's formulas.
- **Formula reading added**: an item of a constant
  (`time_periods['EA']['first_hour']`) and mapping by another table's
  column (`trips.tour_id.map(tours.number_of_participants)`, the same as
  `reindex`).
- Not supported yet (reported before the run): HDF5 table output,
  `output_tables` sorting, `SAVE_TRIPS_TABLE`, parking zones in the trip
  matrices, two- and three-zone systems, `EXPORT_PIPELINE_TABLES`.
- **1.x finding #22**: skim_usage.txt doesn't separate the skims looked up
  from those not, and a skim read for one fixed period (`('WLK_TRN_WLK_XWAIT',
  'MD')`) is also listed under its name as not looked up.

```bash
python asim_core/harness/run_standalone.py --through summarize --out asim_core/work/standalone_mtc_b4
python asim_core/harness/run_standalone.py --example-dir activitysim/examples/prototype_arc \
  --asim1-out asim_core/work/arc/asim1_legacy --through write_tables --out asim_core/work/arc/standalone_b4
python asim_core/harness/run_standalone.py --example-dir activitysim/examples/prototype_mwcog \
  --asim1-out asim_core/work/mwcog/asim1_legacy --through write_trip_matrices --out asim_core/work/mwcog/standalone_b4o
python -m pytest asim_core/tests -k "output_files or omx or constant_items"
# full MTC: a fresh 1.x run of the whole model (no resume), then asim_core against it
python asim_core/harness/run_asim1.py --example-dir asim_core/work/mtc_full/prototype_mtc_full \
  --out asim_core/work/mtc_full/asim1_10k_fresh --households 10000 --patch no_numexpr
python asim_core/harness/run_standalone.py --example-dir asim_core/work/mtc_full/prototype_mtc_full \
  --asim1-out asim_core/work/mtc_full/asim1_10k_fresh --read-skim-cache true --through summarize \
  --out asim_core/work/mtc_full/standalone_10k_outputs
```

