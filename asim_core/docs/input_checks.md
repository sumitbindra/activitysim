# Input tables: definitions and checks before the run

Part 1 of the configuration plan ([config_plan.md](config_plan.md)).
Before the first step runs, asim_core reads the region's input tables
and skims and checks them. Errors (a negative employment count, a
household in a zone that doesn't exist, a household size that doesn't
match its persons, a missing skim value) stop the run, with a report that
says what is wrong, where, and which model files use the column. Warnings
are reported and the run goes on.

## Summary

- **Every injected error is caught on all three regions, and the clean
  example data passes with no false alarms.** Seven kinds of error were
  injected, one at a time, into a copy of each region's data (21 cases,
  `tests/test_input_checks.py`). Each report names that error and
  nothing else.
- **Fast enough to run every time.** A population 100 times ARC's
  (2.4 million households, 6.7 million persons) is read and checked in
  **2.1 s** at **2.8 GB** peak. Reading the same files with pandas, as 1.x
  does, takes 8.2 s and 4.0 GB, without any checks. The three example
  regions take 0.4–0.6 s, skims included. Unchanged inputs are not
  checked again (the report is reused).
- **The model files supply checks that need no setup.** Every land use
  column used as a size term must be there and never negative, and each
  size segment must be positive somewhere (otherwise nothing can be
  chosen). Every skim must be free of missing or infinite values. These
  run even when a region describes none of its columns.
- **The reader matches 1.x's.** Same tables, columns, IDs and values as
  1.x's `read_input_table`, on all three regions (tested). Along the way
  it turned out pandas, which 1.x uses, reads some CSV numbers slightly
  wrong (1.x findings, #11); asim_core reads them correctly rounded by
  default, and like 1.x when asked (`like_asim1=True`).

## Where tables and columns are defined

Two files, in the same format, checked by Pydantic when they load:

| File | What it holds |
|---|---|
| `src/asim_core/inputs/standard_tables.yaml` (ships with asim_core) | The columns every region has under these names and the core relies on: `household_id`, `home_zone_id`, `hhsize`, `person_id`, the person's `household_id`, `zone_id`. |
| `input_tables.yaml` in the region's configs folder | The region's other columns: meaning, unit, rules. Can also change a standard rule. |

The examples' region files are in `asim_core/configs/<example>/` (read
before the example's own configs folder, so 1.x's examples stay
unchanged). Which file holds each table, and how columns are renamed,
still comes from the region's `settings.yaml` (`input_table_list`), so
nothing is written twice.

A column, as written in `input_tables.yaml`:

```yaml
tables:
  land_use:
    columns:
      TOTEMP:
        type: int64
        unit: jobs
        description: Total employment.
        min: 0
        sum_of: [RETEMPN, FPSEMPN, HEREMPN, OTHEMPN, AGREMPN, MWTEMPN]
      area_type:
        type: int64
        description: "Area type: 0 regional core, 1 CBD, 2 urban business, 3 urban, 4 suburban, 5 rural."
        values: [0, 1, 2, 3, 4, 5]
```

The rules:

| Rule | Severity | Meaning |
|---|---|---|
| `type` | error | Values must fit: `int64` needs whole numbers, any number type needs numbers. |
| missing values | error | Empty cells are errors unless `missing_allowed: true`. |
| `min`, `max` | error | Allowed range. |
| `values` | error | The allowed codes. |
| `unique` | error | No value used twice (IDs). |
| `references: table.column` | error | Every value must be one of that column's (zones, households). |
| `count_of: table.column` | error | Equals the number of rows of that table with this row's ID (household size against persons). |
| `warn_below`, `warn_above` | warning | Allowed, but unusual (an age above 100; a park-and-ride lot with no spaces). |
| `sum_of` (with `sum_tolerance`) | warning | Should equal the sum of other columns (total employment and its sectors). |

A misspelled rule, a description with an unquoted comma, a `min` above
its `max`, or a reference to a column that isn't defined is reported when
the definitions load, before any data is read:

```
table definitions (.../standard_tables.yaml, .../input_tables.yaml):
  tables.land_use.columns.TOTEMP: 'mn' is not a setting (settings: type, description, unit, missing_allowed, min, max, values, unique, references, count_of, warn_below, warn_above, sum_of, sum_tolerance)
```

(This caught six mistakes of mine, unquoted commas, while writing the example files.)

Columns with no definition are listed at the end of the report, so it's
clear what is still undescribed. The three examples' region files now
describe every input column.

## Running the checks

```bash
python -m asim_core.inputs -c asim_core/configs/prototype_mtc -c activitysim/examples/prototype_mtc/configs -d activitysim/examples/prototype_mtc/data
```

The exit status is 1 when there are errors. `--cache DIR` keeps one
report per version of the inputs; `--no-skims` skips the skims. From
Python: `check_inputs(configs_dirs, data_dirs, cache_dir)` returns a
`Report`, and `report.raise_if_errors()` stops a run. The milestone run
(`harness/run_milestone1.py`) checks the inputs before anything else and
stops on errors.

A report, for MTC data with four errors put in (zone 8's retail and total
employment negated, one household given an extra person and another a
zone that doesn't exist, one skim value removed):

```
Input checks: 6 errors, 1 warning (rows: households 5,000, persons 8,212, land_use 25, 826 skims; 0.36 s)

ERROR   households.home_zone_id: not found in land_use.zone_id (1 different value) -- 1 row
          household_id 350326: 26
          used in: annotate_households, annotate_persons, annotate_persons_school, annotate_persons_workplace, joint_tour_frequency_annotate_households_preprocessor, non_mandatory_tour_destination, school_location, summarize, and 1 more

ERROR   households.hhsize: differs from the number of persons rows with this household_id -- 1 row
          household_id 112477: hhsize 2, persons rows 1
          used in: annotate_households, free_parking, stop_frequency_eatout, stop_frequency_escort, stop_frequency_othdiscr, stop_frequency_othmaint, stop_frequency_school, stop_frequency_shopping, and 6 more

ERROR   land_use.TOTEMP: below the minimum 0 -- 1 row
          zone_id 8: -4171
          used in: accessibility, annotate_landuse, tour_mode_choice_annotate_choosers_preprocessor, trip_mode_choice_annotate_trips_preprocessor

ERROR   land_use.RETEMPN: below the minimum 0 -- 1 row
          zone_id 8: -344
          used in: accessibility, destination_choice_size_terms

ERROR   land_use.RETEMPN: negative values in a size term (work_low, work_med, work_high, work_veryhigh, escort, shopping ...): a zone's size must not be negative -- 1 row
          zone_id 8: -344
          used in: destination_choice_size_terms, accessibility

ERROR   skims.SOV_TIME__AM: missing or infinite values -- 1 cell
          zone 5 to zone 12: missing
          used in: annotate_persons_school, annotate_persons_workplace, joint_tour_scheduling_annotate_tours_preprocessor, summarize_preprocessor, tour_mode_choice, tour_mode_choice_annotate_choosers_preprocessor, tour_scheduling_atwork_preprocessor, trip_mode_choice, and 1 more

WARNING land_use.TOTEMP: differs from the sum of RETEMPN, FPSEMPN, HEREMPN, OTHEMPN, AGREMPN, MWTEMPN by more than 0 -- 1 row
          zone_id 8: -4171, sum 3483
          used in: accessibility, annotate_landuse, tour_mode_choice_annotate_choosers_preprocessor, trip_mode_choice_annotate_trips_preprocessor
```

The same negative employment in 1.x runs silently and changes results
across the model (1.x findings, #10).

"Used in" lists the config files that mention the column by name (any
spec, preprocessor, size term or settings file other than `settings.yaml`).
It is a text search, so it can list a file where the name means something
else; Part 2's formula checking will make it exact.

## The injected errors

Each case changes one thing in a copy of the region's data and checks
that the report has exactly the expected problems:

| Injected | Caught by | MTC | MWCOG | ARC |
|---|---|---|---|---|
| Total employment negative in one zone | region rule `min: 0` (+ `sum_of` warning) | ✓ | ✓ | ✓ |
| A size-term employment column negative in one zone | size terms (no definition needed) and region rule | ✓ | ✓ | ✓ |
| A person ID used twice | standard rule `unique` | ✓ | ✓ | ✓ |
| A household in a zone that doesn't exist | standard rule `references` | ✓ | ✓ | ✓ |
| Household size one more than its persons | standard rule `count_of` | ✓ | ✓ | ✓ |
| One missing skim value | skims | ✓ | ✓ | ✓ |
| University enrollment zero everywhere (no capacity) | size terms: a segment with zero size | ✓ | ✓ | ✓ |

On zero capacity: none of the three examples has park-and-ride lots, so
"zero capacity" was tested as a whole segment with no size (MTC's and
MWCOG's university segments; in ARC only the trip `univ` segment, since
ARC's school `univ` segment also counts education jobs, and the check said
exactly that). A single zone or lot with zero capacity is allowed but
worth a warning; a region writes it as `warn_below: 1` on the column
(tested on a small made-up table).

## How the checks run

Each table is read with PyArrow (only the columns kept), then checked a
column at a time with NumPy: the column's minimum and maximum first, and
the offending rows looked up only when a limit is crossed; allowed codes
through a lookup table; references and counts through `bincount` when IDs
are small integers, sorting otherwise. Nothing runs row by row (Pydantic
row by row would take about 2.7 s per million rows; measured earlier).
Only ID columns and columns other rules point to are kept in memory as
NumPy arrays.

**Large population** (`harness/bench_input_checks.py --copies 100`: ARC's
households and persons repeated 100 times with new IDs; each run in a
fresh process, 3 runs, median; skims not included):

| | Time | Peak memory |
|---|---|---|
| asim_core: read and check | 2.13 s (1.98–2.16) | 2,849 MB |
| asim_core: read only | 1.19 s | 2,465 MB |
| pandas: read only (as 1.x reads, all columns) | 8.17 s | 3,974 MB |

Reading dominates the peak: PyArrow's parallel parser holds about four
times the final table while it works. Reading in a stream cuts the peak by
about 15% but takes four times as long, so it isn't used.

**Skims** are read from the OMX files and every matrix checked for missing,
infinite and negative values; the land use zones must match the skims'
zones. For the examples this takes 0.2–0.5 s. A region's full skims take
longer, which is why the report is kept per version of the inputs.

**Reports are kept per version of the inputs:** the input files' paths,
sizes and modification times, the definitions, the size terms, the skim
files and this code. Unchanged inputs give the saved report back at once.

## Reading the tables like 1.x

`read_tables(configs_dirs, data_dirs)` follows `input_table_list` as 1.x
does: `filename` (also `.csv.gz` or `.parquet`), `drop_columns`,
`rename_columns`, `keep_columns`, `index_col` (or the table's standard ID),
and the cp1252 fallback for files that aren't UTF-8. The checks see
the original values; `to_pandas(tables, specs)` then sets the ID as the
index and applies `recode_columns` (zones renumbered from 0), giving the
same tables as 1.x's reader. Two things found while matching it:

- **Empty text cells are missing values in 1.x** (pandas), but PyArrow keeps
  them as empty strings by default; the reader uses pandas' list of
  missing-value markers.
- **pandas' default parser reads some numbers slightly wrong.** It keeps
  about 16 significant digits and doesn't round the last one correctly:
  22 of MTC's 250 land use floats differ from the correctly rounded value
  by one unit in the last place, and in MWCOG's configs some values by
  thousands of units (relative error 1e-12). asim_core reads correctly
  rounded by default. For bit-for-bit comparisons with 1.x,
  `read_tables(..., like_asim1=True)` reads with pandas instead. The test
  checks both: identical to 1.x when reading like 1.x, within a few units
  in the last place otherwise. (Reported as 1.x findings, #11.)

## Not done yet

- **`log()` inputs worked out from the specs** (a column used inside
  `log()` must be positive): needs the formula checking of Part 2.
- **Persons' home zone against their household's** (ARC has both): a
  rule kind for "same as the household's" isn't there yet.
- **Tables other than households, persons and land use** (park-and-ride
  lots, MAZ/TAP tables of two- and three-zone systems): the rules work on
  any table, but no example has them.
- **HDF5 input stores** (`input_store`) are read through pandas and
  untested.
- **Running inside asim_core's own first step:** asim_core still starts
  from 1.x's saved state, so the checks run before the milestone run and
  from the command line.
