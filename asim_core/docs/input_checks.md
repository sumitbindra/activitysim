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

## Skims the model would use wrongly, without an error

These checks look for the kind of problem that gives wrong results with no
error at all: the right numbers read for the wrong zone, period or units.
The rule behind them: wherever two inputs are matched by position or by a
code, check the match.

**Errors or warnings.** An error stops the run, so it is kept for problems
that certainly make the run wrong or make it fail: a zone that can't be
found in the skims, or would read another zone's row by the numbering
rules; a missing skim value. Everything that is a judgment call, or found by
a heuristic, is a warning: the run goes on, and the warnings are shown with
the report (on screen, in `input_checks.txt` and in the run's `summary.md`)
for the user to decide. Each was tested by injecting the problem into a copy
of each region's data (8 cases × 3 regions, `tests/test_input_checks.py`):
the report names it, by the right rule, and nothing else; the clean data of
all three regions passes with no problems.

| Check | Catches | Kind | Needs |
|---|---|---|---|
| Zone rows | each zone reads the skim row 1.x would read it from: through the file's zone lookup, or by 1.x's numbering rules (table above) | error | nothing |
| Zone order | a skim file whose zones are in another order than the model assumes (e.g. not ascending, and no zone lookup to say so): its distances don't follow the land use better than a random order would | warning (statistical) | `skims.zone_order` |
| Skims the models use | a skim named in a spec, preprocessor or model file that isn't in the skim files; a period it's used with that has no skim | warning (found by a text search; a step that uses it stops anyway) | nothing |
| Time periods | a skim used with each row's time period that lacks some period of network_los.yaml (`SOV_TIME__EV` missing) | warning (as above) | nothing |
| | skims for a period network_los.yaml doesn't list (1.x skips them) | warning | nothing |
| Duplicates | a skim in two files: with different values (the last file's are used, as in 1.x: an override, or the wrong file?) | warning | nothing |
| | with the same values | warning | nothing |
| Units | a skim's typical value (median of its positive values) outside its measure's range: e.g. fares in dollars instead of cents | warning | `skims.measures` |
| | a time's typical minutes per mile outside its range: e.g. seconds or hours instead of minutes, whatever the size of the region | warning | `skims.measures` |
| Placeholders | values more than 1,000 times the skim's typical value (999999 for "no path"), which the models use as numbers | warning | nothing |
| | values above a measure's `max` | warning | `skims.measures` |
| Within a zone | a time or distance of 0 from a zone to itself | warning | `skims.measures` |

**Which skims the models use** comes from the config files: 1.x's specs and
settings (`odt_skims['SOV_TIME']`, `skim_od[('SOV_TIME', 'AM')]`,
`skim_dict.lookup(..., 'DIST')`) and asim_core's model files
(`skim.SOV_TIME[o, d, out_period]`). Commented-out rows are skipped, as 1.x
skips them (commented-out rows of MWCOG's and ARC's trip specs name three
skims that don't exist). A skim used without a named period and without a
period-free version needs every period.

**The zone order** can only be checked against something else that knows
where zones are. Three tests, each run when the region names the land use
columns, compare the distances with them, and with the same zones in 30
random orders:

| Region | Test | As the data is | Random orders | z |
|---|---|---|---|---|
| MTC, 25 zones | distance within a zone against √area | 0.76 | 0.05 ± 0.22 | 3.3 |
| MTC, 1,454 zones | distance within a zone against √area | 0.81 | 0.01 ± 0.03 | 28 |
| | nearest zone in the same district | 95% | 4% ± 1% | 128 |
| MWCOG, 53 zones | distance within a zone against √area | 0.29 | 0.01 ± 0.14 | 2.0 |
| | distances against straight-line distances | 0.96 | 0.01 ± 0.05 | 18 |
| ARC, 36 zones | distance within a zone against √area | 0.60 | 0.07 ± 0.15 | 3.5 |

(Rank correlations; z: standard deviations above the random orders.) The
check warns when no test is 2 above random. It catches a file in
another order (tested: rows and columns in a random order). It can't catch
a shift by a few zones (zone z read as z + 1): neighbouring zone numbers
are neighbouring places, so the tests barely move (MTC full: 0.81 to 0.71
for a shift of one). Shifts come from numbering rules, and those are
checked directly (zone rows, above).

**Within the steps,** some columns only exist once earlier steps have run,
so one check runs when the step starts: every chooser's segment value
(income segment, school type, tour purpose) should have a segment in the
model. 1.x leaves choosers without a segment without a choice, with no
message (1.x findings, #8, for scheduling). In location choice and
destination choice asim_core does the same, so results stay 1.x's, and
reports it as a warning of the step, naming the values and how many
choosers have them. Steps report warnings through `ctx.warn`; the runner
collects them in its report and the run's `run_manifest.json`, and the
milestone run lists them in `summary.md`. (Scheduling and tour mode choice
still stop on it, as before: making them warn too needs matching what 1.x
does with those tours afterwards, which isn't done yet.)

### Describing a region's skims

In the region's `input_tables.yaml`, next to its tables. MTC's (excerpt):

```yaml
skims:
  distance: DIST              # the region's distance skim (ARC: SOV_FREE_DISTANCE__MD)
  zone_order:                 # land use columns distances should follow
    area: TOTACRE
    district: DISTRICT        # MWCOG also has x: TAZXCRD, y: TAZYCRD
  measures:                   # matched by name without the period, first pattern first
    DISTWALK: {unit: miles, typical: [0.1, 300], no_path: 1000000, intrazonal: positive}
    "*_TIME": {unit: minutes, description: Highway time., typical: [0.5, 300], per_distance: [0.3, 30],
               max: 1440, intrazonal: positive}
    "*IVT": {unit: minutes x 100, description: In-vehicle time., typical: [50, 100000]}
    "*_FAR": {unit: cents, description: Fare., typical: [20, 5000]}
```

Units are mixed within one region's file (MTC: highway times in minutes,
transit times in minutes × 100, fares in cents; ARC: fares in dollars), so
ranges are given per measure; the three regions' files describe their main
measures. `no_path` declares a region's value for "no path" (MTC's walk and
bike distances use 1,000,000 where the models' availability rules expect
it), so it isn't reported. Times per mile are given for highway times only:
transit trips in a small area can take 30 to 60 minutes per mile (driving
away from the destination to a station).

On full MTC (2.9 million households, 826 skims) the checks take 22 s (21 s
before these), with the same two warnings as before (85 households above 20
persons; 21 external stations without land use).

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

On full MTC (826 skims, 7.2 GB once decompressed) the checks take about
25 s, 19 s of it decompressing the OMX file, which the skim import then does
again. So the checks can read the values from the run's skim cache instead
(`check_inputs(..., skim_values="cache", skim_store=store)`): the 32-bit
values the model uses, as 64-bit for the statistics. Skims in more than one
file (compared copy by copy) and any the cache left out (time periods the
model doesn't use) are still read from the OMX files. With the cache in
memory, as it is right after an import, the full MTC checks take 6 s; the
report is the same as from the OMX files on all four test regions and in
every injected-problem test (`tests/test_input_checks.py`). Reading the
OMX files stays the default for `check_inputs` and `python -m
asim_core.inputs`; the milestone script now prepares the skims first and
checks from the cache (`--skim-check-values omx` for the other way). If the
cache isn't in memory, reading it (6.7 GB) can take longer than
decompressing the OMX file, but those are the pages the model's steps read
next.

**Memory after the checks.** Checking full MTC's tables (7.6 million
persons) left the process holding 1.3 GB for the rest of the run: Arrow's
default allocator (mimalloc) keeps what it freed, and so did the C
allocator. The checks now read the tables with Arrow's system allocator and
then ask the C library to return freed memory (`asim_core/memory.py`);
after the checks the process holds 0.35 GB instead of 1.56 GB, with the
same report.

**Which skim row each zone reads** follows 1.x's rules, and is checked:

| Zones | OMX file has a zone lookup | Zone → skim row | Checked |
|---|---|---|---|
| original numbers (MWCOG, ARC) | yes, in any order | matched by number through the lookup | every land use zone is in the lookup (error) |
| original numbers | no | zone z is row z, counting from 1 | every zone is within 1 to the number of skim zones (error) |
| renumbered from 0 (MTC) | yes | the i-th smallest zone is row i; 1.x doesn't use the lookup | the lookup lists the zones in ascending order (error: 1.x findings, #13) |
| renumbered from 0 | no | the i-th smallest zone is row i | can't be checked; zone numbers with gaps give a warning |

Lookups never see zone numbers: they get rows worked out from these rules,
and the number of zones from the skims themselves.

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
