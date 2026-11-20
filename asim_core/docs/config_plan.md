# Plan: configuration and input checks

## Goal

Model settings and inputs that are simple to read and change, checked
before anything runs, and usable by any engine (NumPy today, others later),
without losing the ability to compare every result with ActivitySim 1.x.

## What's wrong today (found while porting)

- **Formulas have no defined meaning.** Two dialects (plain, run by
  `pandas.eval`, and `@`, run as Python) with different rules for names and
  number types; the rules even change above 1M rows. Any new engine has to
  reverse-engineer pandas.
- **Logic is hidden.** Preprocessors test which columns exist to guess
  where they run; random draws hide inside formulas; defaults live in code
  (sample size capped at the zone count, preprocessors skipped, tours left
  unscheduled).
- **Variables aren't defined anywhere.** Names come from input files after
  per-region renames, from earlier steps and from preprocessors. Types,
  units and meaning are nowhere.
- **Bad data runs silently.** Negative employment in one MTC zone changed
  results across the model with no error or warning (1.x findings, #10).
- **A model is spread over many files:** settings, spec, coefficients,
  coefficient template, preprocessor, segments file.

## Decisions

- **One configuration for everyone**, no separate layer for planners. Simpler.
- **Pydantic for definitions** (model settings, table schemas). **PyArrow
  for reading data**, with types from the schemas. **NumPy for checking
  data**, a column at a time, never row by row. No new dependencies.
  Measured: 1M persons read in 0.06 s (pandas 0.25 s), checked in 0.02 s;
  Pydantic row by row would take 2.7 s per million rows.
- **CSV tables stay** (term tables, coefficients), editable in Excel.
- **Today's files keep loading**, so the exact comparison with 1.x stays.

## Phase 1: table schemas and input checks

1. **Table schemas** (Pydantic), one per standard table (households,
   persons, zones, tours, ...): standard column names, types, units,
   descriptions, rules. Names and types seeded from the column lists 1.x's
   runs already write (`data_dict.csv`).
2. **Region input mapping** (YAML, checked): which file and raw column
   provides each standard column (`TAZ` / `maz` / `HHID`), plus region-only
   columns. Same idea as 1.x's `rename_columns`, in one place.
3. **Reader**: PyArrow with the schema's types; CSV or Parquet.
4. **Check before the first step**:
   - column rules: ranges, allowed values, missing values, unique IDs;
   - references: every `home_zone_id` is a zone, every person's household exists;
   - cross-table: household size equals persons; zone employment roughly equals its categories;
   - checks worked out from the configured models: size terms not negative
     and positive somewhere per segment, `log()` inputs positive, skims used
     without missing or negative values;
   - errors stop the run; warnings are reported (e.g. a park-and-ride lot
     with capacity 0);
   - one report: every problem, counts, first examples, and which steps use
     the column;
   - checked once per version of the inputs (like the skim cache).

**Done when:** a set of injected errors (negative employment, duplicate IDs,
missing zone references, household size mismatch, missing skim values, zero
capacity) is caught on all three regions; the clean example data passes
with no false alarms; all ARC households are checked in seconds.

## Phase 2: model file format

1. **One YAML file per model**, checked by its Pydantic class: kind of
   model, choosers, alternatives, term table, coefficients, and every option
   written out with its default (sampling, segments, logsums, ...).
2. **Term tables**: CSV, today's layout (one row per term, one column per
   alternative).
3. **Coefficients**: one table, with a column per segment where segments
   differ (replaces the separate template file).
4. **One formula language**: table-qualified names (`person.age`,
   `dest_zone.TOTEMP`), spreadsheet-like functions (`where`, `clip`, `log`,
   `min`, `max`), `and` / `or` / `not`, skims as `skim.SOV_TIME[od, out_period]`;
   NumPy number rules, stated once; names checked against the table schemas
   when the model loads ("did you mean ...?").
5. **Derived columns** declared per table, replacing preprocessors; no
   tests of which columns exist. **Random draws** declared, each with its
   own named stream.
6. **Regions inherit and override** a base model instead of copying it.

**Done when:** auto ownership and tour mode choice are converted (the
hardest model checks the format); each gives the same results as its
original files (exact wherever the old dialects' number rules agree, listed
differences elsewhere); every behaviour 1.x hides is an explicit option.

## Phase 3: converter and the rest

- A converter from today's files to the new format, for all ported steps;
  new ports written in the new format, with today's reader kept for the 1.x
  comparison.
- A generated page listing every variable and model: meaning, unit, source,
  and which models use it.

## What changes for people using the model

| | Today | New |
|---|---|---|
| Column names differ by region | `rename_columns` in settings.yaml | the region's input mapping (same idea, checked) |
| What a variable means | nowhere | the table schemas, and a generated page |
| Bad input data | runs silently, or crashes far from the cause | stopped before the first step, with a report |
| Formulas | two dialects with different rules | one, with stated rules |
| Preprocessors | column-existence tests, hidden random draws | declared derived columns and random streams |
| Coefficients for segments | coefficients file plus template file | one table, a column per segment |
| Options | some defaults hidden in code | all written out, with defaults |
| Files per model | up to six plus settings | one YAML, a term table, a coefficients table |

What doesn't change: CSV tables editable in Excel; today's configs still
run; every result can still be compared with 1.x.
