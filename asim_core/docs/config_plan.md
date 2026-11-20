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
- **Pydantic for definitions** (model settings, table definitions). **PyArrow
  for reading data**, with types from the definitions. **NumPy for checking
  data**, a column at a time, never row by row. No new dependencies.
  Measured: 1M persons read in 0.06 s (pandas 0.25 s), checked in 0.02 s;
  Pydantic row by row would take 2.7 s per million rows.
- **CSV tables stay** (term tables, coefficients), editable in Excel.
- **Today's files keep loading**, so the exact comparison with 1.x stays.

## Part 1: table definitions and checks before the run (done)

Results: [input_checks.md](input_checks.md).

1. **Table definitions** (YAML, checked by Pydantic): the standard
   columns in asim_core, the region's own columns in its
   `input_tables.yaml`: types, units, descriptions, rules.
2. **Which file holds each table, and the renames:** read from 1.x's
   `input_table_list` in settings.yaml, unchanged.
3. **Reader**: PyArrow; CSV, compressed CSV or Parquet. Same tables as
   1.x's reader.
4. **Checks before the first step**: types, ranges, allowed values,
   missing values, unique IDs; zone and household references; household
   size against persons; sums; size terms and skims worked out from the
   model files. Errors stop the run, warnings don't; one report with
   counts, first examples, and the config files that use each column; a
   report kept per version of the inputs.

**Done:** seven kinds of injected error caught on all three regions;
clean data passes with no false alarms; 6.7 million persons read and
checked in 2.1 s.

Left for later: `log()` inputs worked out from formulas (needs Part 2),
a "same as the household's" rule, and tables none of the examples have
(park-and-ride lots, MAZ/TAP).

## Part 2: model file format, and a converter

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
   NumPy number rules, stated once; names checked against the table
   definitions when the model loads ("did you mean ...?").
5. **Derived columns** declared per table, replacing preprocessors; no
   tests of which columns exist. **Random draws** declared, each with its
   own named stream.
6. **Regions inherit and override** a base model instead of copying it.
7. **A converter** from today's files to the new format, applied to all
   six ported steps; today's reader kept for the 1.x comparison.

**Done when:** the six ported steps are converted, and each gives the same
results as its original files (exact wherever the old dialects' number
rules agree, listed differences elsewhere); every behaviour 1.x hides is
an explicit option.

## What changes for people using the model

| | Today | New |
|---|---|---|
| Column names differ by region | `rename_columns` in settings.yaml | the same `rename_columns`, with every column defined |
| What a variable means | nowhere | the table definitions (`input_tables.yaml`) |
| Bad input data | runs silently, or crashes far from the cause | stopped before the first step, with a report |
| Formulas | two dialects with different rules | one, with stated rules |
| Preprocessors | column-existence tests, hidden random draws | declared derived columns and random streams |
| Coefficients for segments | coefficients file plus template file | one table, a column per segment |
| Options | some defaults hidden in code | all written out, with defaults |
| Files per model | up to six plus settings | one YAML, a term table, a coefficients table |

What doesn't change: CSV tables editable in Excel; today's configs still
run; every result can still be compared with 1.x.
