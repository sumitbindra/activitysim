# Model files and one formula language

Part 2 of the configuration plan ([config_plan.md](config_plan.md)).
asim_core can now read its own model files: one YAML file per model plus
CSV tables, with formulas in one language whose names say which table a
column comes from. A converter writes them from today's files. Today's
files keep loading, so every result can still be compared with 1.x.

## Summary

- **All eight ported steps convert, on all three regions, and give exactly
  1.x's results from the converted files** (school and workplace location
  were added after this write-up, written up in
  [location_choice.md](location_choice.md)). Each step was run twice on the
  same saved 1.x state, from today's files and from the converted ones;
  every output column is identical between the two and with 1.x
  (`harness/check_model_files.py`, and a test per region):

  | Step | MTC | MWCOG | ARC |
  |---|---|---|---|
  | compute_accessibility | exact | exact | exact |
  | auto_ownership | exact | exact | exact |
  | mandatory_tour_scheduling (with mode choice logsums) | exact | exact | exact |
  | non_mandatory_tour_destination (sampling, logsums) | exact | exact | exact |
  | non_mandatory_tour_scheduling | exact | exact | exact |
  | tour_mode_choice (modes and logsums, bit for bit) | exact | exact | exact |
  | school_location, workplace_location (with annotations) | exact | exact | exact |

  End to end on MTC from the model files: 34 steps (6 computed from the
  converted files, 28 replayed from 1.x), all 217 compared columns exact.

- **One language, with NumPy's number rules.** 1.x evaluated 47 of the
  converted formulas (37 in MTC, 7 in MWCOG, 3 in ARC) with
  `pandas.eval`, whose number rules differ from NumPy's and change above
  1M rows. They now follow NumPy's rules at any size. On these examples the
  results are identical.

- **Every column says its table.** Writing `person.is_CBD` instead of
  `df['is_CBD']` exposed a problem in ARC's example: its destination
  terms "CBD area type" and "Urban high area type" read the person's home
  zone, not the candidate destination, so they never change a choice
  (checked on all 30,851 tours; 1.x findings, #12).

- **What 1.x decides in code is written in the file:** the tour purpose
  rule, which times give the skim periods, tests of which columns exist,
  how sampling caps, orders and stores its draws, where random draws come
  from (list below).

- **Mistakes are caught when the model loads, or when the run starts**,
  before any step runs: a misspelled option, an unknown function, a
  column a table doesn't have ("person has no column 'agee'; did you mean
  age?").

## A model file

```yaml
kind: auto_ownership                      # which kind of model; sets the tables formulas see
alternatives: [cars0, cars1, cars2, cars3, cars4]
logit: multinomial
terms: auto_ownership_terms.csv
coefficients: auto_ownership_coefficients.csv
output: auto_ownership
random_stream: households
options:
  utility_shift: false                    # 1.x: only when skip_failed_choices is off
  drop_zero_terms: true                   # 1.x skips terms whose coefficients are all zero
```

The file is checked by the Pydantic class of its kind: a misspelled or
misplaced setting is an error, not ignored. Each kind writes out every
option with 1.x's behaviour as its value. Where only 1.x's behaviour is
implemented so far, the option has one allowed value, so the file still
documents it (e.g. `sample_order: by_zone`).

The CSV tables keep today's layout:

| Table | Columns | Replaces |
|---|---|---|
| terms | `Label, Description, Formula`, then one column per alternative (or segment, or `coefficient`) holding a coefficient name or a number | the spec CSV (`Expression` becomes `Formula`) |
| coefficients | `coefficient_name`, then `value`, or one column per segment (e.g. tour purpose) | the coefficients CSV and the coefficient template |
| derived values | `Description, Name, Formula`, computed in order | preprocessors and annotation-style assignment CSVs |

A term row written `name = formula` defines a value for later rows
(1.x's `_name@formula`). Derived values starting with `_` are only used by
later rows.

Files a model shares with 1.x unchanged (`tour_departure_and_duration_alternatives.csv`,
`destination_choice_size_terms.csv`, network_los.yaml's time periods,
constants.yaml) are named in the model file and found the usual way: next
to it, or in the first configs folder that has them.

The kinds so far: `accessibility`, `auto_ownership`, `location_choice`
(school and workplace), `tour_mode_choice`, `tour_scheduling` and
`tour_destination`. The converted files for the three
examples are in `asim_core/configs/<example>/`.

## The formula language

Python syntax, parsed and checked, never executed; compiled to NumPy code
(`spec/language.py`).

**Names**

| Written | Means |
|---|---|
| `person.age`, `household.income`, `tour.tour_type`, `home_zone.TOTEMP` | a column of a table the model sees (its "role"; listed per kind below) |
| `alt.start`, `alt.alt_dest` | the alternative being evaluated in scheduling (a departure-return option) and destination choice (a candidate zone) |
| `is_joint`, `out_period`, `mode_choice_logsum` | a bare name: one of the model's own values (a derived value, or a value the step works out), a constant, a coefficient where allowed, or a parameter |
| `o`, `d` | the zones the model is evaluated between (origin, destination), set by the step |
| `land_use.TOPOLOGY[d]` | another table's column, looked up by ID (an ID not in the table, e.g. zone -1 for a person without a workplace, gives a missing value, as 1.x's `reindex`) |
| `skim.SOV_TIME[o, d]`, `skim.SOV_TIME[d, o, in_period]`, `skim.DIST[o, d, 'AM']`, `skim.DIST[tour.origin, tour.destination, 'MD']` | skims: from, to, and optionally a time period (a derived value holding each row's period, or a period in quotes); zone -1 gives a missing value, as in 1.x |
| `tt.previous_tour_ends(tour.person_id, alt.start)` | the timetable's functions (scheduling) |

**Operators:** `+ - * / // % **`; comparisons, chained too
(`18 <= person.age < 65`); `x in [1, 2]` and `x not in [...]`; `and`,
`or`, `not`; `a if CONDITION else b`, where the condition must be known
when the model loads (constants, parameters; use `where` for row-by-row
choices).

**Functions:** `where(condition, a, b)`, `min(a, b, ...)`, `max(a, b, ...)`
(element by element), `clip(x, lower, upper)`, `between(x, lower, upper)`,
`log`, `log1p`, `log10`, `log2`, `exp`, `sqrt`, `abs`, `floor`, `ceil`,
`fillna(x, value)`, `map(x, {...})`, `cut(x, bins=[...], labels=[...])`,
`shift(x, rows)`, `count_by(table.column, keys)`,
`sum_by(table.column, table.key, keys)` (e.g. a household's sum over its
persons: `sum_by(persons.x, persons.household_id, household.household_id)`),
`int(x)`, `float(x)`,
`bool(x)`, `int8(x)` ... `float64(x)`, `setting('name')`, and the random
draw `random_lognormal(mean=..., sd=...)` (or `mu=..., sigma=...` for the
underlying normal).

**Number rules: NumPy's**, at any number of rows. Coefficients are
rounded to 32-bit floats before use, as 1.x does.

**What is checked, and when.** When the model loads: syntax, functions and
their arguments, table names, bare names (constants, coefficients, derived
values, parameters), skim forms. When the run starts (the runner's check of
every step, before anything runs): every `table.column` against the
columns the run's tables have, with "did you mean" suggestions.

**Two rules about names**, both checked:
- A derived value can't share a name with a column it hides (e.g. a
  derived `number_of_participants` and `tour.number_of_participants` in the
  same rows).
- When a column is in two tables of a view (ARC's persons and households
  both have `home_zone_id`), the name goes to the first table, as in 1.x's
  merged tables (`person.home_zone_id`).

### Tables each kind of model sees

| Kind | Roles | The step's own values (bare names) |
|---|---|---|
| `auto_ownership` | `household`, `home_zone` (land use and accessibility of the home zone) | |
| `accessibility` | `dest_zone` (the destination zone's land use); skims from `o` (origin) to `d` | |
| `tour_mode_choice` | `tour`, `person`, `household`, `home_zone`; derived values may look up `land_use`, `tours` by ID | `tour_purpose`, `out_period`, `in_period` |
| `tour_scheduling` | terms: `tour`, `person`, `household`, `home_zone`, `alt` (`start`, `end`, `duration`, `tdd`); tours' derived values: `tour`; options' derived values: `alt` | `mode_choice_logsum`, `out_period`, `in_period`, `start_previous`, `end_previous` |
| `tour_destination` | `tour`, `person`, `household`, `home_zone`, `alt` (the candidate zone: `zone_id` in the sample, `alt_dest` in the choice); skims from the tour's origin (`o`) to the candidate (`d`) | `size_term`, `prob`, `pick_count`, `mode_choice_logsum` |
| `location_choice` | `person`, `household`, `home_zone`, `alt` (as `tour_destination`); annotations: their table (`person` or `household`), other tables by ID, skims between any zones | as `tour_destination`, and `shadow_price_size_term_adjustment` (1), `shadow_price_utility_adjustment` (0) |

**Annotations** (1.x's `annotate_persons`, `annotate_households`): derived
values for a whole table after the step's choices, listed under
`annotate:` in the model file. They run in 1.x's order (persons, then
households), so a households annotation sees the persons annotation's
results; a result for a column the table already has replaces it except
where the new value is missing, as 1.x does.

Tour mode choice also runs inside scheduling and destination choice, to
give logsums. There, its `tour` columns are what those steps put in the
rows: in scheduling, the option's times (`tour.start` is the option's
start); in destination choice, the candidate zone and the sample's columns.

## Today's formulas, converted

| 1.x | asim_core |
|---|---|
| `reindex(land_use.TOPOLOGY, df[dest_col_name])` | `land_use.TOPOLOGY[d]` |
| `(reindex(land_use.TOTPOP, df[orig_col_name]) + ...)` | `(land_use.TOTPOP[o] + land_use.TOTEMP[o]) / (land_use.TOTACRE[o] / 640)` |
| `skim_od[('SOVTOLL_TIME', 'AM')] + skim_do[('SOVTOLL_TIME', 'PM')]` | `skim.SOVTOLL_TIME[o, d, 'AM'] + skim.SOVTOLL_TIME[d, o, 'PM']` |
| `(tour_num > 1) & (start < end_previous)` | `tour.tour_num > 1 and alt.start < end_previous` |
| `@df.income_in_thousands.clip(0, 30)` | `clip(household.income_in_thousands, 0, 30)` |
| `@np.minimum(9-alt_tdd.start,48)*(alt_tdd.start<=9) + ...` | `min(9 - alt.start, 48) * (alt.start <= 9) + ...` |
| `df.number_of_participants if _DF_IS_TOUR else 1` | `tour.number_of_participants if _DF_IS_TOUR else 1` |
| `'tour_type' in df.columns` | `has_tour_type` (a parameter, below) |
| `pd.cut(origin_density_measure, bins=[...], labels=[...]).astype(int)` | `int(cut(origin_density_measure, bins=[...], labels=[...]))` |
| `rng.lognormal_for_df(df, mu=X, sigma=Y, broadcast=True, scale=True)` | `random_lognormal(mean=X, sd=Y)` |
| `reindex_i(joint_tour_participants.groupby('person_id').size(), df.person_id)` | `int8(count_by(joint_tour_participants.person_id, tour.person_id))` |

## What 1.x decides in code, now written in the files

| Behaviour | 1.x | Model file |
|---|---|---|
| A tour's purpose (picks mode choice coefficients; mandatory scheduling's spec) | in code: tour type, or `univ` for a university student's school tour | `tour_purpose:` / `segment:` formula, e.g. `where(tour.tour_type == 'school' and person.is_university, 'univ', tour.tour_type)` |
| Which times give the skim periods | in code: `start`, `end` | `time_periods: {out_period: tour.start, in_period: tour.end}` |
| Tests of which columns the choosers have (`'tour_type' in df.columns`): the same preprocessor takes different branches in different steps | hidden in formulas | parameters (`has_tour_type`, ...): the mode choice file gives its own step's values, each step using it for logsums gives its own (`logsums: parameters:`) |
| Tests of an input table's columns (`'access_dist_transit' in land_use`) | worked out at each run | settled when converting (a region's land use doesn't change between steps): the branch that applies is kept |
| Overflow protection | only when `skip_failed_choices` is off | `options: utility_shift:` |
| Terms whose coefficients are all zero | skipped | `options: drop_zero_terms: true` |
| Destination sampling | inverse CDF; at most as many draws as zones with a size; sample sorted by zone; probabilities as 32-bit floats | `options: sample_method`, `sample_size_cap`, `sample_order`, `sample_probabilities` |
| Random draws in preprocessors | from the chooser's stream, in row order | `random_lognormal(...)` rows; `options: random_draws: chooser_stream` |
| Tours whose segment has no spec | left unscheduled (1.x findings, #8) | `options: tours_without_spec: error` |
| Per-segment alternatives preprocessors without logsums | skipped (1.x findings, #9) | only the ones 1.x runs are converted |
| Coefficients for segments | coefficients file + template | one table, a column per segment |

## Converting

```bash
python -m asim_core.converter -c activitysim/examples/prototype_mtc/configs --asim1-run asim_core/work/asim1_legacy --out asim_core/configs/prototype_mtc
```

To run with the converted files, list their folder first among the configs
folders (the milestone run: `--model-files`). To convert and check every
step against today's files and 1.x in one go:

```bash
python asim_core/harness/check_model_files.py --example prototype_mtc
```

How it works:

- **Which table each column comes from** depends on the step: the
  converter reads the column names a 1.x run saved (only their names) and
  applies the step's view (tours, then persons, households, home zone).
- **Tour mode choice runs in four places** (its step, both scheduling
  steps, destination choice). The converter asks each how it would call the
  model, to name every column the model reads anywhere and to work out each
  place's parameter values. A column no place has (in a branch none of them
  takes, e.g. `parent_tour_id`, which tours only get at the at-work subtour
  step) is taken as the tour's; if a step ever takes that branch, the
  start-up check reports it.
- **Numbers** are written exactly as 1.x reads them (including the last
  bits pandas' parser gets wrong, 1.x findings, #11), so the converted
  models compute the same values.
- Comment lines in 1.x's CSV files (`#,, auto peak`) are not carried over.

## Inheriting

A region's model file can start from another and change only what differs:

```yaml
# ARC's destination model without the home-zone CBD and urban terms
extends: ../prototype_arc/non_mandatory_tour_destination.yaml
coefficient_values:
  coef_shopping_cbd: 0
  coef_eatout_cbd: 0
  ...
```

Options are merged; any other setting listed replaces the base's; file
names are relative to the file that names them. This is how the ARC
experiment above was run: the destinations didn't change for any of the
30,851 tours.

## Not done yet

- **Model files for the steps not ported yet** (they keep running from 1.x).
- **Declared random streams** (each draw its own named stream): for exact
  matching, draws still come from the chooser's 1.x stream in row order;
  the option names that.
- **`log()` inputs checked against the input definitions** (Part 1's
  "checks worked out from the models"): the language now makes this
  possible.
- **A generated page of every variable** (meaning, unit, which models use
  it), from the table definitions and the model files.
- **Estimation settings** (`constrain` flags of coefficients are not
  carried over).
