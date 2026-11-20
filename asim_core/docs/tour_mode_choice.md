# Tour mode choice

Tour mode choice is the third ported step, and the first large one: a
nested logit with 15–21 modes, about 300 formulas, ten purpose segments
with their own coefficients, a preprocessor of about 75 rows, and skims
that depend on each tour's departure and return times. It's also what
mandatory tour scheduling spends most of its time on (as logsums; see
[scheduling.md](scheduling.md)), so it had to come first.

## Summary

- **asim_core's tour mode choice matches 1.x exactly on all three regions.**
  Every tour gets the same mode, and every mode choice logsum is bit-for-bit
  identical:

  | Region | 1.x run used | Tours | Same mode | Logsums bit-identical |
  |---|---|---|---|---|
  | MTC | full example | 9,103 | 9,103 | 9,103 |
  | MWCOG | 100-household sample | 133 | 133 | 133 |
  | ARC | all 24,061 households | 68,934 | 68,934 | (ARC doesn't save logsums) |

- **Today's files are read unchanged:** `tour_mode_choice.yaml`, the spec,
  coefficients, coefficient template and preprocessor of each region. The
  formulas are turned into NumPy code you can read
  (`work/milestone1/step_check/tour_mode_choice_simulate/generated/`).
- **Faster and smaller than 1.x (details below).** PROFILE_SUMMARY
- **Nothing tour-mode-specific went into the core.** The step is a
  component (`components/tour_mode_choice.py`, about 300 lines). What it
  needed is general: coefficient templates, preprocessors, table lookups,
  time-of-day skims, normal random draws, logsums from the nested logit.

## What was added to asim_core

- **Coefficient templates (segments).** A model can have one coefficient
  set per segment (tour purpose here), read from today's template file.
  Terms and nests use the generic names; `coefficient_values("work")` gives
  the values for one segment. An empty template cell means the same
  coefficient for every segment, as in 1.x.
- **Coefficients inside formulas.** Mode choice formulas use coefficient
  names as values (`@coef_topology_walk_multiplier * df.dest_topology`). The
  generated code reads them as `k['name']`, so one compiled function serves
  every segment.
- **Preprocessors: decided once, when the step's columns are known.** Rows
  like `df.number_of_participants if 'tour_type' in df.columns else 1` test
  what data exists. asim_core works such tests out during the startup check
  and writes only the branch that applies into the generated code. So the
  code is simpler, and the startup check knows exactly which columns the
  step reads. A test that can't be decided that way is reported as
  unsupported.
- **Other preprocessor features:** lookups in other tables
  (`reindex(land_use.TOPOLOGY, df[dest_col_name])`, including arithmetic on
  the looked-up column, as MWCOG writes it), `setting('cbd_threshold')`,
  `.map()` with a dictionary constant, `pd.cut(...).astype(int)`, and
  scalar results made into columns, as 1.x does.
- **Time-of-day skims.** `odt_skims['SOV_TIME']` looks up each tour's skim
  for its departure period (`SOV_TIME__AM`, ...), `dot_skims` for its return
  period, and so on for 1.x's six skim lookups, including
  `od_skims.reverse(...)` and `od_skims.max(...)`. Periods come from
  `network_los.yaml`'s `skim_time_periods`, labelled as 1.x labels them.
  asim_core looks up period by period, so no 3-D copy of the skims is
  needed.
- **Logsums.** The nested logit piece now also returns each row's logsum,
  computed as 1.x computes it: log of the root's exponentiated utility.
- **Joins through other joins.** Tours see their person's columns (by
  person_id), the household's (by household_id), and the home zone's (by
  home_zone_id), looked up rather than merged. When a column name is in
  several tables, the same table wins as in 1.x's merged table.

## Details that mattered for exact matching

- **The preprocessor draws random numbers.** MTC's and MWCOG's
  preprocessors draw a wait time for taxis and ride-hailing at both ends of
  each tour: six lognormal draws per tour. 1.x makes all six in one call
  before the preprocessor runs, then moves each tour's random stream on by
  six. The mode choice draw comes after that. asim_core makes the same
  draws in the same order (`random.normal`), so the choice draw matches.
  NumPy's normal generator usually uses more uniform numbers than the six
  it returns, so the choice's draw can be one the normal draws already
  used. That's how 1.x does it, so it's reproduced as is.
- **The wait times are constants in these examples.** The standard
  deviations in the settings are all zero, so each "random" wait time is
  `exp(log(mean))`. The draws still move the random streams on.
- **Terms with all-zero coefficients are skipped per segment,** as 1.x
  does. Which terms that is differs by purpose (e.g. 289 terms are computed
  for work tours, 301 for school tours, out of MTC's 315).
- **Segments are processed in 1.x's order,** but nothing depends on it:
  each tour's draws come from its own random stream.
- **The output column keeps 1.x's categories,** sorted, with an empty
  category first, as 1.x's tours table ends up with them.

## Found along the way

- **Replay kept stale values for rows added later.** When a ported step's
  column is carried through later replayed 1.x steps, asim_core keeps its
  own values. All three regions add at-work subtours to the tours table
  after tour mode choice, and choose their modes in a later step. The runner now
  remembers which rows it computed, so rows a later 1.x step adds keep 1.x's
  values.

## Profiles

PROFILE_TABLE

## Not supported yet

Park-and-ride lots, trip mode choice logsums inside tour mode choice
(`COMPUTE_TRIP_MODE_CHOICE_LOGSUMS`), annotation after the model,
multinomial (not nested) mode choice, more than one preprocessor, and
at-work subtour mode choice (a separate 1.x step, still replayed). For
`reindex`, a zone not in the table is an error, where 1.x would give a
missing value.
