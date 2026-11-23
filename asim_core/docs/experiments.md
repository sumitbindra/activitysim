# Swappable pieces and experiments

The core is a framework of swappable pieces. Speed-ups come from experiments:
alternative versions of a piece, plugged in by a setting and measured
against ActivitySim 1.x. Sharrow is not used anywhere in asim_core; 1.x
with sharrow appears only as a baseline in the reports, to check that
nothing is significantly slower.

## Two levels of swapping

1. **Whole steps.** Each model step is a component with declared inputs and
   outputs (see `component.py`). A region picks which components to run, and
   any component can be replaced by another that reads and writes the same
   columns.
2. **Pieces inside a step ("seams").** Each seam is one job, with a
   **reference** implementation (plain NumPy, readable, matches 1.x exactly)
   and any number of alternatives. Which implementation runs is a setting:

   ```python
   Runner(components, out_dir, engine=Engine({"random": "counter"}))
   ```

   Pieces not named use the reference. `step_engines={step: Engine(...)}`
   overrides the choice for individual steps.

| Seam | Job | Reference | Alternatives so far |
|---|---|---|---|
| `lookup` | values of a column at given rows (joins between tables) | `numpy` | |
| `skim_access` | how a step holds the skims it reads | `memory_map` | `prefault`, `in_memory` ([skim_experiments.md](skim_experiments.md)) |
| `skim_reuse` | whether a model's formulas share the skim values they read | `as_written`: every lookup as the formulas are written | `once`: each skim named more than once looked up once per batch, shared until its last use ([skim_experiments.md](skim_experiments.md)) |
| `skim_lookup` | skim values for origin/destination zones | `flat_take`: a one-dimensional gather, positions shared by a step's skims (the default since the skim experiment) | `index_2d` (two-dimensional indexing, the default before) |
| `sample_columns` | which of the choosers' columns trip destination's sampling copies out to every zone | `spec_reads`: only the columns the sample spec reads, as 1.x's `drop_unused_columns` (the default since the trip destination measurement) | `all_columns`: every column the stops carry (the default before; the same results) |
| `size_lookup` | size terms for trip destination's sampling rows (every stop × every zone) | `by_position`: each stop's sizes taken whole as its purpose's column of the size table (the default since the trip destination measurement) | `match`: every row's zone and purpose matched, as 1.x's `DataFrameMatrix.get` (the default before; the same results) |
| `trip_matrix_table` | how `write_trip_matrices` holds the trips while adding up each matrix | `per_matrix`: only the columns some matrix reads; each matrix's table built from its own columns and the matrix written once made (the default since the trip matrices memory change) | `one_table`: every trip column and preprocessor result in one pandas table, a file's matrices written together, as 1.x (the default before; the same matrices) |
| `utilities` | formula values and coefficients → utilities | `matmul`: one matrix product, as 1.x | `accumulate`; `jax`: formulas and sum compiled with JAX ([jax_experiment.md](jax_experiment.md)) |
| `interaction_utilities` | one utility per chooser-alternative row | `sequential`, as 1.x | `jax` (as above) |
| `probabilities` | utilities → probabilities (multinomial logit) | `asim1` | |
| `nested_probabilities` | utilities and a nest tree → probabilities (nested logit) | `asim1` | |
| `sample` | probabilities and draws → sampled alternatives | `binary_search`: 1.x's picks, about log2(alternatives) comparisons per draw (the default since the JAX experiment) | `inverse_cdf`: the same picks, compared as 1.x does, every alternative per draw (the default before) |
| `choose` | probabilities and random draws → choices | `sequential`, as 1.x | |
| `random` | per-row random draws for a step | `vectorized`: 1.x's streams, the same numbers, worked out for all rows at once (the default since the random draws measurement) | `asim1` (reseeded row by row as 1.x, the default before; the same numbers); `counter` (different numbers) |
| `trip_attempts` | how trip scheduling works out one attempt's departures | `arrays`: each trip's row of the probability table found once, the rounds on NumPy arrays, row sums added left to right as pandas does (the default since the trip scheduling measurement) | `pandas`: 1.x's code as ported, a merge with the probability table in every round (the default before; the same departures) |
| `trip_retries` | which of trip scheduling's retried trips sit out an attempt they are certain to fail | `skip_certain_failures`: with `logic_version` 1, while outbound trips are retried, the inbound trips of tours whose outbound trips are not wait, their streams moved on by the draws skipped (the default since the trip scheduling measurement) | `every_attempt`: every retried trip through every attempt, as 1.x (the default before; the same departures) |
| `random_positions` | where each row's place in its random stream is kept between a step's draws | `kept_index`: one pandas index of the IDs per channel, reused while no new IDs appear, positions updated in place (the default since the trip scheduling measurement) | `rebuilt`: a pandas Series of every ID seen so far, rebuilt on every draw (the default before; the same numbers) |

List them with:

```bash
python asim_core/harness/run_experiments.py --list
```

or, with each implementation's description:

```bash
PYTHONPATH=asim_core/src python -m asim_core.seams
```

**When a change becomes the default, the way it worked before stays as an
alternative**, so an agency can run both and compare. Any run script takes
the choices as a flag (`--seams`, JSON; seams not named use the default):

```bash
python asim_core/harness/run_milestone1.py --seams '{"sample_columns": "all_columns", "size_lookup": "match"}'
```

```bash
python asim_core/harness/run_standalone.py --through trip_destination --seams '{"skim_lookup": "index_2d"}'
```

```bash
python asim_core/harness/profile_step.py --engine asim_core --step trip_destination --scratch /tmp/td --out td.csv --seams '{"size_lookup": "match"}'
```

`run_milestone1.py` checks every step against 1.x with the choices given;
`run_standalone.py` runs the steps together and compares the last tables;
`profile_step.py` times one step in fresh processes.

For example, trip destination at 50,000 households (full MTC zones, from
1.x's saved state, skim file warm), each setting giving exactly 1.x's trips:

| `--seams` | Step time | Own memory peak |
|---|---|---|
| `{}` (the defaults: `spec_reads`, `by_position`) | 43.7 s | 991 MB |
| `{"sample_columns": "all_columns", "size_lookup": "match"}` (the way before) | 71.2 s, 75.4 s | 1,132 MB, 1,126 MB |

## How results are compared

Each implementation declares how its results relate to the reference, and
the experiment report checks accordingly:

- **exact:** results identical to the reference.
- **approximate:** identical up to floating-point rounding. The report shows
  the largest utility difference and whether any choice changed.
- **statistical:** individual results differ by design (for example,
  different random numbers), so only distributions can be compared. The
  report shows the largest difference in the share of rows taking any value,
  next to the difference expected by chance.

## Running experiments

List experiments in `harness/experiments.yaml`:

```yaml
step: auto_ownership_simulate
repeat: 5
experiments:
  reference: {}
  counter_random:
    random: counter
```

Then, from the repository root:

```bash
python asim_core/harness/run_experiments.py
```

For each experiment it runs the step from 1.x's saved state and compares the
result with 1.x, then times the step in fresh processes (`repeat` times),
alongside the same step in 1.x legacy and 1.x sharrow. It also times each
piece on its own with synthetic data at larger row counts. Results:
`asim_core/work/experiments/experiments.md` and `.csv`.

## Adding an alternative

Write a function with the same signature as the reference and register it:

```python
import numpy as np
from asim_core.seams import register

@register("choose", "cumulative", agreement="approximate",
          description="first alternative whose cumulative probability reaches the draw")
def choose_cumulative(probs, rands):
    cum = np.cumsum(probs, axis=1)
    return (cum < rands.reshape(-1, 1)).sum(axis=1).astype(np.int32)
```

Import the module before running (built-in alternatives live in
`src/asim_core/alternatives.py`), then name it in `experiments.yaml`. It can
call anything that's callable from Python (Numba, JAX, a C++ extension), as
long as it takes and returns the same data.

## First results

`auto_ownership`, `prototype_mtc` example (5,000 households), Linux, 4 CPUs.
Step times are the median of 5 fresh-process runs.

| Experiment | Result vs 1.x | Households with a different choice | Step time (s) |
|---|---|---|---|
| reference | same as 1.x | 0 | 0.021 |
| `utilities=accumulate` | same as 1.x (utilities differ by at most 7×10⁻¹⁵) | 0 | 0.019 |
| `random=counter` | same distribution: largest share difference 0.72 points, chance level about 1.94 | 2,125 (by design) | 0.010 |
| both | same distribution | 2,125 (by design) | 0.010 |
| 1.x legacy (baseline) | | | 0.180 |
| 1.x sharrow (baseline) | | | 0.188 |

Each piece on its own, synthetic data, 1,000,000 rows (single runs):

| Piece | Implementation | Seconds | Extra memory (MB) |
|---|---|---|---|
| random | `asim1` (the reference then) | 2.28 | 66 |
| random | `counter` | 0.13 | 77 |
| utilities, 29 terms × 5 alternatives | `matmul` (reference) | 0.12 | 260 |
| utilities, 29 terms × 5 alternatives | `accumulate` | 0.29 | 45 |

What this says:

- **Random numbers are the reference's biggest cost** in this step: about
  11 of its 12 ms of calculation. 1.x reseeds a generator for every row in a
  Python loop, and every choice step pays this cost in proportion to its row
  count. `counter` keeps the same rules (results don't depend on row order,
  batching or other rows) and is about 18 times faster at a million rows.
  But it draws different numbers, so results match 1.x only statistically.
  Before relying on it, its statistical quality should be reviewed. Using it
  also means deciding where exact 1.x reproducibility is still needed. One
  option: reference for validation runs, `counter` for production runs.
- **`accumulate` trades speed for memory** in computing utilities: 6 times
  less memory, 2.4 times slower on that piece alone, at a million rows. Here
  utilities are a small part of the step. It becomes worth testing in steps
  where the table of term values is large: mode and destination choice, with
  hundreds of terms.
- **The default is unchanged:** with all reference pieces, results still
  match 1.x exactly.

The example is small, so step times are indicative only. The synthetic
benchmarks show how each piece's cost grows with size.

## Random draws: 1.x's numbers for all rows at once

1.x gives every row its own random stream for each step and reseeds NumPy's
Mersenne Twister for every row, in a Python loop: about 2 µs per row, and
78 to 90% of a simple choice step's time (free parking, auto ownership,
MWCOG's person steps; half of CDAP). A freshly seeded generator's first 113
numbers depend only on part of its seeding (words 0 to 2k and 397 to
396 + 2k for k numbers) and one pass of its twist, so `vectorized` works
the seeding, twist and number assembly out for all rows together in NumPy,
in batches of 65,536 rows. Rows past their first 113 draws in a step, and
normal draws (preprocessors' lognormal values), still go row by row.

- **The same numbers, bit for bit**: against `asim1` (first draws, more
  draws later in the step, subsets of rows, past the first 113, normal then
  uniform) and against 1.x's own `random_for_df` (5 rows in the tests; a
  million rows: identical).
- **The same results everywhere**: every step of MTC, MWCOG (100 and all
  households), ARC and full MTC (10,000 and 50,000 households) exactly
  1.x's, from saved state and end to end; full MTC at 50,000 households
  from the input files: 219 of 219 columns.
- **Faster**: a million rows, one draw each, 2.06 s → 0.28 s (1.x's
  `random_for_df`: 2.42 s); 30 draws per row (destination sampling) 1.6
  times. The whole model, full MTC, 50,000 households, skim file warm:

| `--seams` | Run 1 | Run 2 |
|---|---|---|
| `{"random": "asim1"}` (row by row, the default before) | 148.2 s | 148.4 s |
| `{}` (the default: `vectorized`) | 139.1 s | 134.3 s |

  The largest gains per step: trip mode choice 6.6 → 4.8 to 5.2 s, trip
  scheduling 19.4 → 17.3 to 18.6 s, CDAP 1.1 → 0.5 s, auto ownership and
  free parking 0.2 → 0.05 s. Trip destination hardly changes: its time is
  not in the draws.
- Not done: normal draws (1.x's polar method rejects some pairs, so rows
  use different amounts of their stream) and rows past 113 draws.


## Trip scheduling: the same departures in a fifteenth of the time

Trip scheduling was the one model step no faster than 1.x: it was 1.x's
pandas code, ported as is. Measured from 1.x's saved state, the first
attempt schedules every trip in about 0.6 s (SANDAG one-zone: 416,699
trips); the other 99 attempts, about 14,500 retried trips each, took the
rest. Three changes, each a seam with the way before one flag away, and
each with exactly 1.x's departures:

- **`random_positions`: stream positions in a reused index.** Every draw
  rebuilt a pandas Series of every ID drawn for so far in the step (600
  draws over 442,682 trips on full MTC), half the step's time. `kept_index`
  keeps one index per channel and updates positions in place. It is part of
  the random draws every step makes, so other steps that draw in rounds gain
  too (SANDAG's joint tour participation: about 5.0 → 2.9 s).
- **`trip_retries`: retries certain to fail sit out.** With 1.x's default
  `logic_version` 1, the inbound trips of a tour whose outbound trips are not
  being retried have no earliest period while other tours' outbound trips
  are, so they fail on every such attempt whatever is drawn: about 70% of
  every retry (1.x finding 18). `skip_certain_failures` holds them back and
  puts them back when no outbound trip is left (1.x then gives them their
  usual window again) or on the last attempt, their random streams moved on
  by the draws they skipped, so every later draw is 1.x's.
- **`trip_attempts`: the attempts on arrays.** 1.x merges the trips with the
  probability table in every round of every attempt. `arrays` finds each
  trip's row once and does the rounds on NumPy arrays. pandas adds up each
  row of these tables left to right, while NumPy's own sum adds in another
  order and differs in the last bit in 328 of SANDAG's 600 rounds, so the
  row sums are added left to right too (`_row_sums`; checked against pandas
  for every round of every attempt on SANDAG, 1.1 million trip rows).

Trip scheduling alone, from 1.x's saved state, two runs each (every run:
exactly 1.x's departures):

| `--seams` | SANDAG one-zone | Full MTC, 50,000 households |
|---|---|---|
| `{"trip_attempts": "pandas", "trip_retries": "every_attempt", "random_positions": "rebuilt"}` (all three as before) | 15.3, 18.1 s | 15.8, 21.2 s |
| `{"trip_attempts": "pandas", "trip_retries": "every_attempt"}` (positions kept) | 7.8, 10.5 s | 10.1, 11.2 s |
| `{"trip_attempts": "pandas"}` (and retries certain to fail sit out) | 3.7, 3.7 s | 4.0, 4.5 s |
| `{}` (the defaults: and attempts on arrays) | 0.96, 1.05 s | 1.03, 1.05 s |
| `{"trip_retries": "every_attempt"}` (arrays, every retry) | 2.6, 2.6 s | 4.0, 3.1 s |

The whole model from the input files (`run_standalone.py`), the defaults
against all three as before, runs alternating, skim file in memory:

| | Defaults | As before |
|---|---|---|
| SANDAG one-zone, all 33 steps | 64.3, 68.7 s | 84.9, 86.4 s |
| of which trip scheduling | 0.9, 1.1 s | 15.0, 18.0 s |
| Full MTC, 50,000 households, all 34 steps | 137.8, 133.9 s | 174.1, 170.5 s |
| of which trip scheduling | 1.1, 0.9 s | 18.9, 20.1 s |
| of which trip destination | 33.2, 32.7 s | 43.1, 39.3 s |

(A first full MTC run with the defaults, right after the SANDAG runs, took
216 s: the 7.2 GB skim file was no longer in memory. Trip destination's gain
is the stream positions: at full MTC it samples in many batches, each with
its own draws.) Memory is unchanged (own peak: SANDAG 1.35 to 1.37 GB either
way; full MTC 2.5 GB, against 2.6 GB). Every
run was exactly 1.x's: SANDAG 16 of 16 output files, full MTC 40 of 40,
and MTC, MWCOG and ARC as before.

- **Tests**: `test_trip_scheduling.py` runs every combination of the two
  trip seams on made-up trips (failures common, 3 to 100 attempts,
  `logic_version` 1 and 2) and checks that held trips rejoin both ways;
  `test_random.py` checks that both ways of keeping positions draw the same
  numbers.
- **The one case the retries change could differ**: a held trip's skipped
  draw of exactly 0.0, which would pick the first period instead of failing
  (asim_core's code as before would then stop: a departure outside the
  window). The chance is about one in 10^16 per draw.
