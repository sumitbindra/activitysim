# Milestone 1: a minimal core, checked against ActivitySim 1.x

## Summary

- **A small core exists** (`asim_core/`, about 1,750 lines including
  documentation). It holds data, runs components, saves results, compares
  with 1.x and profiles every step. It depends only on NumPy, pandas,
  PyArrow, Pydantic and PyYAML, and does not import ActivitySim 1.x.
- **`auto_ownership` is ported and matches 1.x exactly.** All 5,000
  households make the same choice. All 25,000 utilities and 25,000
  probabilities are bit-for-bit identical to the values 1.x's own functions
  compute. It reads today's YAML and CSV files unchanged.
- **The model runs end to end** with `auto_ownership` computed and the other
  33 steps replayed from a saved 1.x run. The results still match at the end.
- **Every 1.x step is profiled** (runtime and peak memory), with the legacy
  engine and with sharrow. On this example, trip destination alone is about
  a third of the legacy engine's runtime.
- **What this does not show yet:** anything about beating sharrow on a
  real-size model. The example is tiny (25 zones, 5,000 households), and
  `auto_ownership` is about 0.1% of the runtime.

_Sharrow results are pending: its one-time compile run was still in progress
when this draft was written._

## How it was checked

All runs use the `prototype_mtc` example in this repository (25 zones,
5,000 households, 8,212 persons), single process, with 1.x saving its state
after every step.

1. **Step check.** Load 1.x's saved state after `workplace_location`, run
   asim_core's `auto_ownership`, and compare the result with 1.x's saved state
   after `auto_ownership_simulate`, household by household.
2. **Utility check.** From the same saved state, ask 1.x's own functions
   (`eval_utilities`, `utils_to_probs`, legacy engine) for every household's
   utilities and probabilities, and compare them with asim_core's values.
3. **End to end.** Start from nothing. Replay each 1.x step from the saved
   run, except `auto_ownership_simulate`, which asim_core computes. Whenever a
   later 1.x step rewrites the households table, compare asim_core's
   `auto_ownership` with 1.x's copy, and compare again at the end.

## Matching results

| Check | Compared | Result |
|---|---|---|
| Step check: choices | 5,000 households | exact (same value and type, `int32`) |
| Utility check: utilities | 25,000 values (5,000 x 5 alternatives) | bit-for-bit identical |
| Utility check: probabilities | 25,000 values | bit-for-bit identical |
| End to end, after `cdap_simulate` | 5,000 households | exact |
| End to end, after `joint_tour_frequency` | 5,000 households | exact |
| End to end, end of run | 5,000 households | exact |

Shares by number of cars (identical in both): 0 cars 62.40%, 1 car 35.96%,
2 cars 1.52%, 3 cars 0.04%, 4 cars 0.08%.

### What exact matching needed

These 1.x details all had to be reproduced. Each would cause small
differences if it were changed:

- **Coefficients are rounded to 32-bit floats** before use, so a coefficient
  of 3.0773 is actually 3.0773000717. The rounding is kept by default and can
  be turned off.
- **Terms whose coefficients are all zero are dropped** (legacy engine only).
- **Utilities are one matrix multiplication** of term values (terms ×
  choosers, 64-bit) by coefficients, with the same array layout.
- **No overflow protection** when `skip_failed_choices` is on (1.x's
  default): utilities are exponentiated without subtracting each row's
  maximum first.
- **Choices are made by subtracting** each alternative's probability from the
  random draw in turn, not by comparing with cumulative sums. The two differ
  in the last bits.
- **Random numbers come from per-row streams.** Each stream is seeded from the
  base seed, a hash of the channel name (`households`), a hash of the step
  name (`auto_ownership_simulate`) and the household ID, and restarts every
  step. Nothing has to be saved to reproduce them, but a ported component
  must keep 1.x's step name.

### One deliberate difference

1.x skips households whose probabilities can't be computed
(`skip_failed_choices`, on by default) and lists them in
`households_skipped`. asim_core stops and names them instead, because
silently dropping households makes later comparisons harder to interpret.
No household was skipped in this run, so the difference didn't come up.

## Profiles

### ActivitySim 1.x, full run, legacy engine

101 seconds end to end (94 seconds inside steps). Peak memory 875 MB; the
process already held 396 MB before the first step. These are single runs.

The ten slowest steps:

| Step | Seconds | Share of step time | Extra memory during step (MB) |
|---|---|---|---|
| trip_destination | 30.2 | 32% | 1.7 |
| mandatory_tour_scheduling | 11.2 | 12% | 243.6 |
| trip_scheduling | 7.5 | 8% | 0.3 |
| non_mandatory_tour_destination | 5.7 | 6% | 1.9 |
| tour_mode_choice_simulate | 5.3 | 6% | 0.0 |
| trip_mode_choice | 5.1 | 5% | 0.0 |
| cdap_simulate | 4.7 | 5% | 3.5 |
| workplace_location | 4.3 | 5% | 104.2 |
| school_location | 4.2 | 4% | 65.9 |
| non_mandatory_tour_frequency | 2.8 | 3% | 1.6 |

`auto_ownership_simulate` took 0.12 seconds (0.1%). The full per-step
table, with sharrow next to it, is in `work/milestone1/asim1_profiles.csv`
after running the harness.

### `auto_ownership`, each engine in a fresh process (5 runs each)

| Engine | Seconds (median) | Extra memory (MB) |
|---|---|---|
| asim_core | about 0.02 | about 6 |
| 1.x legacy | about 0.18 | about 22 |
| 1.x sharrow | _pending_ | _pending_ |

### How profiles were measured

- The same profiler measures both engines: wall-clock time, CPU time, and
  peak memory during the step. Peak memory comes from Linux's per-process
  high-water mark, reset before each step. Only the standard library is used.
- **Full-run profiles** time each step inside one complete 1.x run. These
  are single runs.
- **Step profiles** run one step in a fresh process, starting from 1.x's
  saved state, five times per engine. Loading the state is not timed; the
  step, including saving its results, is.
- **Two traps found along the way, both handled:**
  - Inside a long run, memory freed by earlier steps is reused, so a step's
    extra memory looks smaller than it is. For example, `auto_ownership`
    shows 0.2 MB inside the full run but about 22 MB in a fresh process.
    Memory should be compared across engines only under the same conditions.
  - In a fresh process, 1.x spends about a second compiling its choice
    function with Numba on first use: the step took 1.14 seconds without a
    warm-up and 0.17–0.19 seconds with one. In a full run that compile
    happens in an earlier step, so the step profiles do it before timing
    starts.
- Even in a fresh process, the 1.x process holds much more memory before
  any step runs (about 400 MB, versus about 170 MB for asim_core). Most of
  that is the libraries 1.x loads.

## What the `auto_ownership` speed-up does and does not mean

asim_core's `auto_ownership` is faster than 1.x's on this example. Most of
the difference comes from work around the calculation, not the calculation
itself:

- 1.x builds merged copies of the households and persons tables. The persons
  copy is built even though auto ownership doesn't use it; there's a FIXME
  about this in 1.x's code. asim_core looks up only the 16 columns the
  formulas use.
- 1.x saves the whole households table after the step; asim_core saves the
  one new column.

These are the kinds of savings the foundation was designed for, but at
5,000 households and 0.1% of runtime they say nothing yet about the steps
that matter.

## What's next

1. **Profile a full-size region on representative hardware.** On this
   example, trip destination is about a third of the runtime and mandatory
   tour scheduling has the largest memory jump. A real region will rank
   steps differently, because location choice grows with the number of
   zones. The harness runs unchanged on any 1.x example:
   `run_asim1.py --example-dir <your model>`.
2. **Support preprocessors and annotations in the translator.** Of the 17
   model files in the example, only `auto_ownership` loads today, and 15 of
   the other 16 use preprocessors or annotations: formulas that add columns
   before or after a model runs. That's the biggest single unlock for porting
   simple models (`free_parking`, `mandatory_tour_frequency`,
   `joint_tour_frequency`, ...).

   Across all of the example's spec files, the translator already handles
   1,759 of 2,905 formulas (61%). Counts of what blocks the rest (grouped
   roughly by pattern):

   | Needs | Formulas |
   |---|---|
   | skim lookups | 506 |
   | temporary variables in preprocessors (`_name`) | 198 |
   | columns of other tables (`persons.age`, `trips.depart`, ...) | about 160 |
   | `reindex()` | 87 |
   | names defined outside the spec file | 56 |
   | `.map()` / `.astype()` | 27 |
   | other (`other_than()`, `apply`, `setting()`, random draws, ...) | about 110 |
3. **Port one skim-heavy location choice model** (for example
   `workplace_location`). It needs the skim store, sampling of alternatives
   and size terms. This is where the foundation has to show memory and speed
   gains.
4. **Then try alternative engines** (Numba, JAX) behind the same model
   definitions, measured with the same harness.

## Reproducing

See `asim_core/README.md`. In short, from the repository root:

```bash
python asim_core/harness/run_asim1.py --out asim_core/work/asim1_legacy
```

```bash
python asim_core/harness/run_milestone1.py
```

Environment for these results: Linux, 4 CPUs, 15 GB RAM, Python 3.10,
NumPy 2.2.6, pandas 2.3.2, Numba 0.62.0, sharrow 2.16.2.
