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
- **Every 1.x step is profiled** (runtime and peak memory) with both 1.x
  engines:
  - Legacy: 101 seconds, peak 875 MB.
  - Sharrow: 83 seconds, peak 1,010 MB, after about 37 minutes of one-time
    compiling.
  - Trip destination is the slowest step under both engines.
- **1.x's two engines give identical results on this example**, so "exact"
  is the right bar for matching.
- **What this does not show yet:** anything about beating sharrow on a
  real-size model. The example is tiny (25 zones, 5,000 households), and
  `auto_ownership` is about 0.1% of the runtime.

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

### How much 1.x differs from itself

The final results of 1.x's legacy and sharrow runs were compared column by
column: car ownership, daily activity pattern, work and school location,
tour mode, destination and start time, and trip mode and destination.
Every value is identical for every row (5,000 households, 8,212 persons,
9,806 tours, 23,583 trips). On a larger model the two engines may differ
slightly, and this comparison is the way to find out how much. That
difference sets a fair tolerance for asim_core.

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

### ActivitySim 1.x, full runs

| | Legacy engine | Sharrow |
|---|---|---|
| Total run time | 101 s | 83 s |
| Time inside steps | 94 s | 74 s |
| Peak memory | 875 MB | 1,010 MB |
| Memory held before the first step | 396 MB | 396 MB |
| One-time compiling before the first run | none | about 37 minutes, peak 4.4 GB (see below) |

The ten slowest steps under the legacy engine:

| Step | Legacy (s) | Sharrow (s) | Legacy extra memory (MB) | Sharrow extra memory (MB) |
|---|---|---|---|---|
| trip_destination | 30.2 | 19.1 | 1.7 | 137.6 |
| mandatory_tour_scheduling | 11.2 | 7.0 | 243.6 | 185.2 |
| trip_scheduling | 7.5 | 7.3 | 0.3 | 2.0 |
| non_mandatory_tour_destination | 5.7 | 2.4 | 1.9 | 15.1 |
| tour_mode_choice_simulate | 5.3 | 3.1 | 0.0 | 0.0 |
| trip_mode_choice | 5.1 | 3.3 | 0.0 | 22.2 |
| cdap_simulate | 4.7 | 4.0 | 3.5 | 30.1 |
| workplace_location | 4.3 | 2.1 | 104.2 | 32.7 |
| school_location | 4.2 | 4.4 | 65.9 | 60.1 |
| non_mandatory_tour_frequency | 2.8 | 1.4 | 1.6 | 6.2 |

Trip destination alone is 32% of the legacy engine's step time and 26% of
sharrow's. Per step, sharrow ranges from 0.65 to 2.3 times the legacy
engine's speed. Every step, for both engines, is in
`work/milestone1/asim1_profiles.csv` after running the harness. These are
single runs. Extra memory inside a full run is understated: see "How
profiles were measured".

### `auto_ownership`, each engine in a fresh process (5 runs each)

| Engine | Seconds, median (range) | Extra memory, median (MB) | Seconds inside a full 1.x run |
|---|---|---|---|
| asim_core | 0.026 (0.018–0.030) | 6.2 | |
| 1.x legacy | 0.209 (0.179–0.249) | 24.6 | 0.123 |
| 1.x sharrow | 0.201 (0.184–0.239) | 25.4 | 0.113 |

### Sharrow's one-time compile

Before sharrow runs fast, it compiles every model's formulas to machine code
with Numba. On this example, its warm-up run of 100 households:

- **First attempt:** ran for about 27 minutes, all of it on the CPU, and was
  stopped by this environment's 30-minute limit for background jobs while
  still compiling `trip_destination`. The 43 compiled functions (60 MB)
  were saved to the cache.
- **Second attempt:** reused those and finished in 9.6 minutes. Compiling
  `trip_mode_choice` alone took 337 seconds and 2.5 GB of extra memory, and
  `trip_destination` 150 seconds and 1.3 GB. Peak memory was 4.4 GB, five
  times the legacy engine's full run.

The cache is keyed by a fingerprint of each model's formula text, the
values of the constants they use, the data layout and the compile settings
(sharrow `flows.py`). Editing a formula or constant recompiles that model's
whole function. Coefficients are passed in as data, so recalibrating them
doesn't. The cache folder is named after the sharrow and Numba versions
(`~/.cache/ActivitySim/sharrow-2.16-numba-0.62.0`), so upgrading either one
means compiling again.

asim_core has no compile step: its generated code is plain NumPy and takes
milliseconds to prepare. Whether NumPy alone is fast enough for the heavy
steps, or a compiled engine is needed for some of them, is the main open
question for the next milestone. If one is needed, it should compile small
building blocks that don't depend on any model's formulas, so editing a
formula never triggers a compile.

### How profiles were measured

- **One profiler for both engines.** It records wall-clock time, CPU time,
  and peak memory during the step. Peak memory comes from Linux's
  per-process high-water mark, reset before each step. Only the standard
  library is used.
- **Full-run profiles** time each step inside one complete 1.x run.
- **Step profiles** run one step in a fresh process, starting from 1.x's
  saved state, five times per engine. Loading the state is not timed; the
  step, including saving its results, is.
- **Four traps found along the way, all handled:**
  - **Memory reuse inside a long run.** Memory freed by earlier steps is
    reused, so a step's extra memory looks smaller than it is. For example,
    `auto_ownership` shows 0.2 MB inside the full legacy run and about 25 MB
    in a fresh process. Compare memory across engines only under the same
    conditions.
  - **1.x's choice-function compile.** In a fresh process, 1.x first spends
    about a second compiling its choice function with Numba (the step took
    1.14 seconds without a warm-up). In a full run this happens in an
    earlier step, so the step profiles compile it before timing starts.
  - **Sharrow compiles that function a second time** for its 32-bit
    probabilities, another 0.4 seconds, so the warm-up covers both
    versions.
  - **Sharrow loads the whole skim dataset** the first time any sharrow
    model runs in a process, even for models that use no skims. This took
    about 6 seconds in a fresh process here, more than half of it writing a
    skim cache to disk. In a full run it happens in an earlier step, so the
    step profiles do it before timing starts.
- **Memory held before any step runs.** The 1.x process holds 400–525 MB
  before running a step; asim_core holds about 170 MB. Most of 1.x's figure
  is the libraries it loads.

## What the `auto_ownership` speed-up does and does not mean

asim_core's `auto_ownership` is about 8 times faster than either 1.x engine
when run the same way. Most of the difference comes from work around the
calculation, not the calculation itself:

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
   example, trip destination is the slowest step and mandatory tour
   scheduling has the largest memory jump. A real region will rank steps
   differently, because location choice grows with the number of zones.
   The harness runs unchanged on any 1.x example:
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
   gains against sharrow, and where the NumPy-or-compiled question gets
   answered.
4. **Track "time from editing a formula to results"** next to runtime and
   memory for every engine, since compile time is a large part of what users
   experience.

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
