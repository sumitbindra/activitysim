# Findings for ActivitySim 1.x

Problems and possible improvements in ActivitySim 1.x found while building
asim_core and comparing it with 1.x step by step. This file is about 1.x
only: each entry stands on its own, so it can be turned into an issue (or
a pull request) for the ActivitySim project. Nothing here has been
reported yet.

Each entry says how solid the evidence is:

- **Reproduced**: seen in a 1.x run, with a command to see it again.
- **Measured**: numbers from a 1.x run or from 1.x's own functions.
- **From reading the code**: not yet seen in a run.

All runs used this repository's 1.x code and its example models
(`activitysim/examples/prototype_mtc`, `prototype_mwcog`, `prototype_arc`),
on a 4-CPU, 15 GB Linux machine. The scripts referred to are in
`asim_core/harness/`; they import 1.x and work on 1.x's saved state.

## Summary

| # | Finding | Kind | Evidence | Status |
|---|---|---|---|---|
| 1 | `mandatory_tour_frequency` crashes when a household's choice fails | Bug | Reproduced (MWCOG, all households) | Not reported |
| 2 | Tour scheduling computes the same mode choice logsums many times | Speed and memory | Measured; patch tested (results identical) | Not reported |
| 3 | Results depend on how many rows are evaluated at once | Reproducibility | Measured; one-line fix tested | Not reported |
| 4 | Tour mode choice's choice draw reuses random numbers already used | Random numbers | Measured | Not reported |
| 5 | Building every tour × time option at once uses gigabytes | Memory | Measured | Not reported |
| 6 | Numba functions are compiled again in every run | Speed | Measured | Not reported |
| 7 | Time-of-day skim lookups are a third of tour mode choice's time | Speed | Measured | Not reported |
| 8 | Tours of a type with no spec segment are silently left unscheduled | Bug (configuration) | From reading the code | Not reported |
| 9 | Smaller items (5) | Various | From reading the code, one measured | Not reported |
| 10 | Bad input data runs silently: negative employment changed results across the model with no error or warning | Data checking | Reproduced (MTC) | Not reported |
| 11 | Numbers in CSV files are read inexactly (pandas' default parser): up to 1e-12 relative error, in inputs and in coefficients | Accuracy | Measured (all three examples) | Not reported |

---

## 1. `mandatory_tour_frequency` crashes when a household's choice fails

**Evidence: reproduced.** MWCOG, all 29,972 households.

**What happens.** Three households (11310, 14815, 18648) get probability
zero for every alternative in `mandatory_tour_frequency`. With
`skip_failed_choices` on (the default), 1.x marks their choices as -99 and
reports them as skipped. The same step then turns choice positions into
alternative names with `model_spec.columns[choices.values]`
(`activitysim/abm/models/mandatory_tour_frequency.py`, line 136). -99 isn't
a valid position, so the run stops:
`IndexError: index -99 is out of bounds`.

The example's default 100-household sample doesn't include these
households, which is likely why it hasn't been seen.

**To reproduce:**

```bash
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/prototype_mwcog --households 0 --out asim_core/work/mwcog/asim1_legacy_all
```

**Suggested change.** Drop or flag the skipped choosers before mapping
positions to names, as other steps do. Separately, why these three
households have no usable alternative may be worth a look (a data or spec
issue in the MWCOG example).

---

## 2. Tour scheduling computes the same mode choice logsums many times

**Evidence: measured; a patch was tested.**

**What happens.** Mandatory tour scheduling computes a tour mode choice
logsum for every tour × (departure period, return period, duration)
combination (`compute_tour_scheduling_logsums`, with `dedupe_alt_tdd` in
`activitysim/abm/models/util/vectorize_tour_scheduling.py`). On ARC, the
mode choice formulas never use duration, so only the (departure period,
return period) pairs give different results: 21 per tour instead of up to
273. About 13 times more logsums are computed than needed.

**Measured on ARC, 4,000 households:**

| | 1.x | 1.x with the patch |
|---|---|---|
| Mandatory tour scheduling | 44.8 s, +2,322 MB | 13.0 s, +660 MB |
| Whole run | 137 s, peak 2,929 MB | 102 s, peak 1,480 MB |

Logsums computed both ways in the same run (1,000 households): all
1,585,229 bit-for-bit identical. Complete 4,000-household runs with and
without the patch: every table and column identical.

Without the patch, 1.x ran out of memory (about 14 GB) on all 24,061 ARC
households. With it, all 34 steps finished in 347 s with a 5.8 GB peak.

**The patch** (`asim_core/harness/asim1_patches.py`, `logsum_dedupe`)
computes one logsum per (tour, departure period, return period) and joins
it back with 1.x's existing join. It applies only when there is no
`tour_departure_and_duration_segments.csv` and no file of the logsum model
mentions `duration`; otherwise it leaves 1.x alone.

**To reproduce:**

```bash
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/prototype_arc --households 4000 --out asim_core/work/arc/full4000_base
```

```bash
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/prototype_arc --households 4000 --patch logsum_dedupe --out asim_core/work/arc/full4000_patched
```

```bash
python asim_core/harness/compare_asim1_runs.py asim_core/work/arc/full4000_base asim_core/work/arc/full4000_patched
```

**Suggested change.** Deduplicate logsum inputs on the columns the logsum
model actually reads, rather than on a fixed set. A general version: find
the chooser columns used by the mode choice spec and preprocessor, and
compute one logsum per distinct combination. Details:
`asim_core/docs/scheduling.md`.

**Also shown in asim_core**, which implements this rule generally (it
knows which columns every formula reads): on ARC at 4,000 households it
computes 81,759 logsums instead of 1,245,334, and all 6,360,778 (tour,
option) logsums are bit-for-bit identical to 1.x's
(`asim_core/docs/mandatory_scheduling.md`).

---

## 3. Results depend on how many rows are evaluated at once

**Evidence: measured; a one-line fix was tested.**

**What happens.** pandas evaluates Series arithmetic with NumPy, except for
arrays with more than 1,000,000 elements, where it switches to numexpr
(`pandas.core.computation.expressions`, `_MIN_ELEMENTS`). numexpr's type
rules differ from NumPy's. For example:

| Operation | Under 1M elements | Over 1M elements |
|---|---|---|
| float32 Series × 0.3 | computed in float32 | computed in float64 |
| float32 Series × int64 Series | computed in float64 | computed in float32 |

1.x evaluates `@` formulas on pandas Series. So when the number of rows
evaluated at once crosses 1M, the same formula gives slightly different
values. That number depends on the household sample, on the chunk
settings, and on how many choosers a step processes together.

**Measured** on ARC's non-mandatory tour scheduling, first round, 4,000
households (3,069,657 tour × time-option rows), with 1.x's own
`eval_interaction_utilities`:

| Comparison | Bit-identical utilities |
|---|---|
| Whole round at once (1.x's default) vs the same rows in slices of 500,000 | 264,769 of 3,069,657 (largest difference 1.7e-7) |
| Whole round with `pd.set_option("compute.use_numexpr", False)` vs slices | 3,069,657 of 3,069,657 |

Differences of 1e-7 rarely change a choice: none of ARC's 30,851
non-mandatory tours changed with all households. But results aren't
strictly independent of chunk size, and changing the sample size changes
utilities for everyone.

A related inconsistency: plain-style formulas (no `@`) go through
`pandas.eval`, which always uses numexpr. So the same arithmetic gives
different last bits depending on whether a formula is written with `@`.

**To reproduce:** `scheduling_round_utilities` in
`asim_core/harness/asim1_reference.py` computes a scheduling round's
utilities with 1.x's functions, whole or in slices (`rows_per_call`).

**Suggested change.** Set `pd.set_option("compute.use_numexpr", False)`
for 1.x's expression evaluation (tested above: removes the dependence).
Possible cost: numexpr is multi-threaded, so very large arithmetic may be
slower; not measured.

---

## 4. Tour mode choice's choice draw reuses random numbers already used

**Evidence: measured.**

**What happens.** MTC's and MWCOG's tour mode choice preprocessors draw
six lognormal wait times per tour (`rng.lognormal_for_df`). 1.x draws all
six with `normal_for_df(..., size=6)` (`activitysim/core/random.py`), then
moves each tour's random stream position on by 6. But NumPy's legacy
normal generator (polar method) uses two uniform numbers per pair of
normals and throws pairs away about 21% of the time, so six normals use 6,
8, 10, ... uniform numbers. The mode choice draw that follows takes the
7th uniform number of the stream, which has often already been used to
make a wait time.

Measured over 100,000 random seeds: **in 51.4% of rows the choice draw is
a uniform number the wait-time draws already used** (theory:
1 − (π/4)³ ≈ 51.6%).

**Also in mandatory tour scheduling.** The same preprocessor runs inside
the scheduling logsums (MTC, MWCOG), once per tour, before the scheduling
choice draw; that draw then has the same overlap.

**Effect today:** none in these examples, because their wait-time
standard deviations are all zero, so the wait times don't depend on the
draws. In a model with non-zero standard deviations, a tour's wait time and
its mode choice draw (or its scheduling draw) would be correlated.

**Suggested change.** Move the stream position by the number of uniform
numbers actually used; or make each normal from exactly one uniform
(inverse normal CDF); or draw wait times from a separate channel.

---

## 5. Building every tour × time option at once uses gigabytes

**Evidence: measured.**

**What happens.** With default settings, tour scheduling builds one round's
whole table of tour × departure-return options at once
(`tdd_interaction_dataset`, then `interaction_sample_simulate`). On ARC
with all households, non-mandatory scheduling's first round is 18.6
million rows; the step needs **+5.2 GB** (151 s, measured in a fresh
process from 1.x's saved state).

Processing the same rounds in batches of at most 1M rows (each tour's rows
depend only on its own person's timetable) gives identical results with
**+144 MB** (37.5 s), as asim_core shows.

1.x has an `explicit_chunk` setting that should do something similar.
Whether it gives the same saving wasn't tested. Note that because of
finding 3, chunking currently changes 1.x's results in the last bits.

**Suggested change.** Batch scheduling rounds by default (after fixing
finding 3, so batching doesn't change results).

---

## 6. Numba functions are compiled again in every run

**Evidence: measured.**

**What happens.** 1.x's Numba functions (the timetable functions in
`activitysim/core/timetable.py`, `choice_maker` in `choosing.py`,
accessibility's accumulator) use `@nb.njit` without `cache=True`, so
they're compiled in every run, and in every worker process of a
multiprocess run. On MTC, the first scheduling step in a fresh process
took 5.8 s with compiling and 1.9 s without. The timetable functions
alone compile in about 4 s. The compiled code also depends on argument
types and array layout (e.g. a timetable read from a saved run is
column-major), so a function can be compiled more than once per run.

**Suggested change.** `@nb.njit(cache=True)` where possible (check how the
cache folder behaves on read-only installs and shared file systems).

---

## 7. Time-of-day skim lookups are a third of tour mode choice's time

**Evidence: measured** (Python's profiler, MTC, legacy engine).

**What happens.** Tour mode choice's formulas look up time-of-day skims
(`odt_skims['SOV_TIME']`, ...) 2,853 times on MTC. Each lookup maps every
row's period label to a skim with `np.vectorize(dict.get)`
(`Skim3dWrapper` → `SkimDict.lookup_3d`,
`activitysim/core/skim_dictionary.py`) and builds a new pandas Series.
Under the profiler these lookups took 2.8 s of the step's 8.0 s;
`np.vectorize` alone took 0.56 s. Overall, evaluating the formulas one by
one as pandas operations is about 60% of the step.

**Suggested change.** Map period labels to skim positions once per set of
choosers (e.g. categorical codes) and reuse them for every lookup.

---

## 8. Tours of a type with no spec segment are silently left unscheduled

**Evidence: from reading the code** (not seen in the example models).

**What happens.** With `TOUR_SPEC_SEGMENTS` (e.g. MWCOG's non-mandatory
scheduling), `vectorize_tour_scheduling` schedules only tours whose
`tour_type` is listed. Tours of any other type get no choice, and nothing
checks that every tour was scheduled. `assign_in_place` then leaves their
start and end empty, and later steps receive tours without times.

**Suggested change.** Check after scheduling that every chooser has a
choice, and name the unlisted tour types if not.

---

## 9. Smaller items

From reading the code; low priority.

- **Preprocessor results can't overwrite with a missing value.**
  `assign_in_place` updates existing columns with `DataFrame.update`, which
  skips missing values. A preprocessor row that recomputes an existing
  column keeps the old value wherever the new one is missing.
- **`reindex_i` returns 8-bit integers by default**
  (`activitysim/core/util.py`). Counts above 127 would not fit.
- **Tour mode choice re-reads its spec, coefficients and coefficient
  template for every segment** (`run_tour_mode_choice_simulate`). Measured
  at about 2% of the step on MTC.
- **A per-segment `ALTS_PREPROCESSOR` is skipped when there are no
  logsums.** Tour scheduling looks it up by the logsum purpose
  (`_schedule_tours`, marked "FIXME this is not always right" in the
  code), which is empty without `LOGSUM_SETTINGS`, so the preprocessor
  silently doesn't run. The examples all have logsums, so it doesn't
  affect them.
- **A quoted comment line in ARC's `destination_choice_size_terms.csv`
  is read as a size term row.** The line starts with `"#ctramp ...`, in
  quotes, so `read_csv(comment="#")` doesn't skip it: the table gets a
  row with an empty segment and all-zero terms. Harmless today (no model
  selects it), but it would surprise a check that every segment has a
  positive size.

---

## 10. Bad input data runs silently

**Evidence: reproduced** (MTC, 2,000 households, first 17 steps).

**What happens.** The employment columns of one zone (zone 8) were made
negative, as a paste with a flipped sign would. 1.x ran all 17 steps with
no error and no warning about it, and the results changed across the
model:

| | Clean data | Negative employment in zone 8 |
|---|---|---|
| Zones whose accessibility changed | | 25 of 25 |
| Workers with a workplace in zone 8 | 25 | 0 |
| Workers whose workplace changed | | 1,620 of 1,779 |
| Households whose car ownership changed | | 119 of 2,000 |
| Non-mandatory tours to zone 8 | 49 | 5 |

1.x has an optional `input_checker` step (pandera classes written in
Python per region, in a `data_model` folder), but the MTC, MWCOG and ARC
examples don't turn it on, and its row-by-row Pydantic option is marked as
still in development.

**To reproduce:** copy `activitysim/examples/prototype_mtc`, negate the
employment columns of `TAZ` 8 in `data/land_use.csv`, and run both copies
with `run_asim1.py --households 2000 --stop-after non_mandatory_tour_destination`.

**Suggested change.** Basic input checks by default: non-negative
employment and sizes, valid zone references, unique IDs, household size
matching persons. Also checks derived from how models use columns (size
terms can't be negative, log inputs must be positive), all reported before
the first step.

Measured since: asim_core's column-at-a-time checks (NumPy) read and
check 2.4 million households and 6.7 million persons in 2.1 s and 2.8 GB
at peak; reading the same files with pandas, as 1.x does, takes 8.2 s and
4.0 GB without any checks (`asim_core/harness/bench_input_checks.py`).
So checks by default need not slow a run down.

---

## 11. Numbers in CSV files are read inexactly

**Evidence: measured** (all three examples; inputs and configs).

**What happens.** 1.x reads CSV files with `pandas.read_csv` and its
default float parser, which is fast but not exact: it keeps about 16
significant digits and doesn't round the last one correctly. Reading
with `float_precision="round_trip"` gives the closest float to the text
(the same value as Python's `float()` and PyArrow).

| Example | File | Numbers | Read differently | Largest difference |
|---|---|---|---|---|
| MTC | land_use.csv | 250 | 22 | 1 unit in the last place |
| MTC | config CSVs (9 files) | 31,377 | 29 | 1 unit in the last place |
| MWCOG | land use | 265 | 9 | 6 units in the last place |
| MWCOG | config CSVs (4 files) | 181,687 | 9,488 | 7,214 units in the last place (relative 1e-12) |
| ARC | households.csv (`hhincAdj`) | 24,061 | 364 | 1 unit in the last place |

For example, MWCOG's `trip_scheduling_probs_purpose_stops.csv` has
`0.00010113064033479776`, read as `0.0001011306403347`. The MWCOG
files affected include `tour_mode_choice_coeffs.csv` (42 coefficients)
and `trip_mode_choice_coefficients.csv` (34).

The differences are tiny, but they mean 1.x's results depend on which
CSV parser reads the files, and any other tool (PyArrow, Polars, a
database) reading the same inputs starts from slightly different numbers.

**To reproduce:** compare `pd.read_csv(f)` with
`pd.read_csv(f, float_precision="round_trip")` column by column.

**Suggested change.** Pass `float_precision="round_trip"` wherever 1.x
reads CSV (inputs, specs, coefficients, probability tables). Results
would change in the last bits once, on adoption.

---

## Not issues, but worth knowing

- **Legacy and sharrow runs give the same choices on MTC, but stored
  logsums differ by up to 7.5e-6** (sharrow computes in 32-bit). This sets
  the tolerance for comparing the two engines.
- **Sharrow's one-time compile of the MTC example took about 37 minutes**
  and 4.4 GB of memory on this machine.
