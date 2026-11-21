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
| 3 | Results depend on how many rows are evaluated at once (numexpr above 1M elements; padding width per chunk) | Reproducibility | Measured; one-line fix tested | Not reported |
| 4 | Tour mode choice's choice draw reuses random numbers already used | Random numbers | Measured | Not reported |
| 5 | Building every tour × time option at once uses gigabytes | Memory | Measured | Not reported |
| 6 | Numba functions are compiled again in every run | Speed | Measured | Not reported |
| 7 | Time-of-day skim lookups are a third of tour mode choice's time | Speed | Measured | Not reported |
| 8 | Tours of a type with no spec segment are silently left unscheduled | Bug (configuration) | From reading the code | Not reported |
| 9 | Smaller items (7) | Various | From reading the code, one measured | Not reported |
| 10 | Bad input data runs silently: negative employment changed results across the model with no error or warning | Data checking | Reproduced (MTC) | Not reported |
| 11 | Numbers in CSV files are read inexactly (pandas' default parser): up to 1e-12 relative error, in inputs and in coefficients | Accuracy | Measured (all three examples) | Not reported |
| 12 | ARC's destination terms for CBD and urban area read the person's home zone, not the destination, so they never change a choice | Model specification | Reproduced (ARC, all households) | Not reported |
| 13 | With zones renumbered, the skims' zone lookup is not used or checked: a lookup in another order gives every zone another zone's skims, silently | Zone numbering | From reading the code | Not reported |
| 14 | 8-bit integer formulas overflow when evaluated without numexpr (7 tours × start 22 = −102), changing one person's schedule in 50,000 households | Arithmetic | Reproduced (MTC full zones, 50,000 households) | Not reported |
| 15 | CDAP ranks children by person type, not youngest first as its comments and CTRAMP say: in larger households, preschoolers are left out of the joint choice | Model behavior | Measured (MTC full zones, 50,000 households: 616 preschoolers) | Not reported |
| 16 | Joint tour and non-mandatory frequency specs test tour types that don't exist ('eat', 'disc', 'maint' for eatout, othdiscr, othmaint): 13 terms with coefficients never apply | Model specification | Measured (MTC full zones, 50,000 households: 1.7% of joint tours change party type when corrected) | Not reported |
| 17 | MTC's at-work subtour scheduling spec: four 'eating out' terms test business subtours, and the '2 to 3 hours' term covers 1 to 4 hours | Model specification | Measured (MTC full zones, 50,000 households: 70% of at-work subtours get another schedule when the eating-out terms are corrected) | Not reported |

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

**Suggested change.** Turn off numexpr for pandas' arithmetic only
(`pandas.core.computation.expressions.set_use_numexpr(False)`), so Series
arithmetic follows NumPy's rules at any size, while `pandas.eval` keeps
numexpr. Careful: pandas ties the two together (with numexpr off for
arithmetic, `pandas.eval` switches to plain pandas too, and the
`compute.use_numexpr` option does both), and plain pandas lets 8-bit
integer formulas overflow (#14); the engine for `pandas.eval` has to be kept
at numexpr explicitly (`harness/asim1_patches.py`, `no_numexpr`). Possible
cost: numexpr is multi-threaded, so very large arithmetic may be slower; not
measured.

**Seen on full-size zones.** MTC's full zone system (1,454 zones, 10,000
sampled households, 1.x run to tour mode choice): arrays pass 1M elements
everywhere (accessibility evaluates 1,454 × 1,454 zone pairs at once;
location and destination sampling evaluate choosers × 1,454 zones). 1.x's
accessibility differs in all 1,454 zones (up to 6e-7) from the same run
with numexpr off (first tested with `pd.set_option("compute.use_numexpr",
False)`, which has the side effect described in #14), and the location and
destination choice logsums of 4,749 students, 14,134 workers and 14,670
tours differ (about 1e-7). No choice changed. asim_core, which uses NumPy's
rules at any size, matches the numexpr-off run exactly (all 279 compared
columns); against the default run, those 36 columns differ. On the small
example data nothing reaches 1M elements, so this never shows there.

**A second way chunking changes results: the padding width.** When the
alternatives differ per chooser (sampled destinations, a tour's departure
options), `interaction_sample_simulate` pads every chooser's utilities to
the widest chooser *in the chunk* with -999 before adding up the
exponentials (`max_sample_count`). The padding adds zeros, but NumPy adds
each row in pairs, so the row's length changes how the sum is rounded.
With a different chunk size, a chooser gets a different width, and its
logsum and probabilities can change in the last bits.

**Measured** with asim_core, which reproduces 1.x's arithmetic exactly:
padding each batch of 3,000 rows to its own widest chooser, instead of the
whole segment's, changed 12 of 8,212 MTC school location logsums (by up to
3.6e-15). asim_core now pads to the widest chooser of the segment (1.x's
unchunked call), so its results don't depend on its batch size.

**Suggested change.** Pad to a width that doesn't depend on the chunk:
the sample size for destination choice, the number of alternatives for
scheduling.

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
- **At-work subtour mode choice loses `household_id`.** It merges the
  subtours with `persons_merged` with pandas' default suffixes, so a column
  both have (`household_id`) becomes `household_id_x` and `household_id_y`;
  tour mode choice merges the same tables with `suffixes=("", "_r")` and
  keeps the tour's. Harmless today (the spec doesn't use it), but the same
  spec can read a column in one step and fail in the other.
- **Every run's log has an ERROR that isn't one.** Making at-work subtours'
  IDs logs `ERROR - parent_tour_num.dtype: int8` (`canonical_ids.py`)
  whenever the parent tour number isn't 64-bit, which it never is, then
  converts it and carries on. A real error is easy to miss among them.

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

## 12. ARC's CBD and urban destination terms never change a choice

**Evidence: reproduced** (ARC, all 24,061 households, 30,851 non-mandatory tours).

**What happens.** `prototype_arc/configs/non_mandatory_tour_destination.csv`
has two terms meant to describe the candidate destination:

```
CBD area type,@df['is_CBD'],0,0,0,coef_shopping_cbd,coef_eatout_cbd,coef_othmaint_cbd,coef_social_cbd,0
Urban high area type,@df['is_urban'],0,0,0,coef_shopping_urban,...
```

The rows this spec sees hold the sampled zone (`alt_dest`), its sampling
probability and logsum, plus the tour's own columns: the tour, the person,
the household and the *home* zone. Land use columns of the candidate zone
are not there. `is_CBD` and `is_urban` exist on the persons table
(`annotate_persons_after_hh.csv`: the household's home zone area type), so
`df['is_CBD']` silently reads the person's home zone. That value is the
same for every candidate zone of a tour, so it adds the same amount to
every alternative's utility and cancels out of the choice.

Setting the nine CBD and urban coefficients to zero (asim_core, ARC
saved state of all households): **every tour gets the same destination**;
only the destination logsums of 7,056 tours change (by up to 0.714, the
urban coefficient for social tours). The coefficients (e.g. −0.57 for
shopping in the CBD) suggest a destination effect was intended.

Found while converting the models to asim_core's model files, where every
column is written with its table: the converted term reads
`person.is_CBD`, which makes the problem visible.

**To reproduce:** run non-mandatory tour destination twice on the same
saved state, once with those coefficients set to 0, and compare the
destinations (or, in 1.x, add a column test: the term's value is constant
within each tour).

**Suggested change.** In the example: use the destination zone's land use
(e.g. a size-term-like column joined to the alternatives, or a skim-based
measure). In 1.x: warn when an interaction term doesn't vary across a
chooser's alternatives, and/or make the alternatives' columns and the
chooser's columns distinguishable in formulas.

---

## 13. With zones renumbered, the skims' zone lookup is not used or checked

**Evidence: from reading the code** (not reproduced; none of the three
examples has a lookup out of order).

**What happens.** With `recode_pipeline_columns: True` and
`recode_columns: {zone_id: zero-based}` (as in MTC), 1.x renumbers the land
use zones 0, 1, 2, ... in ascending order of their original numbers
(`core/input.py`) and sets `offset_preprocessing`. The skim lookup then uses
a fixed offset of 0 (`core/los.py`, `_override_offset_int = 0`), and
`OffsetMapper.set_offset_int` drops any zone list read from the OMX file's
lookup (`core/skim_dictionary.py`). So zone i is always read from skim row
i. If the OMX file's lookup lists zones in another order (or with zones
missing), every zone reads another zone's skims, with no error: the only
check compares the number of zones (`los.py`, and only for `land_use_taz`).

Without renumbering the lookup is used, in any order. Without renumbering
and without a lookup, zone z is read from row z − 1 ("assume this is a
one-based skim map").

**Found while** checking that asim_core finds skim rows the way 1.x does
(asim_core had the no-lookup case off by one; fixed). asim_core now checks
before the run: with renumbered zones and a lookup, the lookup must list
the zones in ascending order (error otherwise, naming the zones affected);
without a lookup, original zone numbers must be within 1 to the number of
skim zones (error), and renumbered zones with gaps get a warning that the
row order can't be checked.

**To reproduce:** reorder the rows and columns of a small example's skims
and their lookup consistently (the same skims, listed in another order),
then run with renumbering on: results change.

**Suggested change.** When renumbering, map the original zone numbers
through the OMX lookup if there is one (or at least check that it matches
the renumbering, and stop if not).

---

## 14. 8-bit integer formulas can overflow, depending on how they are evaluated

**Evidence: reproduced** (MTC full zones, 50,000 households, one person).

**What happens.** Some columns are 8-bit integers (values up to 127), such
as a person's count of non-mandatory tours and a tour option's start hour.
MTC's non-mandatory scheduling has the plain-style formula
`num_non_escort_tours * start`. 1.x evaluates plain-style formulas with
`pandas.eval`; with numexpr (installed with 1.x, through sharrow) the
integers are widened first, so 7 tours × start 22 = 154. Without numexpr
(not installed, or turned off with pandas' `compute.use_numexpr` option) the
product stays 8-bit and wraps around: 7 × 22 = −102. One person in 50,000
households has 7 non-mandatory tours; the term's coefficient (0.039) then
adds −4.0 instead of +6.0 to the late options of their first tour, which
changes its departure and, through the timetable, their six later tours.

Formulas written with `@` are evaluated with NumPy, which never widens
8-bit integers: the same product written as `@df.num_non_escort_tours *
df.start` would overflow in every 1.x run. (Not checked whether any of the
examples' `@` formulas do.)

**Found while** running asim_core against 1.x with numexpr turned off for
arithmetic (#3): the first version of that patch used pandas' option, which
also switched `pandas.eval` off numexpr. asim_core matches 1.x as installed
(154).

**To reproduce:** `pd.eval("a * b")` with `a`, `b` 8-bit Series of 7 and
22, with and without `pd.set_option("compute.use_numexpr", False)`.

**Suggested change.** Widen 8-bit integer columns before formulas use them
(or keep counts and hours in at least 16 bits), so results don't depend on
the evaluation engine.

---

## 15. CDAP ranks children by person type, not youngest first

**Evidence: measured** (MTC full zones; asim_core's rank code, shown
identical to 1.x's, with the run's own random draws).

**What happens.** CDAP chooses the day patterns of at most five members of
a household jointly; others get a pattern drawn from fixed proportions by
person type, without regard to the household. `assign_cdap_rank`
(`abm/models/util/cdap.py`) decides who the five are. Its comments, quoting
CTRAMP's `reOrderPersonsForCdap`, say: up to two workers, "preferring full
over part, older over younger", then up to three children, "preferring
youngest". The code sorts workers and children by person type only:

```python
children = persons.loc[persons[_ptype_].isin(person_type_map["CHILD"]), [_hh_id_, _ptype_, _age_]]
    .sort_values(by=[_hh_id_, _ptype_], ascending=[True, True]).groupby(_hh_id_).head(3)
```

With MTC's person types (6 driving-age child, 7 school child, 8 preschool)
ascending order puts driving-age children first and preschoolers last: the
opposite of youngest first. Age is selected but not used. Workers of the
same type are taken in table order, not by age.

**How much.** In households of six or more:

| MTC full zones | Persons outside the joint choice | of them children | of them preschoolers | Households where the 3 children differ from youngest-first |
|---|---|---|---|---|
| 5,000 households (example data) | 315 | 54 | 42 | 40 |
| 10,000 households | 1,310 | 222 | 119 | 202 |
| 50,000 households | 6,445 | 1,105 | 616 | 938 |

A preschooler's pattern depends strongly on the household (someone has to
stay home with them, CDAP's interaction terms model exactly that); drawn
from fixed proportions it is not coordinated with anyone.

**Found while** porting CDAP to asim_core (it reproduces 1.x's ranks
exactly, as it must to match 1.x).

**Suggested change.** Either sort children by age (and workers by person
type, then age descending), as documented, or change the comments to say
what the code does. A model change: results change for the affected
households.

---

## 16. Specs test tour types that don't exist ('eat', 'disc', 'maint')

**Evidence: measured** (MTC full zones, 50,000 households; the same steps
run in asim_core from 1.x's saved state, with the specs as shipped and
corrected, the same random numbers).

**What happens.** The joint tour types in all three examples are
`shopping`, `othmaint`, `eatout`, `social` and `othdiscr`
(`joint_tour_frequency_alternatives.csv`). But:

- `joint_tour_composition.csv`: `tour_type=='eat'` and `tour_type=='disc'`
  (4 coefficients, e.g. eating out −0.97 for children-only parties,
  discretionary +0.76 for adults-only);
- `joint_tour_participation_annotate_participants_preprocessor.csv`:
  `tour_type_is_eat` and `tour_type_is_disc` the same way, used in 10
  participation terms (each person type on eating out and discretionary
  tours);
- `non_mandatory_tour_frequency_annotate_persons_preprocessor.csv`:
  `num_hh_joint_maint_tours` counts tours of type `'maint'`, so it is
  always 0, though the spec gives it −1.35 for school-age children.

None of these terms can ever apply, and nothing reports it: comparing a
column with a value it never takes is valid. The same files are in MTC,
MWCOG and ARC.

**How much** (joint tours, as shipped against corrected):

| Joint tours | Tours | Children-only parties | Mixed parties |
|---|---|---|---|
| discretionary (othdiscr) | 979 | 12.2% → 16.1% | 54.3% → 49.6% |
| eating out | 446 | 4.0% → 3.8% | 36.1% → 35.4% |

53 of 3,068 joint tours (1.7%) get another party type; 8,092 participants
become 8,066.

**Found while** porting these steps to asim_core (which reproduces 1.x's
results, so it reproduces this too).

**Suggested change.** Use the tour type names of the alternatives files;
and a check, when a model loads, that each value a spec compares a category
column with is one of the column's categories (asim_core could do this in
its input checks).

---

## 17. At-work subtour scheduling: 'eating out' terms test business subtours (MTC)

**Evidence: measured** (MTC full zones, 50,000 households; asim_core's
`atwork_subtour_scheduling`, which matches 1.x exactly, run from 1.x's
saved state with the spec as shipped and corrected, the same random
numbers).

**What happens.** In MTC's `tour_scheduling_atwork.csv` (the at-work
subtour types are `eat`, `business` and `maint`):

- four terms labelled for eating out test business subtours:
  `util_dummy_for_eating_out_purpose_and_duration_of_1_hour`,
  `..._departure_at_11`, `..._departure_at_12` and `..._departure_at_13`
  are all `(tour_type == 'business') & ...`, with coefficients 0.40, 1.51,
  2.72 and 2.12 (the row before them, labelled business, also tests
  business);
- `util_duration_of_2_to_3_hours` is `(duration >= 1) and (duration <= 4)`,
  so it also applies to 1-hour subtours, whose own term is the reference
  (coefficient fixed at 0), and to 4-hour ones, which the 4-to-5-hours term
  covers too.

MWCOG's and ARC's at-work scheduling specs are different files and test
`'eat'` for their eating-out terms.

**How much** (15,100 at-work subtours):

| Corrected | Subtours with another schedule | Effect |
|---|---|---|
| eating-out terms test `'eat'` | 10,648 (70%) | eat subtours departing in hour 11, 12 or 13: 53% → 91%; business subtours: 85% → 46% |
| 2-to-3-hours term tests 2 to 3 hours | 8,240 (55%) | subtours returning in the hour they left: 53% → 33%; one hour later: 21% → 49% |

As shipped, business subtours get the lunchtime departures and eat
subtours don't.

**Found while** porting these steps to asim_core (which reproduces 1.x's
results, so it reproduces this too).

**Suggested change.** `tour_type == 'eat'` in the four eating-out terms and
`(duration >= 2) and (duration <= 3)` in the 2-to-3-hours term, after
checking against the model's estimation (the labels and coefficients
suggest these were meant; the CTRAMP original wasn't available here).

---

## Not issues, but worth knowing

- **Legacy and sharrow runs give the same choices on MTC, but stored
  logsums differ by up to 7.5e-6** (sharrow computes in 32-bit). This sets
  the tolerance for comparing the two engines.
- **Sharrow's one-time compile of the MTC example took about 37 minutes**
  and 4.4 GB of memory on this machine.
