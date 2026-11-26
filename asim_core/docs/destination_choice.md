# Non-mandatory tour destination, and sampling of alternatives

Destination and location choices have too many alternatives (every zone)
to evaluate the full model for each one. 1.x first **samples** a few
dozen zones per tour with a simple model, then runs the full model,
including mode choice logsums, on the sample only. Sampling is used by
four 1.x steps: workplace location, school location, non-mandatory tour
destination and trip destination. Non-mandatory tour destination was
ported first because it has no shadow pricing (unlike workplace and
school location) and its logsums come from the tour mode choice model
asim_core already has.

## Summary

- **asim_core matches 1.x exactly on all three regions:** every tour gets
  the same destination, and every destination choice logsum is bit-for-bit
  identical:

  | Region | 1.x run used | Tours | Same destination | Destination logsums bit-identical |
  |---|---|---|---|---|
  | MTC | full example | 4,521 | 4,521 | 4,521 |
  | MWCOG | 10,000 households | 6,998 | 6,998 | 6,998 |
  | ARC | 4,000 households | 5,173 | 5,173 | 5,173 |
  | ARC | all 24,061 households | 30,851 | 30,851 | 30,851 |

- **Sampling is a swappable piece (`sample`).** Its reference is 1.x's
  default sampler (inverse CDF), vectorized with NumPy and checked against
  1.x's Numba sampler draw by draw.
- **The tour mode choice model was reused a second time, with one small
  change:** its origin column is now a setting too (MWCOG's logsums start
  from `TAZ`). The model's interface otherwise held: bind to some chooser
  columns and an origin and destination, then evaluate rows.
- **About 5 to 11 times faster than 1.x, with less memory.** All ARC
  households, in a fresh process: 1.9 s and +340 MB against 10.6 s and
  +650 MB; MTC: 0.37 s against 4.2 s (legacy) and 3.2 s (sharrow).

## How 1.x does it, step by step

For each segment (tour type: shopping, othmaint, ...):

1. **Size terms.** Each zone's size is a weighted sum of land use columns
   (`destination_choice_size_terms.csv`), computed as 1.x does (pandas
   `.dot`). Zones with size zero are left out.
2. **Sample.** The sample spec gives every tour × zone a utility (distance
   bands, `log(1 + size)`, ...); probabilities follow; each tour gets
   `SAMPLE_SIZE` random draws, and each draw picks the first zone whose
   running sum of probabilities exceeds it. A zone drawn several times is
   kept once with its `pick_count`.
3. **Logsums.** Tour mode choice gives every sampled zone a logsum, at the
   segment's fixed departure and return periods (`OUT_PERIOD`,
   `IN_PERIOD`). As in 1.x, the mode choice model sees the sampled rows and
   the person's columns, not the tour's (so, for instance, its preprocessor
   takes its "not a tour" branches).
4. **Choice.** The full spec scores each sampled zone, with the logsum and
   the sampling correction `log(pick_count / prob)`; one zone is chosen,
   and the logsum over the sample is kept as `destination_logsum`.

Each tour's random stream is used in that order: the sample draws, the
mode choice preprocessor's draws (once per tour), then the choice.

## Details that mattered for exact matching

Found by comparing each stage with 1.x's own intermediate results (the
first run matched the sampling probabilities and every logsum bit for bit,
but sampled different zones):

- **1.x caps the number of draws at the number of zones that can be
  chosen:** `min(SAMPLE_SIZE, zones with size > 0)`. MTC's example has 25
  zones, so its tours get fewer than 30 draws. This also changes where each
  tour's random stream is when the logsum and choice draws come.
- **1.x returns the sample sorted by zone within each tour,** and the
  choice step sees the zones in that order.
- **Probabilities are stored as 32-bit floats** in the sample, and pick
  counts as 32-bit unsigned integers; the correction term
  `log(pick_count / prob)` is computed from those.

## What was added to asim_core

- `sample`, a new swappable piece (`logit.py`).
- Interaction specs with one coefficient column per segment (1.x's
  `spec_for_segment`), with skim lookups (`skims['DIST']`,
  `skims[('SOV_FREE_DISTANCE', 'MD')]`).
- `.apply(np.log1p)` (and any listed NumPy function) in formulas.
- Logsums from the multinomial logit probabilities piece (1.x's
  `utils_to_probs(..., return_logsums=True)`).
- `TourModeChoiceModel.bind(..., orig=...)`.
- Two views of the choosers: tours with their person, household and home
  zone (for the sample and the choice), and persons with their household
  and home zone (for the logsums, as 1.x's `persons_merged`).

## Profiles

Each step run in a fresh process, starting from 1.x's saved state before
it, the same way for every engine (`harness/profile_step.py`). Not timed:
loading the saved state, getting skims ready, and 1.x's one-time Numba
compiles (its sampler is compiled before timing; Numba's compile events
confirm nothing compiles inside the timed step). Wall time is the median;
extra memory is the step's peak above where it started.

**MTC, 4,521 tours, 5 runs each:**

| | Wall time (median, min–max) | Extra memory | Same step in the full 1.x run |
|---|---|---|---|
| asim_core | 0.37 s (0.33–0.39) | 81 MB | |
| 1.x, legacy | 4.22 s (4.08–4.41) | 115 MB | 5.67 s |
| 1.x, sharrow | 3.17 s (2.89–95.7) | 74 MB | 2.44 s |

One of the five sharrow runs took 95.7 s and +1,061 MB: sharrow compiled
code it didn't find in its cache during that run. The median is the
fairer figure.

**ARC, 3 runs each:**

| | 4,000 households (5,173 tours) | All households (30,851 tours) |
|---|---|---|
| asim_core | 0.38 s (0.33–0.41), +80 MB | 1.90 s (1.80–1.98), +340 MB |
| 1.x, legacy | 4.06 s (3.78–4.11), +135 MB | 10.61 s (10.27–10.81), +650 MB |

## Not supported yet

Shadow pricing (workplace and school location), two- and three-zone
systems (presampling of zones), sampling methods other than 1.x's default
(`eet`, `poisson`), preprocessors in destination choice, saving the
sample table, and estimation.
