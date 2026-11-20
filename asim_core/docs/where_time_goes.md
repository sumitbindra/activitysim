# Where the time goes, piece by piece (full MTC)

Step 2 of the plan after the skim experiment: before trying JAX, find which
pieces of the work dominate once the skims are in memory, since JAX can only
help with some of them.

## How it was measured

`harness/profile_pieces.py`: the 8 ported steps on full MTC (1,475 zones,
10,000 households) in one process, every other step replayed from 1.x's
saved run. The whole run is done twice in the same process; the second is
measured, so the skim pages are already mapped, as in any step of a whole run
after the first ones that read them. Each swappable piece is timed
(`asim_core/seam_timing.py`): its own time, without the pieces it calls (a
skim lookup made while a formula is evaluated counts as a skim lookup). Two
entries are split out of the pieces that consume them: **formulas**
(producing each formula's values) and **random** (drawing random numbers).
**Other** is everything outside the pieces: preprocessors, data movement,
timetables, padding. Default implementations (`flat_take` and so on).

## Results

All 8 steps together, 18.4 s:

| Piece | Seconds | Share | What it is |
|---|---|---|---|
| skim lookups | 4.35 | 24% | gathering skim values for zone pairs (`take`) |
| other | 4.00 | 22% | preprocessors, data movement, timetables, padding, ... |
| formulas | 3.42 | 19% | computing each formula's values |
| utilities | 3.01 | 16% | formula values × coefficients (`utilities` 1.48, `interaction_utilities` 1.53) |
| sampling | 1.86 | 10% | sampling destinations for location and destination choice |
| random numbers | 0.76 | 4% | |
| logsums and probabilities | 1.00 | 5% | nested logsums 0.57, probabilities 0.43 |
| choosing, table lookups | 0.07 | 0.4% | |

Per step (seconds, share of the step):

| Step | Wall | Skim lookups | Formulas | Utilities | Sampling | Random | Other |
|---|---|---|---|---|---|---|---|
| compute_accessibility | 0.56 | 0.25 (45%) | | | | | 0.31 (55%) |
| school_location | 2.12 | 0.48 (22%) | 0.26 (12%) | 0.36 (17%) | 0.35 (17%) | 0.08 (4%) | 0.45 (21%) |
| workplace_location | 5.35 | 1.21 (23%) | 0.80 (15%) | 0.95 (18%) | 0.74 (14%) | 0.17 (3%) | 1.13 (21%) |
| auto_ownership_simulate | 0.05 | | 0.01 (27%) | | | 0.03 (57%) | |
| mandatory_tour_scheduling | 3.60 | 0.83 (23%) | 1.11 (31%) | 0.62 (17%) | | 0.08 (2%) | 0.81 (23%) |
| non_mandatory_tour_destination | 4.80 | 1.20 (25%) | 0.62 (13%) | 0.83 (17%) | 0.77 (16%) | 0.19 (4%) | 0.84 (18%) |
| non_mandatory_tour_scheduling | 0.94 | | 0.46 (49%) | 0.22 (23%) | | 0.06 (7%) | 0.16 (17%) |
| tour_mode_choice_simulate | 1.06 | 0.39 (37%) | 0.16 (15%) | 0.03 (3%) | | 0.15 (14%) | 0.28 (27%) |

(Logsums, probabilities and choosing are left out of this table: 2 to 4%
each at most.)

## What it says

Python's profiler on two steps (workplace location, mandatory scheduling)
shows what is inside the largest pieces:

- **Skim lookups are memory reads.** NumPy's `take` alone is 1.2 s of
  workplace location and 1.0 s of mandatory scheduling: values gathered from
  8.7 MB matrices at scattered positions, so the time is waiting for memory.
  A compiler (JAX) can't make that faster. Fewer reads can: looking up each
  distinct zone pair once, or the skim layout (skim experiment, item 4).
- **Sampling compares every draw with every zone.** It picks, for each of 30
  draws per chooser, the first zone whose running sum of probabilities
  exceeds the draw, by comparing the draw with all 1,475 running sums
  (workplace location: about 600 million comparisons). A binary search per
  draw (`np.searchsorted` row by row) finds the same zone with about 11
  comparisons: identical picks, a fraction of the time. A NumPy change, not
  a JAX one.
- **Formulas and utilities together are 35%** (6.4 s): chains of
  element-wise arithmetic on arrays of 100,000s of rows, each step writing a
  temporary array, then multiplied by coefficients and added up. This is
  what JAX's compiler (XLA) does well on a CPU: it fuses a chain of
  element-wise operations and the sum into one pass over memory, on several
  threads. It is the largest share in the scheduling steps (48% and 72%)
  and about a third in the destination steps.
- **Other is spread out**: preprocessors (formulas outside the pieces),
  sorting and grouping in scheduling (`argsort`, `unique`), the sampling
  step's bookkeeping, timetables. No single item is large.

## For the JAX experiment

The target that fits is **formulas + utilities** (35%), in the destination
and scheduling steps. What it needs and what to expect:

- **Skims stay outside JAX.** Copying the skims into JAX arrays would cost
  6 GB (like the `in_memory` option, the slowest in the skim experiment).
  The formulas' skim lookups would be done first, in NumPy (asim_core knows
  each formula's skims from the formula language), and their values passed
  in.
- **Compiling takes time.** JAX compiles a function for each set of formulas
  and each array shape. Batches would be padded to a few fixed sizes so that
  a compile is reused. At 10,000 households the steps take 1 to 5 s, and
  compiling may cost as much as it saves; the experiment should report
  compile time separately, and also run with more households.
- **Results would be close, not identical.** XLA can add up in a different
  order and fuse multiplications and additions (as numexpr does in 1.x,
  1.x findings, #3). The report would show the largest utility difference
  and whether any choice changes ("approximate" agreement).
- **Strings stay outside JAX** (tour types and similar): formulas compare
  them in NumPy, or they are turned into codes first.

The two NumPy changes above (sampling by binary search, and looking up each
distinct zone pair once) don't need JAX and keep results identical.
