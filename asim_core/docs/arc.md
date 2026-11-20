# Third region: ARC, and nested logit

ARC's auto ownership model is a nested logit, the smallest one among the
example models. It was used to add nested logit to asim_core and check it
against 1.x. The two ported steps were also run on ARC
(`activitysim/examples/prototype_arc`) as a third region.

## Summary

- **Nested logit matches 1.x exactly.**
  - All 24,061 ARC households make the same car-ownership choice.
  - All 96,244 utilities and 96,244 probabilities are bit-for-bit
    identical to 1.x's own nested logit functions.
  - It also matches 1.x's functions bit for bit on 2,000 rows of random
    utilities with a deeper, three-level tree (a unit test).
- **`compute_accessibility` matches 1.x exactly on ARC,** and the
  end-to-end run (2 steps computed, 32 replayed) matches at every check.
- **ARC keeps its original zone numbers** (100, 101, ...), so this also
  tested the zone-lookup rule added for MWCOG. It worked unchanged.
- **1.x ran out of memory on all ARC households** (15 GB machine) in
  `mandatory_tour_scheduling`, a step asim_core hasn't ported yet. Before
  the run I estimated 2–3 GB, by scaling MTC's memory by household count.
  That was wrong. Scheduling memory grows with tours × time-window
  alternatives, and ARC has 6 times as many alternatives per tour as MTC.
  Details below.

## Nested logit in asim_core

- **Model definitions take the `NESTS` tree** from today's settings files.
  Nest coefficients can be numbers or coefficient names. Loading checks that
  every alternative appears in exactly one place, that names are unique,
  and that named coefficients exist.
- **`nested_probabilities` is a new swappable piece.** Its reference version
  follows 1.x's `compute_nested_exp_utilities`,
  `compute_nested_probabilities` and `compute_base_probabilities` operation
  by operation:
  1. each alternative's utility is divided by the product of its nests'
     coefficients and exponentiated;
  2. each nest's value, from the leaves up, is its coefficient × log(sum of
     its children), exponentiated;
  3. each child's probability within its nest is its share of the nest's
     sum, with 1.x's handling of zeros and missing values;
  4. each alternative's probability is the product of the conditional
     probabilities on its path, multiplied from the top down.
- **Details that mattered for exact matching:** nest coefficients stay
  64-bit (only utility coefficients are rounded to 32-bit in 1.x); sums
  within a nest add children left to right; the tree is walked in 1.x's
  order; there's no overflow protection.

## Results

| Check | Compared | Result |
|---|---|---|
| `auto_ownership` (nested logit), all households | 24,061 households | exact |
| ... utilities and probabilities | 96,244 each | bit-for-bit identical |
| `compute_accessibility` | 10 columns × 36 zones | exact |
| Same checks, 100-household run | 100 households, 36 zones | exact |
| End to end, 100 households: 2 computed, 32 replayed | after `cdap_simulate`, after `joint_tour_frequency`, at the end | exact |
| Nested logit vs 1.x functions, 3-level tree, random utilities | 2,000 rows × 7 alternatives | bit-for-bit identical |

Car ownership shares, all households (identical in both): 0 cars 3.03%,
1 car 29.32%, 2 cars 49.19%, 3 or more 18.46%.

## Memory: 1.x's tour scheduling on ARC

The run on all 24,061 households was killed by the operating system in
`mandatory_tour_scheduling`, with about 14 GB in use. ARC uses 30-minute
time periods, so each tour has 1,176 departure–duration alternatives, while
MTC's hourly setup has 189. 1.x holds a table of tours × alternatives in
memory, and ARC turns chunking off (`chunk_training_mode: disabled`).

1.x runs that stop after that step, at four sample sizes:

| Households | Mandatory tours | Scheduling time (s) | Extra memory in scheduling (MB) | Peak memory (MB) |
|---|---|---|---|---|
| 1,000 | 1,393 | 13.1 | 638 | 1,134 |
| 2,000 | 2,876 | 23.0 | 1,271 | 1,821 |
| 4,000 | 5,601 | 43.5 | 2,324 | 2,930 |
| 8,000 | 11,092 | 90.7 | 4,607 | 5,236 |

Memory grows linearly, by about 0.41 MB per mandatory tour. For all
households (about 33,400 tours) that predicts about 13.7 GB of extra
memory, which matches the run that was stopped. Time also grows linearly,
by about 8 ms per tour.

This step is a strong candidate for an early asim_core experiment. On a
real region, chunking keeps 1.x within memory but costs time. A
batch-by-batch design that never builds the full table is what the core is
designed to test.

## Profiles

100-household run; each step in a fresh process from 1.x's saved state, 5
runs, medians. Skims prepared with ARC's own settings (`read_skim_cache:
False`, 845 matrices). Sharrow was not run.

| Step | Engine | Step (s) | Extra memory (MB) | Skims ready (s) | Inside the full 1.x run (s) |
|---|---|---|---|---|---|
| compute_accessibility | asim_core | 0.005 | 1.6 | 0.34 | |
| compute_accessibility | 1.x legacy | 0.089 | 3.7 | 0.44 | 0.41 |
| auto_ownership (nested logit) | asim_core | 0.004 | 1.3 | 0.35 | |
| auto_ownership (nested logit) | 1.x legacy | 0.113 | 4.6 | 0.42 | 0.29 |

The full 1.x run of 100 households took 60 seconds, with a peak of 559 MB.

## Harness changes

- **Profiles are written step by step,** so a run that's killed (as here)
  keeps the profile of every step it finished.
- **`run_asim1.py --stop-after STEP`** runs 1.x only up to a given step, as
  in the scaling runs above.
- **The 1.x reference script also asks 1.x for nested logit
  probabilities,** for the utility-level check.

## Reproducing

From the repository root:

```bash
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/prototype_arc --out asim_core/work/arc/asim1_legacy
```

```bash
python asim_core/harness/run_milestone1.py --example-dir activitysim/examples/prototype_arc --asim1-out asim_core/work/arc/asim1_legacy --asim1-sharrow-out asim_core/work/arc/no_sharrow_run --out asim_core/work/arc/milestone1
```

Memory scaling (one size shown):

```bash
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/prototype_arc --households 4000 --stop-after mandatory_tour_scheduling --out asim_core/work/arc/scaling_4000
```

Avoid `--households 0` (all households) for ARC on a machine with less than
about 16 GB of memory.
