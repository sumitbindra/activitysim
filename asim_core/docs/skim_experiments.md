# Experiment: holding and looking up skims (full MTC zones)

The first experiment with the swappable pieces. On full MTC zones (1,475
zones, 826 skims, 7.2 GB; 10,000 households; see [mtc_full.md](mtc_full.md)),
skims are the largest part of asim_core's time in the steps that use them.
Two questions:

1. Which part costs the time: getting the skim values into the process
   (pages of the memory-mapped file), or the lookups themselves?
2. Does another way of holding or looking up skims make the steps faster,
   with exactly the same results?

## Summary

- **Every variant gives exactly 1.x's results** in all six steps tried (30
  combinations of step and variant; every compared column the same bit for
  bit, against 1.x with numexpr off).
- **Whether the skim file is in the operating system's page cache matters
  most.** Tour mode choice takes 0.87 s when it is (warm) and 6.1 s when it
  isn't (cold); 1.x takes 6.0 s with its skims already in memory. Mandatory
  scheduling: 4.4 s warm, 10.1 s cold, 1.x 16.1 s. In a whole run, the cost
  of a cold cache is paid once per page, not in every step: in one process
  the 8 ported steps take 18 s, against 71 s for 1.x (mtc_full.md).
- **Mapping pages up front (`prefault`) helps when cold, if the step reads
  most of the skims it declares:** tour mode choice 6.1 → 3.6 s, mandatory
  scheduling 10.1 → 8.0 s. It hurts the location and destination steps
  (for example school location 3.7 → 5.6 s): they declare 733 skims but read
  pages of only a quarter of them, and `prefault` reads them all. When warm
  it makes no difference.
- **Copying the skims into memory (`in_memory`, as 1.x holds them) is the
  slowest in every step**, warm or cold, and adds 6.5 GB of the process's own
  memory. Not worth pursuing.
- **A one-dimensional gather (`flat_take`) is 4 to 15% faster in the three
  destination choice steps** (school, workplace, non-mandatory destination:
  0.1 to 0.9 s per step, in every run), and makes no difference elsewhere.
  On its own, it is 30 to 40% faster than NumPy's two-dimensional indexing
  for random zone pairs.
- **What to do next** (end of this page): keep `memory_map` and make
  `flat_take` the default; declare only the skims a step can read, so that
  `prefault` can be on everywhere; work out the flat positions once per set
  of zone pairs instead of once per skim. The largest opportunity is the
  layout: tour mode choice reads 6 GB of pages to use 128 MB of values.

## What was compared

Two seams (swappable pieces), and how the skim file sits in the operating
system's page cache when the step starts:

| Seam | Implementation | What it does |
|---|---|---|
| `skim_access` | `memory_map` (reference) | look up straight from the memory-mapped file; the operating system brings each page in the first time a lookup touches it |
| | `prefault` | before the step, ask the operating system to map every page of the skims the step declares, in one call per run of adjacent skims (`madvise(MADV_POPULATE_READ)`) |
| | `in_memory` | before the step, read the skims it declares from the file into the process's own memory, and look up from the copies |
| `skim_lookup` | `numpy` (reference) | `matrix[orig, dest]` |
| | `flat_take` | `matrix.ravel().take(orig * zones + dest)` |

| Page cache | What happens before each run |
|---|---|
| cold | the skim file is dropped from the cache (`posix_fadvise(DONTNEED)`), as after a reboot or once other programs have pushed it out |
| warm | the skim file is read once, so all of it is in the cache, as when the model ran on the same machine just before |

The time to `prefault` or copy the skims is part of the step's time. Each
combination ran 3 times in a fresh process, from 1.x's saved state before
the step (`harness/run_experiments.py --page-cache cold warm`;
`harness/experiments_skims.yaml`). The share of the skim file in the page
cache when each run started was measured (`mincore`): 0.00 for every cold
run, 1.00 for every warm one.

## Time

Median wall time in seconds, 3 runs each. 1.x: the same step in a fresh
process with its skims already in memory (not timed: loading them).

| Step | Page cache | `memory_map` + `numpy` (reference) | `prefault` | `in_memory` | `flat_take` | `prefault` + `flat_take` | 1.x |
|---|---|---|---|---|---|---|---|
| compute_accessibility | cold | 0.76 | 0.83 | 0.82 | 0.73 | 0.68 | 1.30 |
| | warm | 0.71 | 0.80 | 0.92 | 0.69 | 0.69 | |
| school_location | cold | 3.71 | 5.62 | 8.06 | 3.26 | 5.15 | 7.32 |
| | warm | 2.15 | 2.25 | 9.20 | 2.06 | 2.09 | |
| workplace_location | cold | 6.93 | 9.15 | 11.02 | 6.45 | 8.58 | 18.71 |
| | warm | 6.08 | 6.02 | 12.61 | 5.17 | 5.21 | |
| mandatory_tour_scheduling | cold | 10.09 | 7.97 | 10.72 | 9.61 | 8.55 | 16.08 |
| | warm | 4.38 | 5.18 | 11.93 | 4.58 | 4.14 | |
| non_mandatory_tour_destination | cold | 6.42 | 8.60 | 10.70 | 5.52 | 7.82 | 18.30 |
| | warm | 5.30 | 5.36 | 12.56 | 4.49 | 4.73 | |
| tour_mode_choice_simulate | cold | 6.14 | 3.59 | 5.85 | 6.33 | 4.01 | 5.99 |
| | warm | 0.87 | 0.88 | 8.45 | 0.86 | 0.85 | |

Differences under about 0.5 s are within the spread of the runs, except for
`flat_take` in the three destination choice steps, where all three runs
were faster than the reference's fastest (for example non-mandatory
destination, warm: 4.43 to 4.60 s against 5.30 to 5.56 s).

## Memory

Median MB. Own: the process's own memory, its peak during the step above
where it started. Mapped: the most pages of mapped files it held (the skims,
plus about 65 MB of program libraries). Both sampled every 5 ms (the
profiler's `split_memory`). Warm runs; cold runs are within a few percent.

| Step | Reference: own | Reference: mapped | `prefault`: mapped | `in_memory`: own | 1.x: own, extra (holding 8,000 before) |
|---|---|---|---|---|---|
| compute_accessibility | 378 | 245 | 251 | 561 | 195 |
| school_location | 438 | 1,767 | 6,154 | 6,505 | 825 |
| workplace_location | 474 | 1,828 | 6,205 | 6,608 | 984 |
| mandatory_tour_scheduling | 483 | 6,147 | 6,155 | 6,567 | 517 |
| non_mandatory_tour_destination | 391 | 1,307 | 6,155 | 6,476 | 309 |
| tour_mode_choice_simulate | 59 | 6,054 | 6,154 | 6,156 | 46 |

`flat_take` changes neither. Mapped pages are the operating system's file
cache: shared between processes that read the same file, and dropped when
memory is needed (read again if used). The process's own memory is not.

## Why

**Reading pages costs more than looking up values.** Tour mode choice
declares 733 skims (every time period of every skim its formulas name). Its
32,020 tours use 43,569 distinct zone pairs (each way), 2% of all pairs:
0.17 MB of values in each 8.7 MB skim. But the pairs are spread over almost
every origin, so they touch 95% of each skim's 4 KB pages: the step reads
6 GB of pages to use 128 MB of values. From the page cache, mapping those
pages takes 0.14 s (`prefault`) to 0.26 s (touching them one by one), so a
warm step's time is the work itself. From the disk, it takes 2.5 s at best
on this machine:

| Bringing in the 6.4 GB tour mode choice declares, fresh process | Cold | Warm |
|---|---|---|
| touching one value per page, in order (kernel read-ahead helps) | 2.5 to 4.8 s | 0.26 s |
| `madvise(MADV_POPULATE_READ)` (`prefault`) | 2.5 to 2.6 s | 0.14 s |
| `posix_fadvise(WILLNEED)`, then `MADV_POPULATE_READ` | 2.5 s | |
| 4 or 8 threads reading the file, then `MADV_POPULATE_READ` | 1.9 to 2.1 s | |

During the step, lookups touch pages in no particular order, so the
reference pays more than this (6.1 − 0.87 = 5.3 s): `prefault` turns that
into one ordered read (3.6 − 0.88 = 2.7 s).

**`prefault` reads what a step declares, not what it reads.** The location
and destination steps also declare 733 skims, but take mode choice logsums
at fixed time periods (MTC's school tours: periods 8 and 14), so they read
pages of only 1.3 to 1.8 GB of them. Declared skims are worked out from the
formulas, which name a skim with a period variable (`skim.SOV_TIME[o, d,
period]`), so every period counts.

**`in_memory` copies 6.4 GB in every step.** Into fresh memory the operating
system has to supply page by page, while the file's pages also fill the page
cache: on this 15 GB machine the two together leave little room, and warm
runs are slower than cold ones. 1.x does the same copy once, before the
first step (34 to 36 s, from OMX).

**`flat_take`: NumPy's two-dimensional indexing does more work per value**
(it walks two index arrays and checks both) than a gather with one
precomputed position. Measured on its own, a 1,475 × 1,475 skim:

| Zone pairs | `matrix[orig, dest]` | `flat_take` | gather with positions worked out once |
|---|---|---|---|
| 50,000 | 0.58 ms | 0.33 ms | 0.17 ms |
| 800,000 | 6.22 ms | 4.48 ms | 3.81 ms |

The destination choice steps look up hundreds of skims for every chooser ×
sampled zone (up to 800,000 pairs), which is where the gain shows. Tour mode
choice looks up 32,020 tours: too few for it to show.

## What to do next

1. **Make `flat_take` the default** (exact, plain NumPy, up to 15% faster
   where lookups are large). Kept as an alternative until you decide.
2. **Declare only the skims a step can read**: when the periods come from
   settings (location and destination steps), only those periods. Then
   `prefault` reads only what is needed and can be on for every step (it
   costs nothing when warm).
3. **Work out positions once per set of zone pairs**, and reuse them for
   every skim with those pairs (1.x findings, #7 suggests the same for 1.x's
   time-period lookups): from the table above, another 15 to 50% of each
   lookup's time.
4. **Skim layout.** Values used together could be stored together: for
   example, by blocks of origin zones, or all skims of one period for each
   zone pair. Tour mode choice would read more than 15 times fewer pages
   from the disk; accessibility, which reads whole skims, would read more. A
   cache format change, so worth its own experiment.
5. **Whole-run profiles on full zones** alongside the per-step ones: a
   fresh process pays for reading skim pages in every step, a whole run once.

Memory needs no work here: asim_core's own memory stays under 540 MB in
every step; the mapped pages are the operating system's to manage.

## Reproducing

```bash
python asim_core/harness/run_experiments.py --config asim_core/harness/experiments_skims.yaml \
  --step tour_mode_choice_simulate --repeat 3 --page-cache cold warm --no-baselines \
  --example-dir asim_core/work/mtc_full/prototype_mtc_full \
  --asim1-out asim_core/work/mtc_full/asim1_10k --match-run asim_core/work/mtc_full/asim1_10k_no_numexpr \
  --read-skim-cache true --out asim_core/work/mtc_full/experiments/tour_mode_choice_simulate
```

The 1.x times come from `harness/profile_step.py --engine asim1`, 3 runs per
step, from the same saved state.
