# Mandatory tour scheduling: finding and testing an improvement to 1.x

Before porting mandatory tour scheduling, its time and memory were measured
inside ActivitySim 1.x, part by part. That led to an improvement to 1.x
itself, tested with the harness: same results, much less time and memory.

## Summary

- **On ARC, 74% of mandatory tour scheduling's time and most of its memory
  is the mode choice logsums,** not the scheduling choice itself.
- **1.x computes about 13 times more logsums than ARC's formulas need.**
  Without a segments file, it computes one per tour × (departure period,
  return period, duration): up to 273 per tour. ARC's mode choice formulas
  never read duration, so only the 21 (departure period, return period)
  pairs give different results.
- **Computing each distinct logsum once gives identical results.** That's
  tested at three levels: all 1,585,229 logsums bit for bit; every
  scheduling choice; and complete 4,000-household runs, every table and
  every column.
- **Gains on ARC at 4,000 households:**
  - mandatory tour scheduling 3.4 times faster (44.8 s → 13.0 s), with 3.5
    times less extra memory (2,322 MB → 660 MB);
  - the whole run 25% faster (137 s → 102 s), with half the peak memory
    (2,929 MB → 1,480 MB).
- The change is applied by the harness at run time
  (`run_asim1.py --patch logsum_dedupe`). 1.x's own source is unchanged.

- **With the patch, 1.x runs all 24,061 ARC households on this 15 GB
  machine:** all 34 steps in 347 seconds, with a peak of 5.8 GB. Without
  it, the run was stopped for running out of memory at about 14 GB in
  mandatory tour scheduling. There's no unpatched all-households run to
  compare with, so the evidence that results are unchanged is the
  4,000-household comparison above.

## Where the time and memory go

1.x's `mandatory_tour_scheduling` handles tours in batches (first tours,
then second tours, by purpose). For each batch it:

1. builds a table of every tour × every departure–duration option still
   open in that person's day (1,176 options on ARC, with 30-minute periods);
2. computes **tour mode choice logsums** for those options
   (`LOGSUM_SETTINGS: tour_mode_choice.yaml`);
3. joins tour and person columns onto every option, evaluates the
   scheduling formulas and chooses;
4. records the choice in the person's timetable.

Measured inside a 1.x run on 4,000 ARC households (5,601 mandatory tours):

| Part | Time | Largest memory jump |
|---|---|---|
| 1. Build the tour × option table | 2.8 s | 232 MB |
| 2. Mode choice logsums | 33.3 s (74%) | 2,254 MB |
| 3. Join, formulas and choice | 4.8 s | 493 MB |

## Why so many logsums

A mode choice logsum depends on a tour option only through the columns the
mode choice formulas read. 1.x passes three of the option's columns to the
logsum calculation: departure period, return period and duration. Exact
start and end times aren't passed. So ARC's preprocessor rows like
`(df.start > 26) if 'start' in df.columns else (df.out_period.isin([...]))`
take their period-based branch.

| | Per tour, ARC |
|---|---|
| Departure–duration options | 1,176 |
| Distinct (departure period, return period, duration): what 1.x computes | 273 |
| Distinct (departure period, return period): what the result depends on | 21 |

MTC is different: its mode choice uses duration, for all-day parking cost.
The patch leaves that case alone.

1.x has a related setting, a `tour_departure_and_duration_segments.csv`
file. MTC and MWCOG ship one; ARC doesn't. It isn't a substitute here. With
segments, 1.x passes a representative start and end time into the logsum
calculation, so ARC's preprocessor would take its exact-time branch and the
logsums, and choices, would change.

## The patch

`harness/asim1_patches.py`, `logsum_dedupe`:

- **It applies only when** no segments file is given, and no file of the
  logsum model mentions `duration`. That's a plain-text search of the
  settings, spec, coefficients, template and preprocessor files, so any
  doubt means the patch steps aside.
- **When it applies,** it computes one logsum per (tour, departure period,
  return period) and joins it back to every option with that pair, using
  1.x's existing join.

## How it was checked

| Check | Compared | Result |
|---|---|---|
| Logsums, both ways in the same run (1,000 households) | 1,585,229 | bit-for-bit identical |
| Mandatory scheduling choices (4,000 households) | 5,601 tours: start, end, duration, option | identical |
| Complete runs (4,000 households), `compare_asim1_runs.py` | 13 tables, every column | identical |

## Profiles, complete ARC runs, 4,000 households

| | 1.x | 1.x with the patch |
|---|---|---|
| Whole run | 137 s | 102 s |
| Peak memory | 2,929 MB | 1,480 MB |
| Mandatory tour scheduling | 44.8 s, +2,322 MB | 13.0 s, +660 MB |
| Non-mandatory tour scheduling (no logsums on ARC) | 16.3 s, +731 MB | 15.9 s, +737 MB |

## All ARC households, with the patch

| Step | Time (s) | Extra memory (MB) | Peak memory (MB) |
|---|---|---|---|
| non_mandatory_tour_scheduling | 121.4 | 5,029 | 5,846 |
| mandatory_tour_scheduling | 74.5 | 3,829 | 4,602 |
| trip_scheduling_choice | 6.0 | 1,601 | 2,681 |
| workplace_location | 13.3 | 945 | 1,707 |

Whole run: 347 seconds, peak 5.8 GB. The biggest memory user is now
non-mandatory tour scheduling, which doesn't use logsums on ARC: its memory
is the tour × option table itself. That's the clearest target for
asim_core's batch-by-batch design.

## What this means for asim_core

- **The same rule belongs in asim_core's design:** compute a logsum once
  per distinct combination of the inputs its formulas read. asim_core
  already knows those inputs, because the formula translator records the
  columns every formula uses. It can apply the rule generally, not as a
  special case.
- **Porting mandatory tour scheduling needs tour mode choice first,** since
  most of the step is mode choice logsums. Tour mode choice is the next
  port. Scheduling's own remaining memory (the tour × option table, about
  0.1 MB per tour on ARC) is then the target for asim_core's batch-by-batch
  design.
- **The patch could be offered to the ActivitySim project** as a change
  to `dedupe_alt_tdd`, with these tests as evidence.

## Reproducing

From the repository root:

```bash
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/prototype_arc --households 4000 --out asim_core/work/arc/full4000_base
```

```bash
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/prototype_arc --households 4000 --patch logsum_dedupe --out asim_core/work/arc/full4000_patched
```

```bash
python asim_core/harness/compare_asim1_runs.py asim_core/work/arc/full4000_base asim_core/work/arc/full4000_patched
```
