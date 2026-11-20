# Mandatory tour scheduling, and logsums computed once

Mandatory tour scheduling (work, school and university tours) works like
non-mandatory scheduling ([non_mandatory_scheduling.md](non_mandatory_scheduling.md)),
with one addition that dominates its cost in 1.x: for every tour and every
departure–return option, a **tour mode choice logsum** for that option's
departure and return periods. On ARC that was 74% of the step's time and
most of its memory ([scheduling.md](scheduling.md)).

This port was the test of two claims: that ported pieces can be combined
(scheduling uses the tour mode choice model), and that a known improvement
can be added as a swappable piece and checked exactly.

## Summary

- **asim_core matches 1.x exactly on all three regions.** Every mandatory
  tour gets the same times, every timetable is the same, and **every mode
  choice logsum 1.x computes inside the step is bit-for-bit identical**:

  | Region | 1.x run used | Tours | Same times | Timetable cells identical | Logsums bit-identical |
  |---|---|---|---|---|---|
  | MTC | full example | 4,477 | 4,477 | 172,452 | 831,809 of 831,809 |
  | MWCOG | 10,000 households | 8,479 | 8,479 | 668,600 | 9,549,117 of 9,549,117 |
  | ARC | 4,000 households | 5,601 | 5,601 | 562,550 | 6,360,778 of 6,360,778 |
  | ARC | all 24,061 households | 33,605 | 33,605 | 3,355,050 | (not captured) |

- **Logsums are computed once per distinct input, as a swappable piece.**
  `logsum_keys` decides which option rows share a logsum. The reference
  does what 1.x does; `distinct_inputs` uses only the option columns the
  mode choice formulas actually read. On ARC the two give bit-identical
  logsums on all 6,360,778 rows, and the step runs ~3 times faster.
- **Much less memory, and faster.** (Profiles being measured.)
- **Combining pieces took one refactor.** Tour mode choice had been written
  as a step. It was split into a model (`TourModeChoiceModel`: read the
  files once, then `bind` to some chooser columns and a destination column,
  and `evaluate` rows) and a thin step on top. Scheduling uses the same
  model for its logsums, with its own destination column per purpose
  (`workplace_zone_id`, `school_zone_id`). Tour mode choice's own results
  are unchanged (re-checked on MTC and MWCOG). Non-mandatory and mandatory
  scheduling are now one component with settings.

## How the logsums are computed

For each batch of tours (see non-mandatory scheduling for the batches):

1. every option that fits the person's day gets its departure and return
   period labels;
2. options are grouped by tour and by the columns `logsum_keys` returns;
   one row per group is passed to the tour mode choice model with the
   tour's and person's columns, the period labels and, as in 1.x, either
   the option's duration (no segments file: ARC) or the representative
   start, end and duration of its periods (MTC and MWCOG ship a
   `tour_departure_and_duration_segments.csv`);
3. the model runs its preprocessor (including its random wait-time draws,
   once per tour, as 1.x does) and its nested logit, and returns logsums,
   at most 100,000 rows at a time and never splitting a tour;
4. each option row gets its group's logsum.

**The two `logsum_keys` implementations:**

| | Groups options by | MTC, MWCOG (segments file) | ARC (no segments file) |
|---|---|---|---|
| `asim1` (reference) | departure period, return period, and duration if no segments file (1.x's `dedupe_alt_tdd`) | periods | periods and duration |
| `distinct_inputs` | only the option columns the mode choice model reads | periods (their formulas read duration, which depends on the periods there) | periods only (ARC's formulas never read duration) |

The model knows exactly which columns it reads: the formula translator
records every column of every formula, skims included (a time-of-day skim
lookup reads the period).

(Counts of logsum rows under each rule: being measured.)

## What else was added

- **Segments** (work, school, univ), with the university segment worked
  out as 1.x does (a school tour of a university student).
- **The alternatives' preprocessor** (MWCOG, ARC): columns computed on the
  option rows, e.g. departure shift from 9 a.m.
- **The chooser preprocessor in the tours' table order**, which matters
  for MWCOG's `df.tour_type.shift(-1)` (the next tour's type).
- **Translator:** `shift()`, Python's `abs()`.
- **A 1.x harness check** (`capture_scheduling_logsums` in
  `harness/asim1_reference.py`) that runs 1.x's step from its saved state
  and records every logsum it computes, for bit-for-bit comparison.

## Profiles

(Fresh-process profiles: being measured.)

## Not supported yet

Park-and-ride in the logsums, trip mode choice logsums, a
`LOGSUM_PREPROCESSOR` other than `preprocessor`, annotation after the
model.
