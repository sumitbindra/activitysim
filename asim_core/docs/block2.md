# Block 2: joint tours, non-mandatory tour frequency and at-work subtours

Ten more 1.x steps are ported: the five joint tour steps
(`joint_tour_frequency`, `joint_tour_composition`,
`joint_tour_participation`, `joint_tour_destination`,
`joint_tour_scheduling`), `non_mandatory_tour_frequency`, and the four
at-work subtour steps (`atwork_subtour_frequency`,
`atwork_subtour_destination`, `atwork_subtour_scheduling`,
`atwork_subtour_mode_choice`). With them, **MTC's and ARC's tour-level
model runs in asim_core alone**, from the input files through at-work
subtour mode choice: 23 steps, nothing taken from a 1.x run. (MWCOG still
replays its four region-specific steps, as in block 1.)

## Summary

- **Every new step matches 1.x exactly** on all five test runs: MTC,
  MWCOG and ARC (their example data) and MTC's full zone system with 10,000
  and 50,000 sampled households, each step from 1.x's saved state and end
  to end (`run_milestone1.py`: ALL EXACT; 727 compared columns on MTC,
  1,386 on MWCOG, 1,180 on ARC, 706 on each full MTC run). The full MTC
  1.x runs used for block 1 stopped after tour mode choice; they were
  resumed (1.x's `resume_after`) through the four at-work steps, numexpr
  off as before.
- **Run on its own** (`harness/run_standalone.py`: the 23 steps from the
  input files in one process, nothing from 1.x, then compared with 1.x's
  tables after at-work subtour mode choice): 203 of 203 columns exact on
  MTC (1.1 million values), 335 of 335 on ARC, and 203 of 203 on full MTC
  with 10,000 households (3.4 million values) and with 50,000 (16.6
  million values).
- **Speed** (one process, full MTC, 50,000 households): the ten new steps
  take 15 s against 65 s in 1.x; all 23 steps 104 s against 524 s. asim_core's
  own memory stays under 1.3 GB (the skims are a shared memory-mapped
  file); 1.x holds 8 to 13 GB. At 10,000 households: 4.9 s
  against 22 s for the ten steps, 36 s against 137 s for all 23 (1.x as
  shipped, numexpr on).
- **1.x finding #17**: in MTC's at-work scheduling spec, four terms labelled
  for eating out test business subtours, and the "2 to 3 hours" term covers
  1 to 4 hours. Correcting the eating-out terms changes the schedule of 70%
  of at-work subtours (eat subtours departing at lunchtime: 53% → 91%)
  ([../../asim1_findings/README.md](../../asim1_findings/README.md#17-at-work-subtour-scheduling-eating-out-terms-test-business-subtours-mtc)).
  Finding #16 (joint and non-mandatory specs testing tour types that don't
  exist) also came from this block, and two smaller items (at-work mode
  choice losing `household_id` in a merge; an ERROR logged in every run
  that isn't one).
- **Built from the existing pieces** (simple choice, interaction choice,
  destination with sampling, scheduling, tour mode choice), with four
  additions: steps that add rows to a table (the runner learns the new
  table's columns when the step is checked), a scheduling timetable per
  joint tour or per work tour, formula views (some rows of a table, two
  tables merged), and 1.x's time-window functions.

## The steps

### Joint tours

**joint_tour_frequency** (`components/joint_tour_frequency.py`). Households
with `participates_in_jtf_model` choose among
`joint_tour_frequency_alternatives.csv` (MTC: none, one tour of five
purposes, or two). The preprocessor sees the households' persons and their
shared free time (`hh_time_window_overlap`). Each chosen tour becomes a
`tours` row (category `joint`), given for now to the household's first
person and starting from home, with 1.x's tour IDs (`j_shopping1`, ...);
the tours are added after the existing ones as 1.x's `extend_table` adds
them (categories unioned and sorted, missing text filled with '').

**joint_tour_composition** (`components/joint_tour_composition.py`). Each
joint tour, with its household (annotated again with shared free time),
chooses adults, children or mixed; `tours.composition` is a category of the
spec's alternatives and '' for the other tours.

**joint_tour_participation** (`components/joint_tour_participation.py`):

1. **Candidates**: each joint tour with each member of its household,
   except adults on children-only tours and children on adults-only tours.
   A candidate's ID is `tour_id × 100 + PNUM`; it names their random number
   stream (1.x's `joint_tour_participants` channel).
2. The preprocessor annotates the candidates (with the persons' shared free
   time, `person_time_window_overlap`); a logit gives each candidate's
   probability of going.
3. **Choosing in rounds** (1.x's `participants_chooser`): every candidate
   still choosing draws; a tour is done when its party is valid (two or
   more people; on a mixed tour, an adult and a child); the candidates of
   the other tours draw again, up to `max_participation_choice_iterations`
   rounds.
4. `joint_tour_participants` (tour, household, person, and the
   participant's number within the tour); each joint tour's `person_id`
   becomes its first participant and `number_of_participants` its party
   size; then the persons' annotations.

**joint_tour_destination**: the non-mandatory destination model (sampling of
alternatives, size terms, mode choice logsums) for the joint tours, from
their first participant (`components/tour_destination.py`, the same
component as non-mandatory destination).

**joint_tour_scheduling** (`components/tour_scheduling.py`, category
`joint`). As the other scheduling steps (rounds of tour number, one row per
tour and departure-return option that fits), except that a joint tour must
fit every participant's day: each round builds a timetable with a row per
tour, its participants' windows combined participant by participant (1.x's
`build_joint_tour_timetables`), the previous tour is the household's
previous joint tour, and each round's choices are booked in every
participant's timetable.

### non_mandatory_tour_frequency

`components/non_mandatory_tour_frequency.py`, as 1.x:

1. Choosers: persons with a mandatory or non-mandatory day, annotated by
   the preprocessor (their longest free stretch, `person_max_window`, and
   their household's joint tours).
2. One model per person type (`SPEC_SEGMENTS`), each over **all**
   alternatives of `non_mandatory_tour_frequency_alternatives.csv` (1.x's
   `interaction_simulate`: every chooser with every alternative).
3. `persons.non_mandatory_tour_frequency`: the chosen alternative's row
   number (16-bit; the first alternative for everyone else).
4. **Extension** (1.x's `extend_tour_counts`): a person with 1 to 4 tours
   whose count of a type is that type's most may get one or two more,
   drawn from `non_mandatory_tour_frequency_extension_probs.csv` (one draw
   per type).
5. The tours, with 1.x's IDs, added after the existing tours; then the
   persons' annotations.

### At-work subtours

**atwork_subtour_frequency** (`components/atwork_subtour_frequency.py`).
Each work tour, with its person (1.x's `persons_merged`) and the
preprocessor's columns (the person's joint tours by type, from
`joint_tour_participants` joined with `tours`), chooses among
`atwork_subtour_frequency_alternatives.csv` (none, or up to two eat,
business and maintenance subtours). Each subtour gets its work tour as
`parent_tour_id`, the person and household, the workplace (the work tour's
destination) as origin, and 1.x's ID (`eat1_1`: the eat subtour of work
tour 1). `tours.atwork_subtour_frequency` is missing for tours that aren't
work tours, as in 1.x.

**atwork_subtour_destination**: the same destination component again, for
the subtours, from the workplace, with the defaults 1.x gives this step
for settings a region may leave out (size term selector and segment
`atwork`, origin `workplace_zone_id`).

**atwork_subtour_scheduling** (`components/tour_scheduling.py`, category
`atwork`), as 1.x's `vectorize_subtour_scheduling`:

- the timetable has a row per **work tour**, not per person: each period
  outside the work tour is taken and those within it free (1.x's
  `assign_subtour_mask`), so a subtour fits inside its work tour (it may
  start in the work tour's first hour and end in its last);
- rounds by subtour number; the previous tour is the work tour's previous
  subtour; each round's choices are booked in the work tour's row;
- persons' timetables are left as they are (1.x doesn't book subtours in
  `person_windows`);
- the preprocessor can use `od_skims` (from the subtour's origin to its
  destination) and `od_skims.reverse(...)`, with time-period keys:
  MTC's `od_skims[('SOV_TIME', 'MD')]`.

**atwork_subtour_mode_choice** (`components/tour_mode_choice.py`). The tour
mode choice model (same files, same component) for the subtours only, from
the workplace (`workplace_zone_id`) to the destination, all with the
`atwork` coefficients. Two 1.x details kept: the subtours get no
`tour_purpose` column, and a column both the tours and `persons_merged`
have (`household_id`) is not a chooser column, because 1.x merges them with
pandas' default suffixes, which rename both (tour mode choice itself keeps
the tour's). The preprocessor sees `parent_tour_id`, so MTC's
`work_tour_is_SOV` (the parent work tour's mode, looked up in `tours`)
takes effect here. The other tours keep their mode and logsum.

## Formula reading added

All in these steps' preprocessors and specs, each translated to NumPy at
load time:

| 1.x formula | Example |
|---|---|
| some rows of a table, and their columns | `_JOINT_TOURS = tours[tours.tour_category=='joint']`; `_JOINT_TOURS.tour_type` |
| counts per ID over some rows, filtered further | `reindex_i(_JOINT_TOURS[_JOINT_TOURS.tour_type=='shopping'].groupby('household_id').size(), persons.household_id)` |
| two tables merged, then filtered and counted | `pd.merge(joint_tour_participants[['tour_id', 'person_id']], tours[['tour_type']], left_on='tour_id', right_index=True, how='left')` |
| 1.x's time-window functions (results with several columns, as 1.x's) | `hh_time_window_overlap(households, persons)`, `person_time_window_overlap(persons)`, `person_max_window(persons)` |
| a column of such a result | `_HH_OVERLAPS['aa']` |
| counting lambdas with text built inside (8-bit counts, as 1.x) | `_PTYPE_CDAP_PATTERN_COUNT(1, 'M', households, persons)` |
| skim wrappers with periods, and the reverse direction | `od_skims[('SOV_TIME', 'MD')]`, `od_skims.reverse(('SOV_TIME', 'MD'))` |

## Profiles

**One process, the 23 steps from the input files** (`run_standalone.py`;
seconds, MB; the ten block 2 steps shown, the earlier ones as in
[block1.md](block1.md)). asim_core's memory is its own; "with skim pages"
adds the pages of the memory-mapped skim file the step held. 1.x's column
is the same step in its full run.

Full MTC, 50,000 households (1.x with numexpr off, the run asim_core
matches exactly):

| Step | asim_core s | asim_core own MB | with skim pages | 1.x s |
|---|---|---|---|---|
| joint_tour_frequency | 0.62 | 822 | 7,108 | 0.95 |
| joint_tour_composition | 0.08 | 796 | 7,082 | 0.53 |
| joint_tour_participation | 1.78 | 796 | 7,082 | 2.57 |
| joint_tour_destination | 1.11 | 852 | 7,139 | 7.67 |
| joint_tour_scheduling | 0.26 | 796 | 7,082 | 1.88 |
| non_mandatory_tour_frequency | 5.79 | 1,125 | 7,412 | 20.48 |
| atwork_subtour_frequency | 0.47 | 925 | 7,211 | 4.25 * |
| atwork_subtour_destination | 4.07 | 1,161 | 7,447 | 21.36 |
| atwork_subtour_scheduling | 0.54 | 909 | 7,195 | 3.26 |
| atwork_subtour_mode_choice | 0.71 | 907 | 7,194 | 2.05 |
| **ten block 2 steps** | **15.4** | | | **65.0** |
| the 13 earlier steps | 88.7 | | | 459.1 |
| **all 23 steps** | **104.1** | max 1,258 | | **524.1** |

\* 1.x's resumed run loaded its skims in `atwork_subtour_frequency` (44.3 s
in all); the 40 s of loading (from 1.x's log) is left out, since a full run
loads them once, in `initialize_landuse` (counted in the earlier steps).

Full MTC, 10,000 households (1.x as shipped, numexpr on; its at-work steps
from the shipped run resumed the same way):

| Step | asim_core s | asim_core own MB | with skim pages | 1.x s |
|---|---|---|---|---|
| joint_tour_frequency | 0.15 | 416 | 6,698 | 0.42 |
| joint_tour_composition | 0.03 | 416 | 6,698 | 0.19 |
| joint_tour_participation | 0.93 | 416 | 6,699 | 1.22 |
| joint_tour_destination | 0.40 | 416 | 6,699 | 3.06 |
| joint_tour_scheduling | 0.12 | 416 | 6,699 | 1.09 |
| non_mandatory_tour_frequency | 1.92 | 479 | 6,761 | 4.17 |
| atwork_subtour_frequency | 0.12 | 421 | 6,705 | 3.14 * |
| atwork_subtour_destination | 0.98 | 578 | 6,861 | 5.74 |
| atwork_subtour_scheduling | 0.11 | 436 | 6,719 | 1.99 |
| atwork_subtour_mode_choice | 0.11 | 436 | 6,720 | 1.18 |
| **ten block 2 steps** | **4.9** | | | **22.2** |
| the 13 earlier steps | 30.6 | | | 114.5 |
| **all 23 steps** | **35.5** | max 1,260 | | **136.7** |

\* As above: 44.1 s in all, about 41 s of it loading skims.

What it shows:

- **The new steps are a small part of the run**: 15% of asim_core's time
  at 50,000 households. The biggest are non-mandatory tour frequency
  (5.8 s: every active person against all 96 alternatives, one model per
  person type) and at-work subtour destination (4.1 s: sampling and mode
  choice logsums).
- **joint_tour_participation is the closest to 1.x** (1.8 s against
  2.6 s): building the candidates and the rounds of choosing (which
  parties are complete, who draws again) are 1.x's pandas code, kept so
  the draws come out the same. A
  candidate for NumPy later, like CDAP's tables in block 1.
- **Memory**: no new step adds much; asim_core's own peak for the whole run
  is the 1.26 GB of `initialize_households` (the input files parsed before
  sampling).
- The earlier steps' times vary with the page cache: `mandatory_tour_scheduling`
  took 8.1 s in this run against 5.6 s in block 1's at 10,000 households.
  The same step from block 1's code and from today's both take 4.0 s with
  the skim file's pages in memory (`profile_step.py --page-cache warm`,
  two runs each), so it is the page cache, not the code. (A first run right
  after 1.x's 12 GB run, which pushed the skim file out of memory, took
  16.9 s.)

**Per step, fresh processes** (MTC's example data, 5,000 households; each
step from 1.x's saved state in a new process, median of 3 runs;
`profile_step.py`):

| Step | asim_core s | 1.x s | asim_core extra MB | 1.x extra MB |
|---|---|---|---|---|
| joint_tour_frequency | 0.104 | 0.519 | 15 | 21 |
| joint_tour_composition | 0.026 | 0.308 | 5 | 18 |
| joint_tour_participation | 0.330 | 0.690 | 20 | 18 |
| joint_tour_destination | 0.113 | 2.642 | 15 | 30 |
| joint_tour_scheduling | 0.105 | 0.577 | 19 | 28 |
| non_mandatory_tour_frequency | 0.843 | 3.016 | 44 | 55 |
| atwork_subtour_frequency | 0.095 | 0.306 | 24 | 28 |
| atwork_subtour_destination | 0.110 | 1.198 | 55 | 67 |
| atwork_subtour_scheduling | 0.098 | 0.592 | 14 | 29 |
| atwork_subtour_mode_choice | 0.070 | 0.781 | 16 | 31 |
| **ten steps** | **1.89** | **10.63** | | |

These are small runs (MTC's example has 5,000 households and 25 zones);
the full MTC tables above show how the steps scale.

## Reproducing

```bash
python asim_core/harness/run_milestone1.py --profiles none
python asim_core/harness/run_standalone.py --through atwork_subtour_mode_choice --out asim_core/work/standalone_mtc_b2
python asim_core/harness/run_standalone.py --example-dir activitysim/examples/prototype_arc \
  --asim1-out asim_core/work/arc/asim1_legacy --through atwork_subtour_mode_choice --out asim_core/work/arc/standalone_b2
# full MTC: extend block 1's 1.x run through the at-work steps (a copy, resumed after tour mode choice)
cp -a asim_core/work/mtc_full/asim1_50k_nne2 asim_core/work/mtc_full/asim1_50k_atwork
python asim_core/harness/run_asim1.py --example-dir asim_core/work/mtc_full/prototype_mtc_full \
  --out asim_core/work/mtc_full/asim1_50k_atwork --households 50000 --patch no_numexpr \
  --resume-after tour_mode_choice_simulate --stop-after atwork_subtour_mode_choice
python asim_core/harness/run_milestone1.py --profiles none --skip-utility-check \
  --example-dir asim_core/work/mtc_full/prototype_mtc_full --asim1-out asim_core/work/mtc_full/asim1_50k_atwork \
  --asim1-sharrow-out none --out asim_core/work/mtc_full/milestone_50k_b2c
python -m pytest asim_core/tests/test_against_asim1.py -k block2
for step in joint_tour_frequency atwork_subtour_scheduling; do  # each step, a fresh process
  python asim_core/harness/profile_step.py --engine asim_core --step $step --scratch /tmp/p --out /tmp/p/asim_core.csv
  python asim_core/harness/profile_step.py --engine asim1 --step $step --scratch /tmp/p1 --out /tmp/p/asim1.csv
done
```

`harness/run_milestone1.py` now skips (and names) ported steps a 1.x run
doesn't have, rather than stopping.
