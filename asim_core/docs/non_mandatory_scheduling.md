# Non-mandatory tour scheduling

Non-mandatory tour scheduling chooses a departure and return time for
every non-mandatory tour and fits it into the person's day. It was the
first hard step to port, and it was chosen as the test of asim_core's
design: on ARC it was 1.x's biggest memory user (+5.0 GB with all
households), because 1.x builds a table of every tour × every departure–
return option at once.

## Summary

- **asim_core matches 1.x exactly on all three regions.** Every
  non-mandatory tour gets the same departure, return and duration, and
  every person's timetable afterwards is the same, period by period:

  | Region | 1.x run used | Tours scheduled | Same times | Timetable cells identical |
  |---|---|---|---|---|
  | MTC | full example | 4,521 | 4,521 | 172,452 of 172,452 |
  | MWCOG (six specs, one per tour type) | 10,000 households | 6,998 | 6,998 | 668,600 of 668,600 |
  | ARC | 4,000 households | 5,173 | 5,173 | 562,550 of 562,550 |
  | ARC | all 24,061 households | 30,851 | 30,851 | 3,355,050 of 3,355,050 |

- **Utilities are bit-for-bit identical to 1.x's own functions,** on
  every (tour, option) row of the first round: 501,136 rows on MTC and
  3,069,657 rows on ARC (4,000 households), with one caveat on ARC, below.
- **Much less memory, and faster.** On all ARC households: 38 s and
  +172 MB, against 121 s and +5.0 GB inside 1.x's full run. Profiles below.
- **Found in 1.x: its utilities depend on how many rows it evaluates at
  once.** Details below. It changed no choice in these runs, but it means
  1.x's results aren't strictly independent of its batch size.

## How 1.x schedules, and what asim_core does differently

1.x (`non_mandatory_scheduling.py`, `util/vectorize_tour_scheduling.py`)
works in rounds: all first non-mandatory tours of each person, then all
second tours, and so on. For each round it:

1. builds a table with one row per tour × departure–return option that
   still fits the person's day (190 options on MTC, 1,176 on ARC and MWCOG);
2. adds the tour's and person's columns, the option's start, end and
   duration, and the start and end of the person's previous tour in this
   step;
3. evaluates the spec on every row, adds the terms up into one utility per
   row, chooses one option per tour, and books it into the person's
   timetable.

asim_core does the same, with two differences that don't change results:

- **A round is processed in batches** of at most a million rows (a setting).
  Each tour's rows depend only on its own person's timetable and random
  stream, so batch size can't change any result; the tests check this by
  running with tiny batches. 1.x, with its default settings, builds the
  whole round at once: 3.1 million rows for ARC's first round at 4,000
  households (all households: about six times as many households, so
  presumably several times more rows; not measured).
- **Fitting options into a day is done for a batch of tours at once,** with
  small matrix products over the periods of the day, instead of 1.x's loop
  over every tour–option pair. Same rule (1.x's list of colliding period
  states), checked against 1.x function by function.

## What was added to asim_core

- **Timetables** (`timetable.py`): each person's day as a row of periods,
  with 1.x's period states and collision rules; booking a tour; and the
  functions scheduling formulas call (`tt.adjacent_window_before`,
  `tt.previous_tour_ends`, `tt.remaining_periods_available`,
  `tt.max_time_block_available`, ...), vectorized. Each is checked against
  1.x's on real timetables, including result types.
- **Interaction models**: a spec with one coefficient per row, whose
  alternatives are made at run time. Rows written `_name@formula` define
  temporaries used by later rows, as in 1.x. Plain-style formulas see the
  model's own names (temporaries, constants) before columns, as in 1.x's
  interaction models.
- **A new swappable piece, `interaction_utilities`.** Its reference version
  adds term × coefficient one term at a time into a 64-bit total, as 1.x
  does. That order and precision matter here: each product is computed in
  the term's own type (a 32-bit column times its coefficient stays 32-bit),
  then added.
- **Preprocessor features**: the DF alias (`non_mandatory_tours.destination`),
  `reindex_i(joint_tour_participants.groupby('person_id').size(), ...)`
  (counts per person), `skim_dict.lookup(origins, destinations, ('SOV_DIST', 'MD'))`,
  `np.log10`, `.astype(np.float32)`.
- **Plain-style formulas with arithmetic are evaluated with `pandas.eval`,**
  exactly as 1.x evaluates them. See the next section for why.

## Details that mattered for exact matching

- **pandas.eval doesn't follow NumPy's type rules.** 1.x evaluates
  plain-style formulas (no `@`) with `pandas.eval`, which uses numexpr. For
  a float32 column times an int64 column, numexpr computes in 32 bits;
  NumPy computes in 64 bits. The values differ in the last bits, and here
  they're multiplied by a coefficient in their own precision, so they
  reach the utility. asim_core therefore evaluates plain-style formulas
  that do arithmetic with `pandas.eval` too. Plain-style comparisons and
  logic give the same values either way and stay NumPy code. (The earlier
  steps' arithmetic formulas happen to use 64-bit columns, where both give
  the same values; they now go through pandas.eval too, and still match.)
- **Coefficients act like Python floats.** 1.x multiplies a pandas Series
  by a 32-bit coefficient; pandas treats it like a Python float, so a 0/1
  or small-integer column gives a 64-bit product and a 32-bit column a
  32-bit product. asim_core passes the coefficients as Python floats to get
  the same.
- **Category columns stay categorical.** ARC's spec compares `tour_type`
  with a string in nearly every row. Comparing categorical codes is much
  faster than comparing strings, and gives the same answer; 1.x does the
  same. (This took ARC from 20 s to 6 s at 4,000 households.)
- **Padding.** Tours with fewer options are padded to a rectangle with
  utility −999 before probabilities, as in 1.x. With 1.x's default
  settings (no overflow protection) the padding gets probability exactly 0.
- **The person's "previous tour" starts as option 0** for everyone, as in
  1.x, so `start_previous` and `end_previous` are defined in round 1 too.

## Found in 1.x: results depend on how many rows are evaluated at once

pandas evaluates ordinary Series arithmetic with NumPy, except for arrays
with more than 1,000,000 elements, where it switches to numexpr. numexpr's
type rules differ: for example, a float32 column times 0.3 is computed in
float32 below the threshold and in float64 above it.

1.x evaluates `@` formulas on pandas Series, one round at a time. On MTC
a round has about 500,000 rows, below the threshold. On ARC the first round
has millions of rows, above it. So ARC's utilities in 1.x depend on whether
the round is evaluated at once or in pieces, which depends on the
household sample and on 1.x's chunk settings.

Checked directly, on ARC's first round at 4,000 households (3,069,657
rows), using 1.x's own `eval_interaction_utilities`:

| Comparison | Bit-identical utilities |
|---|---|
| asim_core vs 1.x, evaluated in slices of 500,000 rows | 3,069,657 of 3,069,657 |
| asim_core vs 1.x, whole round at once (1.x's default) | 264,769 |
| 1.x whole round vs 1.x in slices | 264,769 (largest difference 1.7e-7) |

So asim_core reproduces 1.x exactly as 1.x computes below the threshold,
and 1.x disagrees with itself across the threshold. Differences of 1e-7 in
a utility change a choice only if the random draw falls that close to a
boundary between options: none of the 30,851 ARC tours changed. But the
claim that 1.x's results don't depend on chunk size doesn't hold to the
last bit. This could be reported to the ActivitySim project with the check
above (`harness/asim1_reference.py`, `scheduling_round_utilities`).

asim_core's results don't depend on batch size: it never evaluates
formulas through pandas Series arithmetic.

## Profiles

(Fresh-process profiles being measured.)

## Not supported yet

Mode choice logsums inside scheduling (`LOGSUM_SETTINGS`; MTC, MWCOG and
ARC don't use them for non-mandatory tours, but mandatory scheduling
does), preprocessors for the alternatives (`ALTS_PREPROCESSOR`),
annotation after the model, and tour types without a spec segment (1.x
leaves those tours unscheduled; asim_core stops with an error).
