# Block 3: the trip steps

Eight more 1.x steps are ported: the six trip steps of MTC and MWCOG
(`stop_frequency`, which makes the trips, `trip_purpose`,
`trip_destination`, `trip_purpose_and_destination`, `trip_scheduling`,
`trip_mode_choice`) and the two ARC uses instead of trip scheduling
(`trip_scheduling_choice`, `trip_departure_choice`). With them, **every
model step of the three example regions runs in asim_core**; what is
still taken from 1.x is the output and summary steps (writing tables and
trip matrices, the data dictionary, skim usage, `summarize`) and, for
MWCOG, its four region-specific steps.

## Summary

- **Every new step matches 1.x exactly** on all five test runs, each step
  from 1.x's saved state and end to end (`run_milestone1.py`: ALL EXACT):
  MTC (29 of its 34 steps computed, 826 compared columns), MWCOG (29 of 37,
  1,485), ARC (30 of 34, 1,358) and MTC's full zone system with 10,000 and
  50,000 households (29 of 34, 826 each). The full-MTC 1.x runs were
  resumed once more, after at-work mode choice, through the end of the
  model, numexpr off as before.
- **The whole model on its own** (`run_standalone.py`: the 29 model steps
  from the input files in one process, nothing from 1.x, then compared with
  1.x's tables after trip mode choice): 219 of 219 columns exact on MTC (1.4
  million values), on full MTC with 10,000 households (4.7 million values)
  and with 50,000 (23.2 million values); 379 of 379 on ARC (its 30 model
  steps).
- **Speed** (one process, full MTC): the 29 model steps take 58 s against
  227 s in 1.x at 10,000 households (1.x as shipped), and 240 s against
  806 s at 50,000 (1.x with numexpr off, as exact matching needs): 3.9 and
  3.4 times faster. Trip destination is now the largest step (85 s at
  50,000 households; 1.x: 230 s). Trip scheduling is the one step no faster
  than 1.x (25 s against 21 s): it is 1.x's pandas code, kept for exactness,
  and a candidate for later work.
- **Memory**: asim_core's own peak for the whole run is 1.55 GB at 50,000
  households (in trip destination); the skims are a shared memory-mapped
  file. 1.x holds 8 to 13 GB.
- **1.x findings #18 to #20**
  ([../../asim1_findings/README.md](../../asim1_findings/README.md)):
  trip scheduling's default logic forces 11,289 trips (2.6%) to their
  window's edge after 100 failed attempts, against 474 with
  `logic_version: 2` (#18); MTC's stop frequency looks household tour counts
  up by person ID, and 2,735 of 175,579 tours get another stop frequency
  when corrected (#19); ARC's trip scheduling specs count stops one too low,
  test a tour type that doesn't exist and check inbound stop times against
  the outbound leg (#20). Two smaller items.
- **Built from the existing pieces** (simple choice, interaction choice
  with sampling, the mode choice model), with these additions: the mode
  choice model takes its skim lookups and the names its formulas get (trips:
  `trip_period`, `ORIGIN`/`DESTINATION`), interaction formulas can call the
  step's objects (`size_terms.get`), and the formula reading below.

## The steps

### stop_frequency

`components/stop_frequency.py`. Every tour, with its person (1.x's
`tours_merged`), annotated by the preprocessor: `primary_purpose` (the tour
type; `univ` for a university student's school tour; `atwork` for at-work
subtours), the tour's distance (`od_skims`), household counts by person
type, the tour's and household's counts by purpose, accessibility at both
ends. One model per primary purpose (`SPEC_SEGMENTS`), one draw per tour;
`tours.stop_frequency` (`'1out_2in'`, ...) and `tours.primary_purpose`.
Then **the trips table**: for each tour, `out + 1` outbound and `in + 1`
inbound trips, numbered within each direction, outbound from the tour's
origin to its destination and inbound back; 1.x's trip IDs (`tour_id × 2 ×
most trips per leg + trip number`, inbound after outbound).

### trip_purpose

`components/trip_purpose.py`. A direction's last trip goes to the tour's
primary purpose (outbound) or home (inbound; work for at-work subtours);
every other trip draws its purpose from `trip_purpose_probs.csv`, the row
matching its primary purpose, direction and person type whose departure
range holds the tour's start, the row's probabilities divided by their sum.

### trip_destination

`components/trip_destination.py`. The stops between a direction's first
and last trips, in rounds by trip number (a stop's zone becomes the next
trip's origin), per primary purpose:

1. **sample**: every zone gets a utility from the sample spec (distances
   origin → stop → leg destination, the stop's size for the trip's purpose
   from the `trip` size terms); `SAMPLE_SIZE` draws per trip;
2. **logsums**: two trip mode choice logsums per sampled zone, origin →
   stop (`od_logsum`) and stop → leg destination (`dp_logsum`), each with
   the trip mode choice preprocessor run first (1.x runs the second on the
   rows the first annotated, and asim_core does the same);
3. **choice** among the sampled zones; the chosen zone's logsum is
   `destination_logsum`.

A trip with no usable zone fails (destination -1, its next trip starts
where it did); with `CLEANUP`, failed trips are dropped as 1.x does. The
trips table ends up sorted by trip ID, as 1.x leaves it.

### trip_purpose_and_destination

`components/trip_purpose_and_destination.py`. 1.x retries failed trips
here (their purpose and destination again); with none failed, as in every
run checked, it drops the trips' `failed` column, which is what asim_core
does. Retrying failed trips is not supported yet: the step stops with that
message if it ever meets one.

### trip_scheduling

`components/trip_scheduling.py` (departure mode, 1.x's pandas code). Each
trip's window is its tour's, cut by at-work subtours; a direction's first
outbound trip leaves at the tour's start and its last inbound trip at its
end; the others draw a period from `trip_scheduling_probs.csv`, outbound
trips in trip order and inbound ones from the last back, each window
starting (ending) at the departure before (after) it. Only a direction's
first scheduled trip has its probabilities scaled back to 1, so the others
can fail; a failed direction is scheduled again, up to `MAX_ITERATIONS`
(MTC: 100), and on the last attempt a trip that still fails leaves at its
window's edge. 1.x's `logic_version` 1 (its default) and 2 are supported.

### trip_scheduling_choice and trip_departure_choice (ARC)

`components/trip_scheduling_choice.py`, `components/trip_departure_choice.py`.
First, per tour with stops, how its duration splits between the outbound
stops, the main leg and the inbound stops: every split into whole periods
is an alternative, one draw per tour. Then, per leg with stops, how the
leg's time splits over its stops: every pattern is an alternative, its
utility the sum of its stops' (`tour_legs` random stream, leg ID `tour_id +
10**9` outbound, `+ 2 × 10**9` inbound, as 1.x); departures follow from the
pattern. Both follow 1.x's pandas steps where the order of sums matters.

### trip_mode_choice

`components/trip_mode_choice.py`. The tour mode choice model
(`TourModeChoiceModel`), read from `trip_mode_choice.yaml` with the trips'
skim lookups: `odt_skims` / `dot_skims` at the trip's period (from its
departure), `od_skims`; `ORIGIN`, `DESTINATION`, `MIN_PER_PERIOD` and
`PERIODS_PER_HOUR` given to the formulas as 1.x gives them. Choosers are the
trips with their tour's columns listed in `TOURS_MERGED_CHOOSER_COLUMNS`;
per primary purpose, the preprocessor (random waiting times included), then
the nested logit. Trip destination uses the same model for its logsums,
with other origin and destination columns.

## Formula reading added

| 1.x formula | Example |
|---|---|
| some of the rows being annotated, counted | `reindex_i(df[~is_joint & (df.tour_type==SHOP_TOUR)].groupby('person_id').size(), df.person_id)` |
| counting lambdas giving counts per key, used inside `reindex_i` | `_HH_PERSON_COUNT('age < 5', persons)` |
| counting lambdas with a shift before the reindex (ARC) | `(trips.query(exp).groupby('tour_id').size() - 1).reindex(tours.index)` |
| skim time period labels | `network_los.skim_time_period_label(df.start)` |
| `map(dict, na_action='ignore')`, `isnull()` / `notnull()` | `df.atwork_subtour_frequency.map(num_atwork_subtours_map, na_action='ignore')` |
| size terms by zone and purpose (an object the step gives the formulas) | `size_terms.get(df.dest_taz, df.purpose)`, `size_terms.get_cols(df.purpose)` |
| a condition on a coefficient (decided per segment, MWCOG) | `@(...).clip(origin_density_index_max) if origin_density_applied else 0` |
| `np.sin`, `np.cos`, `np.absolute`, `np.pi` (ARC) | `np.sin(2 * np.pi * (df['start']/48))` |
| `LEGACY_COEFFICIENTS` (ARC's trip mode choice), specs without a coefficients file (kept 64-bit, as 1.x does) | |

## Profiles

**One process, the 29 model steps from the input files** (`run_standalone.py`;
seconds, MB; the six trip steps shown, the earlier ones as in
[block1.md](block1.md) and [block2.md](block2.md)). asim_core's memory is its
own; "with skim pages" adds the pages of the memory-mapped skim file the step
held. 1.x's column is the same step in its full run.

Full MTC, 10,000 households (1.x as shipped, numexpr on):

| Step | asim_core s | asim_core own MB | with skim pages | 1.x s |
|---|---|---|---|---|
| stop_frequency | 0.48 | 516 | 6,844 | 3.53 * |
| trip_purpose | 0.11 | 516 | 6,844 | 0.46 |
| trip_destination | 16.68 | 931 | 7,262 | 68.41 |
| trip_purpose_and_destination | 0.05 | 489 | 6,820 | 0.08 |
| trip_scheduling | 9.38 | 489 | 6,821 | 9.51 |
| trip_mode_choice | 1.63 | 565 | 6,896 | 8.56 |
| **six trip steps** | **28.3** | | | **90.5** |
| the 23 tour-level steps | 29.5 | | | 136.7 |
| **all 29 model steps** | **57.9** | max 1,263 | | **227.3** |

Full MTC, 50,000 households (1.x with numexpr off, the run asim_core matches
exactly):

| Step | asim_core s | asim_core own MB | with skim pages | 1.x s |
|---|---|---|---|---|
| stop_frequency | 1.71 | 1,029 | 7,364 | 5.27 * |
| trip_purpose | 0.60 | 1,048 | 7,382 | 1.01 |
| trip_destination | 84.53 | 1,553 | 7,884 | 230.19 |
| trip_purpose_and_destination | 0.23 | 1,010 | 7,342 | 0.28 |
| trip_scheduling | 25.09 | 1,025 | 7,356 | 21.18 |
| trip_mode_choice | 9.93 | 1,378 | 7,709 | 23.66 |
| **six trip steps** | **122.1** | | | **281.6** |
| the 23 tour-level steps | 117.4 | | | 524.1 |
| **all 29 model steps** | **239.5** | max 1,553 | | **805.7** |

\* 1.x's resumed runs loaded their skims in their first step (here
`stop_frequency`, and `atwork_subtour_frequency` in block 2's); that time
(43 to 45 s, from 1.x's log) is left out, since a full run loads them once,
in `initialize_landuse` (counted in the tour-level steps).

What it shows:

- **Trip destination is most of asim_core's trip time** (59% at 10,000
  households, 69% at 50,000) and the largest
  step of the run: for every stop, 30 sampled zones, two mode choice logsums
  each (the trip mode choice preprocessor and nested logit for every row),
  then the choice. 2.7 to 4.1 times faster than 1.x.
- **Trip scheduling is not faster than 1.x** (9.4 s against 9.5 s at 10,000
  households; 25.1 s against 21.2 s at 50,000). It is a port of 1.x's pandas
  code (merges per round of trips, up to 100 attempts), kept so each attempt
  draws as 1.x does.
- **Memory**: at 50,000 households trip destination's batches of trips ×
  zones (and their logsum rows) set the run's peak, 1.55 GB; at 10,000 the
  peak is still reading the input files (1.26 GB).

**Since these tables** (`future_experiments.md`): trip destination's
sampling copies out only the columns its spec reads, and takes each stop's
sizes as its purpose's column of the size table. Same results; at 50,000
households the step went from 68 to 71 s to 46 to 48 s (from 1.x's saved
state, skim file warm), its own memory peak from about 1.1 to 1.0 GB.

**Per step, fresh processes** (the example data: MTC, 5,000 households, and
ARC for its two steps; each step from 1.x's saved state in a new process,
median of 3 runs; `profile_step.py`):

| Step | Example | asim_core s | 1.x s | asim_core extra MB | 1.x extra MB |
|---|---|---|---|---|---|
| stop_frequency | MTC | 0.212 | 1.020 | 33 | 30 |
| trip_purpose | MTC | 0.037 | 0.229 | 15 | 18 |
| trip_destination | MTC | 1.733 | 31.960 | 111 | 127 |
| trip_purpose_and_destination | MTC | 0.053 | 0.199 | 24 | 43 |
| trip_scheduling | MTC | 7.894 | 9.098 | 13 | 22 |
| trip_mode_choice | MTC | 0.542 | 6.260 | 56 | 77 |
| trip_scheduling_choice | ARC | 0.094 | 0.368 | 6 | 6 |
| trip_departure_choice | ARC | 0.072 | 0.261 | 5 | 6 |

On MTC's 25 zones trip destination is 18 times faster than 1.x; on full
MTC's 1,475 zones, 2.7 to 4 times. Why the gain shrinks with more zones
(per-call costs in 1.x, which runs a sample, two logsum and one choice
evaluation per trip number and primary purpose, against work per zone) has
not been measured yet.

## Reproducing

```bash
python asim_core/harness/run_milestone1.py --profiles none
python asim_core/harness/run_standalone.py --through trip_mode_choice --out asim_core/work/standalone_mtc_b3
python asim_core/harness/run_standalone.py --example-dir activitysim/examples/prototype_arc \
  --asim1-out asim_core/work/arc/asim1_legacy --through trip_mode_choice --out asim_core/work/arc/standalone_b3
# full MTC: extend block 2's 1.x run through the end of the model (a copy, resumed after at-work mode choice)
cp -a asim_core/work/mtc_full/asim1_50k_atwork asim_core/work/mtc_full/asim1_50k_full
python asim_core/harness/run_asim1.py --example-dir asim_core/work/mtc_full/prototype_mtc_full \
  --out asim_core/work/mtc_full/asim1_50k_full --households 50000 --patch no_numexpr \
  --resume-after atwork_subtour_mode_choice
python asim_core/harness/run_milestone1.py --profiles none --skip-utility-check \
  --example-dir asim_core/work/mtc_full/prototype_mtc_full --asim1-out asim_core/work/mtc_full/asim1_50k_full \
  --asim1-sharrow-out none --out asim_core/work/mtc_full/milestone_50k_b3
python -m pytest asim_core/tests/test_against_asim1.py -k "block3 or trip_ids"
```

Full MTC's `network_los.yaml` has `write_skim_cache: True` and
`read_skim_cache: False`: every 1.x run writes a 6.7 GB copy of the skims
into its output folder and never reads it. Delete
`<run>/cache/cached_taz.mmap` after each run, or the disk fills.
