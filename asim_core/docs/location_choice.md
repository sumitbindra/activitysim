# School and workplace location

Every student gets a usual school zone and every worker a usual work
zone. These two 1.x steps run right after accessibility and before auto
ownership, so with them asim_core computes the whole start of the model:
accessibility, school location, workplace location, auto ownership.

## Summary

- **asim_core matches 1.x exactly on all three regions,** from today's
  files and from the converted model files: every chosen zone, every
  destination choice logsum (bit for bit), every annotation column of
  persons and households, and the modeled-size tables.

  | Region | 1.x run used | Persons | Students choosing | Workers choosing | Columns compared | Exact |
  |---|---|---|---|---|---|---|
  | MTC | full example (5,000 households) | 8,212 | 1,633 | 4,361 | 20 | all |
  | MWCOG | 100 households | 135 | 40 | 80 | 19 | all |
  | MWCOG | all 29,972 households | 40,137 | 11,103 | 25,471 | 19 | all |
  | ARC | 100 households | 308 | 104 | 160 | 31 | all |
  | ARC | 4,000 households | 11,251 | 3,797 | 6,029 | 31 | all |
  | ARC | all 24,061 households | 67,101 | 22,574 | 35,803 | 31 | all |

  End to end on MTC, asim_core now computes 8 of the 34 steps (accessibility,
  both location steps, auto ownership, both scheduling steps, non-mandatory
  destination and tour mode choice) and replays the other 26 from 1.x; every
  compared column is exact, from today's files and from the model files.

- **Built from pieces asim_core already had.** Location choice is a
  destination choice for persons instead of tours: the same sampling, size
  terms, mode choice logsums and choice code as non-mandatory tour
  destination, with the random numbers taken from each person's stream.
  New were the results written to persons (and a new table), and
  annotations after the model.

- **Annotations after the model**, a capability most of the remaining 1.x
  steps need: columns worked out for whole tables once the step has chosen
  (here: distance and travel times to work and school, a household's sum
  over its workers). They run in 1.x's order and merge as 1.x does.

- **5 to 15 times faster than 1.x's legacy engine, with less memory**
  (ARC, all households: workplace 2.7 s and +382 MB against 13.6 s and
  +1,209 MB). Against sharrow on MTC: 8 to 19 times faster, but sharrow
  needs less memory for workplace location. Details below.

- **Two last-bit details found and fixed** while matching 1.x (below):
  lookups of zone -1, and padding widths. The second was also a 1.x
  problem: its results change with its chunk size (1.x findings, #3).


## How 1.x does it

For each step (`activitysim/abm/models/location_choice.py`, without
shadow pricing, which none of the three examples turn on):

1. **Choosers:** persons with their household, home zone and
   accessibility (`persons_merged`), only those with the filter column set
   (`is_student`, `is_worker`), sorted by person ID.
2. **For each segment, in the order of `SEGMENT_IDS`** (MTC: school types,
   income groups for work; ARC: occupation groups for work), the persons
   whose segment column has that value:
   - **size terms:** each zone's size for the segment (land use columns
     times weights, `destination_choice_size_terms.csv`); zones with size 0
     can't be chosen;
   - **sample** up to 30 zones per person, as in non-mandatory destination
     (capped at the number of zones with a size; one row per zone drawn,
     with its pick count);
   - **logsums** from tour mode choice for each sampled zone, at the
     segment's fixed periods, with its tour purpose (`work`, `school`,
     `univ`);
   - **choice** of one sampled zone, and the logsum over the sample.
3. **Results:** the chosen zone goes to persons (`-1` for everyone else),
   the logsum too (missing for everyone else); the count of persons
   choosing each zone, per segment, becomes a table (`school_modeled_size`,
   `workplace_modeled_size`).
4. **Annotations:** `annotate_persons`, then `annotate_households`.

Each person's random stream is used in that order: the sample's draws, the
mode choice preprocessor's draws, then the choice.

## Details that mattered for exact matching

- **Persons without a location have zone -1, and 1.x looks them up
  anyway.** The annotations compute, for every person,
  `skim_dict.lookup(persons.home_zone_id, persons.workplace_zone_id, 'DIST')`
  and `reindex(land_use.area_type, persons.workplace_zone_id)`, then mask
  the non-workers with `np.where`. In 1.x a skim lookup of zone -1 gives a
  missing value (any other unknown zone is an error), and a reindex of an
  unknown ID gives a missing value. asim_core does the same now (it used
  to stop with an error). ARC's annotations also look up school zones
  in the workplace annotation, and sum, per household, values that are
  missing for non-workers: pandas' groupby sum skips them, and so does
  asim_core (it uses pandas' groupby sum, so the additions are the same).
- **Padding width.** Each chooser's sampled zones are padded to a common
  width before the utilities' exponentials are added up. 1.x pads to the
  widest chooser of the whole call (the segment, unless chunked).
  asim_core padded to the widest chooser of each batch; since NumPy adds a
  row in pairs, the width changes how the sum is rounded, and with batches
  of 3,000 rows 12 of 8,212 MTC school logsums differed by 3.6e-15. asim_core
  now pads to the segment's widest chooser (and, in scheduling, the round's
  widest tour), so results don't depend on its batch size, and match 1.x's
  unchunked run. 1.x's own results do depend on its chunk size this way
  (1.x findings, #3).
- **A spec with a single `coefficient` column** serves every segment
  (1.x's `spec_for_segment`); MTC's workplace specs are written so.
- **The alternatives carry 1.x's shadow price columns**, at their values
  without shadow pricing: `shadow_price_size_term_adjustment` = 1 and
  `shadow_price_utility_adjustment` = 0. The specs multiply and add them.
- **The logsums' preprocessor.** These steps ask for tour mode choice's
  `nontour_preprocessor`. In all three examples it is the same file as
  `preprocessor`, with fewer tables; the converter and loader check that.

## In the model files

`kind: location_choice` (converted for all three regions in
`asim_core/configs/<example>/`). An excerpt of MTC's workplace model:

```yaml
kind: location_choice
filter: person.is_worker
segment: household.income_segment      # 1.x: CHOOSER_SEGMENT_COLUMN_NAME (it is a household column)
segments: {work_low: 1, work_med: 2, work_high: 3, work_veryhigh: 4}
size_terms: {file: destination_choice_size_terms.csv, selector: workplace}
sample_size: 30
origin: person.home_zone_id
logsums:
  model: tour_mode_choice.yaml
  tour_purpose: work
  periods: [8, 17]
outputs:
  zone: workplace_zone_id
  zone_when_none: -1
  logsum: workplace_location_logsum
  modeled_size: workplace_modeled_size
annotate:
  persons: {derived: annotate_persons_workplace.csv, tables: [land_use]}
  households: {derived: annotate_households_workplace.csv, tables: [persons]}
options:
  shadow_pricing: false                 # not implemented yet
```

The annotations, converted:

```
distance_to_work   where(person.workplace_zone_id >= 0, skim.DIST[person.home_zone_id, person.workplace_zone_id], nan)
workplace_in_cbd   land_use.area_type[person.workplace_zone_id] < setting('cbd_threshold')
hh_work_auto_savings_ratio
                   fillna(sum_by(persons.work_auto_savings_ratio, persons.household_id, household.household_id), 0.0)
```

## Profiles

Each step run in a fresh process, starting from 1.x's saved state before
it, the same way for every engine (`harness/profile_step.py`). Not timed:
loading the saved state, getting skims ready, and 1.x's one-time Numba
compiles. Wall time is the median; extra memory is the step's peak above
where it started.

**MTC, 5 runs each** (8,212 persons; 1,633 students, 4,361 workers):

| | School location | Workplace location |
|---|---|---|
| asim_core | 0.13 s (0.12–0.15), +40 MB | 0.38 s (0.37–0.44), +118 MB |
| 1.x, legacy | 1.88 s (1.77–2.80), +53 MB | 3.61 s (3.40–4.04), +139 MB |
| 1.x, sharrow | 2.51 s (2.44–106.1), +50 MB | 2.96 s (2.73–8.19), +62 MB |
| same step in the full 1.x run (legacy / sharrow) | 4.22 s / 4.42 s | 4.28 s / 2.12 s |

One of the five sharrow school location runs took 106 s and +1,046 MB:
sharrow compiled code it didn't find in its cache. On MTC's workplace
location, sharrow needs less memory than asim_core (+62 MB against
+118 MB).

**ARC, all 24,061 households, 3 runs each** (67,101 persons; 22,574
students, 35,803 workers):

| | School location | Workplace location |
|---|---|---|
| asim_core | 0.87 s (0.86–0.91), +330 MB | 2.72 s (2.63–2.76), +382 MB |
| 1.x, legacy | 4.76 s (4.65–5.00), +645 MB | 13.60 s (13.24–14.46), +1,209 MB |

About 5 times faster than 1.x on ARC, with a third to a half of the extra
memory.

**The padding fix costs nothing measurable.** Scheduling now works out the
padding width of each round before its batches. MTC mandatory and
non-mandatory scheduling: 0.99 s and 0.37 s (before: 1.05 s and 0.44 s).
ARC mandatory scheduling, all households, `distinct_inputs`: 18.2 s
(18.1–19.3), +288 MB (before: 19.2 s, +287 MB).

## Not supported yet

Shadow pricing (none of the examples use it; next), preprocessors in
location choice, two- and three-zone systems (presampling), saving the
sample table, estimation, and the mode choice logsum of the chosen zone as
an output (`MODE_CHOICE_LOGSUM_COLUMN_NAME` is implemented but no example
sets it, so it is untested against 1.x).
