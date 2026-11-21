# Block 1: from the input files through mandatory tour scheduling

Five more 1.x steps are ported: `initialize_landuse`,
`initialize_households`, `free_parking`, `cdap_simulate` and
`mandatory_tour_frequency`. With them, **MTC's and ARC's first ten steps
all run in asim_core**, from the input files through mandatory tour
scheduling, without anything taken from a 1.x run. (MWCOG runs four
region-specific steps in between, `work_from_home`, `transit_pass_subsidy`,
`transit_pass_ownership` and `telecommute_frequency`, which are still
replayed from 1.x.)

## Summary

- **Every new step matches 1.x exactly** on all four test runs: MTC,
  MWCOG and ARC (their example data) and MTC's full zone system with
  10,000 sampled households, each step from 1.x's saved state and end to
  end (`run_milestone1.py`: ALL EXACT).
  <!-- 50k -->
- **Run on its own** (`harness/run_standalone.py`: the ten steps from the
  input files in one process, nothing from 1.x, then compared with 1.x's
  tables after mandatory tour scheduling): MTC 181 of 181 columns exact.
  <!-- standalone full -->
- **1.x finding #15**: CDAP ranks children by person type, not youngest
  first as its comments and CTRAMP say; in the 50,000-household full MTC
  run, 616 preschoolers get their day pattern from fixed proportions instead
  of the household's joint choice
  ([../../asim1_findings/README.md](../../asim1_findings/README.md#15-cdap-ranks-children-by-person-type-not-youngest-first)).
- **A correction to asim_core's formula reading**: 1.x evaluates utility
  expressions written without `@`, and query strings, with pandas, where
  `&` and `|` bind like `and` and `or`: `ptype == 7 & age >= 6` means
  `(ptype == 7) & (age >= 6)`. asim_core read them with Python's rules
  (`ptype == (7 & age) >= 6`). The ported specs happened to use
  parentheses, so no earlier result changes; ARC's household annotations
  (`_PERSON_COUNT('agep>=6 & agep<=15', ...)`) were the first to show it.

## The steps

### initialize_landuse and initialize_households

What 1.x does, reproduced (`components/initialize.py`):

- **Reading**: `input_table_list` as in 1.x (renames, kept columns,
  recodes, land use sorted by zone). CSV files are read with exactly 1.x's
  pandas call by default, so numbers are bit for bit 1.x's (finding #11)
  and so are column types (pandas infers them per block of rows, which
  matters for columns mixing numbers and text); `csv_reader="exact"` reads
  numbers correctly rounded instead. The files go straight to pandas: a
  first version converted them to Arrow and back, which on full MTC
  doubled the step's time and memory and left 0.6 GB held afterwards.
- **Sampling households** (`households_sample_size`): NumPy's
  `RandomState` seeded with the base seed and `hash32('sample_households')`,
  `choice` without replacement, households in the order drawn; the traced
  household replaces the first one if it wasn't drawn; `sample_rate` is the
  sample size over the total, written as is (even above 1). Persons of the
  sampled households are kept in file order; every person must have a
  household and every household a person. asim_core picks the sample's
  rows right after parsing (the same rows: 1.x's draw depends only on row
  positions), so the rest of the step handles 10,000 households rather
  than 2.9 million; not when a zero-based recode numbers all rows.
- **Annotations** in the settings' order (MTC: persons, households, persons
  again), with the random draws 1.x makes (`rng.lognormal_for_df`, one
  draw per row per call by default).
- **Made empty**: `accessibility` (the land use's zones), `households_skipped`
  (no rows, the households' columns), `person_windows` (every person's hours,
  all free, as 8-bit integers).
- **Size tables** (`school_destination_size`, `workplace_destination_size`):
  each zone's size per segment from `destination_choice_size_terms.csv`,
  computed as 1.x does (pandas' `dot` on the land use sorted by zone);
  scaled to the choosers only with shadow pricing on a full run.

Not supported yet: `override_hh_ids`, `column_map` in `annotate_tables`.

### free_parking

A logit on the persons with a workplace (`workplace_zone_id > -1`), seeing
their household, home zone land use and accessibility (1.x's
`persons_merged`), with MTC's preprocessor (the workplace's county). True
for `FREE_PARKING_ALT`, False for everyone else
(`components/free_parking.py`). The simple choice builder now takes an
optional chooser filter, preprocessor and output mapping, so this needed no
new choice code.

### cdap_simulate

The coordinated daily activity pattern (`components/cdap.py`), as 1.x:

1. **Rank** members: up to two workers, up to three children (by person
   type), then others in a random order to fill five places (one draw per
   person); then by rank group and age.
2. **Individual utilities** of M, N and H for the first five members.
3. **One joint choice per household**, a separate model per household size
   (1 to 5; 5 also for larger households): utilities are the members'
   individual utilities plus interaction terms generated from
   `cdap_interaction_coefficients.csv` (by activity and the members' person
   types, for 2 and 3 members, or all members). One draw per household.
4. **Extra members** (rank above 5) drawn from fixed proportions by person
   type: a second draw on the person's stream.
5. `cdap_activity` (a category with '', M, N, H), then the persons and
   households annotations.

The interaction specs, ranks and chooser tables follow 1.x's code; tests
compare them with 1.x's own functions (`tests/test_cdap.py`: same ranks
for 915 synthetic persons in 300 households, 28 of them with six or more members; same spec for
every household size, MTC's and MWCOG's coefficients). The pattern
utilities use the same matrix product as 1.x. In every test run some
households have six or more members (MTC's example data: 103 households,
315 persons ranked above 5), so the extra-member draw is exercised.

Not supported yet: `ADD_JOINT_TOUR_UTILITY` (no example uses it).

### mandatory_tour_frequency

A logit on persons with a mandatory day (`cdap_activity == 'M'`) over the
spec's alternatives (work1, work2, school1, school2, work_and_school),
written to `persons.mandatory_tour_frequency`, then the tours
(`components/mandatory_tour_frequency.py`, `components/tours.py`):

- one row per tour from `mandatory_tour_frequency_alternatives.csv`,
  numbered within type and within person (1.x's `create_tours`); for
  work_and_school, non-workers go to school first;
- destination (workplace or school zone), origin (home zone), household;
- **1.x's tour IDs**: 1.x lists every tour a person could make, across all
  tour models ("flavors": `work1`, `school2`, `escort3`, `j_shopping2`,
  `eat1_1`, ...) from the models' settings and alternatives files, sorts
  the list, and numbers a tour `person_id × (list length) + its place`.
  asim_core builds the same list from the same files with 1.x's rules and
  defaults (`canonical_tour_channels`), so tours get 1.x's IDs, which also
  name their random number streams in later steps. Every one of MTC's
  4,477 tours (full MTC: 16,540) has 1.x's ID.

Not supported yet: adding to an existing tours table (1.x does when school
escorting runs first; none of the examples does).

## Formula reading added

All found in these steps' specs, each translated to NumPy at load time and
checked against pandas doing what 1.x's formula does
(`tests/test_initialize.py`, `tests/test_translate.py`):

| 1.x formula | Example |
|---|---|
| counting lambdas, either table first | `_PERSON_COUNT('agep>=6 & agep<=15', persons, households)`; `_PERSON_TOUR_COUNT('tour_type==%s' % _Q('work'), persons, tours)` |
| text-building lambdas, worked out at load time (checked to only build text; no `eval`) | `_Q = lambda s: "'{}'".format(s)` |
| grouped sums of any expression on another table | `(persons.adult & persons.travel_active).astype(int).groupby(persons.household_id).sum().reindex(households.index)` |
| `other_than`, `Series.where`, `np.zeros(len(df))` | `other_than(persons.household_id, persons.age >= 18)` |
| `rng.lognormal_for_df(df, ..., broadcast=False)` (1.x's default) | household value of time |
| pandas' `&` / `|` precedence in queries and plain expressions | `ptype == 7 & age >= 6` |

## Profiles

<!-- profiles -->

## Reproducing

```bash
python asim_core/harness/run_milestone1.py --profiles none
python asim_core/harness/run_standalone.py --through mandatory_tour_scheduling --out asim_core/work/standalone_mtc
python asim_core/harness/run_milestone1.py --profiles none --skip-utility-check \
  --example-dir asim_core/work/mtc_full/prototype_mtc_full --asim1-out asim_core/work/mtc_full/asim1_10k_nne2 \
  --asim1-sharrow-out none --out asim_core/work/mtc_full/milestone_10k_b1
python -m pytest asim_core/tests/test_initialize.py asim_core/tests/test_cdap.py asim_core/tests/test_against_asim1.py
```
