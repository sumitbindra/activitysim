# Block 4: the rest of the model

Blocks 1 to 3 ported every model step MTC and ARC run. This block ports
what is left; so far MWCOG's four person-level steps, which MWCOG and
SEMCOG run and MTC and ARC don't:

| Step | Chooses | Writes |
|---|---|---|
| `work_from_home` | workers (`CHOOSER_FILTER_COLUMN_NAME`, default `is_worker`) | `work_from_home` (bool), `is_out_of_home_worker`, and `workplace_zone_id` -1 for those working from home |
| `transit_pass_subsidy` | every person (or those passing `CHOOSER_FILTER_COLUMN_NAME`) | `transit_pass_subsidy` (the alternative's position, 0 for others, int64) |
| `transit_pass_ownership` | every person | `transit_pass_ownership` (the alternative's position, int32) |
| `telecommute_frequency` | persons with a workplace (`workplace_zone_id > -1`) | `telecommute_frequency` (the alternative's name, "" for others; a category) |

With them, **every model step of the three example regions runs in
asim_core**, and MWCOG's whole model runs from the input files alone, as
MTC's and ARC's do. What is still taken from 1.x: the output and summary
steps (writing tables and trip matrices, the data dictionary, skim usage,
`summarize`).

## Summary

- **All four match 1.x exactly**, each from 1.x's saved state and end to
  end (`run_milestone1.py`: ALL EXACT), on MWCOG's 100-household run (the
  whole model) and on all 29,972 households (40,137 persons; 1.x's run
  stops at `mandatory_tour_frequency` because of the 1.x bug in
  [mwcog.md](mwcog.md), so this one ends after `cdap_simulate`).
- **MWCOG on its own** (`run_standalone.py`, from the input files, nothing
  from 1.x): 310 of 310 columns exact after trip mode choice (100
  households); 245 of 245 after CDAP with all households (6.9 million
  values).
- **Time**: 0.07 to 0.13 s each with all households, against 0.16 to
  0.22 s in 1.x (its own log).
- **Built from the simple choice component** (as `free_parking`), with
  one addition: the preprocessor can run on every row before the choosers
  are picked, so the filter column may be one it makes (1.x's
  `transit_pass_subsidy` does this; the others filter first).

## Details

- The four share how 1.x runs them (`components/person_choice.py`): its
  `simple_simulate` on `persons_merged` (each person with their household,
  home zone and its accessibility), the persons' random stream, the
  utility shift only with `skip_failed_choices` off, and an optional
  preprocessor (MWCOG's work from home: whether others in the household
  are preschool children or non-working adults, with `other_than`).
- Each output keeps 1.x's type: work from home is True for the
  `WORK_FROM_HOME_ALT` alternative; the subsidy goes through
  `fillna(0).astype(int)` in 1.x (int64), ownership keeps the choice's
  int32; telecommute frequency is a category of the alternatives and "".
- Not supported yet (reported before the run): `WORK_FROM_HOME_ITERATIONS`
  above 1 (1.x runs the model again with a constant adjusted towards a
  target share), annotations after these models, and a preprocessor on
  another table before the subsidy's filter. No example uses them.
- Tests: the four against MWCOG's 1.x run, and the subsidy with a filter
  column its preprocessor makes (whether someone else in the household
  works, checked against pandas over every person).
- The harness reads a 1.x run's households sample size from 1.x's log when
  the run has no `run_info.json` (MWCOG's all-households run stopped early
  and has none; asim_core had read 100 households instead of all).

## Reproducing

```bash
python asim_core/harness/run_milestone1.py --profiles none --example-dir activitysim/examples/prototype_mwcog \
  --asim1-out asim_core/work/mwcog/asim1_legacy --asim1-sharrow-out none --out asim_core/work/mwcog/milestone1_b4
python asim_core/harness/run_milestone1.py --profiles none --example-dir activitysim/examples/prototype_mwcog \
  --asim1-out asim_core/work/mwcog/asim1_legacy_all --asim1-sharrow-out none --out asim_core/work/mwcog/milestone1_b4_all
python asim_core/harness/run_standalone.py --example-dir activitysim/examples/prototype_mwcog \
  --asim1-out asim_core/work/mwcog/asim1_legacy --through trip_mode_choice --out asim_core/work/mwcog/standalone_b4
python asim_core/harness/run_standalone.py --example-dir activitysim/examples/prototype_mwcog \
  --asim1-out asim_core/work/mwcog/asim1_legacy_all --through cdap_simulate --out asim_core/work/mwcog/standalone_b4_all
python -m pytest asim_core/tests/test_against_asim1.py -k "mwcog or subsidy"
```

The 1.x runs are made as in [mwcog.md](mwcog.md) (`--households 0` for
all households).
