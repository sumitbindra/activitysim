# An out-of-sample region: SANDAG (one zone system)

ActivitySim's SANDAG example (`activitysim/examples/placeholder_sandag`)
has two setups. The one-zone setup runs MTC's model files on SANDAG's own
data: `configs_1_zone` (SANDAG's settings, constants and skim periods)
layered on `prototype_mtc/configs`, with `data_1`: 90 zones (IDs 3093 to
3849, renumbered from 0 on reading), 48,769 households, 136,477 persons,
826 skims with SANDAG's periods (`EA` twice, then `AM`, `MD`, `PM`, `EV`).
None of asim_core was written or checked against it before, so it is an
out-of-sample test of the ported steps. The two-zone setup (`configs_2_zone`
on PSRC's model files, with zones inside zones) needs two-zone systems,
which asim_core doesn't have yet.

## Summary

- **Exactly 1.x's, on the first run**: every one of its 33 steps from 1.x's
  saved state, and end to end (`run_milestone1.py`: ALL EXACT, 854
  comparisons); the whole model from the input files
  (`run_standalone.py`): 219 of 219 columns (22.9 million values) and all
  16 output files (5 OMX trip matrix files, 7 final tables, checkpoints,
  data dictionary, skim usage).
- **Speed** (one process, all households): 91 s against 370 s in 1.x
  (numexpr off, as exact matching needs), 4.1 times faster. Trip destination
  16 s against 86 s, mandatory tour scheduling 11 s against 67 s; trip
  scheduling 17 s against 15 s (1.x's pandas code, kept for exactness).
- **Memory**: asim_core's own peak 2.4 GB, in `write_trip_matrices` (416,699
  trips with the preprocessor's 122 columns, put in one pandas table; the
  model steps peak at 1.35 GB, in trip destination). 1.x peaks at 3.7 GB, in
  mandatory tour scheduling. Building each matrix's table from just its
  columns would bring the output step down (an exact change, not made yet).

## What it took

1. **Settings inheritance in every settings file.** SANDAG's
   `constants.yaml` starts with `inherit_settings: True`: 1.x fills in the
   rest from MTC's `constants.yaml` and sets `TRANSIT_SCALE_FACTOR` to 1
   (MTC: 100). asim_core applied inheritance only to `settings.yaml`, so
   three steps could not be built (the person type constants missing). Now
   one reader (`spec/legacy.py`, `read_settings`) follows 1.x's
   `read_settings_file` rules for every settings file: `inherit_settings:
   True` (the next copy fills in), `inherit_settings: <file>`,
   `include_settings: <file>`. MTC, MWCOG and ARC use inheritance only in
   `settings.yaml`, so nothing changed for them.
2. **The data folder in the harness.** It took the data folder from the
   last configs folder, which here is MTC's: it would have run SANDAG's
   settings on MTC's households, with no error. Now `--data` names the
   example's data folders (`data_1`), and the default is the data folder of
   the region's own configs folder.
3. **A 1.x run to compare with**: all households, one process
   (`settings.yaml`, not `settings_mp.yaml`'s 28 processes), numexpr off.
   1.x's own SANDAG test uses 30 households and sorted output tables
   (`output_tables: sort`, not supported yet in asim_core), so it was not
   used.

The input checks found no errors; their 718 warnings are -999 in SANDAG's
transit skims, meaning no service. MTC's model files count transit as
available only with a positive time, so -999 is treated as unavailable.

## Reproducing

```bash
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/placeholder_sandag \
  --configs configs_1_zone ../prototype_mtc/configs --data data_1 --patch no_numexpr \
  --out asim_core/work/sandag/asim1_1zone
python asim_core/harness/run_milestone1.py --profiles none --skip-utility-check \
  --example-dir activitysim/examples/placeholder_sandag --configs configs_1_zone ../prototype_mtc/configs \
  --data data_1 --asim1-out asim_core/work/sandag/asim1_1zone --asim1-sharrow-out none \
  --out asim_core/work/sandag/milestone1
python asim_core/harness/run_standalone.py --example-dir activitysim/examples/placeholder_sandag \
  --configs configs_1_zone ../prototype_mtc/configs --data data_1 \
  --asim1-out asim_core/work/sandag/asim1_1zone --through write_tables --out asim_core/work/sandag/standalone
python -m pytest asim_core/tests -k "sandag or settings"
```
