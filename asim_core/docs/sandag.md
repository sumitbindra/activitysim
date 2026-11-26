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
  (numexpr off, as exact matching needs), 4.1 times faster; against 1.x
  with sharrow (after its 34-minute compile), 86 s against 203 s, 2.3 times
  (below). Trip destination
  16 s against 86 s, mandatory tour scheduling 11 s against 67 s; trip
  scheduling 17 s against 15 s (1.x's pandas code, kept for exactness).
  *Since then* trip scheduling takes about 1 s, with the same departures,
  and all the steps 64 to 69 s: 5.4 to 5.7 times faster than 1.x legacy
  (numexpr off) and about 3 times faster than sharrow ("Trip scheduling" in
  [experiments.md](experiments.md)).
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

## Against 1.x with sharrow

1.x has a second engine, sharrow, which compiles each model's formulas to
machine code (Numba). Run the same way (one process, all 48,769
households), the whole model's steps take:

| Run | Steps, s | Peak memory | Results |
|---|---|---|---|
| asim_core | 86 | 1.3 GB own (1.5 GB process) | exactly 1.x legacy's |
| 1.x sharrow (after its compile) | 203 | 3.1 GB | 150 fewer tours, 344 fewer trips; of the rest, 2 destinations, 1 tour start and 5 trip departures differ |
| 1.x legacy, as shipped | 341 | 3.7 GB | the same tours and trips as with numexpr off |
| 1.x legacy, numexpr off | 370 | 3.7 GB | asim_core's reference |

- **asim_core is 2.3 times faster than sharrow**, and 3.9 times faster than
  1.x legacy as shipped, with less than half of either's memory.
- **Sharrow's one-time compile**: about 34 minutes, on 100 households
  (`configs_sh_compile`, sharrow's test mode). It was killed once for want
  of memory in its last model step (trip mode choice; 13.9 GB, the
  compiled code of every earlier model still held, on this 15 GB machine);
  resumed after trip scheduling in a fresh process, trip mode choice
  compiled in 97 s at 1.7 GB. The compiled code is cached (144 MB) and
  reused while the formulas, constants and data layout stay the same;
  SANDAG's constants and skims differ from MTC's, so most of MTC's cached
  code could not be reused. asim_core has no compile step.
- **Where sharrow is faster than asim_core**: non-mandatory tour frequency
  (3.1 s against 4.3 s) and trip scheduling (14.8 s against 16.6 s; legacy
  14.0 s), the two steps left as candidates. *Since this table*, trip
  scheduling takes about 1 s with the same departures, and all the steps
  64 to 69 s instead of 85 to 86 s (memory unchanged): see "Trip
  scheduling" in [experiments.md](experiments.md). Sharrow is slower than legacy
  in non-mandatory tour scheduling (31.9 s against 22.1 s) and loads its
  skims into a dataset first (4.9 s in `initialize_landuse`).
- **Its results are not legacy's**: sharrow computes in 32-bit floats, so a
  few choices come out differently, and every later step follows from them.
- asim_core's own peak (1.3 GB) is now in `write_trip_matrices` and trip
  destination, about equal, with the trip matrices change.

| Step | asim_core s | 1.x sharrow s | 1.x legacy s | 1.x legacy, numexpr off s | asim_core own MB | asim_core process MB | 1.x sharrow MB | 1.x legacy MB | 1.x legacy, numexpr off MB |
|---|---|---|---|---|---|---|---|---|---|
| **Setup** | | | | | | | | | |
| initialize_landuse | 0.01 | 4.94 | 0.52 | 0.67 | 176 | 285 | 414 | 404 | 404 |
| initialize_households | 0.49 | 1.04 | 1.08 | 1.20 | 337 | 447 | 675 | 662 | 659 |
| compute_accessibility | 0.01 | 0.40 | 0.34 | 0.36 | 327 | 439 | 690 | 681 | 676 |
| *Setup, total* | *0.5* | *6.4* | *1.9* | *2.2* | | | | | |
| **Long-term choices** | | | | | | | | | |
| school_location | 3.11 | 7.27 | 20.08 | 21.39 | 949 | 1,084 | 1,840 | 3,076 | 3,074 |
| workplace_location | 5.01 | 7.22 | 28.30 | 30.04 | 959 | 1,093 | 1,441 | 2,068 | 2,066 |
| auto_ownership_simulate | 0.04 | 0.30 | 0.31 | 0.35 | 716 | 850 | 1,232 | 964 | 950 |
| free_parking | 0.05 | 0.44 | 0.42 | 0.49 | 711 | 846 | 1,237 | 940 | 936 |
| *Long-term choices, total* | *8.2* | *15.2* | *49.1* | *52.3* | | | | | |
| **Daily pattern and tours** | | | | | | | | | |
| cdap_simulate | 0.57 | 4.58 | 4.87 | 5.05 | 759 | 893 | 1,260 | 931 | 961 |
| mandatory_tour_frequency | 0.27 | 0.75 | 0.95 | 0.93 | 783 | 926 | 1,269 | 932 | 921 |
| mandatory_tour_scheduling | 10.39 | 52.50 | 60.73 | 67.30 | 1,102 | 1,240 | 2,763 | 3,662 | 3,661 |
| joint_tour_frequency | 0.31 | 0.81 | 0.71 | 0.73 | 785 | 922 | 1,489 | 1,388 | 1,199 |
| joint_tour_composition | 0.06 | 0.27 | 0.22 | 0.21 | 733 | 871 | 1,487 | 1,388 | 1,173 |
| joint_tour_participation | 4.69 | 5.19 | 5.58 | 5.64 | 758 | 902 | 1,489 | 1,388 | 1,174 |
| joint_tour_destination | 0.34 | 2.24 | 4.22 | 4.30 | 783 | 921 | 1,490 | 1,388 | 1,175 |
| joint_tour_scheduling | 0.26 | 1.39 | 1.64 | 1.73 | 733 | 871 | 1,492 | 1,413 | 1,175 |
| non_mandatory_tour_frequency | 4.33 | 3.08 | 11.87 | 15.43 | 1,095 | 1,233 | 1,505 | 1,651 | 1,612 |
| non_mandatory_tour_destination | 6.71 | 9.39 | 40.03 | 45.52 | 1,170 | 1,308 | 1,977 | 2,473 | 2,456 |
| non_mandatory_tour_scheduling | 4.24 | 31.89 | 22.07 | 24.06 | 911 | 1,050 | 3,013 | 2,911 | 2,884 |
| tour_mode_choice_simulate | 1.50 | 4.64 | 9.55 | 9.83 | 1,011 | 1,149 | 1,547 | 1,453 | 1,326 |
| atwork_subtour_frequency | 0.21 | 0.42 | 0.45 | 0.47 | 937 | 1,076 | 1,546 | 1,355 | 1,178 |
| atwork_subtour_destination | 0.72 | 1.92 | 4.60 | 4.83 | 1,169 | 1,308 | 1,523 | 1,712 | 1,610 |
| atwork_subtour_scheduling | 0.24 | 0.69 | 1.22 | 1.14 | 938 | 1,077 | 1,523 | 1,353 | 1,176 |
| atwork_subtour_mode_choice | 0.17 | 0.55 | 1.01 | 1.03 | 938 | 1,077 | 1,524 | 1,353 | 1,176 |
| *Daily pattern and tours, total* | *35.0* | *120.3* | *169.7* | *188.2* | | | | | |
| **Trips** | | | | | | | | | |
| stop_frequency | 0.97 | 2.07 | 1.91 | 2.52 | 1,002 | 1,141 | 1,560 | 1,399 | 1,358 |
| trip_purpose | 0.14 | 0.73 | 0.52 | 0.63 | 1,002 | 1,141 | 1,569 | 1,361 | 1,159 |
| trip_destination | 14.12 | 27.67 | 80.66 | 85.75 | 1,324 | 1,463 | 2,003 | 2,416 | 2,416 |
| trip_purpose_and_destination | 0.16 | 0.19 | 0.17 | 0.23 | 957 | 1,095 | 1,600 | 1,441 | 1,357 |
| trip_scheduling | 16.58 | 14.82 | 14.04 | 15.00 | 993 | 1,132 | 1,603 | 1,442 | 1,358 |
| trip_mode_choice | 4.18 | 7.75 | 15.30 | 15.74 | 1,272 | 1,410 | 1,629 | 1,735 | 1,670 |
| *Trips, total* | *36.2* | *53.2* | *112.6* | *119.9* | | | | | |
| **Outputs** | | | | | | | | | |
| write_data_dictionary | 0.08 | 0.75 | 0.66 | 0.65 | 1,051 | 1,190 | 1,889 | 1,749 | 1,704 |
| track_skim_usage | 0.00 | 0.03 | 0.03 | 0.03 | 1,051 | 1,190 | 1,884 | 1,743 | 1,698 |
| write_trip_matrices | 5.81 | 6.21 | 5.84 | 5.76 | 1,330 | 1,468 | 3,131 | 2,875 | 2,915 |
| write_tables | 0.63 | 0.64 | 0.64 | 0.66 | 993 | 1,132 | 1,875 | 1,709 | 1,668 |
| *Outputs, total* | *6.5* | *7.6* | *7.2* | *7.1* | | | | | |
| **All steps** | **86.4** | **202.8** | **340.5** | **369.7** | **1,330** | **1,468** | **3,131** | **3,662** | **3,661** |

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
# 1.x with sharrow: compile once (100 households), then the timed run; 1.x legacy as shipped
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/placeholder_sandag --data data_1 \
  --configs ../prototype_mtc/configs_sh_compile configs_1_zone ../prototype_mtc/configs --out asim_core/work/sandag/asim1_sharrow_compile
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/placeholder_sandag --data data_1 \
  --configs ../prototype_mtc/configs_sh configs_1_zone ../prototype_mtc/configs --out asim_core/work/sandag/asim1_sharrow
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/placeholder_sandag --data data_1 \
  --configs configs_1_zone ../prototype_mtc/configs --out asim_core/work/sandag/asim1_legacy_shipped
```
