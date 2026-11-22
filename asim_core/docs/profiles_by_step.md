# Time and memory, step by step: asim_core and 1.x

Two regions, every step of the model and the output steps, one process
each, all of asim_core's results exactly 1.x's (every column and every
output file):

- **Full MTC zones** (1,475 zones, 826 skims, 7.2 GB), 50,000 households:
  166 s against 603 s in 1.x, **3.6 times faster**.
- **SANDAG one-zone** (90 zones, MTC's model files on SANDAG's data), all
  48,769 households: 91 s against 370 s, **4.1 times faster**.
- (Full MTC, 10,000 households: 53 s against 249 s, 4.7 times.)

How it was measured: asim_core's whole model from the input files
(`run_standalone.py`, the defaults of today: vectorized random draws,
sampling copying only the columns its spec reads), skim file already in
memory; 1.x's own run of the whole model (`run_asim1.py`, no resume, one
process, numexpr off as exact matching needs; with numexpr on 1.x is
somewhat faster and its results differ, finding #3). Same machine (4 CPUs,
15 GB), each run alone on it.

Memory, three columns:

- **asim_core own**: the memory asim_core allocates (its tables, the
  step's working arrays). The skims are not in it: asim_core maps the skim
  file, whose pages the operating system shares between processes and can
  drop when memory is short.
- **process peak**: everything the process holds during the step,
  including the skim file's pages read so far (asim_core) or the skims 1.x
  loads into its own memory (8 GB for full MTC, from `initialize_landuse`
  on).
- **added by the step**: the peak during the step minus the memory before
  it (asim_core: own memory; 1.x: process).

## What the tables say

- **Faster in every model step but trip scheduling** (0.8 to 0.9 times:
  1.x's pandas code, ported as is for exactness, and the one candidate left
  for a NumPy rewrite). Joint tour participation (1.2 times) and trip purpose
  and destination (1.0) are about even; the output steps that write files
  are about even too (writing the tables: PyArrow in both).
- **Most of the time saved** at full MTC: trip destination (165 → 40 s),
  workplace location (82 → 19 s), non-mandatory tour destination (75 →
  16 s), mandatory tour scheduling (72 → 18 s), loading the skims (1.x:
  34 s in `initialize_landuse`; asim_core: the cached skim file is mapped,
  nothing read up front).
- **Memory**: asim_core's own peak is 1.4 GB in the model steps (trip
  destination) and 2.7 GB with the output steps (`summarize`,
  `write_trip_matrices`); 1.x holds 8 GB of skims all run and peaks at
  12.8 GB (workplace location). Per step, asim_core adds less in every
  large step (school location 0.6 GB against 4.2 GB, workplace location 0.4
  against 4.7, mandatory tour scheduling 0.4 against 2.9, trip destination
  0.4 against 3.2), more in `initialize_households` (1.1 against 0.9 GB) and
  `write_trip_matrices` (1.6 against 1.1 GB: the trips and the
  preprocessor's 122 columns in one pandas table).
- **The output steps set asim_core's peak**, in both regions, in these
  runs.

**Since these tables** (the `trip_matrix_table` seam, `experiments.md`):
the trip matrices step keeps only the columns some matrix reads, builds
each matrix's table from its own columns and writes each matrix once made.
Memory the step adds, from 1.x's saved state, two runs each, every OMX
file exactly 1.x's with both settings:

| Region | before (`one_table`, now behind the flag) | now (`per_matrix`) |
|---|---|---|
| SANDAG, 48,769 households | 1,476 / 1,478 MB | 464 / 464 MB |
| Full MTC, 50,000 households | 1,570 / 1,572 MB | 869 / 856 MB |

Time about the same (SANDAG 5.6 / 5.7 s against 6.4 / 6.9 s; full MTC 13.1
/ 13.2 s against 12.4 / 12.2 s). Full MTC's matrices are 1,475 × 1,475 (17
MB each, 23 a file), so writing each once made matters there; what is left
is mostly the preprocessor's 122 columns. These are from 1.x's saved state
(the step starts with 0.3 GB); in a whole run it starts with about 1 GB, so
it adds to that, and at full MTC `summarize` (2.7 GB) is still the run's
peak: not measured again here.

## Full MTC zones, 50,000 households

| Step | asim_core s | 1.x s | faster | asim_core own peak MB | asim_core process peak MB | 1.x process peak MB | added by the step: asim_core own MB | 1.x MB |
|---|---|---|---|---|---|---|---|---|
| **Setup** | | | | | | | | |
| initialize_landuse | 0.01 | 34.26 | 2362.8x | 162 | 265 | 11,836 | 3 | 11,500 |
| initialize_households | 3.32 | 4.45 | 1.3x | 1,266 | 1,374 | 8,802 | 1,103 | 925 |
| compute_accessibility | 0.60 | 1.60 | 2.7x | 748 | 1,032 | 8,151 | 398 | 201 |
| *Setup, total* | *3.9* | *40.3* | *10.2x* | | | | | |
| **Long-term choices** | | | | | | | | |
| school_location | 8.32 | 31.98 | 3.8x | 931 | 2,875 | 12,184 | 594 | 4,194 |
| workplace_location | 19.36 | 82.30 | 4.3x | 1,068 | 3,835 | 12,751 | 442 | 4,696 |
| auto_ownership_simulate | 0.05 | 0.39 | 8.5x | 754 | 3,522 | 8,053 | 0 | 2 |
| free_parking | 0.05 | 0.50 | 10.4x | 754 | 3,522 | 8,056 | 0 | 4 |
| *Long-term choices, total* | *27.8* | *115.2* | *4.1x* | | | | | |
| **Daily pattern and tours** | | | | | | | | |
| cdap_simulate | 0.51 | 5.48 | 10.6x | 800 | 3,568 | 8,064 | 46 | 12 |
| mandatory_tour_frequency | 0.31 | 1.02 | 3.3x | 788 | 3,556 | 8,061 | 59 | 5 |
| mandatory_tour_scheduling | 17.85 | 72.03 | 4.0x | 1,112 | 7,386 | 10,962 | 381 | 2,906 |
| joint_tour_frequency | 0.34 | 0.70 | 2.1x | 782 | 7,063 | 8,462 | 0 | 0 |
| joint_tour_composition | 0.07 | 0.24 | 3.5x | 756 | 7,038 | 8,460 | 0 | 0 |
| joint_tour_participation | 1.46 | 1.70 | 1.2x | 756 | 7,038 | 8,460 | 0 | 1 |
| joint_tour_destination | 0.86 | 5.60 | 6.5x | 813 | 7,094 | 8,460 | 57 | 0 |
| joint_tour_scheduling | 0.22 | 1.58 | 7.2x | 782 | 7,064 | 8,486 | 26 | 25 |
| non_mandatory_tour_frequency | 4.65 | 14.13 | 3.0x | 1,165 | 7,447 | 8,731 | 383 | 247 |
| non_mandatory_tour_destination | 16.05 | 74.75 | 4.7x | 1,165 | 7,446 | 10,233 | 266 | 1,773 |
| non_mandatory_tour_scheduling | 3.80 | 19.58 | 5.2x | 939 | 7,221 | 9,618 | 32 | 1,159 |
| tour_mode_choice_simulate | 2.15 | 10.53 | 4.9x | 1,034 | 7,316 | 8,487 | 124 | 7 |
| atwork_subtour_frequency | 0.25 | 0.47 | 1.9x | 936 | 7,219 | 8,489 | 0 | 3 |
| atwork_subtour_destination | 2.96 | 13.75 | 4.6x | 1,202 | 7,484 | 9,825 | 266 | 1,339 |
| atwork_subtour_scheduling | 0.38 | 1.51 | 4.0x | 952 | 7,235 | 8,486 | 1 | 0 |
| atwork_subtour_mode_choice | 0.26 | 1.21 | 4.7x | 952 | 7,235 | 8,486 | 0 | 0 |
| *Daily pattern and tours, total* | *52.1* | *224.3* | *4.3x* | | | | | |
| **Trips** | | | | | | | | |
| stop_frequency | 1.15 | 1.94 | 1.7x | 1,064 | 7,355 | 8,490 | 112 | 4 |
| trip_purpose | 0.17 | 0.63 | 3.6x | 995 | 7,278 | 8,496 | 1 | 7 |
| trip_destination | 39.76 | 164.95 | 4.1x | 1,432 | 7,719 | 11,726 | 438 | 3,237 |
| trip_purpose_and_destination | 0.21 | 0.22 | 1.0x | 1,028 | 7,315 | 8,464 | 0 | 0 |
| trip_scheduling | 18.47 | 15.61 | 0.8x | 1,073 | 7,360 | 8,465 | 45 | 1 |
| trip_mode_choice | 5.46 | 18.57 | 3.4x | 1,418 | 7,705 | 8,835 | 424 | 372 |
| *Trips, total* | *65.2* | *201.9* | *3.1x* | | | | | |
| **Outputs** | | | | | | | | |
| write_data_dictionary | 0.11 | 0.71 | 6.4x | 1,062 | 7,349 | 8,721 | 74 | 257 |
| track_skim_usage | 0.00 | 0.04 | 54.6x | 988 | 7,275 | 8,702 | 0 | 0 |
| write_trip_matrices | 12.02 | 13.09 | 1.1x | 2,607 | 8,901 | 9,766 | 1,620 | 1,064 |
| write_tables | 1.16 | 1.12 | 1.0x | 1,987 | 8,281 | 8,771 | 0 | 8 |
| summarize | 3.93 | 6.26 | 1.6x | 2,661 | 8,959 | 9,663 | 714 | 927 |
| *Outputs, total* | *17.2* | *21.2* | *1.2x* | | | | | |
| **All steps** | **166.3** | **602.9** | **3.6x** | **2,661** | **8,959** | **12,751** | | |

## SANDAG one-zone, 48,769 households

| Step | asim_core s | 1.x s | faster | asim_core own peak MB | asim_core process peak MB | 1.x process peak MB | added by the step: asim_core own MB | 1.x MB |
|---|---|---|---|---|---|---|---|---|
| **Setup** | | | | | | | | |
| initialize_landuse | 0.01 | 0.67 | 46.6x | 176 | 289 | 404 | 0 | 69 |
| initialize_households | 0.61 | 1.20 | 2.0x | 338 | 451 | 659 | 161 | 256 |
| compute_accessibility | 0.01 | 0.36 | 65.1x | 324 | 439 | 676 | 0 | 39 |
| *Setup, total* | *0.6* | *2.2* | *3.6x* | | | | | |
| **Long-term choices** | | | | | | | | |
| school_location | 3.13 | 21.39 | 6.8x | 946 | 1,083 | 3,074 | 622 | 2,528 |
| workplace_location | 5.18 | 30.04 | 5.8x | 1,012 | 1,150 | 2,066 | 335 | 1,179 |
| auto_ownership_simulate | 0.04 | 0.35 | 8.1x | 744 | 881 | 950 | 0 | 104 |
| free_parking | 0.05 | 0.49 | 10.5x | 741 | 879 | 936 | 0 | 89 |
| *Long-term choices, total* | *8.4* | *52.3* | *6.2x* | | | | | |
| **Daily pattern and tours** | | | | | | | | |
| cdap_simulate | 0.59 | 5.05 | 8.6x | 767 | 905 | 961 | 26 | 114 |
| mandatory_tour_frequency | 0.28 | 0.93 | 3.4x | 769 | 908 | 921 | 4 | 68 |
| mandatory_tour_scheduling | 10.73 | 67.30 | 6.3x | 1,098 | 1,238 | 3,661 | 328 | 2,809 |
| joint_tour_frequency | 0.32 | 0.73 | 2.2x | 828 | 969 | 1,199 | 0 | 0 |
| joint_tour_composition | 0.06 | 0.21 | 3.3x | 804 | 945 | 1,173 | 0 | 0 |
| joint_tour_participation | 4.66 | 5.64 | 1.2x | 804 | 946 | 1,174 | 0 | 1 |
| joint_tour_destination | 0.31 | 4.30 | 14.0x | 804 | 945 | 1,175 | 0 | 1 |
| joint_tour_scheduling | 0.24 | 1.73 | 7.3x | 830 | 971 | 1,175 | 26 | 0 |
| non_mandatory_tour_frequency | 4.36 | 15.43 | 3.5x | 1,121 | 1,263 | 1,612 | 292 | 438 |
| non_mandatory_tour_destination | 6.98 | 45.52 | 6.5x | 1,187 | 1,329 | 2,456 | 265 | 1,317 |
| non_mandatory_tour_scheduling | 4.24 | 24.06 | 5.7x | 928 | 1,070 | 2,884 | 0 | 1,747 |
| tour_mode_choice_simulate | 1.79 | 9.83 | 5.5x | 1,027 | 1,169 | 1,326 | 99 | 173 |
| atwork_subtour_frequency | 0.22 | 0.47 | 2.1x | 954 | 1,095 | 1,178 | 0 | 2 |
| atwork_subtour_destination | 0.79 | 4.83 | 6.1x | 1,186 | 1,328 | 1,610 | 233 | 434 |
| atwork_subtour_scheduling | 0.28 | 1.14 | 4.1x | 953 | 1,095 | 1,176 | 0 | 0 |
| atwork_subtour_mode_choice | 0.19 | 1.03 | 5.4x | 953 | 1,095 | 1,176 | 0 | 0 |
| *Daily pattern and tours, total* | *36.0* | *188.2* | *5.2x* | | | | | |
| **Trips** | | | | | | | | |
| stop_frequency | 1.02 | 2.52 | 2.5x | 1,027 | 1,179 | 1,358 | 73 | 182 |
| trip_purpose | 0.15 | 0.63 | 4.1x | 1,009 | 1,151 | 1,159 | 1 | 6 |
| trip_destination | 15.67 | 85.75 | 5.5x | 1,349 | 1,491 | 2,416 | 340 | 1,261 |
| trip_purpose_and_destination | 0.16 | 0.23 | 1.4x | 993 | 1,134 | 1,357 | 0 | 0 |
| trip_scheduling | 16.93 | 15.00 | 0.9x | 1,015 | 1,157 | 1,358 | 23 | 1 |
| trip_mode_choice | 4.39 | 15.74 | 3.6x | 1,297 | 1,439 | 1,670 | 352 | 313 |
| *Trips, total* | *38.3* | *119.9* | *3.1x* | | | | | |
| **Outputs** | | | | | | | | |
| write_data_dictionary | 0.09 | 0.65 | 6.9x | 1,017 | 1,159 | 1,704 | 0 | 315 |
| track_skim_usage | 0.00 | 0.03 | 23.6x | 1,017 | 1,159 | 1,698 | 0 | 0 |
| write_trip_matrices | 6.64 | 5.76 | 0.9x | 2,364 | 2,517 | 2,915 | 1,347 | 1,217 |
| write_tables | 0.67 | 0.66 | 1.0x | 1,017 | 1,160 | 1,668 | 0 | 0 |
| *Outputs, total* | *7.4* | *7.1* | *1.0x* | | | | | |
| **All steps** | **90.8** | **369.7** | **4.1x** | **2,364** | **2,517** | **3,661** | | |

asim_core's skims ready in 0.4 s before the first step (not in the table); 1.x loads its skims in initialize_landuse (in the table).

## Reproducing

```bash
# 1.x: the whole model, no resume (full MTC: delete <out>/cache/cached_taz.mmap after; 6.7 GB)
python asim_core/harness/run_asim1.py --example-dir asim_core/work/mtc_full/prototype_mtc_full \
  --out asim_core/work/mtc_full/asim1_50k_fresh --households 50000 --patch no_numexpr
python asim_core/harness/run_standalone.py --example-dir asim_core/work/mtc_full/prototype_mtc_full \
  --asim1-out asim_core/work/mtc_full/asim1_50k_fresh --read-skim-cache true --through summarize \
  --out asim_core/work/mtc_full/standalone_50k_fresh
# SANDAG: see sandag.md; the tables: profile.csv of each run against 1.x's asim_core_profile.csv
```

Earlier documents compared full MTC at 50,000 households with 806 s for
1.x: that run was made in three parts, resumed twice, and 1.x reloads its
skims at each resume (44 s and 48 s, 8 to 13 GB, in the steps after the
resumes). A run of the whole model without a resume takes 603 s.
