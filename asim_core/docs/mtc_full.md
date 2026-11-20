# Full MTC zones, sampled households

The examples so far used small zone systems (MTC's 25 zones, MWCOG's and
ARC's samples). This run uses the **full MTC region**: 1,475 skim zones,
826 skims and all 1,454 land use zones, with a sample of 10,000 of its
2.9 million households. It answers two questions: does asim_core still match
1.x when the arrays are large, and does a full zone system fit on this
machine (15 GB of memory, 4 CPUs)?

## Summary

- **Every choice is the same as 1.x's, and with one 1.x setting changed,
  every compared column is the same bit for bit** (279 of 279: all 8 steps
  asim_core computes, the 11 steps it replays, and the end of the run).
  With 1.x as it is, 13 columns of decimal values (the 10 accessibility
  columns and 3 destination choice logsums) differ in the last bits, at most
  6.3e-7, and those differences are carried into later steps' tables (36
  table columns in all). They come from 1.x: above a million elements,
  pandas hands arithmetic to numexpr, which adds up in a different order
  (1.x findings, #3). With numexpr turned off in 1.x
  (`harness/asim1_patches.py`, `no_numexpr`), 1.x's choices are unchanged
  and every value matches asim_core's.
- **It fits.** 1.x holds all skims in its own memory: 7.9 GB for the whole
  run, with a peak of 10.2 to 12.6 GB while it loads them (two runs). asim_core
  reads its skims from a memory-mapped file: its own memory grows by 60 to
  540 MB in a step; the operating system brings in the skim pages a step
  reads (up to 6.1 GB), shares them between processes and can drop them
  again.
- **The 8 steps asim_core computes take 18 s in one process, against 71 s in
  1.x's full run** (plus 1.x's 34 to 36 s to load the skims). Profiled one
  step at a time in fresh processes, asim_core's times depend mostly on
  whether the skim file is in the operating system's page cache (below).
- **Two fixes were needed in asim_core** (below): importing the OMX file ran
  out of memory, and the input checks treated the 21 external stations as an
  error.
- **Profiles** of the 8 steps on full zones are below. Reading skim pages
  is the largest part of asim_core's time in a fresh process, which led to
  the first experiment ([skim_experiments.md](skim_experiments.md)).

## Setup

- **Data:** `mtc_data_full.tar.zst` from the `activitysim-prototype-mtc`
  release listed in `activitysim/examples/external_example_manifest.yaml`,
  unpacked into its own folder: households, persons, land use and
  `skims.omx` (826 skims of 1,475 × 1,475).
- **Configs:** the prototype MTC configs, unchanged. The example folder
  `asim_core/work/mtc_full/prototype_mtc_full` links `configs` to them and
  `data` to the full data.
- **1.x runs:** `harness/run_asim1.py` with `households_sample_size:
  10000`, no chunking, the legacy engine; once as it is and once with the
  `no_numexpr` patch. 10,000 households have 26,579 persons and 32,020
  tours (16,540 mandatory, 14,823 non-mandatory, 657 joint).
- **asim_core:** `harness/run_milestone1.py` on that example: input checks,
  then each ported step from 1.x's saved state before it (step check), then
  the whole run with the other steps replayed (end to end).

## Results

| 1.x run | Columns compared | Same bit for bit | Choices | Largest difference |
|---|---|---|---|---|
| as it is | 279 | 243 | all the same | 6.3e-7 (an accessibility column) |
| with numexpr off | 279 | 279 | all the same | 0 |

The 36 columns that differ in the first run are 13 distinct columns:
`compute_accessibility`'s 10 (from `auPkRetail` to `nmTotal`), and
`school_location_logsum`, `workplace_location_logsum` and
`destination_logsum`, each also counted again in the tables of later steps.
The other comparisons (zones, times, modes, counts, every annotation) are the
same in both runs.

**Input checks** on the full population (2,875,192 households, 7,566,527
persons, 1,454 zones, 826 skims): no errors, 2 warnings, 21.6 s, of which
1.5 s for the tables and the rest for checking every skim value. One warning
is new: the skims have 21 more zones than the land use (zones 1,455 to
1,475 are MTC's external stations, with no land use). That is a warning now,
not an error; an error stays for land use zones that have no skims.

## Fixes

- **Importing the OMX file ran out of memory** (the process was stopped
  at 15 GB). asim_core opened every skim of the file before reading it, and
  PyTables gives each open array a chunk cache of up to 16 MB: 826 × 16 MB is
  13 GB, on top of the data. asim_core now opens, reads and closes one skim
  at a time (`skims.py`, `import_omx`), and the input checks do the same.
  The import now needs the skims plus a few MB.
- **Skim zones without land use** are a warning, as above.

## Profiles

Wall time in seconds, median of 3 runs, 10,000 households.

- **1.x, fresh process:** each step from 1.x's saved state before it
  (`harness/profile_step.py`); loading skims (about 40 s) is not timed.
- **1.x, full run:** the same step inside the full 1.x run.
- **asim_core, fresh process, cold / warm:** each step from 1.x's saved
  state before it, with the skim file dropped from the operating system's
  page cache first (cold), or read once first (warm); opening the skim file
  is not timed, reading its pages is (`--page-cache`; the reference
  implementations, from the skim experiment).
- **asim_core, one process:** the same steps in the end-to-end run, one
  process after the step checks, so the skim pages were already in it, as in
  any step of a whole run after the first ones that read them.

| Step | 1.x, fresh | 1.x, full run | asim_core cold | asim_core warm | asim_core, one process |
|---|---|---|---|---|---|
| compute_accessibility | 1.30 | 1.97 | 0.76 | 0.71 | 0.69 |
| school_location | 7.32 | 9.01 | 3.71 | 2.15 | 2.01 |
| workplace_location | 18.71 | 17.02 | 6.93 | 6.08 | 5.31 |
| auto_ownership_simulate | 0.31 | 0.16 | 0.06 | 0.06 | 0.03 |
| mandatory_tour_scheduling | 16.08 | 17.83 | 10.09 | 4.38 | 3.50 |
| non_mandatory_tour_destination | 18.30 | 15.85 | 6.42 | 5.30 | 4.90 |
| non_mandatory_tour_scheduling | 4.34 | 3.56 | 1.11 | 1.11 | 0.85 |
| tour_mode_choice_simulate | 5.99 | 5.14 | 6.14 | 0.87 | 0.66 |
| **8 steps** | **72.4** | **70.5** | **35.2** | **20.7** | **18.0** |

Auto ownership and non-mandatory scheduling read no skims, so their cold
and warm columns are the same runs. The cold column adds up the cost of
reading skim pages in every step; a whole run pays it once per page, the
first time a step reads it. Here that is about 2.5 s for the 6.4 GB the
mode choice steps can read (see the skim experiment); 1.x takes 34 to 36 s
to load its skims into memory.

On the small MTC example asim_core was about 10 to 15 times faster than 1.x
in the steps compared; on full zones, a fresh process with a warm cache is
1.8 to 7 times faster per step, and a cold one is as slow as 1.x in tour mode choice. The work that
grows with the zone system, reading skim values for many zone pairs, is now
the largest part of asim_core's time.

### Memory

| Step | 1.x holds before | 1.x extra | asim_core own extra | asim_core mapped skim pages |
|---|---|---|---|---|
| compute_accessibility | 8,008 | 195 | 378 | 245 |
| school_location | 8,034 | 825 | 438 | 1,767 |
| workplace_location | 7,974 | 984 | 474 | 1,828 |
| mandatory_tour_scheduling | 8,059 | 517 | 483 | 6,147 |
| non_mandatory_tour_destination | 8,016 | 309 | 391 | 1,307 |
| tour_mode_choice_simulate | 8,013 | 46 | 59 | 6,054 |

MB, medians. asim_core starts each step holding about 190 MB (the saved
state), and its own memory is sampled every 5 ms apart from the pages of
mapped files (the profiler's `split_memory`); the mapped skim pages are the
most it held during the step (warm runs). Those pages are the operating
system's file cache: shared by every process that reads the file, and
dropped when memory is needed elsewhere (they are read again if used). 1.x's
skims are its own memory, loaded before the step.
