# Skims and `compute_accessibility`

## Summary

- **Skims are read through a memory-mapped cache, as 1.x's own skim cache
  does.**
  - OMX stays the format for exchanging skims with assignment software.
  - The OMX files are imported into a single NumPy file that every process
    shares in memory.
  - The same `network_los.yaml` settings as 1.x control it: which files,
    which time periods, where the cache goes, and whether to reuse it.
  - The cache is reused only if the OMX files haven't changed, so a model
    feedback loop can't pick up stale skims (see "Skim cache and feedback
    loops").
  - Importing needs PyTables; opening a cache doesn't.
- **Assignment specs are supported:** files where each row assigns a
  variable (`Description, Target, Expression`). This includes temporary
  variables (`_name`), skim lookups such as `skim_od[('SOVTOLL_TIME', 'AM')]`
  and functions such as `exp(...)`. Accessibility, preprocessors and
  annotations all use this format.
- **`compute_accessibility` is ported and matches 1.x exactly:** all 10
  columns for all 25 zones, bit for bit.
- **End to end, both ported steps are computed** and the other 32 are
  replayed from 1.x. Everything matches at every check. `auto_ownership`,
  computed from asim_core's own accessibility values, still matches 1.x
  exactly, including all utilities and probabilities.
- **Skim lookup is a swappable piece** (`skim_lookup`), with a plain NumPy
  reference version (since the skim experiment, a one-dimensional gather
  whose positions are shared by the skims a step looks up for the same zone
  pairs; see [skim_experiments.md](skim_experiments.md)).

## Matching results

| Check | Compared | Result |
|---|---|---|
| `compute_accessibility`, from 1.x's saved state | 10 columns × 25 zones | exact |
| ... in batches of 7 or 10 origins | same | exact |
| `auto_ownership`, from 1.x's saved state | 5,000 households | exact |
| `auto_ownership` utilities and probabilities | 25,000 each | bit-for-bit identical |
| End to end: accessibility and car ownership at the end of the run | 25 zones, 5,000 households | exact |

### What exact matching needed

- **Skims are 32-bit.** The OMX file stores 64-bit numbers; 1.x converts
  them to 32-bit when it loads them, and the import does the same.
- **Zone IDs equal matrix rows.** 1.x renumbers zones from 0
  (`recode_columns`), so zone *n* is matrix row *n*. The import records a
  zone ID for each row, so models that keep their original zone numbers
  work too.
- **Number types follow 1.x's rules.** Skim arithmetic stays 32-bit; jobs
  (64-bit integers) × a 32-bit decay factor becomes 64-bit.
- **Totals are added in destination order, in 64-bit,** as 1.x's loop does.
- **`log1p` comes from the C library.** This was the last difference found.
  Every variable compared (23, including all 10 outputs) matched 1.x for all
  625 zone pairs, but 4 of the 250 final values differed in the last digit. The
  cause: on this CPU, NumPy's vectorized `log1p` (which uses AVX-512)
  rounds differently from the C library `log1p` that 1.x's Numba code
  calls. The reference now uses the C library function, once per zone.

## Profiles

Each step run in a fresh process from 1.x's saved state, 5 times per engine;
medians. "Skims ready" is timed separately and is not part of the step.
All engines use the example's own settings (`read_skim_cache: False`, 1.x's
default): 1.x legacy reads the OMX file and asim_core imports it, on every
run. Sharrow reuses its own on-disk skim cache from the earlier run.

| Step | Engine | Step (s) | Extra memory (MB) | Skims ready (s) | Inside a full 1.x run (s) |
|---|---|---|---|---|---|
| compute_accessibility | asim_core | 0.003 | 0.9 | 0.30 | |
| compute_accessibility | 1.x legacy | 0.080 | 17.9 | 0.47 | 0.66 |
| compute_accessibility | 1.x sharrow | 0.130 | 17.6 | 1.73 | 0.58 |
| auto_ownership | asim_core | 0.018 | 5.3 | 0.32 | |
| auto_ownership | 1.x legacy | 0.156 | 18.5 | 0.42 | 0.12 |
| auto_ownership | 1.x sharrow | 0.186 | 26.1 | 1.74 | 0.11 |

With the cache turned on (`read_skim_cache: True`) and skims unchanged, both
engines reuse a cache. 4 runs each:

| Skims ready, cache on | 1.x legacy | asim_core |
|---|---|---|
| First run | 0.16 s | 0.30 s (imports, as there's no cache yet) |
| Later runs | 0.15–0.18 s (copies the whole cache into memory) | 0.007 s (checks the fingerprint, maps the cache) |

An earlier version of this page compared 1.x without a cache against
asim_core with one (0.46 s against 0.001 s). That wasn't a fair
comparison; the two tables above are.

In a full 1.x run, `compute_accessibility` is the first step that uses
skims, so its 0.66 seconds includes loading them. It also includes Numba
compiling 1.x's accumulation function on first use.

## Skim cache and feedback loops

A regional model runs in a loop, typically 4 to 6 global iterations: the
population synthesizer, ActivitySim, a commercial vehicle model, then
assignment, which writes new skims for the next iteration. OMX stays the
exchange format at both ends. Skims come in as OMX, and trip tables go out
as OMX once `write_trip_matrices` is ported (1.x writes them with
`openmatrix`). The NumPy file is a private cache in between, like 1.x's
`cache/cached_taz.mmap`.

The cache follows 1.x's settings in `network_los.yaml`:

| Setting | ActivitySim 1.x | asim_core |
|---|---|---|
| `taz_skims` | one OMX file, a list, or `omx:` / `file:` / `files:`; glob patterns; first data folder that has the file; a skim in two files: the later one wins, with a warning | the same |
| `skim_time_periods.labels` | skims for other time periods skipped, with a warning | the same |
| skim number type | float32 (fixed) | float32 by default |
| `read_skim_cache: False` (default) | reads the OMX files every run | imports the OMX files every run, into a new cache folder |
| `read_skim_cache: True` | reads `cache/cached_taz.mmap`; checks only its shape, so it can be stale | reuses the cache only if the OMX files' size and modification time are unchanged; otherwise imports again |
| `write_skim_cache` | writes the cache for later runs | the cache always exists, since runs read skims from it |
| `network_cache_dir` | where the cache goes (default `<output>/cache`) | the same |

Two safeguards for loops:

- **No stale skims.** Each cache records a fingerprint of the OMX files it
  came from: path, size and modification time. When assignment writes new
  skims, the fingerprint changes and the next run imports again, even with
  `read_skim_cache: True`.
- **No overwriting.** Windows won't let a program overwrite a file another
  process has memory-mapped, and most agencies run on Windows. So each import
  goes into a new folder (`asim_core_taz_<fingerprint>`, plus a timestamp
  when a fresh import is asked for). The folder is renamed into place only
  when complete, so a failed import never leaves a half-written cache.
  Folders no longer needed are removed. On Windows, one still open elsewhere
  is left and removed by a later run.

Prepare skims once per run and hand the folder to any worker processes;
workers open it directly.

### Scaling check (asim_core only, synthetic data)

Accessibility evaluates every zone pair, so its cost grows with the square
of the number of zones. This example has 25 zones. With 1,500 synthetic
zones (2.25 million pairs, 22 skims, a 198 MB skim file):

| Origins per batch | Time (s) | Extra memory (MB) |
|---|---|---|
| all at once | 0.81 | 593 |
| 100 | 0.60 | 235 |

About 200 MB of the batched figure is skim data read from the
memory-mapped file, which other processes would share. Results don't depend
on the batch size (tested). The same comparison against 1.x needs a real
full-size region.

## How it works

1. **Prepare skims for the run:**
   `prepare_skims_from_settings(configs, data_dirs, output_dir)` reads
   `network_los.yaml` and imports or reuses a cache folder holding
   `skims.npy` (all matrices stacked) and `skims.json` (names, zone IDs,
   number type, source fingerprint).
2. **At run time:** `Runner(..., skims=store)`. Components
   declare the skims they use, like table columns:
   `needs={"skims": ["SOVTOLL_TIME__AM", ...]}`. The startup check fails
   if one is missing; reading an undeclared skim is an error.
3. **The accessibility formulas** (`accessibility.csv`) are translated into
   one readable NumPy function, written to
   `work/milestone1/step_check/compute_accessibility/generated/`. For
   example:

   ```python
   # peak round trip distance: skim_od[('SOVTOLL_TIME', 'AM')] + skim_do[('SOVTOLL_TIME', 'PM')]
   v['_auPkTime'] = (s('skim_od', 'SOVTOLL_TIME__AM') + s('skim_do', 'SOVTOLL_TIME__PM'))
   # decay function: exp(_auPkTime * dispersion_parameter_automobile)
   v['_decay'] = np.exp((v['_auPkTime'] * -0.05))
   # auto peak retail: df.RETEMPN * _decay
   v['auPkRetail'] = (c['RETEMPN'] * v['_decay'])
   ```

4. **The component** evaluates that function for each batch of origins
   against every destination, then adds up and takes `log1p` per origin.
   Tracing a zone writes every destination's inputs and variables to a CSV.

## Translator coverage

Across all of the example's spec files, the translator now handles 2,155 of
2,905 formulas (74%), up from 61%. Utility formulas: 1,848 of 2,181 (85%).
Assignment formulas: 307 of 724 (42%). Most of what's left needs names that a
model supplies while it runs:

| Needs | Formulas (roughly) |
|---|---|
| names supplied by the model: coefficients used inside formulas, variables from preprocessors, the timetable `tt`, `skim_dict`, random draws `rng` | 289 |
| columns of other tables (`persons.age`, `trips.depart`, ...) | 180 |
| `reindex()` | 87 |
| other table subscripts | 44 |
| `.map()` / `.astype()` | 27 |
| other (`other_than()`, `.where`, `.apply`, helper functions, ...) | about 70 |

Of these, columns of other tables and `reindex()` are the joins that
preprocessors and annotations use. They're the natural next step for porting
the simple models.
