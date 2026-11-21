# Future experiments

Ideas measured enough to be worth trying, not done yet. Each would go in as
an option next to today's way, which stays the reference until the new one
is shown to be better in every respect.

## 1. A cache of the parsed input tables

**What it is.** Keep the households and persons tables (and land use)
parsed, in a binary file next to the inputs' fingerprints, as the skim cache
does for OMX files: parse each CSV once, then reuse the parsed columns while
the file is unchanged.

**Why.** `initialize_households` is the one ported step slower than 1.x on
full MTC: 4.9 s against 3.4 s at 10,000 households (one process,
`run_standalone.py`). Almost all of it is parsing all 2.9 million
households and 7.6 million persons with pandas, the same call as 1.x (2.7 s
at best on this machine, up to 4.7 s from run to run), then copying the
kept columns (about 0.5 s). Its memory peak, 1.25 GB, is the full parsed
files, before the sample is taken. The input checks parse the same files a
second time (with Arrow, about 0.6 s).

**What it could save.** Reading parsed columns from a binary file takes a
fraction of a second. With a sample, only the sampled households' rows and
their persons need reading, so the peak would follow the sample (about
0.1 to 0.2 GB at 10,000 households instead of 1.25 GB). Roughly 4 s per run
on full MTC.

**When it pays.** Every run reads unchanged inputs: repeated test and
experiment runs, sampled runs, and above all a model's feedback loop, where
each global iteration starts again from `initialize_households` with the
same input files (in agency model loops each global iteration typically runs
the whole model again). A single full-population run saves only the parse time
(all rows are needed, so memory is the same), a small share of a run that
takes an hour or more.

**Design sketch.**

- Parse with exactly 1.x's call (`inputs/reader.py`, `read_frame`), so the
  values are 1.x's to the last bit; then store each column as it came out
  (NumPy arrays: `.npy` per column, or one Parquet file), with the dtypes.
- Key the cache like the skim cache: the file's path, size and
  modification time, the table's settings (renames, kept columns, dtypes),
  and the reader's code version. A changed file or setting makes a new
  cache; an old one is never overwritten (safe with several processes).
- Read only the needed columns, and with a sample, the sampled rows:
  households by position (1.x's draw depends only on positions), persons by
  household ID (a sorted index of persons per household, kept in the cache,
  avoids scanning 7.6 million rows).

**The exactness risk.** Columns mixing numbers and text. pandas infers each
column's type per block of rows when parsing (its default), and 1.x keeps
what comes out: MWCOG's `NAICSP` has text values and missing values (`NaN`)
in one object column; in a larger file a column could hold numbers in some
blocks and text in others. The cache must give back exactly those Python
objects: store such columns as text plus a per-value type code, or keep them
out of the cache (read from the CSV) if they can't round-trip. Parquet would
turn the missing values into `None`, which is not what 1.x holds in memory
(the comparison treats both as missing, but formulas could tell them apart).

**How to test.** The initialize steps from the cache against 1.x on all
four regions (`run_milestone1.py`, and `run_standalone.py` for full MTC at
10,000 and 50,000 households); every column's dtype and every value, missing
values included, the same as from the CSV; a changed CSV (one value) makes
a new cache; two processes building the same cache at once.

## Smaller candidates noticed

- **Re-check skims only when the skims change.** The input checks keep one
  report per version of all inputs, so a change to the households file
  re-checks every skim too (6 s from the skim cache on full MTC, 25 s from
  the OMX file). Keeping the skim part of the report by the skim files'
  fingerprint (plus land use and the skim definitions) would skip that.
- **Give freed memory back between steps.** Freed memory can stay with the
  process: after the input checks it was about 1 GB until the C allocator
  was asked to return it (`asim_core/memory.py`, now called after the
  checks). Calling it after every step is cheap (milliseconds); measure
  whether it lowers what later steps hold.
- **CDAP at the full population.** 1.25 s at 50,000 households, about
  70 s projected for 2.9 million (1.x: about 420 s). Of the 1.25 s,
  0.53 s is random numbers (one seeded generator per row, needed for 1.x's
  numbers), 0.24 s ranking (pandas sorts, kept so ties go as in 1.x) and
  0.14 s the household chooser tables (pandas merges). The merges could be
  done with positions (each member's row by household and rank); the ranks
  only with care for ties.
- **Random numbers in general.** Matching 1.x means reseeding a generator
  for every row, about 3.3 µs per row; no faster exact way in pure NumPy is
  known. The `counter` alternative (not 1.x's numbers) is the fast option
  for those who accept different numbers; it grows in importance with
  steps that draw for every trip.
- **Joint tour participation with positions instead of pandas.** 1.8 s at
  50,000 households against 2.6 s in 1.x: the closest of the block 2 steps
  ([block2.md](block2.md)). Building the candidates (a merge of joint tours
  with household members) and each round's check of which parties are
  complete (a groupby over tour and composition) are 1.x's pandas code. Both
  could work on row positions (members by household, candidates by tour),
  keeping the draw order so the results stay 1.x's.
- **Trip scheduling in NumPy.** asim_core's port is 1.x's pandas code
  (a merge with the probability table per round of trips, up to 100
  attempts) and is no faster than 1.x (25 s against 21 s at 50,000
  households, [block3.md](block3.md)). The rounds could work on positions
  (each trip's probability row found once), keeping the draws in order.
- **Trip destination: measured, logsums are not worth deduplicating.** Full
  MTC, 50,000 households, skim file warm, timers around the step's parts
  (results still 1.x's): 80 s in all; sampling 44 s (55%: every zone's
  utility for each of 91,524 stops, 135 million rows, then 30 draws each),
  logsums 30 s (14 + 15 s, of which the trip mode choice preprocessor 5 s),
  choice 2 s, the rest 4 s. Of 2,094,154 logsum rows each way, only 42 (od)
  and 28 (dp) repeat another row's inputs: the person's value of time, the
  trip number and the zones make nearly every row different. Even the zone
  pairs with their period are 86 to 90% distinct, and 98% of stops have
  their own sample inputs (origin, leg destination, purpose, direction, tour
  mode). Deduplicating saves nothing here.
- **Trip destination's sampling: measured; skims are not the cost.** The
  same run, sampling only (39 s; 133 million stop × zone rows), with
  timers around its parts: building the rows (every column the stops carry,
  copied for every zone) 17.2 s (44%); utilities 18.1 s (46%), of which
  `size_terms.get` 10.3 s, skim fetches 3.5 s (zone → leg destination, a
  column of the matrix, 2.6 s; origin → zone, a row, 0.9 s) and formula
  arithmetic 4.2 s; probabilities, draws, picks and deduplication 3.6 s
  (picks only 1.0 s, so binary-search sampling would not help). Fetching
  whole rows and columns of skims would save at most about 3 s. Two exact
  changes would remove most of the rest: copy out only the columns the
  sample spec reads (about six, of every column the stops carry), and take
  each stop's sizes as its purpose's column of the size table (the zones are
  all zones in order; today 133 million zone IDs and purpose names are
  matched, twice per batch). **Done**: both changes made, results still
  exactly 1.x's (MTC, MWCOG, ARC, full MTC at 10,000 and 50,000
  households); trip destination at 50,000 households went from 68 to 71 s
  to 46 to 48 s (two runs each, skim file warm), its own memory peak from
  1.08 to 1.13 GB to 0.98 to 1.01 GB. They are the defaults of two seams,
  `sample_columns` (`spec_reads`) and `size_lookup` (`by_position`); the
  way it worked before is `all_columns` and `match`, also exactly 1.x's.
  Measured again with each setting (full MTC, 50,000 households, skim file
  warm): the default 43.7 s, own memory peak 991 MB; the way before 71.2
  and 75.4 s, 1,132 and 1,126 MB.
- **Read the skims a step uses before it runs.** The same trip destination
  run took 181 s with the skim file not in memory and 80 s with it read
  first (`cat` of the whole 6.7 GB file: 6 s). The difference is the first
  logsum pass (104 s against 14 s): random page faults on the ~100
  time-of-day skims it reads. A sequential read of the skims a run declares,
  at the start, would avoid this after a reboot or when other jobs have
  pushed the file out of memory.
