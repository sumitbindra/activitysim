# Multiprocessing

asim_core runs a model in several processes the way ActivitySim 1.x does,
from the same settings, so a region's multiprocess configs work unchanged,
and its output files are 1.x's, byte for byte, in 1.x's row order.

## How it works

The plan is 1.x's own settings in `settings.yaml`: `multiprocess`,
`num_processes` and `multiprocess_steps` (MTC's `configs_mp` is the usual
example). Each `multiprocess_steps` entry is a group of steps:

- **A group without `slice`** runs in the main process (`mp_initialize`,
  `mp_summarize`).
- **A group sliced by households** runs in `num_processes` worker
  processes. As 1.x: process k gets every `num_processes`-th household
  starting at the k-th, in table order; persons, tours, trips and any table
  whose index or one of whose columns is the index of a table already split
  follow; tables that don't (land use, accessibility, size terms) go whole
  to every process. Afterwards the split tables are put back together in
  process order, and the others taken from process 0.
- **A group sliced by zones** (`mp_accessibility`) runs once in the main
  process (its values don't depend on the split), with its rows put in the
  order 1.x's split leaves them.

Checkpoints are named as 1.x names them in multiprocess runs (`init`, the
steps of the first group, `mp_initialize`, `mp_accessibility`,
`mp_households`, then the output steps), and the first step of each later
group re-saves every table, as 1.x's fresh process does, so
`final_checkpoints.csv` and the data dictionary match too.

**Worker processes are started fresh on every system** (`spawn`, the only
way Windows can start them): each rebuilds its steps from the configs
(`asim_core.model.build_steps`), opens the skim file itself (memory-mapped,
so the operating system keeps one copy for every process), reads its tables
from Parquet files the main process wrote, and writes its tables back the
same way. Nothing relies on memory inherited from the main process.
Workers keep the main process's thread settings for the numeric libraries,
as 1.x's do: the BLAS library's last digits depend on its thread count
(asim1 findings, 24), so on a 24-CPU Windows PC, workers with 12 threads
against 1.x's 24 made 3 of 39 files differ in logsums' last digits.
`--threads-per-process` sets them instead (e.g. CPUs / processes), which
can change last digits.

Code: `src/asim_core/multiprocess.py` (`plan_groups`, `slice_rules`,
`split_store`, `coalesce`, `run_in_groups`); harness:
`harness/run_multiprocess.py`.

```bash
# 1.x, then asim_core, MTC example in 2 processes (configs_mp)
python asim_core/harness/run_asim1.py --configs configs_mp configs --out asim_core/work/asim1_mp2
python asim_core/harness/run_multiprocess.py --configs configs_mp configs \
    --asim1-out asim_core/work/asim1_mp2 --asim1-single asim_core/work/asim1_legacy \
    --out asim_core/work/multiprocess_mtc
```

A region without multiprocess configs gets them as 1.x's examples do: a
folder with a `settings.yaml` that inherits the rest
(`inherit_settings: True`, `multiprocess: True`, `fail_fast: True`, which
1.x requires in multiprocess runs, `num_processes` and the four groups),
put first in the configs list (`asim_core/work/sandag/configs_mp4`).

## Results are 1.x's

| Region | Processes | Output files against 1.x in as many processes | Values by ID against 1.x in one process |
|---|---|---|---|
| MTC example | 2 (`configs_mp`) | 39 of 39 exact, row order included | the same but `sample_rate` (configs_mp's sample size) and 8 school logsums' last digits, as 1.x's own 2-process run |
| SANDAG one-zone | 4 | 16 of 16 exact | 1 tour's stop frequency, 2 trips and 20 logsums' last digits differ, as in 1.x's own 4-process run |
| SANDAG one-zone | 2 | (no 1.x run) | 177 of 177 columns exact |
| Full MTC, 50,000 households | 2, 4 | (no 1.x run) | 4 to 5 tours' stop frequency and their trips, 8 logsums' last digits |

Where a multiprocess run differs from a run in one process, 1.x differs the
same way: its results depend on the number of processes (asim1 findings,
23). MTC's stop frequency spec looks up household tour counts by person ID
(finding 19), which finds the household whose ID equals the person's ID only
if that household runs in the same process; and some logsums' last digits
depend on which rows are computed together. A run is reproducible for a
given number of processes.

## Time and memory

The times below were measured with CPUs / processes threads per worker (now `--threads-per-process`; on this 4-CPU machine the default makes no difference to the results' bits, see above). This machine has 4 CPUs and 15 GB. One asim_core process already uses
about 2.2 CPUs on average (NumPy's maths libraries run some work in
threads), so on it 4 processes can't be 4 times faster; machines with more
cores leave more room.

**SANDAG one-zone** (48,769 households), whole model:

| Run | All steps | Households steps (school location to trip mode choice) | Own memory |
|---|---|---|---|
| asim_core, one process | 64 to 69 s | about 58 s | 1.36 GB peak |
| asim_core, 2 processes | 64 s | 54 s (each process 50 to 52 s) | 0.84 GB per process |
| asim_core, 3 processes | 54 s | 43 s | 0.69 to 0.72 GB per process |
| asim_core, 4 processes | 44 s | 33 s (each 29 to 31 s) | 0.60 GB per process; main process 0.2 GB then, 0.9 GB writing the trip matrices |
| 1.x, 4 processes | 255 s | | |
| 1.x, one process | 341 s (as shipped) | | 3.7 GB |

(2 and 3 processes measured before the worker start-up savings below; 4
after.)

**Full MTC zones, 50,000 households**, runs alternating, skim file in
memory:

| Run | All steps | Households steps | Own memory |
|---|---|---|---|
| asim_core, one process | 175, 171 s | 144 s | 1.4 GB in the model steps; 2.5 GB peak (summarize) |
| asim_core, 2 processes | 109, 109 s (1.6 times faster) | 80 s (each process 78 s) | 0.84 to 0.87 GB per process and the main process's 0.65 GB while they run (about 2.4 GB in all); main 2.1 GB in summarize |
| asim_core, 4 processes | 76, 79 s (2.2 to 2.3 times faster) | 46 to 49 s (each 43 to 47 s) | 0.63 to 0.68 GB per process and the main process's 0.65 GB (about 3.3 GB in all); main 2.2 GB in summarize |

The skims (7.2 GB) are one memory-mapped file shared by every process, not
counted above. (This machine was slower than in the morning's runs, when
one process took 134 to 138 s; compare within the table only.)

Where the rest of the time goes, at 4 processes:

- **The main process's steps**: setting up (initialize, accessibility),
  then the output steps (trip matrices, tables, summaries) after the
  workers: 10 s of SANDAG's 44, 29 to 30 s of full MTC's 76 to 79. 1.x runs these in one
  process too.
- **Starting a worker**: about 3 to 5 s (Python and its libraries, building
  the steps from the configs, reading and writing its tables). Settings
  files are parsed once per process with PyYAML's C parser (building the
  steps: 5.2 s to 3.2 s), JAX is loaded only if one of its alternatives
  runs, and a worker builds only its own steps.
- **Steps that don't shrink with fewer households**: joint tour
  participation, trip scheduling and non-mandatory tour frequency repeat
  rounds a fixed number of times.

## Not done yet

- **Windows**: checked once on a 24-CPU Windows PC (`harness/windows_check.py`):
  one process exactly 1.x's (40 of 40 files); 2 processes 36 of 39 with
  workers at 12 threads against 1.x's 24, which led to keeping the main
  process's thread settings; to be rerun.
- **Shadow pricing** across processes (1.x shares the prices between
  processes; asim_core doesn't support shadow pricing yet).
- **The output steps in parallel** (each trip matrix file, summaries).
