# asim_core

A minimal, component-based core for ActivitySim, built to be checked step by
step against ActivitySim 1.x and to measure runtime and memory for every step.

The core knows how to hold data, run steps, save results, compare against
1.x and measure. It knows nothing about travel models: every model (auto
ownership, mode choice, park-and-ride, ...) is a **component** that plugs in.
A region chooses its components; the runner checks before starting that
every input each component needs is provided by an earlier step.

Inside a step, the pieces are swappable too (**seams**): turning formulas into
utilities, probabilities, choosing, random numbers, lookups. Each has a plain
NumPy reference implementation that matches 1.x exactly; alternatives are
picked by a setting and tested with the experiment harness. Speed-ups come
from those experiments. Sharrow is not used; 1.x with sharrow is only a
baseline in the reports.

Status: forty components are ported and match 1.x exactly:
`initialize_landuse` and `initialize_households` (reading, household
sampling, annotations, size tables), `compute_accessibility` (with skims),
`school_location` and `workplace_location` (sampling, logsums, annotations
after the model), `auto_ownership` (multinomial or nested logit),
`free_parking`, `cdap_simulate` (the joint household day pattern),
`mandatory_tour_frequency` (and the tours, with 1.x's tour IDs),
`mandatory_tour_scheduling` and `non_mandatory_tour_scheduling`
(timetables, rounds of tours, one utility per tour × departure-return
option, mode choice logsums computed once per distinct input),
`non_mandatory_tour_destination` (sampling of alternatives, size terms),
`tour_mode_choice_simulate` (nested logit, purpose segments, preprocessor,
time-of-day skims), the five joint tour steps (frequency, composition,
participation in rounds, destination, scheduling across participants),
`non_mandatory_tour_frequency` (with the extension draws), the four
at-work subtour steps (frequency, destination, scheduling within the work
tour, mode choice from the workplace), and the trip steps: stop frequency
(and the trips, with 1.x's trip IDs), trip purpose, trip destination (two
mode choice logsums per sampled zone), trip purpose and destination, trip
scheduling (MTC, MWCOG) or trip scheduling choice and trip departure choice
(ARC), and trip mode choice; and MWCOG's own four: work from home, transit
pass subsidy and ownership, telecommute frequency; and the output steps:
trip matrices (OMX), the final tables, the data dictionary, skim usage and
summaries, every file exactly 1.x's. They were checked on three regions' example
models: MTC, MWCOG and ARC (all 24,061 households), and on MTC's full zone
system. **Every model step of the three regions runs in asim_core**, and
each region's whole model runs from the input files alone
([docs/block1.md](docs/block1.md), [docs/block2.md](docs/block2.md),
[docs/block3.md](docs/block3.md), [docs/block4.md](docs/block4.md)), from the
input files to the output files, with nothing taken from 1.x. An
out-of-sample region, SANDAG's one-zone example (MTC's model files on
SANDAG's data, 48,769 households; nothing in asim_core was checked against
it before), was exactly 1.x's on the first run: every step, all 16 output
files, 4.1 times faster ([docs/sandag.md](docs/sandag.md)).
Every other step can be
**replayed** from a 1.x run, so the model runs end to end today. Results: [docs/milestone1.md](docs/milestone1.md),
[docs/block1.md](docs/block1.md), [docs/block2.md](docs/block2.md), [docs/block3.md](docs/block3.md),
[docs/skims_and_accessibility.md](docs/skims_and_accessibility.md),
[docs/mwcog.md](docs/mwcog.md), [docs/arc.md](docs/arc.md),
[docs/tour_mode_choice.md](docs/tour_mode_choice.md),
[docs/non_mandatory_scheduling.md](docs/non_mandatory_scheduling.md),
[docs/mandatory_scheduling.md](docs/mandatory_scheduling.md),
[docs/destination_choice.md](docs/destination_choice.md),
[docs/location_choice.md](docs/location_choice.md).

Inputs are checked before the first step
([docs/input_checks.md](docs/input_checks.md)): every table and column is
defined (standard columns in asim_core, a region's own in its
`input_tables.yaml`), and bad data (negative employment, unknown zones,
household sizes that don't match their persons, missing skim values,
segments with no size) stops the run with one report. 6.7 million
persons are read and checked in about 2 s.

Models can also be written in asim_core's own model files
([docs/model_files.md](docs/model_files.md)): one YAML file per model, CSV
tables, and one formula language in which every column names its table
(`person.age`, `land_use.TOPOLOGY[d]`, `skim.SOV_TIME[o, d, out_period]`).
A converter writes them from today's files; all six ported steps,
converted, give exactly 1.x's results on all three regions.

The harness can also test changes to 1.x itself
([docs/scheduling.md](docs/scheduling.md)): a run-time patch that computes
tour scheduling's mode choice logsums once per distinct input leaves every
ARC result identical, halves peak memory, and lets 1.x run all ARC
households on a 15 GB machine. Swappable
pieces and the first experiments: [docs/experiments.md](docs/experiments.md); ideas measured but
not tried yet: [docs/future_experiments.md](docs/future_experiments.md).

On the **full MTC zone system** (1,475 zones, 826 skims, 7.2 GB; 10,000
households) asim_core matches 1.x bit for bit once 1.x's numexpr switch is
off, and its 8 steps take 18 s in one process against 71 s in 1.x
([docs/mtc_full.md](docs/mtc_full.md)). The whole model and its output steps run
from the input files in 166 s against 603 s in 1.x at 50,000 households (1.x
without a resume, numexpr off), 53 s against 249 s at 10,000; step by step,
time and memory, for full MTC and SANDAG:
[docs/profiles_by_step.md](docs/profiles_by_step.md). The first experiment there, on how
steps hold and look up skims, found that reading skim pages, not the
lookups, is most of the cost
([docs/skim_experiments.md](docs/skim_experiments.md)).

Problems and possible improvements found in 1.x itself are kept separately,
written so they can be reported to the ActivitySim project:
[../asim1_findings/README.md](../asim1_findings/README.md).

## Rules

* **Dependencies:** only widely used libraries maintained by large
  communities: NumPy, pandas, PyArrow, Pydantic, PyYAML. Importing skims from
  OMX also needs PyTables, once per skim file, and so does writing OMX trip
  matrices (`write_trip_matrices`); the model steps themselves don't.
  Anything else needs checking against this rule first. The package never imports ActivitySim
  1.x; only the scripts in `harness/` do, to run 1.x for comparison.
* **Every component declares what it reads and writes.** Reading or writing
  anything else is an error.
* **Every step is profiled** (runtime and peak memory) and can be compared
  with 1.x.

## Layout

| Path | What it is |
|---|---|
| `src/asim_core/store.py` | Data store: each table is an ID index plus NumPy columns, held once. Joins are index lookups, not merged copies. |
| `src/asim_core/component.py` | The component contract (`needs`, `provides`, `run`) and what a component sees while running. |
| `src/asim_core/runner.py` | Runs steps in order: computed by a component, or replayed from 1.x. Startup check, saving, profiling. |
| `src/asim_core/asim1.py` | Reads the state 1.x saves after every step, so a run can start at any step. |
| `src/asim_core/compare.py` | Compares results with 1.x, by row ID: exact, within tolerance, or different. |
| `src/asim_core/profiler.py` | Per-step runtime and peak memory, standard library only. |
| `src/asim_core/seams.py` | The swappable pieces: registry, reference implementations, and `Engine`, which picks an implementation per piece. |
| `src/asim_core/skims.py` | Skims: OMX files imported into a memory-mapped cache shared by every process, following 1.x's `network_los.yaml` settings; reused only while the OMX files are unchanged. |
| `src/asim_core/lookups.py` | Reference implementations of table and skim lookups. |
| `src/asim_core/timetable.py` | Each person's day as time periods (1.x's `person_windows`): which departure-return options still fit, booking a tour, and the functions scheduling formulas call. |
| `src/asim_core/random.py` | Random numbers. Reference: 1.x's per-row streams, reproduced exactly. Alternative: `counter`. |
| `src/asim_core/logit.py` | Reference implementations of utilities, probabilities (multinomial and nested logit) and choices, with 1.x's arithmetic. |
| `src/asim_core/alternatives.py` | Alternative implementations for experiments (e.g. `accumulate` utilities). |
| `src/asim_core/inputs/` | Input tables: definitions (`standard_tables.yaml`, checked by Pydantic), the reader (PyArrow, following `input_table_list`), and the checks before the run. `python -m asim_core.inputs` checks a region. |
| `src/asim_core/spec/language.py` | asim_core's formula language: checked and compiled to NumPy code. |
| `src/asim_core/spec/convert.py`, `src/asim_core/converter.py` | Converting 1.x's formulas and files to asim_core's language and model files (`python -m asim_core.converter`). |
| `src/asim_core/spec/model_file.py` | Reading model files (YAML with `extends`, CSV term, coefficient and derived-value tables). |
| `configs/<example>/` | asim_core's additions to the 1.x examples' configs: each region's `input_tables.yaml`, and the six ported steps converted to model files. |
| `src/asim_core/spec/` | Model definitions (Pydantic): choice models and assignment specs; the loader for today's YAML/CSV files; the formula translator. |
| `src/asim_core/components/` | Components. `simple_choice.py` runs any simple choice model from its definition; `initialize.py`, `auto_ownership.py`, `accessibility.py`, `location_choice.py`, `free_parking.py`, `cdap.py`, `mandatory_tour_frequency.py` (with `tours.py`: making tours and 1.x's tour IDs), `tour_mode_choice.py`, `tour_scheduling.py` and `tour_destination.py` are ported models; `annotate.py` runs annotations after a model. |
| `harness/` | Scripts that run 1.x and compare (these import 1.x). |
| `tests/` | Unit tests, plus an exact-match test against a saved 1.x run. |

## Running it

From the repository root, with ActivitySim 1.x's environment installed
(`uv sync`) and this package installed in it:

```bash
uv pip install -e asim_core
```

Run 1.x once with every step saved and profiled (legacy engine):

```bash
python asim_core/harness/run_asim1.py --out asim_core/work/asim1_legacy
```

Optionally, the same with sharrow. The first command compiles sharrow's code (slow, once); the second is the timed run:

```bash
python asim_core/harness/run_asim1.py --configs configs_sh_compile configs --out asim_core/work/asim1_sharrow_compile
```

```bash
python asim_core/harness/run_asim1.py --configs configs_sh configs --out asim_core/work/asim1_sharrow
```

Then run the checks, the end-to-end run and the profiles (skims are imported
from the example's OMX file the first time):

```bash
python asim_core/harness/run_milestone1.py
```

Profiles are cached: a step is measured again for an engine only when that
engine's code, the profiling harness, the configs, the saved 1.x run or
the machine changed. So 1.x is measured once, and asim_core again after
its code changes. With everything cached, the whole run takes seconds.
`--profiles all` measures everything again; `--profiles none` skips
profiling, for a quick correctness check after a change:

```bash
python asim_core/harness/run_milestone1.py --profiles none
```

To run the ported steps on their own, from the input files, with nothing
taken from 1.x (then compared with 1.x's tables after the last step):

```bash
python asim_core/harness/run_standalone.py --through trip_mode_choice
```

The milestone run checks the example's inputs first and stops on errors.
To check a region's inputs on their own:

```bash
python -m asim_core.inputs -c asim_core/configs/prototype_mtc -c activitysim/examples/prototype_mtc/configs -d activitysim/examples/prototype_mtc/data
```

To run the ported steps from the converted model files instead, add
`--model-files`. To convert an example's ported steps and compare each with
today's files and 1.x:

```bash
python asim_core/harness/check_model_files.py --example prototype_mtc
```

Results land in `asim_core/work/milestone1/` (`summary.md`, `comparisons.csv`,
`utility_check.csv`, `profile_comparison.csv`, `asim1_profiles.csv`, traces,
and the generated NumPy code for each model).

Experiments with alternative pieces, listed in `harness/experiments.yaml`
(see [docs/experiments.md](docs/experiments.md)):

```bash
python asim_core/harness/run_experiments.py
```

Tests:

```bash
python -m pytest asim_core/tests
```

## How a model is read

Today's files are read unchanged. For `auto_ownership` the loader reads
`auto_ownership.yaml`, `auto_ownership.csv` and
`auto_ownership_coefficients.csv` into a checked `ChoiceModelDefinition`.
Loading fails with a list of every problem found: unknown coefficients or
alternatives, or formulas that use something unsupported.

Models with purpose segments (a coefficient template) and preprocessors,
like tour mode choice, are read the same way, unchanged.

Formulas are parsed with Python's `ast` module, not executed, and turned
into plain NumPy code you can read. For example,
`@(df.num_workers==0)*df.density_index.clip(0, 10)` becomes
`((c['num_workers'] == 0) * _clip(c['density_index'], 0, 10))`. The
milestone script writes each model's generated code to
`work/milestone1/step_check/<step>/generated/`.

Preprocessor rows that test what data exists, such as
`df.number_of_participants if 'tour_type' in df.columns else 1`, are worked
out in the startup check, once the step's columns are known; only the branch
that applies is written into the generated code.

## Adding a component

A component is a name, a function, and what it reads and writes:

```python
from asim_core.component import Component

def work_from_home_run(ctx):
    workers = ctx.column("persons", "is_worker")
    ...
    return {"persons": {"work_from_home": choices}}

work_from_home = Component(
    name="work_from_home",            # use the 1.x step name to match 1.x random numbers
    run=work_from_home_run,
    needs={"persons": ["is_worker", "workplace_zone_id"]},
    provides={"persons": ["work_from_home"]},
)
```

Simple choice models, multinomial or nested logit, need no code: build them with
`simple_choice_component(definition, step, view, output_column)`, as
`components/auto_ownership.py` does; optionally only some rows choose, a
preprocessor adds columns first, and the choices are mapped to the values
written (`components/free_parking.py`).

## Not supported yet

Park-and-ride, shadow pricing (workplace and school location), CDAP's
joint tour option, school escorting, `override_hh_ids`, annotations
of tours and trips, Parquet skim files, two- and
three-zone systems (MAZ, TAP), writing trip tables to OMX, starting a run
from asim_core's own saved state, a command-line `run` for agency model
loops, multiprocessing, estimation, and peak-memory measurement outside
Linux. The loader names any unsupported setting it finds instead of running
a partial model.
