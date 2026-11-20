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

Status: two components are ported and match 1.x exactly: `compute_accessibility`
(with skims) and `auto_ownership`, on two regions' example models (MTC and
MWCOG, the latter with all 29,972 households). Every other step can be
**replayed** from a 1.x run, so the model runs end to end today. Results:
[docs/milestone1.md](docs/milestone1.md),
[docs/skims_and_accessibility.md](docs/skims_and_accessibility.md),
[docs/mwcog.md](docs/mwcog.md). Swappable
pieces and the first experiments: [docs/experiments.md](docs/experiments.md).

## Rules

* **Dependencies:** only widely used libraries maintained by large
  communities: NumPy, pandas, PyArrow, Pydantic, PyYAML. Importing skims from
  OMX also needs PyTables, once per skim file; running models doesn't.
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
| `src/asim_core/random.py` | Random numbers. Reference: 1.x's per-row streams, reproduced exactly. Alternative: `counter`. |
| `src/asim_core/logit.py` | Reference implementations of utilities, probabilities and choices, with 1.x's arithmetic. |
| `src/asim_core/alternatives.py` | Alternative implementations for experiments (e.g. `accumulate` utilities). |
| `src/asim_core/spec/` | Model definitions (Pydantic): choice models and assignment specs; the loader for today's YAML/CSV files; the formula translator. |
| `src/asim_core/components/` | Components. `simple_choice.py` runs any simple choice model from its definition; `auto_ownership.py` and `accessibility.py` are ported models. |
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

Formulas are parsed with Python's `ast` module, not executed, and turned
into plain NumPy code you can read. For example,
`@(df.num_workers==0)*df.density_index.clip(0, 10)` becomes
`((c['num_workers'] == 0) * _clip(c['density_index'], 0, 10))`. The
milestone script writes each model's generated code to
`work/milestone1/step_check/<step>/generated/`.

## Adding a component

A component is a name, a function, and what it reads and writes:

```python
from asim_core.component import Component

def free_parking_run(ctx):
    workers = ctx.column("persons", "is_worker")
    ...
    return {"persons": {"free_parking_at_work": choices}}

free_parking = Component(
    name="free_parking",              # use the 1.x step name to match 1.x random numbers
    run=free_parking_run,
    needs={"persons": ["is_worker", "workplace_zone_id"]},
    provides={"persons": ["free_parking_at_work"]},
)
```

Simple choice models need no code: build them with
`simple_choice_component(definition, step, view, output_column)`, as
`components/auto_ownership.py` does.

## Not supported yet

Nested logit, coefficient templates (purpose segments), preprocessors and
annotation (assignment specs work, but joins to other tables like
`persons.age` and `reindex()` don't yet), skims indexed by each row's time
period (`odt_skims`), Parquet skim files, two- and three-zone systems (MAZ,
TAP), writing trip tables to OMX, starting a run from asim_core's own saved
state, a command-line `run` for agency model loops, sampling of alternatives,
multiprocessing, estimation, and peak-memory measurement outside Linux. The loader names any unsupported
setting it finds instead of running a partial model.
