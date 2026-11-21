# Swappable pieces and experiments

The core is a framework of swappable pieces. Speed-ups come from experiments:
alternative versions of a piece, plugged in by a setting and measured
against ActivitySim 1.x. Sharrow is not used anywhere in asim_core; 1.x
with sharrow appears only as a baseline in the reports, to check that
nothing is significantly slower.

## Two levels of swapping

1. **Whole steps.** Each model step is a component with declared inputs and
   outputs (see `component.py`). A region picks which components to run, and
   any component can be replaced by another that reads and writes the same
   columns.
2. **Pieces inside a step ("seams").** Each seam is one job, with a
   **reference** implementation (plain NumPy, readable, matches 1.x exactly)
   and any number of alternatives. Which implementation runs is a setting:

   ```python
   Runner(components, out_dir, engine=Engine({"random": "counter"}))
   ```

   Pieces not named use the reference. `step_engines={step: Engine(...)}`
   overrides the choice for individual steps.

| Seam | Job | Reference | Alternatives so far |
|---|---|---|---|
| `lookup` | values of a column at given rows (joins between tables) | `numpy` | |
| `skim_access` | how a step holds the skims it reads | `memory_map` | `prefault`, `in_memory` ([skim_experiments.md](skim_experiments.md)) |
| `skim_reuse` | whether a model's formulas share the skim values they read | `as_written`: every lookup as the formulas are written | `once`: each skim named more than once looked up once per batch, shared until its last use ([skim_experiments.md](skim_experiments.md)) |
| `skim_lookup` | skim values for origin/destination zones | `flat_take`: a one-dimensional gather, positions shared by a step's skims (the default since the skim experiment) | `index_2d` (two-dimensional indexing, the default before) |
| `sample_columns` | which of the choosers' columns trip destination's sampling copies out to every zone | `spec_reads`: only the columns the sample spec reads, as 1.x's `drop_unused_columns` (the default since the trip destination measurement) | `all_columns`: every column the stops carry (the default before; the same results) |
| `size_lookup` | size terms for trip destination's sampling rows (every stop × every zone) | `by_position`: each stop's sizes taken whole as its purpose's column of the size table (the default since the trip destination measurement) | `match`: every row's zone and purpose matched, as 1.x's `DataFrameMatrix.get` (the default before; the same results) |
| `utilities` | formula values and coefficients → utilities | `matmul`: one matrix product, as 1.x | `accumulate`; `jax`: formulas and sum compiled with JAX ([jax_experiment.md](jax_experiment.md)) |
| `interaction_utilities` | one utility per chooser-alternative row | `sequential`, as 1.x | `jax` (as above) |
| `probabilities` | utilities → probabilities (multinomial logit) | `asim1` | |
| `nested_probabilities` | utilities and a nest tree → probabilities (nested logit) | `asim1` | |
| `sample` | probabilities and draws → sampled alternatives | `binary_search`: 1.x's picks, about log2(alternatives) comparisons per draw (the default since the JAX experiment) | `inverse_cdf`: the same picks, compared as 1.x does, every alternative per draw (the default before) |
| `choose` | probabilities and random draws → choices | `sequential`, as 1.x | |
| `random` | per-row random draws for a step | `asim1`: 1.x's streams, exactly | `counter` |

List them with:

```bash
python asim_core/harness/run_experiments.py --list
```

or, with each implementation's description:

```bash
PYTHONPATH=asim_core/src python -m asim_core.seams
```

**When a change becomes the default, the way it worked before stays as an
alternative**, so an agency can run both and compare. Any run script takes
the choices as a flag (`--seams`, JSON; seams not named use the default):

```bash
python asim_core/harness/run_milestone1.py --seams '{"sample_columns": "all_columns", "size_lookup": "match"}'
```

```bash
python asim_core/harness/run_standalone.py --through trip_destination --seams '{"skim_lookup": "index_2d"}'
```

```bash
python asim_core/harness/profile_step.py --engine asim_core --step trip_destination --scratch /tmp/td --out td.csv --seams '{"size_lookup": "match"}'
```

`run_milestone1.py` checks every step against 1.x with the choices given;
`run_standalone.py` runs the steps together and compares the last tables;
`profile_step.py` times one step in fresh processes.

## How results are compared

Each implementation declares how its results relate to the reference, and
the experiment report checks accordingly:

- **exact:** results identical to the reference.
- **approximate:** identical up to floating-point rounding. The report shows
  the largest utility difference and whether any choice changed.
- **statistical:** individual results differ by design (for example,
  different random numbers), so only distributions can be compared. The
  report shows the largest difference in the share of rows taking any value,
  next to the difference expected by chance.

## Running experiments

List experiments in `harness/experiments.yaml`:

```yaml
step: auto_ownership_simulate
repeat: 5
experiments:
  reference: {}
  counter_random:
    random: counter
```

Then, from the repository root:

```bash
python asim_core/harness/run_experiments.py
```

For each experiment it runs the step from 1.x's saved state and compares the
result with 1.x, then times the step in fresh processes (`repeat` times),
alongside the same step in 1.x legacy and 1.x sharrow. It also times each
piece on its own with synthetic data at larger row counts. Results:
`asim_core/work/experiments/experiments.md` and `.csv`.

## Adding an alternative

Write a function with the same signature as the reference and register it:

```python
import numpy as np
from asim_core.seams import register

@register("choose", "cumulative", agreement="approximate",
          description="first alternative whose cumulative probability reaches the draw")
def choose_cumulative(probs, rands):
    cum = np.cumsum(probs, axis=1)
    return (cum < rands.reshape(-1, 1)).sum(axis=1).astype(np.int32)
```

Import the module before running (built-in alternatives live in
`src/asim_core/alternatives.py`), then name it in `experiments.yaml`. It can
call anything that's callable from Python (Numba, JAX, a C++ extension), as
long as it takes and returns the same data.

## First results

`auto_ownership`, `prototype_mtc` example (5,000 households), Linux, 4 CPUs.
Step times are the median of 5 fresh-process runs.

| Experiment | Result vs 1.x | Households with a different choice | Step time (s) |
|---|---|---|---|
| reference | same as 1.x | 0 | 0.021 |
| `utilities=accumulate` | same as 1.x (utilities differ by at most 7×10⁻¹⁵) | 0 | 0.019 |
| `random=counter` | same distribution: largest share difference 0.72 points, chance level about 1.94 | 2,125 (by design) | 0.010 |
| both | same distribution | 2,125 (by design) | 0.010 |
| 1.x legacy (baseline) | | | 0.180 |
| 1.x sharrow (baseline) | | | 0.188 |

Each piece on its own, synthetic data, 1,000,000 rows (single runs):

| Piece | Implementation | Seconds | Extra memory (MB) |
|---|---|---|---|
| random | `asim1` (reference) | 2.28 | 66 |
| random | `counter` | 0.13 | 77 |
| utilities, 29 terms × 5 alternatives | `matmul` (reference) | 0.12 | 260 |
| utilities, 29 terms × 5 alternatives | `accumulate` | 0.29 | 45 |

What this says:

- **Random numbers are the reference's biggest cost** in this step: about
  11 of its 12 ms of calculation. 1.x reseeds a generator for every row in a
  Python loop, and every choice step pays this cost in proportion to its row
  count. `counter` keeps the same rules (results don't depend on row order,
  batching or other rows) and is about 18 times faster at a million rows.
  But it draws different numbers, so results match 1.x only statistically.
  Before relying on it, its statistical quality should be reviewed. Using it
  also means deciding where exact 1.x reproducibility is still needed. One
  option: reference for validation runs, `counter` for production runs.
- **`accumulate` trades speed for memory** in computing utilities: 6 times
  less memory, 2.4 times slower on that piece alone, at a million rows. Here
  utilities are a small part of the step. It becomes worth testing in steps
  where the table of term values is large: mode and destination choice, with
  hundreds of terms.
- **The default is unchanged:** with all reference pieces, results still
  match 1.x exactly.

The example is small, so step times are indicative only. The synthetic
benchmarks show how each piece's cost grows with size.
