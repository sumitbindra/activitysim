# Experiment: sampling by binary search, and formulas compiled with JAX

Step 3 of the plan after the skim experiment. The profile by piece
([where_time_goes.md](where_time_goes.md)) found two targets on full MTC:
sampling destinations (10% of the time, a NumPy problem) and formulas plus
utilities (35%, the part a compiler such as JAX's XLA could speed up).

Both are **alternatives, not defaults**: the reference implementations stay
the default until an alternative is shown to be better in every respect.
They are hypotheses under test.

## Summary

- **Sampling by binary search (`sample: binary_search`) picks exactly the
  same zones** (all six steps match 1.x bit for bit) and cuts sampling from
  1.94 s to 0.40 s (−79%) on full MTC: about 7% of the 8 ported steps.
  It is never slower in the benchmarks (about equal at 30 alternatives, 3.6
  to 5 times faster at 1,454). **Now the default**; the original stays
  available as `inverse_cdf` (`Engine({"sample": "inverse_cdf"})`).
- **JAX (`utilities: jax`, `interaction_utilities: jax`)**: every formula set
  of every ported step compiles, and **every choice is the same as 1.x's**
  (logsums differ by up to 7.2e-7, as expected from a compiler that rounds
  in a different order). Once compiled, the 8 steps take 12% less time:
  formulas and utilities −20%, skim lookups −26%. But compiling takes 1 to
  7 s per step in every new process even with compiled code kept on disk (2
  to 11 s without), so at 10,000 households each step run on its own is 1.3
  to 8 times slower (2 to 12 times without the disk cache). JAX doesn't pay
  off at this size.
- **Two findings don't need JAX:**
  - the skim lookup saving comes from looking up each distinct skim once per
    formula set (JAX needs that to work at all); NumPy could do the same;
  - JAX's compute gain is largest in scheduling (−21% and −43%), where the
    formula chains are long, and nil or negative in tour mode choice.

## What was tried

### Sampling by binary search

Each of the 30 draws per chooser picks the first zone whose running sum of
probabilities exceeds the draw. The reference finds it by comparing the
draw with every zone's running sum (1,454 comparisons per draw); the
alternative counts the running sums at or below the draw bit by bit, from
the largest power of two down (`logit.sample_binary_search`): 11
comparisons for 1,454 zones, all vectorized. They are the same comparisons,
so the picks are the same; rows with missing probabilities are counted as
the reference does. Tested against the reference with zeros, ties, draws
equal to running sums, draws above the total and missing values
(`tests/test_sampling.py`).

| Choosers × zones, 30 draws | Reference | Binary search |
|---|---|---|
| 3,000 × 1,454 | 227 ms | 45 ms |
| 15,000 × 1,454 | 858 ms | 236 ms |
| 3,000 × 200 | 26 ms | 13 ms |
| 3,000 × 30 | 6 ms | 5 ms |

### Formulas and utilities compiled with JAX

`asim_core/jax_backend.py` (optional: `pip install "asim-core[jax]"`; JAX
0.6.2, CPU). A model's formulas are generated as plain NumPy code; the
alternative runs the same source with `jax.numpy` in place of NumPy and
compiles it together with the sum of utilities into one function. What it
takes:

- **Skim values gathered in NumPy first**, then passed in (copying the
  skims into JAX would cost 6 GB);
- **timetable questions** in scheduling formulas
  (`tt.previous_tour_ends(...)`) worked out in NumPy first, likewise;
- **text columns as codes**: the words formulas compare them with
  (`tour_type == 'escort'`) become numbers, and so do the columns' values;
- **rows padded to a few sizes** (at most 12.5% more rows), since XLA
  compiles once per array shape;
- **compiled code shared** between steps with the same formulas, and kept
  on disk with `ASIM_JAX_CACHE_DIR` for later runs;
- **64-bit floats** turned on; a formula set that can't compile falls back
  to the NumPy code (none did, in the end), and `ASIM_JAX_VERIFY=1`
  compares each formula set's first call with NumPy.

The verification on MTC's example: scheduling within 4e-15 of NumPy,
destination and location choice within 5e-7, tour mode choice within 3e-6
(32-bit skim values, which XLA fuses into multiplications and additions
differently).

## Results (full MTC, 10,000 households)

**Matching 1.x** (every compared column):

| Step | Binary search | JAX |
|---|---|---|
| school_location | same as 1.x | choices same; logsums within 2.3e-7 |
| workplace_location | same as 1.x | choices same; logsums within 1.4e-7 |
| non_mandatory_tour_destination | same as 1.x | choices same; logsums within 1.9e-7 |
| mandatory_tour_scheduling | same as 1.x | same as 1.x |
| non_mandatory_tour_scheduling | same as 1.x | same as 1.x |
| tour_mode_choice_simulate | same as 1.x | choices same; logsums within 7.2e-7 |

**Once compiled, in one process** (`profile_pieces.py`, second run of the
8 steps; seconds):

| Piece | Reference | Binary search | JAX |
|---|---|---|---|
| sampling | 1.94 | **0.40** | 1.97 |
| formulas + utilities | 7.59 | 7.46 | **6.11** |
| skim lookups | 5.18 | 5.85 | **3.82** |
| all 8 steps | 21.1 | 20.2 | **18.6** |
| first run: compiling | | | 42.6 (81 compiles) |

Formulas + utilities per step, reference → JAX: mandatory scheduling 2.07 →
1.63 (−21%), non-mandatory scheduling 0.81 → 0.46 (−43%), non-mandatory
destination 1.73 → 1.41 (−18%), workplace 2.10 → 1.77 (−16%), school 0.69
→ 0.61 (−12%), tour mode choice 0.19 → 0.24 (+26%).

**Each step in a fresh process** (`run_experiments.py`, median of 3,
skims in the page cache; the JAX variants include compiling):

| Step | Reference | Binary search | JAX | JAX, compiled code on disk | JAX compiling (none / on disk) |
|---|---|---|---|---|---|
| school_location | 2.06 | 2.13 | 6.67 | 4.46 | 4.3 / 2.2 |
| workplace_location | 5.57 | 5.06 | 12.86 | 9.89 | 7.4 / 4.3 |
| non_mandatory_tour_destination | 4.71 | 5.03 | 13.07 | 9.27 | 7.9 / 4.2 |
| mandatory_tour_scheduling | 4.73 | 4.18 | 14.98 | 10.24 | 11.1 / 6.2 |
| non_mandatory_tour_scheduling | 1.34 | 1.49 | 2.80 | 1.81 | 1.8 / 0.9 |
| tour_mode_choice_simulate | 1.10 | 1.04 | 12.73 | 8.73 | 11.3 / 7.2 |

Between fresh processes the times vary by about ±10% (mandatory scheduling,
which doesn't sample, came out 12% faster with binary-search sampling), so
the one-process table above is the one to read for the pieces.

Compiled code on disk saves XLA's compile, but not JAX's work before it:
tracing the generated Python (several hundred formulas for tour mode
choice) and lowering it, in every new process. Own memory: within about 60 MB of
the reference, except tour mode choice (+150 MB) and non-mandatory
scheduling (+60 MB), XLA's runtime.

## What it says

- **Binary search is a straight win** at every size measured, with
  identical results. Making it the default is a one-line change.
- **JAX's compute gain is real but modest** (−20% on formulas and
  utilities), for three reasons: much of the work is memory-bound (formula
  values are read once and used once); padding adds up to 12.5% of rows; and
  the inputs are copied in and the results out.
- **Its fixed cost dominates at this size.** Compiling or tracing costs 1 to
  11 s per step per process; the compute saved is up to about 0.4 s per step
  at 10,000 households. With 2.9 million households (290 times more rows, with
  the same compiles) the saving would be roughly 30 to 120 s per step against
  the same few seconds of compiling, so JAX could pay off for full runs,
  if the run happens in one process and the gains hold at that size. That is
  a hypothesis this experiment does not test (it needs a 1.x run with more
  households to start from).
- **Exactness:** JAX can't promise 1.x's results bit for bit; here no
  choice changed, and logsums moved by under 1e-6.

## Possible next steps

1. ~~Make `binary_search` the default for sampling~~ (done: the default
   since; the original stays selectable as `inverse_cdf`).
2. Look up each distinct skim once per formula set in the NumPy path too
   (the −26% on skim lookups, identical results), as a new alternative.
3. JAX at a larger sample (e.g. 100,000 households), whole run in one
   process, to test whether the fixed cost is paid back.

## Reproducing

```bash
python asim_core/harness/run_experiments.py --config asim_core/harness/experiments_jax.yaml \
  --step workplace_location --repeat 3 --page-cache warm --no-baselines \
  --example-dir asim_core/work/mtc_full/prototype_mtc_full \
  --asim1-out asim_core/work/mtc_full/asim1_10k --match-run asim_core/work/mtc_full/asim1_10k_no_numexpr \
  --read-skim-cache true --out asim_core/work/mtc_full/experiments_jax/workplace_location
python asim_core/harness/profile_pieces.py --example-dir asim_core/work/mtc_full/prototype_mtc_full \
  --asim1-out asim_core/work/mtc_full/asim1_10k --out asim_core/work/mtc_full/pieces_jax \
  --seams '{"utilities": "jax", "interaction_utilities": "jax"}'
```
