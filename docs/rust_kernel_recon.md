# Rust Kernel Reconnaissance — Destination / Location Choice Hot Path

**Status:** Reconnaissance deliverable for Section 1 of the "Rewrite ActivitySim
Destination Choice Hot Path in Rust" plan. **No Rust has been written.** This
document maps the existing Python so that the kernel can be ported safely. All
file paths and line numbers are as of the branch
`claude/activitysim-rust-kernel-fjkpi4`.

> Scope reminder: the Rust target is **stages 1 (sample) and 3 (simulate)** of
> destination/location choice — i.e. the interaction-utility evaluation +
> softmax + sampling/choice kernels. Mode-choice **logsums (stage 2)**,
> orchestration, shadow pricing, config/spec parsing, RNG channel management,
> and I/O all stay in Python.

---

## 0. TL;DR — the most important findings

1. **RNG is per-row, not global.** Every chooser row gets its **own
   independent `numpy.random.RandomState` (MT19937 / Mersenne Twister)**, seeded
   deterministically from its index value:
   `row_seed = (base_seed + channel_seed + step_seed + row_index) % 2**32`.
   Randomness for a chooser depends **only** on that chooser's seed and a
   per-step "offset" counter — never on the order in which choosers are
   processed. **This means parallelism over choosers is trivially
   deterministic** (see §5). This is the single best piece of news for the
   project.

2. **Sampling consumes `sample_size` uniforms per chooser**, drawn via
   `random_for_df(probs, n=sample_size)` → shape `(n_choosers, sample_size)`.
   Final choice consumes **1 uniform per chooser** via `random_for_df(probs)`.
   The destination *sample* stage crosses **all** alternatives and samples by
   probability — it does **not** use `choice_for_df` (that path is only hit by
   `interaction_simulate` when `sample_size < len(alternatives)`).

3. **The kernel is linear-in-parameters.** Utilities are
   `expression_values · coefficients`. Every spec row is one expression; the
   utility is the dot product of the per-row expression values with the spec
   coefficient column(s). The expressions themselves are a small, finite set of
   pandas/numpy idioms (see §4) — an IR is feasible.

4. **Skims are a contiguous C-order `float32` array** of shape
   `(num_skims, n_zones, n_zones)`, indexed by fancy indexing
   `skim_data[block_offset, mapped_orig, mapped_dest]`. Zone ids map to 0-based
   offsets via a simple integer offset (usually `-1`) or a lookup Series.
   Zero-copy via rust-numpy is straightforward.

5. **Recommended v1 RNG strategy:** have Python pre-draw the uniforms (calling
   the existing `random_for_df`) and pass the arrays to Rust. Rust then becomes
   purely deterministic given its inputs and we sidestep reimplementing MT19937.
   Reimplement the generator in Rust only if RNG generation itself becomes a
   bottleneck.

---

## 1. Module map (files, roles, key signatures)

| File | Role | Port to Rust? |
|---|---|---|
| `activitysim/core/interaction_sample.py` | Stage 1: MNL sampling of alternatives | **Yes** |
| `activitysim/core/interaction_sample_simulate.py` | Stage 3: final choice among pre-sampled alts | **Yes** |
| `activitysim/core/interaction_simulate.py` | `eval_interaction_utilities()` + full-alts simulate | **Yes** (util eval) |
| `activitysim/core/logit.py` | `utils_to_probs`, `make_choices`, softmax, logsum | **Yes** |
| `activitysim/core/choosing.py` | numba kernels: `choice_maker`, `sample_choices_maker_preserve_ordering` | **Yes** (reference) |
| `activitysim/core/simulate.py` | `eval_utilities` (dot-product), `simple_simulate`, MNL/NL | partial (util eval) |
| `activitysim/core/random.py` | the RNG contract (channels, seeds, draws) | **No** (call from Python in v1) |
| `activitysim/core/skim_dictionary.py` / `skim_dict_factory.py` | skim storage + lookup | No (read buffers zero-copy) |
| `activitysim/core/los.py` / `network_los.py` | skim/los container | No |
| `activitysim/abm/tables/size_terms.py` | destination size terms | No (read buffer zero-copy) |
| `activitysim/abm/models/util/tour_destination.py` | tour dest orchestration (sample→logsum→simulate) | No (dispatch point) |
| `activitysim/abm/models/location_choice.py` | workplace/school + shadow-pricing loop | No |
| `activitysim/abm/models/trip_destination.py` | trip destination flow | No |

### Kernel signatures to port

```python
# activitysim/core/interaction_sample.py
def interaction_sample(
    state, choosers: pd.DataFrame, alternatives: pd.DataFrame, spec: pd.DataFrame,
    sample_size: int, alt_col_name: str, allow_zero_probs: bool = False,
    log_alt_losers: bool = False, skims=None, locals_d=None, chunk_size: int = 0,
    chunk_tag=None, trace_label=None, zone_layer=None, explicit_chunk_size=0,
    compute_settings=None,
) -> pd.DataFrame   # columns: [alt_col_name, prob(float32), pick_count(uint32)], index=chooser id

def make_sample_choices(
    state, choosers, probs, alternatives, sample_size, alternative_count,
    alt_col_name, allow_zero_probs, trace_label, chunk_sizer,
)  # draws rands, returns flat choices_df

# activitysim/core/interaction_sample_simulate.py
def interaction_sample_simulate(
    state, choosers, alternatives, spec, choice_column,
    allow_zero_probs=False, zero_prob_choice_val=None, log_alt_losers=False,
    want_logsums=False, skims=None, locals_d=None, chunk_size=0, chunk_tag=None,
    trace_label=None, trace_choice_name=None, estimator=None, skip_choice=False,
    explicit_chunk_size=0, *, compute_settings=None,
) -> pd.Series | pd.DataFrame   # chosen alt per chooser (+ logsum if want_logsums)

# activitysim/core/interaction_simulate.py
def eval_interaction_utilities(
    state, spec, df, locals_d, trace_label, trace_rows, estimator=None,
    log_alt_losers=False, extra_data=None, zone_layer=None, compute_settings=None,
) -> (pd.DataFrame, dict|None)   # utilities (len(df), 1)

# activitysim/core/logit.py
def utils_to_probs(state, utils, trace_label=None, exponentiated=False,
    allow_zero_probs=False, trace_choosers=None, overflow_protection=True,
    skip_failed_choices=True, return_logsums=False)
def make_choices(state, probs, trace_label=None, trace_choosers=None,
    allow_bad_probs=False) -> (positions: pd.Series, rands: pd.Series)
```

---

## 2. Tour-destination data-flow diagram (the three-stage pattern)

Orchestrated by `run_tour_destination()` in
`activitysim/abm/models/util/tour_destination.py:914`. Per segment:

```
                       tours (choosers)  +  destination_size_terms (alternatives)
                                       |
        ┌──────────────────────────────┴───────────────────────────────┐
        │ STAGE 1 — SAMPLE          run_destination_sample()  (:610)     │
        │   → _destination_sample() (:72)                                │
        │   → interaction_sample()  (:143)   [core/interaction_sample.py]│  ◄── RUST
        │                                                                │
        │   cross ALL alts × choosers → eval utilities (linear)          │
        │   → utils_to_probs (softmax) → make_sample_choices             │
        │     draws random_for_df(probs, n=sample_size)                  │
        │   OUT: (chooser_id, alt_dest, prob, pick_count)                │
        └────────────────────────────┬───────────────────────────────────┘
                                      │  destination_sample (sparse: sample_size rows/chooser)
        ┌────────────────────────────┴───────────────────────────────────┐
        │ STAGE 2 — LOGSUM          run_destination_logsums()  (:695)     │
        │   → logsum.compute_location_choice_logsums() (:753)            │  ◄── STAYS PYTHON
        │   computes mode-choice logsum for each sampled (orig,dest)     │
        │   OUT: destination_sample + "mode_choice_logsum" column         │
        └────────────────────────────┬───────────────────────────────────┘
                                      │
        ┌────────────────────────────┴───────────────────────────────────┐
        │ STAGE 3 — SIMULATE        run_destination_simulate()  (:770)    │
        │   re-attach size_term (:837), join logsum                       │
        │   → interaction_sample_simulate()  (:887)                       │  ◄── RUST
        │     [core/interaction_sample_simulate.py]                       │
        │   join sparse alts ← choosers → eval utilities (incl. logsum)   │
        │   → pad to (n_choosers, max_sample) with -999 → utils_to_probs  │
        │   → make_choices  draws random_for_df(probs)  (1 per chooser)   │
        │   OUT: chosen dest_zone per chooser (+ logsum optional)         │
        └──────────────────────────────────────────────────────────────────┘
```

**Workplace/school location choice** (`location_choice.py`) is the same three
stages — `run_location_sample()` / `run_location_logsums()` /
`run_location_simulate()` inside `run_location_choice()` (:742) — wrapped by the
**shadow-pricing loop** `iterate_location_choice()` (:956): for
`iteration in range(1, max_iterations+1)`, update shadow prices → run choice →
`spc.set_choices()` → `spc.check_fit()` → break on convergence. The iteration
and the size-term adjustment (`aggregate_size_terms()` :276, applies
`shadow_price_utility_adjustment` / `shadow_price_size_term_adjustment`) stay in
Python; only the inner choice kernel is Rust. Note the kernel may be invoked
2–10× per model run because of shadow-price iterations, so kernel speed
compounds.

**Trip destination** (`trip_destination.py`) follows the same
sample→(logsum)→simulate shape via `_destination_sample()` and
`interaction_sample_simulate`, with size terms looked up by `(dest_taz, purpose)`
through `DataFrameMatrix.get(...)`.

### Settings driving the kernel (`TourLocationComponentSettings`)
`SAMPLE_SPEC`, `SAMPLE_SIZE` (e.g. ~30), `SPEC`, `COEFFICIENTS`,
`LOGSUM_SETTINGS`, `CONSTANTS`, `ALT_DEST_COL_NAME`, `CHOOSER_ORIG_COL_NAME`,
`SIZE_TERM_SELECTOR`. These are read in Python and stay there; the Rust boundary
receives already-resolved numeric arrays.

---

## 3. Inner numeric pipeline (what Rust must reproduce)

### 3.1 Utility evaluation
Two layouts exist; both reduce to **linear-in-parameters**:

- **Interaction layout** (`eval_interaction_utilities`,
  `interaction_simulate.py:28`): builds a cross/joined frame of
  `len(choosers) × n_alts` rows, then for each spec row evaluates the
  expression over the whole frame and accumulates
  `utilities += (expr_value * coefficient)`. The spec here has a **single
  coefficient column**. Result shape `(len(choosers)*n_alts, 1)`, reshaped to
  `(n_choosers, n_alts)`.
  - Expressions starting with `_` → temp assignment into `locals_d` (not summed).
  - Expressions starting with `@` → Python `eval` of `expr[1:]`.
  - Otherwise → `fast_eval(df, expr, ...)` (pandas-style).

- **Simple/MNL layout** (`eval_utilities`, `simulate.py:537`): builds
  `expression_values` of shape `(n_exprs, n_choosers)`, then
  `utilities = expression_values.T @ spec.values` → `(n_choosers, n_alts)`.
  Used by mode choice / `simple_simulate`; relevant because the IR/evaluator
  can be shared.

### 3.2 Softmax — `logit.utils_to_probs` (`logit.py:171`)
Constants: `EXP_UTIL_MIN = 1e-300`, `PROB_MIN = 0.0`, `PROB_MAX = 1.0`.
```
if overflow_protection:        # default True (off only when allow_zero_probs)
    shifts = utils.max(axis=1, keepdims=True)
    utils -= shifts
utils = exp(utils)
utils[utils <= 1e-300] = 0
row_sum = utils.sum(axis=1)
if return_logsums:
    logsums = log(row_sum) + squeeze(shifts)     # add the per-row shift back
probs = utils / row_sum[:, None]
probs[isnan(probs)] = 0.0
probs = clip(probs, 0.0, 1.0)
```
- Per-row max-shift before `exp` is the overflow trick — **must match**.
- `utils_to_logsums` (`logit.py:134`) is the *no-shift* variant:
  `logsum[i] = log(sum_j exp(utils[i,j]))`, clip to `[1e-300, inf]`.
- Reduction is `sum(axis=1)` in NumPy (pairwise summation). See §5.4 on FP.

### 3.3 Final choice — `logit.make_choices` (`logit.py:351`)
```
rands = state.get_rn_generator().random_for_df(probs)   # 1 uniform per chooser row
choices = choice_maker(probs.values, rands)             # choosing.py:6
```
`choice_maker` (numba, `choosing.py:6`): for each row, `z = rand`; iterate cols,
`z -= pr[row,col]`; first col where `z <= 0` is chosen. Fallback (loop
exhausts): pick argmax probability. Returns 0-based column positions.

### 3.4 Sampling — `make_sample_choices` (`interaction_sample.py:30`)
```
rands = state.get_rn_generator().random_for_df(probs, n=sample_size)  # (n_choosers, sample_size)
choices, choice_probs = sample_choices_maker_preserve_ordering(
    probs.values, rands, alternatives.index.values)
```
`_sample_choices_maker_preserve_ordering` (`choosing.py:92`): per chooser,
`argsort` the row's rands; walk alternatives accumulating cumulative prob `z`;
assign each sorted random point to the alternative whose cumulative prob first
exceeds it; **write results back to the random point's original position**
(this preserves draw order — required for parity). Output shape
`(sample_size, n_choosers)`; flattened **F-order** into the result frame.
Then duplicate (chooser, alt) picks are collapsed and counted into
`pick_count` (uint32), `prob` cast to float32, sorted by (chooser, alt).

### 3.5 sample_simulate padding (`interaction_sample_simulate.py`)
Sampled alternatives are sparse and variable-length per chooser. Utilities are
padded with `-999` up to `max_sample_count` per chooser, reshaped to
`(n_choosers, max_sample_count)`, softmaxed, then `make_choices` picks a column;
the column position is mapped back to the alt via `first_row_offsets`. Rust must
reproduce the **-999 padding** and the offset arithmetic exactly.

---

## 4. Expression IR inventory (sizing the IR)

Source specs inspected under
`activitysim/examples/prototype_mtc/configs/`:
`*_tour_destination_sample.csv`, `*_tour_destination.csv`,
`trip_destination_sample.csv`, `trip_destination.csv`,
`atwork_subtour_destination*.csv`.

The set of expression forms actually used is small and finite:

| Category | Concrete examples | IR need |
|---|---|---|
| Skim lookup (2D) | `@skims['DIST']`, `_od_DIST@od_skims['DIST']` | skim read by (O,D) |
| Skim lookup + clip/arith | `@skims['DIST'].clip(0,1)`, `@(skims['DIST']-1).clip(0,1)`, `@(skims['DIST']-2).clip(0,3)` | sub, clip(lo,hi) |
| 3D skim (time period) | `od_skims['SOV_TIME']` style (dim3 = period) | skim read by (O,D,period) |
| Size term + log | `@df['size_term'].apply(np.log1p)`, `@np.log1p(size_terms.get(df.dest_taz, df.purpose))` | size lookup, log1p |
| Availability / zero-size | `@df['size_term']==0`, `@size_terms.get(df.dest_taz, df.purpose) == 0`, `size_term==0` | eq compare → 0/1 |
| Boolean gate × value | `@(~df.is_joint & df.outbound) * (_od_DIST + _dp_DIST)`, `@df.outbound * _od_DIST` | and/not, mul, add |
| Distance/mode availability | `@(df.tour_mode_is_walk) & (od_skims['DISTWALK'] > max_walk_distance)` | compare, and |
| Plain column passthrough | `mode_choice_logsum`, `od_logsum`, `dp_logsum` | column ref |
| min/clip caps | `@np.minimum(np.log(df.pick_count/df.prob), 60)` | div, log, min |
| Temp assignment | leading `_name@expr` (e.g. `_od_DIST@od_skims['DIST']`) | named temp binding |

**IR operations required (v1):** column ref (chooser/alt), constant, skim
lookup (2D and 3D by O/D[/period]), size-term lookup `(zone, segment)`,
binary `+ - * /`, comparisons `== > < & ~`, `clip(lo,hi)`,
`np.minimum`/`np.maximum`, `log`, `log1p`, `apply(np.log1p)`, and named temp
bindings (`_`-prefixed). Locals available in the eval context: `skims`,
`od_skims`/`dp_skims`, `df`, `size_terms` (a `DataFrameMatrix`), `np`, `pd`,
plus named constants from `CONSTANTS` (e.g. `max_walk_distance`).

**Fallback policy:** the Python dispatcher compiles each spec to IR at load
time; if any expression is unsupported, that component silently uses the
existing Python path. This keeps the port partial and safe (Milestone 5).

**De-risking (Milestone 1):** build a pure-Python reference IR evaluator and
assert it matches pandas evaluation of the original spec **before** any Rust.

---

## 5. The RNG contract (highest risk) — precise

Source: `activitysim/core/random.py`, accessed via
`state.get_rn_generator()` → `Random` (`workflow/state.py:880`).

### 5.1 Generator
Underlying PRNG is **`numpy.random.RandomState` = MT19937 (Mersenne
Twister)**. A single `RandomState` object is reused and **reseeded per row**
on the fly (creating one object per row would cost ~5KB each).

### 5.2 Seed derivation (`random.py:104`, `:24`)
```
channel_seed = hash32(channel_name)     # md5(name).hexdigest() → int & 0xFFFFFFFF
step_seed    = hash32(step_name)
row_seed     = (base_seed + channel_seed + step_seed + row_index) % 2**32
```
- `base_seed`: global, default `0`; set once via `set_base_seed` before step 1.
- `row_index`: the actual DataFrame index **value** (e.g. `tour_id`,
  `person_id`) — **not** positional. Hence stable, repeatable, restartable
  streams keyed to entity ids.
- `hash32` uses **MD5** of the UTF-8 name, low 32 bits. (`_MAX_SEED = 1<<32`.)

### 5.3 Draw production & offsets (`random.py:174`, `:207`)
Per-row state table `row_states[["row_seed","offset"]]`. To draw for a df:
```
for each row (in df index order):
    prng.seed(row.row_seed)
    if row.offset: prng.rand(row.offset)   # fast-forward: consume prior draws this step
    yield prng.rand(n)                      # the n new draws
row_states.loc[df.index, "offset"] += n     # advance offset
```
- `random_for_df(df, n)` → array shape **`(len(df), n)`**, row order = df index
  order. Each row's `n` draws come from a stream that is **independent per row**
  (its own seed) and **position-tracked** by `offset` within the step.
- `choice_for_df(df, a, size, replace)` → per row `prng.choice(a, size, replace)`,
  concatenated; advances offset by `size` (unless `multi_choice_offset` set).
- `normal_for_df` / `lognormal_for_df` exist for other models (not the dest
  kernel).

### 5.4 What the dest kernel consumes (exact order)
- **Sample stage:** exactly one call `random_for_df(probs, n=sample_size)` →
  `(n_choosers, sample_size)` uniforms in `[0,1)`. Consumed by
  `_sample_choices_maker_preserve_ordering` which **argsorts each row's draws**.
- **Simulate stage:** exactly one call `random_for_df(probs)` (n=1) →
  `(n_choosers, 1)` uniforms. Consumed by `choice_maker`.
- The destination *sample* stage does **not** call `choice_for_df`
  (`_interaction_sample` crosses all alts with
  `interaction_dataset(..., sample_size=alternative_count)`). `choice_for_df`
  is only reached by `interaction_simulate` when `sample_size < len(alts)`.

### 5.5 Consequences for Rust (the good news)
Because each chooser's randomness is **fully determined by its own
`row_seed` + entry `offset`**, the kernel is **embarrassingly parallel and
order-independent**. v1 plan:
1. Python calls the existing `random_for_df(probs, n=...)` to materialize the
   `(n_choosers, sample_size)` (or `(n_choosers,1)`) uniform array.
2. Pass that array to Rust; Rust consumes it **row-major, per chooser**,
   applying `argsort` (sample) or direct cumulative walk (choice) exactly as the
   numba kernels do.
3. Rust output is then independent of thread scheduling → single- vs
   multi-threaded equality is guaranteed by construction.

A later optimization could reimplement MT19937 `rand()`
(double = `((a>>5)*2**26 + (b>>6)) / 2**53` from two 32-bit words) and the
seed/fast-forward in Rust, but **only if** Python draw generation proves to be
a bottleneck. The `prng.choice(replace=False)` path (Fisher–Yates permutation)
is more involved and is **not** needed for the destination-sample kernel.

**Parity anchor for a future Rust RNG:** `activitysim/core/test/test_random.py`
hard-codes the expected draws for fixed seeds — e.g. for persons indexed
`[1..5]` with `base_seed=0` and step `"test_step"`, the first
`random_for_df` returns `[0.1733218, 0.1255693, 0.7384256, 0.3485183,
0.9012387]`, the second (offset-advanced) call returns `[0.9105223, 0.5718418,
0.7222742, 0.9062284, 0.3929369]`, and `choice_for_df` outputs are likewise
pinned. Any Rust MT19937 reimplementation must reproduce these exactly
(also requires bit-identical MD5-based `hash32` and `u32` wrapping arithmetic).

### 5.6 Floating-point / determinism policy (decide explicitly in tests)
- NumPy `sum(axis=1)` uses pairwise summation; naive sequential summation in
  Rust can differ in the last bits. To start, **match NumPy's reduction order**
  (or accept a documented tolerance). Recommend: **bit-identical for choices**
  (the discrete outcome), **within tolerance for probabilities/logsums**.
  Encode the chosen policy in the parity tests (Milestone 2+).

---

## 6. Skim & size-term memory layout (for zero-copy)

Source: `skim_dictionary.py`, `skim_dict_factory.py`, `size_terms.py`.

### 6.1 Skim buffer
- Backing store: contiguous **`numpy.ndarray` (or `memmap`), dtype `float32`**,
  C-order, shape **`(num_skims, n_zones, n_zones)`** when `ROW_MAJOR_LAYOUT`
  (the default). `SkimData` (`skim_dict_factory.py:22`) wraps it and enforces
  3-index access.
- `SkimInfo` holds `block_offsets: dict[key -> int]` (the index into dim 0) and
  `skim_dim3: dict[base_key -> {period -> offset}]` for time-period skims.

### 6.2 Lookup (`skim_dictionary.py:248`)
```
mapped_orig = offset_mapper.map(orig_zone_ids)   # zone id → 0-based offset
mapped_dest = offset_mapper.map(dest_zone_ids)
result = skim_data[block_offset, mapped_orig, mapped_dest]   # fancy index
```
- `OffsetMapper`: usually a scalar `offset_int` (commonly `-1`, i.e. 1-based zone
  ids → 0-based array indices); for sparse/non-contiguous zones it's a
  `pd.Series` mapping. Rust should receive the precomputed **offset arrays**
  (or the scalar offset) rather than re-deriving them.
- 3D: `lookup_3d(orig,dest,dim3,key)` maps `dim3` period strings →
  per-row `block_offsets` via `skim_dim3[key]`, then the same fancy index.
- `SkimWrapper`/`Skim3dWrapper` bind a chooser df's origin/dest/period columns
  so `skims['DIST']` returns a Series aligned to the df index.

### 6.3 Size terms (`size_terms.py`, `DataFrameMatrix` `skim_dictionary.py:844`)
- `tour_destination_size_terms()` returns a DataFrame indexed by `zone_id`, one
  column per segment; each cell is `land_use[attrs] · coefficients`
  (`size_term()` :20).
- `DataFrameMatrix` wraps `df.values` (2D `float64`, shape
  `(n_zones, n_segments)`) plus a `zone_id→row` offset map and a
  `segment_name→col` dict. `get(row_ids, col_ids)` does
  `data[row_indexes, col_indexes]` fancy indexing.
- Rust receives this 2D buffer zero-copy plus the row-offset and
  segment-column maps.

### 6.4 Rust zero-copy summary
| Buffer | dtype / shape / order | index op | passed to Rust as |
|---|---|---|---|
| skim data | f32 `(num_skims, n_zones, n_zones)` C | `[blk, o, d]` | `PyReadonlyArray3<f32>` |
| zone→offset | scalar int or i64 array | `id + off` / `map` | scalar or `PyReadonlyArray1` |
| block_offsets | `dict[key→int]` | O(1) | precomputed i32 array keyed by skim enum |
| dim3 table | `dict[key→{period→int}]` | O(1) | precomputed 2D i32 table |
| size terms | f64 `(n_zones, n_segments)` C | `[zone, seg]` | `PyReadonlyArray2<f64>` + maps |

Never materialize OD matrices; index the shared buffers in place.

---

## 7. Python ↔ Rust boundary (proposed, coarse & numeric)

**Python prepares** (per chunk of choosers): compiled spec IR (op tags +
coefficient arrays), chooser columns as contiguous numeric arrays, alternative
zone ids + size terms, skim buffers (zero-copy) + origin/dest/period index
vectors, **pre-drawn uniform array** from `random_for_df`, `sample_size`, flags
(`allow_zero_probs`, logit type).

**Rust returns** — sampling: `(chooser_index, alt_dest, prob, pick_count)`;
simulate: chosen alt per chooser (+ optional logsum). Python reassembles into
the DataFrames the pipeline expects. The single dispatch point lives in
`interaction_sample.py` / `interaction_sample_simulate.py` behind a
`use_rust_kernel` setting, with automatic fallback to Python (Milestone 5).

---

## 8. Open questions / things to validate before Milestone 2

1. **`fast_eval` semantics** (`core/fast_eval.py`) vs plain pandas `eval` — the
   non-`@` expression path. Confirm the IR matches its numeric behavior
   (dtype promotion, NaN handling) on the dest specs.
2. **dtype path through softmax**: utilities are float64 in interaction eval;
   `prob` is cast to float32 on output. Confirm where casts happen so the IR
   reproduces them (affects the FP tolerance policy).
3. **`allow_zero_probs` rows**: choosers whose probs all sum to 0 are dropped
   in `make_sample_choices` (and handled in sample_simulate via
   `zero_prob_choice_val`). Replicate exactly.
4. **`uniquify_spec_index`**: duplicate expression strings are made unique;
   ensure the IR compiler consumes the post-uniquified spec.
5. **Multizone / TAZ presampling** path (`destination_presample`) and the
   `MazSkimDict` facade — out of scope for v1 but note for Milestone 7.
6. **Sharrow**: when `settings.sharrow` is enabled the eval path differs
   (`apply_flow`). v1 should target the non-sharrow path and either disable
   sharrow or fall back when it's on.

---

*End of reconnaissance. Per the plan (Section 1 & Section 10), no Rust is to be
written until this document is reviewed.*
