# activitysim-kernel

Optional Rust kernel for ActivitySim's destination/location **choice hot path**
(the interaction utility evaluation + softmax + sampling/choice). It is a
*surgical* extension: orchestration, mode-choice logsums, shadow pricing, config
and spec parsing, RNG channel management, and all I/O stay in Python. See
[`../docs/rust_kernel_recon.md`](../docs/rust_kernel_recon.md) for the design and
the exact Python contract this mirrors.

## Status

Milestone 0 (scaffolding) + the pure-Rust numeric core for the choice/sampling
kernels:

* `utils_to_probs` — softmax with per-row overflow protection and logsums
  (mirrors `activitysim/core/logit.py`).
* `choice_maker` — inverse-CDF single choice (mirrors
  `activitysim/core/choosing.py`).
* `sample_choices_maker_preserve_ordering` — order-preserving alternative
  sampling (mirrors `activitysim/core/choosing.py`).

Randomness is **passed in** (pre-drawn by Python via `random_for_df`), so the
kernel is deterministic given its inputs — this is what makes parallelism over
choosers safe (recon §5).

## Layout

```
activitysim-kernel/
  Cargo.toml          # crate-type cdylib + rlib; PyO3/numpy behind `python` feature
  pyproject.toml      # maturin build config (builds the `activitysim_kernel` module)
  src/
    lib.rs            # module wiring + PyO3 bindings (feature = "python")
    choice.rs         # utils_to_probs / choice_maker / sample_choices_maker
    error.rs          # KernelError
  tests/parity.rs     # numeric parity tests (run with --no-default-features)
```

## Building / testing

The numeric core is pure Rust and needs **no** Python toolchain:

```bash
cargo test --no-default-features          # verify the numeric core
```

To build the importable Python extension (requires a Python dev environment and
NumPy installed):

```bash
pip install maturin
maturin develop --release                 # installs `import activitysim_kernel`
```

The Rust path is **strictly optional**: ActivitySim imports `activitysim_kernel`
lazily and falls back to the existing pure-Python code path when the extension is
not built (see `activitysim/core/rust_kernel.py`). Contributors without a Rust
toolchain are unaffected.
