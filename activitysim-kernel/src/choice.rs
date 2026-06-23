//! Pure-Rust ports of ActivitySim's choice/sampling numerics.
//!
//! These mirror, operation-for-operation, the reference implementations in:
//!   * `activitysim/core/logit.py`        -> [`utils_to_probs`]
//!   * `activitysim/core/choosing.py`     -> [`choice_maker`],
//!                                           [`sample_choices_maker_preserve_ordering`]
//!
//! Randomness is *passed in* (pre-drawn by Python via `random_for_df`), so these
//! functions are deterministic given their inputs and contain no RNG. See
//! `docs/rust_kernel_recon.md` section 5 for why this guarantees that
//! single-threaded and multi-threaded runs are identical.

/// Smallest exponentiated utility kept; anything `<=` this becomes exactly 0.0.
/// Mirrors `EXP_UTIL_MIN` in `logit.py`.
pub const EXP_UTIL_MIN: f64 = 1e-300;

/// Convert a row-major `(n_rows, n_cols)` matrix of utilities into choice
/// probabilities, in place. Mirrors `logit.utils_to_probs`.
///
/// * `utils` is mutated to hold the probabilities on return.
/// * If `overflow_protection` is true, each row is shifted by its max before
///   exponentiation (the standard softmax overflow trick).
/// * If `return_logsums` is true, the per-row logsum is returned, computed as
///   `log(sum(exp(shifted))) + shift` so it equals `log(sum(exp(util)))`.
///
/// `allow_zero_probs` mirrors the Python flag: when true, rows whose
/// exponentiated utilities all underflow to zero are left as all-zeros (rather
/// than producing NaNs); when false such rows would be reported as bad choices
/// upstream — here we simply leave them as zeros, the caller decides.
pub fn utils_to_probs(
    utils: &mut [f64],
    n_rows: usize,
    n_cols: usize,
    overflow_protection: bool,
    return_logsums: bool,
) -> Option<Vec<f64>> {
    assert_eq!(utils.len(), n_rows * n_cols, "utils length mismatch");
    let mut logsums = if return_logsums {
        Some(vec![0.0_f64; n_rows])
    } else {
        None
    };

    for r in 0..n_rows {
        let row = &mut utils[r * n_cols..(r + 1) * n_cols];

        // 1. overflow protection: subtract the per-row max
        let shift = if overflow_protection {
            let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for v in row.iter_mut() {
                *v -= m;
            }
            m
        } else {
            0.0
        };

        // 2. exponentiate, flushing tiny values to exactly zero
        for v in row.iter_mut() {
            let e = v.exp();
            *v = if e <= EXP_UTIL_MIN { 0.0 } else { e };
        }

        // 3. row sum
        let sum: f64 = row.iter().sum();

        if let Some(ref mut ls) = logsums {
            // log(sum(exp(shifted))) + shift  ==  log(sum(exp(util)))
            ls[r] = sum.ln() + shift;
        }

        // 4. normalize -> probabilities, clamp to [0, 1], NaN -> 0
        for v in row.iter_mut() {
            let p = *v / sum;
            *v = if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
        }
    }

    logsums
}

/// Make one choice per row given a `(n_rows, n_cols)` row-major probability
/// matrix and one uniform random number per row. Mirrors
/// `choosing.choice_maker`.
///
/// Returns a vector of chosen column indices, one per row.
pub fn choice_maker(pr: &[f64], n_rows: usize, n_cols: usize, rn: &[f64]) -> Vec<i64> {
    assert_eq!(pr.len(), n_rows * n_cols, "pr length mismatch");
    assert_eq!(rn.len(), n_rows, "rn length mismatch");
    let mut out = vec![0_i64; n_rows];

    for row in 0..n_rows {
        let base = row * n_cols;
        let mut z = rn[row];
        let mut chosen: Option<usize> = None;
        for col in 0..n_cols {
            z -= pr[base + col];
            if z <= 0.0 {
                chosen = Some(col);
                break;
            }
        }
        out[row] = match chosen {
            Some(c) => c as i64,
            None => {
                // Fallback (numerical edge case): pick the argmax probability.
                let mut max_pr = 0.0_f64;
                let mut best = 0usize;
                for col in 0..n_cols {
                    if pr[base + col] > max_pr {
                        max_pr = pr[base + col];
                        best = col;
                    }
                }
                best as i64
            }
        };
    }
    out
}

/// Result of [`sample_choices_maker_preserve_ordering`]: for each
/// `(chooser, sample)` pair, the chosen alternative id and its probability,
/// laid out row-major as `(n_choosers, sample_size)`.
pub struct SampleChoices {
    pub choices: Vec<i64>,
    pub choice_probs: Vec<f64>,
}

/// Draw `sample_size` alternatives per chooser by inverse-CDF sampling,
/// preserving the original ordering of the random draws. Mirrors
/// `choosing._sample_choices_maker_preserve_ordering`.
///
/// * `prob_array` is row-major `(n_choosers, n_alts)`.
/// * `random_array` is row-major `(n_choosers, sample_size)` of uniforms in
///   `[0, 1)` (pre-drawn by Python).
/// * `alts_array` holds the `n_alts` alternative ids.
///
/// Output arrays are `(n_choosers, sample_size)` row-major. Note the legacy
/// numba kernel materializes a transposed `(sample_size, n_choosers)` buffer
/// internally; the *mapping of each draw to an alternative* is identical here.
pub fn sample_choices_maker_preserve_ordering(
    prob_array: &[f64],
    n_choosers: usize,
    n_alts: usize,
    random_array: &[f64],
    sample_size: usize,
    alts_array: &[i64],
) -> SampleChoices {
    assert_eq!(prob_array.len(), n_choosers * n_alts, "prob_array length");
    assert_eq!(
        random_array.len(),
        n_choosers * sample_size,
        "random_array length"
    );
    assert_eq!(alts_array.len(), n_alts, "alts_array length");

    let mut choices = vec![0_i64; n_choosers * sample_size];
    let mut choice_probs = vec![0.0_f64; n_choosers * sample_size];

    for c in 0..n_choosers {
        let p_base = c * n_alts;
        let r_base = c * sample_size;
        let rands = &random_array[r_base..r_base + sample_size];

        // argsort of this chooser's random draws (ascending). Random doubles
        // are distinct in practice; ties resolve by index for stability.
        let mut order: Vec<usize> = (0..sample_size).collect();
        order.sort_by(|&a, &b| {
            rands[a]
                .partial_cmp(&rands[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mut s = 0usize;
        let mut z = 0.0_f64;
        let mut a = 0usize;
        while a < n_alts {
            z += prob_array[p_base + a];
            while s < sample_size && z > rands[order[s]] {
                let orig = order[s];
                choices[r_base + orig] = alts_array[a];
                choice_probs[r_base + orig] = prob_array[p_base + a];
                s += 1;
            }
            if s >= sample_size {
                break;
            }
            a += 1;
        }
        if s < sample_size {
            // Rare numerical edge case: slip back to the last non-trivial alt.
            let mut a = n_alts - 1;
            while prob_array[p_base + a] < 1e-30 && a > 0 {
                a -= 1;
            }
            while s < sample_size {
                let orig = order[s];
                choices[r_base + orig] = alts_array[a];
                choice_probs[r_base + orig] = prob_array[p_base + a];
                s += 1;
            }
        }
    }

    SampleChoices {
        choices,
        choice_probs,
    }
}
