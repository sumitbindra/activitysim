//! Numeric parity tests for the pure-Rust choice/sampling core.
//!
//! Expected values are computed by hand from the reference algorithms in
//! `activitysim/core/logit.py` and `activitysim/core/choosing.py`, so these run
//! with `cargo test --no-default-features` (no Python required).

use activitysim_kernel::choice::{
    choice_maker, sample_choices_maker_preserve_ordering, utils_to_probs,
};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn utils_to_probs_softmax_with_overflow_protection() {
    // utils = [1, 2, 3]; shift by max=3 -> [-2,-1,0]; exp/sum -> softmax.
    let mut u = vec![1.0, 2.0, 3.0];
    let logsums = utils_to_probs(&mut u, 1, 3, true, true).unwrap();

    // exp([-2,-1,0]) = [0.13533528, 0.36787944, 1.0]; sum = 1.50321472
    let expected = [0.0900306, 0.2447285, 0.6652409];
    for (got, exp) in u.iter().zip(expected.iter()) {
        assert!(approx(*got, *exp, 1e-6), "prob got {got} want {exp}");
    }
    // probabilities sum to 1
    assert!(approx(u.iter().sum::<f64>(), 1.0, 1e-12));
    // logsum = log(e^1 + e^2 + e^3) = 3.4076059
    assert!(approx(logsums[0], 3.4076059, 1e-6), "logsum {}", logsums[0]);
}

#[test]
fn utils_to_probs_no_overflow_protection_matches() {
    // Without the shift, the resulting probabilities are identical (softmax is
    // shift-invariant); only intermediate magnitudes differ.
    let mut a = vec![1.0, 2.0, 3.0];
    let mut b = vec![1.0, 2.0, 3.0];
    utils_to_probs(&mut a, 1, 3, true, false);
    utils_to_probs(&mut b, 1, 3, false, false);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(approx(*x, *y, 1e-12), "{x} vs {y}");
    }
}

#[test]
fn choice_maker_inverse_cdf() {
    // probs row = [0.2, 0.5, 0.3].
    let pr = vec![0.2, 0.5, 0.3];
    // rand 0.1 -> col 0; 0.25 -> col 1; 0.95 -> col 2.
    let out = choice_maker(&pr.repeat(3), 3, 3, &[0.1, 0.25, 0.95]);
    assert_eq!(out, vec![0, 1, 2]);
}

#[test]
fn choice_maker_fallback_when_probs_short() {
    // Probabilities sum to < 1 and rand exceeds the sum -> argmax fallback.
    let pr = vec![0.1, 0.2, 0.0]; // argmax is col 1
    let out = choice_maker(&pr, 1, 3, &[0.99]);
    assert_eq!(out, vec![1]);
}

#[test]
fn sample_choices_preserve_ordering_basic() {
    // probs = [0.2, 0.5, 0.3], alts ids = [10, 20, 30], sample_size = 2.
    // rands (in original order) = [0.9, 0.1].
    // sorted ascending -> [idx1 (0.1), idx0 (0.9)].
    //   alt0 cumsum 0.2 > 0.1  -> draw at orig pos 1 gets alt 10 (prob 0.2)
    //   alt2 cumsum 1.0 > 0.9  -> draw at orig pos 0 gets alt 30 (prob 0.3)
    let pr = vec![0.2, 0.5, 0.3];
    let alts = vec![10_i64, 20, 30];
    let res = sample_choices_maker_preserve_ordering(&pr, 1, 3, &[0.9, 0.1], 2, &alts);
    assert_eq!(res.choices, vec![30, 10]); // original-position order preserved
    assert!(approx(res.choice_probs[0], 0.3, 1e-12));
    assert!(approx(res.choice_probs[1], 0.2, 1e-12));
}

#[test]
fn sample_choices_two_choosers_independent() {
    // Two choosers with identical probs but different draws; results depend only
    // on each chooser's own draws (the determinism property).
    let pr = vec![0.2, 0.5, 0.3, 0.2, 0.5, 0.3];
    let alts = vec![10_i64, 20, 30];
    let rands = vec![0.9, 0.1 /*chooser 0*/, 0.05, 0.6 /*chooser 1*/];
    let res = sample_choices_maker_preserve_ordering(&pr, 2, 3, &rands, 2, &alts);
    // chooser 0 (same as previous test): [30, 10]
    assert_eq!(&res.choices[0..2], &[30, 10]);
    // chooser 1: sorted [0.05, 0.6] -> 0.05<0.2 -> alt10; 0.6<0.7 (0.2+0.5) -> alt20
    // original order rands=[0.05,0.6] already ascending so positions preserved.
    assert_eq!(&res.choices[2..4], &[10, 20]);
}
