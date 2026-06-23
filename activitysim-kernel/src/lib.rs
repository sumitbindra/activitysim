//! ActivitySim Rust kernel.
//!
//! Surgical Rust port of the destination/location choice hot path. The numeric
//! core (this crate's modules) is pure Rust and unit-tested with `cargo test`.
//! The Python-importable extension module is gated behind the `python` feature
//! (PyO3 + rust-numpy) so the pure-Rust core can be built and tested on machines
//! without a Python toolchain, and so the surrounding ActivitySim package always
//! builds even when no Rust toolchain is present.
//!
//! See `docs/rust_kernel_recon.md` for the design and the contract this mirrors.

pub mod choice;
pub mod error;
pub mod spec;

/// Crate version, surfaced to Python for diagnostics.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Python bindings (optional)
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
mod python {
    use crate::choice;
    use numpy::ndarray::Array2;
    use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;

    /// Return the kernel version string (sanity check that the extension loaded).
    #[pyfunction]
    fn version() -> &'static str {
        crate::VERSION
    }

    /// Round-trip a 1-D float array (Milestone 0 hello-world: proves zero-copy
    /// array passing works end to end).
    #[pyfunction]
    fn echo_f64<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
        let v = x.as_array().to_vec();
        PyArray1::from_vec(py, v)
    }

    /// Make one choice per row given a probability matrix and one uniform per
    /// row. Thin wrapper over [`choice::choice_maker`].
    #[pyfunction]
    fn make_choices<'py>(
        py: Python<'py>,
        probs: PyReadonlyArray2<'py, f64>,
        rands: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<i64>> {
        let p = probs.as_array();
        let (n_rows, n_cols) = (p.shape()[0], p.shape()[1]);
        let p_flat: Vec<f64> = p.iter().copied().collect();
        let r: Vec<f64> = rands.as_array().to_vec();
        let out = choice::choice_maker(&p_flat, n_rows, n_cols, &r);
        PyArray1::from_vec(py, out)
    }

    /// Convert utilities to probabilities in place-style (returns a new array).
    /// Thin wrapper over [`choice::utils_to_probs`].
    #[pyfunction]
    #[pyo3(signature = (utils, overflow_protection=true))]
    fn utils_to_probs<'py>(
        py: Python<'py>,
        utils: PyReadonlyArray2<'py, f64>,
        overflow_protection: bool,
    ) -> Bound<'py, PyArray2<f64>> {
        let u = utils.as_array();
        let (n_rows, n_cols) = (u.shape()[0], u.shape()[1]);
        let mut flat: Vec<f64> = u.iter().copied().collect();
        choice::utils_to_probs(&mut flat, n_rows, n_cols, overflow_protection, false);
        let arr = Array2::from_shape_vec((n_rows, n_cols), flat).expect("shape");
        PyArray2::from_owned_array(py, arr)
    }

    #[pymodule]
    fn activitysim_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", crate::VERSION)?;
        m.add_function(wrap_pyfunction!(version, m)?)?;
        m.add_function(wrap_pyfunction!(echo_f64, m)?)?;
        m.add_function(wrap_pyfunction!(make_choices, m)?)?;
        m.add_function(wrap_pyfunction!(utils_to_probs, m)?)?;
        Ok(())
    }
}
