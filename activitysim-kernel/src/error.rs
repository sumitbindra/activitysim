//! Error type for the kernel. Kept minimal for the scaffolding milestone.

use std::fmt;

#[derive(Debug)]
pub enum KernelError {
    /// A shape/length invariant was violated at the Python boundary.
    Shape(String),
    /// The compiled spec referenced an operation the IR does not support.
    UnsupportedOp(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::Shape(m) => write!(f, "shape error: {m}"),
            KernelError::UnsupportedOp(m) => write!(f, "unsupported op: {m}"),
        }
    }
}

impl std::error::Error for KernelError {}
