//! Rust evaluation of a compiled utility-spec IR (Milestone 2).
//!
//! This mirrors, node-for-node, the reference IR in
//! `activitysim/core/choice_ir.py`. The Python↔Rust boundary is deliberately
//! coarse: Python materializes every *leaf* value that needs pandas/skim
//! machinery (chooser/alt columns, resolved skim lookups, size-term lookups)
//! into a set of per-row input vectors, then passes the arithmetic tree with
//! [`Node::Input`] leaves referencing those vectors by index. Rust performs the
//! per-row arithmetic, comparisons and function applications.
//!
//! Booleans are represented as `0.0`/`1.0` so the whole evaluation is `f64`,
//! matching numpy's behavior when boolean arrays participate in arithmetic.
//!
//! Skim/size-term lookups *inside* Rust (zero-copy over the shared 3-D skim
//! buffer, recon §6) are a later optimization (Milestone 6); v1 keeps them in
//! Python for correctness.

/// Binary arithmetic / bitwise operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    And,
    Or,
}

/// Comparison operators (produce 0.0/1.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Un {
    Neg,
    Invert,
}

/// Supported functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fun {
    Log,
    Log1p,
    Exp,
    Abs,
    Sqrt,
    Clip,    // (x, lo, hi)
    Minimum, // (a, b)
    Maximum, // (a, b)
    Where,   // (cond, a, b)
}

/// An IR node. Leaves are numeric constants or references to pre-materialized
/// per-row input vectors.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Num(f64),
    Input(usize),
    Bin(Op, Box<Node>, Box<Node>),
    Unary(Un, Box<Node>),
    Compare(Cmp, Box<Node>, Box<Node>),
    Func(Fun, Vec<Node>),
}

/// Per-row input vectors, addressed by `Node::Input(idx)`. All vectors must
/// have length `n_rows`.
pub struct Inputs<'a> {
    pub cols: &'a [&'a [f64]],
    pub n_rows: usize,
}

impl<'a> Inputs<'a> {
    pub fn new(cols: &'a [&'a [f64]], n_rows: usize) -> Self {
        for c in cols {
            assert_eq!(c.len(), n_rows, "input column length mismatch");
        }
        Inputs { cols, n_rows }
    }
}

fn bool_f64(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// Evaluate an IR node into a per-row vector (length `inputs.n_rows`).
pub fn evaluate(node: &Node, inputs: &Inputs) -> Vec<f64> {
    let n = inputs.n_rows;
    match node {
        Node::Num(v) => vec![*v; n],
        Node::Input(i) => inputs.cols[*i].to_vec(),
        Node::Unary(op, x) => {
            let mut v = evaluate(x, inputs);
            match op {
                Un::Neg => v.iter_mut().for_each(|e| *e = -*e),
                Un::Invert => v.iter_mut().for_each(|e| *e = bool_f64(*e == 0.0)),
            }
            v
        }
        Node::Bin(op, a, b) => {
            let av = evaluate(a, inputs);
            let bv = evaluate(b, inputs);
            let mut out = vec![0.0; n];
            for r in 0..n {
                out[r] = apply_bin(*op, av[r], bv[r]);
            }
            out
        }
        Node::Compare(op, a, b) => {
            let av = evaluate(a, inputs);
            let bv = evaluate(b, inputs);
            let mut out = vec![0.0; n];
            for r in 0..n {
                out[r] = bool_f64(apply_cmp(*op, av[r], bv[r]));
            }
            out
        }
        Node::Func(f, args) => {
            let evaled: Vec<Vec<f64>> = args.iter().map(|a| evaluate(a, inputs)).collect();
            let mut out = vec![0.0; n];
            for r in 0..n {
                out[r] = apply_func(*f, &evaled, r);
            }
            out
        }
    }
}

fn apply_bin(op: Op, a: f64, b: f64) -> f64 {
    match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Div => a / b,
        Op::Pow => a.powf(b),
        Op::And => bool_f64(a != 0.0 && b != 0.0),
        Op::Or => bool_f64(a != 0.0 || b != 0.0),
    }
}

fn apply_cmp(op: Cmp, a: f64, b: f64) -> bool {
    match op {
        Cmp::Eq => a == b,
        Cmp::Ne => a != b,
        Cmp::Lt => a < b,
        Cmp::Le => a <= b,
        Cmp::Gt => a > b,
        Cmp::Ge => a >= b,
    }
}

fn apply_func(f: Fun, args: &[Vec<f64>], r: usize) -> f64 {
    match f {
        Fun::Log => args[0][r].ln(),
        Fun::Log1p => args[0][r].ln_1p(),
        Fun::Exp => args[0][r].exp(),
        Fun::Abs => args[0][r].abs(),
        Fun::Sqrt => args[0][r].sqrt(),
        Fun::Clip => args[0][r].clamp(args[1][r], args[2][r]),
        Fun::Minimum => args[0][r].min(args[1][r]),
        Fun::Maximum => args[0][r].max(args[1][r]),
        Fun::Where => {
            if args[0][r] != 0.0 {
                args[1][r]
            } else {
                args[2][r]
            }
        }
    }
}

/// Accumulate a linear-in-parameters utility: `sum_i coef[i] * eval(terms[i])`.
/// Returns a per-row utility vector. Mirrors how `eval_interaction_utilities`
/// sums coefficient-weighted expression values.
pub fn compute_utilities(terms: &[(Node, f64)], inputs: &Inputs) -> Vec<f64> {
    let mut util = vec![0.0_f64; inputs.n_rows];
    for (node, coef) in terms {
        let v = evaluate(node, inputs);
        for r in 0..inputs.n_rows {
            util[r] += coef * v[r];
        }
    }
    util
}
