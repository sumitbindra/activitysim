//! Parity tests for the Rust IR evaluator (`spec.rs`).
//!
//! These build IR trees equivalent to the representative destination-choice
//! expressions and check the evaluator against a direct Rust computation of the
//! same formula. Combined with the Python parity test
//! (`activitysim/core/test/test_choice_ir.py`), which proves the IR *semantics*
//! match numpy, this confirms the Rust tree-walker composes them correctly.

use activitysim_kernel::spec::{compute_utilities, evaluate, Cmp, Fun, Inputs, Node, Op, Un};

fn approx_slice(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() <= tol, "row {i}: {x} vs {y}");
    }
}

fn b(n: Node) -> Box<Node> {
    Box::new(n)
}

#[test]
fn clip_shifted_skim() {
    // (skims['DIST'] - 1).clip(0, 3)
    let dist = [0.0, 1.5, 2.0, 5.0];
    let cols: Vec<&[f64]> = vec![&dist];
    let inputs = Inputs::new(&cols, 4);

    let node = Node::Func(
        Fun::Clip,
        vec![
            Node::Bin(Op::Sub, b(Node::Input(0)), b(Node::Num(1.0))),
            Node::Num(0.0),
            Node::Num(3.0),
        ],
    );
    let got = evaluate(&node, &inputs);
    let expect: Vec<f64> = dist.iter().map(|d| (d - 1.0).clamp(0.0, 3.0)).collect();
    approx_slice(&got, &expect, 1e-12);
}

#[test]
fn bool_gate_times_sum() {
    // (~is_joint & outbound) * (od + dp)
    let is_joint = [1.0, 0.0, 0.0, 1.0];
    let outbound = [1.0, 1.0, 0.0, 0.0];
    let od = [3.0, 4.0, 5.0, 6.0];
    let dp = [1.0, 1.0, 1.0, 1.0];
    let cols: Vec<&[f64]> = vec![&is_joint, &outbound, &od, &dp];
    let inputs = Inputs::new(&cols, 4);

    let gate = Node::Bin(
        Op::And,
        b(Node::Unary(Un::Invert, b(Node::Input(0)))),
        b(Node::Input(1)),
    );
    let sum = Node::Bin(Op::Add, b(Node::Input(2)), b(Node::Input(3)));
    let node = Node::Bin(Op::Mul, b(gate), b(sum));

    let got = evaluate(&node, &inputs);
    let expect: Vec<f64> = (0..4)
        .map(|i| {
            let g = if is_joint[i] == 0.0 && outbound[i] != 0.0 {
                1.0
            } else {
                0.0
            };
            g * (od[i] + dp[i])
        })
        .collect();
    approx_slice(&got, &expect, 1e-12);
    // only row 1 (not joint, outbound) is active
    approx_slice(&got, &[0.0, 5.0, 0.0, 0.0], 1e-12);
}

#[test]
fn where_and_minimum_log() {
    // np.where(outbound, od, dp)
    let outbound = [1.0, 0.0, 1.0];
    let od = [10.0, 20.0, 30.0];
    let dp = [1.0, 2.0, 3.0];
    let pick = [4.0, 9.0, 2.0];
    let prob = [0.5, 0.1, 1.0];
    let cols: Vec<&[f64]> = vec![&outbound, &od, &dp, &pick, &prob];
    let inputs = Inputs::new(&cols, 3);

    let wnode = Node::Func(
        Fun::Where,
        vec![Node::Input(0), Node::Input(1), Node::Input(2)],
    );
    let got = evaluate(&wnode, &inputs);
    approx_slice(&got, &[10.0, 2.0, 30.0], 1e-12);

    // np.minimum(np.log(pick / prob), 60)
    let mnode = Node::Func(
        Fun::Minimum,
        vec![
            Node::Func(
                Fun::Log,
                vec![Node::Bin(Op::Div, b(Node::Input(3)), b(Node::Input(4)))],
            ),
            Node::Num(60.0),
        ],
    );
    let got = evaluate(&mnode, &inputs);
    let expect: Vec<f64> = (0..3).map(|i| (pick[i] / prob[i]).ln().min(60.0)).collect();
    approx_slice(&got, &expect, 1e-12);
}

#[test]
fn equality_produces_bool() {
    // size_term == 0  -> 1.0 where zero else 0.0
    let size_term = [0.0, 1.0, 0.0, 5.0];
    let cols: Vec<&[f64]> = vec![&size_term];
    let inputs = Inputs::new(&cols, 4);
    let node = Node::Compare(Cmp::Eq, b(Node::Input(0)), b(Node::Num(0.0)));
    let got = evaluate(&node, &inputs);
    approx_slice(&got, &[1.0, 0.0, 1.0, 0.0], 1e-12);
}

#[test]
fn linear_utility_assembly() {
    // utilities = -0.8*clip(DIST,0,1) + 1.0*log1p(size_term)
    let dist = [0.2, 0.9, 2.0];
    let size_term = [0.0, 10.0, 100.0];
    let cols: Vec<&[f64]> = vec![&dist, &size_term];
    let inputs = Inputs::new(&cols, 3);

    let terms = vec![
        (
            Node::Func(
                Fun::Clip,
                vec![Node::Input(0), Node::Num(0.0), Node::Num(1.0)],
            ),
            -0.8,
        ),
        (Node::Func(Fun::Log1p, vec![Node::Input(1)]), 1.0),
    ];
    let got = compute_utilities(&terms, &inputs);
    let expect: Vec<f64> = (0..3)
        .map(|i| -0.8 * dist[i].clamp(0.0, 1.0) + 1.0 * size_term[i].ln_1p())
        .collect();
    approx_slice(&got, &expect, 1e-12);
}
