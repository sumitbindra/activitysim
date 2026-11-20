import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asim_core.logit import nested_probabilities, probabilities
from asim_core.spec.definition import ChoiceModelDefinition, Coefficient, NestTree, Term

ARC_CONFIGS = Path(__file__).resolve().parents[2] / "activitysim/examples/prototype_arc/configs"

# three levels, coefficients named and literal, alternatives listed out of order
NESTS = {
    "name": "root", "coefficient": 1.0, "alternatives": [
        {"name": "AUTO", "coefficient": "coef_nest_auto", "alternatives": [
            {"name": "DRIVE", "coefficient": 0.35, "alternatives": ["sov", "hov"]},
            "taxi",
        ]},
        {"name": "TRANSIT", "coefficient": 0.6, "alternatives": ["bus", "rail", "ferry"]},
        "walk",
    ],
}
ALTS = ["walk", "sov", "hov", "taxi", "bus", "rail", "ferry"]


def definition(nests=NESTS, alternatives=ALTS, **kw):
    return ChoiceModelDefinition(
        name="toy", logit_type="NL", alternatives=alternatives, nests=nests,
        terms=[Term(formula="1", coefficients={alternatives[0]: "0"})],
        coefficients={"coef_nest_auto": Coefficient(value=0.72)}, **kw,
    )


def random_utils(n=2000, k=len(ALTS), seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(scale=3.0, size=(n, k))
    u[rng.random((n, k)) < 0.05] = -999.0  # some unavailable alternatives
    return u


def test_probabilities_sum_to_one():
    probs, bad = nested_probabilities(random_utils(), ALTS, definition().nest_tree())
    assert not bad.any()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)


def test_all_coefficients_one_is_multinomial_logit():
    flat = NestTree("root", 1.0, [NestTree("A", 1.0, ALTS[:3]), NestTree("B", 1.0, ALTS[3:])])
    u = random_utils()
    nl, _ = nested_probabilities(u, ALTS, flat)
    mnl, _ = probabilities(u)
    np.testing.assert_allclose(nl, mnl, rtol=1e-12, atol=1e-15)


def test_nest_problems_are_reported_at_load():
    bad = {"name": "root", "coefficient": "coef_missing", "alternatives": [
        {"name": "A", "coefficient": 0.5, "alternatives": ["walk", "walk"]}, "spaceship"]}
    with pytest.raises(ValueError) as err:
        definition(nests=bad)
    msg = str(err.value)
    assert "unknown coefficient 'coef_missing'" in msg
    assert "used more than once" in msg
    assert "not in any nest" in msg
    assert "unknown alternatives: ['spaceship']" in msg
    with pytest.raises(ValueError, match="needs NESTS"):
        definition(nests=None)


def test_coefficients_resolve_by_name_and_numbers_in_text():
    nests = dict(NESTS, coefficient="1.0")  # a number written as text, as YAML may give
    tree = definition(nests=nests).nest_tree()
    assert tree.coefficient == 1.0
    assert tree.alternatives[0].coefficient == 0.72  # coef_nest_auto, from the coefficients


def test_matches_activitysim_1x_bit_for_bit():
    simulate = pytest.importorskip("activitysim.core.simulate")
    from activitysim.core import workflow
    from activitysim.core.configuration import Settings

    state = workflow.State()
    state.settings = Settings()
    u = random_utils()
    spec = pd.DataFrame(columns=ALTS)
    nest_spec = simulate.eval_nest_coefficients(  # changes its argument: give it a copy
        copy.deepcopy(NESTS), {"coef_nest_auto": 0.72}, "test")
    raw = pd.DataFrame(u, columns=ALTS)
    exp_u = simulate.compute_nested_exp_utilities(raw, nest_spec)
    nested = simulate.compute_nested_probabilities(state, exp_u, nest_spec, "test")
    ref = simulate.compute_base_probabilities(nested, nest_spec, spec).to_numpy()

    ours, _ = nested_probabilities(u, ALTS, definition().nest_tree())
    assert np.array_equal(ours, ref)


@pytest.mark.skipif(not ARC_CONFIGS.exists(), reason="ARC example not present")
def test_arc_auto_ownership_loads_as_nested_logit():
    from asim_core.spec import load_choice_model

    d = load_choice_model([ARC_CONFIGS], "auto_ownership.yaml")
    assert d.logit_type == "NL"
    tree = d.nest_tree()
    assert [n.name for n in tree.alternatives] == ["LOW_CAR", "MORE_CAR"]
    assert tree.alternatives[1].coefficient == 0.507
