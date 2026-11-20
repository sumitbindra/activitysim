from pathlib import Path

import numpy as np
import pytest

from asim_core.spec import ChoiceModelDefinition, Coefficient, Term, load_choice_model

MTC_CONFIGS = Path(__file__).resolve().parents[2] / "activitysim/examples/prototype_mtc/configs"


def definition(**changes):
    kw = dict(
        name="toy",
        alternatives=["a", "b"],
        terms=[Term(label="t1", formula="@df.x.clip(0, 3)", coefficients={"b": "coef_x"})],
        coefficients={"coef_x": Coefficient(value=0.1)},
    )
    kw.update(changes)
    return ChoiceModelDefinition(**kw)


def test_valid_definition_loads():
    d = definition()
    assert d.columns_used() == {"x"}


def test_all_problems_are_reported_together():
    with pytest.raises(ValueError) as err:
        definition(terms=[
            Term(label="t1", formula="@df.x.rolling(2)", coefficients={"b": "coef_missing"}),
            Term(label="t2", formula="y > 1", coefficients={"c": "coef_x"}),
        ])
    msg = str(err.value)
    assert "3 problem(s)" in msg
    assert "unknown coefficient 'coef_missing'" in msg
    assert "unknown alternative 'c'" in msg
    assert "rolling" in msg


def test_coefficients_are_rounded_to_float32_like_1x():
    d = definition()
    m = d.coefficient_matrix()
    assert m[0, 1] == np.float64(np.float32(0.1))
    assert m[0, 0] == 0.0
    assert d.coefficient_matrix(float32_rounding=False)[0, 1] == 0.1


def test_coefficient_cells_can_be_arithmetic():
    d = definition(terms=[Term(formula="1", coefficients={"a": "2 * coef_x", "b": "-1.5"})])
    np.testing.assert_allclose(d.coefficient_matrix(float32_rounding=False), [[0.2, -1.5]])


@pytest.mark.skipif(not MTC_CONFIGS.exists(), reason="example configs not present")
def test_todays_auto_ownership_files_load():
    d = load_choice_model([MTC_CONFIGS], "auto_ownership.yaml")
    assert d.alternatives == ["cars0", "cars1", "cars2", "cars3", "cars4"]
    assert len(d.terms) == 29
    assert "county_id" in d.columns_used()
    _, source = d.utility_function()
    assert "def auto_ownership_terms(c, s=None, k=None):" in source


@pytest.mark.skipif(not MTC_CONFIGS.exists(), reason="example configs not present")
def test_unsupported_settings_are_named():
    # tour mode choice has a preprocessor, which only its own component handles
    with pytest.raises(NotImplementedError, match="preprocessor"):
        load_choice_model([MTC_CONFIGS], "tour_mode_choice.yaml")


@pytest.mark.skipif(not MTC_CONFIGS.exists(), reason="example configs not present")
def test_coefficient_template_segments():
    d = load_choice_model([MTC_CONFIGS], "tour_mode_choice.yaml", handled=["preprocessor"],
                          formulas_see_coefficients=True,
                          skim_accessors=["odt_skims", "dot_skims", "odr_skims", "dor_skims",
                                          "od_skims", "do_skims"])
    assert "work" in d.segments and "univ" in d.segments
    work, atwork = d.coefficient_values("work"), d.coefficient_values("atwork")
    # a coefficient that differs by segment, and an empty template cell (same name for all)
    assert work["coef_topology_walk_multiplier"] == 15.0
    assert atwork["coef_topology_walk_multiplier"] == 7.5
    assert work["coef_one"] == atwork["coef_one"] == 1.0
    # formulas can use coefficient names; their values are supplied per segment
    tr = next(t for t in d.translated if "coef_topology_walk_multiplier" in t.formula)
    assert "k['coef_topology_walk_multiplier']" in tr.source
    assert d.nest_tree("work").coefficient == work["coef_nest_root"]
    assert not (d.coefficient_matrix(segment="work") == d.coefficient_matrix(segment="school")).all()
