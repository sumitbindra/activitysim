"""
CDAP's and tour creation's pieces, against 1.x's own functions where they
can run without a 1.x state (skipped if ActivitySim 1.x isn't installed).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asim_core.components.cdap import (
    _interaction_code,
    assign_cdap_rank,
    build_cdap_spec,
    build_cdap_spec_numpy,
    read_interaction_coefficients,
)
from asim_core.components.tours import create_tours, set_tour_index
from asim_core.spec.legacy import load_choice_model

REPO = Path(__file__).resolve().parents[2]
PERSON_TYPE_MAP = {"WORKER": [1, 2], "CHILD": [6, 7, 8]}


def _persons(n_hh=300, seed=5):
    rng = np.random.default_rng(seed)
    sizes = rng.choice([1, 2, 3, 4, 5, 6, 7, 9], size=n_hh, p=[.2, .25, .2, .15, .1, .05, .03, .02])
    hh = np.repeat(np.arange(1000, 1000 + n_hh), sizes)
    n = len(hh)
    df = pd.DataFrame({"household_id": hh, "ptype": rng.integers(1, 9, n).astype(np.int8),
                       "age": rng.integers(0, 90, n).astype(np.int8)},
                      index=pd.Index(rng.permutation(np.arange(1, n + 1)) * 3, name="person_id"))
    df["hhsize"] = df.groupby("household_id").household_id.transform("size")
    return df


def test_interaction_codes_numeric_and_text_agree():
    rng = np.random.default_rng(1)
    ptypes = [rng.integers(1, 9, 500) for _ in range(3)]
    for k in (2, 3):
        numeric = _interaction_code(ptypes[:k])
        text = pd.Series(ptypes[0]).astype(str)
        for p in ptypes[1:k]:
            text = text + pd.Series(p).astype(str)
        np.testing.assert_array_equal(numeric, text.apply(lambda x: "".join(sorted(x))).astype(int))


def test_cdap_rank_matches_1x():
    cdap1 = pytest.importorskip("activitysim.abm.models.util.cdap")
    persons = _persons()
    ref = cdap1.assign_cdap_rank(None, persons.copy(), PERSON_TYPE_MAP)  # 1.x: default_rng(0) order
    order = np.random.default_rng(seed=0).uniform(size=len(persons))
    ours = assign_cdap_rank(persons.index.to_numpy(), persons.household_id.to_numpy(), persons.ptype.to_numpy(),
                            persons.age.to_numpy(), PERSON_TYPE_MAP, order)
    np.testing.assert_array_equal(ours, ref.to_numpy())
    # every household's ranks are 1..n
    counts = pd.Series(ours).groupby(persons.household_id.to_numpy()).agg(["min", "max", "size"])
    assert (counts["min"] == 1).all() and (counts["max"] == counts["size"]).all()


def _interaction(example):
    configs = [REPO / "activitysim/examples" / example / "configs"]
    indiv = load_choice_model(configs, "cdap.yaml", handled=["annotate_persons", "annotate_households"],
                              spec_key="INDIV_AND_HHSIZE1_SPEC")
    return read_interaction_coefficients(configs[0] / "cdap_interaction_coefficients.csv",
                                         {k: c.value for k, c in indiv.coefficients.items()})


@pytest.mark.parametrize("example", ["prototype_mtc", "prototype_mwcog", "prototype_arc"])
def test_numpy_interaction_spec_is_the_same(example):
    coefficients = _interaction(example)
    for hhsize in range(1, 7):
        ref = build_cdap_spec(coefficients, hhsize)
        new = build_cdap_spec_numpy(coefficients, hhsize)
        assert list(new.index) == list(ref.index) and list(new.columns) == list(ref.columns)
        a, b = ref.astype(np.float64).values, new.astype(np.float64).values
        assert np.array_equal(a, b) and a.flags.f_contiguous == b.flags.f_contiguous


@pytest.mark.parametrize("example", ["prototype_mtc", "prototype_mwcog"])
def test_interaction_spec_matches_1x(example):
    cdap1 = pytest.importorskip("activitysim.abm.models.util.cdap")
    configs = [REPO / "activitysim/examples" / example / "configs"]
    indiv = load_choice_model(configs, "cdap.yaml", handled=["annotate_persons", "annotate_households"],
                              spec_key="INDIV_AND_HHSIZE1_SPEC")
    coefficients = read_interaction_coefficients(configs[0] / "cdap_interaction_coefficients.csv",
                                                 {k: c.value for k, c in indiv.coefficients.items()})
    ref_coefficients = coefficients.drop(columns=["cardinality", "slug"])
    ref_coefficients["interaction_ptypes"] = pd.read_csv(configs[0] / "cdap_interaction_coefficients.csv",
                                                         comment="#").interaction_ptypes
    ref_coefficients = cdap1.preprocess_interaction_coefficients(ref_coefficients)
    pd.testing.assert_frame_equal(coefficients, ref_coefficients)
    for hhsize in range(2, 6):
        ours = build_cdap_spec(coefficients, hhsize)
        ref = cdap1.build_cdap_spec(None, ref_coefficients, hhsize, cache=False)
        pd.testing.assert_frame_equal(ours.astype(np.float64), ref.astype(np.float64))
        assert list(ours.index) == list(ref.index)


def test_create_tours_matches_1x():
    tf1 = pytest.importorskip("activitysim.abm.models.util.tour_frequency")
    rng = np.random.default_rng(3)
    counts = pd.DataFrame({"work": rng.integers(0, 3, 50), "school": rng.integers(0, 3, 50)},
                          index=pd.Index(rng.permutation(np.arange(100, 150)), name="person_id"))
    counts = counts[(counts.work + counts.school) > 0]
    pd.testing.assert_frame_equal(create_tours(counts, "mandatory"), tf1.create_tours(counts, "mandatory"))


def test_tour_ids():
    tours = pd.DataFrame({"person_id": [7, 7, 9], "tour_type": ["work", "work", "school"],
                          "tour_type_num": [1, 2, 1]})
    channels = ["school1", "school2", "work1", "work2"]
    out = set_tour_index(tours, channels)
    assert list(out.index) == [7 * 4 + 2, 7 * 4 + 3, 9 * 4 + 0]
    with pytest.raises(ValueError, match="not among the possible tours"):
        set_tour_index(pd.DataFrame({"person_id": [1], "tour_type": ["work"], "tour_type_num": [3]}), channels)
