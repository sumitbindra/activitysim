import numpy as np

from asim_core.components.segments import segment_problem


def test_every_chooser_in_a_segment_passes():
    assert segment_problem("tours", "tour_type", np.array(["work", "shop", "work"]), ["work", "shop"]) is None


def test_choosers_without_a_segment_are_named_and_counted():
    msg = segment_problem("persons", "income_segment", np.array([1, 2, 5, 5, 7]), [1, 2, 3, 4])
    assert "3 persons" in msg and "5: 2" in msg and "7: 1" in msg and "[1, 2, 3, 4]" in msg
