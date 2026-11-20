import numpy as np
import pytest

from asim_core.components.segments import SegmentError, check_segment_values


def test_every_chooser_in_a_segment_passes():
    check_segment_values("step", "tours", "tour_type", np.array(["work", "shop", "work"]), ["work", "shop"])


def test_choosers_without_a_segment_are_named_and_counted():
    with pytest.raises(SegmentError) as err:
        check_segment_values("location", "persons", "income_segment", np.array([1, 2, 5, 5, 7]), [1, 2, 3, 4])
    msg = str(err.value)
    assert "3 persons" in msg and "5: 2" in msg and "7: 1" in msg and "[1, 2, 3, 4]" in msg
