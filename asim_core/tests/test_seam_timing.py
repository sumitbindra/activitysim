import time

import numpy as np

from asim_core.seam_timing import SeamTimes, TimedEngine
from asim_core.seams import Engine


def test_pieces_are_timed_without_the_pieces_they_call():
    times = SeamTimes()
    e = TimedEngine(Engine(), times)
    m = np.arange(16.0).reshape(4, 4)

    def terms():
        time.sleep(0.02)  # formula work
        yield e.skim_lookup(m, np.array([0, 1]), np.array([2, 3])) + 1.0  # a skim lookup inside
        yield np.ones(2)

    utils = e.utilities(terms(), np.ones((2, 3)), 2)
    assert utils.shape == (2, 3)
    assert times.calls["utilities"] == 1 and times.calls["skim_lookup"] == 1
    assert times.calls["formulas"] == 3  # two values, then the end of the terms
    assert times.seconds["formulas"] >= 0.02 > times.seconds["utilities"]
    draws = e.random("step", 0).uniform("households", np.arange(5))
    assert len(draws) == 5 and times.calls["random"] == 2  # setting up the streams, then drawing
    assert e.changed == {} and e.agreement == "exact"
