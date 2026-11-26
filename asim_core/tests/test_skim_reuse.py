from collections import Counter

import numpy as np
import pytest

from asim_core.seams import Engine
from asim_core.skim_reuse import SharedSkims, skim_uses

SOURCE = '''def terms(c, s=None, k=None):
    # a comment naming s('odt_skims', 'IGNORED')
    yield s('odt_skims', 'SOV_TIME') + s('dot_skims', 'SOV_TIME')
    yield s('odt_skims', 'SOV_TIME') * 2
    yield s('skims', 'DIST', 'reverse') + s('skims', 'DIST')
'''


class _Lookups:
    def __init__(self):
        self.calls = Counter()

    def __call__(self, accessor, key, how="get"):
        self.calls[(accessor, key, how)] += 1
        return np.arange(5, dtype=np.float32) + hash((accessor, key, how)) % 7


def test_uses_are_counted_from_the_generated_source():
    assert skim_uses(SOURCE) == Counter({("odt_skims", "SOV_TIME", "get"): 2, ("dot_skims", "SOV_TIME", "get"): 1,
                                         ("skims", "DIST", "reverse"): 1, ("skims", "DIST", "get"): 1})


def test_skims_used_twice_are_looked_up_once_and_then_let_go():
    inner = _Lookups()
    s = Engine({"skim_reuse": "once"}).skim_reuse(inner, [SOURCE])
    assert isinstance(s, SharedSkims)
    first = s("odt_skims", "SOV_TIME")
    s("dot_skims", "SOV_TIME")
    second = s("odt_skims", "SOV_TIME")
    np.testing.assert_array_equal(first, second)
    assert inner.calls[("odt_skims", "SOV_TIME", "get")] == 1
    assert not s._values  # released after its last use
    with pytest.raises(ValueError):
        first[0] = 99.0  # shared values are read-only
    s("odt_skims", "SOV_TIME")  # a use beyond those foreseen: looked up as written
    assert inner.calls[("odt_skims", "SOV_TIME", "get")] == 2


def test_as_written_is_the_default_and_changes_nothing():
    inner = _Lookups()
    assert Engine().choices["skim_reuse"] == "as_written"
    assert Engine().skim_reuse(inner, [SOURCE]) is inner
    assert Engine({"skim_reuse": "once"}).skim_reuse(inner, ["def f(c, s):\n    yield s('a', 'B')\n"]) is inner
