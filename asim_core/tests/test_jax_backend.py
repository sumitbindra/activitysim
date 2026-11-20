"""The JAX alternative for formulas and utilities (an experiment; skipped without JAX)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from asim_core import jax_backend as jb  # noqa: E402
from asim_core.logit import FormulaValues, interaction_utilities, utilities  # noqa: E402
from asim_core.spec.translate import compile_function  # noqa: E402

SOURCE = '''def terms(c, n, tt=None, s=None):
    v = {}
    v['_D'] = _full(s('skims', 'DIST'), n)
    yield _clip(v['_D'], 0, 3)
    yield (c['size'] > 0) * np.log1p(c['size'])
    yield (c['tour_type'] == 'escort') * c['x']
    yield (c['tour_type'] != 'work') * 1.5
    yield _pandas_eval("x * (tour_type == 'shopping')", {'x': c['x'], 'tour_type': c['tour_type']})
    yield tt.previous_tour_ends(c['person_id'], c['start'])
'''


class _TT:
    def previous_tour_ends(self, person_ids, periods):
        return (np.asarray(person_ids) % 3 == 0) & (np.asarray(periods) > 5)


def _inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    c = {"size": rng.random(n) * (rng.random(n) > 0.3), "x": rng.normal(size=n).astype(np.float32),
         "tour_type": rng.choice(np.array(["escort", "work", "shopping", "othmaint"], dtype=object), n),
         "person_id": rng.integers(0, 100, n), "start": rng.integers(0, 20, n)}
    dist = rng.random(n).astype(np.float32) * 5
    return c, (lambda acc, key, how="get": dist)


@pytest.mark.parametrize("n", [10, 1500, 5000])
def test_interaction_utilities_close_to_numpy(n):
    fn = compile_function(SOURCE, "terms")
    c, s = _inputs(n)
    coefs = [0.5, -0.25, 1.0, 0.1, 2.0, -3.0]
    ref = interaction_utilities(FormulaValues(fn, c, n, _TT(), s), coefs, n)
    got = jb.interaction_utilities_jax(FormulaValues(fn, c, n, _TT(), s), coefs, n)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)
    comp = jb._compiled(fn)
    assert comp.ok and comp.calls["jax"] >= 1, comp.reason


def test_utilities_close_to_numpy():
    src = '''def terms(c, s=None, k=None):
    yield c['a'] * k['coef_x']
    yield np.where(c['b'] > 0.5, c['a'], 0.0)
    yield (s('odt_skims', 'TIME') + s('dot_skims', 'TIME'))
'''
    fn = compile_function(src, "terms")
    rng = np.random.default_rng(1)
    n = 3000
    c = {"a": rng.random(n), "b": rng.random(n)}
    t = rng.random(n).astype(np.float32)
    coefs = rng.normal(size=(3, 4))
    args = (c, (lambda acc, key, how="get": t), {"coef_x": 0.7})
    ref = utilities(FormulaValues(fn, *args), coefs, n)
    got = jb.utilities_jax(FormulaValues(fn, *args), coefs, n)
    assert got.shape == (n, 4)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_unsupported_formulas_fall_back_to_numpy():
    src = '''def terms(c, n, tt=None, s=None):
    yield _map(c['a'], {1: 2.0})
'''
    fn = compile_function(src, "terms")
    c = {"a": np.array([1, 2, 1])}
    got = jb.interaction_utilities_jax(FormulaValues(fn, c, 3, None, None), [1.0], 3)
    np.testing.assert_array_equal(np.isnan(got), [False, True, False])
    comp = jb._compiled(fn)
    assert not comp.ok and "_map" in comp.reason and comp.calls["numpy"] == 1


def test_rows_are_padded_to_a_few_sizes():
    assert jb.padded_rows(1) == jb.padded_rows(1024) == 1024
    sizes = {jb.padded_rows(n) for n in range(1025, 2049)}
    assert len(sizes) == jb.PAD_STEPS and max(sizes) == 2048
    assert all(jb.padded_rows(n) >= n for n in range(1, 100_000, 997))
