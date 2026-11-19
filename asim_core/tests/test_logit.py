import numpy as np

from asim_core import logit


def test_probabilities_sum_to_one():
    utils = np.array([[0.0, 1.0, -2.0], [5.0, 5.0, 5.0]])
    probs, bad = logit.probabilities(utils)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert not bad.any()
    np.testing.assert_allclose(probs[1], [1 / 3] * 3)


def test_unusable_rows_are_flagged():
    utils = np.array([[-1000.0, -1000.0], [0.0, np.nan]])
    _, bad = logit.probabilities(utils)
    assert bad.tolist() == [True, True]


def test_shift_protects_against_overflow():
    utils = np.array([[800.0, 801.0]])
    _, bad = logit.probabilities(utils, shift=False)
    assert bad.all()
    probs, bad = logit.probabilities(utils, shift=True)
    assert not bad.any()
    np.testing.assert_allclose(probs.sum(), 1.0)


def test_choose_takes_first_alternative_that_uses_up_the_draw():
    probs = np.array([[0.2, 0.3, 0.5]] * 4)
    rands = np.array([0.1, 0.2, 0.25, 0.99])
    assert logit.choose(probs, rands).tolist() == [0, 0, 1, 2]


def test_choose_marks_bad_probabilities():
    probs = np.array([[0.2, 0.2], [0.5, 0.5]])
    assert logit.choose(probs, np.array([0.1, 0.1])).tolist() == [logit.BAD_CHOICE, 0]


def test_utilities_are_values_times_coefficients():
    terms = [np.array([1.0, 0.0]), np.array([2.0, 3.0])]  # 2 terms, 2 choosers
    coef = np.array([[0.0, 1.0], [0.5, -1.0]])  # 2 terms x 2 alternatives
    np.testing.assert_allclose(logit.utilities(iter(terms), coef, 2), [[1.0, -1.0], [1.5, -3.0]])


def test_wrong_number_of_terms_is_an_error():
    import pytest

    with pytest.raises(ValueError, match="expected 2 term values"):
        logit.utilities(iter([np.ones(2)]), np.ones((2, 2)), 2)
