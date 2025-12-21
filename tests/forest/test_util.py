import numpy as np
import pytest

from src.models.forest.util import gini_impurity, highest_probability_arg, majority_vote


@pytest.mark.parametrize(
    "proba, expected",
    [
        (np.array([1]), (0,)),
        (np.array([0.5, 0.5]), (0, 1)),
        (np.array([0.9, 0.1]), (0,)),
        (np.array([0.1, 0.9]), (1,)),
        (np.array([0.33, 0.33, 0.33]), (0, 1, 2)),
    ],
)
def test_highest_probability(proba: np.ndarray, expected: tuple[int]):
    arg = highest_probability_arg(proba)
    assert arg in expected


@pytest.mark.parametrize(
    "votes, expected",
    [
        (np.array([1]), (1,)),
        (np.array([0, 1]), (0, 1)),
        (np.array([1, 1]), (1,)),
        (np.array([0, 1, 1]), (1,)),
        (np.array([1, 1, 1]), (1,)),
        (np.array([0, 1, 0, 1]), (0, 1)),
        (np.array([0, 0, 1, 0]), (0,)),
    ],
)
def test_majority_vote(votes: np.ndarray, expected: tuple[int]):
    label = majority_vote(votes)
    assert label in expected


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.array([]), 0),
        (np.array([1]), 0),
        (np.array([1, 1, 1, 1]), 0.75),
        (np.array([4]), 0),
        (np.array([2, 2]), 0.5),
    ],
)
def test_gini(array: np.ndarray, expected: float):
    assert gini_impurity(array) == expected
