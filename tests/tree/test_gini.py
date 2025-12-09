import numpy as np
import pytest

from src.tree.gini import gini_gain, gini_impurity, labels_probabilities


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.array([]), np.array([])),
        (np.array([1]), np.array([1])),
        (np.array([1, 2, 3, 4]), np.array([0.25, 0.25, 0.25, 0.25])),
        (np.array([1, 1, 1, 1]), np.array([1])),
        (np.array([1, 1, 2, 2]), np.array([0.5, 0.5])),
    ],
)
def test_labels_probabilities(array: np.ndarray, expected: np.ndarray):
    assert (labels_probabilities(array) == expected).all()


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.array([]), 0),
        (np.array([1]), 0),
        (np.array([1, 2, 3, 4]), 0.75),
        (np.array([1, 1, 1, 1]), 0),
        (np.array([1, 1, 2, 2]), 0.5),
    ],
)
def test_gini(array: np.ndarray, expected: float):
    assert gini_impurity(array) == expected


@pytest.mark.parametrize(
    "parent, left, right, expected",
    [
        (np.array([]), np.array([]), np.array([]), 0.0),
        (np.array([1]), np.array([1]), np.array([]), 0.0),
        (np.array([1, 1, 2, 2]), np.array([1, 1]), np.array([2, 2]), 0.5),
        (np.array([1, 1, 1, 2]), np.array([1, 1, 1]), np.array([2]), 0.375),
    ],
)
def test_gini_gain(parent: np.ndarray, left: np.ndarray, right: np.ndarray, expected: float):
    gain = gini_gain(parent, left, right)
    assert np.isclose(gain, expected, atol=1e-3)
