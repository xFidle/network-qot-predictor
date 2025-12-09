import numpy as np
import pytest

from src.tree.split import split_at_threshold


@pytest.mark.parametrize(
    "dataset, feature_index, threshold, expected_above, expected_below",
    [
        (
            np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            0,
            4,
            np.array([[5, 6], [7, 8]]),
            np.array([[1, 2], [3, 4]]),
        ),
        (np.array([[1, 5], [2, 3], [3, 7]]), 1, 5, np.array([[1, 5], [3, 7]]), np.array([[2, 3]])),
        (np.array([[10, 20], [15, 25]]), 0, 5, np.array([[10, 20], [15, 25]]), np.empty((0, 2))),
        (np.array([[1, 2], [2, 3]]), 0, 5, np.empty((0, 2)), np.array([[1, 2], [2, 3]])),
    ],
)
def test_split(
    dataset: np.ndarray,
    feature_index: int,
    threshold: float,
    expected_above: np.ndarray,
    expected_below: np.ndarray,
):
    above, below = split_at_threshold(dataset, feature_index, threshold)

    assert (above == expected_above).all()
    assert (below == expected_below).all()
