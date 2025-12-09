import numpy as np


def labels_probabilities(Y: np.ndarray) -> np.ndarray:
    n_labels = Y.size
    _, counts = np.unique(Y, return_counts=True)
    return counts / n_labels


def gini_impurity(Y: np.ndarray) -> float:
    probabilities = labels_probabilities(Y)
    if probabilities.size == 0:
        return 0.0

    impurity = float(1 - np.sum(np.power(probabilities, 2)))
    return impurity


def gini_gain(parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    total = parent.size
    if total == 0:
        return 0.0

    return (
        gini_impurity(parent)
        - (left.size / total) * gini_impurity(left)
        - (right.size / total) * gini_impurity(right)
    )
