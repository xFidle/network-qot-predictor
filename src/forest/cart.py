from dataclasses import dataclass

import numpy as np

from src.forest.gini import gini_gain
from src.forest.util import highest_probability_arg, split_at_threshold


@dataclass
class DecisionNode:
    feature_index: int
    threshold: float
    left: "DecisionNode | Leaf"
    right: "DecisionNode | Leaf"


@dataclass
class Leaf:
    probabilities: np.ndarray


@dataclass
class CARTConfig:
    max_depth: int
    min_samples_split: int


class CART:
    def __init__(self, config: CARTConfig) -> None:
        self.root: DecisionNode | Leaf | None = None
        self.max_depth = config.max_depth
        self.min_samples_split = config.min_samples_split

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.root is not None:
            self.root = None

        dataset = np.concatenate((X_train, y_train[:, np.newaxis]), axis=1)
        self.root = self._build_tree(dataset, np.unique(y_train).size)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("The root is not initialized, call fit() first.")

        return np.array([self._pl(self.root, sample) for sample in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("The root is not initialized, call fit() first.")

        return np.array([self._pp(self.root, sample) for sample in X])

    def _pl(self, node: DecisionNode | Leaf, sample: np.ndarray) -> int:
        if isinstance(node, Leaf):
            return highest_probability_arg(node.probabilities)
        child = node.left if sample[node.feature_index] >= node.threshold else node.right

        return self._pl(child, sample)

    def _pp(self, node: DecisionNode | Leaf, sample: np.ndarray) -> np.ndarray:
        if isinstance(node, Leaf):
            return node.probabilities
        child = node.left if sample[node.feature_index] >= node.threshold else node.right

        return self._pp(child, sample)

    def _build_tree(
        self, dataset: np.ndarray, n_labels: int, current_depth: int = 0
    ) -> DecisionNode | Leaf:
        n_samples = dataset.shape[0]
        dataset_labels = dataset[:, -1]

        split = self._find_best_split(dataset)
        if split is None or n_samples < self.min_samples_split or current_depth >= self.max_depth:
            proba = np.zeros(n_labels)
            unique, counts = np.unique(dataset_labels, return_counts=True)
            proba[unique.astype(int)] = counts / np.sum(counts)
            return Leaf(proba)

        feature, threshold, left_split, right_split = split

        left_subtree = self._build_tree(left_split, n_labels, current_depth + 1)
        right_subtree = self._build_tree(right_split, n_labels, current_depth + 1)

        return DecisionNode(feature, threshold, left_subtree, right_subtree)

    def _find_best_split(
        self, dataset: np.ndarray
    ) -> tuple[int, float, np.ndarray, np.ndarray] | None:
        n_features = dataset.shape[1] - 1
        best_gain = -np.inf
        best_split = None

        for feature_index in range(n_features):
            unique_values = np.unique(dataset[:, feature_index])

            for threshold in unique_values:
                left_split, right_split = split_at_threshold(dataset, feature_index, threshold)

                if left_split.size == 0 or right_split.size == 0:
                    continue

                dataset_labels = dataset[:, -1]
                left_labels = left_split[:, -1]
                right_labels = right_split[:, -1]

                gain = gini_gain(dataset_labels, left_labels, right_labels)

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, left_split, right_split)

        return best_split
