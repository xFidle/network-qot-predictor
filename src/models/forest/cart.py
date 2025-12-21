from dataclasses import dataclass

import numpy as np

from src.models.forest.util import gini_impurity

from .util import highest_probability_arg


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
        self.classes = np.unique(y_train)
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
        child = node.right if sample[node.feature_index] >= node.threshold else node.left

        return self._pl(child, sample)

    def _pp(self, node: DecisionNode | Leaf, sample: np.ndarray) -> np.ndarray:
        if isinstance(node, Leaf):
            return node.probabilities
        child = node.right if sample[node.feature_index] >= node.threshold else node.left

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

        feature, threshold, left_mask, right_mask = split

        left_subtree = self._build_tree(dataset[left_mask], n_labels, current_depth + 1)
        right_subtree = self._build_tree(dataset[right_mask], n_labels, current_depth + 1)

        return DecisionNode(feature, threshold, left_subtree, right_subtree)

    def _find_best_split(
        self, dataset: np.ndarray
    ) -> tuple[int, float, np.ndarray, np.ndarray] | None:
        n_samples, n_features = dataset.shape[0], dataset.shape[1] - 1
        labels = dataset[:, -1].astype(int)

        initial_right = np.bincount(labels)
        parent_impurity = gini_impurity(initial_right)

        best_gain = -np.inf
        best_split = None

        for feature_index in range(n_features):
            feature_column = dataset[:, feature_index]

            sort_indices = np.argsort(feature_column)
            features_sorted = feature_column[sort_indices]
            labels_sorted = labels[sort_indices]

            right_counts = np.copy(initial_right)
            left_counts = np.zeros_like(right_counts)

            for i in range(n_samples - 1):
                label = labels_sorted[i]
                right_counts[label] -= 1
                left_counts[label] += 1

                if features_sorted[i] == features_sorted[i + 1]:
                    continue

                gain = (
                    parent_impurity
                    - (left_counts.sum() / n_samples) * gini_impurity(left_counts)
                    - (right_counts.sum() / n_samples) * gini_impurity(right_counts)
                )

                if gain > best_gain:
                    best_gain = gain
                    thr = (features_sorted[i] + features_sorted[i + 1]) / 2
                    left_mask = feature_column < thr
                    right_mask = feature_column >= thr
                    best_split = (feature_index, thr, left_mask, right_mask)

        return best_split
