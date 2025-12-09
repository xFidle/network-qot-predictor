from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.tree.split import split_at_threshold


class SplitEvalFunction(Protocol):
    def __call__(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float: ...


@dataclass
class InnerNode:
    feature_index: int
    threshold: float
    left: "InnerNode | Leaf"
    right: "InnerNode | Leaf"


@dataclass
class Leaf:
    label: int


@dataclass
class CARTConfig:
    max_depth: int
    min_samples_split: int
    eval_function: SplitEvalFunction


class CART:
    def __init__(self, config: CARTConfig) -> None:
        self.root: InnerNode | Leaf | None = None
        self.max_depth: int = config.max_depth
        self.min_samples_split: int = config.min_samples_split
        self.eval_function: SplitEvalFunction = config.eval_function

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
        dataset = np.concatenate((X_train, Y_train), axis=1)
        self.root = self._build_tree(dataset)

    def predict(self, node: InnerNode | Leaf | None, sample: np.ndarray) -> int:
        if node is None:
            raise ValueError("The root is not initalized, call fit() first.")

        return self._predict_recursively(node, sample)

    def _predict_recursively(self, node: InnerNode | Leaf, sample: np.ndarray) -> int:
        if isinstance(node, Leaf):
            return node.label

        if sample[node.feature_index] >= node.threshold:
            return self._predict_recursively(node.left, sample)
        else:
            return self._predict_recursively(node.right, sample)

    def _build_tree(self, dataset: np.ndarray, current_depth: int = 0) -> InnerNode | Leaf:
        n_samples = dataset.shape[0]

        dataset_labels = dataset[:, -1]
        unique, counts = np.unique(dataset_labels, return_counts=True)
        if unique.size == 1 or n_samples < self.min_samples_split or current_depth > self.max_depth:
            return Leaf(unique[np.argmax(counts)])

        feature, threshold, left_split, right_split = self._find_best_split(dataset)

        left_subtree = self._build_tree(left_split, current_depth + 1)
        right_subtree = self._build_tree(right_split, current_depth + 1)

        return InnerNode(feature, threshold, left_subtree, right_subtree)

    def _find_best_split(self, dataset: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray]:
        n_features = dataset.shape[1] - 1
        best_gain = -np.inf
        best_split = None

        for feature_index in range(n_features):
            unique_values = np.unique(dataset[:, feature_index])

            for threshold in unique_values:
                left_split, right_split = split_at_threshold(dataset, feature_index, threshold)

                dataset_labels = dataset[:, -1]
                left_labels = left_split[:, -1]
                right_labels = right_split[:, -1]

                gain = self.eval_function(dataset_labels, left_labels, right_labels)

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, left_split, right_split)

        assert best_split is not None
        return best_split
