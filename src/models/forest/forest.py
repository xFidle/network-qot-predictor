from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .cart import CART, CARTConfig
from .util import majority_vote


@dataclass
class RandomForestConfig:
    n_trees: int
    tree_config: CARTConfig
    seed: int = 42


class RandomForest:
    def __init__(self, config: RandomForestConfig) -> None:
        self.trees: list[CART] = []
        self.selected_features: list[np.ndarray] = []
        self.n_trees = config.n_trees
        self.tree_config = config.tree_config
        self.forest_rng = np.random.default_rng(config.seed)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if len(self.trees) == 0:
            self.trees = []
            self.selected_features = []

        self.classes = np.unique(y_train)

        with ProcessPoolExecutor() as executor:
            seeds = self.forest_rng.integers(0, 2**32 - 1, size=self.n_trees)
            result = executor.map(
                self._build_single_tree, [X_train] * self.n_trees, [y_train] * self.n_trees, seeds
            )

        for tree, indices in result:
            self.trees.append(tree)
            self.selected_features.append(indices)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError("Forest is not initalized, call fit() first.")

        all_predictions = self._collect_tree_labels(X)

        return np.apply_along_axis(majority_vote, 1, all_predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError("Forest is not initalized, call fit() first.")

        all_predictions = self._collect_tree_proba(X)

        return np.mean(all_predictions, axis=1)

    def _collect_tree_labels(self, X: np.ndarray) -> np.ndarray:
        return np.stack(
            [tree.predict(X[:, self.selected_features[i]]) for i, tree in enumerate(self.trees)],
            axis=1,
        )

    def _collect_tree_proba(self, X: np.ndarray) -> np.ndarray:
        all_proba: list[np.ndarray] = []

        for i, tree in enumerate(self.trees):
            proba = tree.predict_proba(X[:, self.selected_features[i]])
            aligned = np.zeros((proba.shape[0], self.classes.shape[0]))

            indices = np.searchsorted(self.classes, tree.classes)
            aligned[:, indices] = proba

            all_proba.append(aligned)

        return np.stack(all_proba, axis=1)

    def _build_single_tree(
        self, X_train: np.ndarray, y_train: np.ndarray, seed: int
    ) -> tuple[CART, np.ndarray]:
        n_samples, n_features = X_train.shape

        rng = np.random.default_rng(seed)

        samples_indices = rng.choice(n_samples, int(n_samples), replace=True)
        features_indices = rng.choice(n_features, int(np.sqrt(n_features)), replace=False)

        X_bootstrap = X_train[np.ix_(samples_indices, features_indices)]
        y_bootstrap = y_train[samples_indices]

        tree = CART(self.tree_config)
        tree.fit(X_bootstrap, y_bootstrap)

        return tree, features_indices
