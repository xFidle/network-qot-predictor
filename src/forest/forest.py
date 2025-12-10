from dataclasses import dataclass

import numpy as np

from src.forest.cart import CART, CARTConfig


@dataclass
class RandomForestConfig:
    n_trees: int
    tree_config: CARTConfig


class RandomForest:
    def __init__(self, config: RandomForestConfig) -> None:
        self.trees: list[CART] = []
        self.selected_features: list[np.ndarray] = []
        self.n_trees: int = config.n_trees
        self.tree_config: CARTConfig = config.tree_config

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
        for _ in range(self.n_trees):
            tree, indices = self._build_single_tree(X_train, Y_train)
            self.trees.append(tree)
            self.selected_features.append(indices)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError("Forest is not initalized, call fit() first.")

        all_predictions = np.empty((samples.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            all_predictions[:, i] = tree.predict(samples[:, self.selected_features[i]])

        return np.apply_along_axis(self._majority_vote, 1, all_predictions)

    def _majority_vote(self, votes: np.ndarray) -> int:
        unique, counts = np.unique(votes, return_counts=True)
        most_frequent = unique[counts == counts.max()]

        return np.random.choice(most_frequent)

    def _build_single_tree(
        self, X_train: np.ndarray, Y_train: np.ndarray
    ) -> tuple[CART, np.ndarray]:
        n_samples, n_features = X_train.shape

        rng = np.random.default_rng()

        samples_indices = rng.choice(n_samples, int(n_samples), replace=True)
        features_indices = rng.choice(n_features, int(np.sqrt(n_features)), replace=False)

        X_bootstrap = X_train[np.ix_(samples_indices, features_indices)]
        Y_bootstrap = Y_train[samples_indices]

        tree = CART(self.tree_config)
        tree.fit(X_bootstrap, Y_bootstrap)

        return tree, features_indices
