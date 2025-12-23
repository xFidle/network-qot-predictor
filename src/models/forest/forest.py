from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from src.config import register_config

from .cart import CART, CARTConfig
from .util import majority_vote


@register_config(name="forest")
@dataclass
class RandomForestConfig:
    tree_config: CARTConfig = field(default_factory=CARTConfig)
    n_trees: int = 100
    multiprocessing: bool = False
    seed: int = 42


class RandomForest:
    def __init__(self, config: RandomForestConfig) -> None:
        self._trees: list[CART] = []
        self._selected_features: list[np.ndarray] = []
        self._n_trees = config.n_trees
        self._tree_config = config.tree_config
        self._multiprocessing = config.multiprocessing
        self._forest_rng = np.random.default_rng(config.seed)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if len(self._trees) == 0:
            self._trees = []
            self._selected_features = []

        self.classes = np.unique(y_train)

        seeds = self._forest_rng.integers(0, 2**32 - 1, size=self._n_trees)
        if self._multiprocessing:
            with ProcessPoolExecutor() as executor:
                result = executor.map(
                    self._build_single_tree,
                    [X_train] * self._n_trees,
                    [y_train] * self._n_trees,
                    seeds,
                )

            for tree, indices in result:
                self._trees.append(tree)
                self._selected_features.append(indices)

        else:
            for i in range(self._n_trees):
                tree, indices = self._build_single_tree(X_train, y_train, seeds[i])
                self._trees.append(tree)
                self._selected_features.append(indices)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(self._trees) == 0:
            raise ValueError("Forest is not initalized, call fit() first.")

        all_predictions = self._collect_tree_labels(X)

        return np.apply_along_axis(majority_vote, 1, all_predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(self._trees) == 0:
            raise ValueError("Forest is not initalized, call fit() first.")

        all_predictions = self._collect_tree_proba(X)

        return np.mean(all_predictions, axis=1)

    def _collect_tree_labels(self, X: np.ndarray) -> np.ndarray:
        return np.stack(
            [tree.predict(X[:, self._selected_features[i]]) for i, tree in enumerate(self._trees)],
            axis=1,
        )

    def _collect_tree_proba(self, X: np.ndarray) -> np.ndarray:
        all_proba: list[np.ndarray] = []

        for i, tree in enumerate(self._trees):
            proba = tree.predict_proba(X[:, self._selected_features[i]])
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

        tree = CART(self._tree_config)
        tree.fit(X_bootstrap, y_bootstrap)

        return tree, features_indices
