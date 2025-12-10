from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from src.forest.cart import CART, CARTConfig
from src.forest.mode import PredictionMode
from src.forest.util import majority_vote


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
        with ProcessPoolExecutor() as executor:
            result = executor.map(
                self._build_single_tree, [X_train] * self.n_trees, [Y_train] * self.n_trees
            )
            executor.shutdown()

        for tree, indices in result:
            self.trees.append(tree)
            self.selected_features.append(indices)

    def predict(self, samples: np.ndarray, mode: PredictionMode) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError("Forest is not initalized, call fit() first.")

        all_predicitions = self._collect_tree_predictions(samples, mode)
        match mode:
            case PredictionMode.LABELS:
                return np.apply_along_axis(majority_vote, 1, all_predicitions)
            case PredictionMode.PROBABILITIES:
                return np.mean(all_predicitions, axis=1)

    def _collect_tree_predictions(self, samples: np.ndarray, mode: PredictionMode) -> np.ndarray:
        first = self.trees[0].predict(samples[:, self.selected_features[0]], mode)

        out_shape = (samples.shape[0], self.n_trees, *first.shape[1:])
        all_predictions = np.empty(out_shape)

        for i, tree in enumerate(self.trees):
            all_predictions[:, i] = tree.predict(samples[:, self.selected_features[i]], mode)

        return all_predictions

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
