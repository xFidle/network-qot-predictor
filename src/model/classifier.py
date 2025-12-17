from typing import Literal, Protocol

import numpy as np

from src.forest.forest import CARTConfig, RandomForest, RandomForestConfig
from src.svm.svm import SVM, SVMConfig


class Classifier(Protocol):
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def resolve_model(name: Literal["svm", "forest"]) -> Classifier:
    match name:
        case "svm":
            svm_config = SVMConfig(learning_rate=0.1, penalty=100, iter_count=1000)
            return SVM(svm_config)

        case "forest":
            cart_config = CARTConfig(max_depth=10, min_samples_split=2)
            forest_config = RandomForestConfig(n_trees=100, tree_config=cart_config)
            return RandomForest(forest_config)
