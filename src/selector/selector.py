from typing import Protocol

import numpy as np

from src.model.classifier import Classifier


class Selector(Protocol):
    def __call__(self, X: np.ndarray, batch_size: int = 5) -> np.ndarray: ...


class UncertaintySelector:
    def __init__(self, classifier: Classifier) -> None:
        self.classifier: Classifier = classifier

    def __call__(self, X: np.ndarray, batch_size: int = 5) -> np.ndarray:
        proba = self.classifier.predict_proba(X)
        max_proba = np.max(proba, axis=1)

        return np.argsort(np.abs(max_proba - 0.5))[:batch_size]


class DiversitySelector:
    def __call__(self, X: np.ndarray, batch_size: int = 5) -> np.ndarray:
        distances = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
        mean_distance = np.mean(distances, axis=1)

        return np.argsort(mean_distance)[-batch_size:]


class RandomSelector:
    def __init__(self, random_state: int | None = None) -> None:
        self.random_state: int | None = random_state

    def __call__(self, X: np.ndarray, batch_size: int = 5) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)

        return rng.choice(X)
