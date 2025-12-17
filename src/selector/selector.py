from typing import Literal, Protocol

import numpy as np

from src.model.classifier import Classifier


class Selector(Protocol):
    def __call__(
        self, X_unlabeled: np.ndarray, X_train: np.ndarray, batch_size: int = 5
    ) -> np.ndarray: ...


class UncertaintySelector:
    def __init__(self, classifier: Classifier) -> None:
        self.classifier: Classifier = classifier

    def __call__(
        self, X_unlabeled: np.ndarray, X_train: np.ndarray, batch_size: int = 5
    ) -> np.ndarray:
        proba = self.classifier.predict_proba(X_unlabeled)
        max_proba = np.max(proba, axis=1)
        size = min(batch_size, X_unlabeled.shape[0])

        return np.argsort(1 - max_proba)[-size:]


class DiversitySelector:
    def __call__(
        self, X_unlabeled: np.ndarray, X_train: np.ndarray, batch_size: int = 5
    ) -> np.ndarray:
        distances = np.linalg.norm(
            X_unlabeled[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
        )
        min_distance = np.min(distances, axis=1)
        size = min(batch_size, X_unlabeled.shape[0])

        return np.argsort(min_distance)[-size:]


class RandomSelector:
    def __init__(self, random_state: int | None = None) -> None:
        self.random_state: int | None = random_state

    def __call__(
        self, X_unlabeled: np.ndarray, X_train: np.ndarray, batch_size: int = 5
    ) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        size = min(batch_size, X_unlabeled.shape[0])

        return rng.choice(X_unlabeled.shape[0], size, replace=False)


def resolve_selector(
    name: Literal["uncertainty", "diversity", "random"], classifier: Classifier
) -> Selector:
    match name:
        case "uncertainty":
            return UncertaintySelector(classifier)
        case "diversity":
            return DiversitySelector()
        case "random":
            return RandomSelector()
