from typing import Literal, Protocol

import numpy as np

from src.models.classifier import Classifier


type SelectorName = Literal["uncertainty", "diversity", "random"]


class Selector(Protocol):
    name: SelectorName

    def __call__(
        self, X_pool: np.ndarray, labeled_mask: np.ndarray, batch_size: int = 5
    ) -> np.ndarray: ...


class UncertaintySelector:
    name: SelectorName = "uncertainty"

    def __init__(self, classifier: Classifier) -> None:
        self.classifier: Classifier = classifier

    def __call__(
        self, X_pool: np.ndarray, labeled_mask: np.ndarray, batch_size: int = 5
    ) -> np.ndarray:
        unlabeled_indices = np.flatnonzero(~labeled_mask)
        X_unlabeled = X_pool[unlabeled_indices, :]

        proba = self.classifier.predict_proba(X_unlabeled)
        max_proba = np.max(proba, axis=1)
        size = min(batch_size, X_unlabeled.shape[0])
        relative_indices = np.argsort(1 - max_proba)[-size:]

        return unlabeled_indices[relative_indices]


class DiversitySelector:
    name: SelectorName = "diversity"

    def __call__(
        self, X_pool: np.ndarray, labeled_mask: np.ndarray, batch_size: int = 5
    ) -> np.ndarray:
        X_train = X_pool[labeled_mask, :]

        unlabeled_indices = np.flatnonzero(~labeled_mask)
        X_unlabeled = X_pool[unlabeled_indices, :]

        distances = np.linalg.norm(
            X_unlabeled[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
        )
        min_distance = np.min(distances, axis=1)
        size = min(batch_size, X_unlabeled.shape[0])
        relative_indices = np.argsort(min_distance)[-size:]

        return unlabeled_indices[relative_indices]


class RandomSelector:
    name: SelectorName = "random"

    def __init__(self, random_state: int | None = None) -> None:
        self.rng = np.random.default_rng(random_state)

    def __call__(
        self, X_pool: np.ndarray, labeled_mask: np.ndarray, batch_size: int = 5
    ) -> np.ndarray:
        unlabeled_indices = np.flatnonzero(~labeled_mask)
        X_unlabeled = X_pool[unlabeled_indices, :]

        size = min(batch_size, X_unlabeled.shape[0])
        relative_indices = self.rng.choice(X_unlabeled.shape[0], size, replace=False)

        return unlabeled_indices[relative_indices]


def resolve_selector(name: SelectorName, classifier: Classifier) -> Selector:
    match name:
        case "uncertainty":
            return UncertaintySelector(classifier)
        case "diversity":
            return DiversitySelector()
        case "random":
            return RandomSelector()
