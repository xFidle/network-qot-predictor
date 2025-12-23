from typing import Literal, Protocol

import numpy as np

from src.config import ConfigParser

from .forest.forest import RandomForest, RandomForestConfig
from .svm.svm import SVM, SVMConfig

type ClassifierName = Literal["svm", "forest"]


class Classifier(Protocol):
    name: ClassifierName

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def resolve_classifier(name: ClassifierName, p: ConfigParser) -> Classifier:
    match name:
        case "svm":
            return SVM(p.get(SVMConfig))

        case "forest":
            return RandomForest(p.get(RandomForestConfig))
