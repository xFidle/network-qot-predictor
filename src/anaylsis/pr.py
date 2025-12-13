from dataclasses import dataclass

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from src.model.classifier import Classifier


@dataclass
class PRMetrics:
    precision: np.ndarray
    recall: np.ndarray
    auc_score: float


def calculate_pr_metrics(
    classifier: Classifier, X_test: np.ndarray, y_test: np.ndarray
) -> PRMetrics:
    proba = classifier.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    auc_score = auc(recall, precision)

    return PRMetrics(precision, recall, float(auc_score))
