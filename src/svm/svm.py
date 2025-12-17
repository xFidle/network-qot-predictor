from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class SVMConfig:
    learning_rate: float
    penalty: float  # C param
    iter_count: int


class SVM:
    def __init__(self, config: SVMConfig) -> None:
        self.learning_rate: float = config.learning_rate
        self.penalty: float = config.penalty
        self.iter_count: int = config.iter_count
        self.w: np.ndarray | None = None
        self.b: float | None = None
        self.platt_a: float | None = None
        self.platt_b: float | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        _, n_features = X_train.shape

        w = np.zeros(n_features)
        b = 0.0

        y = np.where(y_train <= 0, -1, 1)

        for _ in range(self.iter_count):
            for idx, x_i in enumerate(X_train):
                condition = y[idx] * (np.dot(x_i, w) + b) >= 1

                if condition:
                    w -= self.learning_rate * w
                else:
                    w -= self.learning_rate * (w - self.penalty * y[idx] * x_i)
                    b -= self.learning_rate * (-self.penalty * y[idx])

        self.w = w
        self.b = b

        self._fit_platt_scaling(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.sign(self._decision_function(X))
        return np.where(pred <= 0, 0, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.platt_a is None or self.platt_b is None:
            raise ValueError("Model not trained. Call fit() first.")

        decision = self._decision_function(X)
        proba_class_1 = 1 / (1 + np.exp(self.platt_a * decision + self.platt_b))
        proba_class_0 = 1 - proba_class_1
        return np.column_stack([proba_class_0, proba_class_1])

    def _fit_platt_scaling(self, X: np.ndarray, Y: np.ndarray) -> None:
        decision_values = self._decision_function(X)

        y_binary = np.where(Y <= 0, 0, 1).flatten()

        n_pos = np.sum(y_binary == 1)
        n_neg = np.sum(y_binary == 0)
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1 / (n_neg + 2)

        targets = np.where(y_binary == 1, t_pos, t_neg)

        def neg_log_likelihood(params):
            a, b = params
            pred_proba = 1 / (1 + np.exp(a * decision_values + b))
            pred_proba = np.clip(pred_proba, 1e-15, 1 - 1e-15)
            nll = -np.sum(targets * np.log(pred_proba) + (1 - targets) * np.log(1 - pred_proba))
            return nll

        result = minimize(neg_log_likelihood, x0=[0.0, 0.0], method="BFGS")
        self.platt_a, self.platt_b = result.x

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None or self.b is None:
            raise ValueError("Model not trained. Call fit() first.")

        return np.dot(X, self.w) + self.b
