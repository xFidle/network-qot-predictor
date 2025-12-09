import numpy as np


def split_at_threshold(
    dataset: np.ndarray, feature_index: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    feature_column = dataset[:, feature_index]
    return dataset[feature_column >= threshold], dataset[feature_column < threshold]
