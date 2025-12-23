import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.base import register_config
from src.config.parser import ConfigParser

logger = logging.getLogger(__name__)


def parse_percentage(val: int, _: ConfigParser) -> int:
    if not (0 <= val <= 100):
        raise ValueError(f"Invalid percentage: {val}. Must be between 0 and 100")
    return val


@register_config(name="data_processing", field_parsers={"unlabeled_percentage": parse_percentage})
@dataclass
class DataProcessingConfig:
    output_dir: Path = Path("data/active_learning")
    unlabeled_percentage: int = 75
    majority_ratio: int = 20


def shrink_minority_class(
    majority_ratio: int, majority_size: int, minority_data: pd.DataFrame
) -> pd.DataFrame:
    current_minority_size = len(minority_data)
    desired_minority_size = majority_size // majority_ratio

    samples_to_remove = current_minority_size - desired_minority_size

    logger.info(f"Current minority size: {current_minority_size}")
    logger.info(f"Desired minority size: {desired_minority_size}")
    logger.info(f"Samples to remove: {max(0, samples_to_remove)}")

    if samples_to_remove <= 0:
        logger.info("No removal needed - minority class already at or below desired size")
        return minority_data

    indices = np.random.choice(minority_data.index, size=desired_minority_size, replace=False)
    reduced_data = minority_data.loc[indices]

    logger.info(
        f"Reduced minority class from {current_minority_size} to {len(reduced_data)} samples"
    )

    return reduced_data


def merge_datasets(*datasets: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat(datasets, ignore_index=True).sample(frac=1)
    return merged


def save_to_file(output_file: Path, data: pd.DataFrame):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index=False)
    logger.info(f"saved {str(output_file)}")


def get_labeled_unlabeled(
    unlabeled_percentage: int, classes: tuple[pd.DataFrame, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove a percentage of samples from each class to create unlabeled set.

    Args:
        unlabeled_percentage: Percentage of samples to remove (e.g., 75 for 75%)
        classes: Tuple of 2 DataFrames, one for each class

    Returns:
        Tuple of (labeled_merged, unlabeled_combined)
    """
    labeled_dfs = []
    unlabeled_dfs = []

    for i, class_df in enumerate(classes):
        total_size = len(class_df)
        unlabeled_size = int(total_size * unlabeled_percentage / 100)
        labeled_size = total_size - unlabeled_size

        logger.info(
            f"Class {i}: Total={total_size}, Labeled={labeled_size}, Unlabeled={unlabeled_size}"
        )

        unlabeled_indices = np.random.choice(class_df.index, size=unlabeled_size, replace=False)

        unlabeled_df = class_df.loc[unlabeled_indices]
        labeled_df = class_df[~class_df.index.isin(unlabeled_indices)]

        labeled_dfs.append(labeled_df)
        unlabeled_dfs.append(unlabeled_df)

    unlabeled_combined = pd.concat(unlabeled_dfs, ignore_index=True)
    unlabeled_combined = unlabeled_combined.iloc[np.random.permutation(len(unlabeled_combined))]

    labeled_combined = pd.concat(labeled_dfs, ignore_index=True)
    labeled_combined = labeled_combined.iloc[np.random.permutation(len(labeled_combined))]

    logger.info(f"Created labeled set: {len(labeled_combined)}")
    logger.info(f"Created unlabeled set: {len(unlabeled_combined)} samples")

    return labeled_combined, unlabeled_combined
