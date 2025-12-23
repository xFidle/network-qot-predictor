import logging
import shutil
from pathlib import Path

import kagglehub

from src.config import ConfigParser
from src.data_processing.data_processing import (
    DataProcessingConfig,
    merge_datasets,
    save_to_file,
    shrink_minority_class,
)
from src.image_processing import FeatureExtractor
from src.image_processing.config import ImageProcessingConfig
from src.utils.logger import LoggerConfig, setup_root_logger

logger = logging.getLogger(__name__)


def download_data(
    data_dir: Path | str,
    dataset: str,
    force_download: bool = False,
    subdirs_to_copy: list[str] | None = None,
):
    """
    Download dataset and optionally copy only specific subdirectories.

    Args:
        data_dir: Target directory for the data
        force_download: Force re-download even if data exists
        dataset: Kaggle dataset identifier
        subdirs_to_copy: List of subdirectory paths to copy (e.g., ["train/dandelion", "train/sunflower"])
                        If None, copies everything
    """
    data_dir = Path(data_dir)

    if data_dir.exists() and not force_download:
        logger.info("Data directory exists, skipping download")
        return

    logger.info(f"Downloading dataset: {dataset}")
    download_path = Path(kagglehub.dataset_download(dataset, force_download=force_download))

    data_dir.parent.mkdir(parents=True, exist_ok=True)

    if subdirs_to_copy is None:
        shutil.copytree(download_path, data_dir, dirs_exist_ok=True)
        logger.info(f"Copied all data to: {data_dir}")
    else:
        for subdir in subdirs_to_copy:
            src = download_path / subdir
            dst = data_dir / subdir

            if not src.exists():
                logger.warning(f"{subdir} not found in downloaded dataset")
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            logger.info(f"Copied {subdir} to {dst}")


def main():
    config_parser = ConfigParser("image_processing_config")
    image_processing_config = config_parser.get(ImageProcessingConfig)
    logger_config = config_parser.get(LoggerConfig)
    data_procesing_config = config_parser.get(DataProcessingConfig)

    setup_root_logger(logger_config)

    download_data(
        image_processing_config.data_dir,
        dataset="imsparsh/flowers-dataset",
        force_download=image_processing_config.force_download,
        subdirs_to_copy=["train/dandelion", "train/sunflower"],
    )

    data_dir = image_processing_config.data_dir / "train"
    output_dir = data_procesing_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = FeatureExtractor(image_processing_config.model)

    logger.info("Processing DANDELION images (class 0)")
    dandelion_dir = data_dir / "dandelion"

    df_dandelion = feature_extractor.process_directory(image_dir=dandelion_dir, class_label=0)
    logger.info(f"Dandelion dataset shape: {df_dandelion.shape}")

    logger.info("Processing SUNFLOWER images (class 1)")
    sunflower_dir = data_dir / "sunflower"

    df_sunflower = feature_extractor.process_directory(image_dir=sunflower_dir, class_label=1)
    logger.info(f"Sunflower dataset shape: {df_sunflower.shape}")

    df_sunflower = shrink_minority_class(
        data_procesing_config.majority_ratio, len(df_dandelion), df_sunflower
    )

    merged = merge_datasets(df_dandelion, df_sunflower)
    output_file = output_dir / "flowers.csv"
    save_to_file(output_file, merged)


if __name__ == "__main__":
    main()
