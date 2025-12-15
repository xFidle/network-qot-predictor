import argparse
import logging
import shutil
from pathlib import Path

import kagglehub

from src.image_processing import FeatureExtractor
from src.utils.argparse_logger import add_logger_arguments, get_logger_config_from_args
from src.utils.logger import setup_logger

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process flower images and extract features")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/flowers/images",
        help="Directory to download/store image data",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/flowers/processed",
        help="Directory to save processed features",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet50", "vgg16"],
        default="resnet50",
        help="Model to use for feature extraction",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of dataset even if it exists",
    )

    add_logger_arguments(parser)

    return parser.parse_args()


def main():
    args = parse_args()

    logger_config = get_logger_config_from_args(args)
    logger = setup_logger(logger_config)

    download_dir = Path(args.data_dir)
    download_data(
        download_dir,
        dataset="imsparsh/flowers-dataset",
        force_download=args.force_download,
        subdirs_to_copy=["train/dandelion", "train/sunflower"],
    )

    data_dir = download_dir / "train"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    feature_extractor = FeatureExtractor(model_name)

    logger.info("Processing DANDELION images (class 0)")
    dandelion_dir = data_dir / "dandelion"
    dandelion_csv = output_dir / f"dandelion_features_{model_name}.csv"

    df_dandelion = feature_extractor.process_directory(
        image_dir=dandelion_dir, class_label=0, output_csv=dandelion_csv
    )
    logger.info(f"Dandelion dataset shape: {df_dandelion.shape}")

    logger.info("Processing SUNFLOWER images (class 1)")
    sunflower_dir = data_dir / "sunflower"
    sunflower_csv = output_dir / f"sunflower_features_{model_name}.csv"

    df_sunflower = feature_extractor.process_directory(
        image_dir=sunflower_dir, class_label=1, output_csv=sunflower_csv
    )
    logger.info(f"Sunflower dataset shape: {df_sunflower.shape}")

    logger.info(f"Dandelion samples: {len(df_dandelion)}")
    logger.info(f"Sunflower samples: {len(df_sunflower)}")
    logger.info(f"Total samples: {len(df_dandelion) + len(df_sunflower)}")
    logger.info(f"Output files:\n  - {dandelion_csv}\n  - {sunflower_csv}")


if __name__ == "__main__":
    main()
