from pathlib import Path
import shutil

from src.image_processing import FeatureExtractor

import kagglehub


def download_data(
    data_dir: Path | str, force_download: bool = False, dataset: str = "imsparsh/flowers-dataset"
):
    data_dir = Path(data_dir)

    if data_dir.exists() and not force_download:
        print("Data directory exists, skipping download")
        return

    print(f"Downloading dataset: {dataset}")
    download_path = kagglehub.dataset_download(dataset, force_download=force_download)

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(download_path, data_dir, dirs_exist_ok=True)


def main():
    download_dir = Path("data/flowers/images")
    download_data(download_dir)

    data_dir = download_dir / "train"
    output_dir = Path("data/flowers/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = "resnet50"
    feature_extractor = FeatureExtractor(model_name)

    print("Processing DANDELION images (class 0)")
    dandelion_dir = data_dir / "dandelion"
    dandelion_csv = output_dir / f"dandelion_features_{model_name}.csv"

    df_dandelion = feature_extractor.process_directory(
        image_dir=dandelion_dir, class_label=0, output_csv=dandelion_csv
    )
    print(f"Dandelion dataset shape: {df_dandelion.shape}")

    print("Processing SUNFLOWER images (class 1)")
    sunflower_dir = data_dir / "sunflower"
    sunflower_csv = output_dir / f"sunflower_features_{model_name}.csv"

    df_sunflower = feature_extractor.process_directory(
        image_dir=sunflower_dir, class_label=1, output_csv=sunflower_csv
    )
    print(f"Sunflower dataset shape: {df_sunflower.shape}")

    print(f"Dandelion samples: {len(df_dandelion)}")
    print(f"Sunflower samples: {len(df_sunflower)}")
    print(f"Total samples: {len(df_dandelion) + len(df_sunflower)}")
    print("Output files:")
    print(f"  - {dandelion_csv}")
    print(f"  - {sunflower_csv}")


if __name__ == "__main__":
    main()
