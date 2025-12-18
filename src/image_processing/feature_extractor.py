import logging
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class ModelSetup(Protocol):
    def __call__(self) -> tuple[nn.Module, transforms.Compose]: ...


MODEL_SETUP: dict[str, ModelSetup] = {
    "resnet50": lambda: _setup_resnet50(),
    "vgg16": lambda: _setup_vgg16(),
}


def _setup_resnet50() -> tuple[nn.Module, transforms.Compose]:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    transform = weights.transforms()
    return model, transform


def _setup_vgg16() -> tuple[nn.Module, transforms.Compose]:
    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)
    model = model.features
    model = nn.Sequential(model, nn.AdaptiveAvgPool2d((1, 1)))
    transform = weights.transforms()
    return model, transform


class FeatureExtractor:
    def __init__(self, model_name: Literal["resnet50", "vgg16"] = "resnet50"):
        self.model_name: Literal["resnet50", "vgg16"] = model_name
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: nn.Module
        self.transform: transforms.Compose
        self.model, self.transform = self._load_model_transform()

    def _load_model_transform(self) -> tuple[nn.Module, transforms.Compose]:
        model_setup = MODEL_SETUP.get(self.model_name, None)
        if model_setup is None:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model, transform = model_setup()
        model = model.to(self.device)
        model.eval()
        return model, transform

    def extract_features(self, image_path: Path | str) -> npt.NDArray[np.float32]:
        image: Image.Image = Image.open(image_path).convert("RGB")
        image_tensor: torch.Tensor = self.transform(image).unsqueeze(0).to(self.device)  # type: ignore[attr-defined]

        with torch.no_grad():
            features: torch.Tensor = self.model(image_tensor)

        features_array: npt.NDArray[np.float32] = features.cpu().numpy().flatten()
        return features_array

    def process_directory(self, image_dir: Path | str, class_label: int) -> pd.DataFrame:
        image_dir = Path(image_dir)

        image_extensions: set[str] = {".jpg", ".jpeg", ".png"}

        image_files: list[Path] = [
            f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            error_msg = f"No images found in {image_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Processing {len(image_files)} images from {image_dir}")

        data: list[dict[str, int | float | str]] = []
        for image_file in image_files:
            features: npt.NDArray[np.float32] = self.extract_features(image_file)
            row: dict[str, int | float | str] = {
                **{f"feature_{j}": features[j] for j in range(len(features))},
                "class": class_label,
            }
            data.append(row)

        df: pd.DataFrame = pd.DataFrame(data)

        return df
