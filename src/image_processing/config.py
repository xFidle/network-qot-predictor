from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config.base import register_config


@register_config(name="image_processing")
@dataclass
class ImageProcessingConfig:
    model: Literal["resnet50", "vgg16"] = "resnet50"
    data_dir: Path = Path("data/flowers/images")
    force_download: bool = False
