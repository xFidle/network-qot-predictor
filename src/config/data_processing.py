from dataclasses import dataclass
from pathlib import Path

from src.config.base import register_config
from src.config.parser import ConfigParser


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
