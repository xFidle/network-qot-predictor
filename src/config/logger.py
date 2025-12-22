import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.config.base import register_config

LogOutput = Literal["file", "stdout"]


@register_config(
    name="logging",
    field_mappings={"output": "log_output", "level": "log_level", "format_string": "log_format"},
)
@dataclass
class LoggerConfig:
    level: str = logging.getLevelName(logging.INFO)
    output: list[LogOutput] = field(default_factory=lambda: ["stdout"])
    log_file: Path | None = None
    format_string: str = "%(levelname)s - %(message)s"
