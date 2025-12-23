from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.config.base import register_config

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

COLORS = {"WARNING": YELLOW, "INFO": GREEN, "DEBUG": BLUE, "CRITICAL": RED, "ERROR": RED}


LogOutput = Literal["file", "stdout"]


@register_config(name="logging")
@dataclass
class LoggerConfig:
    level: str = logging.getLevelName(logging.INFO)
    output: list[LogOutput] = field(default_factory=lambda: ["stdout"])
    file: Path | None = None
    format_string: str = "%(levelname)s [%(asctime)s]: %(filename)s:%(funcName)s - %(message)s"


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        levelname = record.levelname
        color = COLOR_SEQ % (30 + COLORS.get(levelname, 0))
        record.levelname = f"{color}{levelname}{RESET_SEQ}"
        return super().format(record)


def configrue_root_logger(logger: logging.Logger, config: LoggerConfig) -> None:
    logger.setLevel(config.level)

    colored_formatter = ColoredFormatter(config.format_string)
    plain_formatter = logging.Formatter(config.format_string)

    def add_handler(handler: logging.Handler, formatter: logging.Formatter) -> None:
        handler.setLevel(config.level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if "file" in config.output:
        if config.file is None:
            raise ValueError("log_file must be specified when 'file' is in output")

        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        add_handler(logging.FileHandler(log_path), plain_formatter)

    if "stdout" in config.output:
        add_handler(logging.StreamHandler(sys.stdout), colored_formatter)


def setup_root_logger(config: LoggerConfig) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    configrue_root_logger(root_logger, config)
