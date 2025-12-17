import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

LogOutput = Literal["file", "stdout"]

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

COLORS = {"WARNING": YELLOW, "INFO": GREEN, "DEBUG": BLUE, "CRITICAL": RED, "ERROR": RED}

level_mapping = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        levelname = record.levelname
        color = COLOR_SEQ % (30 + COLORS.get(levelname, 0))
        record.levelname = f"{color}{levelname}{RESET_SEQ}"
        return super().format(record)


@dataclass
class LoggerConfig:
    level: int = logging.INFO
    output: list[LogOutput] = field(default_factory=lambda: ["stdout"])
    log_file: Path | str | None = None
    format_string: str | None = "%(levelname)s - %(message)s"


def _configure_logger(logger: logging.Logger, config: LoggerConfig) -> logging.Logger:
    logger.setLevel(config.level)

    colored_formatter = ColoredFormatter(config.format_string)
    plain_formatter = logging.Formatter(config.format_string)

    if "file" in config.output:
        if config.log_file is None:
            raise ValueError("log_file must be specified when 'file' is in output")

        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(config.level)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    if "stdout" in config.output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

    return logger


def setup_logger(config: LoggerConfig) -> logging.Logger:
    global _global_logger_configured

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    _configure_logger(root_logger, config)

    return root_logger
