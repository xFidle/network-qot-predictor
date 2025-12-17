import argparse
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

    @staticmethod
    def parse_log_level(level_str: str) -> int:
        level_upper = level_str.upper()
        if level_upper not in level_mapping:
            raise ValueError(
                f"Invalid log level: {level_str}. Must be one of: {', '.join(level_mapping.keys())}"
            )

        return level_mapping[level_upper]

    def argparse_overrides(self, args: argparse.Namespace) -> "LoggerConfig":
        overrides = {}

        if args.log_level is not None:
            overrides["level"] = self.parse_log_level(args.log_level)

        if args.log_output is not None:
            overrides["output"] = args.log_output

        if args.log_file is not None:
            overrides["log_file"] = Path(args.log_file)

        if args.log_format is not None:
            overrides["format_string"] = args.log_format

        final_output = overrides.get("output", self.output)
        final_log_file = overrides.get("log_file", self.log_file)

        if "file" in final_output and final_log_file is None:
            raise ValueError("--log-file must be specified when 'file' is in --log-output")

        for key, value in overrides.items():
            setattr(self, key, value)

        return self


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
