import argparse
import logging
from pathlib import Path

from src.utils.logger import LoggerConfig, LogOutput


def add_logger_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    logger_group = parser.add_argument_group("logging options")

    logger_group.add_argument(
        "--log-output",
        nargs="+",
        choices=["stdout", "file"],
        default=["stdout"],
        help="Logger output destinations (default: stdout)",
    )

    logger_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (required if 'file' is in --log-output)",
    )

    logger_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    logger_group.add_argument(
        "--log-format",
        type=str,
        default="%(levelname)s - %(message)s",
        help="Custom log format string (default: '%%(levelname)s - %%(message)s')",
    )

    return parser


def get_logger_config_from_args(args: argparse.Namespace) -> LoggerConfig:
    level = getattr(logging, args.log_level)

    output: list[LogOutput] = args.log_output

    log_file = Path(args.log_file) if args.log_file else None

    if "file" in output and log_file is None:
        raise ValueError("--log-file must be specified when 'file' is in --log-output")

    format_string = args.log_format

    return LoggerConfig(level=level, output=output, log_file=log_file, format_string=format_string)
