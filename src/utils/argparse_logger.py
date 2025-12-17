import argparse


def add_logger_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    logger_group = parser.add_argument_group("logging options")

    logger_group.add_argument(
        "--log-output",
        nargs="+",
        choices=["stdout", "file"],
        default=None,
        help="Logger output destinations",
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
        default=None,
        help="Logging level",
    )

    logger_group.add_argument(
        "--log-format", type=str, default=None, help="Custom log format string"
    )

    return parser
