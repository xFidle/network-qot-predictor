from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ConfigParser
from src.config.logger import LoggerConfig
from src.learner.learner import ActiveLearner, ActiveLearnerConfig, LearningData
from src.model.classifier import resolve_model
from src.selector.selector import resolve_selector
from src.utils.logger import setup_logger


def parse_config():
    config_parser = ConfigParser()
    logger_config = config_parser.get(LoggerConfig)
    setup_logger(logger_config)


def main():
    args = parse_args()

    logger_config = get_logger_config_from_args(args)
    _ = setup_logger(logger_config)

    df = pd.read_csv("./data/wine-quality.csv")
    x, y = df.iloc[:, :-2], df.iloc[:, -2:-1]

    x = x.to_numpy()
    y = y.to_numpy()

    y = np.where(y <= 6, 1, 0).ravel()

    x_train, x_test, y_train, y_test = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], train_test_split(x, y, test_size=0.8)
    )

    classifier = resolve_model(args.model)
    selector = resolve_selector(args.selector, classifier)

    learner_config = ActiveLearnerConfig(
        classifier,
        selector,
        Path(args.session_name),
        LearningData(x_train[:-10, :], y_train[:-10], x_train[-10:, :]),
    )
    learner = ActiveLearner(learner_config)
    learner.loop(x_test, y_test)


if __name__ == "__main__":
    main()
