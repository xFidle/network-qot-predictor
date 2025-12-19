from pathlib import Path

import pandas as pd

from src.active_learning.learner import ActiveLearnerConfig, LearnerTester, TesterConfig
from src.active_learning.selector import resolve_selector
from src.config import ConfigParser, LoggerConfig
from src.models.classifier import resolve_model
from src.utils.logger import setup_logger


def main():
    config_parser = ConfigParser()
    logger_config = config_parser.get(LoggerConfig)
    setup_logger(logger_config)

    df = pd.read_csv("data/active_learning/flowers.csv")

    x, y = df.iloc[:, :-1], df.iloc[:, -1]

    x = x.to_numpy()
    y = y.to_numpy()

    classifier = resolve_model("forest")
    selector = resolve_selector("random", classifier)

    learner_config = ActiveLearnerConfig(classifier, selector, 10, True)
    tester_config = TesterConfig(Path("results/forest-test"), n_splits=5, n_repeats=3)
    tester = LearnerTester(learner_config, tester_config)
    print(tester.aggregate_results(x, y))


if __name__ == "__main__":
    main()
