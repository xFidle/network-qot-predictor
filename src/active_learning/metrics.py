from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from .learner import ActiveLearner, ActiveLearnerConfig, ExperimentResults, LearningData


@dataclass
class TesterConfig:
    save_dir: str
    n_splits: int = 5
    n_repeats: int = 3
    labeled_ratio: float = 0.2
    seed: int = 42
    thresholds: list[float] = field(default_factory=lambda: [0.25, 0.3, 0.4, 0.5, 0.75, 1.0])


@dataclass
class PRResult:
    threshold: float
    precision: np.ndarray
    recall: np.ndarray


class LearnerTester:
    def __init__(self, learner_config: ActiveLearnerConfig, config: TesterConfig) -> None:
        self.learner_config = learner_config
        self.save_dir = Path(config.save_dir)
        self.n_splits = config.n_splits
        self.n_repeats = config.n_repeats
        self.labeled_ratio = config.labeled_ratio
        self.thresholds = config.thresholds
        self.tester_rng = np.random.default_rng(config.seed)

    def aggregate_results(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self.learner_config.store_results:
            raise ValueError("Learener must store metrics to aggregate them later")

        learners: list[ActiveLearner] = []
        learners_X_test: list[np.ndarray] = []
        learners_y_test: list[np.ndarray] = []

        rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats)
        for train_index, test_index in rskf.split(X, y):
            X_train, y_train = X[train_index, :], y[train_index]
            X_test, y_test = X[test_index, :], y[test_index]

            n_train = X_train.shape[0]
            n_labeled = int(n_train * self.labeled_ratio)
            labeled_mask = np.zeros(n_train, dtype=bool)
            labeled_mask[:n_labeled] = True
            self.tester_rng.shuffle(labeled_mask)

            learner = ActiveLearner(
                self.learner_config, LearningData(X_train, y_train, labeled_mask)
            )

            learners.append(learner)
            learners_X_test.append(X_test)
            learners_y_test.append(y_test)

        with ProcessPoolExecutor() as executor:
            proccss_pool_result = executor.map(
                self._run_single_learner, learners, learners_X_test, learners_y_test
            )

        trials = [result for result in proccss_pool_result]

        aucs = self._extract_aucs(trials)
        prs = self._extract_prs(trials)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._save_aucs(trials[0].labeled_ratio, aucs)
        self._save_prs(prs)

    def _extract_aucs(self, trials: list[ExperimentResults]) -> np.ndarray:
        aucs = np.array(
            [
                [average_precision_score(result.y_test, proba) for proba in result.proba]
                for result in trials
            ]
        )
        return aucs

    def _extract_prs(self, trials: list[ExperimentResults]) -> list[PRResult]:
        y_test = np.concatenate([result.y_test for result in trials])

        ratios = trials[0].labeled_ratio
        result: list[PRResult] = []

        min_ratio = ratios[0]

        for thr in self.thresholds:
            if thr < min_ratio:
                continue

            idx = next((i for i, ratio in enumerate(ratios) if ratio >= thr), None)
            if idx is None:
                break

            thr_proba = np.concatenate([result.proba[idx] for result in trials])
            precision, recall, _ = precision_recall_curve(y_test, thr_proba)
            result.append(PRResult(thr, precision, recall))

        return result

    def _save_aucs(self, ratios: list[float], aucs: np.ndarray) -> None:
        data = {
            "labeled_ratio": ratios,
            "mean": np.mean(aucs, axis=0),
            "std": np.std(aucs, axis=0),
            "min": np.min(aucs, axis=0),
            "max": np.max(aucs, axis=0),
        }

        pd.DataFrame(data).to_csv(self.save_dir / "auc-results.csv", index=False)

    def _save_prs(self, prs: list[PRResult]) -> None:
        for pr in prs:
            data = {"precision": pr.precision, "recall": pr.recall}
            pd.DataFrame(data).to_csv(
                self.save_dir / f"precision-recall-{round(pr.threshold, 2) * 100}.csv", index=False
            )

    def _run_single_learner(
        self, learner: ActiveLearner, X_test: np.ndarray, y_test: np.ndarray
    ) -> ExperimentResults:
        learner.loop(X_test, y_test)
        return learner.results
