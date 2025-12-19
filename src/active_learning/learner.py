from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.models.classifier import Classifier

from .metrics import calculate_pr_metrics
from .selector import Selector

type AUCHistory = list[tuple[float, float]]
type PRHistory = list[tuple[float, np.ndarray, np.ndarray]]


@dataclass
class ActiveLearnerConfig:
    classifier: Classifier
    selector: Selector
    batch_size: int
    store_metrics: bool


@dataclass
class LearningData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_unlabeled: np.ndarray
    y_oracle: np.ndarray


class ActiveLearner:
    def __init__(self, config: ActiveLearnerConfig, data: LearningData) -> None:
        self.classifier = config.classifier
        self.selector = config.selector
        self.batch_size = config.batch_size
        self.store_metrics = config.store_metrics
        self.data = data

        if config.store_metrics:
            self.thresholds: list[float] = [0.25, 0.35, 0.5, 0.75, 1.0]
            self.auc: AUCHistory = []
            self.pr: PRHistory = []

    def loop(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        self.classifier.fit(self.data.X_train, self.data.y_train)
        while self.data.X_unlabeled.shape[0] != 0:
            samples_indices = self.selector(
                self.data.X_unlabeled, self.data.X_train, self.batch_size
            )

            for sample_index in sorted(samples_indices, reverse=True):
                self._label_sample_using_oracle(sample_index)

            self.classifier.fit(self.data.X_train, self.data.y_train)

            if self.store_metrics:
                self._store_metrics(X_test, y_test)

    def _label_sample_using_oracle(self, sample_index: int) -> None:
        sample = self.data.X_unlabeled[sample_index, :]
        label = self.data.y_oracle[sample_index]

        self.data.X_unlabeled = np.delete(self.data.X_unlabeled, sample_index, axis=0)
        self.data.X_train = np.concatenate((self.data.X_train, sample[np.newaxis, :]), axis=0)

        self.data.y_oracle = np.delete(self.data.y_oracle, sample_index)
        self.data.y_train = np.append(self.data.y_train, int(label))

    def _store_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        precision, recall, auc = calculate_pr_metrics(self.classifier, X_test, y_test)
        thr = self._get_closest_unsaved_threshold()
        if thr is not None:
            recall_grid = np.linspace(0, 100, 1)
            self.pr.append(
                (thr, np.interp(recall_grid, recall[::-1], precision[::-1]), recall_grid)
            )
        self.auc.append((self._get_labeled_ratio(), auc))

    def _get_closest_unsaved_threshold(self) -> float | None:
        labeled_ratio = self._get_labeled_ratio()
        for thr in self.thresholds:
            if labeled_ratio >= thr:
                if not any([saved_thr == thr for saved_thr, _, _ in self.pr]):
                    return thr
            else:
                break

    def _get_labeled_ratio(self) -> float:
        return self.data.X_train.shape[0] / (
            self.data.X_train.shape[0] + self.data.X_unlabeled.shape[0]
        )


@dataclass
class TesterConfig:
    save_dir: Path
    n_splits: int = 5
    n_repeats: int = 3
    unlabeled_ratio: float = 0.2
    seed: int = 42


class LearnerTester:
    def __init__(self, learner_config: ActiveLearnerConfig, config: TesterConfig) -> None:
        self.learner_config = learner_config
        self.save_dir = config.save_dir
        self.n_splits = config.n_splits
        self.n_repeats = config.n_repeats
        self.unlabeled_ratio = config.unlabeled_ratio
        self.tester_rng = np.random.default_rng(config.seed)

    def aggregate_results(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self.learner_config.store_metrics:
            raise ValueError("Learener must store metrics to aggregate them later")

        learners: list[ActiveLearner] = []
        learners_X_test: list[np.ndarray] = []
        learners_y_test: list[np.ndarray] = []

        rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats)
        for pool_index, test_index in rskf.split(X, y):
            X_pool, y_pool = X[pool_index, :], y[pool_index]
            X_test, y_test = X[test_index, :], y[test_index]

            threshold = int(X_pool.shape[0] * self.unlabeled_ratio)
            X_train, y_train = X_pool[threshold:, :], y_pool[threshold:]
            X_unlabeled, y_oracle = X_pool[:threshold, :], y_pool[:threshold]

            learner = ActiveLearner(
                self.learner_config, LearningData(X_train, y_train, X_unlabeled, y_oracle)
            )
            learners.append(learner)
            learners_X_test.append(X_test)
            learners_y_test.append(y_test)

        with ProcessPoolExecutor() as executor:
            result = executor.map(
                self._run_single_learner, learners, learners_X_test, learners_y_test
            )
            executor.shutdown()

        all_auc: list[AUCHistory] = []
        all_pr: list[PRHistory] = []
        for auc, pr in result:
            all_auc.append(auc)
            all_pr.append(pr)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._process_auc(all_auc)
        self._process_pr(all_pr)

    def _process_auc(self, all_auc: list[AUCHistory]) -> None:
        labeled_ratio = np.array([round(t[0], 2) for t in all_auc[0]])
        values = np.array([[t[1] for t in record] for record in all_auc])

        data = {
            "labeled_ratio": labeled_ratio,
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0),
            "min": np.min(values, axis=0),
            "max": np.max(values, axis=0),
        }

        pd.DataFrame(data).to_csv(self.save_dir / "auc-results.csv")

    def _process_pr(self, all_pr: list[PRHistory]) -> None:
        thresholds = np.array([t[0] for t in all_pr[0]])
        precision = np.array([[t[1] for t in record] for record in all_pr])
        recall = np.array([[t[2] for t in record] for record in all_pr])

        precision_mean = np.mean(precision, axis=0)
        recall_mean = np.mean(recall, axis=0)

        for i, thr in enumerate(thresholds):
            data = {"recall": recall_mean[i, :], "precision": precision_mean[i, :]}
            pd.DataFrame(data).to_csv(self.save_dir / f"pr-curve-{round(thr, 2) * 100}.csv")

    def _run_single_learner(
        self, learner: ActiveLearner, X_test: np.ndarray, y_test: np.ndarray
    ) -> tuple[AUCHistory, PRHistory]:
        learner.loop(X_test, y_test)
        return learner.auc, learner.pr
