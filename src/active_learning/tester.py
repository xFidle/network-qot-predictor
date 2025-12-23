import logging
import os
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Manager, Queue
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import register_config
from src.utils.progress_bar import setup_progress_bars

from .learner import (
    ActiveLearner,
    ActiveLearnerConfig,
    ExperimentResults,
    LearningData,
    MultiprocessingContext,
)

logger = logging.getLogger(__name__)


@register_config(name="tester")
@dataclass
class TesterConfig:
    save_dir: str = "results"
    n_splits: int = 5
    n_repeats: int = 3
    labeled_ratio: float = 0.25
    seed: int = 42
    thresholds: list[float] = field(default_factory=lambda: [0.25, 0.3, 0.4, 0.5, 0.75, 1.0])


@dataclass
class PRResult:
    threshold: float
    precision: np.ndarray
    recall: np.ndarray


@dataclass
class LearningBatch:
    data: list[LearningData]
    input: list[np.ndarray]
    target: list[np.ndarray]


class LearnerTester:
    def __init__(self, learner_config: ActiveLearnerConfig, config: TesterConfig) -> None:
        self._learner_config = learner_config
        self._save_dir = Path(config.save_dir)
        self._n_splits = config.n_splits
        self._n_repeats = config.n_repeats
        self._labeled_ratio = config.labeled_ratio
        self._thresholds = config.thresholds
        self._tester_rng = np.random.default_rng(config.seed)

    def run(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self._learner_config.should_store_results:
            raise ValueError("Learener must store metrics to aggregate them later")

        batch = self._get_splits(X, y)
        logger.info("Splits for repeated K-fold created")

        logger.info("Processing splits")
        trials = self._process_data(batch)

        aucs = self._extract_aucs(trials)
        logger.info("PR AUCs extracted from experiments")

        prs = self._extract_prs(trials)
        logger.info("Precision-recall curves exctracted from experiments")

        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._save_aucs(trials[0].labeled_ratio, aucs)
        logger.info("PR AUCs saved to .csv file")

        self._save_prs(prs)
        logger.info("Precision-recall curves saved to .csv files")

    def _get_splits(self, X: np.ndarray, y: np.ndarray) -> LearningBatch:
        data: list[LearningData] = []
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        rskf = RepeatedStratifiedKFold(n_splits=self._n_splits, n_repeats=self._n_repeats)
        for train_index, test_index in rskf.split(X, y):
            X_train, y_train = X[train_index, :], y[train_index]
            X_test, y_test = X[test_index, :], y[test_index]

            n_train = X_train.shape[0]
            n_labeled = int(n_train * self._labeled_ratio)
            labeled_mask = np.zeros(n_train, dtype=bool)
            labeled_mask[:n_labeled] = True
            self._tester_rng.shuffle(labeled_mask)

            data.append(LearningData(X_train, y_train, labeled_mask))
            inputs.append(X_test)
            targets.append(y_test)

        return LearningBatch(data, inputs, targets)

    def _process_data(self, batch: LearningBatch) -> list[ExperimentResults]:
        n_learners = len(batch.data)
        with setup_progress_bars() as progress:
            with Manager() as manager:
                futures: list[Future[ExperimentResults]] = []
                update_queue: "Queue[dict[str, Any]]" = manager.Queue()  # type: ignore
                overall_progress = progress.add_task("[green]Total learners progress:")

                with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                    for i in range(n_learners):
                        learner_id = progress.add_task(f"Learner {i:02d}", visible=True)
                        futures.append(
                            executor.submit(
                                self._run_single_learner,
                                batch.data[i],
                                batch.input[i],
                                batch.target[i],
                                MultiprocessingContext(learner_id, update_queue),
                            )
                        )

                    n_finished = 0
                    while n_finished < len(futures):
                        event = update_queue.get()
                        progress.update(overall_progress, completed=n_finished, total=n_learners)

                        task_id = event["task_id"]
                        completed = event["completed"]
                        total = event["total"]
                        progress.update(task_id, completed=completed, total=total)

                        if completed == total:
                            n_finished += 1

                    progress.update(overall_progress, completed=n_learners)

        return [f.result() for f in futures]

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

        for thr in self._thresholds:
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
            "median": np.median(aucs, axis=0),
            "std": np.std(aucs, axis=0),
            "min": np.min(aucs, axis=0),
            "max": np.max(aucs, axis=0),
            "p25": np.percentile(aucs, 25, axis=0),
            "p75": np.percentile(aucs, 75, axis=0),
        }

        pd.DataFrame(data).to_csv(self._save_dir / "auc-results.csv", index=False)

    def _save_prs(self, prs: list[PRResult]) -> None:
        for pr in prs:
            data = {"precision": pr.precision, "recall": pr.recall}
            pd.DataFrame(data).to_csv(
                self._save_dir / f"precision-recall-{round(pr.threshold, 2) * 100}.csv", index=False
            )

    def _run_single_learner(
        self,
        learning_data: LearningData,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ctx: MultiprocessingContext,
    ) -> ExperimentResults:
        learner = ActiveLearner(self._learner_config, learning_data)
        learner.loop(X_test, y_test, ctx)
        return learner.results
