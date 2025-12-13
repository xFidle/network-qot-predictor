import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.anaylsis.pr import PRMetrics, calculate_pr_metrics
from src.model.classifier import Classifier
from src.selector.selector import Selector

DEFAULT_STORE_DIR = ".tmp"
DATASET_DIR_NAME = "dataset"
METRICS_DIR_NAME = "metrics"
SAVING_THRESHOLDS = [1.0, 0.5, 0.4, 0.3, 0.25]


class QuitLabeling(Exception):
    pass


@dataclass
class LearningData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_unlabeled: np.ndarray


@dataclass
class ActiveLearnerConfig:
    classifier: Classifier
    selector: Selector
    save_dir: Path
    learning_data: LearningData | None


class ActiveLearner:
    def __init__(self, config: ActiveLearnerConfig) -> None:
        self.classifier: Classifier = config.classifier
        self.selector: Selector = config.selector
        self.save_dir: Path = DEFAULT_STORE_DIR / config.save_dir
        self.data: LearningData

        if self.save_dir.exists():
            self.data = self._load_learning_data()

        elif config.learning_data is not None:
            self.data = config.learning_data

        else:
            raise ValueError(
                "No learning data avaliable. Either provide learning_data in config or path with exsiting data."
            )

    def loop(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        try:
            self.classifier.fit(self.data.X_train, self.data.y_train)
            while self.data.X_unlabeled.shape[0] != 0:
                samples_indices = self.selector(self.data.X_unlabeled, batch_size=5)
                print("Samples to label have been chosen!")
                for sample in sorted(samples_indices, reverse=True):
                    self._label_sample(sample)
                print("Selected labels saved.")

                self.classifier.fit(self.data.X_train, self.data.y_train)
                metrics = calculate_pr_metrics(self.classifier, X_test, y_test)
                self._save_metrics(metrics)
                print("Current PR AUC score: ", metrics.auc_score, "\n")

            print("All samples already labeled.")
            self._save_data()

        except (KeyboardInterrupt, QuitLabeling):
            self._save_data()

    def _label_sample(self, sample_index: int) -> None:
        # TODO: display sample's image, not feature vector
        print("Sample: ", self.data.X_unlabeled[sample_index, :])
        while (label := input("Enter label (0 or 1): ")) not in ("0", "1", "q"):
            print("Invalid label")

        if label == "q":
            raise QuitLabeling()

        sample = self.data.X_unlabeled[sample_index, :]
        self.data.X_unlabeled = np.delete(self.data.X_unlabeled, sample_index, axis=0)
        self.data.X_train = np.concatenate((self.data.X_train, sample[np.newaxis, :]), axis=0)
        self.data.y_train = np.append(self.data.y_train, int(label))

    def _load_learning_data(self) -> LearningData:
        data_dir = self.save_dir / DATASET_DIR_NAME

        X_train = np.load(data_dir / "x_train.npy")
        y_train = np.load(data_dir / "y_train.npy")
        x_unlabeled = np.load(data_dir / "x_unlabeled.npy")

        return LearningData(X_train, y_train, x_unlabeled)

    def _save_data(self) -> None:
        data_dir = self.save_dir / DATASET_DIR_NAME
        data_dir.mkdir(parents=True, exist_ok=True)

        np.save(data_dir / "x_train", self.data.X_train)
        np.save(data_dir / "y_train", self.data.y_train)
        np.save(data_dir / "x_unlabeled", self.data.X_unlabeled)

    def _save_metrics(self, metrics: PRMetrics) -> None:
        metrics_dir = self.save_dir / METRICS_DIR_NAME
        metrics_dir.mkdir(parents=True, exist_ok=True)

        thresholds = SAVING_THRESHOLDS
        labeled_ratio = self.data.X_train.shape[0] / (
            self.data.X_train.shape[0] + self.data.X_unlabeled.shape[0]
        )

        for thr in thresholds:
            if labeled_ratio >= thr:
                thr_pct = int(round(thr * 100, 2))
                precision_file = metrics_dir / f"precision-{thr_pct}"
                recall_file = metrics_dir / f"recall-{thr_pct}"

                if not precision_file.exists() and not recall_file.exists():
                    np.save(precision_file, metrics.precision)
                    np.save(recall_file, metrics.recall)
            break

        auc_file = metrics_dir / "pr-auc.csv"
        auc_exists = auc_file.exists()

        with open(auc_file, mode="a") as fh:
            writer = csv.writer(fh)
            if not auc_exists:
                writer.writerow(["labeled_ratio(%)", "auc_score"])
            writer.writerow([round(labeled_ratio * 100, 2), metrics.auc_score])
