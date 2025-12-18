import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.analysis.pr import PRMetrics, calculate_pr_metrics
from src.learner.gui import LabelingWindow
from src.model.classifier import Classifier
from src.selector.selector import Selector

DEFAULT_STORE_DIR = "sessions"
TRAINING_DATA_DIR = "train"
METRICS_DIR = "metrics"

logger = logging.getLogger(__name__)


class QuitLabeling(Exception):
    pass


@dataclass
class LearningData:
    X_train: np.ndarray
    X_unlabeled: np.ndarray
    y_train: np.ndarray
    X_train_images_paths: list[str]
    X_unlabeled_images_paths: list[str]


@dataclass
class ActiveLearnerConfig:
    classifier: Classifier
    selector: Selector
    save_dir: Path
    learning_data: LearningData | None


class ActiveLearner:
    def __init__(self, config: ActiveLearnerConfig) -> None:
        self.classifier = config.classifier
        self.selector = config.selector
        self.save_dir = DEFAULT_STORE_DIR / config.save_dir
        self.data
        if self.save_dir.exists():
            logger.info("Session restored, data loaded from files")
            self.data = self._load_data()

        elif config.learning_data is not None:
            logger.info("New session created, data loaded from config")
            self.data = config.learning_data

        else:
            msg = (
                "Learning data is not available. Provide it using config or relative directory path"
            )
            logger.error(msg)
            raise ValueError(msg)

    def loop(self, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 5) -> None:
        try:
            self.classifier.fit(self.data.X_train, self.data.y_train)
            logger.info("Initial training done")

            window = LabelingWindow(
                ("sunflower", "dandelion"),
                self.data.X_train.shape[0],
                self.data.X_unlabeled.shape[0],
            )
            while self.data.X_unlabeled.shape[0] != 0:
                logger.info(f"Samples remaining: {self.data.X_unlabeled.shape[0]}")

                samples_indices = self.selector(
                    self.data.X_unlabeled, self.data.X_train, batch_size
                )
                size = len(samples_indices)
                logger.info(f"Selected next {size} samples to label")

                for index in sorted(samples_indices, reverse=True):
                    img_path = Path(self.data.X_unlabeled_images_paths[int(index)])
                    window.set_sample(img_path)

                    label = window.wait_for_label()

                    if label == "q":
                        window.quit()
                        raise QuitLabeling()

                    self._label_sample(index, int(label))
                    logger.info(f"Image {img_path} labeled")

                    window.update_progress_bar(
                        self.data.X_train.shape[0], self.data.X_unlabeled.shape[0]
                    )

                logger.info(f"Samples batch of size {size} successfully labeled")

                window.show_training_status()
                self.classifier.fit(self.data.X_train, self.data.y_train)
                logger.info("Training with new data finished successfully")
                window.hide_training_status()

                metrics = calculate_pr_metrics(self.classifier, X_test, y_test)
                self._save_metrics(metrics)
                logger.info(f"PR Metrics calculated and saved in {self.save_dir / METRICS_DIR}")

            logger.info("Successfully labeled all samples. Learning finished")

            window.quit()
            self._save_data()
            logger.info(f"Learning data saved to {self.save_dir / TRAINING_DATA_DIR}")

        except (KeyboardInterrupt, QuitLabeling):
            logger.info("Process interrupted by user")

            self._save_data()
            logger.info(f"Learning data saved to {self.save_dir / TRAINING_DATA_DIR}")

    def _label_sample(self, sample_index: int, label: int) -> None:
        sample = self.data.X_unlabeled[sample_index, :]
        self.data.X_unlabeled = np.delete(self.data.X_unlabeled, sample_index, axis=0)
        self.data.X_train = np.concatenate((self.data.X_train, sample[np.newaxis, :]), axis=0)
        self.data.y_train = np.append(self.data.y_train, int(label))

        sampled_image = self.data.X_unlabeled_images_paths.pop(sample_index)
        self.data.X_train_images_paths.append(sampled_image)

    def _load_data(self) -> LearningData:
        data_dir = self.save_dir / TRAINING_DATA_DIR

        X_train = np.load(data_dir / "x_train.npy")
        y_train = np.load(data_dir / "y_train.npy")
        x_unlabeled = np.load(data_dir / "x_unlabeled.npy")

        with open(data_dir / "x_train_images_paths", "r") as fh:
            images_train = [line.strip() for line in fh.readlines()]

        with open(data_dir / "x_unlabeled_images_paths", "r") as fh:
            images_unlabeled = [line.strip() for line in fh.readlines()]

        return LearningData(X_train, x_unlabeled, y_train, images_train, images_unlabeled)

    def _save_data(self) -> None:
        data_dir = self.save_dir / TRAINING_DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)

        np.save(data_dir / "x_train", self.data.X_train)
        np.save(data_dir / "y_train", self.data.y_train)
        np.save(data_dir / "x_unlabeled", self.data.X_unlabeled)

        with open(data_dir / "x_train_images_paths", "w") as fh:
            fh.writelines([file + "\n" for file in self.data.X_train_images_paths])

        with open(data_dir / "x_unlabeled_images_paths", "w") as fh:
            fh.writelines([file + "\n" for file in self.data.X_unlabeled_images_paths])

    def _save_metrics(self, metrics: PRMetrics) -> None:
        metrics_dir = self.save_dir / METRICS_DIR
        metrics_dir.mkdir(parents=True, exist_ok=True)

        thresholds = [1.0, 0.5, 0.4, 0.3, 0.25]
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
