"""
Baseline model for the project.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.constants import PATHS, Paths


def train_baseline(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Trains a baseline logistic regression classifier."""

    classifier = LogisticRegression(max_iter=2000, n_jobs=None)
    classifier.fit(x_train, y_train)
    return classifier


def save_baseline(classifier: LogisticRegression, paths: Paths = PATHS) -> None:
    """Saves a baseline logistic regression classifier in the artifacts directory."""

    paths.artifacts.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, paths.artifacts / "baseline_logreg.joblib")


def load_baseline(paths: Paths = PATHS) -> LogisticRegression:
    """Loads a baseline logistic regression classifier from the artifacts directory."""

    return joblib.load(paths.artifacts / "baseline_logreg.joblib")
