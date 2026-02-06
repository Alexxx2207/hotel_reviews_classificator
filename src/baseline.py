"""
Baseline model for the project.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.constants import PATHS, Paths


def train_baseline(
    x_train: np.ndarray, y_train: np.ndarray
) -> LogisticRegression:
    """Trains a baseline logistic regression classifier."""

    classifier = LogisticRegression(max_iter=2000, n_jobs=None)
    classifier.fit(x_train, y_train)
    return classifier


def evaluate_baseline(
    classifier: LogisticRegression,
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Evaluates a baseline logistic regression classifier."""

    prediction = classifier.predict(x)
    return classification_report(y, prediction, output_dict=True)


def save_baseline(
    classifier: LogisticRegression,
    paths: Paths | None = None,
) -> None:
    """Saves a baseline logistic regression classifier in the artifacts directory."""
    p = paths or PATHS
    p.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, p.artifacts / "baseline_logreg.joblib")


def load_baseline(paths: Paths | None = None) -> LogisticRegression:
    """Loads a baseline logistic regression classifier from the artifacts directory."""
    p = paths or PATHS
    return joblib.load(p.artifacts / "baseline_logreg.joblib")
