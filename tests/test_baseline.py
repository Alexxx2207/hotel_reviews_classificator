"""Tests for src.baseline."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.constants import PATHS, Paths
from src.baseline import (
    evaluate_baseline,
    load_baseline,
    save_baseline,
    train_baseline,
)


def test_train_baseline_returns_classifier() -> None:
    """train_baseline returns a fitted LogisticRegression."""
    rng = np.random.default_rng(100)
    X = rng.random((20, 10))
    y = (rng.random(20) > 0.5).astype(np.int64)
    classifier = train_baseline(X, y)
    assert isinstance(classifier, LogisticRegression)
    assert classifier.predict(X[:3]).shape == (3,)


def test_evaluate_baseline_returns_dict() -> None:
    """evaluate_baseline returns a classification report dict."""

    rng = np.random.default_rng(100)
    X = rng.random((20, 10))
    y = (rng.random(20) > 0.5).astype(np.int64)
    classifier = train_baseline(X, y)
    report = evaluate_baseline(classifier, X[:5], y[:5])
    assert isinstance(report, dict)
    assert "accuracy" in report or "0" in report or "1" in report


def test_save_and_load_baseline(tmp_path_in_project: Path) -> None:
    """Saving and loading round-trips the baseline classifier."""
    paths = Paths(project_root=tmp_path_in_project)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(100)
    X = rng.random((30, 5))
    y = (rng.random(30) > 0.5).astype(np.int64)
    classifier = train_baseline(X, y)
    save_baseline(classifier, paths)
    assert (tmp_path_in_project / "artifacts" / "baseline_logreg.joblib").exists()
    loaded = load_baseline(paths)
    assert isinstance(loaded, LogisticRegression)
    np.testing.assert_array_equal(classifier.predict(X), loaded.predict(X))


def test_baseline_saved_under_artifacts() -> None:
    """Baseline is saved under PATHS.artifacts when using default paths."""
    assert (PATHS.artifacts / "baseline_logreg.joblib").parent == PATHS.artifacts
