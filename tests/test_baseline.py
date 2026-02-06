"""Tests for src.baseline."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src import baseline, constants
from src.constants import PATHS
from src.baseline import (
    BASELINE_PATH,
    evaluate_baseline,
    load_baseline,
    save_baseline,
    train_baseline,
)


def test_train_baseline_returns_classifier() -> None:
    """train_baseline returns a fitted LogisticRegression."""
    rng = np.random.default_rng(42)
    X = rng.random((20, 10))
    y = (rng.random(20) > 0.5).astype(np.int64)
    clf = train_baseline(X, y)
    assert isinstance(clf, LogisticRegression)
    pred = clf.predict(X[:3])
    assert pred.shape == (3,)


def test_evaluate_baseline_returns_dict() -> None:
    """evaluate_baseline returns a classification report dict."""

    rng = np.random.default_rng(42)
    X = rng.random((20, 10))
    y = (rng.random(20) > 0.5).astype(np.int64)
    clf = train_baseline(X, y)
    report = evaluate_baseline(clf, X[:5], y[:5])
    assert isinstance(report, dict)
    assert "accuracy" in report or "0" in report or "1" in report


def test_save_and_load_baseline(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving and loading round-trips the baseline classifier."""
    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    monkeypatch.setattr(baseline, "PATHS", constants.PATHS)
    monkeypatch.setattr(baseline, "BASELINE_PATH", tmp_path_in_project / "baseline.joblib")

    rng = np.random.default_rng(42)
    X = rng.random((30, 5))
    y = (rng.random(30) > 0.5).astype(np.int64)
    clf = train_baseline(X, y)
    save_baseline(clf)
    assert (tmp_path_in_project / "baseline.joblib").exists()
    loaded = load_baseline()
    assert isinstance(loaded, LogisticRegression)
    np.testing.assert_array_equal(clf.predict(X), loaded.predict(X))


def test_baseline_path_in_artifacts() -> None:
    """BASELINE_PATH is under PATHS.artifacts."""
    assert BASELINE_PATH == PATHS.artifacts / "baseline_logreg.joblib"
