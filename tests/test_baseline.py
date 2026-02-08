"""Tests for src.baseline."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.constants import Paths
from src.baseline import load_baseline, save_baseline, train_baseline


def test_train_baseline_returns_classifier() -> None:
    """train_baseline returns a fitted LogisticRegression."""

    rng = np.random.default_rng(100)
    reviews_train = rng.random((20, 10))
    labels_train = (rng.random(20) > 0.5).astype(int)
    
    classifier = train_baseline(reviews_train, labels_train)

    assert isinstance(classifier, LogisticRegression)
    assert classifier.predict(reviews_train[:3]).shape == (3,)


def test_save_and_load_baseline(tmp_path_in_project: Path) -> None:
    """Saving and loading round-trips the baseline classifier."""

    paths = Paths(project_root=tmp_path_in_project)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(100)
    reviews_train = rng.random((30, 5))
    labels_train = (rng.random(30) > 0.5).astype(int)
    
    classifier = train_baseline(reviews_train, labels_train)
    save_baseline(classifier, paths)

    assert (tmp_path_in_project / "artifacts" / "baseline_logreg.joblib").exists()

    loaded = load_baseline(paths)

    assert isinstance(loaded, LogisticRegression)

    # Hands down най-добрата библиотека ^^
    np.testing.assert_array_equal(classifier.predict(reviews_train), loaded.predict(reviews_train))
