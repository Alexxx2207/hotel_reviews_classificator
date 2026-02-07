"""Tests for src.features."""

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.constants import Paths
from src.features import (
    fit_vectorizer,
    load_vectorizer,
    save_vectorizer,
    transform,
)


def test_fit_vectorizer_returns_vectorizer_and_matrix() -> None:
    """fit_vectorizer returns a fitted vectorizer and dense feature matrix."""
    
    reviews = ["great hotel", "bad room", "great stay"]
    vec, matrix = fit_vectorizer(reviews)
    assert isinstance(vec, TfidfVectorizer)
    assert matrix.shape == (3, matrix.shape[1])
    assert matrix.dtype in (np.float32, np.float64)
    assert not np.any(np.isnan(matrix))


def test_transform_uses_fitted_vectorizer() -> None:
    """transform produces same vocab shape as fit_vectorizer."""
    
    train = ["good", "bad", "nice"]
    vec, train_matrix = fit_vectorizer(train)
    test_matrix = transform(vec, ["good nice"])
    assert test_matrix.shape[0] == 1
    assert test_matrix.shape[1] == train_matrix.shape[1]


def test_save_and_load_vectorizer(tmp_path_in_project: Path) -> None:
    """Saving and loading round-trips the vectorizer."""
    paths = Paths(project_root=tmp_path_in_project)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    reviews = ["one two", "two three", "three four"]
    vec, _ = fit_vectorizer(reviews)
    save_vectorizer(vec, paths)
    assert (tmp_path_in_project / "artifacts" / "tfidf.joblib").exists()
    loaded = load_vectorizer(paths)
    assert loaded.vocabulary_ == vec.vocabulary_
    matrix_orig = transform(vec, ["one three"])
    matrix_loaded = transform(loaded, ["one three"])
    np.testing.assert_array_almost_equal(matrix_orig, matrix_loaded)



