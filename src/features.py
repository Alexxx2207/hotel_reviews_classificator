"""
Features' processing functions.
"""

from __future__ import annotations

from typing import cast

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.constants import PATHS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, Paths


def fit_vectorizer(reviews: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    """Fits a TF-IDF vectorizer to the reviews and returns the vectorizer and feature matrix."""

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        lowercase=True,
        strip_accents=None,
    )
    feature_matrix = cast(
        np.ndarray,
        vectorizer.fit_transform(reviews).toarray(),  # pyright: ignore[reportAttributeAccessIssue]
    )

    return vectorizer, feature_matrix


def transform(vectorizer: TfidfVectorizer, reviews: list[str]) -> np.ndarray:
    """Transforms the reviews using the given vectorizer; returns dense matrix."""

    return cast(
        np.ndarray,
        vectorizer.transform(reviews).toarray(),  # pyright: ignore[reportAttributeAccessIssue]
    )


def save_vectorizer(vectorizer: TfidfVectorizer, paths: Paths = PATHS) -> None:
    """Saves a TF-IDF vectorizer to the artifacts directory."""

    paths.artifacts.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, paths.artifacts / "tfidf.joblib")


def load_vectorizer(paths: Paths = PATHS) -> TfidfVectorizer:
    """Loads a TF-IDF vectorizer from the artifacts directory."""

    return joblib.load(paths.artifacts / "tfidf.joblib")
