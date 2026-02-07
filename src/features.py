"""
Features' processing functions.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.constants import PATHS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, Paths


def fit_vectorizer(reviews: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    """Fits a TF-IDF vectorizer to the reviews and returns dense feature matrix."""

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        lowercase=True,
        strip_accents=None,
    )
    feature_matrix = vectorizer.fit_transform(reviews).toarray()  # type: ignore[union-attr]
    return vectorizer, feature_matrix


def transform(
    vectorizer: TfidfVectorizer, reviews: list[str]
) -> np.ndarray:
    """Transforms the reviews using the given vectorizer; returns dense matrix."""

    return vectorizer.transform(reviews).toarray()  # type: ignore[union-attr]


def save_vectorizer(
    vectorizer: TfidfVectorizer,
    paths: Paths | None = None,
) -> None:
    """Saves a TF-IDF vectorizer to the artifacts directory."""

    p = paths or PATHS
    p.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, p.artifacts / "tfidf.joblib")


def load_vectorizer(paths: Paths | None = None) -> TfidfVectorizer:
    """Loads a TF-IDF vectorizer from the artifacts directory."""

    p = paths or PATHS
    return joblib.load(p.artifacts / "tfidf.joblib")
