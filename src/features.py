"""
Features' processing functions.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.constants import PATHS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE

VECTORIZER_PATH = PATHS.artifacts / "tfidf.joblib"


def fit_vectorizer(reviews: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    """Fits a TF-IDF vectorizer to the reviews."""

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        lowercase=True,
        strip_accents=None,
    )
    feature_matrix = vectorizer.fit_transform(reviews).toarray()
    return vectorizer, feature_matrix


def transform(
    vectorizer: TfidfVectorizer, reviews: list[str]
) -> np.ndarray:
    """Transforms the reviews using the given vectorizer."""

    return vectorizer.transform(reviews).toarray()


def save_vectorizer(vectorizer: TfidfVectorizer) -> None:
    """Saves a TF-IDF vectorizer to the artifacts directory."""

    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)


def load_vectorizer() -> TfidfVectorizer:
    """Loads a TF-IDF vectorizer from the artifacts directory."""

    return joblib.load(VECTORIZER_PATH)
