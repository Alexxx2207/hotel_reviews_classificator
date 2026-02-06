"""Tests for src.constants."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.constants import (
    BATCH_SIZE,
    DATASET_FILE_NAME,
    DATASET_NAME,
    EPOCHS,
    LR,
    MLP_DROPOUT,
    MLP_HIDDEN,
    NEG_RATINGS,
    PATHS,
    POS_RATINGS,
    RANDOM_STATE,
    RATING_COL,
    TEST_SIZE,
    TEXT_COL,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    WEIGHT_DECAY,
    Paths,
)


def test_paths_dataclass() -> None:
    """Paths has expected attributes and project_root is a parent of others."""
    assert PATHS.project_root.is_dir() or not PATHS.project_root.exists()
    assert PATHS.data_processed == PATHS.project_root / "data"
    assert PATHS.artifacts == PATHS.project_root / "artifacts"
    assert PATHS.metrics == PATHS.artifacts / "metrics"
    assert PATHS.plots == PATHS.artifacts / "plots"


def test_paths_with_custom_root(tmp_path_in_project: Path) -> None:
    """Paths(project_root=X) derives other paths from X."""
    paths = Paths(project_root=tmp_path_in_project)
    assert paths.project_root == tmp_path_in_project
    assert paths.data_processed == tmp_path_in_project / "data"
    assert paths.artifacts == tmp_path_in_project / "artifacts"
    assert paths.metrics == tmp_path_in_project / "artifacts" / "metrics"
    assert paths.plots == tmp_path_in_project / "artifacts" / "plots"


def test_paths_is_frozen() -> None:
    """Paths is frozen so attributes cannot be reassigned."""
    with pytest.raises(FrozenInstanceError):
        PATHS.project_root = Path("/other")


def test_constants_values() -> None:
    """Constants have expected values."""
    assert DATASET_NAME == "andrewmvd/trip-advisor-hotel-reviews"
    assert DATASET_FILE_NAME == "tripadvisor_hotel_reviews.csv"
    assert TEXT_COL == "Review"
    assert RATING_COL == "Rating"
    assert RANDOM_STATE == 42
    assert TEST_SIZE == 0.2
    assert NEG_RATINGS == {1, 2}
    assert POS_RATINGS == {4, 5}
    assert TFIDF_MAX_FEATURES == 40000
    assert TFIDF_NGRAM_RANGE == (1, 2)
    assert MLP_HIDDEN == 256
    assert MLP_DROPOUT == 0.2
    assert EPOCHS == 8
    assert BATCH_SIZE == 256
    assert LR == 1e-3
    assert WEIGHT_DECAY == 1e-4
