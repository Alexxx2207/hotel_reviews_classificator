"""
Constants for the project.
"""

from dataclasses import dataclass, field
from pathlib import Path


DATASET_NAME = "andrewmvd/trip-advisor-hotel-reviews"
DATASET_FILE_NAME = "tripadvisor_hotel_reviews.csv"

TEXT_COL = "Review"
RATING_COL = "Rating"

RANDOM_STATE = 42
TEST_SIZE = 0.2

NEG_RATINGS = {1, 2}
POS_RATINGS = {4, 5}

TFIDF_MAX_FEATURES = 40000
TFIDF_NGRAM_RANGE = (1, 2)

MLP_HIDDEN = 256
MLP_DROPOUT = 0.2
EPOCHS = 8
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4

@dataclass(frozen=True)
class Paths:
    """Paths to the important directories for the project."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_processed: Path = field(init=False)
    artifacts: Path = field(init=False)
    metrics: Path = field(init=False)
    plots: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_processed", self.project_root / "data")
        object.__setattr__(self, "artifacts", self.project_root / "artifacts")
        object.__setattr__(self, "metrics", self.artifacts / "metrics")
        object.__setattr__(self, "plots", self.artifacts / "plots")


PATHS = Paths()
