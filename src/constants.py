"""
Constants for the project.
"""

from dataclasses import dataclass
from pathlib import Path


DATASET_NAME = "andrewmvd/trip-advisor-hotel-reviews"
DATASET_FILE_NAME = "tripadvisor_hotel_reviews.csv"

TEXT_COL = "Review"
RATING_COL = "Rating"

TEST_SIZE = 0.2

NEG_RATINGS = [1, 2]
POS_RATINGS = [4, 5]

TFIDF_MAX_FEATURES = 40000
TFIDF_NGRAM_RANGE = (1, 2)

MLP_HIDDEN = 256
MLP_DROPOUT = 0.2
EPOCHS = 8
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4

HF_MODEL_NAME = "kmack/HotelReviewClassifier"

@dataclass(frozen=True)
class Paths:
    """Paths to the important directories for the project."""

    project_root: Path = Path(__file__).resolve().parents[1]

    @property
    def data_processed(self) -> Path:
        """Returns the path to the processed data."""

        return self.project_root / "data"

    @property
    def artifacts(self) -> Path:
        """Returns the path to the artifacts."""

        return self.project_root / "artifacts"

    @property
    def metrics(self) -> Path:
        """Returns the path to the metrics."""

        return self.artifacts / "metrics"

    @property
    def plots(self) -> Path:
        """Returns the path to the plots."""

        return self.artifacts / "plots"


PATHS = Paths()
