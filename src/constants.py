from dataclasses import dataclass
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
    project_root: Path = Path(__file__).resolve().parents[1]
    data_processed: Path = project_root / "data"
    artifacts: Path = project_root / "artifacts"
    metrics: Path = artifacts / "metrics"
    plots: Path = artifacts / "plots"

PATHS = Paths()