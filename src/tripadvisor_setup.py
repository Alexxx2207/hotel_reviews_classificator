"""
Sets up the TripAdvisor dataset for the project.
"""

from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import (
    DATASET_FILE_NAME,
    DATASET_NAME,
    NEG_RATINGS,
    PATHS,
    POS_RATINGS,
    RANDOM_STATE,
    RATING_COL,
    TEST_SIZE,
    TEXT_COL,
    Paths,
)


def load_dataset_from_csv() -> pd.DataFrame:
    """Load data from the Kaggle dataset CSV into a DataFrame."""

    dataset_dir = kagglehub.dataset_download(DATASET_NAME)
    df = pd.read_csv(Path(dataset_dir) / DATASET_FILE_NAME)
    df = df[[TEXT_COL, RATING_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[RATING_COL] = df[RATING_COL].astype(int)
    return df


def to_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ratings with binary labels (0=negative, 1=positive)."""

    df = df[df[RATING_COL].isin(NEG_RATINGS | POS_RATINGS)].copy()
    df["label"] = df[RATING_COL].apply(
        lambda r: 0 if r in NEG_RATINGS else 1
    )
    return df[[TEXT_COL, "label"]]


def split_train_test(
    df: pd.DataFrame,
) -> tuple[
    tuple[pd.Series, pd.Series],
    tuple[pd.Series, pd.Series],
]:
    """Split dataset into train and test with stratification."""

    x_train, x_test, y_train, y_test = train_test_split(
        df[TEXT_COL],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    return (x_train, y_train), (x_test, y_test)


def save_splits(
    train: tuple[pd.Series, pd.Series],
    test: tuple[pd.Series, pd.Series],
    paths: Paths | None = None,
) -> None:
    """Save train and test splits to CSV in the processed data directory."""
    p = paths or PATHS
    (x_train, y_train) = train
    (x_test, y_test) = test
    p.data_processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TEXT_COL: x_train, "label": y_train}).to_csv(
        p.data_processed / "train.csv", index=False
    )
    pd.DataFrame({TEXT_COL: x_test, "label": y_test}).to_csv(
        p.data_processed / "test.csv", index=False
    )


def main(paths: Paths | None = None) -> None:
    """Main function to run the setup."""
    p = paths or PATHS
    df = load_dataset_from_csv()
    df = to_binary_labels(df)
    train, test = split_train_test(df)
    save_splits(train, test, p)


if __name__ == "__main__":
    main()
