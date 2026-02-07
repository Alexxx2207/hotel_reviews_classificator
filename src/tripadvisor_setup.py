"""
Sets up the TripAdvisor dataset for the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import (
    DATASET_FILE_NAME,
    DATASET_NAME,
    NEG_RATINGS,
    PATHS,
    POS_RATINGS,
    RATING_COL,
    TEST_SIZE,
    TEXT_COL,
    Paths,
)


def load_dataset_from_csv() -> pd.DataFrame:
    """Load data from the Kaggle dataset CSV into a DataFrame."""

    dataset_dir = kagglehub.dataset_download(DATASET_NAME)

    df = pd.read_csv(Path(dataset_dir) / DATASET_FILE_NAME)

    df = cast(
        pd.DataFrame,
        df[[TEXT_COL, RATING_COL]].dropna(),
    )

    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[RATING_COL] = df[RATING_COL].astype(int)

    return df


def to_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ratings with binary labels (0=negative, 1=positive)."""

    df = cast(
        pd.DataFrame,
        df[df[RATING_COL].isin(NEG_RATINGS + POS_RATINGS)].copy(),
    )

    df["label"] = df[RATING_COL].apply(
        lambda r: 0 if r in NEG_RATINGS else 1
    )

    return cast(
        pd.DataFrame,
        df[[TEXT_COL, "label"]],
    )


def split_train_test(
    df: pd.DataFrame,
) -> tuple[
    tuple[pd.Series, pd.Series],
    tuple[pd.Series, pd.Series],
]:
    """Split dataset into train and test with stratification."""

    reviews_train, reviews_test, label_train, label_test = train_test_split(
        df[TEXT_COL],
        df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
    )
    return cast(
        tuple[tuple[pd.Series, pd.Series], tuple[pd.Series, pd.Series]],
        ((reviews_train, label_train), (reviews_test, label_test)),
    )


def save_splits(
    train: tuple[pd.Series, pd.Series],
    test: tuple[pd.Series, pd.Series],
    paths: Paths | None = None,
) -> None:
    """Save train and test splits to CSV in the processed data directory."""

    p = paths or PATHS

    (reviews_train, label_train) = train
    (reviews_test, label_test) = test

    p.data_processed.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({TEXT_COL: reviews_train, "label": label_train}).to_csv(
        p.data_processed / "train.csv", index=False
    )
    pd.DataFrame({TEXT_COL: reviews_test, "label": label_test}).to_csv(
        p.data_processed / "test.csv", index=False
    )


def main(paths: Paths | None = None) -> None:
    """Main function to run the setup."""

    p = paths or PATHS

    train, test = split_train_test(to_binary_labels(load_dataset_from_csv()))

    save_splits(train, test, p)


if __name__ == "__main__":
    main()
