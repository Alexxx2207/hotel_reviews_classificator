"""Tests for src.tripadvisor_setup."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.constants import Paths, RATING_COL, TEXT_COL, DATASET_FILE_NAME
from src.tripadvisor_setup import (
    load_dataset_from_csv,
    to_binary_labels,
    split_train_test,
    main as tripadvisor_main,
)


def test_to_binary_labels() -> None:
    """to_binary_labels keeps only neg/pos ratings and maps to 0/1."""

    df = pd.DataFrame({
        TEXT_COL: ["a", "b", "c", "d", "e"],
        RATING_COL: [1, 2, 4, 5, 3],
    })
    out = to_binary_labels(df)

    assert list(out.columns) == [TEXT_COL, "label"]
    assert len(out) == 4
    assert set(out["label"]) == {0, 1}
    assert out["label"].iloc[0] == 0
    assert out["label"].iloc[2] == 1


def test_split_train_test_stratified() -> None:
    """split_train_test returns train and test tuples with stratification."""
    
    df = pd.DataFrame({
        TEXT_COL: [f"text_{i}" for i in range(100)],
        "label": [0] * 50 + [1] * 50,
    })
    (x_train, y_train), (x_test, y_test) = split_train_test(df)

    assert len(x_train) + len(x_test) == 100
    assert len(y_train) + len(y_test) == 100
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    assert 70 <= len(x_train) <= 90
    assert 10 <= len(x_test) <= 30
    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}


def test_load_dataset_from_csv_mocked(tmp_path_in_project: Path) -> None:
    """load_dataset_from_csv returns DataFrame when kagglehub returns a local CSV path."""

    fake_csv = tmp_path_in_project / DATASET_FILE_NAME
    pd.DataFrame({
        TEXT_COL: ["review one", "review two", "review three"],
        RATING_COL: [1, 5, 3],
    }).to_csv(fake_csv, index=False)
    with patch(
        "src.tripadvisor_setup.kagglehub.dataset_download",
        return_value=str(tmp_path_in_project),
    ):
        df = load_dataset_from_csv()

    assert len(df) == 3
    assert list(df.columns) == [TEXT_COL, RATING_COL]
    assert df[TEXT_COL].dtype == object
    assert df[RATING_COL].dtype in (pd.Int64Dtype(), "int64", "int32")


def test_tripadvisor_main_mocked(tmp_path_in_project: Path) -> None:
    """main() runs pipeline when load_dataset_from_csv is mocked."""

    paths = Paths(project_root=tmp_path_in_project)
    mock_df = pd.DataFrame({
        TEXT_COL: [f"text_{i}" for i in range(20)],
        RATING_COL: [1, 2, 4, 5] * 5,
    })
    with patch("src.tripadvisor_setup.load_dataset_from_csv", return_value=mock_df):
        tripadvisor_main(paths)

    assert (tmp_path_in_project / "data" / "train.csv").exists()
    assert (tmp_path_in_project / "data" / "test.csv").exists()

    train_df = pd.read_csv(tmp_path_in_project / "data" / "train.csv")
    test_df = pd.read_csv(tmp_path_in_project / "data" / "test.csv")

    assert len(train_df) + len(test_df) == 20
    assert set(train_df["label"]) <= {0, 1}
    assert set(test_df["label"]) <= {0, 1}
