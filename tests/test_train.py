"""Tests for src.train."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import constants
from src.constants import PATHS, TEXT_COL
from src.train import (
    ensure_directories_exist,
    load_split,
    train_mlp_only_train,
)


def test_load_split_returns_texts_and_labels(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_split returns (list of texts, array of labels)."""
    from src import constants

    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    (tmp_path_in_project / "data").mkdir(parents=True)
    df = pd.DataFrame({
        TEXT_COL: ["a", "b", "c"],
        "label": [0, 1, 0],
    })
    df.to_csv(tmp_path_in_project / "data" / "train.csv", index=False)
    monkeypatch.setattr("src.train.PATHS", constants.PATHS)

    texts, labels = load_split("train")
    assert texts == ["a", "b", "c"]
    np.testing.assert_array_equal(labels, np.array([0, 1, 0]))


def test_ensure_directories_exist(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ensure_directories_exist creates artifacts, metrics, plots."""
    from src import constants

    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    monkeypatch.setattr("src.train.PATHS", constants.PATHS)

    ensure_directories_exist()
    assert (tmp_path_in_project / "artifacts").is_dir()
    assert (tmp_path_in_project / "artifacts" / "metrics").is_dir()
    assert (tmp_path_in_project / "artifacts" / "plots").is_dir()


def test_train_mlp_only_train_returns_history(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_mlp_only_train returns history and saves checkpoint."""

    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    monkeypatch.setattr("src.train.PATHS", constants.PATHS)
    (tmp_path_in_project / "artifacts").mkdir(parents=True)

    rng = np.random.default_rng(42)
    n, in_features = 50, 20
    x_train = rng.random((n, in_features)).astype(np.float32)
    y_train = (rng.random(n) > 0.5).astype(np.int64)

    # Override EPOCHS to 1 for speed
    import src.train as train_mod
    monkeypatch.setattr(train_mod, "EPOCHS", 1)

    history = train_mlp_only_train(x_train, y_train, in_features)
    assert isinstance(history, list)
    assert len(history) == 1
    assert "epoch" in history[0]
    assert "train_loss" in history[0]
    assert "train_macro_f1" in history[0]
    assert (tmp_path_in_project / "artifacts" / "mlp_last.pt").exists()


def test_train_main(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() runs full pipeline and creates artifacts + history CSV."""
    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    monkeypatch.setattr("src.train.PATHS", constants.PATHS)
    monkeypatch.setattr("src.features.PATHS", constants.PATHS)
    monkeypatch.setattr("src.features.VECTORIZER_PATH", tmp_path_in_project / "artifacts" / "tfidf.joblib")
    monkeypatch.setattr("src.baseline.PATHS", constants.PATHS)
    monkeypatch.setattr("src.baseline.BASELINE_PATH", tmp_path_in_project / "artifacts" / "baseline_logreg.joblib")

    (tmp_path_in_project / "data").mkdir(parents=True)
    pd.DataFrame({
        TEXT_COL: ["good hotel"] * 10 + ["bad room"] * 10,
        "label": [1] * 10 + [0] * 10,
    }).to_csv(tmp_path_in_project / "data" / "train.csv", index=False)

    import src.train as train_mod
    monkeypatch.setattr(train_mod, "EPOCHS", 1)
    train_mod.main()

    assert (tmp_path_in_project / "artifacts" / "tfidf.joblib").exists()
    assert (tmp_path_in_project / "artifacts" / "baseline_logreg.joblib").exists()
    assert (tmp_path_in_project / "artifacts" / "mlp_last.pt").exists()
    assert (tmp_path_in_project / "artifacts" / "metrics" / "mlp_history.csv").exists()
    hist = pd.read_csv(tmp_path_in_project / "artifacts" / "metrics" / "mlp_history.csv")
    assert list(hist.columns) == ["epoch", "train_loss", "train_macro_f1"]
    assert len(hist) == 1
