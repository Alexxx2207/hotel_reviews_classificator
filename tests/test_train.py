"""Tests for src.train."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import EPOCHS, PATHS, Paths, TEXT_COL
from src.train import (
    main as train_main,
    train_mlp_only_train,
)


def test_train_mlp_only_train_returns_history(tmp_path_in_project: Path) -> None:
    """train_mlp_only_train returns history and saves checkpoint."""

    paths = Paths(project_root=tmp_path_in_project)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(100)
    n, in_features = 50, 20
    reviews_train = rng.random((n, in_features)).astype(float)
    labels_train = (rng.random(n) > 0.5).astype(int)

    history = train_mlp_only_train(
        reviews_train, labels_train, in_features, paths=paths
    )

    assert isinstance(history, list)
    assert len(history) == EPOCHS
    assert "epoch" in history[0]
    assert "train_loss" in history[0]
    assert "train_macro_f1" in history[0]
    assert (tmp_path_in_project / "artifacts" / "mlp_last.pt").exists()


def test_train_main(tmp_path_in_project: Path) -> None:
    """main() runs full pipeline and creates artifacts + history CSV."""

    paths = Paths(project_root=tmp_path_in_project)
    (tmp_path_in_project / "data").mkdir(parents=True)

    pd.DataFrame({
        TEXT_COL: ["good hotel"] * 10 + ["bad room"] * 10,
        "label": [1] * 10 + [0] * 10,
    }).to_csv(tmp_path_in_project / "data" / "train.csv", index=False)

    train_main(paths=paths)

    assert (tmp_path_in_project / "artifacts" / "tfidf.joblib").exists()
    assert (tmp_path_in_project / "artifacts" / "baseline_logreg.joblib").exists()
    assert (tmp_path_in_project / "artifacts" / "mlp_last.pt").exists()
    assert (tmp_path_in_project / "artifacts" / "metrics" / "mlp_history.csv").exists()

    history = pd.read_csv(tmp_path_in_project / "artifacts" / "metrics" / "mlp_history.csv")

    assert list(history.columns) == ["epoch", "train_loss", "train_macro_f1"]
    assert len(history) == EPOCHS
