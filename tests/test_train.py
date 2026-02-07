"""Tests for src.train."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import PATHS, Paths, TEXT_COL
from src.train import (
    ensure_directories_exist,
    load_split,
    main as train_main,
    train_mlp_only_train,
)


def test_train_mlp_only_train_returns_history(tmp_path_in_project: Path) -> None:
    """train_mlp_only_train returns history and saves checkpoint."""
    paths = Paths(project_root=tmp_path_in_project)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n, in_features = 50, 20
    x_train = rng.random((n, in_features)).astype(np.float32)
    y_train = (rng.random(n) > 0.5).astype(np.int64)

    history = train_mlp_only_train(
        x_train, y_train, in_features, paths=paths, epochs=1
    )
    assert isinstance(history, list)
    assert len(history) == 1
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

    train_main(paths=paths, epochs=1)

    assert (tmp_path_in_project / "artifacts" / "tfidf.joblib").exists()
    assert (tmp_path_in_project / "artifacts" / "baseline_logreg.joblib").exists()
    assert (tmp_path_in_project / "artifacts" / "mlp_last.pt").exists()
    assert (tmp_path_in_project / "artifacts" / "metrics" / "mlp_history.csv").exists()
    hist = pd.read_csv(tmp_path_in_project / "artifacts" / "metrics" / "mlp_history.csv")
    assert list(hist.columns) == ["epoch", "train_loss", "train_macro_f1"]
    assert len(hist) == 1
