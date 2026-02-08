"""Tests for src.evaluate."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

from src.baseline import save_baseline, train_baseline
from src.constants import MLP_HIDDEN, Paths, TEXT_COL
from src.evaluate import main as evaluate_main, plot_history
from src.features import fit_vectorizer, save_vectorizer
from src.mlp import MLP


def test_plot_history_returns_early_if_no_file(tmp_path_in_project: Path) -> None:
    """plot_history returns without error when mlp_history.csv is missing."""

    paths = Paths(project_root=tmp_path_in_project)
    paths.metrics.mkdir(parents=True, exist_ok=True)

    plot_history(paths)


def test_plot_history_creates_plots(tmp_path_in_project: Path) -> None:
    """plot_history creates loss and f1 plots when history exists."""

    paths = Paths(project_root=tmp_path_in_project)
    paths.metrics.mkdir(parents=True, exist_ok=True)
    paths.plots.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "epoch": [1, 2],
        "train_loss": [0.5, 0.3],
        "train_macro_f1": [0.7, 0.8],
    }).to_csv(paths.metrics / "mlp_history.csv", index=False)

    plot_history(paths)

    assert (paths.plots / "mlp_train_loss.png").exists()
    assert (paths.plots / "mlp_train_macro_f1.png").exists()


def test_evaluate_main(tmp_path_in_project: Path) -> None:
    """main() runs full evaluation and writes reports and plots."""

    paths = Paths(project_root=tmp_path_in_project)
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    paths.metrics.mkdir(parents=True, exist_ok=True)
    paths.plots.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        TEXT_COL: ["great", "terrible"],
        "label": [1, 0],
    }).to_csv(paths.data_processed / "test.csv", index=False)

    texts = ["great stay"] * 5 + ["awful"] * 5
    labels = np.array([1] * 5 + [0] * 5)
    
    vec, X = fit_vectorizer(texts)
    save_vectorizer(vec, paths)

    clf = train_baseline(X, labels)
    save_baseline(clf, paths)

    model = MLP(in_features=X.shape[1], hidden=MLP_HIDDEN, dropout=0.0)
    torch.save(
        {"model_state": model.state_dict(), "in_features": X.shape[1]},
        paths.artifacts / "mlp_last.pt",
    )

    evaluate_main(paths)

    assert (paths.metrics / "baseline_test.json").exists()
    assert (paths.metrics / "mlp_test.json").exists()
    assert (paths.plots / "baseline_test.png").exists()
    assert (paths.plots / "mlp_test.png").exists()
