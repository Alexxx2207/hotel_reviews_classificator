"""Tests for src.evaluate."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

from src.baseline import save_baseline, train_baseline
from src.constants import BATCH_SIZE, MLP_HIDDEN, Paths, TEXT_COL
from src.evaluate import (
    eval_baseline,
    eval_mlp,
    load_split,
    load_mlp_checkpoint,
    main as evaluate_main,
    plot_history,
    save_confusion_matrix,
)
from src.features import fit_vectorizer, save_vectorizer, transform
from src.mlp import MLP


def test_load_split_evaluate(tmp_path_in_project: Path) -> None:
    """load_split in evaluate loads CSV and returns texts and labels."""
    paths = Paths(project_root=tmp_path_in_project)
    (tmp_path_in_project / "data").mkdir(parents=True)
    pd.DataFrame({
        TEXT_COL: ["x", "y"],
        "label": [1, 0],
    }).to_csv(tmp_path_in_project / "data" / "test.csv", index=False)

    texts, labels = load_split("test", paths)
    assert texts == ["x", "y"]
    np.testing.assert_array_equal(labels, np.array([1, 0]))


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


def test_save_confusion_matrix_writes_file(tmp_path_in_project: Path) -> None:
    """save_confusion_matrix writes a PNG file."""
    (tmp_path_in_project / "artifacts" / "plots").mkdir(parents=True)
    cm = np.array([[5, 2], [1, 7]])
    out_path = tmp_path_in_project / "artifacts" / "plots" / "cm.png"
    save_confusion_matrix(cm, "Test CM", out_path)
    assert out_path.exists()


def test_eval_baseline_and_eval_mlp(tmp_path_in_project: Path) -> None:
    """eval_baseline and eval_mlp work when models are saved in tmp_path_in_project."""
    paths = Paths(project_root=tmp_path_in_project)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    (tmp_path_in_project / "data").mkdir(parents=True)

    texts = ["good hotel"] * 5 + ["bad room"] * 5
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    vec, X = fit_vectorizer(texts)
    save_vectorizer(vec, paths)
    clf = train_baseline(X, labels)
    save_baseline(clf, paths)

    model = MLP(in_features=X.shape[1], hidden=MLP_HIDDEN, dropout=0.0)
    torch.save(
        {"model_state": model.state_dict(), "in_features": X.shape[1]},
        paths.artifacts / "mlp_last.pt",
    )

    X_test = transform(vec, ["good", "bad"])
    y_test = np.array([1, 0])
    b_report, b_cm = eval_baseline(X_test, y_test, paths)
    assert isinstance(b_report, dict)
    assert b_cm.shape == (2, 2)

    m_report, m_cm = eval_mlp(
        X_test, y_test,
        paths.artifacts / "mlp_last.pt",
        batch_size=BATCH_SIZE,
    )
    assert isinstance(m_report, dict)
    assert m_cm.shape == (2, 2)


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

    assert (paths.metrics / "baseline_test_report.json").exists()
    assert (paths.metrics / "mlp_test_report.json").exists()
    assert (paths.plots / "baseline_cm.png").exists()
    assert (paths.plots / "mlp_cm.png").exists()


def test_load_mlp_checkpoint(tmp_path_in_project: Path) -> None:
    """load_mlp_checkpoint returns model and device."""
    (tmp_path_in_project / "artifacts").mkdir(parents=True)
    model = MLP(
        in_features=4, hidden=MLP_HIDDEN, dropout=0.0
    )
    ckpt_path = tmp_path_in_project / "artifacts" / "mlp_last.pt"
    torch.save(
        {"model_state": model.state_dict(), "in_features": 4},
        ckpt_path,
    )

    loaded_model, _ = load_mlp_checkpoint(ckpt_path)
    assert isinstance(loaded_model, MLP)
    assert loaded_model.hidden_layer.weight.shape == (MLP_HIDDEN, 4)
    x = torch.randn(1, 4)
    with torch.no_grad():
        _ = loaded_model(x)
