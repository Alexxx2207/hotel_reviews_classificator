"""Tests for src.evaluate."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import baseline as baseline_mod
from src import constants
from src import features as features_mod
from src.baseline import save_baseline, train_baseline
from src.constants import BATCH_SIZE, MLP_HIDDEN, PATHS, TEXT_COL
from src.evaluate import (
    eval_baseline,
    eval_mlp,
    load_split,
    load_mlp_checkpoint,
    plot_history,
    save_confusion_matrix,
)
from src.features import fit_vectorizer, save_vectorizer, transform
from src.mlp import MLP

def test_load_split_evaluate(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_split in evaluate loads CSV and returns texts and labels."""

    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    monkeypatch.setattr("src.evaluate.PATHS", constants.PATHS)
    (tmp_path_in_project / "data").mkdir(parents=True)
    pd.DataFrame({
        TEXT_COL: ["x", "y"],
        "label": [1, 0],
    }).to_csv(tmp_path_in_project / "data" / "test.csv", index=False)

    texts, labels = load_split("test")
    assert texts == ["x", "y"]
    np.testing.assert_array_equal(labels, np.array([1, 0]))


def test_plot_history_returns_early_if_no_file(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plot_history returns without error when mlp_history.csv is missing."""

    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    monkeypatch.setattr("src.evaluate.PATHS", constants.PATHS)
    (tmp_path_in_project / "artifacts" / "metrics").mkdir(parents=True)
    plot_history()


def test_plot_history_creates_plots(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plot_history creates loss and f1 plots when history exists."""

    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    monkeypatch.setattr("src.evaluate.PATHS", constants.PATHS)
    (tmp_path_in_project / "artifacts" / "metrics").mkdir(parents=True)
    (tmp_path_in_project / "artifacts" / "plots").mkdir(parents=True)
    pd.DataFrame({
        "epoch": [1, 2],
        "train_loss": [0.5, 0.3],
        "train_macro_f1": [0.7, 0.8],
    }).to_csv(tmp_path_in_project / "artifacts" / "metrics" / "mlp_history.csv", index=False)

    plot_history()
    assert (tmp_path_in_project / "artifacts" / "plots" / "mlp_train_loss.png").exists()
    assert (tmp_path_in_project / "artifacts" / "plots" / "mlp_train_macro_f1.png").exists()


def test_save_confusion_matrix_writes_file(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_confusion_matrix writes a PNG file."""
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    monkeypatch.setattr("src.evaluate.PATHS", constants.PATHS)
    (tmp_path_in_project / "artifacts" / "plots").mkdir(parents=True)
    cm = np.array([[5, 2], [1, 7]])
    out_path = tmp_path_in_project / "artifacts" / "plots" / "cm.png"
    save_confusion_matrix(cm, "Test CM", out_path)
    assert out_path.exists()


def test_eval_baseline_and_eval_mlp(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """eval_baseline and eval_mlp work when models are saved in tmp_path_in_project."""


    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    monkeypatch.setattr("src.evaluate.PATHS", constants.PATHS)
    monkeypatch.setattr(baseline_mod, "PATHS", constants.PATHS)
    monkeypatch.setattr(
        baseline_mod, "BASELINE_PATH", tmp_path_in_project / "artifacts" / "baseline_logreg.joblib"
    )
    monkeypatch.setattr(features_mod, "PATHS", constants.PATHS)
    monkeypatch.setattr(
        features_mod,
        "VECTORIZER_PATH",
        tmp_path_in_project / "artifacts" / "tfidf.joblib",
    )

    (tmp_path_in_project / "artifacts").mkdir(parents=True)
    (tmp_path_in_project / "data").mkdir(parents=True)

    # Create minimal train data and fit vectorizer + baseline
    texts = ["good hotel"] * 5 + ["bad room"] * 5
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    vec, X = fit_vectorizer(texts)
    save_vectorizer(vec)
    clf = train_baseline(X, labels)
    save_baseline(clf)

    # Save MLP checkpoint
    model = MLP(in_features=X.shape[1], hidden=MLP_HIDDEN, dropout=0.0)
    torch.save(
        {"model_state": model.state_dict(), "in_features": X.shape[1]},
        tmp_path_in_project / "artifacts" / "mlp_last.pt",
    )

    # Eval baseline
    X_test = transform(vec, ["good", "bad"])
    y_test = np.array([1, 0])
    b_report, b_cm = eval_baseline(X_test, y_test)
    assert isinstance(b_report, dict)
    assert b_cm.shape == (2, 2)

    # Eval MLP
    m_report, m_cm = eval_mlp(
        X_test, y_test,
        tmp_path_in_project / "artifacts" / "mlp_last.pt",
        batch_size=BATCH_SIZE,
    )
    assert isinstance(m_report, dict)
    assert m_cm.shape == (2, 2)


def test_evaluate_main(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() runs full evaluation and writes reports and plots."""
    import matplotlib
    matplotlib.use("Agg")
    paths = constants.Paths(project_root=tmp_path_in_project)
    monkeypatch.setattr(constants, "PATHS", paths)
    monkeypatch.setattr("src.evaluate.PATHS", paths)
    monkeypatch.setattr(baseline_mod, "PATHS", paths)
    monkeypatch.setattr(
        baseline_mod, "BASELINE_PATH", tmp_path_in_project / "artifacts" / "baseline_logreg.joblib"
    )
    monkeypatch.setattr(features_mod, "PATHS", paths)
    monkeypatch.setattr(
        features_mod, "VECTORIZER_PATH", tmp_path_in_project / "artifacts" / "tfidf.joblib"
    )

    (tmp_path_in_project / "data").mkdir(parents=True)
    (tmp_path_in_project / "artifacts").mkdir(parents=True)
    (tmp_path_in_project / "artifacts" / "metrics").mkdir(parents=True)
    (tmp_path_in_project / "artifacts" / "plots").mkdir(parents=True)

    pd.DataFrame({
        TEXT_COL: ["great", "terrible"],
        "label": [1, 0],
    }).to_csv(tmp_path_in_project / "data" / "test.csv", index=False)
    texts = ["great stay"] * 5 + ["awful"] * 5
    labels = np.array([1] * 5 + [0] * 5)
    vec, X = fit_vectorizer(texts)
    save_vectorizer(vec)
    clf = train_baseline(X, labels)
    save_baseline(clf)
    model = MLP(in_features=X.shape[1], hidden=MLP_HIDDEN, dropout=0.0)
    torch.save(
        {"model_state": model.state_dict(), "in_features": X.shape[1]},
        tmp_path_in_project / "artifacts" / "mlp_last.pt",
    )

    from src.evaluate import main as evaluate_main
    evaluate_main()

    assert (tmp_path_in_project / "artifacts" / "metrics" / "baseline_test_report.json").exists()
    assert (tmp_path_in_project / "artifacts" / "metrics" / "mlp_test_report.json").exists()
    assert (tmp_path_in_project / "artifacts" / "plots" / "baseline_cm.png").exists()
    assert (tmp_path_in_project / "artifacts" / "plots" / "mlp_cm.png").exists()


def test_load_mlp_checkpoint(
    tmp_path_in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_mlp_checkpoint returns model and device."""
    from src import constants

    monkeypatch.setattr(constants, "PATHS", constants.Paths(project_root=tmp_path_in_project))
    monkeypatch.setattr("src.evaluate.PATHS", constants.PATHS)
    (tmp_path_in_project / "artifacts").mkdir(parents=True)
    model = MLP(
        in_features=4, hidden=MLP_HIDDEN, dropout=0.0
    )
    ckpt_path = tmp_path_in_project / "artifacts" / "mlp_last.pt"
    torch.save(
        {"model_state": model.state_dict(), "in_features": 4},
        ckpt_path,
    )

    loaded_model, device = load_mlp_checkpoint(ckpt_path)
    assert isinstance(loaded_model, MLP)
    assert loaded_model.hidden_weight.shape == (4, MLP_HIDDEN)
    x = torch.randn(1, 4)
    with torch.no_grad():
        _ = loaded_model(x)
