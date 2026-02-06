"""
Evaluates the models and saves the results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.baseline import load_baseline
from src.constants import BATCH_SIZE, MLP_HIDDEN, PATHS, TEXT_COL
from src.features import load_vectorizer, transform
from src.mlp import (MLP, iter_minibatches)


def load_split(name: str) -> tuple[list[str], np.ndarray]:
    """Loads the split data from the given name(train or test)."""

    df = pd.read_csv(PATHS.data_processed / f"{name}.csv")
    return df[TEXT_COL].tolist(), df["label"].to_numpy(dtype=np.int64)


def plot_history() -> None:
    """Plots the training history of the MLP model."""

    hist_path = PATHS.metrics / "mlp_history.csv"
    if not hist_path.exists():
        return
    df = pd.read_csv(hist_path)
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("MLP Train Loss")
    plt.savefig(
        PATHS.plots / "mlp_train_loss.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    plt.figure()
    plt.plot(df["epoch"], df["train_macro_f1"])
    plt.xlabel("Epoch")
    plt.ylabel("Train macro F1")
    plt.title("MLP Train Macro-F1")
    plt.savefig(
        PATHS.plots / "mlp_train_macro_f1.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_confusion_matrix(
    cm: np.ndarray, title: str, out_path: Path
) -> None:
    """Saves a confusion matrix to the given path."""

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def eval_baseline(
    reviews_test: np.ndarray, y_test: np.ndarray
) -> tuple[dict, np.ndarray]:
    """Evaluates the baseline model on the test data."""

    baseline = load_baseline()
    y_pred = baseline.predict(reviews_test)
    report = classification_report(
        y_test, y_pred, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    return report, cm


def load_mlp_checkpoint(path: Path) -> tuple[MLP, torch.device]:
    """Loads the MLP model from the given checkpoint path."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    in_features = checkpoint["in_features"]
    model = MLP(
        in_features=in_features,
        hidden=MLP_HIDDEN,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


def eval_mlp(
    reviews_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: Path,
    batch_size: int = BATCH_SIZE,
) -> tuple[dict, np.ndarray]:
    """Evaluates the MLP model with the given checkpoint path."""

    model, device = load_mlp_checkpoint(checkpoint_path)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batched_reviews, _ in iter_minibatches(
            reviews_test, y_test, batch_size=batch_size, shuffle=False
        ):
            batched_reviews_t = torch.tensor(
                batched_reviews, dtype=torch.float32, device=device
            )
            logits = model(batched_reviews_t)
            pred = (
                torch.argmax(logits, dim=1).detach().cpu().numpy()
            )
            preds.append(pred)
    y_pred = np.concatenate(preds)
    report = classification_report(
        y_test, y_pred, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    return report, cm


def main() -> None:
    """Main function to run the evaluation."""

    PATHS.metrics.mkdir(parents=True, exist_ok=True)
    PATHS.plots.mkdir(parents=True, exist_ok=True)
    reviews_test_txt, y_test = load_split("test")
    vec = load_vectorizer()
    reviews_test = transform(vec, reviews_test_txt)
    b_report, b_cm = eval_baseline(reviews_test, y_test)
    with open(
        PATHS.metrics / "baseline_test_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(b_report, f, ensure_ascii=False, indent=2)
    save_confusion_matrix(
        b_cm,
        "Baseline Confusion Matrix (Test)",
        PATHS.plots / "baseline_cm.png",
    )
    ckpt = PATHS.artifacts / "mlp_last.pt"
    m_report, m_cm = eval_mlp(reviews_test, y_test, ckpt)
    with open(
        PATHS.metrics / "mlp_test_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(m_report, f, ensure_ascii=False, indent=2)
    save_confusion_matrix(
        m_cm,
        "MLP Confusion Matrix (Test)",
        PATHS.plots / "mlp_cm.png",
    )
    plot_history()
    print("Saved metrics to:", PATHS.metrics)
    print("Saved plots to:", PATHS.plots)


if __name__ == "__main__":
    main()
