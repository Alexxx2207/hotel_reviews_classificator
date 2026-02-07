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
from src.constants import BATCH_SIZE, MLP_HIDDEN, PATHS, TEXT_COL, Paths
from src.features import load_vectorizer, transform
from src.hf_model import load_hf_model, predict_hf
from src.mlp import (MLP, iter_minibatches)


ReportType = dict[str, dict[str, float] | float] | str


def load_split(name: str, paths: Paths | None = None) -> tuple[list[str], np.ndarray]:
    """Loads the split data from the given name(train or test)."""

    p = paths or PATHS

    df = pd.read_csv(p.data_processed / f"{name}.csv")

    return df[TEXT_COL].tolist(), df["label"].to_numpy(dtype=int)


def plot_history(paths: Paths | None = None) -> None:
    """Plots the training history(loss and F1) of the MLP model."""

    p = paths or PATHS
    hist_path = p.metrics / "mlp_history.csv"

    if not hist_path.exists():
        return

    df = pd.read_csv(hist_path)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("MLP Train Loss")
    plt.savefig(
        p.plots / "mlp_train_loss.png",
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_macro_f1"])
    plt.xlabel("Epoch")
    plt.ylabel("Train macro F1")
    plt.title("MLP Train Macro-F1")
    plt.savefig(
        p.plots / "mlp_train_macro_f1.png",
        bbox_inches="tight",
    )
    plt.close()


def save_confusion_matrix(
    cm: np.ndarray, title: str, out_path: Path
) -> None:
    """Saves a confusion matrix to the given path."""

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def eval_baseline(
    reviews_test: np.ndarray,
    expected_labels: np.ndarray,
    paths: Paths | None = None,
) -> tuple[ReportType, np.ndarray]:
    """Evaluates the baseline model on the test data."""

    baseline = load_baseline(paths)
    predicted_labels = baseline.predict(reviews_test)

    report = classification_report(
        expected_labels, predicted_labels, output_dict=True
    )
    cm = confusion_matrix(expected_labels, predicted_labels)

    return report, cm


def eval_hf(
    reviews_test_txt: list[str],
    expected_labels: np.ndarray
) -> tuple[ReportType, np.ndarray]:
    """Evaluates the Hugging Face model on texts."""

    model, tokenizer, device = load_hf_model()

    predicted_labels, _ = predict_hf(
        model, tokenizer, device, reviews_test_txt
    )

    report = classification_report(
        expected_labels, predicted_labels, output_dict=True
    )
    cm = confusion_matrix(expected_labels, predicted_labels)
    
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
    expected_labels: np.ndarray,
    checkpoint_path: Path,
    batch_size: int = BATCH_SIZE,
) -> tuple[ReportType, np.ndarray]:
    """Evaluates the MLP model with the given checkpoint path."""

    model, device = load_mlp_checkpoint(checkpoint_path)
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for batched_reviews, _ in iter_minibatches(
            reviews_test, expected_labels, batch_size=batch_size, shuffle=False
        ):
            batched_reviews_t = torch.tensor(
                batched_reviews, dtype=torch.float32, device=device
            )
            logits = model(batched_reviews_t)
            prediction = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.append(prediction)
    predicted_labels = np.concatenate(predictions)

    report = classification_report(
        expected_labels, predicted_labels, output_dict=True
    )
    cm = confusion_matrix(expected_labels, predicted_labels)

    return report, cm


def main(paths: Paths | None = None) -> None:
    """Main function to run the evaluation."""
    p = paths or PATHS

    p.metrics.mkdir(parents=True, exist_ok=True)
    p.plots.mkdir(parents=True, exist_ok=True)

    reviews_test_txt, expected_labels = load_split("test", p)

    vec = load_vectorizer(p)
    reviews_test = transform(vec, reviews_test_txt)

    b_report, b_cm = eval_baseline(reviews_test, expected_labels, p)

    with open(p.metrics / "baseline_test_report.json", "w", encoding="utf-8") as f:
        json.dump(b_report, f, ensure_ascii=False, indent=2)

    save_confusion_matrix(
        b_cm,
        "Baseline Confusion Matrix (Test)",
        p.plots / "baseline_cm.png",
    )

    m_report, m_cm = eval_mlp(reviews_test, expected_labels, p.artifacts / "mlp_last.pt")

    with open(p.metrics / "mlp_test_report.json", "w", encoding="utf-8") as f:
        json.dump(m_report, f, ensure_ascii=False, indent=2)

    save_confusion_matrix(
        m_cm,
        "MLP Confusion Matrix (Test)",
        p.plots / "mlp_cm.png",
    )

    hf_report, hf_cm = eval_hf(reviews_test_txt, expected_labels)

    with open(p.metrics / "hf_test_report.json", "w", encoding="utf-8") as f:
        json.dump(hf_report, f, ensure_ascii=False, indent=2)

    save_confusion_matrix(
        hf_cm,
        "Hugging Face Confusion Matrix (Test)",
        p.plots / "hf_cm.png",
    )

    plot_history(p)


if __name__ == "__main__":
    main()
