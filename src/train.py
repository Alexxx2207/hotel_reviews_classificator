"""
Trains the models and saves the results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.baseline import save_baseline, train_baseline
from src.constants import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    MLP_DROPOUT,
    MLP_HIDDEN,
    PATHS,
    TEXT_COL,
    WEIGHT_DECAY,
    Paths,
)
from src.features import fit_vectorizer, save_vectorizer
from src.mlp import MLP, iter_minibatches


def load_split(name: str, paths: Paths | None = None) -> tuple[list[str], np.ndarray]:
    """Loads the split data from the given name(train or test)."""

    p = paths or PATHS

    df = pd.read_csv(p.data_processed / f"{name}.csv")

    return df[TEXT_COL].tolist(), df["label"].to_numpy(dtype=int)


def ensure_directories_exist(paths: Paths | None = None) -> None:
    """Ensures the important directories exist."""

    p = paths or PATHS

    p.artifacts.mkdir(parents=True, exist_ok=True)
    p.metrics.mkdir(parents=True, exist_ok=True)
    p.plots.mkdir(parents=True, exist_ok=True)

# pylint: disable=too-many-locals
def train_mlp_only_train(
    reviews_training: np.ndarray,
    label_training: np.ndarray,
    in_features: int,
    paths: Paths | None = None,
) -> list[dict[str, Any]]:
    """Trains the MLP model only on the training data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(
        in_features=in_features,
        hidden=MLP_HIDDEN,
        dropout=MLP_DROPOUT,
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    history: list[dict[str, Any]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses: list[float] = []
        for reviews_batched, labels_batched in tqdm(
            iter_minibatches(reviews_training, label_training, BATCH_SIZE, shuffle=True),
            desc=f"Epoch {epoch}/{EPOCHS}",
            leave=False,
        ):
            reviews_batched_t = torch.tensor(reviews_batched, dtype=torch.float32, device=device)
            labels_batched_t = torch.tensor(labels_batched, dtype=torch.int64, device=device)

            opt.zero_grad(set_to_none=True)
            logits = model(reviews_batched_t)
            loss = loss_fn(logits, labels_batched_t)
            loss.backward()
            opt.step()

            epoch_losses.append(loss.cpu().item())

        model.eval()
        with torch.no_grad():
            predictions: list[np.ndarray] = []

            for reviews_batched, labels_batched in iter_minibatches(
                reviews_training, label_training, BATCH_SIZE, shuffle=False
            ):
                reviews_batched_t = torch.tensor(
                    reviews_batched,
                    dtype=torch.float32,
                    device=device,
                )
                logits = model(reviews_batched_t)
                predictions.append(torch.argmax(logits, dim=1).cpu().numpy())

            label_prediction = np.concatenate(predictions)

        train_f1 = f1_score(label_training, label_prediction, average="macro")
        train_loss = np.mean(epoch_losses)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
        })

    p = paths or PATHS

    torch.save(
        {"model_state": model.state_dict(), "in_features": in_features},
        p.artifacts / "mlp_last.pt",
    )

    return history


def main(paths: Paths | None = None) -> None:
    """Main function to run the training."""
    p = paths or PATHS

    ensure_directories_exist(p)

    x_train_txt, label_training = load_split("train", p)

    vec, reviews_training = fit_vectorizer(x_train_txt)
    save_vectorizer(vec, p)

    baseline = train_baseline(reviews_training, label_training)
    save_baseline(baseline, p)

    history = train_mlp_only_train(
        reviews_training, label_training, reviews_training.shape[1], paths=p
    )

    pd.DataFrame(history).to_csv(p.metrics / "mlp_history.csv", index=False)


if __name__ == "__main__":  # pragma: no cover
    main()
