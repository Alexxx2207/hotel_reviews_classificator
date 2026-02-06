"""
Trains the models and saves the results.
"""

from __future__ import annotations

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
)
from src.features import fit_vectorizer, save_vectorizer
from src.mlp import MLP, iter_minibatches


def load_split(name: str) -> tuple[list[str], np.ndarray]:
    """Loads the split data from the given name(train or test)."""

    df = pd.read_csv(PATHS.data_processed / f"{name}.csv")
    return df[TEXT_COL].tolist(), df["label"].to_numpy(dtype=np.int64)


def ensure_directories_exist() -> None:
    """Ensures the important directories exist."""

    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    PATHS.metrics.mkdir(parents=True, exist_ok=True)
    PATHS.plots.mkdir(parents=True, exist_ok=True)

# pylint: disable=too-many-locals
def train_mlp_only_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    in_features: int,
) -> list[dict]:
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
    history: list[dict] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in tqdm(
            iter_minibatches(x_train, y_train, BATCH_SIZE, shuffle=True),
            desc=f"Epoch {epoch}/{EPOCHS}",
            leave=False,
        ):
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            yb_t = torch.tensor(yb, dtype=torch.int64, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb_t)
            loss = loss_fn(logits, yb_t)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            preds: list[np.ndarray] = []
            for xb, yb in iter_minibatches(
                x_train, y_train, BATCH_SIZE, shuffle=False
            ):
                xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
                logits = model(xb_t)
                pred = (
                    torch.argmax(logits, dim=1).detach().cpu().numpy()
                )
                preds.append(pred)
            y_pred = np.concatenate(preds)

        train_f1 = f1_score(y_train, y_pred, average="macro")
        train_loss = float(np.mean(epoch_losses))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": float(train_f1),
        })

    torch.save(
        {"model_state": model.state_dict(), "in_features": in_features},
        PATHS.artifacts / "mlp_last.pt",
    )
    return history


def main() -> None:
    """Main function to run the training."""

    ensure_directories_exist()
    x_train_txt, y_train = load_split("train")
    vec, x_train = fit_vectorizer(x_train_txt)
    save_vectorizer(vec)
    baseline = train_baseline(x_train, y_train)
    save_baseline(baseline)
    in_features = x_train.shape[1]
    history = train_mlp_only_train(x_train, y_train, in_features)
    pd.DataFrame(history).to_csv(
        PATHS.metrics / "mlp_history.csv", index=False
    )


if __name__ == "__main__":
    main()
