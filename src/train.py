import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.constants import (
    PATHS, TEXT_COL,
    MLP_HIDDEN, MLP_DROPOUT, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY
)
from src.features import fit_vectorizer, save_vectorizer
from src.baseline import train_baseline, save_baseline
from src.mlp import SparseMLP, csr_to_torch_sparse, iter_minibatches

def load_split(name: str):
    df = pd.read_csv(PATHS.data_processed / f"{name}.csv")
    return df[TEXT_COL].tolist(), df["label"].to_numpy(dtype=np.int64)

def ensure_dirs():
    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    PATHS.metrics.mkdir(parents=True, exist_ok=True)
    PATHS.plots.mkdir(parents=True, exist_ok=True)

def train_mlp_only_train(X_train_csr, y_train, in_features: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SparseMLP(in_features=in_features, hidden=MLP_HIDDEN, dropout=MLP_DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = []
    last_path = PATHS.artifacts / "mlp_last.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for Xb_csr, yb in tqdm(
            iter_minibatches(X_train_csr, y_train, BATCH_SIZE, shuffle=True),
            desc=f"Epoch {epoch}/{EPOCHS}", leave=False
        ):
            Xb = csr_to_torch_sparse(Xb_csr, device)
            yb_t = torch.tensor(yb, dtype=torch.int64, device=device)

            opt.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = loss_fn(logits, yb_t)
            loss.backward()
            opt.step()

            epoch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            preds = []
            for Xb_csr, yb in iter_minibatches(X_train_csr, y_train, BATCH_SIZE, shuffle=False):
                Xb = csr_to_torch_sparse(Xb_csr, device)
                logits = model(Xb)
                pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
                preds.append(pred)
            y_pred = np.concatenate(preds)

        train_f1 = f1_score(y_train, y_pred, average="macro")
        train_loss = float(np.mean(epoch_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": float(train_f1),
        })

    torch.save({"model_state": model.state_dict(), "in_features": in_features}, last_path)
    return history, last_path

def main():
    ensure_dirs()

    # Load train
    X_train_txt, y_train = load_split("train")

    # TF-IDF (fit only on train)
    vec, X_train_csr = fit_vectorizer(X_train_txt)
    save_vectorizer(vec)

    # Baseline (train only)
    baseline = train_baseline(X_train_csr, y_train)
    save_baseline(baseline)

    # MLP (train only)
    in_features = X_train_csr.shape[1]
    history, last_path = train_mlp_only_train(X_train_csr, y_train, in_features)

    pd.DataFrame(history).to_csv(PATHS.metrics / "mlp_history.csv", index=False)

    # Save a small training summary
    summary = {
        "mlp_checkpoint": str(last_path),
        "last_epoch": history[-1],
    }
    with open(PATHS.metrics / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved artifacts to:", PATHS.artifacts)
    print("MLP checkpoint:", last_path)

if __name__ == "__main__":
    main()
