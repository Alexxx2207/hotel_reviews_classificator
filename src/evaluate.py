import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.constants import PATHS, TEXT_COL, MLP_HIDDEN, BATCH_SIZE
from src.features import load_vectorizer, transform
from src.baseline import load_baseline
from src.mlp import SparseMLP, csr_to_torch_sparse, iter_minibatches

def load_split(name: str):
    df = pd.read_csv(PATHS.data_processed / f"{name}.csv")
    return df[TEXT_COL].tolist(), df["label"].to_numpy(dtype=np.int64)

def plot_history():
    hist_path = PATHS.metrics / "mlp_history.csv"
    if not hist_path.exists():
        return
    df = pd.read_csv(hist_path)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("MLP Train Loss")
    plt.savefig(PATHS.plots / "mlp_train_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_macro_f1"])
    plt.xlabel("Epoch")
    plt.ylabel("Train macro F1")
    plt.title("MLP Train Macro-F1")
    plt.savefig(PATHS.plots / "mlp_train_macro_f1.png", dpi=200, bbox_inches="tight")
    plt.close()

def save_cm(cm, title, out_path):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def eval_baseline(X_test_csr, y_test):
    baseline = load_baseline()
    y_pred = baseline.predict(X_test_csr)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm

def load_mlp_checkpoint(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    in_features = ckpt["in_features"]
    model = SparseMLP(in_features=in_features, hidden=MLP_HIDDEN, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device

def eval_mlp(X_test_csr, y_test, ckpt_path, batch_size=BATCH_SIZE):
    model, device = load_mlp_checkpoint(ckpt_path)

    preds = []
    with torch.no_grad():
        for Xb_csr, yb in iter_minibatches(X_test_csr, y_test, batch_size=batch_size, shuffle=False):
            Xb = csr_to_torch_sparse(Xb_csr, device)
            logits = model(Xb)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.append(pred)

    y_pred = np.concatenate(preds)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm

def main():
    PATHS.metrics.mkdir(parents=True, exist_ok=True)
    PATHS.plots.mkdir(parents=True, exist_ok=True)

    X_test_txt, y_test = load_split("test")
    vec = load_vectorizer()
    X_test_csr = transform(vec, X_test_txt)

    b_report, b_cm = eval_baseline(X_test_csr, y_test)
    with open(PATHS.metrics / "baseline_test_report.json", "w", encoding="utf-8") as f:
        json.dump(b_report, f, ensure_ascii=False, indent=2)
    save_cm(b_cm, "Baseline Confusion Matrix (Test)", PATHS.plots / "baseline_cm.png")

    ckpt = PATHS.artifacts / "mlp_last.pt"
    m_report, m_cm = eval_mlp(X_test_csr, y_test, ckpt)
    with open(PATHS.metrics / "mlp_test_report.json", "w", encoding="utf-8") as f:
        json.dump(m_report, f, ensure_ascii=False, indent=2)
    save_cm(m_cm, "MLP Confusion Matrix (Test)", PATHS.plots / "mlp_cm.png")

    plot_history()

    print("Saved metrics to:", PATHS.metrics)
    print("Saved plots to:", PATHS.plots)

if __name__ == "__main__":
    main()
