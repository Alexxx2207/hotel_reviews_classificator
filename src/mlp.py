from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

@dataclass
class Batch:
    X: torch.Tensor
    y: torch.Tensor

def csr_to_torch_sparse(csr: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
    csr = csr.tocoo()
    indices = torch.tensor(np.vstack((csr.row, csr.col)), dtype=torch.int64, device=device)
    values = torch.tensor(csr.data, dtype=torch.float32, device=device)
    shape = torch.Size(csr.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

class SparseMLP(nn.Module):
    """
    1-скрит слой MLP:
      sparse TF-IDF -> Linear (через sparse.mm) -> ReLU -> Dropout -> Linear
    """
    def __init__(self, in_features: int, hidden: int, dropout: float, num_classes: int = 2) -> None:
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(in_features, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.fc2 = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W1)

    def forward(self, x_sparse: torch.Tensor) -> torch.Tensor:
        # x_sparse: [B, in_features] sparse
        h = torch.sparse.mm(x_sparse, self.W1) + self.b1  # [B, hidden]
        h = F.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits

def iter_minibatches(X_csr: sparse.csr_matrix, y: np.ndarray, batch_size: int, shuffle: bool = True):
    n = X_csr.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X_csr[batch_idx], y[batch_idx]
