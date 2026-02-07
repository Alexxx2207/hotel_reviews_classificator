"""Tests for src.mlp."""

import numpy as np
import torch

from src.mlp import MLP, iter_minibatches


def test_mlp_forward_shape() -> None:
    """MLP forward returns logits with shape (batch, num_classes)."""

    model = MLP(in_features=10, hidden=8, dropout=0.0, num_classes=2)
    x = torch.randn(4, 10)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_iter_minibatches_yields_batches() -> None:
    """iter_minibatches yields (features_batch, labels_batch) tuples."""

    X = np.arange(10).reshape(10, 1).astype(np.float64)
    y = np.arange(10)
    batches = list(iter_minibatches(X, y, batch_size=3, shuffle=False))
    assert len(batches) == 4
    assert batches[0][0].shape == (3, 1)
    assert batches[0][1].shape == (3,)
    assert batches[-1][0].shape[0] == 1


def test_iter_minibatches_empty_batch_size_larger_than_data() -> None:
    """When batch_size > n, we get one batch with all samples."""
    
    X = np.ones((3, 2))
    y = np.array([0, 1, 0])
    batches = list(iter_minibatches(X, y, batch_size=10, shuffle=False))
    assert len(batches) == 1
    assert batches[0][0].shape == (3, 2)
    assert batches[0][1].shape == (3,)


def test_mlp_num_classes_three() -> None:
    """MLP with num_classes=3 returns logits shape (batch, 3)."""
    model = MLP(in_features=5, hidden=4, dropout=0.0, num_classes=3)
    x = torch.randn(2, 5)
    logits = model(x)
    assert logits.shape == (2, 3)


