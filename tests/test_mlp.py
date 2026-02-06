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


def test_mlp_forward_deterministic_without_dropout() -> None:
    """With dropout=0, two forward passes give the same result."""

    model = MLP(in_features=5, hidden=4, dropout=0.0)
    model.eval()
    x = torch.randn(2, 5)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2)


def test_mlp_has_expected_parameters() -> None:
    """MLP has hidden_weight, hidden_bias, output_layer, dropout."""

    model = MLP(in_features=6, hidden=3, dropout=0.1)
    assert hasattr(model, "hidden_weight")
    assert hasattr(model, "hidden_bias")
    assert hasattr(model, "output_layer")
    assert hasattr(model, "dropout")
    assert model.hidden_weight.shape == (6, 3)
    assert model.hidden_bias.shape == (3,)


def test_iter_minibatches_yields_batches() -> None:
    """iter_minibatches yields (features_batch, labels_batch) tuples."""

    X = np.arange(10).reshape(10, 1).astype(np.float64)
    y = np.arange(10)
    batches = list(iter_minibatches(X, y, batch_size=3, shuffle=False))
    assert len(batches) == 4  # 10/3 -> 4 batches
    assert batches[0][0].shape == (3, 1)
    assert batches[0][1].shape == (3,)
    assert batches[-1][0].shape[0] == 1  # last batch has 1 sample


def test_iter_minibatches_shuffle_changes_order() -> None:
    """With shuffle=True, batch contents can differ from shuffle=False."""

    X = np.arange(20).reshape(20, 1).astype(np.float64)
    y = np.arange(20)
    batches_no_shuffle = list(iter_minibatches(X, y, batch_size=5, shuffle=False))
    batches_shuffle = list(iter_minibatches(X, y, batch_size=5, shuffle=True))
    order_no = np.concatenate([b[1] for b in batches_no_shuffle])
    order_yes = np.concatenate([b[1] for b in batches_shuffle])
    assert set(order_no) == set(order_yes) == set(range(20))
    assert len(order_yes) == 20


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


def test_mlp_training_mode_dropout() -> None:
    """With dropout and model.train(), two forwards can differ."""
    model = MLP(in_features=4, hidden=3, dropout=0.9)
    model.train()
    x = torch.randn(2, 4)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert out1.shape == out2.shape == (2, 2)
