"""
Multilayer Perceptron model for the project.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np

import torch
from torch import nn


class MLP(nn.Module):
    """Single hidden-layer MLP: dense TF-IDF -> Linear -> ReLU -> Dropout -> Linear."""

    def __init__(
        self,
        in_features: int,
        hidden: int,
        dropout: float,
        num_classes: int = 2,
    ) -> None:
        """Initializes the MLP."""

        super().__init__()
        self.hidden_layer = nn.Linear(in_features, hidden)
        self.output_layer = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch: nn.Tensor) -> nn.Tensor:
        """Forward pass of the MLP."""
        hidden_activation = nn.functional.relu(self.hidden_layer(input_batch))
        hidden_dropped = self.dropout(hidden_activation)
        logits = self.output_layer(hidden_dropped)
        return logits


def iter_minibatches(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Iterates over the data in batches."""

    num_samples = features.shape[0]
    sample_indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(sample_indices)
    for batch_start in range(0, num_samples, batch_size):
        batch_indices = sample_indices[
            batch_start : batch_start + batch_size
        ]
        yield features[batch_indices], labels[batch_indices]
