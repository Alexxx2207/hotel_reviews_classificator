"""
Multilayer Perceptron model for the project.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np

import torch


class MLP(torch.nn.Module):
    """Single hidden-layer MLP: TF-IDF -> Hidden matrix -> ReLU -> Dropout -> Output matrix."""

    def __init__(
        self,
        in_features: int,
        hidden: int,
        dropout: float
    ) -> None:
        """Initializes the MLP."""

        super().__init__()
        self.hidden_layer = torch.nn.Linear(in_features, hidden)
        self.output_layer = torch.nn.Linear(hidden, 2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""

        hidden_activation = torch.nn.functional.relu(self.hidden_layer(input_batch))

        return self.output_layer(self.dropout(hidden_activation))


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
        batch_indices = sample_indices[batch_start : batch_start + batch_size]
        yield features[batch_indices], labels[batch_indices]
