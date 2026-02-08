"""
API to use the Hugging Face pre-trained model for hotel review sentiment classification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.constants import BATCH_SIZE, HF_MODEL_NAME


def load_hf_model(
    model_name: str = HF_MODEL_NAME,
):
    """Load the Hugging Face model and tokenizer from the Hub."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_hf(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    reviews: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict sentiment (0=negative, 1=positive) for a list of reviews."""

    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []

    for i in range(0, len(reviews), BATCH_SIZE):
        inputs = tokenizer(
            reviews[i : i + BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.cpu()
        probabilities = torch.softmax(logits, dim=1).numpy()
        predictions = np.argmax(probabilities, axis=1)

        all_predictions.extend(predictions.tolist())
        all_probabilities.append(probabilities)

    predictions = np.array(all_predictions, dtype=np.int64)
    probabilities = np.vstack(all_probabilities)

    return predictions, probabilities
