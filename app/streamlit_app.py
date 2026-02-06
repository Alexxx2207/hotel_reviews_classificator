from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import load_baseline
from src.constants import MLP_HIDDEN, PATHS
from src.features import load_vectorizer, transform
from src.mlp import MLP, _remap_state_dict_for_compat as remap_mlp_state_dict

# Page configuration
st.set_page_config(
    page_title="Hotel Review Classifier",
    page_icon="🏨",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #0e1117;
        border: 2px solid #28a745;
    }
    .negative {
        background-color: #0e1117;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models() -> tuple[
    TfidfVectorizer,
    Optional[LogisticRegression],
    Optional[MLP],
    Optional[torch.device],
]:
    """Load ML models and vectorizer with caching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        vectorizer = load_vectorizer()
    except FileNotFoundError:
        st.error("Vectorizer not found. Please run the training script first.")
        st.stop()

    try:
        baseline_model = load_baseline()
    except FileNotFoundError:
        st.warning(
            "Baseline model not found. Only MLP predictions will be available."
        )
        baseline_model = None

    mlp_model: Optional[MLP] = None
    mlp_device: Optional[torch.device] = None
    ckpt_path = PATHS.artifacts / "mlp_last.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            in_features = ckpt["in_features"]
            mlp_model = MLP(
                in_features=in_features,
                hidden=MLP_HIDDEN,
                dropout=0.0,
            ).to(device)
            mlp_model.load_state_dict(remap_mlp_state_dict(ckpt["model_state"]))
            mlp_model.eval()
            mlp_device = device
        except Exception as e:
            st.warning(f"Could not load MLP model: {e}")
    else:
        st.warning(
            "MLP model checkpoint not found. "
            "Only baseline predictions will be available."
        )
    return vectorizer, baseline_model, mlp_model, mlp_device


def predict_baseline(
    vectorizer: TfidfVectorizer,
    model: Optional[LogisticRegression],
    text: str,
) -> tuple[Optional[int], Optional[np.ndarray]]:
    """Predict using baseline logistic regression model."""
    if model is None:
        return None, None
    X = transform(vectorizer, [text])
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    return prediction, probabilities


def predict_mlp(
    vectorizer: TfidfVectorizer,
    model: Optional[MLP],
    device: Optional[torch.device],
    text: str,
) -> tuple[Optional[int], Optional[np.ndarray]]:
    """Predict using MLP model."""
    if model is None or device is None:
        return None, None
    X = transform(vectorizer, [text])
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(X_t)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = int(np.argmax(probabilities))
    return prediction, probabilities


def get_label_name(label: int) -> str:
    """Convert numeric label to readable name."""
    return "Positive" if label == 1 else "Negative"


def main() -> None:
    st.markdown(
        '<h1 class="main-header">Hotel Review Classifier</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    with st.spinner("Loading models..."):
        vectorizer, baseline_model, mlp_model, mlp_device = load_models()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Enter Hotel Review")
        review_text = st.text_area(
            "Review Text",
            height=200,
            placeholder=(
                "Enter your hotel review here...\n\n"
                "Example: 'The hotel was amazing! Great service, "
                "clean rooms, and excellent location. Highly recommend!'"
            ),
            help="Type or paste a hotel review to classify it as positive or negative.",
        )
        st.subheader("Try Example Reviews")
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            if st.button("Example Positive Review"):
                st.session_state.example_review = (
                    "This hotel exceeded all my expectations! "
                    "The staff was incredibly friendly and helpful, "
                    "the rooms were spotless and comfortable, "
                    "and the breakfast buffet was outstanding. "
                    "The location is perfect - close to all major attractions. "
                    "I will definitely stay here again!"
                )
        with example_col2:
            if st.button("Example Negative Review"):
                st.session_state.example_review = (
                    "Very disappointed with my stay. The room was dirty, "
                    "the bed was uncomfortable, and the staff was unhelpful. "
                    "The WiFi didn't work, and the noise from the street "
                    "kept me awake all night. Would not recommend this hotel."
                )
        if "example_review" in st.session_state:
            review_text = st.text_area(
                "Review Text",
                value=st.session_state.example_review,
                height=200,
            )
            del st.session_state.example_review

    with col2:
        st.header("Classification")
        if st.button("Classify Review", type="primary", use_container_width=True):
            if not review_text.strip():
                st.warning("Please enter a review text before classifying.")
            else:
                baseline_pred, baseline_probs = predict_baseline(
                    vectorizer, baseline_model, review_text
                )
                mlp_pred, mlp_probs = predict_mlp(
                    vectorizer, mlp_model, mlp_device, review_text
                )
                st.markdown("---")
                if baseline_pred is not None and baseline_probs is not None:
                    st.subheader("Baseline Model (Logistic Regression)")
                    label_name = get_label_name(baseline_pred)
                    confidence = baseline_probs[baseline_pred] * 100
                    css_class = (
                        "positive" if baseline_pred == 1 else "negative"
                    )
                    st.markdown(
                        f'<div class="prediction-box {css_class}">'
                        f"<h3>{label_name} Review</h3>"
                        f"<p><strong>Confidence:</strong> {confidence:.2f}%</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                if mlp_pred is not None and mlp_probs is not None:
                    st.subheader("MLP Model (Neural Network)")
                    label_name = get_label_name(mlp_pred)
                    confidence = mlp_probs[mlp_pred] * 100
                    css_class = "positive" if mlp_pred == 1 else "negative"
                    st.markdown(
                        f'<div class="prediction-box {css_class}">'
                        f"<h3>{label_name} Review</h3>"
                        f"<p><strong>Confidence:</strong> {confidence:.2f}%</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                if baseline_pred is not None and mlp_pred is not None:
                    st.markdown("---")
                    if baseline_pred == mlp_pred:
                        st.success(
                            f"Both models agree: "
                            f"**{get_label_name(baseline_pred)}** review"
                        )
                    else:
                        st.warning(
                            f"Models disagree: Baseline predicts "
                            f"**{get_label_name(baseline_pred)}**, "
                            f"MLP predicts **{get_label_name(mlp_pred)}**"
                        )


if __name__ == "__main__":
    main()
