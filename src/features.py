import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from src.constants import PATHS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE

VECTORIZER_PATH = PATHS.artifacts / "tfidf.joblib"

def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        lowercase=True,
        strip_accents=None
    )

def fit_vectorizer(texts):
    vec = build_vectorizer()
    X = vec.fit_transform(texts)
    return vec, X

def transform(vec: TfidfVectorizer, texts):
    X = vec.transform(texts)
    return X

def save_vectorizer(vec: TfidfVectorizer):
    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, VECTORIZER_PATH)

def load_vectorizer() -> TfidfVectorizer:
    return joblib.load(VECTORIZER_PATH)
