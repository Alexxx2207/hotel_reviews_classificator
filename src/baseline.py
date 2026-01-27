import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.constants import PATHS

BASELINE_PATH = PATHS.artifacts / "baseline_logreg.joblib"

def train_baseline(x_train, y_train):
    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    clf.fit(x_train, y_train)
    return clf

def evaluate_baseline(clf, x, y):
    y_pred = clf.predict(x)
    return classification_report(y, y_pred, output_dict=True)

def save_baseline(clf):
    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, BASELINE_PATH)

def load_baseline():
    return joblib.load(BASELINE_PATH)
