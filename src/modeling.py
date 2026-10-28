"""Pipeline construction, model zoo and hyper-parameter grids."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .transformers import SentinelToNaN

# Simplicity order is used as the tie-breaker during model selection (simpler wins ties).
SIMPLICITY_ORDER = ["Logistic Regression", "KNN", "SVM (RBF)", "Random Forest",
                    "Gradient Boosting", "Soft-Voting Ensemble"]


def build_preprocessor(features) -> ColumnTransformer:
    """Numeric -> median impute + standardise; binary -> mode impute; category -> mode impute + one-hot."""
    num = [f.name for f in features if f.kind in ("numeric", "integer")]
    binary = [f.name for f in features if f.kind == "binary"]
    cat = [f.name for f in features if f.kind == "category"]
    parts = []
    if num:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), num))
    if binary:
        parts.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    if cat:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                                sparse_output=False))]), cat))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(features, sentinels, estimator) -> Pipeline:
    return Pipeline([
        ("sentinel", SentinelToNaN(sentinels=dict(sentinels))),
        ("prep", build_preprocessor(features)),
        ("clf", estimator),
    ])


def model_zoo(seed: int) -> dict:
    """Candidate classifiers with default hyper-parameters (used for the baseline CV round)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=seed),
        "KNN": KNeighborsClassifier(),
        "SVM (RBF)": SVC(probability=True, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }


def baseline_dummy() -> DummyClassifier:
    return DummyClassifier(strategy="prior")


# Hyper-parameter grids (prefix `clf__` because the estimator is the last pipeline step)
GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__class_weight": [None, "balanced"]},
    "KNN": {"clf__n_neighbors": [3, 5, 7, 9, 11, 15], "clf__weights": ["uniform", "distance"]},
    "SVM (RBF)": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]},
    "Random Forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 4, 8],
                      "clf__min_samples_leaf": [1, 3, 5]},
    "Gradient Boosting": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.03, 0.1],
                          "clf__max_depth": [2, 3]},
}

ENSEMBLE_MEMBERS = ["Logistic Regression", "SVM (RBF)", "Random Forest", "Gradient Boosting"]


def build_ensemble(tuned: dict) -> VotingClassifier:
    """Soft-voting ensemble over already-tuned full pipelines (each keeps its own preprocessing)."""
    from sklearn.base import clone
    return VotingClassifier([(n, clone(tuned[n])) for n in ENSEMBLE_MEMBERS], voting="soft")
