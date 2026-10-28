"""Custom scikit-learn transformers used inside the saved model pipelines.

Keeping this logic *inside* the pipeline (rather than in the training script) guarantees the
exact same cleaning is applied at training time and at serving time (no train/serve skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SentinelToNaN(BaseEstimator, TransformerMixin):
    """Replace sentinel / physiologically impossible codes with NaN (stateless).

    Example: in the Pima diabetes data a BMI of 0 or a blood pressure of 0 is a missing-value
    placeholder, not a measurement. ``sentinels = {"BMI": [0], "Glucose": [0]}``.
    """

    def __init__(self, sentinels=None):
        self.sentinels = sentinels

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy().astype(float)
        for col, values in (self.sentinels or {}).items():
            if col in X.columns:
                X[col] = X[col].mask(X[col].isin(values))
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
