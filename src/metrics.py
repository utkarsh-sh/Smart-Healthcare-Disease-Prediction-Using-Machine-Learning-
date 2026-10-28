"""Evaluation helpers: classification metrics, bootstrap CIs and threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
                             roc_curve)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),       # sensitivity
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


def bootstrap_ci(y_true, proba, n_resamples: int, seed: int, threshold: float = 0.5, alpha: float = 0.05) -> dict:
    """Percentile bootstrap 95% CIs (resampling test rows with replacement)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    draws = {k: [] for k in keys}
    while len(draws["accuracy"]) < n_resamples:
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:      # need both classes for AUC
            continue
        m = classification_metrics(y_true[idx], proba[idx], threshold)
        for k in keys:
            draws[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in draws.items()}


def roc_points(y_true, proba, max_points: int = 200) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": [float(x) for x in fpr], "tpr": [float(x) for x in tpr]}


def screening_threshold(y_true, oof_proba, target_recall: float) -> float:
    """Highest probability threshold whose out-of-fold recall is >= target_recall."""
    fpr, tpr, thr = roc_curve(y_true, oof_proba)
    idx = int(np.argmax(tpr >= target_recall))
    t = float(thr[idx])
    return float(min(max(t, 0.0), 1.0))
