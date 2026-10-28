"""Unit tests for the evaluation helpers (known-answer tests)."""
import numpy as np
import pytest

from src.metrics import bootstrap_ci, classification_metrics, roc_points, screening_threshold


def test_classification_metrics_known_answer():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .7, .3, .6, .2, .1, .1, .2, .3])       # 3 TP, 1 FN, 1 FP, 5 TN at threshold 0.5
    m = classification_metrics(y, p, 0.5)
    assert m["confusion"] == {"tn": 5, "fp": 1, "fn": 1, "tp": 3}
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(5 / 6) and m["f1"] == pytest.approx(0.75)
    assert 0.9 < m["roc_auc"] <= 1.0


def test_perfect_and_worst_rankings():
    y = np.array([0, 0, 1, 1])
    assert classification_metrics(y, np.array([.1, .2, .8, .9]))["roc_auc"] == 1.0
    assert classification_metrics(y, np.array([.9, .8, .2, .1]))["roc_auc"] == 0.0


def test_bootstrap_ci_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    a = bootstrap_ci(y, p, 300, seed=1)
    b = bootstrap_ci(y, p, 300, seed=1)
    assert a == b
    point = classification_metrics(y, p)
    for k in ("accuracy", "recall", "roc_auc"):
        lo, hi = a[k]
        assert lo <= point[k] <= hi and hi - lo > 0


def test_screening_threshold_reaches_target_recall():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 500), 0, 1)
    thr = screening_threshold(y, p, 0.9)
    assert 0 <= thr <= 1
    assert classification_metrics(y, p, thr)["recall"] >= 0.9


def test_roc_points_are_monotone_and_thinned():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 1000)
    pts = roc_points(y, rng.random(1000), max_points=50)
    assert len(pts["fpr"]) == len(pts["tpr"]) <= 50
    assert pts["fpr"] == sorted(pts["fpr"]) and pts["tpr"] == sorted(pts["tpr"])
