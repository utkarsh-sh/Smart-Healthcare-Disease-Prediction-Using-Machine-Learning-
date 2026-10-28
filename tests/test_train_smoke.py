"""End-to-end smoke test of the training pipeline (fast mode, writes only to a temporary directory)."""
import json

import joblib
import pandas as pd
import pytest

from src import config, train
from src.predictor import DiseasePredictor


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "MODELS_DIR", out)
    try:
        result = train.run_disease("heart", fast=True)
    finally:
        mp.undo()
    return out, result


def test_artifacts_are_written(trained):
    out, _ = trained
    for name in ("heart_model.joblib", "heart_meta.json", "heart_holdout.csv"):
        assert (out / name).exists()
    meta = json.loads((out / "heart_meta.json").read_text())
    assert meta["thresholds"]["default"] == 0.5 and 0 < meta["thresholds"]["screening"] < 1


def test_protocol_outputs_are_complete_and_sane(trained):
    _, r = trained
    assert r["data"]["n_train"] + r["data"]["n_test"] == r["data"]["n_samples"] == 302
    assert set(r["cv_baseline"]) >= {"Majority-class baseline", "Logistic Regression", "Random Forest"}
    assert r["cv_baseline"]["Majority-class baseline"]["roc_auc"]["mean"] == pytest.approx(0.5)
    assert r["test"]["roc_auc"] > 0.8                       # far above chance on held-out data
    assert r["selected_model"] in r["selection"]
    lo, hi = r["test_ci95"]["accuracy"]
    assert lo <= r["test"]["accuracy"] <= hi
    assert r["stability"]["accuracy"]["min"] <= r["stability"]["accuracy"]["max"]
    assert r["permutation_importance"][0]["mean_auc_drop"] >= r["permutation_importance"][-1]["mean_auc_drop"]


def test_saved_model_loads_and_predicts(trained):
    out, _ = trained
    model = joblib.load(out / "heart_model.joblib")
    hold = pd.read_csv(out / "heart_holdout.csv")
    proba = model.predict_proba(hold.drop(columns="y"))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all() and len(proba) == len(hold)
    served = DiseasePredictor(out)
    assert served.available() == ["heart"]


def test_model_selection_prefers_simpler_model_on_ties():
    sel = {"Random Forest": {"roc_auc": {"mean": 0.900}}, "Logistic Regression": {"roc_auc": {"mean": 0.8995}},
           "Soft-Voting Ensemble": {"roc_auc": {"mean": 0.901}}}
    assert train.choose_model(sel) == "Logistic Regression"          # all within 0.002 -> simplest wins
    sel["Logistic Regression"]["roc_auc"]["mean"] = 0.85
    assert train.choose_model(sel) == "Random Forest"                # LR now clearly worse -> next simplest tied


def test_tie_tolerance_is_float_safe_at_the_boundary():
    sel = {"Logistic Regression": {"roc_auc": {"mean": 0.899}}, "Soft-Voting Ensemble": {"roc_auc": {"mean": 0.901}}}
    assert train.choose_model(sel) == "Logistic Regression"          # difference is exactly the 0.002 tolerance
