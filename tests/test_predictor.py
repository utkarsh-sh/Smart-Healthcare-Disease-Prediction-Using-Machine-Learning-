"""Inference-service tests: validation, prediction, explanation, batch scoring and artefact integrity."""
import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.metrics import classification_metrics
from src.predictor import ValidationError
from src.registry import DISEASES, get_spec


def valid_payload(examples, key, which="positive"):
    return copy.deepcopy(examples[key][which]["values"])


# ------------------------------------------------------------------ catalogue
def test_all_three_modules_are_loaded(predictor):
    assert predictor.available() == ["diabetes", "heart", "breast"]


def test_describe_exposes_ui_contract(predictor):
    d = predictor.describe("heart")
    assert {"features", "thresholds", "metrics", "examples", "train_stats"} <= set(d)
    assert len(d["features"]) == 13
    assert 0 < d["thresholds"]["screening"] < 1


def test_unknown_or_untrained_module_raises(predictor):
    with pytest.raises(KeyError):
        predictor.predict("malaria", {})


# ------------------------------------------------------------------ prediction
@pytest.mark.parametrize("key", list(DISEASES))
def test_prediction_is_a_probability_and_deterministic(predictor, examples, key):
    payload = valid_payload(examples, key)
    a = predictor.predict(key, payload)
    b = predictor.predict(key, payload)
    assert 0.0 <= a["probability"] <= 1.0
    assert a["probability"] == b["probability"]
    assert a["prediction"] in (0, 1)
    assert a["prediction"] == int(a["probability"] >= a["threshold"])
    assert a["screening_flag"] == (a["probability"] >= a["screening_threshold"])
    assert a["latency_ms"] > 0 and "disclaimer" in a


@pytest.mark.parametrize("key", list(DISEASES))
def test_typical_positive_scores_higher_than_typical_negative(predictor, examples, key):
    pos = predictor.predict(key, valid_payload(examples, key, "positive"))["probability"]
    neg = predictor.predict(key, valid_payload(examples, key, "negative"))["probability"]
    assert pos > neg


@pytest.mark.parametrize("key", list(DISEASES))
def test_serving_path_matches_training_pipeline(predictor, key):
    """Train/serve parity: scoring a raw hold-out row through the service must equal scoring it directly
    through the saved pipeline (sentinels such as Pima zeros are mapped to 'missing' on both paths)."""
    spec = get_spec(key)
    hold = pd.read_csv(config.MODELS_DIR / f"{key}_holdout.csv")
    raw = hold[spec.feature_names].iloc[[0]]
    direct = float(predictor.models[key].predict_proba(raw)[0, 1])
    payload = {}
    for f in spec.features:
        v = float(raw.iloc[0][f.name])
        if v in spec.sentinels.get(f.name, []):
            v = None                                  # what a user would leave blank
        payload[f.name] = None if v is None else (int(v) if f.kind != "numeric" else v)
    served = predictor.predict(key, payload, explain=False)["probability"]
    assert served == pytest.approx(direct, abs=1e-12)


@pytest.mark.parametrize("key", list(DISEASES))
def test_saved_model_reproduces_reported_holdout_metrics(predictor, key):
    """Artefact-integrity regression test: reloading the model must give exactly the metrics in the report."""
    hold = pd.read_csv(config.MODELS_DIR / f"{key}_holdout.csv")
    y = hold.pop("y")
    proba = predictor.models[key].predict_proba(hold)[:, 1]
    reported = predictor.meta[key]["metrics"]["test"]
    now = classification_metrics(y, proba, 0.5)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert now[k] == pytest.approx(reported[k], abs=1e-9), k
    assert len(hold) == predictor.meta[key]["n_test"]


# ------------------------------------------------------------------ validation
def test_optional_fields_may_be_blank_and_produce_warning(predictor, examples):
    p = valid_payload(examples, "diabetes")
    p["Insulin"] = None
    p.pop("SkinThickness", None)
    r = predictor.predict("diabetes", p)
    warnings = " ".join(r["warnings"]).lower()
    assert "insulin" in warnings and "skin-fold" in warnings
    assert 0 <= r["probability"] <= 1


def test_required_field_missing_is_reported(predictor, examples):
    p = valid_payload(examples, "diabetes")
    del p["Age"]
    with pytest.raises(ValidationError) as exc:
        predictor.predict("diabetes", p)
    assert exc.value.errors == {"Age": "Required."}


@pytest.mark.parametrize("field,value,fragment", [
    ("Glucose", 5000, "Must be between"),
    ("Glucose", 0, "Must be between"),                 # 0 is a missing-value code, not a measurement
    ("Age", "forty", "Must be a number"),
    ("Age", True, "Must be a number"),                 # bool is not a number
    ("Age", 40.5, "whole number"),
    ("BMI", float("nan"), "finite"),
    ("BMI", float("inf"), "finite"),
])
def test_invalid_numeric_values_are_rejected(predictor, examples, field, value, fragment):
    p = valid_payload(examples, "diabetes")
    p[field] = value
    with pytest.raises(ValidationError) as exc:
        predictor.predict("diabetes", p)
    assert fragment in exc.value.errors[field]


def test_invalid_category_and_unknown_field_rejected(predictor, examples):
    p = valid_payload(examples, "heart")
    p["cp"] = 9
    p["blood_type"] = 1
    with pytest.raises(ValidationError) as exc:
        predictor.predict("heart", p)
    assert "Must be one of" in exc.value.errors["cp"]
    assert exc.value.errors["blood_type"] == "Unknown field."


def test_all_errors_are_collected_at_once(predictor):
    with pytest.raises(ValidationError) as exc:
        predictor.predict("breast", {"mean_radius": "x"})
    assert len(exc.value.errors) == 10                 # 1 wrong type + 9 missing


@pytest.mark.parametrize("bad", [[], "text", 42, None])
def test_non_object_payload_rejected(predictor, bad):
    with pytest.raises(ValidationError) as exc:
        predictor.predict("heart", bad)
    assert "_payload" in exc.value.errors


def test_out_of_distribution_value_triggers_warning_but_still_predicts(predictor, examples):
    p = valid_payload(examples, "diabetes")
    p["Insulin"] = 880                                  # valid range, far beyond the training 99th percentile
    r = predictor.predict("diabetes", p)
    assert any("outside the range typical" in w for w in r["warnings"])


# ------------------------------------------------------------------ explanation
@pytest.mark.parametrize("key", list(DISEASES))
def test_explanation_structure(predictor, examples, key):
    r = predictor.predict(key, valid_payload(examples, key), top_k=3)
    f = r["top_factors"]
    assert len(f) == 3
    assert [abs(x["effect_pp"]) for x in f] == sorted((abs(x["effect_pp"]) for x in f), reverse=True)
    assert all(math.isfinite(x["effect_pp"]) and abs(x["effect_pp"]) <= 100 for x in f)
    assert all(x["direction"] in ("raises risk", "lowers risk", "neutral") for x in f)


def test_explanation_skips_features_the_user_did_not_provide(predictor, examples):
    p = valid_payload(examples, "diabetes")
    p["Insulin"] = None
    names = {x["feature"] for x in predictor.predict("diabetes", p, top_k=8)["top_factors"]}
    assert "Insulin" not in names


def test_explain_can_be_disabled(predictor, examples):
    assert "top_factors" not in predictor.predict("heart", valid_payload(examples, "heart"), explain=False)


# ------------------------------------------------------------------ batch
def test_batch_matches_single_predictions_and_reports_invalid_rows(predictor, examples):
    good = [valid_payload(examples, "heart", "positive"), valid_payload(examples, "heart", "negative")]
    bad = valid_payload(examples, "heart")
    bad["age"] = 500
    res = predictor.predict_batch("heart", [good[0], bad, good[1]])
    assert (res["n_rows"], res["n_scored"], res["n_invalid"]) == (3, 2, 1)
    assert "age" in res["errors"]["1"]
    singles = [predictor.predict("heart", g, explain=False)["probability"] for g in good]
    assert [r["probability"] for r in res["results"]] == pytest.approx(singles, abs=1e-12)
    assert [r["index"] for r in res["results"]] == [0, 2]


def test_metadata_is_json_serialisable_and_versioned(predictor):
    for key in DISEASES:
        json.dumps(predictor.describe(key))
        assert predictor.meta[key]["version"] == config.MODEL_VERSION
