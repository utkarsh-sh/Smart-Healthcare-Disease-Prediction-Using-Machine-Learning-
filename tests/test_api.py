"""REST API + UI integration tests (Flask test client, real trained models, temporary SQLite database)."""
import copy

import pytest

from src.registry import DISEASES


def payload(examples, key, which="positive"):
    return copy.deepcopy(examples[key][which]["values"])


# ------------------------------------------------------------------ pages and metadata
def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Smart Healthcare Disease Prediction" in html and "NOT a medical diagnosis" in html


def test_static_assets_are_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_health_reports_modules_and_database(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["modules"] == ["diabetes", "heart", "breast"]


def test_disease_catalogue_and_detail(client):
    cat = client.get("/api/diseases").get_json()
    assert {d["key"] for d in cat} == set(DISEASES)
    for d in cat:
        assert d["features"] and 0 <= d["metrics"]["test"]["roc_auc"] <= 1
    one = client.get("/api/diseases/heart").get_json()
    assert one["name"] == "Heart Disease" and len(one["features"]) == 13


def test_unknown_disease_returns_404_json(client):
    assert client.get("/api/diseases/malaria").status_code == 404
    r = client.post("/api/predict/malaria", json={})
    assert r.status_code == 404 and "error" in r.get_json()


# ------------------------------------------------------------------ prediction
@pytest.mark.parametrize("key", list(DISEASES))
def test_predict_each_module_end_to_end(client, examples, key):
    r = client.post(f"/api/predict/{key}", json=payload(examples, key))
    assert r.status_code == 200
    body = r.get_json()
    assert 0 <= body["probability"] <= 1 and body["disease"] == key
    assert len(body["top_factors"]) == 3 and body["history_id"] >= 1
    assert body["model"]["version"]


def test_positive_example_gets_higher_risk_than_negative_through_api(client, examples):
    for key in DISEASES:
        hi = client.post(f"/api/predict/{key}", json=payload(examples, key, "positive")).get_json()
        lo = client.post(f"/api/predict/{key}", json=payload(examples, key, "negative")).get_json()
        assert hi["probability"] > lo["probability"]


def test_explain_query_flag(client, examples):
    r = client.post("/api/predict/heart?explain=false", json=payload(examples, "heart"))
    assert "top_factors" not in r.get_json()


def test_invalid_values_return_422_with_field_errors(client, examples):
    p = payload(examples, "diabetes")
    p["Glucose"] = 9999
    p["Age"] = "abc"
    r = client.post("/api/predict/diabetes", json=p)
    assert r.status_code == 422
    fields = r.get_json()["fields"]
    assert set(fields) == {"Glucose", "Age"}


def test_missing_required_field_returns_422(client):
    r = client.post("/api/predict/breast", json={"mean_radius": 14})
    assert r.status_code == 422 and len(r.get_json()["fields"]) == 9


@pytest.mark.parametrize("kwargs", [
    {"data": "not json", "content_type": "application/json"},
    {"data": "{}", "content_type": "text/plain"},
    {"json": [1, 2, 3]},
])
def test_malformed_bodies_return_400(client, kwargs):
    r = client.post("/api/predict/heart", **kwargs)
    assert r.status_code == 400 and "error" in r.get_json()


def test_wrong_http_method_returns_json_405(client):
    r = client.get("/api/predict/heart")
    assert r.status_code == 405 and r.is_json


def test_unknown_api_route_returns_json_404(client):
    r = client.get("/api/nothing-here")
    assert r.status_code == 404 and r.is_json


def test_oversized_body_returns_413(client):
    r = client.post("/api/predict/heart", data="x" * 1_100_000, content_type="application/json")
    assert r.status_code == 413


# ------------------------------------------------------------------ batch
def test_batch_endpoint(client, examples):
    rows = [payload(examples, "heart", "positive"), payload(examples, "heart", "negative")]
    bad = payload(examples, "heart")
    bad["chol"] = -5
    r = client.post("/api/predict/heart/batch", json={"rows": rows + [bad]})
    body = r.get_json()
    assert r.status_code == 200 and (body["n_scored"], body["n_invalid"]) == (2, 1)
    assert "chol" in body["errors"]["2"]


@pytest.mark.parametrize("body", [{}, {"rows": []}, {"rows": "x"}, [1]])
def test_batch_rejects_bad_requests(client, body):
    assert client.post("/api/predict/heart/batch", json=body).status_code == 400


def test_batch_limit_is_enforced(client, examples):
    rows = [payload(examples, "heart")] * 501
    assert client.post("/api/predict/heart/batch", json={"rows": rows}).status_code == 400


# ------------------------------------------------------------------ database / history
def test_predictions_are_logged_and_listed_newest_first(client, examples):
    client.post("/api/predict/heart", json=payload(examples, "heart", "positive"))
    client.post("/api/predict/diabetes", json=payload(examples, "diabetes", "negative"))
    h = client.get("/api/history").get_json()
    assert [i["disease"] for i in h["items"]] == ["diabetes", "heart"]
    assert h["counts"] == {"diabetes": 1, "heart": 1}
    assert h["items"][1]["inputs"]["age"] == examples["heart"]["positive"]["values"]["age"]


def test_history_filters_and_validates_limit(client, examples):
    for _ in range(3):
        client.post("/api/predict/breast", json=payload(examples, "breast"))
    client.post("/api/predict/heart", json=payload(examples, "heart"))
    assert len(client.get("/api/history?disease=breast&limit=2").get_json()["items"]) == 2
    assert client.get("/api/history?limit=abc").status_code == 400


def test_failed_validation_is_not_logged(client):
    client.post("/api/predict/heart", json={"age": 1})
    assert client.get("/api/history").get_json()["items"] == []


def test_logging_can_be_disabled(tmp_path, examples):
    from app.api import create_app
    app = create_app({"DB_PATH": str(tmp_path / "x.db"), "LOG_PREDICTIONS": False, "TESTING": True})
    c = app.test_client()
    r = c.post("/api/predict/heart", json=payload(examples, "heart"))
    assert r.status_code == 200 and "history_id" not in r.get_json()
    assert c.get("/api/history").get_json()["items"] == []
