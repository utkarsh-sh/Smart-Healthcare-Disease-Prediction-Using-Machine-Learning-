"""SQLite store tests: schema constraints, CRUD and ordering."""
import sqlite3

import pytest

from src.store import PredictionStore


def result(prob=0.7, disease="heart"):
    return {"disease": disease, "probability": prob, "prediction": int(prob >= 0.5), "screening_flag": True,
            "model": {"name": "Logistic Regression", "version": "1.0.0"}, "latency_ms": 3.2}


@pytest.fixture()
def store(tmp_path):
    return PredictionStore(tmp_path / "sub" / "p.db")           # also checks parent dir creation


def test_add_and_recent_roundtrip(store):
    rid = store.add(result(0.81), {"age": 55, "sex": 1})
    row = store.recent(1)[0]
    assert row["id"] == rid and row["inputs"] == {"age": 55, "sex": 1}
    assert row["probability"] == pytest.approx(0.81) and row["screening_flag"] == 1


def test_recent_orders_newest_first_and_filters(store):
    for p, d in [(0.1, "heart"), (0.2, "breast"), (0.3, "heart")]:
        store.add(result(p, d), {})
    assert [r["probability"] for r in store.recent(10)] == [0.3, 0.2, 0.1]
    assert [r["disease"] for r in store.recent(10, "heart")] == ["heart", "heart"]
    assert store.counts() == {"heart": 2, "breast": 1}


def test_database_rejects_out_of_range_probability(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add(result(1.5), {})


def test_clear_empties_table(store):
    store.add(result(), {})
    store.clear()
    assert store.recent() == [] and store.counts() == {}


def test_schema_creation_is_idempotent(tmp_path):
    PredictionStore(tmp_path / "a.db").add(result(), {})
    assert len(PredictionStore(tmp_path / "a.db").recent()) == 1   # re-opening keeps data
