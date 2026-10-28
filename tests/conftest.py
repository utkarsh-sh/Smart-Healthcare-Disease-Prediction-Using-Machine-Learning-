import pytest

from app.api import create_app
from src.predictor import DiseasePredictor
from src.registry import DISEASES


@pytest.fixture(scope="session")
def predictor():
    p = DiseasePredictor()
    assert p.available(), "Trained models not found - run `PYTHONPATH=. python -m src.train` first."
    return p


@pytest.fixture()
def app(tmp_path):
    return create_app({"DB_PATH": str(tmp_path / "test_predictions.db"), "TESTING": True})


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def examples(predictor):
    """The 'typical positive / negative' cases stored with every trained module."""
    return {k: predictor.meta[k]["examples"] for k in DISEASES}
