"""API tests using FastAPI's TestClient (no live server needed)."""
import pytest
from fastapi.testclient import TestClient

import config
from app.main import app
from tests.test_predict import HIGH_RISK, LOW_RISK


@pytest.fixture(scope="module")
def client():
    if not config.MODEL_PATH.exists():
        pytest.skip("model not trained; run `python src/train.py`")
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_ready"] is True


def test_predict_endpoint(client):
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}


def test_predict_validation_error(client):
    bad = dict(LOW_RISK)
    bad["Contract"] = "Weekly"  # not an allowed Literal
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_endpoint(client):
    r = client.post("/predict/batch", json={"records": [LOW_RISK, HIGH_RISK]})
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_empty_batch_rejected(client):
    r = client.post("/predict/batch", json={"records": []})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json=LOW_RISK)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.json()["total_requests"] >= 1
