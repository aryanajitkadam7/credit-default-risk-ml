import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from credit_risk import config as C

pytestmark = pytest.mark.skipif(not (C.MODELS_DIR / "model.joblib").exists(), reason="train first")


@pytest.fixture(scope="module")
def client():
    from credit_risk.serve import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_example(client):
    payload = json.loads((Path(__file__).parents[1] / "examples" / "request.json").read_text())
    r = client.post("/predict", json=payload)
    body = r.json()
    assert r.status_code == 200
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] in {"review", "approve"}


def test_risky_profile_scores_higher_than_clean_profile(client):
    clean = {"LIMIT_BAL": 200000, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 40, "PAY": [-1] * 6,
             "BILL_AMT": [5000] * 6, "PAY_AMT": [5000] * 6}
    risky = {"LIMIT_BAL": 20000, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 25, "PAY": [3, 2, 2, 2, 2, 2],
             "BILL_AMT": [19500] * 6, "PAY_AMT": [0] * 6}
    p_clean = client.post("/predict", json=clean).json()["probability_of_default"]
    p_risky = client.post("/predict", json=risky).json()["probability_of_default"]
    assert p_risky > p_clean


def test_rejects_invalid_input(client):
    bad = {"LIMIT_BAL": -5, "EDUCATION": 1, "MARRIAGE": 1, "AGE": 30, "PAY": [0] * 6,
           "BILL_AMT": [0] * 6, "PAY_AMT": [0] * 6}
    assert client.post("/predict", json=bad).status_code == 422
