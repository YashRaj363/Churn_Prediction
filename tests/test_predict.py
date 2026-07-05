"""Model artifact + inference tests. Requires `python src/train.py` to have run."""
import pytest

import config
from src import predict as predict_mod

LOW_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 70, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 60.0,
    "TotalCharges": 4200.0,
}
HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 95.0,
}


@pytest.fixture(scope="module", autouse=True)
def _require_model():
    if not config.MODEL_PATH.exists():
        pytest.skip("model not trained; run `python src/train.py`")


def test_prediction_shape_and_bounds():
    r = predict_mod.predict_one(LOW_RISK)
    assert set(r) >= {"churn_probability", "churn_prediction", "risk_band", "model_version"}
    assert 0.0 <= r["churn_probability"] <= 1.0
    assert isinstance(r["churn_prediction"], bool)
    assert r["risk_band"] in {"low", "medium", "high"}


def test_high_risk_scores_above_low_risk():
    low = predict_mod.predict_one(LOW_RISK)["churn_probability"]
    high = predict_mod.predict_one(HIGH_RISK)["churn_probability"]
    assert high > low


def test_batch_matches_single():
    import pandas as pd
    df = pd.DataFrame([LOW_RISK, HIGH_RISK])
    out = predict_mod.predict_frame(df)
    assert len(out) == 2
    assert out[0]["churn_probability"] == predict_mod.predict_one(LOW_RISK)["churn_probability"]
