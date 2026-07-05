"""Drift math tests: no-drift on reference-like data, drift on shifted data."""
import numpy as np
import pandas as pd
import pytest

import config
from src.monitoring import drift as drift_mod


@pytest.fixture(scope="module", autouse=True)
def _require_profile():
    if not config.REFERENCE_PROFILE.exists():
        pytest.skip("reference profile missing; run `python src/data/load.py`")


def _reference_like(n=200):
    df = pd.read_csv(config.RAW_CSV)
    feat = [c for c in df.columns if c not in (config.ID_COL, config.TARGET)]
    return df.sample(n, random_state=7)[feat].reset_index(drop=True)


def test_insufficient_data_returns_status():
    small = _reference_like(5)
    rep = drift_mod.compute_drift(small)
    assert rep["status"] == "insufficient_data"


def test_no_drift_on_reference_like_sample():
    rep = drift_mod.compute_drift(_reference_like(300))
    assert rep["status"] == "ok"
    assert rep["dataset_drift"] is False


def test_drift_detected_on_shifted_data():
    df = _reference_like(300)
    # Force an extreme shift on enough features to cross the dataset-level
    # threshold (>30% of features drifting).
    df["tenure"] = 0
    df["TotalCharges"] = 0.0
    df["MonthlyCharges"] = 119.0
    df["Contract"] = "Month-to-month"
    df["InternetService"] = "Fiber optic"
    df["PaymentMethod"] = "Electronic check"
    df["PaperlessBilling"] = "Yes"
    rep = drift_mod.compute_drift(df)
    assert rep["status"] == "ok"
    # Individual shifted features must be flagged...
    assert "tenure" in rep["drifting_features"]
    assert "MonthlyCharges" in rep["drifting_features"]
    # ...and enough of them to raise the dataset-level flag.
    assert rep["dataset_drift"] is True


def test_psi_zero_for_identical_distributions():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert drift_mod._psi(p, p) == pytest.approx(0.0, abs=1e-9)
