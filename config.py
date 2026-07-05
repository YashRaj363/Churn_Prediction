"""Central configuration: paths, feature schema, thresholds, settings.

Everything downstream (training, serving, monitoring) imports from here so the
feature contract and thresholds live in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

for _d in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "telco_churn.csv"
REFERENCE_PROFILE = DATA_DIR / "reference_profile.json"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
MODEL_METADATA = MODELS_DIR / "metadata.json"

REQUEST_LOG = LOGS_DIR / "requests.jsonl"
ALERT_LOG = LOGS_DIR / "alerts.log"

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Model / decision
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Probability >= this => predicted "will churn". Tunable per business cost.
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# Monitoring thresholds
# ---------------------------------------------------------------------------
# Population Stability Index interpretation:
#   < 0.1 no significant change, 0.1-0.25 moderate, > 0.25 major shift
PSI_WARN = 0.1
PSI_ALERT = 0.25
# KS p-value below this flags a distribution change for numeric features.
KS_PVALUE_ALERT = 0.05
# Fraction of monitored features drifting that triggers a dataset-level alert.
DRIFT_FEATURE_FRACTION_ALERT = 0.3

# Operational alert thresholds
LATENCY_MS_ALERT = 1000.0
ERROR_RATE_ALERT = 0.05

# Optional outbound alert webhook (Slack/Teams/etc). Empty => log only.
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")

# Minimum number of recent requests before drift is computed.
DRIFT_MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "50"))
