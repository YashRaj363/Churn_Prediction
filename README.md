# Customer Churn Prediction — End-to-End MLOps

Predicts whether a telecom customer is likely to churn, served as a FastAPI REST
API with built-in request logging, data-drift detection, and alerting. Packaged
for Docker and deployable to Google Cloud Run.

Covers the full lifecycle: **train → save → serve → containerize → deploy →
monitor → detect drift → log → alert.**

---

## Model

Trained on the **IBM Telco Customer Churn** dataset (7,043 customers; downloaded
automatically, with a synthetic fallback for offline use).

| | |
|---|---|
| Selected model | `GradientBoostingClassifier` (beat LogisticRegression on CV ROC-AUC) |
| Hold-out ROC-AUC | **0.843** |
| Accuracy | 0.803 |
| Precision / Recall / F1 (churn) | 0.67 / 0.52 / 0.58 |
| Class balance | 26.5% churn |

Preprocessing (median/most-frequent impute → one-hot encode → scale) and the
classifier live in a single sklearn `Pipeline`, so serving is just
`DataFrame in → probability out`. Full details in `models/metadata.json`.

---

## Project layout

```
churn-mlops/
├── config.py                  # paths, feature contract, thresholds (single source of truth)
├── requirements.txt
├── Dockerfile                 # Cloud Run-compatible image (port 8080)
├── cloudbuild.yaml            # Cloud Build: build → push → deploy
├── deploy/cloudrun.sh         # one-command Cloud Run deploy
├── data/                      # raw CSV + reference_profile.json (drift baseline)
├── models/                    # churn_model.joblib + metadata.json
├── logs/                      # requests.jsonl + alerts.log
├── src/
│   ├── data/load.py           # download IBM Telco (synthetic fallback) + build reference profile
│   ├── train.py               # model selection, training, artifact persistence
│   ├── schema.py              # Pydantic request/response models
│   ├── predict.py             # cached model load + inference
│   └── monitoring/
│       ├── logger.py          # JSONL request/prediction logger
│       ├── drift.py           # PSI + KS drift vs reference profile
│       └── alerts.py          # threshold rules → alerts.log + optional webhook
├── app/main.py                # FastAPI app
└── tests/                     # pytest: predict, api, drift
```

---

## Quickstart (local)

```bash
# 1. Install deps
python -m pip install -r requirements.txt

# 2. Get data + build drift baseline
python src/data/load.py            # add --synthetic to skip the download

# 3. Train the model (writes models/churn_model.joblib + metadata.json)
python src/train.py

# 4. Serve
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
#    interactive docs at http://localhost:8080/docs
```

### Example request

```bash
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender":"Female","SeniorCitizen":1,"Partner":"No","Dependents":"No",
    "tenure":1,"PhoneService":"Yes","MultipleLines":"No",
    "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
    "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
    "StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check","MonthlyCharges":95.0,"TotalCharges":95.0
  }'
# => {"churn_probability":0.8927,"churn_prediction":true,"risk_band":"high",...}
```

---

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service info |
| GET | `/health` | Liveness/readiness (200 if model loaded, else 503) |
| GET | `/model` | Model metadata + metrics |
| POST | `/predict` | Score one customer |
| POST | `/predict/batch` | Score many customers |
| GET | `/metrics` | Operational metrics from the request log |
| GET | `/monitoring/report` | Drift report vs training reference + triggered alerts |
| GET | `/docs` | Swagger UI |

Invalid categorical values are rejected at the edge with a **422** (Pydantic
`Literal` constraints), never silently mishandled.

---

## Monitoring, drift & alerting

Every prediction is appended to `logs/requests.jsonl` (timestamp, request id,
latency, features, prediction). From those logs:

- **`/metrics`** — request volume, error rate, avg/p95 latency, predicted churn
  rate, average probability.
- **`/monitoring/report`** — compares recent live traffic to the training
  **reference profile** (`data/reference_profile.json`):
  - Numeric features → **PSI** over fixed reference bins + **KS** two-sample test.
  - Categorical features → **PSI** over category frequencies.
  - A feature is flagged when `PSI ≥ 0.25` (or KS p-value `< 0.05`); a
    **dataset-level drift** flag fires when `> 30%` of features drift.
- **Alerts** — drift, high latency, and high error-rate breaches are written to
  `logs/alerts.log` and, if `ALERT_WEBHOOK_URL` is set, POSTed to that webhook
  (e.g. Slack). Thresholds are all configurable in `config.py`.

Verified end-to-end: an in-distribution batch reports **0/19 drifting, 0
alerts**; a batch with short tenure / fiber / month-to-month / high charges
reports **6/19 drifting** and raises a `critical data_drift` alert.

---

## Tests

```bash
python -m pytest -q      # 13 tests: inference, API (TestClient), drift math
```

Tests skip gracefully with a clear message if the model hasn't been trained yet.

---

## Docker & Cloud Run deploy

> Note: the image build and cloud deploy were **not executed** in the
> development environment (no Docker/gcloud there); the artifacts below are
> ready to run wherever those tools are available.

### Local container

```bash
docker build -t churn-api .
docker run -p 8080:8080 churn-api
```

### Google Cloud Run

```bash
# Option A — scripted (creates Artifact Registry repo, builds, deploys)
PROJECT_ID=my-gcp-project ./deploy/cloudrun.sh

# Option B — Cloud Build pipeline (build → push → deploy)
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_REGION=us-central1,_REPO=churn,_SERVICE=churn-api
```

The container binds to Cloud Run's injected `$PORT` (default 8080). The trained
model is baked into the image, so the container is self-contained — retrain and
rebuild to ship a new model version.

---

## Configuration

Key knobs (env-overridable) in `config.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `DECISION_THRESHOLD` | 0.5 | Probability ≥ this ⇒ predicted churn |
| `PSI_ALERT` | 0.25 | Per-feature PSI drift threshold |
| `KS_PVALUE_ALERT` | 0.05 | Numeric-feature KS drift threshold |
| `DRIFT_FEATURE_FRACTION_ALERT` | 0.30 | Fraction of features drifting ⇒ dataset drift |
| `LATENCY_MS_ALERT` | 1000 | p95 latency alert |
| `ERROR_RATE_ALERT` | 0.05 | Error-rate alert |
| `ALERT_WEBHOOK_URL` | "" | Optional outbound alert webhook |
