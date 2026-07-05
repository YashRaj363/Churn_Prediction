"""FastAPI service for churn prediction + monitoring.

Endpoints
  GET  /                     service info
  GET  /health              liveness/readiness (model loaded?)
  GET  /model               model metadata + metrics
  POST /predict             score a single customer
  POST /predict/batch       score many customers
  GET  /metrics             operational metrics from the request log
  GET  /monitoring/report   data-drift report vs training reference + alerts

Every prediction is logged to logs/requests.jsonl and used to compute live
operational metrics and drift.
"""
from __future__ import annotations

import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import predict as predict_mod  # noqa: E402
from src.monitoring import alerts as alerts_mod  # noqa: E402
from src.monitoring import drift as drift_mod  # noqa: E402
from src.monitoring import logger as req_logger  # noqa: E402
from src.schema import (  # noqa: E402
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model on startup so the first request isn't slow (and we fail
    # fast if artifacts are missing).
    try:
        predict_mod.get_model()
        app.state.model_ready = True
    except predict_mod.ModelNotTrained:
        app.state.model_ready = False
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts telecom customer churn with built-in drift monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "churn-prediction",
        "version": app.version,
        "dashboard": "/ui",
        "docs": "/docs",
        "endpoints": ["/health", "/model", "/predict", "/predict/batch", "/metrics", "/monitoring/report"],
    }


@app.get("/health")
def health():
    ready = getattr(app.state, "model_ready", False)
    status_code = 200 if ready else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if ready else "model_not_loaded", "model_ready": ready},
    )


@app.get("/model")
def model_info():
    try:
        return predict_mod.get_metadata()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


def _score(features: dict) -> dict:
    """Score one record, logging the outcome (success or failure)."""
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        result = predict_mod.predict_one(features)
        latency_ms = (time.perf_counter() - t0) * 1000
        req_logger.log_prediction(
            features=features, result=result, latency_ms=latency_ms,
            request_id=request_id, status="ok",
        )
        return result
    except predict_mod.ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        req_logger.log_prediction(
            features=features, result={}, latency_ms=latency_ms,
            request_id=request_id, status="error", error=repr(exc),
        )
        raise HTTPException(status_code=500, detail="prediction failed")


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    return _score(customer.model_dump())


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not req.records:
        raise HTTPException(status_code=422, detail="records must not be empty")
    preds = [_score(r.model_dump()) for r in req.records]
    return {"predictions": preds, "count": len(preds)}


def _operational_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"total_requests": 0}
    latencies = np.array([r.get("latency_ms", 0.0) for r in rows], dtype=float)
    errors = sum(1 for r in rows if r.get("status") == "error")
    probs = [r["churn_probability"] for r in rows if r.get("churn_probability") is not None]
    positives = sum(1 for r in rows if r.get("churn_prediction") is True)
    return {
        "total_requests": len(rows),
        "error_count": errors,
        "error_rate": round(errors / len(rows), 4),
        "latency_ms_avg": round(float(latencies.mean()), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "predicted_churn_rate": round(positives / len(rows), 4),
        "avg_churn_probability": round(float(np.mean(probs)), 4) if probs else None,
    }


@app.get("/metrics")
def metrics():
    rows = req_logger.read_recent(limit=5000)
    return _operational_metrics(rows)


@app.get("/monitoring/report")
def monitoring_report():
    rows = req_logger.read_recent(limit=5000)
    ops = _operational_metrics(rows)

    # Build a live feature frame from logged successful predictions.
    feats = [r["features"] for r in rows if r.get("status") == "ok" and r.get("features")]
    if feats:
        live_df = pd.DataFrame(feats)
        drift_report = drift_mod.compute_drift(live_df)
    else:
        drift_report = {"status": "insufficient_data", "n_samples": 0}

    triggered = alerts_mod.evaluate(drift_report=drift_report, ops=ops)
    return {
        "operational": ops,
        "drift": drift_report,
        "alerts": triggered,
        "alerts_triggered": len(triggered),
    }


# ---- Frontend dashboard ----
_FRONTEND_DIR = config.ROOT / "frontend"


@app.get("/ui", include_in_schema=False)
def dashboard_ui():
    """Serve the ChurnGuard dashboard."""
    return FileResponse(_FRONTEND_DIR / "index.html")


if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")
