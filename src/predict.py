"""Model loading and inference.

The trained artifact is a full sklearn Pipeline, so inference is just a DataFrame
in -> probabilities out. The model is loaded once and cached.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


class ModelNotTrained(RuntimeError):
    """Raised when inference is attempted before training has produced artifacts."""


@lru_cache(maxsize=1)
def get_model():
    if not config.MODEL_PATH.exists():
        raise ModelNotTrained(
            f"{config.MODEL_PATH} not found. Run `python src/train.py` first."
        )
    return joblib.load(config.MODEL_PATH)


@lru_cache(maxsize=1)
def get_metadata() -> dict:
    if config.MODEL_METADATA.exists():
        with open(config.MODEL_METADATA, encoding="utf-8") as fh:
            return json.load(fh)
    return {"model_version": "unknown"}


def _risk_band(prob: float) -> str:
    if prob >= 0.66:
        return "high"
    if prob >= 0.33:
        return "medium"
    return "low"


def predict_frame(df: pd.DataFrame) -> list[dict]:
    """Score a DataFrame of feature rows; returns one result dict per row."""
    model = get_model()
    meta = get_metadata()
    threshold = float(meta.get("decision_threshold", config.DECISION_THRESHOLD))

    probs = model.predict_proba(df[config.FEATURE_COLUMNS])[:, 1]
    out = []
    for p in probs:
        p = float(p)
        out.append({
            "churn_probability": round(p, 4),
            "churn_prediction": bool(p >= threshold),
            "risk_band": _risk_band(p),
            "decision_threshold": threshold,
            "model_version": meta.get("model_version", "unknown"),
        })
    return out


def predict_one(features: dict) -> dict:
    return predict_frame(pd.DataFrame([features]))[0]
