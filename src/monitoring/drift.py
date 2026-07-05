"""Data drift detection against the training reference profile.

Numeric features  -> Population Stability Index (PSI) over the reference bin
                     edges, plus a two-sample Kolmogorov-Smirnov test.
Categorical feats -> PSI over category frequencies.

A feature is flagged as drifting when PSI >= PSI_ALERT (or, for numeric
features, when the KS p-value < KS_PVALUE_ALERT). A dataset-level drift flag is
raised when the fraction of drifting features exceeds
DRIFT_FEATURE_FRACTION_ALERT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402

_EPS = 1e-6


def _load_profile() -> dict | None:
    if not config.REFERENCE_PROFILE.exists():
        return None
    with open(config.REFERENCE_PROFILE, encoding="utf-8") as fh:
        return json.load(fh)


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI between two probability distributions (already normalised)."""
    expected = np.clip(expected, _EPS, None)
    actual = np.clip(actual, _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_drift(col: str, ref: dict, live: pd.Series) -> dict:
    edges = np.array(ref["bin_edges"], dtype=float)
    live = pd.to_numeric(live, errors="coerce").dropna().to_numpy()

    # Expected distribution over the fixed reference bins comes from the stored
    # reference sample; live distribution uses the same edges.
    ref_sample = np.array(ref["sample"], dtype=float)
    exp_counts, _ = np.histogram(ref_sample, bins=edges)
    act_counts, _ = np.histogram(live, bins=edges)
    exp = exp_counts / max(exp_counts.sum(), 1)
    act = act_counts / max(act_counts.sum(), 1)
    psi = _psi(exp, act)

    ks_p = 1.0
    if len(live) > 1 and len(ref_sample) > 1:
        ks_p = float(stats.ks_2samp(ref_sample, live).pvalue)

    drifting = psi >= config.PSI_ALERT or ks_p < config.KS_PVALUE_ALERT
    return {
        "type": "numeric",
        "psi": round(psi, 4),
        "ks_pvalue": round(ks_p, 4),
        "live_mean": round(float(live.mean()), 4) if len(live) else None,
        "ref_mean": round(float(ref["mean"]), 4),
        "drifting": bool(drifting),
        "severity": _severity(psi),
    }


def _categorical_drift(col: str, ref: dict, live: pd.Series) -> dict:
    live_freq = live.astype(str).value_counts(normalize=True).to_dict()
    categories = set(ref) | set(live_freq)
    exp = np.array([ref.get(c, 0.0) for c in categories])
    act = np.array([live_freq.get(c, 0.0) for c in categories])
    psi = _psi(exp, act)
    return {
        "type": "categorical",
        "psi": round(psi, 4),
        "drifting": bool(psi >= config.PSI_ALERT),
        "severity": _severity(psi),
    }


def _severity(psi: float) -> str:
    if psi >= config.PSI_ALERT:
        return "major"
    if psi >= config.PSI_WARN:
        return "moderate"
    return "none"


def compute_drift(live_df: pd.DataFrame) -> dict:
    """Compare a batch of live feature rows to the reference profile."""
    profile = _load_profile()
    if profile is None:
        return {"status": "no_reference_profile"}

    if len(live_df) < config.DRIFT_MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "n_samples": int(len(live_df)),
            "min_required": config.DRIFT_MIN_SAMPLES,
        }

    features: dict[str, dict] = {}
    for col in config.NUMERIC_FEATURES:
        if col in live_df.columns and col in profile["numeric"]:
            features[col] = _numeric_drift(col, profile["numeric"][col], live_df[col])
    for col in config.CATEGORICAL_FEATURES:
        if col in live_df.columns and col in profile["categorical"]:
            features[col] = _categorical_drift(col, profile["categorical"][col], live_df[col])

    drifting = [c for c, r in features.items() if r["drifting"]]
    frac = len(drifting) / max(len(features), 1)
    return {
        "status": "ok",
        "n_samples": int(len(live_df)),
        "n_features_checked": len(features),
        "n_features_drifting": len(drifting),
        "drifting_features": drifting,
        "dataset_drift": bool(frac >= config.DRIFT_FEATURE_FRACTION_ALERT),
        "drift_fraction": round(frac, 3),
        "features": features,
    }
