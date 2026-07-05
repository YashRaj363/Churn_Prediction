"""Train the churn model.

Builds an sklearn Pipeline (impute + encode + scale -> classifier), compares a
couple of candidate models by cross-validated ROC-AUC, refits the winner on the
full training split, evaluates on a hold-out, and persists:

  models/churn_model.joblib   the fitted Pipeline (preprocessing + model)
  models/metadata.json        features, metrics, threshold, provenance

Run: python src/train.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.data.load import build_reference_profile, load_data  # noqa: E402


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }


def main() -> None:
    # Reuse existing raw data if present, else download/generate.
    if config.RAW_CSV.exists():
        df = pd.read_csv(config.RAW_CSV)
        source = "cached"
    else:
        df, source = load_data()

    # Ensure the reference profile exists (drift baseline).
    if not config.REFERENCE_PROFILE.exists():
        with open(config.REFERENCE_PROFILE, "w", encoding="utf-8") as fh:
            json.dump(build_reference_profile(df), fh, indent=2)

    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    preprocessor = build_preprocessor()

    # Model selection by 5-fold CV ROC-AUC on the training split.
    results = {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        results[name] = float(scores.mean())
        print(f"[train] {name:22s} cv_roc_auc={scores.mean():.4f} (+/-{scores.std():.4f})")

    best_name = max(results, key=results.get)
    print(f"[train] selected: {best_name}")

    best_pipe = Pipeline([("prep", build_preprocessor()), ("model", candidate_models()[best_name])])
    best_pipe.fit(X_train, y_train)

    # Hold-out evaluation.
    proba = best_pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= config.DECISION_THRESHOLD).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    print("[train] hold-out metrics:")
    for k, v in metrics.items():
        print(f"        {k:10s} {v:.4f}")
    print(classification_report(y_test, preds, target_names=["stay", "churn"]))

    # Persist artifacts.
    joblib.dump(best_pipe, config.MODEL_PATH)

    metadata = {
        "model_type": best_name,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target": config.TARGET,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "decision_threshold": config.DECISION_THRESHOLD,
        "cv_roc_auc_by_model": results,
        "holdout_metrics": metrics,
        "class_balance": {"churn": float(y.mean()), "stay": float(1 - y.mean())},
    }
    with open(config.MODEL_METADATA, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"[train] saved {config.MODEL_PATH.name} and {config.MODEL_METADATA.name}")


if __name__ == "__main__":
    main()
