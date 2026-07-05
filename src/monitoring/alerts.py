"""Alerting: evaluate monitoring signals against thresholds and emit alerts.

Alerts are always written to logs/alerts.log. If ALERT_WEBHOOK_URL is set, a
compact JSON payload is also POSTed there (best-effort; failures are swallowed).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402


def _emit(alert: dict) -> None:
    alert = {"ts": datetime.now(timezone.utc).isoformat(), **alert}
    try:
        with open(config.ALERT_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[alerts] failed to write alert log: {exc!r}")

    if config.ALERT_WEBHOOK_URL:
        try:
            data = json.dumps({"text": f"[churn-api] {alert['level']}: {alert['message']}"}).encode()
            req = urllib.request.Request(
                config.ALERT_WEBHOOK_URL, data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:  # noqa: BLE001
            print(f"[alerts] webhook post failed: {exc!r}")


def evaluate(*, drift_report: dict | None, ops: dict | None) -> list[dict]:
    """Check signals against thresholds; emit and return any triggered alerts."""
    alerts: list[dict] = []

    if drift_report and drift_report.get("status") == "ok":
        if drift_report.get("dataset_drift"):
            alerts.append({
                "level": "critical",
                "kind": "data_drift",
                "message": (
                    f"Dataset drift detected: "
                    f"{drift_report['n_features_drifting']}/{drift_report['n_features_checked']} "
                    f"features drifting ({drift_report['drift_fraction']:.0%})"
                ),
                "drifting_features": drift_report.get("drifting_features", []),
            })

    if ops:
        p95 = ops.get("latency_p95_ms")
        if p95 is not None and p95 > config.LATENCY_MS_ALERT:
            alerts.append({
                "level": "warning",
                "kind": "latency",
                "message": f"p95 latency {p95:.0f}ms exceeds {config.LATENCY_MS_ALERT:.0f}ms",
            })
        err = ops.get("error_rate")
        if err is not None and err > config.ERROR_RATE_ALERT:
            alerts.append({
                "level": "warning",
                "kind": "error_rate",
                "message": f"error rate {err:.1%} exceeds {config.ERROR_RATE_ALERT:.1%}",
            })

    for a in alerts:
        _emit(a)
    return alerts
