"""Structured request/prediction logging to a JSONL file.

Each scored request appends one line so the file can be tailed, replayed for
drift analysis, or shipped to a log sink. Writes are best-effort: a logging
failure must never break a prediction response.
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402

_lock = threading.Lock()


def log_prediction(
    *,
    features: dict,
    result: dict,
    latency_ms: float,
    request_id: str,
    status: str = "ok",
    error: str | None = None,
) -> None:
    """Append a single prediction event as one JSON line."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "features": features,
        "churn_probability": result.get("churn_probability") if result else None,
        "churn_prediction": result.get("churn_prediction") if result else None,
        "model_version": result.get("model_version") if result else None,
        "error": error,
    }
    try:
        line = json.dumps(record, default=str)
        with _lock:
            with open(config.REQUEST_LOG, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 - logging must not raise
        print(f"[logger] failed to write log: {exc!r}")


def read_recent(limit: int = 1000) -> list[dict]:
    """Return up to `limit` most recent successful prediction records."""
    if not config.REQUEST_LOG.exists():
        return []
    rows: list[dict] = []
    with open(config.REQUEST_LOG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]
