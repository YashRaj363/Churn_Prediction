"""Data loading and reference-profile construction.

Primary path: download the IBM Telco Customer Churn CSV.
Fallback: deterministically generate a Telco-like synthetic dataset so the
pipeline is fully runnable offline.

The reference profile captures per-feature distributions from the training data
and is later used by the monitoring layer as the drift baseline.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script (python src/data/load.py) or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw Telco columns into the modelling contract."""
    df = df.copy()

    # TotalCharges arrives as a string with blanks for tenure-0 customers.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # SeniorCitizen is 0/1 already; make sure it's numeric.
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # Target: Yes/No -> 1/0
    if config.TARGET in df.columns and df[config.TARGET].dtype == object:
        df[config.TARGET] = (df[config.TARGET].str.strip().str.lower() == "yes").astype(int)

    return df


def download_telco() -> pd.DataFrame:
    """Fetch the IBM Telco CSV. Raises on any network/parse failure."""
    req = urllib.request.Request(config.TELCO_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    from io import StringIO

    df = pd.read_csv(StringIO(raw))
    if config.TARGET not in df.columns:
        raise ValueError("Downloaded CSV missing target column")
    return _clean(df)


def generate_synthetic(n: int = 7043, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate a Telco-like dataset with realistic churn correlations."""
    rng = np.random.default_rng(seed)

    def choice(options, p=None):
        return rng.choice(options, size=n, p=p)

    contract = choice(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])
    tenure = np.clip(rng.gamma(2.0, 12.0, n).astype(int), 0, 72)
    internet = choice(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])
    monthly = np.round(
        20 + (internet == "DSL") * 25 + (internet == "Fiber optic") * 55
        + rng.normal(0, 8, n), 2
    ).clip(18, 120)
    total = np.round(monthly * np.maximum(tenure, 1) * rng.uniform(0.85, 1.05, n), 2)

    def dep(yes_p):
        return choice(["Yes", "No"], [yes_p, 1 - yes_p])

    df = pd.DataFrame({
        config.ID_COL: [f"SYN-{i:05d}" for i in range(n)],
        "gender": choice(["Male", "Female"]),
        "SeniorCitizen": choice([0, 1], [0.84, 0.16]),
        "Partner": dep(0.48),
        "Dependents": dep(0.30),
        "tenure": tenure,
        "PhoneService": choice(["Yes", "No"], [0.90, 0.10]),
        "MultipleLines": choice(["Yes", "No", "No phone service"], [0.42, 0.48, 0.10]),
        "InternetService": internet,
        "OnlineSecurity": dep(0.29),
        "OnlineBackup": dep(0.34),
        "DeviceProtection": dep(0.34),
        "TechSupport": dep(0.29),
        "StreamingTV": dep(0.38),
        "StreamingMovies": dep(0.38),
        "Contract": contract,
        "PaperlessBilling": dep(0.59),
        "PaymentMethod": choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            [0.34, 0.23, 0.22, 0.21],
        ),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    })

    # Churn probability driven by known risk factors.
    logit = (
        -1.6
        + (contract == "Month-to-month") * 1.4
        - (contract == "Two year") * 1.1
        + (internet == "Fiber optic") * 0.7
        - tenure / 36.0
        + (df["PaymentMethod"] == "Electronic check") * 0.5
        + (df["SeniorCitizen"] == 1) * 0.3
        + rng.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df[config.TARGET] = (rng.uniform(0, 1, n) < prob).astype(int)
    return _clean(df)


def build_reference_profile(df: pd.DataFrame) -> dict:
    """Summarise training distributions used as the drift baseline."""
    profile: dict = {"n_rows": int(len(df)), "numeric": {}, "categorical": {}}

    for col in config.NUMERIC_FEATURES:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        # Fixed bin edges (deciles) so live data can be bucketed identically.
        quantiles = np.quantile(s, np.linspace(0, 1, 11)).tolist()
        # Deduplicate edges to avoid zero-width bins.
        edges = sorted(set(quantiles))
        profile["numeric"][col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "bin_edges": edges,
            "sample": s.sample(min(len(s), 500), random_state=config.RANDOM_STATE).tolist(),
        }

    for col in config.CATEGORICAL_FEATURES:
        counts = df[col].astype(str).value_counts(normalize=True)
        profile["categorical"][col] = {str(k): float(v) for k, v in counts.items()}

    profile["target_rate"] = float(df[config.TARGET].mean())
    return profile


def load_data(force_synthetic: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source) where source is 'telco' or 'synthetic'."""
    if not force_synthetic:
        try:
            df = download_telco()
            df.to_csv(config.RAW_CSV, index=False)
            return df, "telco"
        except Exception as exc:  # noqa: BLE001
            print(f"[load] download failed ({exc!r}); using synthetic fallback")

    df = generate_synthetic()
    df.to_csv(config.RAW_CSV, index=False)
    return df, "synthetic"


def main() -> None:
    force = "--synthetic" in sys.argv
    df, source = load_data(force_synthetic=force)
    profile = build_reference_profile(df)
    with open(config.REFERENCE_PROFILE, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)
    print(f"[load] source={source} rows={len(df)} churn_rate={df[config.TARGET].mean():.3f}")
    print(f"[load] wrote {config.RAW_CSV.name} and {config.REFERENCE_PROFILE.name}")


if __name__ == "__main__":
    main()
