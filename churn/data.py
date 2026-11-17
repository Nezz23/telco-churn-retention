"""Dataset schema, loading, fingerprinting and the leakage-proof three-way split."""

import hashlib
from pathlib import Path

import pandas as pd

DATA = Path("data/telco.csv")
SEED = 42
# TotalCharges is left out: it is ~tenure x MonthlyCharges (r=0.9996), so it adds no signal and splits explanation
# credit with tenure. It stays in the frame for revenue reporting only.
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges"]
CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL
# Each customer's split comes from a hash of their ID, so it never changes as new rows arrive: a challenger is never
# scored on a customer the champion trained on, and each stage below only ever sees its own customers.
SPLITS = {"train": (0, 60), "calibration": (60, 80), "test": (80, 100)}  # train: model selection; calibration:
# calibrator + decision threshold; test: touched once, for the final report.


def load(path=DATA):
    df = pd.read_csv(path)
    # 11 brand-new customers (tenure 0) have a blank TotalCharges: they have not been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["churned"] = (df["Churn"] == "Yes").astype(int)
    return df


def fingerprint(path=DATA):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bucket(customer_id: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{customer_id}".encode()).hexdigest(), 16) % 100


def split(df):
    b = df["customerID"].map(bucket)
    return {name: df[(b >= lo) & (b < hi)].reset_index(drop=True) for name, (lo, hi) in SPLITS.items()}
