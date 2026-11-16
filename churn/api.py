"""FastAPI service: scoring, explanations, batch drift checks, model evidence and live operational metrics."""

import json
import time
from collections import deque
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from churn import drift, explain, registry

YesNo = Literal["Yes", "No"]
Extra = Literal["Yes", "No", "No internet service"]
WEB = Path(__file__).resolve().parent.parent / "web"


class Customer(BaseModel):
    """One customer as the billing system records them. Unknown extra columns (e.g. Churn, TotalCharges) are ignored."""

    customerID: str | None = None
    SeniorCitizen: Literal[0, 1]
    tenure: int = Field(ge=0, le=120)
    MonthlyCharges: float = Field(ge=0, le=1000)
    gender: Literal["Male", "Female"]
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Extra
    OnlineBackup: Extra
    DeviceProtection: Extra
    TechSupport: Extra
    StreamingTV: Extra
    StreamingMovies: Extra
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]

    @model_validator(mode="after")
    def consistent(self):
        """Reject combinations that never occur in the data, so the model never scores what it has not seen."""
        addons = [
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies,
        ]
        no_internet = self.InternetService == "No"
        if any((a == "No internet service") != no_internet for a in addons):
            raise ValueError("add-ons must be 'No internet service' exactly when InternetService is 'No'")
        if (self.MultipleLines == "No phone service") != (self.PhoneService == "No"):
            raise ValueError("MultipleLines must be 'No phone service' exactly when PhoneService is 'No'")
        return self


class Reason(BaseModel):
    feature: str
    value: str | int | float
    contribution: float = Field(description="push on the churn log-odds (SHAP)")
    group: str = Field(description="correlated-feature group this feature explains jointly with")


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float
    expected_value: float = Field(description="expected $ from sending the offer")
    send_offer: bool
    top_reasons: list[Reason]


class Explanation(BaseModel):
    churn_probability: float
    raw_probability: float = Field(description="model score before calibration")
    expected_value: float
    send_offer: bool
    margin: float
    base_log_odds: float
    contributions: list[Reason] = Field(description="every feature, largest absolute push first")


model, report = registry.load()
STARTED = time.time()
LATENCY: dict[str, deque] = {}
COUNTS: dict[str, list[int]] = {}  # route -> [requests, errors]

app = FastAPI(
    title="Telco Churn Retention API",
    version=report["version"],
    description="Calibrated churn risk, expected-value retention decisions and SHAP reasons.",
)


@app.exception_handler(RequestValidationError)
async def invalid(request: Request, exc: RequestValidationError):
    # Drop the echoed input: it can hold NaN/Infinity, which JSON cannot carry (the default handler would 500).
    errors = [{k: v for k, v in e.items() if k in ("loc", "msg", "type")} for e in exc.errors()]
    return JSONResponse({"detail": errors}, status_code=422)


@app.middleware("http")
async def record(request: Request, call_next):
    start, status = time.perf_counter(), 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:  # counts server errors too, not only responses that made it back
        route = request.scope.get("route")
        if route is not None and request.url.path.startswith(("/predict", "/explain", "/drift")):
            LATENCY.setdefault(route.path, deque(maxlen=5000)).append((time.perf_counter() - start) * 1000)
            counts = COUNTS.setdefault(route.path, [0, 0])
            counts[0] += 1
            counts[1] += status >= 400


def frame(customers):
    return pd.DataFrame([c.model_dump() for c in customers])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": report["version"],
        "artifact_sha256": report["artifact_sha256"],
        "family": report["family"],
        "calibrator": report["calibration"]["method"],
        "uptime_seconds": round(time.time() - STARTED, 1),
    }


@app.post("/predict", response_model=list[Prediction])
def predict(customers: Annotated[list[Customer], Body(min_length=1, max_length=10_000)]):
    X = frame(customers)
    p, ev, send = model.decide(X)
    contribs, _ = model.contributions(X)
    reasons = explain.reasons(contribs, X, model.groups)
    return [
        {
            "customerID": c,
            "churn_probability": round(float(a), 4),
            "expected_value": round(float(v), 2),
            "send_offer": bool(s),
            "top_reasons": r,
        }
        for c, a, v, s, r in zip(X["customerID"], p, ev, send, reasons, strict=True)
    ]


@app.post("/explain", response_model=Explanation)
def explain_one(customer: Customer):
    X = frame([customer])
    p, ev, send = model.decide(X)
    contribs, base = model.contributions(X)
    row = contribs.iloc[0].reindex(contribs.iloc[0].abs().sort_values(ascending=False).index)
    return {
        "churn_probability": round(float(p[0]), 4),
        "raw_probability": round(float(model.raw(X)[0]), 4),
        "expected_value": round(float(ev[0]), 2),
        "send_offer": bool(send[0]),
        "margin": model.margin,
        "base_log_odds": round(float(base[0]), 4),
        "contributions": [
            {
                "feature": f,
                "value": explain._plain(X[f].iloc[0]),
                "contribution": round(float(v), 4),
                "group": model.groups.get(f, f),
            }
            for f, v in row.items()
        ],
    }


@app.post("/drift")
def drift_check(customers: Annotated[list[Customer], Body(min_length=30, max_length=10_000)]):
    """Compare a batch against the model's reference customers (PSI per feature and on the score)."""
    X = frame(customers)
    return drift.compare(model.reference, X, model.predict_proba(X))


@app.get("/evidence")
def evidence():
    """The champion's full evaluation report: selection, calibration, test metrics, subgroups, errors, SHAP, drift."""
    return report


@app.get("/ops")
def ops():
    routes = {}
    for path, lat in LATENCY.items():
        a = np.fromiter(lat, float)
        routes[path] = {
            "requests": COUNTS[path][0],
            "errors": COUNTS[path][1],
            "window": len(a),
            **{f"p{q}_ms": round(float(np.percentile(a, q)), 2) for q in (50, 95, 99)},
        }
    ix = registry.index()
    versions = {}
    for v in {e["version"] for e in ix["events"]}:
        path = registry.ROOT / v / "report.json"
        if path.exists():
            r = json.loads(path.read_text())
            m = r["test"]["models"][r["selection"]["chosen"]]
            versions[v] = {
                "created_at": r["created_at"],
                "family": r["family"],
                "data_sha256": r["data_sha256"][:12],
                "roc_auc": m["roc_auc"]["value"],
                "ece": m["ece"]["value"],
                "profit": m["profit"]["value"],
                "gate_passed": r["gate"]["passed"],
            }
    bench = Path("reports/benchmark.json")
    return {
        "uptime_seconds": round(time.time() - STARTED, 1),
        "routes": routes,
        "registry": {"champion": ix["champion"], "events": ix["events"], "versions": versions},
        "gate": report["gate"],
        "drift_simulation": report["drift_simulation"],
        "benchmark": json.loads(bench.read_text()) if bench.exists() else None,
    }


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB / "index.html")


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")
