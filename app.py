"""FastAPI churn-scoring service."""
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

from train import FEATURES, OFFER_COST, SAVE_RATE, expected_value, reasons

YesNo = Literal["Yes", "No"]
Extra = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    SeniorCitizen: Literal[0, 1]
    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
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


pipe = joblib.load("model/model.joblib")
app = FastAPI(title="Telco Churn Retention API")


@app.get("/health")
def health():
    return {"status": "ok", "offer_cost": OFFER_COST, "save_rate": SAVE_RATE}


@app.post("/predict")
def predict(customers: Annotated[list[Customer], Body(max_length=10_000)]):
    X = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    probs = pipe.predict_proba(X)[:, 1]
    values = expected_value(probs, X["MonthlyCharges"].to_numpy())
    return [{"churn_probability": round(float(p), 4), "expected_value": round(float(v), 2),
             "send_offer": bool(v > 0), "top_reasons": r}
            for p, v, r in zip(probs, values, reasons(pipe, X))]
