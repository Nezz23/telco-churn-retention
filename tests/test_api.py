import json

from fastapi.testclient import TestClient

from app import app
from train import load

client = TestClient(app)
X, y, _ = load()
predict = lambda rows: client.post("/predict", json=rows).json()

# A textbook flight risk: new, month-to-month, fibre, no support, paying by electronic check.
RISKY = {"SeniorCitizen": 1, "tenure": 1, "MonthlyCharges": 95.0, "TotalCharges": 95.0, "gender": "Female",
         "Partner": "No", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
         "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
         "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
         "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"}
# A loyal customer: six years in, two-year contract, automatic payment.
LOYAL = RISKY | {"SeniorCitizen": 0, "tenure": 72, "TotalCharges": 6840.0, "Partner": "Yes", "Dependents": "Yes",
                 "OnlineSecurity": "Yes", "TechSupport": "Yes", "Contract": "Two year",
                 "PaymentMethod": "Credit card (automatic)"}


def test_flight_risk_gets_an_offer_and_loyal_customer_does_not():
    risky, loyal = predict([RISKY, LOYAL])
    assert risky["churn_probability"] > 0.6 and risky["send_offer"]
    assert loyal["churn_probability"] < 0.1 and not loyal["send_offer"]


def test_cheap_plan_is_not_worth_the_offer_even_at_high_risk():
    # $20/month x 12 x 30% save rate = $72 at best, below the $100 offer cost.
    (cheap,) = predict([RISKY | {"MonthlyCharges": 20.0, "TotalCharges": 20.0}])
    assert not cheap["send_offer"]


def test_reasons_name_what_makes_the_customer_risky():
    (risky,) = predict([RISKY])
    assert "Contract=Month-to-month" in risky["top_reasons"] or "tenure=1" in risky["top_reasons"]


def test_offers_go_mostly_to_customers_who_actually_churned():
    churners = sum(r["send_offer"] for r in predict(X[y == 1].head(300).to_dict("records")))
    stayers = sum(r["send_offer"] for r in predict(X[y == 0].head(300).to_dict("records")))
    assert churners > 3 * stayers


def test_rejects_bad_input():
    assert client.post("/predict", json=[RISKY | {"Contract": "Forever"}]).status_code == 422
    assert client.post("/predict", json=[RISKY | {"tenure": -1}]).status_code == 422


def test_retrained_model_still_beats_contacting_everyone():
    m = json.load(open("model/metrics.json"))
    assert m["test_roc_auc"] >= 0.80
    assert m["profit"]["model_expected_value"] > max(0, m["profit"]["contact_everyone"])
