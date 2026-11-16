import json

import numpy as np
import pytest
from conftest import LOYAL, RISKY
from fastapi.testclient import TestClient

from churn import registry
from churn.api import app

client = TestClient(app)


def predict(rows):
    r = client.post("/predict", json=rows)
    assert r.status_code == 200, r.text
    return r.json()


def test_flight_risk_gets_an_offer_and_loyal_customer_does_not():
    risky, loyal = predict([RISKY, LOYAL])
    assert risky["churn_probability"] > 0.6 and risky["send_offer"]
    assert loyal["churn_probability"] < 0.1 and not loyal["send_offer"]


def test_cheap_plan_is_not_worth_the_offer_even_at_high_risk():
    # $20/month x 12 x 30% save rate = $72 at best, below the $100 offer cost.
    (cheap,) = predict([RISKY | {"MonthlyCharges": 20.0}])
    assert not cheap["send_offer"] and cheap["expected_value"] < 0


def test_reasons_name_what_makes_the_customer_risky():
    (risky,) = predict([RISKY])
    names = {r["feature"] for r in risky["top_reasons"]}
    assert names & {"Contract", "tenure"}
    assert all(r["contribution"] > 0 for r in risky["top_reasons"])


def test_raw_billing_export_rows_are_accepted_and_ids_echoed(csv_rows):
    out = predict(csv_rows[:50])
    assert [o["customerID"] for o in out] == [r["customerID"] for r in csv_rows[:50]]


def test_offers_go_mostly_to_customers_who_actually_churned(csv_rows):
    churners = [r for r in csv_rows if r["Churn"] == "Yes"][:300]
    stayers = [r for r in csv_rows if r["Churn"] == "No"][:300]
    assert sum(o["send_offer"] for o in predict(churners)) > 3 * sum(o["send_offer"] for o in predict(stayers))


@pytest.mark.parametrize("bad", [{"Contract": "Forever"}, {"tenure": -1}, {"MonthlyCharges": "lots"}])
def test_rejects_bad_input(bad):
    assert client.post("/predict", json=[RISKY | bad]).status_code == 422


def test_rejects_combinations_that_cannot_exist():
    assert client.post("/predict", json=[RISKY | {"InternetService": "No"}]).status_code == 422  # add-ons still "No"
    assert client.post("/predict", json=[RISKY | {"PhoneService": "No"}]).status_code == 422  # MultipleLines still "No"


def test_non_finite_numbers_are_a_clean_422_not_a_crash():
    body = json.dumps([RISKY]).replace('"MonthlyCharges": 95.0', '"MonthlyCharges": NaN')
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422 and "input" not in r.json()["detail"][0]


def test_rejects_empty_batch():
    assert client.post("/predict", json=[]).status_code == 422


def test_explanation_adds_up_to_the_raw_score():
    e = client.post("/explain", json=RISKY).json()
    log_odds = e["base_log_odds"] + sum(c["contribution"] for c in e["contributions"])
    assert abs(1 / (1 + np.exp(-log_odds)) - e["raw_probability"]) < 1e-3
    assert e["send_offer"] == (e["expected_value"] > e["margin"])


def test_drift_check_is_quiet_on_real_customers_and_loud_on_shifted_ones(csv_rows):
    assert client.post("/drift", json=csv_rows[:1000]).json()["status"] == "stable"
    shifted = [r | {"MonthlyCharges": r["MonthlyCharges"] * 1.5, "Contract": "Month-to-month"} for r in csv_rows[:1000]]
    assert client.post("/drift", json=shifted).json()["status"] == "major"


def test_health_reports_the_registered_champion():
    h = client.get("/health").json()
    ix = registry.index()
    assert h["model_version"] == ix["champion"]
    assert h["artifact_sha256"] == registry.sha256(registry.ROOT / ix["champion"] / "model.joblib")


def test_ops_records_latency_for_scoring_routes():
    predict([RISKY])
    route = client.get("/ops").json()["routes"]["/predict"]
    assert route["requests"] >= 1 and route["p50_ms"] > 0


def test_home_page_serves_the_app():
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
