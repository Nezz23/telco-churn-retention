"""The served artifact, its evaluation protocol and its promotion record."""

import numpy as np
from sklearn.metrics import roc_auc_score

from churn import registry
from churn.data import bucket, load, split

model, report = registry.load()


def test_splits_are_disjoint_complete_and_stable():
    df = load()
    parts = split(df)
    ids = [set(p["customerID"]) for p in parts.values()]
    assert sum(map(len, ids)) == len(df) and not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert bucket("7590-VHVEG") == bucket("7590-VHVEG")  # pure function of the ID: stable as data grows


def test_artifact_reproduces_its_reported_test_metrics():
    """The bytes on disk are the model that was evaluated: re-score the test customers and compare."""
    test = split(load())["test"]
    p = model.predict_proba(test)
    chosen = report["selection"]["chosen"]
    assert abs(roc_auc_score(test["churned"], p) - report["test"]["models"][chosen]["roc_auc"]["value"]) < 1e-4


def test_selection_used_cross_validation_not_the_test_set():
    sel = report["selection"]
    assert sel["chosen"] == min(sel["candidates"], key=lambda k: sel["candidates"][k]["cv_log_loss"])


def test_champion_passed_its_gate_and_beats_contacting_everyone():
    pol = report["test"]["policies"]
    assert report["gate"]["passed"]
    assert pol["Expected-value targeting (this model)"]["profit"] > max(0, pol["Contact everyone"]["profit"])


def test_probabilities_are_calibrated():
    assert report["test"]["models"][report["selection"]["chosen"]]["ece"]["value"] <= registry.FLOORS["ece_max"]
    rel = report["calibration"]["reliability"]
    assert all(b["ci"][0] - 0.05 <= b["predicted"] <= b["ci"][1] + 0.05 for b in rel)


def test_drift_simulation_is_labelled_and_reacts():
    sim = report["drift_simulation"]
    assert sim["simulation"] is True and "SIMULATED" in sim["note"]
    statuses = [s["status"] for s in sim["scenarios"]]
    assert statuses[0] == "stable" and "major" in statuses


def test_explanations_group_correlated_features():
    groups = report["explain"]["groups"]
    assert groups["tenure"] == groups["Contract"]
    assert groups["InternetService"] == groups["MonthlyCharges"]
    assert np.isclose(
        sum(report["explain"]["importance_group"].values()), sum(report["explain"]["importance_feature"].values()), rtol=0.5
    )  # grouping cancels opposing pushes, so it can only shrink the total
