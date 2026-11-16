"""Data and prediction drift against the training reference, and a clearly labelled drift SIMULATION."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from churn.data import FEATURES, NUMERIC

PSI_MODERATE, PSI_MAJOR = 0.10, 0.25  # industry rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 act
EPS = 1e-4


def profile(X, scores):
    """Reference distribution of every feature and of the model score, stored inside the model artifact."""
    prof = {"n": len(X), "features": {}}
    for f in FEATURES:
        if f in NUMERIC and X[f].nunique() > 2:
            edges = np.unique(np.quantile(X[f], np.linspace(0, 1, 11)))
            prof["features"][f] = {
                "kind": "numeric",
                "edges": edges.tolist(),
                "sample": X[f].sample(min(2000, len(X)), random_state=0).tolist(),
            }
        else:
            prof["features"][f] = {"kind": "categorical", "shares": X[f].astype(str).value_counts(normalize=True).to_dict()}
    prof["score"] = {
        "kind": "numeric",
        "edges": np.unique(np.quantile(scores, np.linspace(0, 1, 11))).tolist(),
        "sample": pd.Series(scores).sample(min(2000, len(scores)), random_state=0).tolist(),
    }
    return prof


def _shares(values, ref):
    if ref["kind"] == "numeric":
        inner = np.asarray(ref["edges"])[1:-1]
        counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
        expected = np.bincount(np.searchsorted(inner, ref["sample"], side="right"), minlength=len(counts)) / len(ref["sample"])
        return counts / counts.sum(), expected
    levels = list(ref["shares"])
    actual = pd.Series(values).astype(str).value_counts(normalize=True)
    new = actual.drop(levels, errors="ignore").sum()  # unseen levels get their own bucket
    return np.append(actual.reindex(levels, fill_value=0).to_numpy(), new), np.append([ref["shares"][k] for k in levels], 0)


def psi(actual, expected):
    a, e = np.clip(actual, EPS, None), np.clip(expected, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value):
    return "major" if value > PSI_MAJOR else "moderate" if value > PSI_MODERATE else "stable"


def compare(reference, X, scores):
    """PSI for every feature and the score, plus a KS (numeric) or chi-square (categorical) test."""
    rows = []
    for name, values in [(f, X[f].to_numpy()) for f in FEATURES] + [("model score", np.asarray(scores))]:
        ref = reference["score"] if name == "model score" else reference["features"][name]
        actual, expected = _shares(values, ref)
        if ref["kind"] == "numeric":
            p_value = ks_2samp(values, ref["sample"]).pvalue
        else:
            counts = np.vstack([actual * len(values), expected * reference["n"]])
            counts = counts[:, counts.sum(axis=0) > 0]
            p_value = chi2_contingency(counts)[1] if counts.shape[1] > 1 else 1.0
        value = psi(actual, expected)
        rows.append({"feature": name, "psi": round(value, 4), "p_value": float(f"{p_value:.3g}"), "status": status(value)})
    worst = max(r["psi"] for r in rows)
    return {"n": len(X), "status": status(worst), "features": sorted(rows, key=lambda r: -r["psi"])}


# SIMULATION ONLY. The dataset is a single static snapshot, so these scenarios synthesise plausible shifts of the
# held-out customers to show the monitor reacting. They are not observations of real drift.
def scenarios(test, seed=0):
    rng = np.random.default_rng(seed)
    t = test.copy()
    short = t[t["tenure"] <= 12]
    out = {"No shift (held-out customers as they are)": (t, True)}
    out["Acquisition surge: new customers over-represented 3x"] = (
        pd.concat([t, short, short]).sample(frac=1, random_state=seed).reset_index(drop=True),
        True,
    )
    s = t.copy()
    s["MonthlyCharges"] = (s["MonthlyCharges"] * 1.15).round(2)
    out["Price rise: every bill +15%"] = (s, False)
    s = t.copy()
    mask = (s["InternetService"] == "DSL") & (rng.random(len(s)) < 0.5)
    s.loc[mask, "InternetService"] = "Fiber optic"
    s.loc[mask, "MonthlyCharges"] = s.loc[mask, "MonthlyCharges"] + 20
    out["Fibre migration: half of DSL customers moved to fibre (+$20/mo)"] = (s, False)
    s = t.copy()
    mask = (s["Contract"] != "Month-to-month") & (rng.random(len(s)) < 0.4)
    s.loc[mask, "Contract"] = "Month-to-month"
    out["Contract mix: 40% of term contracts lapse to month-to-month"] = (s, False)
    return out


def simulate(model, test, evaluate):
    """Run every scenario through the monitor. Labels are only meaningful when a scenario resamples real customers;
    when features are edited the true outcome is unknown, so performance is not reported."""
    results = []
    for name, (frame, labels_valid) in scenarios(test).items():
        p = model.predict_proba(frame)
        report = compare(model.reference, frame, p)
        results.append(
            {
                "scenario": name,
                "status": report["status"],
                "n": len(frame),
                "mean_score": round(float(p.mean()), 4),
                "performance": evaluate(frame, p) if labels_valid else None,
                "features": report["features"][:6],
            }
        )
    return {
        "simulation": True,
        "note": "SIMULATED shifts of the held-out customers. The dataset is a static snapshot; no real drift was observed.",
        "thresholds": {"moderate": PSI_MODERATE, "major": PSI_MAJOR},
        "scenarios": results,
    }
