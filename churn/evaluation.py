"""Held-out evaluation with paired-bootstrap uncertainty, calibration, subgroups and error analysis."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

B = 2000  # bootstrap resamples


def ece(y, p, bins=10):
    """Expected calibration error: |observed - predicted| per equal-width bin, weighted by the bin's share."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(np.mean(idx == b) * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if np.any(idx == b)))


METRICS = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "log_loss": lambda y, p: log_loss(y, p, labels=[0, 1]),
    "brier": brier_score_loss,
    "ece": ece,
}


def resamples(n, b=B, seed=0):
    return np.random.default_rng(seed).integers(0, n, (b, n))


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def compare(y, probs: dict, decisions: dict, econ, monthly, reference):
    """Point estimate and 95% CI of every metric for every model, plus the paired difference to `reference`
    (same resampled customers for both models, so the comparison is not swamped by sample noise)."""
    idx = resamples(len(y))
    out = {}
    for name, p in probs.items():
        gain = econ.realised(decisions[name], y, monthly)
        stats = {k: np.array([f(y[i], p[i]) for i in idx]) for k, f in METRICS.items()}
        stats["profit"] = gain[idx].sum(axis=1)
        out[name] = {
            k: {"value": round(float(METRICS[k](y, p) if k in METRICS else gain.sum()), 4), "ci": ci(v)} for k, v in stats.items()
        }
        out[name]["_draws"] = stats
    for name in probs:
        if name != reference:
            out[name]["vs_" + reference] = {
                k: {
                    "diff": round(float(out[name][k]["value"] - out[reference][k]["value"]), 4),
                    "ci": ci(out[name]["_draws"][k] - out[reference]["_draws"][k]),
                }
                for k in out[name]["_draws"]
            }
    for name in probs:
        del out[name]["_draws"]
    return out


def reliability(y, p, bins=10):
    """Quantile-binned reliability curve with a Wilson 95% interval on each bin's observed rate."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        n, k = int(m.sum()), float(y[m].sum())
        z, phat = 1.96, k / n
        centre, half = (
            (phat + z * z / (2 * n)) / (1 + z * z / n),
            z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n),
        )
        rows.append(
            {
                "predicted": round(float(p[m].mean()), 4),
                "observed": round(phat, 4),
                "n": n,
                "ci": [round(centre - half, 4), round(centre + half, 4)],
            }
        )
    return rows


def profit_curve(y, ev, econ, monthly, points=60):
    """Cumulative realised profit when contacting the top-k customers ranked by expected value."""
    order = np.argsort(-ev)
    cum = np.cumsum(econ.realised(np.ones(len(y)), y[order], monthly[order]))
    ks = np.unique(np.linspace(0, len(y), points).astype(int))
    return [
        {
            "contacted": int(k),
            "profit": round(float(cum[k - 1]) if k else 0.0, 0),
            "expected_value_cutoff": round(float(ev[order][k - 1]), 2) if k else None,
        }
        for k in ks
    ]


def subgroups(frame, y, p, target, econ, b=500):
    """Per-segment discrimination, calibration and policy outcomes. Small segments get wide CIs; that is the point."""
    frame = frame.assign(
        tenure_band=pd.cut(frame["tenure"], [-1, 12, 24, 48, 72], labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]).astype(
            str
        ),
        SeniorCitizen=frame["SeniorCitizen"].map({0: "No", 1: "Yes"}),
    )
    rng = np.random.default_rng(1)
    rows = []
    for col in [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "tenure_band",
        "SeniorCitizen",
        "gender",
        "Partner",
        "Dependents",
        "PaperlessBilling",
    ]:
        for level, m in frame.groupby(col).indices.items():
            ys, ps, ts = y[m], p[m], target[m]
            auc = roc_auc_score(ys, ps) if 0 < ys.sum() < len(ys) else None
            draws = []
            if auc is not None:
                for _ in range(b):
                    i = rng.integers(0, len(m), len(m))
                    if 0 < ys[i].sum() < len(i):
                        draws.append(roc_auc_score(ys[i], ps[i]))
            rows.append(
                {
                    "column": col,
                    "level": str(level),
                    "n": len(m),
                    "churn_rate": round(float(ys.mean()), 4),
                    "mean_predicted": round(float(ps.mean()), 4),
                    "calibration_gap": round(float(ps.mean() - ys.mean()), 4),
                    "roc_auc": round(auc, 4) if auc is not None else None,
                    "roc_auc_ci": ci(draws) if draws else None,
                    "contacted_share": round(float(ts.mean()), 4),
                    "recall": round(float(ts[ys == 1].mean()), 4) if ys.sum() else None,
                    "precision": round(float(ys[ts].mean()), 4) if ts.sum() else None,
                }
            )
    return rows


def error_analysis(frame, y, p, target, ev, econ, reasons, top=12):
    """Where the policy's mistakes cost money, and which segments over-produce them."""
    monthly = frame["MonthlyCharges"].to_numpy()
    fp, fn = target & (y == 0), ~target & (y == 1)
    missed_value = econ.save_rate * econ.horizon * monthly  # what saving a missed churner would have returned
    fn_worth = fn & (missed_value > econ.offer_cost)  # missed churners an offer would have paid off on
    segments = []
    for col in ["Contract", "InternetService", "PaymentMethod", "tenure_band"]:
        values = (
            frame[col]
            if col in frame
            else pd.cut(frame["tenure"], [-1, 12, 24, 48, 72], labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]).astype(str)
        )
        for level in values.unique():
            m = (values == level).to_numpy()
            segments.append(
                {
                    "segment": f"{col}={level}",
                    "n": int(m.sum()),
                    "false_offer_rate": round(float(fp[m].mean()), 4),
                    "missed_churner_rate": round(float(fn_worth[m].mean()), 4),
                    "cost": round(float(fp[m].sum() * econ.offer_cost + (missed_value[fn_worth & m] - econ.offer_cost).sum()), 0),
                }
            )
    cost = np.where(fp, econ.offer_cost, 0) + np.where(fn_worth, missed_value - econ.offer_cost, 0)
    worst = np.argsort(-cost)[:top]
    return {
        "wasted_offers": {"count": int(fp.sum()), "cost": round(float(fp.sum() * econ.offer_cost), 0)},
        "missed_profitable_churners": {
            "count": int(fn_worth.sum()),
            "forgone_profit": round(float((missed_value[fn_worth] - econ.offer_cost).sum()), 0),
        },
        "missed_unprofitable_churners": int((fn & ~fn_worth).sum()),
        "segments": sorted(segments, key=lambda s: -s["cost"]),
        "costliest": [
            {
                "kind": "wasted offer" if fp[i] else "missed churner",
                "probability": round(float(p[i]), 4),
                "expected_value": round(float(ev[i]), 2),
                "cost": round(float(cost[i]), 2),
                "monthly": float(monthly[i]),
                "tenure": int(frame["tenure"].iloc[i]),
                "contract": frame["Contract"].iloc[i],
                "reasons": reasons[i],
            }
            for i in worst
            if cost[i] > 0
        ],
    }


def sensitivity(y, p, monthly, econ_cls, save_rates=(0.1, 0.2, 0.3, 0.4, 0.5), costs=(50, 100, 150, 200), b=500):
    """Realised profit of re-optimised EV targeting vs contacting everyone across assumption grids, with CIs."""
    idx = resamples(len(y), b, seed=2)
    rows = []
    for s in save_rates:
        for c in costs:
            e = econ_cls(offer_cost=float(c), save_rate=s)
            model = e.realised(e.expected_value(p, monthly) > 0, y, monthly)
            everyone = e.realised(np.ones(len(y)), y, monthly)
            rows.append(
                {
                    "save_rate": s,
                    "offer_cost": c,
                    "model": round(float(model.sum()), 0),
                    "model_ci": ci(model[idx].sum(axis=1)),
                    "everyone": round(float(everyone.sum()), 0),
                    "contacted": int((e.expected_value(p, monthly) > 0).sum()),
                }
            )
    return rows
