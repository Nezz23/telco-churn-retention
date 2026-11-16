"""Train, calibrate, evaluate, explain and register a challenger; promote it only if it passes the gate.

Stages and the customers each one may see (hash split, see churn.data):
  1. model selection   train        Optuna 5-fold CV log loss, logistic vs XGBoost
  2. calibration       calibration  calibrator chosen and fitted
  3. threshold         calibration  EV margin tuned on cross-fitted calibrated probabilities
  4. final evaluation  test         touched once; every number in the report comes from here
"""

import argparse
import json
import platform
import subprocess
import time
from datetime import UTC, datetime

import numpy as np

from churn import drift, evaluation, explain, registry
from churn.data import FEATURES, fingerprint, load, split
from churn.economics import Economics, best_margin
from churn.models import XGB_DEFAULT, ChurnModel, choose_calibrator, cross_fitted, cv_log_loss, tune

ECON = Economics()


def finalize(family, params, train, calib, groups):
    """Fit on train, calibrate on calibration, tune the decision margin on cross-fitted calibration probabilities."""
    from churn.models import Calibrator, build

    pipe = build(family, params).fit(train[FEATURES], train["churned"])
    raw, y = pipe.predict_proba(calib[FEATURES])[:, 1], calib["churned"].to_numpy()
    method, cal_scores = choose_calibrator(raw, y)
    honest = cross_fitted(raw, y, method)
    margin, by_margin = best_margin(
        ECON.expected_value(honest, calib["MonthlyCharges"]), y, calib["MonthlyCharges"].to_numpy(), ECON
    )
    model = ChurnModel(family, params, pipe, Calibrator(method).fit(raw, y), margin, ECON, groups)
    Z = pipe["pre"].transform(train[FEATURES])
    model.reference = {
        "transformed_mean": np.asarray(Z.mean(axis=0)).ravel(),
        **drift.profile(calib[FEATURES], model.predict_proba(calib)),
    }
    return model, {
        "method": method,
        "cv_log_loss": cal_scores,
        "margin": margin,
        "profit_by_margin": {str(k): round(v, 0) for k, v in by_margin.items()},
    }


def policy_stats(target, y, monthly):
    gain = ECON.realised(target, y, monthly)
    return {
        "contacted": int(target.sum()),
        "profit": round(float(gain.sum()), 0),
        "profit_ci": evaluation.ci(gain[evaluation.resamples(len(y))].sum(axis=1)),
        "precision": round(float(y[target].mean()), 4) if target.sum() else None,
        "recall": round(float(target[y == 1].mean()), 4),
    }


def paired_profit(a, b, y, monthly):
    d = ECON.realised(a, y, monthly) - ECON.realised(b, y, monthly)
    return {"diff": round(float(d.sum()), 0), "ci": evaluation.ci(d[evaluation.resamples(len(y))].sum(axis=1))}


def claims(test, y, p, target, monthly, n_customers):
    """Numbers for the resume verification matrix (see docs/resume_verification.md)."""
    churn_p = np.sort(p[y == 1])[::-1]
    t86 = float(churn_p[int(np.ceil(0.86 * len(churn_p))) - 1])  # highest cutoff that still reaches 86% recall
    at86 = p >= t86
    scale = n_customers / len(test)
    return {
        "n_customers": n_customers,
        "policy_recall": round(float(target[y == 1].mean()), 4),
        "threshold_for_86pct_recall": round(t86, 4),
        "at_86pct_recall": {
            "precision": round(float(y[at86].mean()), 4),
            "contacted_share": round(float(at86.mean()), 4),
            "profit": round(float(ECON.realised(at86, y, monthly).sum()), 0),
        },
        "revenue_at_risk": {
            "expected_annual_test": round(ECON.revenue_at_risk(p, monthly), 0),
            "expected_annual_extrapolated_to_base": round(ECON.revenue_at_risk(p, monthly) * scale, 0),
            "realised_annual_test_churners": round(float(monthly[y == 1].sum() * ECON.horizon), 0),
            "realised_monthly_all_churners": None,  # filled from the full dataset below
        },
        "expected_retention_profit_test": round(float(ECON.realised(target, y, monthly).sum()), 0),
    }


def head_to_head(test, model, champion):
    y, monthly = test["churned"].to_numpy(), test["MonthlyCharges"].to_numpy()
    p_new, _, t_new = model.decide(test)
    p_old, _, t_old = champion.decide(test)
    cmp = evaluation.compare(
        y, {"challenger": p_new, "champion": p_old}, {"challenger": t_new, "champion": t_old}, ECON, monthly, "champion"
    )
    return cmp


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=100, help="Optuna trials for XGBoost (logistic gets a third)")
    ap.add_argument("--promote", action="store_true", help="promote the challenger if it passes the gate")
    args = ap.parse_args()
    started = time.time()

    df = load()
    parts = split(df)
    train, calib, test = parts["train"], parts["calibration"], parts["test"]
    Xtr, ytr = train[FEATURES], train["churned"]

    # 1. Model selection: on train only.
    groups = explain.correlated_groups(explain.association_matrix(Xtr))
    studies = {
        "xgboost": tune("xgboost", Xtr, ytr, args.trials),
        "logistic": tune("logistic", Xtr, ytr, max(10, args.trials // 3)),
    }
    candidates = {
        "logistic": ("logistic", studies["logistic"].best_params, studies["logistic"].best_value),
        "xgboost_default": ("xgboost", XGB_DEFAULT, cv_log_loss("xgboost", XGB_DEFAULT, Xtr, ytr)),
        "xgboost_tuned": ("xgboost", studies["xgboost"].best_params, studies["xgboost"].best_value),
    }
    chosen = min(candidates, key=lambda k: candidates[k][2])
    # Ablation behind dropping TotalCharges (~tenure x MonthlyCharges): XGBoost passes extra numerics through.
    ablation = {
        "without_total_charges": round(candidates["xgboost_default"][2], 5),
        "with_total_charges": round(cv_log_loss("xgboost", XGB_DEFAULT, train[FEATURES + ["TotalCharges"]], ytr), 5),
    }

    # 2-3. Calibration and threshold: on calibration customers only. Every candidate gets the same treatment.
    fitted = {name: finalize(fam, params, train, calib, groups) for name, (fam, params, _) in candidates.items()}
    model, cal_report = fitted[chosen]

    # 4. Final evaluation: the test customers, once.
    y, monthly = test["churned"].to_numpy(), test["MonthlyCharges"].to_numpy()
    probs, decisions = {}, {}
    for name, (m, _) in fitted.items():
        probs[name], _, decisions[name] = m.decide(test)
    reference = "logistic" if chosen != "logistic" else "xgboost_tuned"
    comparison = evaluation.compare(y, probs, decisions, ECON, monthly, reference)
    p, ev, target = model.decide(test)
    raw = model.raw(test)

    contribs, base = model.contributions(test)
    reasons = explain.reasons(contribs, test, groups)
    by_feature, by_group = explain.global_importance(contribs, groups)
    assoc = explain.association_matrix(Xtr)
    subgroups = evaluation.subgroups(test, y, p, target, ECON)
    everyone = np.ones(len(y), bool)

    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "data_sha256": fingerprint(),
        "n_customers": len(df),
        "splits": {k: {"n": len(v), "churn_rate": round(float(v["churned"].mean()), 4)} for k, v in parts.items()},
        "features": FEATURES,
        "economics": ECON.dict(),
        "family": model.family,
        "params": model.params,
        "selection": {
            "criterion": "5-fold stratified CV log loss on the train split",
            "chosen": chosen,
            "ablation_cv_log_loss": ablation,
            "candidates": {k: {"family": f, "params": prm, "cv_log_loss": round(s, 5)} for k, (f, prm, s) in candidates.items()},
            "optuna": {
                fam: {
                    "n_trials": len(st.trials),
                    "pruned": sum(t.state.name == "PRUNED" for t in st.trials),
                    "history": [{"trial": t.number, "value": round(t.value, 5)} for t in st.trials if t.value is not None],
                }
                for fam, st in studies.items()
            },
        },
        "calibration": {
            **{k: v for k, v in cal_report.items() if k != "profit_by_margin"},
            "reliability_raw": evaluation.reliability(y, raw),
            "reliability": evaluation.reliability(y, p),
            "ece_raw": round(evaluation.ece(y, raw), 4),
        },
        "threshold": {
            "margin": model.margin,
            "bayes_optimal_margin": 0.0,
            "profit_by_margin_on_calibration": cal_report["profit_by_margin"],
            "test_profit_at_tuned_margin": round(float(ECON.realised(ev > model.margin, y, monthly).sum()), 0),
            "test_profit_at_zero_margin": round(float(ECON.realised(ev > 0, y, monthly).sum()), 0),
        },
        "test": {
            "models": comparison,
            "reference_model": reference,
            "policies": {
                "Expected-value targeting (this model)": policy_stats(target, y, monthly),
                "Contact if P(churn) > 0.5": policy_stats(p > 0.5, y, monthly),
                "Contact everyone": policy_stats(everyone, y, monthly),
                "Perfect foresight (ceiling)": policy_stats((y == 1) & (ECON.expected_value(1.0, monthly) > 0), y, monthly),
            },
            # Paired: does EV targeting beat the naive 0.5 cutoff on the same customers, beyond sampling noise?
            "ev_vs_p05_profit": paired_profit(target, p > 0.5, y, monthly),
            "profit_curve": evaluation.profit_curve(y, ev, ECON, monthly),
            "sensitivity": evaluation.sensitivity(y, p, monthly, Economics),
        },
        "subgroups": subgroups,
        "errors": evaluation.error_analysis(test, y, p, target, ev, ECON, reasons),
        "explain": {
            "groups": groups,
            "group_threshold": explain.GROUP_THRESHOLD,
            "association": {"features": FEATURES, "matrix": assoc.round(3).values.tolist()},
            "importance_feature": by_feature,
            "importance_group": by_group,
            "beeswarm": explain.beeswarm(contribs, test),
            "base_log_odds": round(float(base.mean()), 4),
        },
        "drift_simulation": drift.simulate(
            model,
            test,
            lambda f, s: {k: round(float(evaluation.METRICS[k](f["churned"].to_numpy(), s)), 4) for k in ("roc_auc", "ece")},
        ),
        "claims": claims(test, y, p, target, monthly, len(df)),
        "test_predictions": [
            {"p": round(float(a), 4), "monthly": float(m), "churned": int(c), "contract": k, "tenure": int(t)}
            for a, m, c, k, t in zip(p, monthly, y, test["Contract"], test["tenure"], strict=True)
        ],
    }
    report["claims"]["revenue_at_risk"]["realised_monthly_all_churners"] = round(
        float(df.loc[df.churned == 1, "MonthlyCharges"].sum()), 0
    )

    # Champion-challenger: both models on the same test customers.
    champion = None
    if registry.index()["champion"]:
        champ_model, champ_report = registry.load()
        h2h = head_to_head(test, model, champ_model)
        report["head_to_head"] = {"champion_version": champ_report["version"], "models": h2h}
        champion = {"metrics": h2h["champion"]}
        gate_input = {"metrics": h2h["challenger"], "vs_champion": h2h["challenger"]["vs_champion"]}
    else:
        gate_input = {"metrics": comparison[chosen]}
    gate_input |= {"subgroups": subgroups, "everyone_profit": report["test"]["policies"]["Contact everyone"]["profit"]}
    report["gate"] = registry.gate(gate_input, champion)
    report["train_seconds"] = round(time.time() - started, 1)

    version = registry.register(model, report)
    log_mlflow(version, report, studies)
    print(
        json.dumps(
            {
                "version": version,
                "chosen": chosen,
                "test": {k: v["value"] for k, v in comparison[chosen].items() if isinstance(v, dict) and "value" in v},
                "gate": report["gate"]["passed"],
            },
            indent=2,
        )
    )
    for c in report["gate"]["checks"]:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    if args.promote:
        registry.promote(version, report["gate"])
        from churn.card import write_docs

        write_docs()
        print(f"promoted {version}")
    elif not report["gate"]["passed"]:
        raise SystemExit("challenger failed the promotion gate")


def log_mlflow(version, report, studies):
    import mlflow  # imported here so the API never loads it

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("telco-churn")
    chosen = report["selection"]["chosen"]
    with mlflow.start_run(run_name=version):
        mlflow.set_tags(
            {
                "version": version,
                "family": report["family"],
                "data_sha256": report["data_sha256"],
                "gate_passed": report["gate"]["passed"],
            }
        )
        mlflow.log_params(
            {
                **report["params"],
                **report["economics"],
                "calibrator": report["calibration"]["method"],
                "margin": report["threshold"]["margin"],
            }
        )
        for name, m in report["test"]["models"].items():
            mlflow.log_metrics({f"{name}.{k}": v["value"] for k, v in m.items() if "value" in v})
        mlflow.log_metrics({f"cv_log_loss.{k}": v["cv_log_loss"] for k, v in report["selection"]["candidates"].items()})
        mlflow.log_artifact(f"models/{version}/report.json")
        mlflow.log_artifact(f"models/{version}/model.joblib")
        for fam, st in studies.items():
            for t in st.trials:
                if t.value is not None:
                    with mlflow.start_run(run_name=f"{fam}-trial-{t.number}", nested=True):
                        mlflow.log_params(t.params)
                        mlflow.log_metric("cv_log_loss", t.value)
    return chosen


if __name__ == "__main__":
    main()
