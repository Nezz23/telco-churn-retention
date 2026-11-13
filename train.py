"""Train the churn model and its retention-targeting policy, tracked in MLflow."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import DMatrix, XGBClassifier

SEED = 42
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL
PARAMS = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}

# Retention-campaign assumptions. Not in the dataset: change them to match a real business.
OFFER_COST = 100.0  # $ per customer contacted (discount + outreach)
SAVE_RATE = 0.30    # share of would-be churners the offer keeps
HORIZON = 12        # months of revenue a saved customer is worth


def load(path="data/telco.csv"):
    df = pd.read_csv(path)
    # 11 brand-new customers (tenure 0) have a blank TotalCharges: they have not been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df[FEATURES], (df["Churn"] == "Yes").astype(int), df


def build():
    # No class re-weighting: it would distort the probabilities the profit policy relies on.
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough")
    model = XGBClassifier(**PARAMS, eval_metric="logloss", random_state=SEED)
    return Pipeline([("pre", pre), ("model", model)])


def expected_value(p, monthly, save_rate=SAVE_RATE):
    """Expected $ from contacting a customer: chance they churn x chance we save them x their value, minus cost."""
    return p * save_rate * HORIZON * monthly - OFFER_COST


def realised_profit(target, churned, monthly, save_rate=SAVE_RATE):
    """Profit a targeting decision actually earns on labelled customers."""
    return float(np.sum(target * (churned * save_rate * HORIZON * monthly - OFFER_COST)))


def reasons(pipe, X, k=3):
    """Top-k features pushing each customer's churn risk up (XGBoost SHAP contributions)."""
    pre, model = pipe["pre"], pipe["model"]
    contribs = model.get_booster().predict(DMatrix(pre.transform(X)), pred_contribs=True)[:, :-1]
    # Sum one-hot columns back into their source feature with a 0/1 mapping matrix.
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    mapping = np.array([[n == f or n.startswith(f + "_") for f in FEATURES] for n in names], dtype=float)
    per_feature = contribs @ mapping
    top = np.argsort(-per_feature, axis=1)[:, :k]
    values = X[FEATURES].to_numpy()
    return [[f"{FEATURES[j]}={values[i, j]}" for j in row if per_feature[i, j] > 0] for i, row in enumerate(top)]


def main():
    X, y, df = load()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = build()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    pipe.fit(X_tr, y_tr)

    p = pipe.predict_proba(X_te)[:, 1]
    y_arr, monthly = y_te.to_numpy(), X_te["MonthlyCharges"].to_numpy()
    target = expected_value(p, monthly) > 0
    frac_pos, mean_pred = calibration_curve(y_arr, p, n_bins=10, strategy="quantile")

    policies = {  # same test customers, different targeting rules
        "model_expected_value": target,
        "contact_everyone": np.ones_like(target),
        "contact_p_over_0.5": p > 0.5,
        "oracle_perfect_foresight": (y_arr == 1) & (expected_value(1.0, monthly) > 0),
    }
    metrics = {
        "cv_roc_auc": round(roc_auc_score(y_tr, oof), 4),
        "test_roc_auc": round(roc_auc_score(y_te, p), 4),
        "test_brier": round(brier_score_loss(y_te, p), 4),
        "test_calibration_error": round(float(np.mean(np.abs(frac_pos - mean_pred))), 4),
        "policy_recall": round(recall_score(y_te, target), 4),
        "policy_precision": round(precision_score(y_te, target), 4),
        "policy_contacted": int(target.sum()),
        "profit": {k: round(realised_profit(v, y_arr, monthly), 0) for k, v in policies.items()},
        # Does the policy still beat "contact everyone" if offers work less (or more) often than assumed?
        "profit_by_save_rate": {
            str(s): {"model": round(realised_profit(expected_value(p, monthly, s) > 0, y_arr, monthly, s), 0),
                     "everyone": round(realised_profit(np.ones_like(target), y_arr, monthly, s), 0)}
            for s in (0.1, 0.2, 0.3, 0.5)},
        "assumptions": {"offer_cost": OFFER_COST, "save_rate": SAVE_RATE, "horizon_months": HORIZON},
        "n_customers": len(df), "n_test": len(X_te),
    }

    import mlflow  # imported here so the API server does not load it

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("telco-churn")
    with mlflow.start_run():
        mlflow.log_params({**PARAMS, **metrics["assumptions"], "seed": SEED})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)}
                           | {f"profit_{k}": v for k, v in metrics["profit"].items()})

    Path("model").mkdir(exist_ok=True)
    joblib.dump(pipe, "model/model.joblib")
    Path("model/metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
