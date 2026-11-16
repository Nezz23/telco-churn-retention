"""Candidate models, Optuna tuning, probability calibration, and the deployable ChurnModel."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import DMatrix, XGBClassifier

from churn.data import CATEGORICAL, FEATURES, NUMERIC, SEED
from churn.economics import Economics

XGB_DEFAULT = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}


def build(family, params):
    # No class re-weighting: it would distort the probabilities the profit policy multiplies by.
    if family == "xgboost":
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough")
        model = XGBClassifier(**params, eval_metric="logloss", random_state=SEED, n_jobs=4)
    else:
        pre = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL), ("num", StandardScaler(), NUMERIC)]
        )
        model = LogisticRegression(**params, max_iter=5000)
    return Pipeline([("pre", pre), ("model", model)])


def cv_log_loss(family, params, X, y, trial=None):
    """5-fold CV log loss: the selection criterion, because the profit policy needs good probabilities."""
    losses = []
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
        pipe = build(family, params).fit(X.iloc[tr], y.iloc[tr])
        losses.append(log_loss(y.iloc[va], pipe.predict_proba(X.iloc[va])[:, 1]))
        if trial is not None:
            trial.report(float(np.mean(losses)), k)
            if trial.should_prune():
                from optuna import TrialPruned

                raise TrialPruned()
    return float(np.mean(losses))


SPACES = {
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 1200, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 6),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 50, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 0.1, 50, log=True),
    },
    "logistic": lambda t: {"C": t.suggest_float("C", 1e-3, 10, log=True)},
}


def tune(family, X, y, n_trials):
    import optuna  # training-only dependency: the API unpickles this module and must not need it

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(lambda t: cv_log_loss(family, SPACES[family](t), X, y, t), n_trials=n_trials)
    return study


@dataclass
class Calibrator:
    """Maps raw model scores to calibrated probabilities. 'identity' keeps the raw score."""

    method: str
    fitted: object = None

    @staticmethod
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw, y):
        if self.method == "sigmoid":
            self.fitted = LogisticRegression(C=1e6).fit(self.logit(raw), y)
        elif self.method == "isotonic":
            self.fitted = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(raw, y)
        return self

    def __call__(self, raw):
        raw = np.asarray(raw, dtype=float)
        if self.method == "sigmoid":
            return self.fitted.predict_proba(self.logit(raw))[:, 1]
        if self.method == "isotonic":
            return self.fitted.predict(raw)
        return raw


def choose_calibrator(raw, y):
    """Pick identity / sigmoid / isotonic by 5-fold CV log loss on the calibration customers only."""
    scores = {}
    for method in ("identity", "sigmoid", "isotonic"):
        losses = [
            log_loss(y[va], Calibrator(method).fit(raw[tr], y[tr])(raw[va]))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(raw, y)
        ]
        scores[method] = float(np.mean(losses))
    return min(scores, key=scores.get), scores


def cross_fitted(raw, y, method):
    """Calibrated probabilities for each calibration customer from a calibrator that never saw them,
    so the decision threshold is tuned on honest probabilities."""
    out = np.empty(len(raw))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED + 1).split(raw, y):
        out[va] = Calibrator(method).fit(raw[tr], y[tr])(raw[va])
    return out


@dataclass
class ChurnModel:
    """Everything needed to score a customer: the fitted pipeline, its calibrator, the decision margin and the
    economics it was tuned for, plus the reference profile (calibration customers) drift is measured against."""

    family: str
    params: dict
    pipeline: Pipeline
    calibrator: Calibrator
    margin: float
    economics: Economics
    groups: dict = field(default_factory=dict)  # feature -> correlated-feature group name
    reference: dict = field(default_factory=dict)  # drift reference profile

    def raw(self, X):
        return self.pipeline.predict_proba(X[FEATURES])[:, 1]

    def predict_proba(self, X):
        return self.calibrator(self.raw(X))

    def decide(self, X):
        p = self.predict_proba(X)
        ev = self.economics.expected_value(p, X["MonthlyCharges"].to_numpy())
        return p, ev, ev > self.margin

    def contributions(self, X):
        """SHAP values in log-odds per source feature (one-hot columns summed back), plus the base value.
        XGBoost: exact TreeSHAP. Logistic: exact linear SHAP, w * (x - mean), under feature independence."""
        pre, model = self.pipeline["pre"], self.pipeline["model"]
        Z = pre.transform(X[FEATURES])
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z, dtype=float)
        if self.family == "xgboost":
            c = model.get_booster().predict(DMatrix(Z), pred_contribs=True)
            contribs, base = c[:, :-1], c[:, -1]
        else:
            mean = self.reference["transformed_mean"]
            contribs = (Z - mean) * model.coef_[0]
            base = np.full(len(Z), float(model.intercept_[0] + mean @ model.coef_[0]))
        names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
        mapping = np.array([[n == f or n.startswith(f + "_") for f in FEATURES] for n in names], dtype=float)
        return pd.DataFrame(contribs @ mapping, columns=FEATURES, index=X.index), base
