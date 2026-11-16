"""Global and individual explanations, with correlated features grouped so credit is not double counted."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, spearmanr

from churn.data import CATEGORICAL, FEATURES, NUMERIC

GROUP_THRESHOLD = 0.5  # features with association above this (average linkage) explain as one group


def association(a: pd.Series, b: pd.Series) -> float:
    """|Spearman| for two numerics, Cramer's V for two categoricals, correlation ratio (eta) for mixed."""
    if a.name in NUMERIC and b.name in NUMERIC:
        return abs(spearmanr(a, b).statistic)
    if a.name in CATEGORICAL and b.name in CATEGORICAL:
        table = pd.crosstab(a, b)
        chi2 = chi2_contingency(table, correction=False)[0]
        return float(np.sqrt(chi2 / (table.values.sum() * max(1, min(table.shape) - 1))))
    num, cat = (a, b) if a.name in NUMERIC else (b, a)
    means = num.groupby(cat).agg(["mean", "count"])
    between = (means["count"] * (means["mean"] - num.mean()) ** 2).sum()
    total = ((num - num.mean()) ** 2).sum()
    return float(np.sqrt(between / total)) if total else 0.0


def association_matrix(X):
    m = pd.DataFrame(1.0, index=FEATURES, columns=FEATURES)
    for i, f in enumerate(FEATURES):
        for g in FEATURES[i + 1 :]:
            m.loc[f, g] = m.loc[g, f] = association(X[f], X[g])
    return m


def correlated_groups(assoc):
    """Average-linkage clustering on 1 - association. Singletons keep their own name."""
    labels = fcluster(linkage(squareform(1 - assoc.values, checks=False), "average"), t=1 - GROUP_THRESHOLD, criterion="distance")
    members = pd.Series(FEATURES).groupby(labels).apply(list)
    groups = {}
    for feats in members:
        name = feats[0] if len(feats) == 1 else " + ".join(feats) if len(feats) <= 2 else f"{feats[0]} + {len(feats) - 1} related"
        groups.update({f: name for f in feats})
    return groups


def global_importance(contribs, groups):
    per_feature = contribs.abs().mean().sort_values(ascending=False)
    per_group = contribs.T.groupby(groups).sum().T.abs().mean().sort_values(ascending=False)
    return ({k: round(float(v), 4) for k, v in per_feature.items()}, {k: round(float(v), 4) for k, v in per_group.items()})


def beeswarm(contribs, X, n=400, seed=0):
    """A sample of (value, SHAP) pairs per feature for the global summary plot."""
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    return {f: {"values": X[f].iloc[idx].tolist(), "shap": contribs[f].iloc[idx].round(4).tolist()} for f in FEATURES}


def reasons(contribs, X, groups, k=3):
    """Top-k features pushing each customer's risk up, with their value, log-odds push and correlated group."""
    C = contribs.to_numpy()
    values = X[list(contribs.columns)].to_numpy(dtype=object)
    names = list(contribs.columns)
    top = np.argsort(-C, axis=1)[:, :k]
    return [
        [
            {
                "feature": names[j],
                "value": _plain(values[i, j]),
                "contribution": round(float(C[i, j]), 4),
                "group": groups.get(names[j], names[j]),
            }
            for j in row
            if C[i, j] > 0
        ]
        for i, row in enumerate(top)
    ]


def _plain(v):
    return v.item() if hasattr(v, "item") else v
