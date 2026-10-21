"""Probability-of-default models (gradient boosted trees) with out-of-sample predictions."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BASE_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=15, max_iter=400,
                   min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def make_model(params: dict | None = None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**{**BASE_PARAMS, **(params or {})})


def oos_predict(X, y, params=None, n_splits=5, seed=0) -> np.ndarray:
    """Out-of-sample PD for every row of the training set (K-fold). This is what the
    market simulator must use: in-sample predictions would make every lender look clairvoyant."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(make_model(params), X, y, cv=cv, method="predict_proba")[:, 1]


def fit_full(X, y, params=None) -> HistGradientBoostingClassifier:
    return make_model(params).fit(X, y)


def scores(y, p) -> dict:
    return {"auc": roc_auc_score(y, p), "logloss": log_loss(y, p), "brier": brier_score_loss(y, p)}


def calibration_table(y, p, n_bins=10) -> pd.DataFrame:
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"p": p, "y": y, "bin": bins})
    return df.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), realized=("y", "mean"))


def tune(X, y, grid: list[dict], n_splits=3, seed=0):
    """Small grid search on CV log-loss (the right metric: we need calibrated probabilities)."""
    rows, best = [], None
    for params in grid:
        p = oos_predict(X, y, params, n_splits, seed)
        ll = log_loss(y, p)
        rows.append({**params, "logloss": ll, "auc": roc_auc_score(y, p)})
        if best is None or ll < best[0]:
            best = (ll, params)
    return best[1], pd.DataFrame(rows).sort_values("logloss")
