"""
Boruta feature-selection wrapper used by step 8 / 8b.

Boruta tests each real feature against shuffled "shadow" copies of the same
feature using a binomial significance test, producing a principled
keep / reject / undecided verdict per column. We use it to filter the
inflated PFF state-space feature surface before Optuna tuning, on a
per-(pos × target-model) basis.

NaN handling: RandomForest cannot consume NaN, so we median-impute per
column inside the wrapper. Indicator columns naturally land at 0.

Returns column names — in the original input order — bucketed into
{confirmed, tentative, rejected} plus a sklearn-style ranking dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Boruta uses an old sklearn-API-style attribute (`get_params(deep=True)`)
# that all sklearn estimators still implement; nothing to monkey-patch in
# the current versions on this machine (boruta 0.3 + sklearn 1.7 +
# numpy 1.26 all play together).


def _coerce_numeric_matrix(X: pd.DataFrame) -> np.ndarray:
    """float64, NaN -> per-column median (0 if the column is all-NaN)."""
    arr = X.to_numpy(dtype=float, na_value=np.nan, copy=True)
    if not np.isnan(arr).any():
        return arr
    # Per-column median imputation; all-NaN columns -> 0.
    col_med = np.nanmedian(arr, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        idx = np.where(nan_mask)
        arr[idx] = col_med[idx[1]]
    return arr


def run_boruta(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_estimators: int | str = "auto",
    max_iter: int = 100,
    percentile: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
    classifier: bool = False,
    rf_n_jobs: int = -1,
) -> dict[str, Any]:
    """
    Run BorutaPy on (X, y) and return verdicts in original column order.

    `classifier=True` uses RandomForestClassifier; default is regressor
    (right choice for soft probability targets like role_target ∈ [0,1]
    and the continuous impact heads).

    Returns:
        {
          "confirmed": [col, ...],       # support_ == True
          "tentative": [col, ...],       # support_weak_ == True
          "rejected":  [col, ...],       # neither
          "ranking":   {col: int},       # boruta ranking_ (1=best)
          "n_features_in":  int,
          "n_confirmed":    int,
          "n_tentative":    int,
          "n_rejected":     int,
          "max_iter":       int,
          "percentile":     int,
          "alpha":          float,
        }
    """
    if X.empty or len(X.columns) == 0:
        return _empty_result(max_iter, percentile, alpha)

    cols = list(X.columns)
    X_arr = _coerce_numeric_matrix(X)
    y_arr = np.asarray(y, dtype=float).ravel()

    if classifier:
        from sklearn.ensemble import RandomForestClassifier as _RF
        # Boruta-internal n_estimators="auto" doubles each iter; the base
        # RF only needs sane defaults — Boruta sets n_estimators on the
        # estimator each iteration.
        rf = _RF(n_jobs=rf_n_jobs, random_state=random_state, max_depth=5)
    else:
        from sklearn.ensemble import RandomForestRegressor as _RF
        rf = _RF(n_jobs=rf_n_jobs, random_state=random_state, max_depth=5)

    from boruta import BorutaPy

    b = BorutaPy(
        estimator=rf,
        n_estimators=n_estimators,
        perc=percentile,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )
    b.fit(X_arr, y_arr)

    support = np.asarray(b.support_, dtype=bool)
    support_weak = np.asarray(getattr(b, "support_weak_", np.zeros_like(support)), dtype=bool)
    ranking = np.asarray(b.ranking_, dtype=int)

    confirmed = [c for c, s in zip(cols, support) if s]
    tentative = [c for c, s, w in zip(cols, support, support_weak) if (not s) and w]
    rejected  = [c for c, s, w in zip(cols, support, support_weak) if (not s) and (not w)]

    return {
        "confirmed": confirmed,
        "tentative": tentative,
        "rejected":  rejected,
        "ranking":   {c: int(r) for c, r in zip(cols, ranking)},
        "n_features_in": len(cols),
        "n_confirmed":   len(confirmed),
        "n_tentative":   len(tentative),
        "n_rejected":    len(rejected),
        "max_iter":      int(max_iter),
        "percentile":    int(percentile),
        "alpha":         float(alpha),
    }


def _empty_result(max_iter: int, percentile: int, alpha: float) -> dict[str, Any]:
    return {
        "confirmed": [],
        "tentative": [],
        "rejected":  [],
        "ranking":   {},
        "n_features_in": 0,
        "n_confirmed":   0,
        "n_tentative":   0,
        "n_rejected":    0,
        "max_iter":      int(max_iter),
        "percentile":    int(percentile),
        "alpha":         float(alpha),
    }


def select_features(verdict: dict, *, include_tentative: bool = False) -> list[str]:
    """Filter feature_cols from a verdict dict, preserving original order via
    the union of confirmed (+ tentative if requested)."""
    keep = set(verdict.get("confirmed", []))
    if include_tentative:
        keep |= set(verdict.get("tentative", []))
    # Preserve original ordering using the ranking dict's insertion order
    # (which mirrors X.columns at fit time).
    ordered = list(verdict.get("ranking", {}).keys())
    return [c for c in ordered if c in keep]
