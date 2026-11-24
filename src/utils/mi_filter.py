"""
Mutual-information-based feature ranking utility.

Used by step 8 / 8b to cull the dilution-prone PFF state-space block
(~240 cols, ~14% total LightGBM importance) down to a curated top-N
before Optuna tunes hyperparams. MI is a one-shot deterministic rank
(no Optuna seed dependence, no shadow-feature stochasticity), so the
selected subset is stable run-to-run — better calibrated than Boruta
for noisy small-n heads.

For continuous targets (impact, impact_wav) and soft probability
targets (role_target ∈ [0,1]) we use `mutual_info_regression`. MI is
information-theoretic and doesn't require a class label; it captures
both linear and nonlinear dependence.

NaN handling: sklearn's MI doesn't tolerate NaN, so we median-impute
per column inside the wrapper. Indicator columns naturally land at 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _coerce_numeric_matrix(X: pd.DataFrame) -> np.ndarray:
    """float64, NaN -> per-column median (0 if the column is all-NaN)."""
    arr = X.to_numpy(dtype=float, na_value=np.nan, copy=True)
    if not np.isnan(arr).any():
        return arr
    col_med = np.nanmedian(arr, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        idx = np.where(nan_mask)
        arr[idx] = col_med[idx[1]]
    return arr


def rank_by_mi(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Rank columns of X by mutual information against y (descending).

    Returns:
        {
          "ranking":      [col, ...]    # cols sorted by MI desc
          "scores":       {col: float}  # MI score per col
          "n_features_in": int,
          "n_samples":    int,
        }
    """
    if X.empty or len(X.columns) == 0:
        return {"ranking": [], "scores": {}, "n_features_in": 0, "n_samples": 0}
    cols = list(X.columns)
    X_arr = _coerce_numeric_matrix(X)
    y_arr = np.asarray(y, dtype=float).ravel()
    mask = ~np.isnan(y_arr)
    if not mask.all():
        X_arr = X_arr[mask]
        y_arr = y_arr[mask]
    if len(y_arr) < 30:
        return {"ranking": cols, "scores": {c: 0.0 for c in cols},
                "n_features_in": len(cols), "n_samples": int(len(y_arr))}

    from sklearn.feature_selection import mutual_info_regression
    scores = mutual_info_regression(
        X_arr, y_arr, n_neighbors=n_neighbors, random_state=random_state,
    )
    scores_d = {c: float(s) for c, s in zip(cols, scores)}
    ranking = sorted(cols, key=lambda c: -scores_d[c])
    return {
        "ranking":      ranking,
        "scores":       scores_d,
        "n_features_in": len(cols),
        "n_samples":    int(len(y_arr)),
    }


def select_top_mi(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_top: int,
    *,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> tuple[list[str], dict[str, float]]:
    """Convenience: return (top_n_cols_in_X_order, scores_dict)."""
    verdict = rank_by_mi(X, y, n_neighbors=n_neighbors, random_state=random_state)
    keep = set(verdict["ranking"][:n_top])
    # Preserve original X column order for the kept set.
    top_in_order = [c for c in X.columns if c in keep]
    return top_in_order, verdict["scores"]
