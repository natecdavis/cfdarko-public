"""T3 — Adaptive Lasso fallback NaN safety.

Verifies that:
  1. Fitting on a frame with NaN values in some context columns succeeds —
     NaN features are mean-imputed inside the pipeline (rows are kept).
  2. Predict on a frame containing NaN values returns finite predictions
     (the pipeline's SimpleImputer fills training means).
  3. When AL fires (cv_r2 < trigger; with trigger=+inf in current config,
     it always fires), the resulting pipeline still produces finite outputs.

These cover the regression class where a malformed contextual feature could
silently propagate NaN through the model's predictions, poisoning step 5
residuals downstream.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.contextual import ContextualAdjustmentModel


def _make_frame(n_rows: int = 200, n_seasons: int = 8, seed: int = 0) -> pd.DataFrame:
    """Synthetic bridge-style frame: a target stat ('rec'), a few context cols
    with deliberate NaN holes, and season_year for the expanding-window CV."""
    rng = np.random.default_rng(seed)
    seasons = np.repeat(np.arange(2010, 2010 + n_seasons), n_rows // n_seasons + 1)[:n_rows]
    df = pd.DataFrame({
        "season_year":              seasons,
        "opp_rolling_def_ppa":      rng.normal(0, 1, n_rows),
        "opp_rolling_off_ppa":      rng.normal(0, 1, n_rows),
        "opp_rolling_def_success_rate": rng.normal(0.5, 0.1, n_rows),
        "opp_rolling_def_stuff_rate":   rng.normal(0.2, 0.05, n_rows),
        "opp_rolling_off_success_rate": rng.normal(0.5, 0.1, n_rows),
        "opp_rolling_off_explosiveness":rng.normal(1.0, 0.2, n_rows),
        "opp_rolling_def_explosiveness":rng.normal(1.0, 0.2, n_rows),
        "opp_rolling_havoc_total":      rng.normal(0.2, 0.05, n_rows),
        "opp_rolling_havoc_front_seven":rng.normal(0.1, 0.03, n_rows),
        "opp_rolling_havoc_db":         rng.normal(0.1, 0.03, n_rows),
        "conf_tier_power4":         rng.binomial(1, 0.4, n_rows).astype(float),
        "conf_tier_g5":             rng.binomial(1, 0.3, n_rows).astype(float),
        "is_home":                  rng.binomial(1, 0.5, n_rows).astype(float),
        "is_neutral":               rng.binomial(1, 0.05, n_rows).astype(float),
        "days_rest":                7.0,
        "opp_days_rest":            7.0,
        "game_hour":                15.0,
        "tz_diff":                  0.0,
        "temperature":              60.0,
        "wind_speed":               5.0,
        "precipitation":            0.0,
        "game_indoors":             0.0,
        # League-level features (lg_*) — set to small noise; build_context_features
        # adds them as zeros if absent. Including them avoids the "no feature" warn.
        **{f: rng.normal(0, 0.1, n_rows) for f in [
            "lg_off_ppa", "lg_off_success_rate", "lg_off_explosiveness", "lg_off_stuff_rate",
            "lg_def_ppa", "lg_def_success_rate", "lg_def_stuff_rate",
            "lg_pass_rate", "lg_pass_att", "lg_rush_att",
            "lg_pass_tds", "lg_rush_tds", "lg_interceptions",
        ]},
        # Target: 'rec' is rate-stat-ish for WR — use raw column name; the model
        # falls back to the raw column when no _p100 column exists.
        "rec":                      rng.normal(5.0, 2.0, n_rows),
    })

    # Punch NaN holes in two context columns to exercise the .notna().all() mask
    # in fit() and the .fillna(0) imputation in predict().
    nan_idx = rng.choice(n_rows, size=n_rows // 5, replace=False)
    df.loc[nan_idx[: len(nan_idx) // 2], "opp_rolling_def_ppa"] = np.nan
    df.loc[nan_idx[len(nan_idx) // 2:], "temperature"] = np.nan

    return df


def test_fit_drops_nan_rows_predict_returns_finite():
    """Holes in context columns do not poison fit or predict."""
    df = _make_frame()
    model = ContextualAdjustmentModel(stat="rec", position_group="WR")
    model.fit(df)
    preds = model.predict(df)
    assert len(preds) == len(df), "predict must return one row per input"
    assert np.isfinite(preds.values).all(), \
        "NaN propagated through predict — check StandardScaler / fillna path"


def test_fit_succeeds_when_one_column_is_all_nan():
    """An entirely-NaN column should be excluded by _get_feature_cols
    (.notna().any() guard) without raising."""
    df = _make_frame()
    df["wind_speed"] = np.nan      # entire column NaN
    model = ContextualAdjustmentModel(stat="rec", position_group="WR")
    model.fit(df)
    assert "wind_speed" not in model.feature_cols, \
        "all-NaN column should be excluded from feature_cols"
    preds = model.predict(df)
    assert np.isfinite(preds.values).all()


def test_predict_handles_nan_in_inference_frame():
    """A clean fit frame, but the inference frame has NaN in features —
    predict() fillna(0)s and must not return NaN."""
    train = _make_frame()
    train = train.dropna()         # clean training data
    model = ContextualAdjustmentModel(stat="rec", position_group="WR")
    model.fit(train)

    infer = _make_frame(seed=1)
    infer.loc[:10, "opp_rolling_def_ppa"] = np.nan
    infer.loc[5:15, "is_home"] = np.nan
    preds = model.predict(infer)
    assert np.isfinite(preds.values).all(), \
        "predict() did not impute NaN before model — found NaN in output"
