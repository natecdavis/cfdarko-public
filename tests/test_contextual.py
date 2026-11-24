"""
Unit tests for src/models/contextual.py

Tests context feature building, stat column selection, Ridge model fit/predict,
expanding CV no-leakage, and group-level model fitting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.models.contextual import (
    build_context_features,
    get_stat_column,
    ContextualAdjustmentModel,
    _ExpandingSeasonCV,
    fit_contextual_models_for_group,
)
import config as C


# ── build_context_features ────────────────────────────────────────────────────

def _bare_df(n=50):
    """Minimal DataFrame with no contextual features."""
    return pd.DataFrame({"player_id": range(n), "game_id": range(n)})


def test_build_context_features_all_cols_present():
    """build_context_features() must create all non-lg_* CONTEXT_FEATURES columns.
    League-average features (lg_*) are joined upstream in Step 1 and are not
    added by build_context_features itself."""
    df = _bare_df()
    result = build_context_features(df)
    # Only check columns that build_context_features is responsible for creating
    columns_set_by_function = [
        col for col in C.CONTEXT_FEATURES if not col.startswith("lg_")
    ]
    for col in columns_set_by_function:
        assert col in result.columns, f"Missing CONTEXT_FEATURES column: {col!r}"


def test_build_context_features_is_home_default():
    df = _bare_df()
    result = build_context_features(df)
    assert (result["is_home"] == 0.5).all()


def test_build_context_features_days_rest_default():
    df = _bare_df()
    result = build_context_features(df)
    assert (result["days_rest"] == 7.0).all()


def test_build_context_features_game_indoors_default():
    df = _bare_df()
    result = build_context_features(df)
    assert (result["game_indoors"] == 0.0).all()


def test_build_context_features_preserves_existing_values():
    """Existing column values must not be overwritten."""
    df = pd.DataFrame({"is_home": [1.0, 0.0, 1.0]})
    result = build_context_features(df)
    assert list(result["is_home"]) == [1.0, 0.0, 1.0]


def test_build_context_features_conference_tier_from_column():
    df = pd.DataFrame({"conference_tier": [1, 2, 3, 1, 2]})
    result = build_context_features(df)
    assert "conf_tier_power4" in result.columns
    assert "conf_tier_g5" in result.columns
    # P4 rows
    assert (result.loc[result["conference_tier"] == 1, "conf_tier_power4"] == 1.0).all()
    # G5 rows
    assert (result.loc[result["conference_tier"] == 2, "conf_tier_g5"] == 1.0).all()
    # FCS rows — both dummies 0
    fcs = result[result["conference_tier"] == 3]
    assert (fcs["conf_tier_power4"] == 0.0).all()
    assert (fcs["conf_tier_g5"] == 0.0).all()


def test_build_context_features_conference_fallback():
    """Unknown conference string → DEFAULT_CONFERENCE_TIER."""
    df = pd.DataFrame({"conference": ["UnknownConf"] * 5})
    result = build_context_features(df)
    # No exception; conf_tier_* columns exist
    assert "conf_tier_power4" in result.columns


# ── get_stat_column ───────────────────────────────────────────────────────────

def test_get_stat_column_counting_stat_with_p100():
    """Counting stat with _p100 column present → returns _p100 column."""
    stat = "pass_yards"
    df = pd.DataFrame({f"{stat}_p100": [100.0], stat: [50.0]})
    col = get_stat_column(stat, df)
    assert col == f"{stat}_p100"


def test_get_stat_column_rate_stat():
    """Rate stat → returns raw column (no _p100)."""
    stat = "completion_pct"   # in RATE_STAT_ATTEMPT_COL
    df = pd.DataFrame({stat: [0.65], f"{stat}_p100": [65.0]})
    col = get_stat_column(stat, df)
    assert col == stat


def test_get_stat_column_counting_stat_missing_p100():
    """Counting stat but _p100 column absent → falls back to raw column."""
    stat = "pass_yards"
    df = pd.DataFrame({stat: [200.0]})  # no _p100 column
    col = get_stat_column(stat, df)
    assert col == stat


# ── _ExpandingSeasonCV ────────────────────────────────────────────────────────

def test_expanding_cv_no_future_leakage():
    """Test seasons in every fold must be strictly after all train seasons."""
    n = 200
    rng = np.random.default_rng(0)
    seasons = rng.integers(2004, 2020, n)
    X = np.ones((n, 3))
    y = np.zeros(n)
    cv = _ExpandingSeasonCV(min_train_seasons=3, min_test_size=5)
    for train_idx, test_idx in cv.split(X, y, groups=seasons):
        max_train_season = seasons[train_idx].max()
        min_test_season = seasons[test_idx].min()
        assert min_test_season > max_train_season, (
            f"Future leakage: test season {min_test_season} <= "
            f"max train season {max_train_season}"
        )


def test_expanding_cv_min_train_size():
    """CV must not yield folds before min_train_seasons have been seen."""
    n = 100
    seasons = np.array([2004 + (i % 15) for i in range(n)])
    X = np.ones((n, 2))
    y = np.zeros(n)
    cv = _ExpandingSeasonCV(min_train_seasons=5, min_test_size=1)
    for train_idx, test_idx in cv.split(X, y, groups=seasons):
        unique_train_seasons = len(set(seasons[train_idx]))
        assert unique_train_seasons >= 5


# ── ContextualAdjustmentModel ─────────────────────────────────────────────────

def _make_model_df(n=100, stat="pass_yards", seed=42):
    """Synthetic training DataFrame for contextual model."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        f"{stat}_p100": rng.normal(200, 50, n),
        stat: rng.normal(150, 40, n),
        "season_year": rng.integers(2010, 2020, n),
        "is_home": rng.choice([0.0, 1.0], n),
        "is_neutral": np.zeros(n),
        "days_rest": rng.integers(5, 14, n).astype(float),
        "opp_days_rest": rng.integers(5, 14, n).astype(float),
        "game_hour": rng.uniform(12, 20, n),
        "tz_diff": rng.integers(-3, 3, n).astype(float),
        "temperature": rng.uniform(40, 90, n),
        "wind_speed": rng.uniform(0, 20, n),
        "precipitation": np.zeros(n),
        "game_indoors": np.zeros(n),
        "opp_rolling_def_ppa": rng.normal(0, 0.1, n),
        "opp_rolling_off_ppa": rng.normal(0, 0.1, n),
        "opp_rolling_def_success_rate": rng.uniform(0.3, 0.5, n),
        "opp_rolling_def_stuff_rate": rng.uniform(0.15, 0.25, n),
        "opp_rolling_off_success_rate": rng.uniform(0.3, 0.5, n),
        "opp_rolling_off_explosiveness": rng.uniform(1.1, 1.4, n),
        "opp_rolling_def_explosiveness": rng.uniform(1.1, 1.4, n),
        "opp_rolling_havoc_total": rng.uniform(0.1, 0.3, n),
        "opp_rolling_havoc_front_seven": rng.uniform(0.05, 0.2, n),
        "opp_rolling_havoc_db": rng.uniform(0.02, 0.1, n),
        "conf_tier_power4": rng.choice([0.0, 1.0], n),
        "conf_tier_g5": rng.choice([0.0, 1.0], n),
    })
    # League-average features
    for feat in C.CFB_SEASON_AVG_FEATURES:
        df[feat] = rng.normal(0, 0.1, n)
    return df


def test_contextual_model_fit_no_error():
    """Model must fit without raising an exception."""
    df = _make_model_df()
    model = ContextualAdjustmentModel(stat="pass_yards", position_group="QB")
    model.fit(df)
    assert model.pipeline is not None


def test_contextual_model_residuals_mean_near_zero():
    """Residuals on training data should have mean ≈ 0 (model fits mean)."""
    df = _make_model_df(n=120)
    model = ContextualAdjustmentModel(stat="pass_yards", position_group="QB")
    model.fit(df)
    residuals = model.compute_residuals(df)
    assert abs(residuals.mean()) < 5.0, (
        f"Residual mean {residuals.mean():.3f} too far from 0"
    )


def test_contextual_model_missing_feature_fills_zero():
    """
    Column present in training but absent from predict df imputes the training mean,
    not raise an exception.
    """
    df_train = _make_model_df(n=100)
    model = ContextualAdjustmentModel(stat="pass_yards", position_group="QB")
    model.fit(df_train)

    # Build predict df missing some contextual features
    df_pred = df_train[["pass_yards_p100", "pass_yards"]].copy()
    # Should not raise; predictions may be biased but not crash
    preds = model.predict(df_pred)
    assert not preds.isna().any(), "Predictions should not be NaN for missing features"


def test_contextual_model_cv_r2_is_clipped():
    """cv_r2 must be within [-2, 1] (clipped in fit())."""
    df = _make_model_df(n=60)
    model = ContextualAdjustmentModel(stat="pass_yards", position_group="QB")
    model.fit(df)
    if model.cv_r2 is not None:
        assert -2.0 <= model.cv_r2 <= 1.0, f"cv_r2={model.cv_r2} not in [-2, 1]"


def test_contextual_model_compute_residuals_missing_stat_col():
    """If stat column absent from df, compute_residuals returns a Series of NaN."""
    df_train = _make_model_df(n=100)
    model = ContextualAdjustmentModel(stat="pass_yards", position_group="QB")
    model.fit(df_train)

    df_pred = df_train.drop(columns=["pass_yards_p100", "pass_yards"])
    residuals = model.compute_residuals(df_pred)
    # Should be all NaN (raw - expected where raw = NaN)
    assert residuals.isna().all(), "Residuals should be NaN when stat col is missing"


# ── fit_contextual_models_for_group ──────────────────────────────────────────

def test_fit_models_for_group_returns_dict():
    df = _make_model_df(n=100)
    models = fit_contextual_models_for_group(
        bridge_seasons=df,
        team_ratings=pd.DataFrame(),
        stats=["pass_yards"],
        position_group="QB",
        save_dir=None,
    )
    # Should return a dict (may be empty if no save_dir provided — but shouldn't crash)
    assert isinstance(models, dict)


def test_fit_models_for_group_skips_thin_stats(tmp_path):
    """Stat with < 30 rows after NaN filtering → model skipped but no crash."""
    rng = np.random.default_rng(0)
    n = 100
    df = _make_model_df(n=n)
    # A new stat with mostly NaN
    df["rare_stat"] = np.where(rng.random(n) < 0.9, np.nan, rng.normal(0, 1, n))
    models = fit_contextual_models_for_group(
        bridge_seasons=df,
        team_ratings=pd.DataFrame(),
        stats=["pass_yards", "rare_stat"],
        position_group="QB",
        save_dir=tmp_path,
    )
    # rare_stat may or may not be in models, but no exception raised
    assert isinstance(models, dict)


def test_nullable_dtype_columns_fit_and_predict():
    """Nullable pandas dtypes (Float64/Int64 with pd.NA) must survive fit and
    predict — pd.NA breaks numpy.asarray(float) unless extraction goes through
    to_numpy(na_value=np.nan). Regression test for the 2026-07-20 step-4/5
    failures where nullable columns crashed every model refit."""
    rng = np.random.default_rng(7)
    n = 200
    df = _make_model_df(n=n, seed=7)
    df["temperature"] = pd.array(
        np.where(rng.random(n) < 0.25, np.nan, df["temperature"]), dtype="Float64")
    df["is_home"] = pd.array(df["is_home"].astype(int), dtype="Int64")

    model = ContextualAdjustmentModel("pass_yards", "QB")
    model.fit(df)
    preds = model.predict(df)
    assert np.isfinite(preds).all(), "predictions must be finite with pd.NA inputs"

    # NaN feature at predict time imputes the training mean (not raw 0)
    row_nan = df.head(1).copy()
    row_mean = df.head(1).copy()
    row_nan["temperature"] = pd.array([pd.NA], dtype="Float64")
    row_mean["temperature"] = float(
        model.pipeline.named_steps["imputer"].statistics_[
            model.feature_cols.index("temperature")])
    assert abs(model.predict(row_nan).iloc[0] - model.predict(row_mean).iloc[0]) < 1e-9
