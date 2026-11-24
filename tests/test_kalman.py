"""
Unit tests for src/models/kalman.py

Tests invariants: variance monotonicity, Kalman gain bounds, NaN/zero handling,
prior estimation, transfer shock, age multiplier, and batch processing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.models.kalman import (
    KalmanState,
    kalman_predict,
    kalman_update,
    apply_transfer_shock,
    _age_q_multiplier,
    estimate_prior_vars,
    run_kalman_all_players,
)
import config as C


# ── kalman_predict ─────────────────────────────────────────────────────────────

def test_predict_variance_increases():
    state = KalmanState(mean=0.0, var=1.0)
    new = kalman_predict(state, days_elapsed=30)
    assert new.var > state.var


def test_predict_variance_proportional_to_days():
    state = KalmanState(mean=0.0, var=1.0)
    new_30 = kalman_predict(state, days_elapsed=30)
    new_60 = kalman_predict(state, days_elapsed=60)
    # Twice the days → twice the added noise
    assert abs((new_60.var - state.var) - 2 * (new_30.var - state.var)) < 1e-10


def test_predict_zero_days_unchanged():
    state = KalmanState(mean=5.0, var=2.0)
    new = kalman_predict(state, days_elapsed=0)
    assert new.var == state.var
    assert new.mean == state.mean


def test_predict_negative_days_clamped():
    state = KalmanState(mean=0.0, var=1.0)
    neg = kalman_predict(state, days_elapsed=-10)
    zero = kalman_predict(state, days_elapsed=0)
    assert neg.var == zero.var


def test_predict_preserves_mean():
    state = KalmanState(mean=3.7, var=1.0)
    new = kalman_predict(state, days_elapsed=100)
    assert new.mean == state.mean


# ── kalman_update ─────────────────────────────────────────────────────────────

def test_update_variance_decreases():
    state = KalmanState(mean=0.0, var=2.0)
    new = kalman_update(state, observation=1.0, obs_noise=1.0, weight=1.0)
    assert new.var < state.var


def test_update_variance_nonnegative():
    state = KalmanState(mean=0.0, var=0.01)
    new = kalman_update(state, observation=100.0, obs_noise=0.001, weight=1.0)
    assert new.var >= 0.0


def test_update_kalman_gain_in_01():
    # K = var / (var + R_eff); must be in [0, 1]
    for var in [0.1, 1.0, 10.0]:
        for R in [0.01, 1.0, 100.0]:
            state = KalmanState(mean=0.0, var=var)
            new = kalman_update(state, observation=5.0, obs_noise=R, weight=1.0,
                                innovation_clip_sigma=None)
            K = 1.0 - new.var / state.var   # K = 1 - (1-K) where new_var = (1-K)*var
            assert 0.0 <= K <= 1.0, f"K={K} out of [0,1] for var={var}, R={R}"


def test_update_nan_observation_no_change():
    state = KalmanState(mean=3.0, var=2.0, n_obs=5)
    new = kalman_update(state, observation=float("nan"), obs_noise=1.0)
    assert new.mean == state.mean
    assert new.var == state.var
    assert new.n_obs == state.n_obs


def test_update_zero_weight_no_change():
    state = KalmanState(mean=3.0, var=2.0)
    new = kalman_update(state, observation=100.0, obs_noise=1.0, weight=0.0)
    assert new.mean == state.mean
    assert new.var == state.var


def test_update_negative_weight_no_change():
    state = KalmanState(mean=3.0, var=2.0)
    new = kalman_update(state, observation=100.0, obs_noise=1.0, weight=-1.0)
    assert new.mean == state.mean
    assert new.var == state.var


def test_update_innovation_clip():
    """Extreme observation should be clipped to clip_sigma * sqrt(R_eff)."""
    state = KalmanState(mean=0.0, var=100.0)
    obs_noise = 1.0
    clip_sigma = 3.0
    # sqrt(R_eff / var) * clip_sigma bounds the mean shift
    new_clipped = kalman_update(state, observation=1e6, obs_noise=obs_noise,
                                weight=1.0, innovation_clip_sigma=clip_sigma)
    new_unclipped = kalman_update(state, observation=1e6, obs_noise=obs_noise,
                                  weight=1.0, innovation_clip_sigma=None)
    # Clipped version should move the mean less
    assert abs(new_clipped.mean) < abs(new_unclipped.mean)


def test_update_n_obs_increments():
    state = KalmanState(mean=0.0, var=1.0, n_obs=0)
    new = kalman_update(state, observation=1.0, obs_noise=1.0)
    assert new.n_obs == 1


def test_update_high_weight_lower_effective_noise():
    """Higher weight → lower effective R → larger Kalman gain → more update."""
    state = KalmanState(mean=0.0, var=2.0)
    new_low_w = kalman_update(state, observation=5.0, obs_noise=1.0, weight=0.5,
                              innovation_clip_sigma=None)
    new_high_w = kalman_update(state, observation=5.0, obs_noise=1.0, weight=2.0,
                               innovation_clip_sigma=None)
    assert abs(new_high_w.mean) > abs(new_low_w.mean)


# ── apply_transfer_shock ──────────────────────────────────────────────────────

def test_transfer_shock_increases_variance():
    state = KalmanState(mean=1.0, var=2.0)
    multiplier = 2.5
    new = apply_transfer_shock(state, multiplier=multiplier)
    assert abs(new.var - state.var * multiplier) < 1e-10


def test_transfer_shock_preserves_mean():
    state = KalmanState(mean=3.7, var=1.0)
    new = apply_transfer_shock(state, multiplier=2.0)
    assert new.mean == state.mean


# ── _age_q_multiplier ─────────────────────────────────────────────────────────

def test_age_multiplier_young_player():
    young_age = C.AGE_Q_REFERENCE_AGE - 3
    mult = _age_q_multiplier(young_age)
    assert mult > 1.0


def test_age_multiplier_reference_age():
    mult = _age_q_multiplier(C.AGE_Q_REFERENCE_AGE)
    assert abs(mult - 1.0) < 1e-10


def test_age_multiplier_old_player_floor():
    old_age = C.AGE_Q_REFERENCE_AGE + 10
    mult = _age_q_multiplier(old_age)
    assert mult == 1.0


def test_age_multiplier_bounded_by_max():
    very_young = C.AGE_Q_REFERENCE_AGE - 20
    mult = _age_q_multiplier(very_young)
    assert mult <= C.AGE_Q_MAX_MULTIPLIER


def test_age_multiplier_monotone_decreasing():
    ages = [18, 19, 20, 21, 22, 23, 24]
    mults = [_age_q_multiplier(a) for a in ages]
    for i in range(len(mults) - 1):
        assert mults[i] >= mults[i + 1], f"Multiplier not non-increasing at age {ages[i]}"


# ── estimate_prior_vars ───────────────────────────────────────────────────────

def _make_adjusted_df(stat="pass_yards", n_players=20, games_per_player=8, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        true_skill = rng.normal(0, 3)
        for _ in range(games_per_player):
            rows.append({
                "player_id": f"P{pid}",
                f"adjusted_{stat}": true_skill + rng.normal(0, 1),
            })
    return pd.DataFrame(rows)


def test_estimate_prior_vars_positive():
    df = _make_adjusted_df(n_players=20, games_per_player=8)
    result = estimate_prior_vars(df, stats=["pass_yards"])
    assert "pass_yards" in result
    assert result["pass_yards"] >= 0.01   # floored at 0.01


def test_estimate_prior_vars_sparse_uses_config_default():
    """Fewer than 10 qualifying players → fall back to config default."""
    df = _make_adjusted_df(n_players=5, games_per_player=8)
    result = estimate_prior_vars(df, stats=["pass_yards"])
    expected = C.KALMAN_PRIOR_VAR_BY_STAT.get("pass_yards", C.KALMAN_PRIOR_VAR)
    assert result["pass_yards"] == expected


def test_estimate_prior_vars_missing_column_uses_default():
    """Missing adjusted_{stat} column → config default."""
    df = pd.DataFrame({"player_id": ["P1", "P2"]})
    result = estimate_prior_vars(df, stats=["pass_yards"])
    assert "pass_yards" in result


# ── run_kalman_all_players ────────────────────────────────────────────────────

def _make_kalman_input(n_players=10, games_per_player=8, stat="pass_yards", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        base_date = pd.Timestamp("2018-09-01")
        for g in range(games_per_player):
            rows.append({
                "player_id": f"P{pid}",
                "game_date": base_date + pd.Timedelta(days=7 * g),
                "season_year": 2018,
                "school": "TestU",
                "snaps": float(rng.integers(20, 80)),
                "snaps_are_real": True,
                f"adjusted_{stat}": rng.normal(0, 3),
            })
    return pd.DataFrame(rows)


def test_run_kalman_output_shape():
    stat = "pass_yards"
    df = _make_kalman_input(n_players=5, games_per_player=6)
    betas = {stat: 0.999}
    result = run_kalman_all_players(df, decay_betas=betas, stats=[stat])
    assert len(result) == len(df)


def test_run_kalman_variances_positive():
    stat = "pass_yards"
    df = _make_kalman_input(n_players=5, games_per_player=6)
    result = run_kalman_all_players(df, decay_betas={stat: 0.999}, stats=[stat])
    var_col = f"kalman_var_{stat}"
    assert var_col in result.columns
    assert (result[var_col] > 0).all(), "Some Kalman variances are non-positive"


def test_run_kalman_no_nan_means():
    """kalman_mean must not be NaN for any row — unobserved players get prior."""
    stat = "pass_yards"
    df = _make_kalman_input(n_players=5, games_per_player=6)
    result = run_kalman_all_players(df, decay_betas={stat: 0.999}, stats=[stat])
    mean_col = f"kalman_mean_{stat}"
    assert result[mean_col].notna().all(), "NaN kalman_mean for some rows"


def test_run_kalman_prior_for_no_obs_player():
    """Player whose adjusted_{stat} is all NaN stays at prior mean."""
    stat = "pass_yards"
    df = _make_kalman_input(n_players=3, games_per_player=5)
    # Wipe one player's observations
    df.loc[df["player_id"] == "P2", f"adjusted_{stat}"] = np.nan
    result = run_kalman_all_players(df, decay_betas={stat: 0.999}, stats=[stat])
    prior_mean = C.KALMAN_PRIOR_MEAN
    p2_means = result.loc[result["player_id"] == "P2", f"kalman_mean_{stat}"]
    assert (p2_means == prior_mean).all(), "No-obs player should stay at prior mean"


def test_run_kalman_variance_shrinks_with_more_games():
    """Player with more observations should have lower final variance."""
    stat = "pass_yards"
    rows_few = []
    rows_many = []
    for g in range(3):
        rows_few.append({
            "player_id": "Few", "game_date": pd.Timestamp("2018-09-01") + pd.Timedelta(days=7*g),
            "school": "U", "snaps": 50.0, "snaps_are_real": True, f"adjusted_{stat}": 1.0
        })
    for g in range(12):
        rows_many.append({
            "player_id": "Many", "game_date": pd.Timestamp("2018-09-01") + pd.Timedelta(days=7*g),
            "school": "U", "snaps": 50.0, "snaps_are_real": True, f"adjusted_{stat}": 1.0
        })
    df = pd.DataFrame(rows_few + rows_many)
    result = run_kalman_all_players(df, decay_betas={stat: 0.999}, stats=[stat])
    var_few = result.loc[result["player_id"] == "Few", f"kalman_var_{stat}"].iloc[-1]
    var_many = result.loc[result["player_id"] == "Many", f"kalman_var_{stat}"].iloc[-1]
    assert var_many < var_few, "More games should reduce Kalman variance"


def test_run_kalman_pre2016_higher_noise():
    """Pre-2016 games (snaps_are_real=False) should get higher obs noise,
    so variance decreases less per game than with real snaps."""
    stat = "pass_yards"
    base_date = pd.Timestamp("2013-09-01")
    rows_real = []
    rows_proxy = []
    for g in range(5):
        dt = base_date + pd.Timedelta(days=7 * g)
        rows_real.append({
            "player_id": "Real", "game_date": dt, "school": "U",
            "snaps": 70.0, "snaps_are_real": True, f"adjusted_{stat}": 0.0
        })
        rows_proxy.append({
            "player_id": "Proxy", "game_date": dt, "school": "U",
            "snaps": 70.0, "snaps_are_real": False, f"adjusted_{stat}": 0.0
        })
    df = pd.DataFrame(rows_real + rows_proxy)
    result = run_kalman_all_players(df, decay_betas={stat: 0.999}, stats=[stat])
    var_real  = result.loc[result["player_id"] == "Real",  f"kalman_var_{stat}"].iloc[-1]
    var_proxy = result.loc[result["player_id"] == "Proxy", f"kalman_var_{stat}"].iloc[-1]
    # Proxy snaps → higher effective obs noise → less variance reduction per game
    assert var_proxy > var_real, (
        "Pre-2016 proxy snaps should result in higher final variance than real snaps"
    )
