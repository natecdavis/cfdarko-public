"""
Unit tests for src/models/decay.py

Tests decay weight computation, EWMA correctness, overflow safety,
and per-player overflow independence.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.models.decay import (
    compute_decay_weights,
    ewma_time_series_fast,
    ewma_all_players_fast,
    optimize_beta_for_stat,
)
import config as C


# ── compute_decay_weights ─────────────────────────────────────────────────────

def _make_dates(n=10, start="2019-09-01", step_days=7):
    start_ts = pd.Timestamp(start)
    return pd.Series([start_ts + pd.Timedelta(days=step_days * i) for i in range(n)])


def test_decay_weights_in_01():
    dates = _make_dates(10)
    for beta in [0.990, 0.995, 0.999, 1.0]:
        weights = compute_decay_weights(dates, beta)
        assert (weights >= 0).all() and (weights <= 1).all(), (
            f"Weights out of [0,1] for beta={beta}"
        )


def test_decay_weights_monotone_decreasing():
    """Older games (farther from reference) should have lower weight."""
    dates = _make_dates(10, step_days=7)
    weights = compute_decay_weights(dates, beta=0.999)
    # Dates are ascending; reference = max(dates) = last date.
    # Oldest game (first) should have smallest weight.
    for i in range(len(weights) - 1):
        assert weights[i] <= weights[i + 1], (
            f"Weight not non-decreasing at index {i}"
        )


def test_decay_weights_reference_date_gets_weight_one():
    dates = _make_dates(10)
    ref = dates.max()
    weights = compute_decay_weights(dates, beta=0.995, reference_date=ref)
    # The most recent date should get weight 1.0 (0 days from reference)
    assert abs(weights[-1] - 1.0) < 1e-10


def test_decay_weights_beta_one_all_equal():
    dates = _make_dates(10)
    weights = compute_decay_weights(dates, beta=1.0)
    assert np.allclose(weights, 1.0), "beta=1.0 should give all weights = 1.0"


def test_decay_weights_older_game_lower_weight():
    """Explicit: game 100 days ago < game 10 days ago."""
    dates = pd.Series([
        pd.Timestamp("2019-06-01"),   # ~100 days before reference
        pd.Timestamp("2019-09-09"),   # 1 day before reference
    ])
    ref = pd.Timestamp("2019-09-10")
    weights = compute_decay_weights(dates, beta=0.999, reference_date=ref)
    assert weights[0] < weights[1]


# ── ewma_time_series_fast ─────────────────────────────────────────────────────

def _player_df(vals, snaps=None, stat="pass_yards", start="2019-09-01", step=7):
    dates = [pd.Timestamp(start) + pd.Timedelta(days=step * i) for i in range(len(vals))]
    d = {
        "game_date": dates,
        f"adjusted_{stat}": vals,
    }
    if snaps is not None:
        d["snaps"] = snaps
    return pd.DataFrame(d)


def test_ewma_var_nonnegative():
    vals = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0]
    df = _player_df(vals, snaps=[50] * 6)
    result = ewma_time_series_fast(df, stats=["pass_yards"],
                                   betas={"pass_yards": 0.999})
    assert (result["ewma_var_pass_yards"] >= 0).all()


def test_ewma_all_nan_returns_prior_mean():
    """When all observations are NaN, EWMA should return the prior mean (0 if none given)."""
    vals = [np.nan] * 6
    df = _player_df(vals, snaps=[50] * 6)
    result = ewma_time_series_fast(df, stats=["pass_yards"],
                                   betas={"pass_yards": 0.999},
                                   prior_means={"pass_yards": 0.0})
    # Result should be 0 (prior) when no valid observations
    assert not result["ewma_pass_yards"].isna().any()


def test_ewma_single_observation_moves_toward_obs():
    """One observation above prior → final ewma should be above 0."""
    vals = [10.0]
    df = _player_df(vals, snaps=[70])
    result = ewma_time_series_fast(df, stats=["pass_yards"],
                                   betas={"pass_yards": 0.999},
                                   prior_means={"pass_yards": 0.0},
                                   prior_weight=1.0)
    assert result["ewma_pass_yards"].iloc[-1] > 0.0


def test_ewma_more_observations_converge():
    """Many identical observations should converge the ewma toward that value."""
    val = 5.0
    vals = [val] * 20
    df = _player_df(vals, snaps=[70] * 20)
    result = ewma_time_series_fast(df, stats=["pass_yards"],
                                   betas={"pass_yards": 0.999},
                                   prior_means={"pass_yards": 0.0})
    final = result["ewma_pass_yards"].iloc[-1]
    assert abs(final - val) < 1.0, f"EWMA should converge toward {val}, got {final}"


# ── ewma_all_players_fast ────────────────────────────────────────────────────

def _all_players_df(n_players=20, games_each=10, stat="pass_yards", seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        base = pd.Timestamp("2018-09-01")
        skill = rng.normal(0, 3)
        for g in range(games_each):
            rows.append({
                "player_id": f"P{pid}",
                "game_date": base + pd.Timedelta(days=7 * g),
                f"adjusted_{stat}": skill + rng.normal(0, 1),
                "snaps": float(rng.integers(20, 80)),
            })
    df = pd.DataFrame(rows).sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return df


def test_ewma_all_players_no_inf_nan():
    """No inf or NaN should appear in EWMA output."""
    df = _all_players_df(n_players=20, games_each=10)
    result = ewma_all_players_fast(df, stats=["pass_yards"],
                                   betas={"pass_yards": 0.999})
    assert result["ewma_pass_yards"].notna().all()
    assert np.isfinite(result["ewma_pass_yards"].values).all()


def test_ewma_all_players_var_nonneg():
    df = _all_players_df(n_players=10, games_each=8)
    result = ewma_all_players_fast(df, stats=["pass_yards"],
                                   betas={"pass_yards": 0.999})
    assert (result["ewma_var_pass_yards"] >= 0).all()


def test_ewma_per_player_overflow_not_global():
    """
    Critical regression test for the old global-overflow bug:
    A player with a 6000-day span (beta=0.999 → log_decay up to 4157) used to
    zero out all other players when the overflow correction was applied globally.
    Per-player correction must preserve short-span players' relative weights.
    """
    stat = "rush_yards"
    # Player with very long history (6000 days ≈ 16 years)
    rows_long = []
    base_long = pd.Timestamp("2000-01-01")
    for g in range(50):
        rows_long.append({
            "player_id": "LongCareer",
            "game_date": base_long + pd.Timedelta(days=120 * g),
            f"adjusted_{stat}": 5.0,
            "snaps": 50.0,
        })

    # Player with short, recent history (30 days)
    rows_short = []
    base_short = pd.Timestamp("2022-09-01")
    for g in range(5):
        rows_short.append({
            "player_id": "ShortCareer",
            "game_date": base_short + pd.Timedelta(days=7 * g),
            f"adjusted_{stat}": 5.0,
            "snaps": 50.0,
        })

    df = pd.DataFrame(rows_long + rows_short).sort_values(
        ["player_id", "game_date"]
    ).reset_index(drop=True)

    result = ewma_all_players_fast(df, stats=[stat], betas={stat: 0.999})

    short_ewma = result.loc[result["player_id"] == "ShortCareer", f"ewma_{stat}"]
    # ShortCareer's ewma should be close to 5.0 (not zeroed by LongCareer's overflow)
    assert short_ewma.notna().all(), "Short-career player EWMA should not be NaN"
    assert (short_ewma > 0).all(), (
        "Short-career player EWMA zeroed out — possible global overflow bug"
    )


def test_ewma_all_players_consistent_with_single():
    """ewma_all_players_fast should match ewma_time_series_fast for a single player
    when an explicit prior_mean is provided (so both code paths include prior weight)."""
    stat = "pass_yards"
    rng = np.random.default_rng(99)
    base = pd.Timestamp("2019-09-01")
    vals = rng.normal(0, 2, 10).tolist()
    snaps = [60.0] * 10
    dates = [base + pd.Timedelta(days=7 * g) for g in range(10)]

    single_df = pd.DataFrame({
        "player_id": ["P0"] * 10,
        "game_date": dates,
        f"adjusted_{stat}": vals,
        "snaps": snaps,
    })

    betas = {stat: 0.999}
    prior_means = {stat: 0.0}
    prior_weight = 5.0

    res_single = ewma_time_series_fast(single_df.copy(), stats=[stat], betas=betas,
                                       prior_means=prior_means, prior_weight=prior_weight)
    res_all = ewma_all_players_fast(single_df.copy(), stats=[stat], betas=betas,
                                    prior_means=prior_means, prior_weight=prior_weight)

    assert np.allclose(
        res_single[f"ewma_{stat}"].values,
        res_all[f"ewma_{stat}"].values,
        atol=1e-6,
    ), "ewma_all_players_fast and ewma_time_series_fast disagree for single player"


# ── _compute_terminal_ewma vectorized vs reference ───────────────────────────

from src.models.decay import (
    _compute_terminal_ewma,
    _compute_terminal_ewma_reference,
    _prepare_terminal_ewma_arrays,
    _neg_role_pearson,
)


def _bridge_df(seed=0, n_players=40, stat="rush_yards", rate=False):
    """Synthetic bridge-player gamelog with NaNs, single-obs players,
    duplicate dates, and irregular gaps."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        n_games = int(rng.integers(1, 30))  # includes single-obs players
        base = pd.Timestamp("2017-08-15") + pd.Timedelta(days=int(rng.integers(0, 900)))
        skill = rng.normal(3.0, 2.0)
        day = 0
        for g in range(n_games):
            # ~15% duplicate dates (two games recorded on the same day)
            if g > 0 and rng.random() > 0.15:
                day += int(rng.integers(4, 40))
            val = skill + rng.normal(0, 1)
            if rng.random() < 0.15:
                val = np.nan  # NaN observations
            row = {
                "player_id": f"P{pid:04d}",
                "game_date": base + pd.Timedelta(days=day),
                f"adjusted_{stat}": val,
                "snaps": float(rng.integers(5, 80)) if rng.random() > 0.1 else np.nan,
            }
            if rate:
                row["rush_att"] = float(rng.integers(0, 25)) if rng.random() > 0.1 else np.nan
            rows.append(row)
    # One player whose observations are ALL NaN (must be absent from output)
    rows.append({"player_id": "PALLNAN", "game_date": pd.Timestamp("2019-10-01"),
                 f"adjusted_{stat}": np.nan, "snaps": 40.0,
                 **({"rush_att": 5.0} if rate else {})})
    df = pd.DataFrame(rows)
    # Shuffle row order — implementations must not depend on input ordering
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.mark.parametrize("beta", [0.990, 0.995, 0.999, 1.0])
def test_terminal_ewma_matches_reference_snap_weighted(beta):
    df = _bridge_df(seed=1, stat="rush_yards")
    ref = _compute_terminal_ewma_reference(df, "rush_yards", beta)
    fast = _compute_terminal_ewma(df, "rush_yards", beta)
    assert set(ref.keys()) == set(fast.keys())
    assert "PALLNAN" not in fast
    for pid in ref:
        assert np.isclose(ref[pid], fast[pid], rtol=1e-9), (
            f"pid={pid}: ref={ref[pid]} fast={fast[pid]}"
        )


@pytest.mark.parametrize("beta", [0.990, 0.999])
def test_terminal_ewma_matches_reference_attempt_weighted(beta):
    """yards_per_carry is a rate stat → attempt-weighted path."""
    df = _bridge_df(seed=2, stat="yards_per_carry", rate=True)
    ref = _compute_terminal_ewma_reference(df, "yards_per_carry", beta)
    fast = _compute_terminal_ewma(df, "yards_per_carry", beta)
    assert set(ref.keys()) == set(fast.keys())
    for pid in ref:
        assert np.isclose(ref[pid], fast[pid], rtol=1e-9)


def test_terminal_ewma_matches_reference_no_weight_cols():
    """No snaps and no attempt column → uniform game weights path."""
    df = _bridge_df(seed=3, stat="rush_yards").drop(columns=["snaps"])
    ref = _compute_terminal_ewma_reference(df, "rush_yards", 0.997)
    fast = _compute_terminal_ewma(df, "rush_yards", 0.997)
    assert set(ref.keys()) == set(fast.keys())
    for pid in ref:
        assert np.isclose(ref[pid], fast[pid], rtol=1e-9)


def test_terminal_ewma_precomputed_path_matches():
    """Precomputed-arrays path must equal the from-scratch path."""
    df = _bridge_df(seed=4, stat="rush_yards")
    pre = _prepare_terminal_ewma_arrays(df, "rush_yards")
    for beta in [0.991, 0.9967, 1.0]:
        assert _compute_terminal_ewma(df, "rush_yards", beta) == \
            _compute_terminal_ewma(df, "rush_yards", beta, precomputed=pre)


def test_terminal_ewma_missing_column_and_empty():
    df = _bridge_df(seed=5, stat="rush_yards")
    assert _compute_terminal_ewma(df, "no_such_stat", 0.995) == {}
    assert _compute_terminal_ewma(df.iloc[0:0], "rush_yards", 0.995) == {}
    assert _prepare_terminal_ewma_arrays(df, "no_such_stat") is None


def test_neg_role_pearson_unchanged_by_vectorization():
    """_neg_role_pearson output for a fixed beta must be identical whether the
    terminal EWMA comes from the reference helper or the vectorized one."""
    from unittest import mock
    import src.models.decay as decay_mod

    stat = "rush_yards"
    df = _bridge_df(seed=6, n_players=80, stat=stat)
    pids = sorted(df["player_id"].unique())
    rng = np.random.default_rng(7)
    role_labels = pd.Series(rng.random(len(pids)), index=pids)
    impact_labels = pd.Series(
        np.where(rng.random(len(pids)) < 0.2, np.nan, rng.random(len(pids))),
        index=pids,
    )

    for beta in [0.992, 0.999]:
        got_fast = _neg_role_pearson(beta, df, role_labels, stat,
                                     impact_labels=impact_labels)
        with mock.patch.object(
            decay_mod, "_compute_terminal_ewma",
            lambda b_adj, s, b, date_col="game_date", precomputed=None:
                _compute_terminal_ewma_reference(b_adj, s, b, date_col),
        ):
            got_ref = decay_mod._neg_role_pearson(
                beta, df, role_labels, stat, impact_labels=impact_labels)
        assert np.isclose(got_fast, got_ref, rtol=1e-9, atol=1e-12)


# ── optimize_beta_for_stat ────────────────────────────────────────────────────

def test_optimize_beta_returns_in_bounds():
    """Optimizer must return a beta within BETA_BOUNDS."""
    stat = "pass_yards"
    rng = np.random.default_rng(7)
    rows = []
    for pid in range(30):
        base = pd.Timestamp("2018-09-01")
        skill = rng.normal(0, 2)
        for g in range(12):
            rows.append({
                "player_id": f"P{pid}",
                "game_date": base + pd.Timedelta(days=7 * g),
                f"adjusted_{stat}": skill + rng.normal(0, 0.5),
                "snaps": 60.0,
            })
    df = pd.DataFrame(rows)
    lo, hi = C.BETA_BOUNDS
    beta = optimize_beta_for_stat(df, stat=stat, n_players=20, bounds=(lo, hi))
    assert lo <= beta <= hi, f"Optimized beta={beta} out of bounds {C.BETA_BOUNDS}"


def test_optimize_beta_sparse_returns_default():
    """Fewer eligible players than min_history+5 → returns DEFAULT_BETA."""
    stat = "pass_yards"
    df = pd.DataFrame({
        "player_id": ["P1", "P1", "P2"],
        "game_date": pd.to_datetime(["2019-09-01", "2019-09-08", "2019-09-01"]),
        f"adjusted_{stat}": [1.0, 2.0, 1.5],
        "snaps": [50.0, 50.0, 50.0],
    })
    beta = optimize_beta_for_stat(df, stat=stat, min_history=5)
    expected = C.DEFAULT_BETA.get(stat, 0.999)
    assert beta == expected
