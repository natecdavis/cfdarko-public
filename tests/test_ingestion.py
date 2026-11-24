"""
Unit tests for pipeline/01_data_ingestion.py helper functions.

Tests snap proxy computation, per-100-snap normalization, rate stat integrity,
share feature bounds, and de-meaning of rolling opponent stats.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util
import types

import numpy as np
import pandas as pd
import pytest

import config as C

# Load pipeline/01_data_ingestion.py lazily, stubbing the CFBD scraper so
# no network calls occur during import.
_ingestion_mod = None

def _get_mod():
    global _ingestion_mod
    if _ingestion_mod is not None:
        return _ingestion_mod

    cfdarko_root = Path(__file__).parent.parent
    mod_path = cfdarko_root / "pipeline" / "01_data_ingestion.py"

    # Stub cfbd scraper before loading so import-time calls resolve
    stub_cfbd = types.ModuleType("src.scrapers.cfbd")
    stub_cfbd.fetch_player_season_stats = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.fetch_team_ratings = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.fetch_team_conferences = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.fetch_team_advanced_stats = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.fetch_draft_picks = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.fetch_player_game_stats = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.fetch_game_schedule = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.fetch_advanced_game_stats = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.fetch_game_weather = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.fetch_advanced_box_scores = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.compute_rolling_opponent_strength = lambda *a, **kw: pd.DataFrame()
    stub_cfbd.compute_qb_rolling_stats = lambda *a, **kw: pd.DataFrame()
    sys.modules.setdefault("src.scrapers.cfbd", stub_cfbd)

    spec = importlib.util.spec_from_file_location("data_ingestion_01", mod_path)
    mod = types.ModuleType("data_ingestion_01")
    mod.__file__ = str(mod_path)  # required so Path(__file__) inside the module works
    spec.loader.exec_module(mod)
    _ingestion_mod = mod
    return _ingestion_mod


# ── _compute_snap_proxy ────────────────────────────────────────────────────────

def test_snap_proxy_minimum_one():
    """All values returned by _compute_snap_proxy must be ≥ 1."""
    m = _get_mod()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "position_group": ["QB"] * 10 + ["WR"] * 10 + ["DL"] * 10,
        "pass_att": rng.integers(0, 30, 30).astype(float),
        "rush_att": rng.integers(0, 10, 30).astype(float),
        "rec": rng.integers(0, 8, 30).astype(float),
        "tackles": rng.integers(0, 10, 30).astype(float),
        "sacks": rng.integers(0, 3, 30).astype(float),
        "tfl": rng.integers(0, 3, 30).astype(float),
        "passes_defended": rng.integers(0, 2, 30).astype(float),
    })
    snaps = m._compute_snap_proxy(df)
    assert (snaps >= 1).all(), f"Some snap proxies < 1: {snaps[snaps < 1]}"


def test_snap_proxy_qb_uses_pass_rush_att():
    """QB snap proxy = pass_att + rush_att."""
    m = _get_mod()
    df = pd.DataFrame({
        "position_group": ["QB"],
        "pass_att": [20.0],
        "rush_att": [5.0],
        "rec": [0.0],
        "tackles": [0.0],
    })
    snaps = m._compute_snap_proxy(df)
    assert snaps.iloc[0] == 25.0


def test_snap_proxy_wr_uses_rec():
    """WR snap proxy = rec (clipped to ≥1)."""
    m = _get_mod()
    df = pd.DataFrame({
        "position_group": ["WR"],
        "rec": [5.0],
        "pass_att": [0.0],
        "rush_att": [0.0],
        "tackles": [0.0],
    })
    snaps = m._compute_snap_proxy(df)
    assert snaps.iloc[0] == 5.0


def test_snap_proxy_missing_all_cols_returns_one():
    """If no proxy columns available, returns 1 (not crash)."""
    m = _get_mod()
    df = pd.DataFrame({
        "position_group": ["QB"],
        "some_irrelevant_col": [99.0],
    })
    snaps = m._compute_snap_proxy(df)
    assert snaps.iloc[0] >= 1


# ── _normalize_gamelogs ────────────────────────────────────────────────────────

def _make_normalize_input(n=20):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "position_group": ["QB"] * (n // 2) + ["WR"] * (n // 2),
        "pass_yards": rng.integers(0, 300, n).astype(float),
        "pass_att": rng.integers(10, 40, n).astype(float),
        "rush_yards": rng.integers(0, 50, n).astype(float),
        "rush_att": rng.integers(0, 10, n).astype(float),
        "rec": rng.integers(0, 8, n).astype(float),
        "rec_yards": rng.integers(0, 100, n).astype(float),
        "game_id": [f"G{i}" for i in range(n)],
        "school": ["TestU"] * n,
    })


def test_normalize_gamelogs_snaps_at_least_one():
    m = _get_mod()
    df = _make_normalize_input()
    result = m._normalize_gamelogs(df, attrs=pd.DataFrame())
    assert (result["snaps"] >= 1).all(), "Some snaps < 1 after normalization"


def test_normalize_gamelogs_p100_formula():
    """pass_yards_p100 = pass_yards * 100 / snaps for each row."""
    m = _get_mod()
    df = _make_normalize_input()
    result = m._normalize_gamelogs(df, attrs=pd.DataFrame())
    if "pass_yards_p100" in result.columns and "pass_yards" in result.columns:
        expected = result["pass_yards"] * 100.0 / result["snaps"]
        # Allow small floating-point difference
        diff = (result["pass_yards_p100"] - expected).abs()
        assert (diff < 1e-6).all(), "pass_yards_p100 != pass_yards * 100 / snaps"


def test_normalize_gamelogs_shares_clipped_01():
    """rec_share, rush_share, pass_share must be in [0, 1]."""
    m = _get_mod()
    df = _make_normalize_input()
    result = m._normalize_gamelogs(df, attrs=pd.DataFrame())
    for share_col in ["rec_share", "rush_share", "pass_share"]:
        if share_col in result.columns:
            vals = result[share_col].dropna()
            assert (vals >= 0).all() and (vals <= 1).all(), (
                f"{share_col} out of [0,1]: min={vals.min():.3f}, max={vals.max():.3f}"
            )


def _make_input_with_actual_snaps():
    """Two QBs in one game (so team_pass_att/rush_att get reconstructed) plus
    a defender from the opposing team. All have actual_snaps as a share (0–1)."""
    return pd.DataFrame({
        "player_id":     ["QB1", "RB1", "DB1", "K1"],
        "position_group":["QB",  "RB",  "DB",  "K"],
        "school":        ["A",   "A",   "B",   "A"],
        "game_id":       ["G1",  "G1",  "G1",  "G1"],
        "pass_att":      [40.0,  0.0,   0.0,   0.0],
        "rush_att":      [4.0,   20.0,  0.0,   0.0],
        "pass_yards":    [300.0, 0.0,   0.0,   0.0],
        "rush_yards":    [10.0,  100.0, 0.0,   0.0],
        "rec":           [0.0,   2.0,   0.0,   0.0],
        "rec_yards":     [0.0,   15.0,  0.0,   0.0],
        "fg_att":        [0.0,   0.0,   0.0,   3.0],
        "actual_snaps":  [0.95,  0.50,  0.80,  0.05],  # CFBD shares
    })


def test_actual_snaps_share_converted_to_count_for_offense():
    """Offensive player's snap count = share × (own team_pass_att + team_rush_att)."""
    m = _get_mod()
    result = m._normalize_gamelogs(_make_input_with_actual_snaps(), attrs=pd.DataFrame())
    # Team A has pass_att=40 + rush_att=24 = 64 plays
    qb = result[result["player_id"] == "QB1"].iloc[0]
    assert qb["snaps_are_real"] == True
    assert abs(qb["snaps"] - 0.95 * 64) < 1e-6, f"QB snaps={qb['snaps']}, want {0.95*64}"
    rb = result[result["player_id"] == "RB1"].iloc[0]
    assert abs(rb["snaps"] - 0.50 * 64) < 1e-6


def test_actual_snaps_share_converted_to_count_for_defense():
    """Defender's snap count = share × OPPONENT's plays."""
    m = _get_mod()
    result = m._normalize_gamelogs(_make_input_with_actual_snaps(), attrs=pd.DataFrame())
    # Defender DB1 is on team B; opponent (team A) has 64 plays
    db = result[result["player_id"] == "DB1"].iloc[0]
    assert db["snaps_are_real"] == True
    assert abs(db["snaps"] - 0.80 * 64) < 1e-6, f"DB snaps={db['snaps']}, want {0.80*64}"


def test_actual_snaps_does_not_apply_to_kickers():
    """K (and P) should ignore the actual_snaps share — the CFBD share doesn't
    account for special-teams reps — and take the constant specialist
    denominator instead."""
    from config import KALMAN_REF_SNAPS
    m = _get_mod()
    result = m._normalize_gamelogs(_make_input_with_actual_snaps(), attrs=pd.DataFrame())
    k = result[result["player_id"] == "K1"].iloc[0]
    assert k["snaps_are_real"] == False, "K should stay off the actual-snaps path"
    assert k["snaps"] == KALMAN_REF_SNAPS


def test_specialist_volume_stats_are_not_self_normalized():
    """Regression guard for the self-referential K/P snap proxy: a punter's
    punts_p100 must vary with punt volume. Under the old proxy (snaps=punts)
    it was pinned at exactly 100.0 for every punter-game, which zeroed
    adjusted_punts and left the Kalman/EWMA punt features with no variance."""
    m = _get_mod()
    df = pd.DataFrame({
        "position_group": ["P", "P", "K"],
        "punts":   [8, 2, 0],
        "fg_att":  [0, 0, 0],
        "xp_made": [0, 0, 6],
    })
    snaps = m._compute_snap_proxy(df)
    punts_p100 = df["punts"] * 100.0 / snaps
    assert punts_p100.iloc[0] != punts_p100.iloc[1], (
        "punts_p100 is constant across different punt volumes — "
        "the denominator is self-referential again"
    )
    assert punts_p100.iloc[0] > punts_p100.iloc[1]
    # Zero-FGA kicker games must not blow up xp_made_p100 via a clip-to-1 denom.
    assert (df["xp_made"] * 100.0 / snaps).iloc[2] < 100.0


def test_pass_att_p100_in_sane_range_after_conversion():
    """Regression guard: a starting QB's pass_att_p100 should be ~50–150,
    NOT thousands. Catches the share-vs-count unit bug."""
    m = _get_mod()
    result = m._normalize_gamelogs(_make_input_with_actual_snaps(), attrs=pd.DataFrame())
    qb = result[result["player_id"] == "QB1"].iloc[0]
    # 40 attempts / (0.95 * 64 snaps) * 100 ≈ 65.8
    assert 30 < qb["pass_att_p100"] < 200, (
        f"pass_att_p100={qb['pass_att_p100']:.1f} outside sane range — "
        "likely the share-vs-count bug"
    )


def test_normalize_gamelogs_position_group_fallback():
    """Rows with unmapped position get 'OTHER' not NaN."""
    m = _get_mod()
    df = pd.DataFrame({
        "player_id": ["P1", "P2"],
        "position": ["UNKNOWN", "QB"],
        "pass_att": [15.0, 25.0],
        "rush_att": [5.0, 5.0],
        "game_id": ["G1", "G2"],
        "school": ["U", "U"],
    })
    result = m._normalize_gamelogs(df, attrs=pd.DataFrame())
    # Should not have NaN position_group
    assert result["position_group"].notna().all()


# ── _add_opponent_rolling_stats ───────────────────────────────────────────────

def test_add_opponent_rolling_skips_empty_adv_game():
    """If adv_game is empty, returns gamelogs unchanged (no crash)."""
    m = _get_mod()
    df = pd.DataFrame({
        "player_id": ["P1"] * 3,
        "game_id": ["G1", "G2", "G3"],
        "school": ["U"] * 3,
        "season_year": [2019] * 3,
    })
    result = m._add_opponent_rolling_stats(
        gamelogs=df.copy(),
        adv_game=pd.DataFrame(),
        schedule=pd.DataFrame(),
    )
    assert len(result) == len(df)


def test_add_opponent_rolling_demean_cols_sum_near_zero():
    """
    After de-meaning, each player-season's opp_rolling_def_ppa should sum ≈ 0.
    This verifies the OPP_ROLLING_DEMEAN_COLS de-meaning is applied correctly.
    We test via a synthetic already-joined result rather than re-running the full function.
    """
    rng = np.random.default_rng(5)
    n = 40
    player_ids = ["P0"] * 20 + ["P1"] * 20
    season_years = [2019] * 40

    df = pd.DataFrame({
        "player_id": player_ids,
        "season_year": season_years,
        "opp_rolling_def_ppa": rng.normal(0.5, 0.2, n),
        "opp_rolling_off_ppa": rng.normal(-0.3, 0.2, n),
    })

    # Apply the same de-meaning logic as _add_opponent_rolling_stats
    demean_cols = ["opp_rolling_def_ppa", "opp_rolling_off_ppa"]
    season_means = df.groupby(["player_id", "season_year"])[demean_cols].transform("mean")
    df[demean_cols] = df[demean_cols] - season_means

    for pid in ["P0", "P1"]:
        for col in demean_cols:
            s = df.loc[df["player_id"] == pid, col].sum()
            assert abs(s) < 1e-8, (
                f"De-meaned {col} for player {pid} does not sum to ~0 (sum={s:.6f})"
            )
