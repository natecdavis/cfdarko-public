"""
Bayesian-shrinkage triplet aggregation in `load_pff_career_features`.

Guards the contract:
  - Shrinkage formula: shrunk = (n*obs + k*prior) / (n+k)
  - Two-family k: grade columns use a smaller factor than rate columns
  - Snap-weighted mean across seasons
  - Peak is taken AFTER shrinkage (small-sample outliers can't dominate)
  - Volume stats (no denominator mapping) keep the legacy unweighted triplet
  - Seasons with n in {NaN, 0} are excluded from shrinkage aggregation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scrapers.pff import (
    _compute_position_priors,
    _is_grade_col,
    _is_shrunk_stat,
    _resolve_prior,
    _shrinkage_k_for,
    load_pff_career_features,
)
import config


# ── Formula primitives ──────────────────────────────────────────────────────────

def test_is_grade_col_detection():
    assert _is_grade_col("pff_grades_pass")
    assert _is_grade_col("pff_grades_offense")
    assert _is_grade_col("pff_man_grades_pass_route")
    assert _is_grade_col("pff_pressure_grades_pass")
    assert _is_grade_col("pff_grades_fgep_kicker")
    assert _is_grade_col("pff_grades_punter")
    assert not _is_grade_col("pff_yprr")
    assert not _is_grade_col("pff_sacks_per_pr_snap")
    assert not _is_grade_col("pff_routes")


def test_shrinkage_k_two_family():
    """Grade-family k should be strictly smaller than rate-family k for stats
    that share a denominator scale."""
    k_grade = _shrinkage_k_for("pff_grades_pass_route")          # denom=pff_routes
    k_rate  = _shrinkage_k_for("pff_yprr")                        # denom=pff_routes
    assert k_grade < k_rate
    # Both should be a fraction of the routes reference (300.0 in config).
    ref_routes = config.PFF_REF_DENOM_BY_STAT["pff_routes"]
    assert k_grade == pytest.approx(config.PFF_SHRINKAGE_K_FACTOR_GRADES * ref_routes)
    assert k_rate  == pytest.approx(config.PFF_SHRINKAGE_K_FACTOR_RATE   * ref_routes)


def test_is_shrunk_stat_excludes_volume():
    # Rate / grade: shrunk
    assert _is_shrunk_stat("pff_grades_pass_route")
    assert _is_shrunk_stat("pff_yprr")
    # Volume / denominator-stats themselves: NOT shrunk
    assert not _is_shrunk_stat("pff_snap_counts_offense")   # mapped to None
    assert not _is_shrunk_stat("pff_routes")                 # mapped to None
    # Unknown stat: NOT shrunk
    assert not _is_shrunk_stat("pff_made_up_stat")


# ── Prior calculation ──────────────────────────────────────────────────────────

def _build_panel():
    """Minimal panel: three WRs, each with two seasons of pff_yprr + pff_routes."""
    return pd.DataFrame({
        "player_id":       ["p1", "p1", "p2", "p2", "p3", "p3"],
        "season_year":     [2019, 2020, 2019, 2020, 2019, 2020],
        "position_group":  ["WR"] * 6,
        "pff_position":    ["WR"] * 6,
        "pff_yprr":        [1.0, 2.0, 0.5, 1.5, 3.0, 1.0],
        "pff_routes":      [100, 400, 50,  150, 200, 100],
    })


def test_compute_position_priors_denom_weighted():
    """Prior = Σ(n*val) / Σ(n) within (pos_group, stat)."""
    df = _build_panel()
    pos_priors, fallback = _compute_position_priors(df, ["pff_yprr"])
    # Σ(n*val) = 100*1 + 400*2 + 50*0.5 + 150*1.5 + 200*3 + 100*1 = 1850
    # Σ(n)     = 100 + 400 + 50 + 150 + 200 + 100 = 1000
    expected = 1850 / 1000.0
    assert pos_priors[("WR", "pff_yprr")] == pytest.approx(expected)
    # Fallback (un-stratified) matches when only one position group exists.
    assert fallback["pff_yprr"] == pytest.approx(expected)


def test_compute_position_priors_skips_volume_stats():
    """Volume stats (no denominator mapping) get no prior — they're not shrunk."""
    df = _build_panel()
    df["pff_total_pressures"] = [3, 5, 1, 2, 8, 4]
    pos_priors, fallback = _compute_position_priors(
        df, ["pff_yprr", "pff_total_pressures"]
    )
    assert ("WR", "pff_yprr") in pos_priors
    assert ("WR", "pff_total_pressures") not in pos_priors
    assert "pff_total_pressures" not in fallback


def test_resolve_prior_falls_back_to_global():
    """If the (pos_group, stat) key is missing, fall back to the un-stratified
    fallback, then to 0.0."""
    pos_priors = {("WR", "pff_yprr"): 1.2}
    fallback = {"pff_yprr": 1.0}
    # Direct hit
    assert _resolve_prior("pff_yprr", "WR", pos_priors, fallback) == 1.2
    # Missing pos_group → fallback
    assert _resolve_prior("pff_yprr", "TE", pos_priors, fallback) == 1.0
    # Missing stat entirely → 0.0
    assert _resolve_prior("pff_unknown", "WR", pos_priors, fallback) == 0.0
    # NaN/None pos_group → fallback
    assert _resolve_prior("pff_yprr", None, pos_priors, fallback) == 1.0


# ── End-to-end aggregation ─────────────────────────────────────────────────────

def _write_panel_and_load(tmp_path, panel: pd.DataFrame) -> pd.DataFrame:
    """Save a synthetic panel and run the full aggregator on it."""
    panel.to_parquet(tmp_path / "pff_season_grades.parquet", index=False)
    snap = pd.DataFrame({"player_id": panel["player_id"].unique()})
    return load_pff_career_features(snap, tmp_path)


def test_shrinkage_formula_extremes(tmp_path):
    """n=0 → shrunk = prior; n → ∞ → shrunk → obs."""
    panel = _build_panel()
    # Add an extra player who is grade-but-zero-routes — should be NaN
    # (we exclude n in {NaN, 0} from the shrinkage aggregation).
    extra = pd.DataFrame({
        "player_id":      ["p4"],
        "season_year":    [2020],
        "position_group": ["WR"],
        "pff_position":   ["WR"],
        "pff_yprr":       [99.0],
        "pff_routes":     [0],
    })
    panel = pd.concat([panel, extra], ignore_index=True)
    out = _write_panel_and_load(tmp_path, panel)

    # p4 had only n=0; should yield NaN triplet for pff_yprr.
    p4 = out[out["player_id"] == "p4"].iloc[0]
    assert pd.isna(p4["pff_yprr_mean"])
    assert pd.isna(p4["pff_yprr_peak"])
    assert pd.isna(p4["pff_yprr_last"])

    # p1's big season is n=400, observed 2.0. Verify the shrinkage formula
    # matches with the active config k.
    k_rate = config.PFF_SHRINKAGE_K_FACTOR_RATE * config.PFF_REF_DENOM_BY_STAT["pff_routes"]
    prior = (100*1 + 400*2 + 50*0.5 + 150*1.5 + 200*3 + 100*1) / (100+400+50+150+200+100)
    # p1 season 2020: obs=2, n=400
    shrunk_2020 = (400 * 2 + k_rate * prior) / (400 + k_rate)
    # p1 season 2019: obs=1, n=100
    shrunk_2019 = (100 * 1 + k_rate * prior) / (100 + k_rate)
    p1 = out[out["player_id"] == "p1"].iloc[0]
    # Last = 2020's shrunk val
    assert p1["pff_yprr_last"] == pytest.approx(shrunk_2020, rel=1e-6)
    # Peak = max of shrunk values
    assert p1["pff_yprr_peak"] == pytest.approx(max(shrunk_2019, shrunk_2020), rel=1e-6)
    # Mean = snap-weighted across seasons
    expected_mean = (shrunk_2019 * 100 + shrunk_2020 * 400) / (100 + 400)
    assert p1["pff_yprr_mean"] == pytest.approx(expected_mean, rel=1e-6)


def test_peak_is_post_shrinkage(tmp_path):
    """A tiny-sample outlier season must NOT dominate _peak after shrinkage.

    Player with one 5-route 95-grade season and one 600-route 70-grade season:
    the raw max is 95, but the shrunk max should be strictly less. Whether
    the high-n season or the small-n season wins the post-shrinkage peak
    depends on k, but the key invariant is "shrunk_peak < raw_peak".
    """
    panel = pd.DataFrame({
        "player_id":      ["p1", "p1", "p2", "p2"],
        "season_year":    [2019, 2020, 2019, 2020],
        "position_group": ["WR"] * 4,
        "pff_position":   ["WR"] * 4,
        "pff_grades_pass_route": [95.0, 70.0, 60.0, 60.0],
        "pff_routes":            [5,    600,  300,  300],
    })
    out = _write_panel_and_load(tmp_path, panel)
    p1 = out[out["player_id"] == "p1"].iloc[0]

    # Raw max was 95; shrunk peak must be strictly less.
    assert p1["pff_grades_pass_route_peak"] < 95.0

    # Verify the exact value matches max of the two shrunk seasons.
    k_grade = config.PFF_SHRINKAGE_K_FACTOR_GRADES * config.PFF_REF_DENOM_BY_STAT["pff_routes"]
    prior = (5*95 + 600*70 + 300*60 + 300*60) / (5 + 600 + 300 + 300)
    shrunk_2019 = (5   * 95.0 + k_grade * prior) / (5   + k_grade)
    shrunk_2020 = (600 * 70.0 + k_grade * prior) / (600 + k_grade)
    expected_peak = max(shrunk_2019, shrunk_2020)
    assert p1["pff_grades_pass_route_peak"] == pytest.approx(expected_peak, rel=1e-6)


def test_extreme_small_sample_doesnt_dominate_peak(tmp_path):
    """At a tiny enough denominator (1 route), even the relatively small k
    pulls the season hard enough to the prior that the high-n season wins."""
    panel = pd.DataFrame({
        "player_id":      ["p1", "p1", "p2", "p2"],
        "season_year":    [2019, 2020, 2019, 2020],
        "position_group": ["WR"] * 4,
        "pff_position":   ["WR"] * 4,
        "pff_grades_pass_route": [95.0, 70.0, 60.0, 60.0],
        # 1 route is so small that even k_grade ≈ 15 dominates: shrunk
        # weight on obs = 1/16 ≈ 6%. The 600-route 70-grade season wins.
        "pff_routes":            [1,    600,  300,  300],
    })
    out = _write_panel_and_load(tmp_path, panel)
    p1 = out[out["player_id"] == "p1"].iloc[0]
    k_grade = config.PFF_SHRINKAGE_K_FACTOR_GRADES * config.PFF_REF_DENOM_BY_STAT["pff_routes"]
    prior = (1*95 + 600*70 + 300*60 + 300*60) / (1 + 600 + 300 + 300)
    shrunk_2019 = (1   * 95.0 + k_grade * prior) / (1   + k_grade)
    shrunk_2020 = (600 * 70.0 + k_grade * prior) / (600 + k_grade)
    # 600-route season should win (its obs weight ≈ 97.5% vs 1-route's ~6%)
    assert shrunk_2020 > shrunk_2019
    assert p1["pff_grades_pass_route_peak"] == pytest.approx(shrunk_2020, rel=1e-6)


def test_mean_is_snap_weighted(tmp_path):
    """A 10-route frosh year and a 600-route senior year should weight ~60:1
    toward the senior, not 1:1 like the legacy unweighted mean."""
    panel = pd.DataFrame({
        "player_id":      ["p1", "p1", "p2", "p2"],
        "season_year":    [2019, 2020, 2019, 2020],
        "position_group": ["WR"] * 4,
        "pff_position":   ["WR"] * 4,
        "pff_yprr":       [3.0, 1.5, 1.4, 1.6],
        "pff_routes":     [10,  600, 300, 300],
    })
    out = _write_panel_and_load(tmp_path, panel)
    p1 = out[out["player_id"] == "p1"].iloc[0]
    # Snap-weighted shrunk mean should be much closer to the 1.5 senior season
    # than to the unweighted mean (3.0+1.5)/2 = 2.25.
    assert p1["pff_yprr_mean"] < 2.0
    assert p1["pff_yprr_mean"] < 1.8  # tighter — even after some shrinkage pull


def test_volume_stats_unchanged(tmp_path):
    """Pure count stats keep the legacy unweighted mean/peak/last."""
    panel = pd.DataFrame({
        "player_id":         ["p1", "p1", "p1"],
        "season_year":       [2018, 2019, 2020],
        "position_group":    ["DL"] * 3,
        "pff_position":      ["DI"] * 3,
        "pff_total_pressures": [3.0, 8.0, 5.0],
        # Note: no denominator column needed for volume stats — they bypass
        # the shrinkage path.
    })
    out = _write_panel_and_load(tmp_path, panel)
    p1 = out[out["player_id"] == "p1"].iloc[0]
    # Legacy: unweighted mean/max/last
    assert p1["pff_total_pressures_mean"] == pytest.approx((3 + 8 + 5) / 3.0)
    assert p1["pff_total_pressures_peak"] == 8.0
    assert p1["pff_total_pressures_last"] == 5.0


def test_pff_seasons_and_position_mode_preserved(tmp_path):
    panel = _build_panel()
    out = _write_panel_and_load(tmp_path, panel)
    assert "pff_seasons" in out.columns
    assert "pff_position_mode" in out.columns
    p1 = out[out["player_id"] == "p1"].iloc[0]
    assert p1["pff_seasons"] == 2
    assert p1["pff_position_mode"] == "WR"


def test_no_pff_data_returns_empty(tmp_path):
    """Missing parquet: function should return an empty frame, not crash."""
    snap = pd.DataFrame({"player_id": ["p1"]})
    out = load_pff_career_features(snap, tmp_path)
    assert out.empty


# ── Career-total reliability features ──────────────────────────────────────────

def test_career_totals_summed_across_seasons(tmp_path):
    """Each PFF_DENOM_STATS column produces a _total feature equal to the
    sum of that denominator across the player's seasons."""
    panel = pd.DataFrame({
        "player_id":      ["p1", "p1", "p1"],
        "season_year":    [2018, 2019, 2020],
        "position_group": ["WR"] * 3,
        "pff_position":   ["WR"] * 3,
        "pff_yprr":       [1.5, 1.8, 1.6],
        "pff_routes":     [100, 200, 300],
        "pff_targets":    [25, 40, 60],
    })
    out = _write_panel_and_load(tmp_path, panel)
    p1 = out[out["player_id"] == "p1"].iloc[0]
    assert p1["pff_routes_total"] == pytest.approx(600.0)
    assert p1["pff_targets_total"] == pytest.approx(125.0)


def test_career_total_nan_when_denom_never_observed(tmp_path):
    """If the player has no rows with that denom observed (e.g. a WR has
    no pass-rush snaps), the _total is NaN — LightGBM routes via missing
    direction rather than treating it as 0."""
    panel = pd.DataFrame({
        "player_id":              ["p1", "p1"],
        "season_year":            [2019, 2020],
        "position_group":         ["WR"] * 2,
        "pff_position":           ["WR"] * 2,
        "pff_yprr":               [1.5, 1.8],
        "pff_routes":             [200, 300],
        # No pass_rush snaps for a WR
        "pff_snap_counts_pass_rush": [np.nan, np.nan],
    })
    out = _write_panel_and_load(tmp_path, panel)
    p1 = out[out["player_id"] == "p1"].iloc[0]
    assert p1["pff_routes_total"] == pytest.approx(500.0)
    assert pd.isna(p1["pff_snap_counts_pass_rush_total"])


def test_career_totals_present_per_position_in_feature_cols():
    """`_build_feature_cols` should gate _total cols by _PFF_DENOM_RELEVANCE
    — QB sees pff_dropbacks_total but not pff_snap_counts_coverage_total."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "step8", str(Path(__file__).resolve().parent.parent / "pipeline" / "08_draft_models.py")
    )
    step8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step8)

    # Build a synthetic snapshot with the _total cols we care about. The
    # feature builder gates on column presence, so include the ones we
    # want testable and confirm only the relevant ones survive.
    snapshot = pd.DataFrame({
        "player_id": ["p1"],
        # PFF totals — varied positions' denominators
        "pff_dropbacks_total":            [400.0],   # QB
        "pff_passing_snaps_total":        [450.0],   # QB
        "pff_attempts_total":             [300.0],   # QB / RB / P
        "pff_snap_counts_offense_total":  [700.0],   # offense pos groups
        "pff_snap_counts_pass_rush_total":[350.0],   # DL / LB
        "pff_snap_counts_coverage_total": [400.0],   # LB / DB
        "pff_routes_total":               [300.0],   # RB / WR / TE
    })

    qb_cols = step8._build_feature_cols(snapshot, "QB")
    db_cols = step8._build_feature_cols(snapshot, "DB")

    assert "pff_dropbacks_total" in qb_cols
    assert "pff_passing_snaps_total" in qb_cols
    assert "pff_attempts_total" in qb_cols
    assert "pff_snap_counts_offense_total" in qb_cols
    assert "pff_snap_counts_coverage_total" not in qb_cols   # QB doesn't cover
    assert "pff_snap_counts_pass_rush_total" not in qb_cols  # QB doesn't pass-rush
    assert "pff_routes_total" not in qb_cols                  # QB doesn't run routes

    assert "pff_snap_counts_coverage_total" in db_cols
    assert "pff_dropbacks_total" not in db_cols
    assert "pff_routes_total" not in db_cols
    assert "pff_snap_counts_offense_total" not in db_cols    # DB is defense


def test_percent_columns_clamped_to_valid_range():
    """Raw PFF `*_percent` / `*_rate` columns are percentage-scaled, so values
    above 100 are impossible. The real feed carries them (pass_block_percent
    up to 200, forced_incompletion_rate up to 400) on tiny denominators."""
    from src.scrapers.pff import _clamp_percent_columns

    df = pd.DataFrame({
        "pff_pass_block_percent":       [100.2, 200.0, 50.0, None],
        "pff_forced_incompletion_rate": [400.0, -5.0, 0.0, 99.9],
        "pff_grades_pass_block":        [77.1, 88.0, 60.0, 70.0],
        "pff_snap_counts_offense":      [951.0, 14.0, 300.0, 400.0],
    })
    out = _clamp_percent_columns(df.copy())

    assert out["pff_pass_block_percent"].max() == 100.0
    assert out["pff_forced_incompletion_rate"].max() == 100.0
    assert out["pff_forced_incompletion_rate"].min() == 0.0
    # In-range values and NaN are untouched.
    assert out["pff_pass_block_percent"].iloc[2] == 50.0
    assert pd.isna(out["pff_pass_block_percent"].iloc[3])
    assert out["pff_forced_incompletion_rate"].iloc[3] == 99.9
    # Non-percentage columns (grades run 0-100 but ratings/counts don't) are
    # left alone — only *_percent / *_rate are clamped.
    assert out["pff_snap_counts_offense"].max() == 951.0
