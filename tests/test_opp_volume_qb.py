"""
Tests for opponent-volume and opponent-QB rolling features (bundle #1 + #7b).

Coverage:
  - compute_rolling_opp_offense_volume: no future leakage, cross-team
    isolation, single-prior correctness, empty-input safety, min_plays filter
  - opp_qb_rolling join (in _add_opp_qb_rolling_stats): rename produces the
    correct column names and the merge keys are (game_id, opponent)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.cfbd import (
    compute_rolling_opp_offense_volume,
    compute_qb_rolling_stats,
)


def _gl(game_id, school, opponent, date, pass_att, rush_att,
        position_group="QB", player_id="P", **kw):
    base = {
        "game_id":         game_id,
        "school":          school,
        "opponent":        opponent,
        "game_date":       pd.Timestamp(date),
        "team_pass_att":   float(pass_att),
        "team_rush_att":   float(rush_att),
        "position_group":  position_group,
        "player_id":       player_id,
        "pass_att":        float(pass_att),
    }
    base.update(kw)
    return base


def test_opp_volume_no_future_leakage_first_game_nan():
    """First game's opp has no strictly-prior history → NaN."""
    gl = pd.DataFrame([_gl("g1", "A", "B", "2024-09-01", 30, 30)])
    out = compute_rolling_opp_offense_volume(gl)
    assert out[["opp_rolling_off_pass_att",
                "opp_rolling_off_rush_att",
                "opp_rolling_off_plays"]].isna().all().all()


def test_opp_volume_single_prior_equals_prior_value():
    """A's g2 vs B: B's g1 panel row had pass_att=40, rush_att=20 → A's
    opp_rolling_off_pass_att == 40 (single-prior EWMA collapses to the value)."""
    gl = pd.DataFrame([
        _gl("g1", "B", "C", "2024-09-01", 40, 20),    # B's first game (offense vs C)
        _gl("g2", "A", "B", "2024-09-08", 30, 30),    # A faces B
    ])
    out = compute_rolling_opp_offense_volume(gl)
    a_g2 = out[(out["game_id"] == "g2") & (out["school"] == "A")].iloc[0]
    assert a_g2["opp_rolling_off_pass_att"] == 40.0
    assert a_g2["opp_rolling_off_rush_att"] == 20.0
    assert a_g2["opp_rolling_off_plays"]    == 60.0


def test_opp_volume_cross_team_isolation():
    """Team B's history must not leak into team C's rolling."""
    gl = pd.DataFrame([
        _gl("g1", "B", "X", "2024-09-01", 60, 10),   # heavy-pass B
        _gl("g2", "C", "Y", "2024-09-01", 10, 60),   # heavy-rush C
        _gl("g3", "A", "B", "2024-09-08", 30, 30),   # A faces B
        _gl("g4", "D", "C", "2024-09-08", 30, 30),   # D faces C
    ])
    out = compute_rolling_opp_offense_volume(gl)
    a = out[(out["game_id"] == "g3") & (out["school"] == "A")].iloc[0]
    d = out[(out["game_id"] == "g4") & (out["school"] == "D")].iloc[0]
    # A faces B (heavy-pass)
    assert a["opp_rolling_off_pass_att"] == 60.0
    assert a["opp_rolling_off_rush_att"] == 10.0
    # D faces C (heavy-rush) — must not pick up B's profile
    assert d["opp_rolling_off_pass_att"] == 10.0
    assert d["opp_rolling_off_rush_att"] == 60.0


def test_opp_volume_min_plays_filter():
    """If opponent's only prior game is below min_plays, it's filtered from
    panel → focal team's opp_rolling is NaN."""
    gl = pd.DataFrame([
        _gl("g1", "B", "C", "2024-09-01", 5, 5),     # 10 plays — below 30
        _gl("g2", "A", "B", "2024-09-08", 30, 30),
    ])
    out = compute_rolling_opp_offense_volume(gl, min_plays=30)
    a = out[(out["game_id"] == "g2") & (out["school"] == "A")].iloc[0]
    assert pd.isna(a["opp_rolling_off_pass_att"])


def test_opp_volume_empty_inputs_safe():
    out = compute_rolling_opp_offense_volume(pd.DataFrame())
    assert out.empty


def test_opp_volume_missing_cols_returns_empty():
    bad = pd.DataFrame([{"game_id": "g1", "school": "A"}])
    out = compute_rolling_opp_offense_volume(bad)
    assert out.empty


def test_opp_qb_rolling_rename_produces_expected_names():
    """The qb_rolling output, renamed for opponent-perspective merge, must
    produce columns prefixed opp_qb_rolling_*."""
    gl = pd.DataFrame([
        _gl("g1", "B", "C", "2024-09-01", 30, 30,
            position_group="QB", player_id="qb1",
            pass_yards=300, pass_tds=2, completions=20, interceptions=0,
            completion_pct=0.7, yards_per_attempt=10.0, ppa_avg_passing=0.4),
        _gl("g2", "B", "D", "2024-09-08", 30, 30,
            position_group="QB", player_id="qb1",
            pass_yards=250, pass_tds=1, completions=18, interceptions=1,
            completion_pct=0.6, yards_per_attempt=8.3, ppa_avg_passing=0.2),
    ])
    qb_roll = compute_qb_rolling_stats(gl)
    assert "qb_rolling_passer_rating" in qb_roll.columns
    rename_map = {"school": "opponent"}
    rename_map.update({
        c: c.replace("qb_rolling_", "opp_qb_rolling_")
        for c in qb_roll.columns if c.startswith("qb_rolling_")
    })
    opp_qb = qb_roll.rename(columns=rename_map)
    expected = {
        "opp_qb_rolling_passer_rating",
        "opp_qb_rolling_completion_pct",
        "opp_qb_rolling_ppa_avg_passing",
        "opp_qb_rolling_ypa",
        "opp_qb_rolling_td_rate",
        "opp_qb_rolling_int_rate",
    }
    assert expected.issubset(opp_qb.columns)
    assert "opponent" in opp_qb.columns
    assert "school" not in opp_qb.columns


def test_opp_qb_rolling_merge_finds_opponent_history():
    """End-to-end: A plays B in g2 (A's defenders want B's QB rolling).
    Merge on (game_id, opponent=B) recovers B's QB rate stats from g1."""
    gl = pd.DataFrame([
        # B's QB plays in g1 — establishes B's qb_rolling for g2
        _gl("g1", "B", "X", "2024-09-01", 30, 30,
            position_group="QB", player_id="qbB",
            pass_yards=300, pass_tds=3, completions=20, interceptions=0,
            completion_pct=0.7, yards_per_attempt=10.0, ppa_avg_passing=0.5),
        # A's QB plays in g1 — separate school, must not contaminate
        _gl("g1", "A", "Y", "2024-09-01", 30, 30,
            position_group="QB", player_id="qbA",
            pass_yards=100, pass_tds=0, completions=10, interceptions=3,
            completion_pct=0.3, yards_per_attempt=3.3, ppa_avg_passing=-0.5),
        # A faces B in g2 — A's defenders should see B's prior QB profile
        _gl("g2", "A", "B", "2024-09-08", 30, 30,
            position_group="QB", player_id="qbA",
            pass_yards=200, pass_tds=1, completions=15, interceptions=1,
            completion_pct=0.5, yards_per_attempt=6.7, ppa_avg_passing=0.1),
        # B's own row in g2 (mirrors production: both schools have gamelog rows
        # per game). Required for compute_qb_rolling_stats to emit a (g2, B)
        # target row that the opponent-side merge can find.
        _gl("g2", "B", "A", "2024-09-08", 30, 30,
            position_group="QB", player_id="qbB",
            pass_yards=180, pass_tds=1, completions=12, interceptions=2,
            completion_pct=0.4, yards_per_attempt=6.0, ppa_avg_passing=0.0),
    ])
    qb_roll = compute_qb_rolling_stats(gl)
    rename_map = {"school": "opponent"}
    rename_map.update({
        c: c.replace("qb_rolling_", "opp_qb_rolling_")
        for c in qb_roll.columns if c.startswith("qb_rolling_")
    })
    opp_qb = qb_roll.rename(columns=rename_map)

    targets = gl[["game_id", "school", "opponent"]].drop_duplicates(
        ["game_id", "school"])
    merged = targets.merge(
        opp_qb[["game_id", "opponent",
                "opp_qb_rolling_passer_rating",
                "opp_qb_rolling_completion_pct"]],
        on=["game_id", "opponent"], how="left",
    )
    a_g2 = merged[(merged["game_id"] == "g2") & (merged["school"] == "A")].iloc[0]
    # B's only prior was g1 with cmp%=0.7 — single-prior EWMA = 0.7
    assert abs(a_g2["opp_qb_rolling_completion_pct"] - 0.7) < 1e-9
