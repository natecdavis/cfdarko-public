"""
Tests for opponent-defense-allowed rolling computation in
`src/scrapers/cfbd.py::compute_rolling_opp_defense_allowed` and
`compute_team_rolling_scoring_env`.

Coverage:
  - opp_rolling_def_*: no future leakage; correct inverse-perspective lookup
    (the defense being rolled is target's opponent, panel rows are offensive
    stats produced AGAINST that defense)
  - opp_rolling_def_*: cross-defense isolation
  - team_rolling scoring_env: no future leakage; min_plays filter
  - both: empty inputs return empty without crashing
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.cfbd import (
    compute_rolling_opp_defense_allowed,
    compute_team_rolling_scoring_env,
)


# ── opp defense allowed ─────────────────────────────────────────────────────


def _box(game_id, team, **stats):
    base = {"game_id": game_id, "team": team}
    base.update(stats)
    return base


def _sched(game_id, home, away, date):
    return {
        "game_id":    game_id,
        "home_team":  home,
        "away_team":  away,
        "game_date":  pd.Timestamp(date),
    }


OPP_DEF_STATS = ["rushing_stuff_rate", "rushing_power_success"]


def test_opp_def_allowed_no_future_leakage():
    """For a defense's first appearance in the panel, its allowed-stats are
    NaN (no strictly-prior games)."""
    schedule = pd.DataFrame([
        _sched("g1", "A", "B", "2024-09-01"),
    ])
    team_box = pd.DataFrame([
        _box("g1", "A", rushing_stuff_rate=0.3, rushing_power_success=0.7),
        _box("g1", "B", rushing_stuff_rate=0.1, rushing_power_success=0.9),
    ])
    out = compute_rolling_opp_defense_allowed(team_box, schedule,
                                               stat_cols=OPP_DEF_STATS)
    # Both teams' first game → all rolling cols NaN
    assert out[["opp_rolling_def_rushing_stuff_rate",
                "opp_rolling_def_rushing_power_success"]].isna().all().all()


def test_opp_def_allowed_inverse_perspective_lookup():
    """When team A faces defense B in game 2, A's opp_rolling_def_* should
    reflect what B has been allowing — i.e. the offensive stats produced
    against B in B's prior games."""
    schedule = pd.DataFrame([
        _sched("g1", "B", "C", "2024-09-01"),   # B's first game (D faces C; C produces vs B's D)
        _sched("g2", "A", "B", "2024-09-08"),   # A faces B's defense
    ])
    team_box = pd.DataFrame([
        # g1: B is home (their offense); C is away (their offense vs B's D)
        _box("g1", "B", rushing_stuff_rate=0.5, rushing_power_success=0.5),
        _box("g1", "C", rushing_stuff_rate=0.42, rushing_power_success=0.33),
        # g2: A and B both have offensive lines
        _box("g2", "A", rushing_stuff_rate=0.2, rushing_power_success=0.8),
        _box("g2", "B", rushing_stuff_rate=0.6, rushing_power_success=0.4),
    ])
    out = compute_rolling_opp_defense_allowed(team_box, schedule,
                                               stat_cols=OPP_DEF_STATS)
    # For target (g2, school=A, opp=B): rolling over panel rows where
    # opponent==B → only g1's "C" row (which faced B's defense).
    row_A_g2 = out[(out["game_id"] == "g2") & (out["school"] == "A")].iloc[0]
    assert row_A_g2["opp_rolling_def_rushing_stuff_rate"] == 0.42
    assert row_A_g2["opp_rolling_def_rushing_power_success"] == 0.33


def test_opp_def_allowed_cross_defense_isolation():
    """B's defense allowed stats must not leak into the rolling history of
    defense C."""
    schedule = pd.DataFrame([
        _sched("g1", "A", "B", "2024-09-01"),
        _sched("g2", "D", "C", "2024-09-08"),
        _sched("g3", "E", "B", "2024-09-15"),   # E faces B's defense
        _sched("g4", "F", "C", "2024-09-15"),   # F faces C's defense
    ])
    team_box = pd.DataFrame([
        _box("g1", "A", rushing_stuff_rate=0.10, rushing_power_success=0.90),
        _box("g1", "B", rushing_stuff_rate=0.20, rushing_power_success=0.80),
        _box("g2", "D", rushing_stuff_rate=0.30, rushing_power_success=0.70),
        _box("g2", "C", rushing_stuff_rate=0.40, rushing_power_success=0.60),
        _box("g3", "E", rushing_stuff_rate=0.99, rushing_power_success=0.99),
        _box("g3", "B", rushing_stuff_rate=0.50, rushing_power_success=0.50),
        _box("g4", "F", rushing_stuff_rate=0.99, rushing_power_success=0.99),
        _box("g4", "C", rushing_stuff_rate=0.50, rushing_power_success=0.50),
    ])
    out = compute_rolling_opp_defense_allowed(team_box, schedule,
                                               stat_cols=OPP_DEF_STATS)
    # E faces B → rolling pulls only A's stats from g1 (the team that faced B)
    row_E = out[(out["game_id"] == "g3") & (out["school"] == "E")].iloc[0]
    assert row_E["opp_rolling_def_rushing_stuff_rate"] == 0.10
    # F faces C → rolling pulls only D's stats from g2
    row_F = out[(out["game_id"] == "g4") & (out["school"] == "F")].iloc[0]
    assert row_F["opp_rolling_def_rushing_stuff_rate"] == 0.30


def test_opp_def_allowed_empty_inputs_safe():
    out = compute_rolling_opp_defense_allowed(pd.DataFrame(), pd.DataFrame(),
                                               stat_cols=OPP_DEF_STATS)
    assert out.empty


# ── team rolling scoring env ────────────────────────────────────────────────


def _gl(game_id, school, date, team_pass_att, team_rush_att):
    return {
        "game_id":       game_id,
        "school":        school,
        "game_date":     pd.Timestamp(date),
        "team_pass_att": float(team_pass_att),
        "team_rush_att": float(team_rush_att),
    }


SCORING_STATS = ["scoring_opportunities", "points_per_opportunity"]


def test_team_scoring_env_no_future_leakage():
    gl = pd.DataFrame([
        _gl("g1", "A", "2024-09-01", 30, 30),
        _gl("g2", "A", "2024-09-08", 30, 30),
    ])
    tb = pd.DataFrame([
        _box("g1", "A", scoring_opportunities=5, points_per_opportunity=4.0),
        _box("g2", "A", scoring_opportunities=3, points_per_opportunity=6.0),
    ])
    out = compute_team_rolling_scoring_env(gl, tb, stat_cols=SCORING_STATS)
    g1 = out[out["game_id"] == "g1"].iloc[0]
    g2 = out[out["game_id"] == "g2"].iloc[0]
    # g1 has no prior → NaN
    assert pd.isna(g1["team_rolling_scoring_opportunities"])
    # g2's prior is g1 only → must equal g1's box value (single-prior EWMA)
    assert g2["team_rolling_scoring_opportunities"] == 5.0
    assert g2["team_rolling_points_per_opportunity"] == 4.0


def test_team_scoring_env_min_plays_filter():
    """Games with pace < min_plays are excluded from the panel — so g2's
    rolling should be NaN if g1 was below the threshold."""
    gl = pd.DataFrame([
        _gl("g1", "A", "2024-09-01", 5, 5),     # 10 plays — below min_plays=30
        _gl("g2", "A", "2024-09-08", 30, 30),
    ])
    tb = pd.DataFrame([
        _box("g1", "A", scoring_opportunities=5, points_per_opportunity=4.0),
        _box("g2", "A", scoring_opportunities=3, points_per_opportunity=6.0),
    ])
    out = compute_team_rolling_scoring_env(gl, tb, stat_cols=SCORING_STATS,
                                            min_plays=30)
    g2 = out[out["game_id"] == "g2"].iloc[0]
    assert pd.isna(g2["team_rolling_scoring_opportunities"])


def test_team_scoring_env_cross_school_isolation():
    gl = pd.DataFrame([
        _gl("g1", "A", "2024-09-01", 30, 30),
        _gl("g2", "B", "2024-09-08", 30, 30),
    ])
    tb = pd.DataFrame([
        _box("g1", "A", scoring_opportunities=10, points_per_opportunity=5.0),
        _box("g2", "B", scoring_opportunities=1,  points_per_opportunity=1.0),
    ])
    out = compute_team_rolling_scoring_env(gl, tb, stat_cols=SCORING_STATS)
    g2_B = out[(out["game_id"] == "g2") & (out["school"] == "B")].iloc[0]
    # B's g2 must NOT pick up A's g1 stats
    assert pd.isna(g2_B["team_rolling_scoring_opportunities"])


def test_team_scoring_env_empty_inputs_safe():
    out = compute_team_rolling_scoring_env(pd.DataFrame(), pd.DataFrame(),
                                            stat_cols=SCORING_STATS)
    assert out.empty
