"""
Tests for own-team rolling context computation in
`src/scrapers/cfbd.py::compute_team_rolling_context`.

Coverage:
  - No future leakage: the rolling value at game G strictly excludes G itself
  - Single-prior game: rolling pace/pass_rate equal the prior game's values
  - OL run-block columns: populated from team_box_scores join, NaN-safe when
    team_box_scores is empty or missing rows
  - min_plays filter: forfeits / heavy-weather games are excluded from panel
  - Cross-school isolation: a school's team_rolling_* does not see another
    school's prior games
  - Targets contract: every (game_id, school) in gamelogs gets an output row
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.cfbd import compute_team_rolling_context


def _gamerow(game_id, school, game_date, team_pass_att, team_rush_att,
             player_id="PLAYER", position_group="WR"):
    return {
        "game_id":          game_id,
        "school":           school,
        "game_date":        pd.Timestamp(game_date),
        "team_pass_att":    float(team_pass_att),
        "team_rush_att":    float(team_rush_att),
        "player_id":        player_id,
        "position_group":   position_group,
    }


def _boxrow(game_id, team, line_yards_avg=None, second_level_yards_avg=None,
            open_field_yards_avg=None, rushing_power_success=None):
    return {
        "game_id":                game_id,
        "team":                   team,
        "line_yards_avg":         line_yards_avg,
        "second_level_yards_avg": second_level_yards_avg,
        "open_field_yards_avg":   open_field_yards_avg,
        "rushing_power_success":  rushing_power_success,
    }


def test_no_future_leakage_game_one_is_nan():
    """First game in a school's history has no prior, so all
    team_rolling_* must be NaN."""
    gamelogs = pd.DataFrame([
        _gamerow("G1", "SchoolA", "2020-09-05", team_pass_att=30, team_rush_att=35),
    ])
    boxes = pd.DataFrame([_boxrow("G1", "SchoolA", line_yards_avg=3.0)])

    out = compute_team_rolling_context(gamelogs, boxes)
    assert len(out) == 1
    row = out.iloc[0]
    for col in ("team_rolling_pace", "team_rolling_pass_rate",
                "team_rolling_line_yards_avg",
                "team_rolling_second_level_yards_avg",
                "team_rolling_open_field_yards_avg",
                "team_rolling_rushing_power_success"):
        assert pd.isna(row[col]), f"{col} should be NaN (no prior history)"


def test_game_two_equals_game_one_exactly():
    """With one prior game, game 2's team_rolling_* must equal game 1's
    panel values (single-observation EWMA reduces to the single value)."""
    g1 = _gamerow("G1", "SchoolA", "2020-09-05", team_pass_att=30, team_rush_att=35)
    g2 = _gamerow("G2", "SchoolA", "2020-09-12", team_pass_att=25, team_rush_att=40)
    boxes = pd.DataFrame([
        _boxrow("G1", "SchoolA", line_yards_avg=3.2,
                second_level_yards_avg=1.5, open_field_yards_avg=1.1,
                rushing_power_success=0.72),
        _boxrow("G2", "SchoolA", line_yards_avg=2.5,
                second_level_yards_avg=0.9, open_field_yards_avg=0.5,
                rushing_power_success=0.60),
    ])

    out = compute_team_rolling_context(pd.DataFrame([g1, g2]), boxes)
    g2_row = out[out["game_id"] == "G2"].iloc[0]

    # Pace and pass_rate from game 1's panel row
    expected_pace = 30 + 35
    expected_pass_rate = 30 / (30 + 35)
    assert math.isclose(g2_row["team_rolling_pace"], expected_pace, rel_tol=1e-9)
    assert math.isclose(g2_row["team_rolling_pass_rate"], expected_pass_rate, rel_tol=1e-9)
    # OL stats from game 1's box-score row
    assert math.isclose(g2_row["team_rolling_line_yards_avg"], 3.2, rel_tol=1e-9)
    assert math.isclose(g2_row["team_rolling_second_level_yards_avg"], 1.5, rel_tol=1e-9)
    assert math.isclose(g2_row["team_rolling_open_field_yards_avg"], 1.1, rel_tol=1e-9)
    assert math.isclose(g2_row["team_rolling_rushing_power_success"], 0.72, rel_tol=1e-9)


def test_ol_columns_nan_when_team_box_scores_empty():
    """When team_box_scores is empty (or missing the team), pace + pass_rate
    still populate but OL columns are NaN — the gamelogs-derived stats are
    independent of the box-score join."""
    g1 = _gamerow("G1", "SchoolA", "2020-09-05", team_pass_att=30, team_rush_att=35)
    g2 = _gamerow("G2", "SchoolA", "2020-09-12", team_pass_att=25, team_rush_att=40)

    out = compute_team_rolling_context(pd.DataFrame([g1, g2]), pd.DataFrame())
    g2_row = out[out["game_id"] == "G2"].iloc[0]

    # pace + pass_rate still populated
    assert math.isclose(g2_row["team_rolling_pace"], 65, rel_tol=1e-9)
    assert math.isclose(g2_row["team_rolling_pass_rate"], 30 / 65, rel_tol=1e-9)
    # OL columns NaN
    for col in ("team_rolling_line_yards_avg",
                "team_rolling_second_level_yards_avg",
                "team_rolling_open_field_yards_avg",
                "team_rolling_rushing_power_success"):
        assert pd.isna(g2_row[col]), f"{col} should be NaN"


def test_min_plays_filter_excludes_low_volume_game():
    """A game with pace < min_plays (e.g. 20 plays, forfeit) is dropped
    from the panel. The next game's team_rolling_* must reflect only
    games above the threshold."""
    g1_forfeit = _gamerow("G1", "SchoolA", "2020-09-05",
                          team_pass_att=10, team_rush_att=5)   # 15 plays
    g2_real    = _gamerow("G2", "SchoolA", "2020-09-12",
                          team_pass_att=30, team_rush_att=40)  # 70 plays
    g3         = _gamerow("G3", "SchoolA", "2020-09-19",
                          team_pass_att=25, team_rush_att=40)
    boxes = pd.DataFrame([
        _boxrow("G1", "SchoolA", line_yards_avg=1.0),
        _boxrow("G2", "SchoolA", line_yards_avg=3.0),
    ])

    out = compute_team_rolling_context(
        pd.DataFrame([g1_forfeit, g2_real, g3]), boxes, min_plays=30,
    )
    g3_row = out[out["game_id"] == "G3"].iloc[0]

    # G3's rolling should reflect only G2 (G1 dropped by min_plays=30)
    # so pace == 70, pass_rate == 30/70, line_yards_avg == 3.0
    assert math.isclose(g3_row["team_rolling_pace"], 70, rel_tol=1e-9)
    assert math.isclose(g3_row["team_rolling_pass_rate"], 30 / 70, rel_tol=1e-9)
    assert math.isclose(g3_row["team_rolling_line_yards_avg"], 3.0, rel_tol=1e-9)


def test_cross_school_isolation():
    """SchoolA's team_rolling_* must not see SchoolB's prior games. A
    transfer player's USC games must not be tainted by his OSU history."""
    osu = _gamerow("G_OSU1", "OhioState", "2020-09-05",
                   team_pass_att=20, team_rush_att=55)  # run-heavy, 75 plays
    usc1 = _gamerow("G_USC1", "USC", "2020-09-05",
                    team_pass_att=45, team_rush_att=20)  # pass-heavy, 65 plays
    usc2 = _gamerow("G_USC2", "USC", "2020-09-12",
                    team_pass_att=40, team_rush_att=25)
    boxes = pd.DataFrame([
        _boxrow("G_OSU1", "OhioState", line_yards_avg=5.5),  # great OL
        _boxrow("G_USC1", "USC", line_yards_avg=1.5),         # bad OL
    ])

    out = compute_team_rolling_context(
        pd.DataFrame([osu, usc1, usc2]), boxes,
    )
    usc_g2 = out[(out["school"] == "USC") & (out["game_id"] == "G_USC2")].iloc[0]

    # USC G2 should reflect only USC G1: pace=65, pass_rate=45/65, OL=1.5
    # If OSU bled in, pace would be (65+75)/2=70 and OL would be ~3.5
    assert math.isclose(usc_g2["team_rolling_pace"], 65, rel_tol=1e-9)
    assert math.isclose(usc_g2["team_rolling_pass_rate"], 45 / 65, rel_tol=1e-9)
    assert math.isclose(usc_g2["team_rolling_line_yards_avg"], 1.5, rel_tol=1e-9)


def test_targets_contract_one_row_per_game_school():
    """Every (game_id, school) pair in gamelogs gets exactly one output row,
    regardless of player-row multiplicity. Step 1 relies on this for the
    merge to be 1:N (one team-row joined to all of that team's player-rows
    in the game)."""
    # Same team-game with two players → one output row
    g1_p1 = _gamerow("G1", "SchoolA", "2020-09-05",
                     team_pass_att=30, team_rush_att=35, player_id="P1")
    g1_p2 = _gamerow("G1", "SchoolA", "2020-09-05",
                     team_pass_att=30, team_rush_att=35, player_id="P2")
    # Different team in the same game
    g1_opp = _gamerow("G1", "SchoolB", "2020-09-05",
                      team_pass_att=20, team_rush_att=40, player_id="P3")

    out = compute_team_rolling_context(
        pd.DataFrame([g1_p1, g1_p2, g1_opp]), pd.DataFrame(),
    )
    # Two (game_id, school) pairs → 2 rows; both NaN (first game)
    assert len(out) == 2
    assert set(zip(out["game_id"], out["school"])) == {
        ("G1", "SchoolA"), ("G1", "SchoolB"),
    }


def test_empty_gamelogs_returns_empty():
    """Defensive: no gamelogs → no output (not a crash)."""
    out = compute_team_rolling_context(pd.DataFrame(), pd.DataFrame())
    assert out.empty


def test_missing_required_cols_returns_empty():
    """Missing required columns → empty output with a warning log."""
    bad = pd.DataFrame([{"game_id": "G1", "school": "A"}])  # no game_date/atts
    out = compute_team_rolling_context(bad, pd.DataFrame())
    assert out.empty
