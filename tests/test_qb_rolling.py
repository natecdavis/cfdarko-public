"""
Tests for teammate-QB rolling quality computation in
`src/scrapers/cfbd.py::compute_qb_rolling_stats`.

Coverage:
  - No future leakage: the rolling value at game G strictly excludes G itself
  - Two-QB games: prior-game value blends pass_att-weighted across all QBs
  - min_pass_att filter: low-volume QB games are excluded from the prior panel
  - Cross-school isolation: a school's qb_rolling_* does not see another
    school's QB history
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.cfbd import compute_qb_rolling_stats


def _mkrow(player_id, game_id, school, game_date, position_group,
           pass_att, pass_yards, pass_tds, completions, interceptions,
           completion_pct=None, yards_per_attempt=None,
           ppa_avg_passing=None):
    cmp_pct = (completion_pct if completion_pct is not None
               else (completions / pass_att if pass_att else float("nan")))
    ypa = (yards_per_attempt if yards_per_attempt is not None
           else (pass_yards / pass_att if pass_att else float("nan")))
    return {
        "player_id":           player_id,
        "game_id":             game_id,
        "school":              school,
        "game_date":           pd.Timestamp(game_date),
        "position_group":      position_group,
        "pass_att":            float(pass_att),
        "pass_yards":          float(pass_yards),
        "pass_tds":            float(pass_tds),
        "completions":         float(completions),
        "interceptions":       float(interceptions),
        "completion_pct":      float(cmp_pct),
        "yards_per_attempt":   float(ypa),
        "ppa_avg_passing":     (float(ppa_avg_passing)
                                 if ppa_avg_passing is not None else float("nan")),
    }


def _passer_rating(pass_yards, pass_tds, completions, interceptions, pass_att):
    pa = max(pass_att, 1.0)
    return (8.4 * pass_yards
            + 330.0 * pass_tds
            + 100.0 * completions
            - 200.0 * interceptions) / pa


def test_no_future_leakage_game_one_is_nan():
    """The very first game in a school's history has no prior, so all
    qb_rolling_* must be NaN."""
    rows = [
        _mkrow("QB1", "G1", "SchoolA", "2020-09-05", "QB",
               pass_att=30, pass_yards=300, pass_tds=2, completions=20,
               interceptions=0, ppa_avg_passing=0.30),
    ]
    out = compute_qb_rolling_stats(pd.DataFrame(rows))

    # one (game_id, school) target → one row
    assert len(out) == 1
    row = out.iloc[0]
    for col in ("qb_rolling_passer_rating", "qb_rolling_completion_pct",
                "qb_rolling_ppa_avg_passing", "qb_rolling_ypa",
                "qb_rolling_td_rate", "qb_rolling_int_rate"):
        assert pd.isna(row[col]), f"{col} should be NaN (no prior history)"


def test_game_two_equals_game_one_exactly():
    """With a single prior game and no other QB activity, game 2's
    qb_rolling_* must equal game 1's atomic rate stats (one observation in
    the EWMA, weight is positive)."""
    g1 = _mkrow("QB1", "G1", "SchoolA", "2020-09-05", "QB",
                pass_att=30, pass_yards=300, pass_tds=2,
                completions=20, interceptions=1, ppa_avg_passing=0.30)
    g2 = _mkrow("QB1", "G2", "SchoolA", "2020-09-12", "QB",
                pass_att=25, pass_yards=200, pass_tds=1,
                completions=15, interceptions=0, ppa_avg_passing=0.10)
    out = compute_qb_rolling_stats(pd.DataFrame([g1, g2]))
    g2_row = out[out["game_id"] == "G2"].iloc[0]

    expected_rating = _passer_rating(300, 2, 20, 1, 30)
    assert math.isclose(g2_row["qb_rolling_passer_rating"], expected_rating, rel_tol=1e-9)
    assert math.isclose(g2_row["qb_rolling_completion_pct"], 20 / 30, rel_tol=1e-9)
    assert math.isclose(g2_row["qb_rolling_ppa_avg_passing"], 0.30, rel_tol=1e-9)
    assert math.isclose(g2_row["qb_rolling_ypa"], 300 / 30, rel_tol=1e-9)
    assert math.isclose(g2_row["qb_rolling_td_rate"], 2 / 30, rel_tol=1e-9)
    assert math.isclose(g2_row["qb_rolling_int_rate"], 1 / 30, rel_tol=1e-9)


def test_two_qb_game_blends_pass_att_weighted():
    """In a 2-QB game (QB1: 15 att, QB2: 12 att), the panel observation is
    the pass_att-weighted average of their rates."""
    g1a = _mkrow("QB1", "G1", "SchoolA", "2020-09-05", "QB",
                 pass_att=15, pass_yards=150, pass_tds=1,
                 completions=10, interceptions=0)
    g1b = _mkrow("QB2", "G1", "SchoolA", "2020-09-05", "QB",
                 pass_att=12, pass_yards=72,  pass_tds=0,
                 completions=6,  interceptions=2)
    # game 2 (next week) — to read back the rolling value from game 1's blend
    g2 = _mkrow("QB1", "G2", "SchoolA", "2020-09-12", "QB",
                pass_att=20, pass_yards=200, pass_tds=1,
                completions=14, interceptions=0)
    out = compute_qb_rolling_stats(pd.DataFrame([g1a, g1b, g2]))
    g2_row = out[out["game_id"] == "G2"].iloc[0]

    # Expected blend: weights are pass_att (15 and 12) → total 27
    # Per-QB passer ratings:
    r1 = _passer_rating(150, 1, 10, 0, 15)
    r2 = _passer_rating(72,  0, 6,  2, 12)
    expected_rating = (15 * r1 + 12 * r2) / 27
    assert math.isclose(g2_row["qb_rolling_passer_rating"], expected_rating, rel_tol=1e-9)

    # Completion %: blend of QB1's 10/15 and QB2's 6/12, weighted by pass_att.
    cmp1, cmp2 = 10 / 15, 6 / 12
    expected_cmp = (15 * cmp1 + 12 * cmp2) / 27
    assert math.isclose(g2_row["qb_rolling_completion_pct"], expected_cmp, rel_tol=1e-9)

    # td_rate blend
    td1, td2 = 1 / 15, 0 / 12
    expected_td = (15 * td1 + 12 * td2) / 27
    assert math.isclose(g2_row["qb_rolling_td_rate"], expected_td, rel_tol=1e-9)


def test_min_pass_att_filter_excludes_low_volume_qb_row():
    """A QB row with pass_att < min_pass_att (e.g. 3) is dropped from the
    prior panel; the next game's qb_rolling_* should reflect ONLY the
    other QB's history."""
    # Game 1: noisy QB1 throws 3 attempts (below default min_pass_att=5)
    g1_noisy = _mkrow("QB1", "G1", "SchoolA", "2020-09-05", "QB",
                       pass_att=3, pass_yards=10, pass_tds=0,
                       completions=1, interceptions=2)
    # Game 1: QB2 throws 25 attempts (above min) on the same team
    g1_real = _mkrow("QB2", "G1", "SchoolA", "2020-09-05", "QB",
                      pass_att=25, pass_yards=300, pass_tds=2,
                      completions=18, interceptions=0)
    g2 = _mkrow("QB1", "G2", "SchoolA", "2020-09-12", "QB",
                pass_att=20, pass_yards=180, pass_tds=1,
                completions=12, interceptions=0)
    out = compute_qb_rolling_stats(pd.DataFrame([g1_noisy, g1_real, g2]))
    g2_row = out[out["game_id"] == "G2"].iloc[0]

    # Expected: only QB2's game-1 stats inform the rolling value (noisy QB1
    # filtered out by min_pass_att). The panel for game 1 then has only
    # QB2's row → unblended.
    expected_rating = _passer_rating(300, 2, 18, 0, 25)
    assert math.isclose(g2_row["qb_rolling_passer_rating"], expected_rating, rel_tol=1e-9)
    assert math.isclose(g2_row["qb_rolling_completion_pct"], 18 / 25, rel_tol=1e-9)


def test_cross_school_isolation():
    """SchoolA's qb_rolling_* must not see SchoolB's QB history (the join
    is panel_entity_col='school' == target_entity_col='school'). A transfer
    receiver's USC games must not be tainted by his OSU QB's history."""
    osu = _mkrow("QB_OSU", "G_OSU1", "OhioState", "2020-09-05", "QB",
                  pass_att=40, pass_yards=500, pass_tds=5,
                  completions=30, interceptions=0)
    usc1 = _mkrow("QB_USC", "G_USC1", "USC", "2020-09-05", "QB",
                   pass_att=20, pass_yards=180, pass_tds=0,
                   completions=12, interceptions=2)
    usc2 = _mkrow("QB_USC", "G_USC2", "USC", "2020-09-12", "QB",
                   pass_att=22, pass_yards=200, pass_tds=1,
                   completions=14, interceptions=1)
    out = compute_qb_rolling_stats(pd.DataFrame([osu, usc1, usc2]))
    usc_g2 = out[(out["school"] == "USC") & (out["game_id"] == "G_USC2")].iloc[0]

    # USC game 2's rolling should reflect only QB_USC's USC game 1 (low rating)
    expected_rating = _passer_rating(180, 0, 12, 2, 20)
    assert math.isclose(usc_g2["qb_rolling_passer_rating"], expected_rating, rel_tol=1e-9)
    # Sanity: OSU's high rating (which would skew the average upward if
    # cross-school leaked in) is far from USC's value
    osu_rating = _passer_rating(500, 5, 30, 0, 40)
    assert abs(usc_g2["qb_rolling_passer_rating"] - osu_rating) > 50


def test_target_includes_non_qb_rows():
    """Every (game_id, school) pair in gamelogs gets an output row, even
    games where no QB met min_pass_att — those rows have NaN qb_rolling_*.
    Verifies the merge contract that step 1 relies on."""
    # SchoolA: QB only had a 3-attempt game (filtered out)
    qb_low = _mkrow("QB1", "G1", "SchoolA", "2020-09-05", "QB",
                     pass_att=3, pass_yards=15, pass_tds=0,
                     completions=2, interceptions=0)
    # SchoolA: a WR played in the same game (non-QB row, no QB metadata)
    wr = {
        "player_id": "WR1",
        "game_id": "G1",
        "school": "SchoolA",
        "game_date": pd.Timestamp("2020-09-05"),
        "position_group": "WR",
        "pass_att": float("nan"),
    }
    out = compute_qb_rolling_stats(pd.DataFrame([qb_low, wr]))

    # One (game_id, school) → one row, all qb_rolling_* NaN (no qualifying prior)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["school"] == "SchoolA" and row["game_id"] == "G1"
    for col in row.index:
        if col.startswith("qb_rolling_"):
            assert pd.isna(row[col]), f"{col} should be NaN"


def test_empty_gamelogs_returns_empty():
    """Defensive: no gamelogs → no output (not a crash)."""
    out = compute_qb_rolling_stats(pd.DataFrame())
    assert out.empty
