"""
Tests for own-defense teammate rolling computation in
`src/scrapers/cfbd.py::compute_defense_rolling_stats`.

Coverage:
  - No future leakage: first game returns NaN
  - Single-prior correctness: rate computed from prior game's counters /
    opp_pass_att (looked up via the opponent's own row in the same game)
  - All-NaN defensive stats (pre-2018 CFBD coverage gap) emit NaN, not 0
  - Cross-school isolation
  - min_opp_pass_att / min_opp_plays filters
  - Empty / missing-cols safe
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.cfbd import compute_defense_rolling_stats


def _gl(game_id, school, opponent, date, team_pass_att, team_rush_att,
        sacks=None, qb_hurries=None, passes_defended=None, ints=None,
        player_id="P"):
    return {
        "game_id":          game_id,
        "school":           school,
        "opponent":         opponent,
        "game_date":        pd.Timestamp(date),
        "team_pass_att":    float(team_pass_att),
        "team_rush_att":    float(team_rush_att),
        "sacks":            float(sacks)            if sacks            is not None else np.nan,
        "qb_hurries":       float(qb_hurries)       if qb_hurries       is not None else np.nan,
        "passes_defended":  float(passes_defended)  if passes_defended  is not None else np.nan,
        "ints":             float(ints)             if ints             is not None else np.nan,
        "player_id":        player_id,
    }


def test_no_future_leakage_first_game_nan():
    gl = pd.DataFrame([
        # Single game in panel — both teams' first game → all NaN
        _gl("g1", "A", "B", "2024-09-01", 30, 30,
            sacks=2, qb_hurries=3, passes_defended=4, ints=1),
        _gl("g1", "B", "A", "2024-09-01", 30, 30,
            sacks=1, qb_hurries=1, passes_defended=2, ints=0),
    ])
    out = compute_defense_rolling_stats(gl)
    assert out["def_rolling_pressure_rate"].isna().all()
    assert out["def_rolling_coverage_rate"].isna().all()


def test_single_prior_correct_rates():
    """A's g2: rolling reflects A's g1 stats; rate = (sacks + hurries) /
    opp_pass_att where opp_pass_att is B's team_pass_att in g1 (B was A's
    opponent in g1)."""
    gl = pd.DataFrame([
        # g1: A vs B. A's defense: 2 sacks + 3 hurries; B threw 25 passes (opp_pass_att=25)
        _gl("g1", "A", "B", "2024-09-01", 30, 20,
            sacks=2, qb_hurries=3, passes_defended=4, ints=1),
        _gl("g1", "B", "A", "2024-09-01", 25, 35,
            sacks=1, qb_hurries=1, passes_defended=2, ints=0),
        # g2: A vs C. Need both rows in the panel so A's g2 target gets emitted.
        _gl("g2", "A", "C", "2024-09-08", 30, 30,
            sacks=0, qb_hurries=0, passes_defended=0, ints=0),
        _gl("g2", "C", "A", "2024-09-08", 30, 30,
            sacks=0, qb_hurries=0, passes_defended=0, ints=0),
    ])
    out = compute_defense_rolling_stats(gl)
    a_g2 = out[(out["game_id"] == "g2") & (out["school"] == "A")].iloc[0]
    # pressure_rate = (2 + 3) / 25 = 0.2 (B's pass_att in g1 was 25)
    assert abs(a_g2["def_rolling_pressure_rate"] - 0.2) < 1e-9
    # coverage_rate = (4 + 1) / 25 = 0.2
    assert abs(a_g2["def_rolling_coverage_rate"] - 0.2) < 1e-9


def test_all_nan_defensive_stats_emit_nan_not_zero():
    """Pre-2018 games have NaN sacks/hurries/passes_defended/ints. Sum with
    min_count=1 must emit NaN per group, not 0. Otherwise the rate would be
    0 / opp_pass_att = 0 — a false 'no pressure' signal."""
    gl = pd.DataFrame([
        # g1: A vs B, all defensive stats NaN (pre-2018 style)
        _gl("g1", "A", "B", "2015-09-01", 30, 20),
        _gl("g1", "B", "A", "2015-09-01", 25, 35),
        # g2: A vs C, also all NaN
        _gl("g2", "A", "C", "2015-09-08", 30, 30),
        _gl("g2", "C", "A", "2015-09-08", 30, 30),
    ])
    out = compute_defense_rolling_stats(gl)
    # A's g2 rolling should be NaN — sum of all-NaN sacks/hurries via
    # min_count=1 yields NaN, which propagates through rate calc.
    a_g2 = out[(out["game_id"] == "g2") & (out["school"] == "A")].iloc[0]
    assert pd.isna(a_g2["def_rolling_pressure_rate"])
    assert pd.isna(a_g2["def_rolling_coverage_rate"])


def test_cross_school_isolation():
    gl = pd.DataFrame([
        _gl("g1", "A", "B", "2024-09-01", 30, 30,
            sacks=10, qb_hurries=10, passes_defended=10, ints=10),
        _gl("g1", "B", "A", "2024-09-01", 30, 30,
            sacks=0, qb_hurries=0, passes_defended=0, ints=0),
        _gl("g2", "C", "D", "2024-09-08", 30, 30,
            sacks=0, qb_hurries=0, passes_defended=0, ints=0),
        _gl("g2", "D", "C", "2024-09-08", 30, 30,
            sacks=0, qb_hurries=0, passes_defended=0, ints=0),
    ])
    out = compute_defense_rolling_stats(gl)
    # C's g2 rolling must NOT pick up A's massive g1 numbers
    c_g2 = out[(out["game_id"] == "g2") & (out["school"] == "C")].iloc[0]
    assert pd.isna(c_g2["def_rolling_pressure_rate"])


def test_min_opp_pass_att_filter():
    """Panel rows with opp_pass_att below threshold are dropped (e.g. heavy-
    weather / triple-option games with tiny pass volume). A's g2 with only
    a low-pass-att opponent in history → rolling NaN."""
    gl = pd.DataFrame([
        # B threw 5 passes (below min_opp_pass_att=10) — A's defensive rates
        # for g1 should be excluded from panel
        _gl("g1", "A", "B", "2024-09-01", 30, 30,
            sacks=2, qb_hurries=2, passes_defended=1, ints=0),
        _gl("g1", "B", "A", "2024-09-01", 5, 55,
            sacks=0, qb_hurries=0, passes_defended=0, ints=0),
        _gl("g2", "A", "C", "2024-09-08", 30, 30,
            sacks=0, qb_hurries=0, passes_defended=0, ints=0),
        _gl("g2", "C", "A", "2024-09-08", 30, 30,
            sacks=0, qb_hurries=0, passes_defended=0, ints=0),
    ])
    out = compute_defense_rolling_stats(gl, min_opp_pass_att=10)
    a_g2 = out[(out["game_id"] == "g2") & (out["school"] == "A")].iloc[0]
    assert pd.isna(a_g2["def_rolling_pressure_rate"])


def test_empty_inputs_safe():
    out = compute_defense_rolling_stats(pd.DataFrame())
    assert out.empty


def test_missing_cols_returns_empty():
    bad = pd.DataFrame([{"game_id": "g1", "school": "A"}])
    out = compute_defense_rolling_stats(bad)
    assert out.empty
