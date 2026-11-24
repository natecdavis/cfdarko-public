"""
Unit tests for pipeline/01f_player_plays.py — pivot, wind decomposition,
score_diff handling, and garbage-time classification.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util
import math

import numpy as np
import pandas as pd
import pytest


# Load the step module by file path (filename starts with a digit)
_spec = importlib.util.spec_from_file_location(
    "_step_01f", str(Path(__file__).parent.parent / "pipeline" / "01f_player_plays.py")
)
step_01f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step_01f)


# ── _pivot_play_stats ───────────────────────────────────────────────────────


def test_pivot_play_stats_unique_per_play_athlete():
    """A receiver who caught a pass for a TD has 3 stat_type rows that
    should collapse to ONE wide row with three populated stat columns."""
    long = pd.DataFrame({
        "play_id":     [1, 1, 1],
        "athlete_id":  ["A", "A", "A"],
        "stat_type":   ["Reception", "Target", "Touchdown"],
        "stat":        [18.0, 1.0, 1.0],
        "team":        ["GA"] * 3,
        "opponent":    ["AL"] * 3,
        "season_year": [2024] * 3,
        "week":        [5] * 3,
        "game_id":     [10] * 3,
        "conference":  ["SEC"] * 3,
    })
    wide = step_01f._pivot_play_stats(long)
    assert len(wide) == 1
    assert wide["reception"].iloc[0] == 18.0
    assert wide["target"].iloc[0] == 1.0
    assert wide["touchdown"].iloc[0] == 1.0


def test_pivot_separates_athletes_on_same_play():
    """QB completing a pass to WR → 2 wide rows: one for QB, one for WR."""
    long = pd.DataFrame({
        "play_id":     [1, 1, 1],
        "athlete_id":  ["QB1", "WR1", "WR1"],
        "stat_type":   ["Completion", "Reception", "Target"],
        "stat":        [20.0, 20.0, 1.0],
        "team":        ["GA"] * 3,
        "opponent":    ["AL"] * 3,
        "season_year": [2024] * 3,
        "week":        [5] * 3,
        "game_id":     [10] * 3,
        "conference":  ["SEC"] * 3,
    })
    wide = step_01f._pivot_play_stats(long)
    assert len(wide) == 2
    qb_row = wide[wide["athlete_id"] == "QB1"].iloc[0]
    wr_row = wide[wide["athlete_id"] == "WR1"].iloc[0]
    assert qb_row["completion"] == 20.0
    assert pd.isna(qb_row.get("reception", float("nan")))
    assert wr_row["reception"] == 20.0
    assert wr_row["target"] == 1.0


# ── _wind_decomposition ─────────────────────────────────────────────────────


def test_wind_decomposition_parallel():
    """Wind blowing along the field axis: parallel ≈ wind_speed, perp ≈ 0."""
    ws = pd.Series([10.0, 10.0, 10.0])
    wd = pd.Series([0.0, 90.0, 45.0])
    orient = pd.Series([0.0, 90.0, 45.0])  # field along these axes
    par, perp = step_01f._wind_decomposition(ws, wd, orient)
    # Wind direction == field orientation: parallel = speed, perp = 0
    assert par.iloc[0] == pytest.approx(10.0)
    assert perp.iloc[0] == pytest.approx(0.0, abs=1e-10)
    assert par.iloc[1] == pytest.approx(10.0)
    assert perp.iloc[1] == pytest.approx(0.0, abs=1e-10)


def test_wind_decomposition_perpendicular():
    """Wind 90° from field axis: parallel=0, perp=speed."""
    ws = pd.Series([10.0])
    wd = pd.Series([90.0])
    orient = pd.Series([0.0])
    par, perp = step_01f._wind_decomposition(ws, wd, orient)
    assert par.iloc[0] == pytest.approx(0.0, abs=1e-10)
    assert perp.iloc[0] == pytest.approx(10.0)


def test_wind_decomposition_magnitudes_nonnegative():
    """Both components returned as absolute magnitudes."""
    ws = pd.Series([10.0])
    wd = pd.Series([180.0])  # opposite direction
    orient = pd.Series([0.0])
    par, perp = step_01f._wind_decomposition(ws, wd, orient)
    assert par.iloc[0] == pytest.approx(10.0)  # not -10
    assert perp.iloc[0] == pytest.approx(0.0, abs=1e-10)


# ── _classify_garbage_time ──────────────────────────────────────────────────


def test_garbage_time_blowout_q4():
    score_diff = pd.Series([21, -22, 14, 28, 30])
    period     = pd.Series([4, 4, 4, 3, 4])
    out = step_01f._classify_garbage_time(score_diff, period)
    # 21 in Q4 → garbage; -22 in Q4 → garbage; 14 in Q4 → not (under threshold);
    # 28 in Q3 → not (period < 4); 30 in Q4 → garbage
    assert out.tolist() == [1, 1, 0, 0, 1]


def test_garbage_time_dtype_int8():
    out = step_01f._classify_garbage_time(pd.Series([0]), pd.Series([1]))
    assert out.dtype.name == "int8"
