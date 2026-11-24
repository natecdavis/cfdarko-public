"""
Unit tests for pipeline/01d_play_ingestion.py and src/scrapers/cfbd.py play
endpoints.

Covers schema invariants and helper-function behavior. Live API calls are
mocked — these tests do not hit CFBD.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


# ── fetch_plays output schema ─────────────────────────────────────────────


def _mock_play(play_id=1, game_id=10, drive_id=100, **overrides):
    """Build a mock cfbd Play-like object."""
    m = MagicMock()
    m.id = play_id
    m.drive_id = drive_id
    m.game_id = game_id
    m.drive_number = 1
    m.play_number = 1
    m.offense = overrides.get("offense", "Georgia")
    m.defense = overrides.get("defense", "Alabama")
    m.offense_conference = "SEC"
    m.defense_conference = "SEC"
    m.home = "Georgia"
    m.away = "Alabama"
    m.offense_score = 0
    m.defense_score = 0
    m.period = 1
    m.clock = MagicMock(minutes=10, seconds=30)
    m.offense_timeouts = 3
    m.defense_timeouts = 3
    m.yardline = 25
    m.yards_to_goal = 75
    m.down = 1
    m.distance = 10
    m.yards_gained = overrides.get("yards_gained", 5)
    m.scoring = False
    m.play_type = overrides.get("play_type", "Rush")
    m.ppa = 0.1
    return m


def test_fetch_plays_schema():
    from src.scrapers.cfbd import fetch_plays
    plays_mock = [_mock_play(play_id=i) for i in range(3)]
    with patch("src.scrapers.cfbd.cfbd.PlaysApi") as api_class, \
         patch("src.scrapers.cfbd._api_call_with_retry", return_value=plays_mock), \
         patch("src.scrapers.cfbd._rate_limit"):
        df = fetch_plays(year=2024, week=5)

    # Schema: 25 cols including the ones step 1f relies on
    required = {"play_id", "drive_id", "game_id", "season_year", "week",
                "offense", "defense", "down", "distance", "yards_to_goal",
                "yards_gained", "play_type", "ppa", "period", "clock_seconds",
                "offense_score", "defense_score"}
    assert required.issubset(set(df.columns)), \
        f"Missing cols: {required - set(df.columns)}"
    assert len(df) == 3
    assert df["season_year"].unique().tolist() == [2024]
    assert df["week"].unique().tolist() == [5]
    # clock_seconds derived from minutes*60+seconds
    assert (df["clock_seconds"] == 10 * 60 + 30).all()


def test_fetch_plays_handles_empty():
    from src.scrapers.cfbd import fetch_plays
    with patch("src.scrapers.cfbd._api_call_with_retry", return_value=[]), \
         patch("src.scrapers.cfbd._rate_limit"):
        df = fetch_plays(year=2024, week=5)
    assert df.empty


# ── fetch_play_stats output schema ──────────────────────────────────────


def _mock_play_stat(play_id=1, stat_type="Rush", stat=5, athlete_id="abc"):
    m = MagicMock()
    m.play_id = play_id
    m.drive_id = 100
    m.game_id = 10
    m.season = 2024
    m.week = 5
    m.team = "Georgia"
    m.conference = "SEC"
    m.opponent = "Alabama"
    m.athlete_id = athlete_id
    m.athlete_name = "Test Player"
    m.stat_type = stat_type
    m.stat = stat
    return m


def test_fetch_play_stats_schema():
    from src.scrapers.cfbd import fetch_play_stats
    rows = [
        _mock_play_stat(play_id=1, stat_type="Rush", stat=12),
        _mock_play_stat(play_id=2, stat_type="Reception", stat=18, athlete_id="def"),
        _mock_play_stat(play_id=2, stat_type="Target", stat=1, athlete_id="def"),
    ]
    with patch("src.scrapers.cfbd._api_call_with_retry", return_value=rows), \
         patch("src.scrapers.cfbd._rate_limit"):
        df = fetch_play_stats(year=2024, week=5, team="Georgia")
    required = {"play_id", "drive_id", "game_id", "season_year", "week",
                "team", "athlete_id", "athlete_name", "stat_type", "stat"}
    assert required.issubset(set(df.columns))
    assert len(df) == 3
    assert df["stat_type"].tolist() == ["Rush", "Reception", "Target"]


def test_fetch_play_stats_warns_on_2000_cap(caplog):
    """If response returns exactly 2000 rows, log a warning about truncation."""
    import logging
    from src.scrapers.cfbd import fetch_play_stats
    rows = [_mock_play_stat(play_id=i) for i in range(2000)]
    with patch("src.scrapers.cfbd._api_call_with_retry", return_value=rows), \
         patch("src.scrapers.cfbd._rate_limit"), \
         caplog.at_level(logging.WARNING, logger="src.scrapers.cfbd"):
        df = fetch_play_stats(year=2024, week=5, team="Georgia")
    assert any("2000-row cap" in rec.message for rec in caplog.records), \
        "Expected 2000-cap warning"
    assert len(df) == 2000
