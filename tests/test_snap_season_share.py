"""
Snap pct as a share of the team's season (2026-10-31).

The cached snap rows hold the mean pct over games APPEARED; every consumer
must count missed games as 0 (season_snap_shares).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

import src.scrapers.nfl_data as nd


@pytest.fixture
def snaps():
    return pd.DataFrame({
        "pfr_player_id": ["a", "a", "b"],
        "season":        [2020, 2021, 2021],
        "offense_pct":   [0.80, 0.50, 1.00],
        "defense_pct":   [0.00, 0.00, 0.00],
        "st_pct":        [0.40, 0.17, 0.00],
        "n_games":       [4, 17, 18],
    })


def test_scales_by_team_games(snaps):
    out = nd.season_snap_shares(snaps)
    assert out["offense_pct"].tolist() == pytest.approx([0.20, 0.50, 1.00])
    assert out["st_pct"].iloc[0] == pytest.approx(0.10)
    assert snaps["offense_pct"].iloc[0] == 0.80        # input untouched


def test_without_n_games_is_unchanged(snaps):
    raw = snaps.drop(columns="n_games")
    assert nd.season_snap_shares(raw)["offense_pct"].tolist() == [0.80, 0.50, 1.00]


def test_reg_season_games():
    assert nd.reg_season_games(2020) == 16 and nd.reg_season_games(2021) == 17


def test_trajectory_seasonal_features_use_season_share(snaps, monkeypatch):
    monkeypatch.setattr(nd, "_nfl_available", False)
    bp = pd.DataFrame({"pfr_id": ["a"], "draft_year": [2020], "position_group": ["WR"]})
    f = nd.fetch_nfl_seasonal_features(bp, snaps, max_years=2)
    assert f.at["a", "y1_snap_pct"] == pytest.approx(0.20)
    assert f.at["a", "y2_snap_pct"] == pytest.approx(0.50)
    assert f.at["a", "y1_n_games"] == 4


def test_snap_features_use_season_share(snaps):
    bp = pd.DataFrame({"pfr_id": ["a"], "draft_year": [2020], "position_group": ["WR"]})
    f = nd.fetch_snap_features(bp, snaps, window=2)
    assert f.at["a", "year1_snap_pct"] == pytest.approx(0.20)
    assert f.at["a", "mean_snap_pct_4yr"] == pytest.approx(0.35)
