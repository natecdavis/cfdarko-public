"""Tests for src/models/draft_pool.py."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.draft_pool import draft_pool  # noqa: E402


def _setup(tmp_path, next_rosters=True):
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir()
    proc.mkdir()
    # first seasons: 1 → 2022 (eligible), 2 → 2024 freshman (not), 3 → 2023,
    # 4 → 2024 but listed JR (JUCO transfer: eligible), 5 → returned, 6 → sat out
    pd.DataFrame({"player_id": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
                  "season_year": [2022, 2024, 2023, 2025, 2022, 2023, 2025, 2025, 2025, 2025, 2025],
                  "class_year_int": [1, 1, 1, 3, 1, 2, 4, 2, 3, 3, 4]}
                 ).to_parquet(raw / "rosters.parquet")
    pd.DataFrame({"player_id": [1], "season_year": [2022]}).to_parquet(raw / "player_seasons.parquet")
    if next_rosters:
        pd.DataFrame({"player_id": [5]}).to_parquet(raw / "rosters_2026_season.parquet")
    pd.DataFrame({"player_id": [6.0], "draft_year": [2026]}).to_parquet(proc / "bridge_players.parquet")
    pd.DataFrame({"player_id": [6], "draft_year": [2026], "hasan_rank": [50]}
                 ).to_parquet(proc / "hasan_ranks.parquet")
    players = pd.DataFrame({"player_id": ["1", "2", "3", "4", "5", "6", "7"],
                            "last_college_season_year": [2025, 2025, 2025, 2025, 2025, 2023, 2025]})
    return players, raw, proc


def test_pool_rules(tmp_path):
    players, raw, proc = _setup(tmp_path)
    pool = draft_pool(players, 2026, raw, proc)
    assert pool.tolist() == [
        True,    # 1: first season 2022, left
        False,   # 2: freshman who left — not three years out of high school
        True,    # 3: first season 2023
        True,    # 4: first FBS season 2025 but listed JR
        False,   # 5: back on a 2026 roster
        True,    # 6: sat out 2024-25, drafted in 2026
        False,   # 7: no season history at all
    ]


def test_pool_unknown_without_next_season_rosters(tmp_path):
    players, raw, proc = _setup(tmp_path, next_rosters=False)
    assert draft_pool(players, 2026, raw, proc) is None
