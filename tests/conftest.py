"""
Shared fixtures for cfDARKO unit tests.
All fixtures return in-memory DataFrames — no parquet files, no API calls.
"""
import sys
from pathlib import Path

# Ensure cfdarko root is on the path so src/pipeline modules resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


# ── Gamelogs fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_gamelogs():
    """Synthetic player-game DataFrame for Steps 4–6."""
    n = 60
    rng = np.random.default_rng(0)

    player_ids = [f"P{i:03d}" for i in range(10)]
    rows = []
    for pid in player_ids:
        n_games = rng.integers(4, 10)
        base_date = pd.Timestamp("2019-09-01")
        for g in range(n_games):
            rows.append({
                "player_id": pid,
                "game_id": f"G{pid}_{g}",
                "season_year": 2019 + g // 12,
                "school": "TestU",
                "position_group": "QB" if pid < "P005" else "WR",
                "game_date": base_date + pd.Timedelta(days=7 * g),
                "snaps": rng.integers(20, 80),
                "snaps_are_real": bool(rng.integers(0, 2)),
                "adjusted_pass_yards": rng.normal(0, 5),
                "adjusted_rush_yards": rng.normal(0, 3),
                "adjusted_rec_yards": rng.normal(0, 4),
                "adjusted_tackles": rng.normal(0, 2),
                "pass_yards": rng.integers(50, 300),
                "rush_yards": rng.integers(0, 100),
                "rec_yards": rng.integers(0, 150),
                "tackles": rng.integers(0, 10),
                "pass_att": rng.integers(10, 40),
                "rush_att": rng.integers(0, 20),
                "rec": rng.integers(0, 8),
                "pass_yards_p100": rng.uniform(100, 400),
                "rush_yards_p100": rng.uniform(0, 150),
                "rec_yards_p100": rng.uniform(0, 200),
                "tackles_p100": rng.uniform(0, 20),
            })

    return pd.DataFrame(rows)


# ── Bridge players fixture ─────────────────────────────────────────────────────

@pytest.fixture
def mock_bridge_players():
    """Synthetic bridge player DataFrame."""
    rng = np.random.default_rng(1)
    n = 20
    return pd.DataFrame({
        "player_id": [f"P{i:03d}" for i in range(n)],
        "player_name": [f"Player {i}" for i in range(n)],
        "pfr_id": [f"pfr{i:04d}" for i in range(n)],
        "college_team": ["TestU"] * n,
        "draft_year": rng.integers(2010, 2022, n),
        "last_college_season_year": rng.integers(2009, 2021, n),
        "position_group": (["QB"] * 5 + ["WR"] * 5 + ["RB"] * 5 + ["DL"] * 5),
        "position": (["QB"] * 5 + ["WR"] * 5 + ["RB"] * 5 + ["DE"] * 5),
        "career_w_av": rng.integers(0, 60, n).astype(float),
        "is_udfa": [False] * n,
    })


# ── Skill snapshot fixture ─────────────────────────────────────────────────────

@pytest.fixture
def mock_skill_snapshot():
    """Synthetic player skill snapshot (one row per player)."""
    rng = np.random.default_rng(2)
    n = 30
    groups = (["QB"] * 8 + ["WR"] * 8 + ["RB"] * 7 + ["DL"] * 7)
    return pd.DataFrame({
        "player_id": [f"P{i:03d}" for i in range(n)],
        "player_name": [f"Player {i}" for i in range(n)],
        "school": ["TestU"] * n,
        "position_group": groups,
        "last_college_season_year": rng.integers(2015, 2023, n),
        "n_games": rng.integers(5, 40, n),
        "n_seasons": rng.integers(1, 5, n),
        "kalman_mean_pass_yards": rng.normal(0, 3, n),
        "kalman_var_pass_yards": rng.uniform(0.1, 5, n),
        "ewma_pass_yards": rng.normal(0, 3, n),
        "ewma_var_pass_yards": rng.uniform(0.1, 5, n),
        "kalman_mean_rec_yards": rng.normal(0, 3, n),
        "kalman_var_rec_yards": rng.uniform(0.1, 5, n),
        "ewma_rec_yards": rng.normal(0, 3, n),
        "kalman_mean_tackles": rng.normal(0, 2, n),
        "kalman_var_tackles": rng.uniform(0.1, 3, n),
        "ewma_tackles": rng.normal(0, 2, n),
        "kalman_mean_sacks": rng.normal(0, 1, n),
        "kalman_mean_tfl": rng.normal(0, 1, n),
        "kalman_mean_qb_hurries": rng.normal(0, 0.5, n),
        "height_in": rng.uniform(68, 80, n),
        "weight_lbs": rng.uniform(170, 310, n),
        "conference_tier": rng.choice([1, 2, 3], n),
    })


# ── Labels fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_labels():
    """Synthetic NFL draft labels."""
    rng = np.random.default_rng(3)
    n = 30
    return pd.DataFrame({
        "player_id": [f"P{i:03d}" for i in range(n)],
        "pfr_id": [f"pfr{i:04d}" for i in range(n)],
        "draft_year": rng.integers(2010, 2022, n),
        "last_college_season_year": rng.integers(2009, 2021, n),
        "role_target": rng.choice([0.0, 1.0], n),
        "impact_target": rng.uniform(0, 0.05, n),
        "career_w_av": rng.integers(0, 60, n).astype(float),
        "is_bridge": [True] * n,
    })


# ── Snap data fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_snap_data():
    """Synthetic NFL snap percentage data."""
    rng = np.random.default_rng(4)
    n = 60
    return pd.DataFrame({
        "pfr_player_id": [f"pfr{i:04d}" for i in range(n // 4) for _ in range(4)],
        "season": [2018 + j for _ in range(n // 4) for j in range(4)],
        "offense_pct": rng.uniform(0, 1, n),
        "defense_pct": rng.uniform(0, 1, n),
        "st_pct": rng.uniform(0, 0.5, n),
        "n_games": rng.integers(1, 17, n),
    })


# ── Contracts fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_contracts_df():
    """
    Synthetic contracts DataFrame with several contract types:
    - Rookie deals (excluded from second contract)
    - 5th-year options for round-1 picks (excluded)
    - Qualifying second contracts
    - Third+ contracts
    """
    return pd.DataFrame({
        "pfr_id": [
            # Player A: rookie + qualifying second
            "pfrA", "pfrA",
            # Player B: rookie + 5th-year option + qualifying second
            "pfrB", "pfrB", "pfrB",
            # Player C: rookie only (washout)
            "pfrC",
            # Player D: rookie + third contract only (no qualifying second?)
            "pfrD", "pfrD",
        ],
        "year_signed": [
            2015, 2019,       # A: rookie 2015, second 2019
            2016, 2020, 2022, # B: rookie 2016, 5th-yr-opt 2020, second 2022
            2017,             # C: rookie 2017 only
            2018, 2022,       # D: rookie 2018, next deal 2022
        ],
        "draft_year": [
            2015, 2015,
            2016, 2016, 2016,
            2017,
            2018, 2018,
        ],
        "draft_round": [
            3, 3,   # A is round 3
            1, 1, 1,  # B is round 1 (5th yr option applicable)
            4,      # C is round 4
            2, 2,   # D is round 2
        ],
        "years": [
            4, 5,           # A: 4-yr rookie, 5-yr second
            4, 1, 4,        # B: 4-yr rookie, 1-yr 5th-yr-opt, 4-yr second
            4,              # C: 4-yr rookie only
            4, 4,           # D: 4-yr rookie, 4-yr next deal
        ],
        "apy_cap_pct": [
            0.005, 0.025,         # A
            0.008, 0.000, 0.030,  # B
            0.006,                # C
            0.004, 0.018,         # D
        ],
    })


# ── Comps fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_comps():
    """Synthetic comps DataFrame."""
    rows = []
    for pid in [f"P{i:03d}" for i in range(5)]:
        for tier in ("all", "bridge"):
            for rank in range(1, 4):
                rows.append({
                    "player_id": pid,
                    "comp_tier": tier,
                    "comp_rank": rank,
                    "comp_player_id": f"C{rank:02d}",
                    "comp_player_name": f"Comp {rank}",
                    "comp_school": "OldU",
                    "comp_role_target": float(rank % 2),
                    "comp_impact_target": rank * 0.01,
                    "comp_career_w_av": rank * 10.0,
                    "comp_first_nfl_season": 2010 + rank,
                    "last_college_season_year": 2018,
                })
    return pd.DataFrame(rows)
