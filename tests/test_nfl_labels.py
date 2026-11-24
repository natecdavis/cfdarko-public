"""
Unit tests for src/scrapers/nfl_data.py label computation functions.

Tests compute_second_contract_label and compute_nfl_labels invariants:
correct exclusions, binary role targets, non-negative impact targets.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

# nfl_data.py tries to import nfl_data_py at module level (graceful fallback if missing)
# It also imports lightgbm (graceful) and rapidfuzz (graceful). No stubs needed.
from src.scrapers.nfl_data import compute_second_contract_label, compute_nfl_labels


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def contracts_df():
    """
    Contracts with several types:
      pfrA: rookie(2015) + qualifying second(2019)
      pfrB: rookie(2016) + 5th-year-opt(2020) + qualifying second(2022)
      pfrC: rookie(2017) only — washout
      pfrD: rookie(2018) + extension(2022)
    """
    return pd.DataFrame({
        "pfr_id": [
            "pfrA", "pfrA",
            "pfrB", "pfrB", "pfrB",
            "pfrC",
            "pfrD", "pfrD",
        ],
        "year_signed": [
            2015, 2019,
            2016, 2020, 2022,
            2017,
            2018, 2022,
        ],
        "draft_year": [
            2015, 2015,
            2016, 2016, 2016,
            2017,
            2018, 2018,
        ],
        "draft_round": [
            3, 3,
            1, 1, 1,
            4,
            2, 2,
        ],
        "years": [
            4, 5,
            4, 1, 4,
            4,
            4, 4,
        ],
        "apy_cap_pct": [
            0.005, 0.025,
            0.008, 0.000, 0.030,
            0.006,
            0.004, 0.018,
        ],
    })


@pytest.fixture
def bridge_players_df():
    return pd.DataFrame({
        "player_id": ["P001", "P002", "P003", "P004"],
        "player_name": ["Alice", "Bob", "Charlie", "Dave"],
        "pfr_id": ["pfrA", "pfrB", "pfrC", "pfrD"],
        "college_team": ["TestU"] * 4,
        "draft_year": [2015, 2016, 2017, 2018],
        "position_group": ["QB", "WR", "RB", "DL"],
        "career_w_av": [30.0, 45.0, 5.0, 20.0],
        "is_udfa": [False] * 4,
    })


# ── compute_second_contract_label ─────────────────────────────────────────────

def test_second_contract_excludes_rookie_deal(contracts_df):
    """Contract signed in draft_year should be excluded (rookie deal)."""
    result = compute_second_contract_label(contracts_df)
    # pfrA's first contract is year_signed==draft_year=2015 → excluded
    # pfrA's qualifying second is 2019 → APY = 0.025
    assert "pfrA" in result.index
    assert abs(result.loc["pfrA", "label_apy_raw"] - 0.025) < 1e-10


def test_second_contract_excludes_fifth_year_option(contracts_df):
    """Round-1 pick, years==1, year_signed==draft_year+4 → excluded."""
    result = compute_second_contract_label(contracts_df)
    # pfrB: 5th-yr opt is 2020 (=2016+4), years=1, round=1 → excluded
    # pfrB's qualifying second is 2022 → APY = 0.030
    assert "pfrB" in result.index
    assert abs(result.loc["pfrB", "label_apy_raw"] - 0.030) < 1e-10


def test_second_contract_washout_is_zero(contracts_df):
    """Player with only a rookie deal gets label 0.0."""
    result = compute_second_contract_label(contracts_df)
    assert "pfrC" in result.index
    assert result.loc["pfrC", "label_apy_raw"] == 0.0


def test_second_contract_picks_earliest_qualifying(contracts_df):
    """Picks the *first* qualifying contract, not the most recent."""
    # pfrD has qualifying contracts at 2018 (rookie → excluded) and 2022
    result = compute_second_contract_label(contracts_df)
    assert "pfrD" in result.index
    assert abs(result.loc["pfrD", "label_apy_raw"] - 0.018) < 1e-10


def test_second_contract_all_values_nonneg(contracts_df):
    """All returned APY cap % values must be ≥ 0."""
    result = compute_second_contract_label(contracts_df)
    vals = result["label_apy_raw"]
    assert (vals >= 0).all(), f"Negative apy_cap_pct: {vals[vals < 0]}"


def test_second_contract_indexed_by_pfr_id(contracts_df):
    """Result is a per-pfr_id frame (was a bare Series before the gate work
    needed to return the raw value, the rebased value and the role flag)."""
    result = compute_second_contract_label(contracts_df)
    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == {"pfrA", "pfrB", "pfrC", "pfrD"}
    assert "label_apy_raw" in result.columns


def test_second_contract_empty_df():
    """Empty contracts DataFrame → empty or zero-filled frame, no crash."""
    empty = pd.DataFrame(columns=["pfr_id", "year_signed", "draft_year",
                                   "draft_round", "years", "apy_cap_pct"])
    result = compute_second_contract_label(empty)
    assert isinstance(result, pd.DataFrame)


# ── compute_nfl_labels ────────────────────────────────────────────────────────

def test_role_target_binary(bridge_players_df, contracts_df):
    """role_target must be in {0.0, 1.0} for all rows."""
    contract_label = compute_second_contract_label(contracts_df)
    result = compute_nfl_labels(
        bridge_players=bridge_players_df,
        snap_data=pd.DataFrame(),
        contract_label=contract_label,
    )
    valid = {0.0, 1.0}
    bad = result["role_target"][~result["role_target"].isin(valid)]
    assert bad.empty, f"Non-binary role_target values: {bad.values}"


def test_impact_target_nonneg(bridge_players_df, contracts_df):
    """impact_target must be ≥ 0.0 for all rows."""
    contract_label = compute_second_contract_label(contracts_df)
    result = compute_nfl_labels(
        bridge_players=bridge_players_df,
        snap_data=pd.DataFrame(),
        contract_label=contract_label,
    )
    assert (result["impact_target"] >= 0).all(), (
        f"Negative impact_target: {result['impact_target'][result['impact_target'] < 0]}"
    )


def test_no_null_role_target_for_bridge(bridge_players_df, contracts_df):
    """All bridge players must get a non-null role_target."""
    contract_label = compute_second_contract_label(contracts_df)
    result = compute_nfl_labels(
        bridge_players=bridge_players_df,
        snap_data=pd.DataFrame(),
        contract_label=contract_label,
    )
    assert result["role_target"].notna().all()


def test_missing_contract_label_gives_zero(bridge_players_df):
    """contract_label=None → role_target and impact_target are 0.0 for all."""
    result = compute_nfl_labels(
        bridge_players=bridge_players_df,
        snap_data=pd.DataFrame(),
        contract_label=None,
    )
    assert (result["role_target"] == 0.0).all()
    assert (result["impact_target"] == 0.0).all()


def test_player_with_contract_gets_role_one(bridge_players_df, contracts_df):
    """Player with qualifying second contract (apy_cap_pct > 0) → role_target = 1."""
    contract_label = compute_second_contract_label(contracts_df)
    result = compute_nfl_labels(
        bridge_players=bridge_players_df,
        snap_data=pd.DataFrame(),
        contract_label=contract_label,
    )
    # pfrA and pfrB have qualifying second contracts → role = 1
    alice = result[result["pfr_id"] == "pfrA"]["role_target"].iloc[0]
    bob = result[result["pfr_id"] == "pfrB"]["role_target"].iloc[0]
    assert alice == 1.0
    assert bob == 1.0


def test_washout_gets_role_zero(bridge_players_df, contracts_df):
    """Player with only rookie deal (pfrC) → role_target = 0."""
    contract_label = compute_second_contract_label(contracts_df)
    result = compute_nfl_labels(
        bridge_players=bridge_players_df,
        snap_data=pd.DataFrame(),
        contract_label=contract_label,
    )
    charlie = result[result["pfr_id"] == "pfrC"]["role_target"].iloc[0]
    assert charlie == 0.0


def test_missing_pfr_id_gets_zero(contracts_df):
    """Players without pfr_id get role=0, impact=0."""
    players = pd.DataFrame({
        "player_id": ["X001"],
        "player_name": ["No PFR"],
        "pfr_id": [None],
        "college_team": ["U"],
        "draft_year": [2018],
        "position_group": ["QB"],
        "career_w_av": [5.0],
        "is_udfa": [False],
    })
    contract_label = compute_second_contract_label(contracts_df)
    result = compute_nfl_labels(players, snap_data=pd.DataFrame(),
                                contract_label=contract_label)
    assert result["role_target"].iloc[0] == 0.0
    assert result["impact_target"].iloc[0] == 0.0
