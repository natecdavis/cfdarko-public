"""
Unit tests for config additions for the play layer.

Verifies eligibility matrices and naming conventions don't have stale
references that would break downstream steps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import config as C


def test_play_stat_col_maps_all_known_stats():
    """Every PLAY_STAT in ALL_PLAY_STAT_TYPES must have a snake_case column."""
    for s in C.ALL_PLAY_STAT_TYPES:
        assert s in C.PLAY_STAT_COL, f"Missing PLAY_STAT_COL entry for {s!r}"
        # Snake_case sanity
        col = C.PLAY_STAT_COL[s]
        assert col == col.lower(), f"PLAY_STAT_COL value not lowercase: {col!r}"
        assert " " not in col, f"PLAY_STAT_COL value has space: {col!r}"


def test_play_continuous_stats_subset_of_all():
    """All continuous-stat names must exist in the master stat list."""
    assert C.PLAY_CONTINUOUS_STATS.issubset(set(C.ALL_PLAY_STAT_TYPES)), \
        f"PLAY_CONTINUOUS_STATS - ALL_PLAY_STAT_TYPES = " \
        f"{C.PLAY_CONTINUOUS_STATS - set(C.ALL_PLAY_STAT_TYPES)}"


def test_position_eligible_play_types_covers_all_modeled_positions():
    """Every position we model at play grain (i.e., not OL) should have an
    eligibility set."""
    expected = {"QB", "RB", "WR", "TE", "DL", "EDGE", "LB", "DB", "K", "P"}
    assert set(C.POSITION_ELIGIBLE_PLAY_TYPES.keys()) == expected, \
        f"Position eligibility missing: {expected - set(C.POSITION_ELIGIBLE_PLAY_TYPES.keys())}"


def test_position_eligible_play_types_nonempty():
    for pos, plays in C.POSITION_ELIGIBLE_PLAY_TYPES.items():
        assert len(plays) > 0, f"Empty eligibility for {pos}"


def test_position_side_covers_modeled_positions():
    """All modeled positions have a side assignment for catch-all gating."""
    for pos in C.POSITION_ELIGIBLE_PLAY_TYPES.keys():
        assert pos in C.POSITION_SIDE, f"Missing POSITION_SIDE for {pos}"
        assert C.POSITION_SIDE[pos] in ("offense", "defense"), \
            f"Invalid side for {pos}: {C.POSITION_SIDE[pos]}"


def test_eligibility_matrix_includes_basic_play_types():
    """The most common play_types must have eligibility entries."""
    required = {"Rush", "Pass", "Pass Reception", "Pass Incompletion", "Sack",
                "Field Goal Good", "Punt", "Kickoff"}
    for pt in required:
        assert pt in C.PLAY_TYPE_STAT_ELIGIBILITY, \
            f"Missing PLAY_TYPE_STAT_ELIGIBILITY entry for {pt!r}"


def test_eligibility_matrix_consistency():
    """Stats appearing in PLAY_TYPE_STAT_ELIGIBILITY must all be known stats."""
    known = set(C.ALL_PLAY_STAT_TYPES)
    for pt, stats in C.PLAY_TYPE_STAT_ELIGIBILITY.items():
        unknown = stats - known
        assert not unknown, \
            f"Unknown stat_types in PLAY_TYPE_STAT_ELIGIBILITY[{pt!r}]: {unknown}"


def test_lgbm_play_params_have_required_keys():
    """LightGBM training requires these keys."""
    required = {"objective", "num_leaves", "learning_rate", "num_boost_round"}
    assert required.issubset(set(C.LGBM_PLAY_PARAMS.keys())), \
        f"Missing keys: {required - set(C.LGBM_PLAY_PARAMS.keys())}"


def test_play_context_features_has_categoricals():
    """PLAY_CATEGORICAL_FEATURES must be a subset of PLAY_CONTEXT_FEATURES."""
    assert set(C.PLAY_CATEGORICAL_FEATURES).issubset(set(C.PLAY_CONTEXT_FEATURES)), \
        f"Categorical feature not in context features: " \
        f"{set(C.PLAY_CATEGORICAL_FEATURES) - set(C.PLAY_CONTEXT_FEATURES)}"


def test_garbage_time_rules_keys():
    """Required keys for the garbage-time computation."""
    required = {"score_diff_abs_gte", "period_gte"}
    assert required.issubset(set(C.GARBAGE_TIME_RULES.keys()))


def test_play_skill_stat_positions_gate_defenders_out():
    """Offensive play-grain skill stats must never route to positions that
    can't produce them. Defenders / OL / P produce no rushing, receiving,
    passing or field-goal production, so none of them may appear in any
    PLAY_SKILL_STAT_POSITIONS value; K only gets field-goal stats."""
    non_producers = {"DL", "EDGE", "LB", "DB", "OL", "P"}
    for stat, positions in C.PLAY_SKILL_STAT_POSITIONS.items():
        assert not (non_producers & positions), \
            f"{stat} wrongly routed to non-producing position(s): {non_producers & positions}"
    # K is field-goal only.
    for stat, positions in C.PLAY_SKILL_STAT_POSITIONS.items():
        if "K" in positions:
            assert stat.startswith("field_goal"), \
                f"K wrongly gets non-kicking play stat {stat}"
