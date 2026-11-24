"""
Config consistency tests.
Checks that the config constants are internally self-consistent —
no orphaned stats, no missing mappings, no out-of-bounds defaults.
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config as C


# ── Stat categorization ────────────────────────────────────────────────────────

def test_all_modeled_stats_categorized():
    """Every stat in MODELED_STATS must be either a snap-counting stat or have
    an attempt column mapping. No stat should fall into neither bucket."""
    uncategorized = []
    for pos, stats in C.MODELED_STATS.items():
        for stat in stats:
            in_counting = stat in C.SNAP_COUNTING_STATS
            in_rate = stat in C.RATE_STAT_ATTEMPT_COL
            if not in_counting and not in_rate:
                uncategorized.append((pos, stat))
    assert uncategorized == [], (
        f"Stats in MODELED_STATS but in neither SNAP_COUNTING_STATS nor RATE_STAT_ATTEMPT_COL: "
        f"{uncategorized}"
    )


def test_all_stats_no_duplicates():
    """ALL_STATS should have no duplicate entries."""
    assert len(C.ALL_STATS) == len(set(C.ALL_STATS)), "ALL_STATS has duplicate entries"


def test_ol_has_no_modeled_stats():
    """OL has no per-play stats — only physical/team features."""
    assert C.MODELED_STATS["OL"] == [], "OL should have empty MODELED_STATS"


def test_rate_stat_attempt_cols_are_strings():
    """Every value in RATE_STAT_ATTEMPT_COL should be a non-empty string."""
    for stat, col in C.RATE_STAT_ATTEMPT_COL.items():
        assert isinstance(col, str) and len(col) > 0, (
            f"RATE_STAT_ATTEMPT_COL[{stat!r}] = {col!r} is not a valid column name"
        )


# ── Position group maps ────────────────────────────────────────────────────────

def test_position_to_group_roundtrip():
    """Every position in POSITION_GROUPS[group] must map back to that group
    via POSITION_TO_GROUP."""
    for group, positions in C.POSITION_GROUPS.items():
        for pos in positions:
            mapped = C.POSITION_TO_GROUP.get(pos)
            assert mapped == group, (
                f"POSITION_TO_GROUP[{pos!r}] = {mapped!r}, expected {group!r}"
            )


def test_snap_proxy_position_groups_valid():
    """Every key in SNAP_PROXY_BY_POSITION must appear in POSITION_GROUPS."""
    for pos_group in C.SNAP_PROXY_BY_POSITION:
        assert pos_group in C.POSITION_GROUPS, (
            f"SNAP_PROXY_BY_POSITION has key {pos_group!r} not in POSITION_GROUPS"
        )


def test_all_position_groups_in_snap_proxy():
    """Every position group in POSITION_GROUPS should have an entry in SNAP_PROXY_BY_POSITION."""
    for group in C.POSITION_GROUPS:
        assert group in C.SNAP_PROXY_BY_POSITION, (
            f"POSITION_GROUPS key {group!r} missing from SNAP_PROXY_BY_POSITION"
        )


def test_role_threshold_keys_match_position_groups():
    """ROLE_SNAP_PCT_THRESHOLD_BY_GROUP keys must match POSITION_GROUPS keys exactly."""
    assert set(C.ROLE_SNAP_PCT_THRESHOLD_BY_GROUP.keys()) == set(C.POSITION_GROUPS.keys()), (
        f"Mismatch: threshold keys={set(C.ROLE_SNAP_PCT_THRESHOLD_BY_GROUP.keys())}, "
        f"position_group keys={set(C.POSITION_GROUPS.keys())}"
    )


# ── Conference tiers ───────────────────────────────────────────────────────────

def test_conference_tiers_valid_values():
    """Every value in CONFERENCE_TIERS must be 1, 2, or 3."""
    valid = {1, 2, 3}
    bad = {k: v for k, v in C.CONFERENCE_TIERS.items() if v not in valid}
    assert not bad, f"CONFERENCE_TIERS has invalid tier values: {bad}"


def test_default_conference_tier_valid():
    """DEFAULT_CONFERENCE_TIER must be in {1, 2, 3}."""
    assert C.DEFAULT_CONFERENCE_TIER in {1, 2, 3}, (
        f"DEFAULT_CONFERENCE_TIER={C.DEFAULT_CONFERENCE_TIER} not in {{1, 2, 3}}"
    )


# ── Kalman parameters ──────────────────────────────────────────────────────────

def test_kalman_obs_noise_covers_all_stats():
    """KALMAN_OBS_NOISE_BY_STAT should have an entry for every stat in ALL_STATS,
    or fall back to the global KALMAN_OBS_NOISE default."""
    # We don't require every stat to have a specific entry (global fallback exists),
    # but every entry that IS present should be positive.
    for stat, noise in C.KALMAN_OBS_NOISE_BY_STAT.items():
        assert noise > 0, f"KALMAN_OBS_NOISE_BY_STAT[{stat!r}] = {noise} is not positive"


def test_kalman_prior_var_by_stat_all_positive():
    """All per-stat prior variances must be positive."""
    for stat, var in C.KALMAN_PRIOR_VAR_BY_STAT.items():
        assert var > 0, f"KALMAN_PRIOR_VAR_BY_STAT[{stat!r}] = {var} is not positive"


def test_kalman_process_noise_positive():
    """Global and per-stat process noise must be positive."""
    assert C.KALMAN_PROCESS_NOISE_PER_DAY > 0
    for stat, q in C.KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT.items():
        assert q > 0, f"KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT[{stat!r}] = {q} not positive"


# ── EWMA parameters ────────────────────────────────────────────────────────────

def test_default_beta_in_bounds():
    """Every value in DEFAULT_BETA must be within BETA_BOUNDS [0.990, 1.000]."""
    lo, hi = C.BETA_BOUNDS
    bad = {stat: b for stat, b in C.DEFAULT_BETA.items() if not (lo <= b <= hi)}
    assert not bad, f"DEFAULT_BETA values out of bounds {C.BETA_BOUNDS}: {bad}"


def test_default_beta_covers_all_stats():
    """DEFAULT_BETA should have an entry for every stat in ALL_STATS."""
    missing = [s for s in C.ALL_STATS if s not in C.DEFAULT_BETA]
    assert not missing, f"ALL_STATS stats missing from DEFAULT_BETA: {missing}"


# ── Ref counts ────────────────────────────────────────────────────────────────

def test_kalman_ref_snaps_positive():
    assert C.KALMAN_REF_SNAPS > 0


def test_kalman_ref_attempts_positive():
    for stat, ref in C.KALMAN_REF_ATTEMPTS.items():
        assert ref > 0, f"KALMAN_REF_ATTEMPTS[{stat!r}] = {ref} not positive"


def test_ref_attempts_covers_rate_stats():
    """Every rate stat should have a ref attempt count."""
    missing = [s for s in C.RATE_STAT_ATTEMPT_COL if s not in C.KALMAN_REF_ATTEMPTS]
    assert not missing, f"Rate stats missing from KALMAN_REF_ATTEMPTS: {missing}"


# ── Reverted context-feature gate ─────────────────────────────────────────────

def test_reverted_context_families_not_in_context_features():
    """The four reverted families gated by COMPUTE_REVERTED_CONTEXT must stay
    out of CONTEXT_FEATURES while the gate is off — if a column is re-added to
    CONTEXT_FEATURES, the gate must be flipped on (step 1 would otherwise emit
    NaN-only features)."""
    gated = (set(C.TEAM_ROLLING_FEATURES) | set(C.DEF_ROLLING_FEATURES)
             | set(C.OPP_OFF_VOLUME_FEATURES) | set(C.OPP_QB_ROLLING_FEATURES))
    if not C.COMPUTE_REVERTED_CONTEXT:
        overlap = gated & set(C.CONTEXT_FEATURES)
        assert not overlap, (
            f"COMPUTE_REVERTED_CONTEXT is False but CONTEXT_FEATURES contains "
            f"gated columns (step 1 will not compute them): {sorted(overlap)}"
        )
