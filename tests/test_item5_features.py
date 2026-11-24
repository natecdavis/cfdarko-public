"""
Unit tests for the audit item-5 feature batch (2026-07-25).

Six new features across two grains:
  game-grain   capacity_diff_from_home                      (step 01)
  career       career_rec_long_max, career_rush_long_max,
               career_int_return_yards_per_int,
               career_solo_tackle_share                     (step 03)
  career       career_q1_play_share, career_q4_play_share    (step 03)

The invariant these tests exist to protect is MISSING-vs-ZERO. Every one of
these is a max or a rate over a career, and every one has a legitimate "no
measurement" state that must stay NaN: a player with 0 interceptions has no
return average, a venue with capacity 0 means unknown seating, a defender before
2016 has no solo-tackle data at all. Filling any of those with 0 would tell the
model "measured, and bad" instead of "not measured", and the model would read a
1990s-style missing-data pattern as ability.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

import config as C


# ── capacity_diff_from_home (game grain) ───────────────────────────────────────

def _cap_diff(game_cap, home_cap, bucket=None):
    """Reimplements step 01's capacity diff on scalars, including the
    capacity<=0 nulling, so the arithmetic contract can be tested without
    standing up the whole venue join."""
    bucket = float(bucket or C.CAPACITY_DIFF_BUCKET)
    g = pd.Series([game_cap], dtype="float64")
    h = pd.Series([home_cap], dtype="float64")
    g = g.where(g > 0)
    h = h.where(h > 0)
    return float(((g - h) / bucket).round().iloc[0] * bucket) if (g.notna() & h.notna()).iloc[0] else np.nan


def test_capacity_diff_home_game_is_zero():
    """A player at their own stadium has no crowd shock."""
    assert _cap_diff(80_000, 80_000) == 0.0


def test_capacity_diff_signed_and_bucketed():
    """Away at a bigger venue is positive; rounding is to CAPACITY_DIFF_BUCKET."""
    # 107k Michigan vs a 40k home → +67k → bucket to +70k at 10k granularity
    assert _cap_diff(107_601, 40_000) == 70_000.0
    # and the reverse trip is negative
    assert _cap_diff(40_000, 107_601) == -70_000.0


def test_capacity_zero_is_unknown_not_empty_stadium():
    """70 of 840 venues report capacity 0. That is missing data, not a stadium
    with no seats — a 0 would fabricate a -100k hostility shock."""
    assert np.isnan(_cap_diff(0, 80_000))
    assert np.isnan(_cap_diff(80_000, 0))
    assert np.isnan(_cap_diff(np.nan, 80_000))


def test_capacity_diff_is_built_but_withheld_from_the_model():
    """Built in step 01 and defaulted in the context builder, but deliberately
    NOT in CONTEXT_FEATURES: it has never been evaluated, and enabling it forces
    an adjusted_gamelogs rebuild plus 04b/04d recalibration. Keeping it out means
    it cannot ride along untested on the next retune."""
    assert "capacity_diff_from_home" not in C.CONTEXT_FEATURES


def test_capacity_diff_dropped_for_ol():
    """OL has no per-game stats, so venue features are inert there — it follows
    the same drop convention as altitude/surface rather than being left dangling."""
    assert "capacity_diff_from_home" in C.CONTEXT_FEATURE_DROP_BY_POSITION.get("OL", [])


def test_context_builder_emits_every_context_feature():
    """Guards the failure this batch actually hit: config listed the feature
    before src/models/contextual.py knew how to default it, so a fixture without
    the step-01 join raised instead of yielding NaN."""
    from src.models.contextual import build_context_features
    df = pd.DataFrame({
        "player_id": ["1", "2"],
        "game_id": [10, 11],
        "season_year": [2019, 2019],
        "week": [1, 2],
        "position_group": ["WR", "WR"],
    })
    out = build_context_features(df)
    assert "capacity_diff_from_home" in out.columns


# ── career aggregates (step 03 semantics) ──────────────────────────────────────

def _career_rates(rows: pd.DataFrame) -> pd.Series:
    """Reimplements the two career RATES from step 03 for one player."""
    s = rows.sum(numeric_only=True)
    out = {}
    denom_int = s.get("interceptions_int", 0) or np.nan
    out["career_int_return_yards_per_int"] = s.get("interceptions_yds", np.nan) / denom_int
    denom_tkl = s.get("tackles", 0) or np.nan
    share = s.get("tackles_solo", np.nan) / denom_tkl
    out["career_solo_tackle_share"] = share if (pd.isna(share) or share <= 1.0) else np.nan
    return pd.Series(out)


def test_int_return_average_is_a_rate_over_career_sums():
    rows = pd.DataFrame({"interceptions_yds": [30.0, 12.0],
                         "interceptions_int": [2.0, 1.0]})
    assert _career_rates(rows)["career_int_return_yards_per_int"] == pytest.approx(14.0)


def test_zero_interceptions_is_nan_not_zero_yards():
    """A defender who never picked one off has no ball-skill measurement. A 0
    here would rank him equal to someone with three 0-yard returns."""
    rows = pd.DataFrame({"interceptions_yds": [0.0], "interceptions_int": [0.0]})
    assert np.isnan(_career_rates(rows)["career_int_return_yards_per_int"])


def test_solo_share_is_a_fraction():
    rows = pd.DataFrame({"tackles_solo": [40.0, 20.0], "tackles": [80.0, 20.0]})
    assert _career_rates(rows)["career_solo_tackle_share"] == pytest.approx(0.6)


def test_solo_share_above_one_is_rejected_not_clipped():
    """tackles_solo is a subset of tackles, so >1 means the feed contradicts
    itself. Clipping to 1.0 would silently assert 'perfect solo rate'."""
    rows = pd.DataFrame({"tackles_solo": [90.0], "tackles": [80.0]})
    assert np.isnan(_career_rates(rows)["career_solo_tackle_share"])


def test_zero_tackles_is_nan_not_zero_share():
    rows = pd.DataFrame({"tackles_solo": [0.0], "tackles": [0.0]})
    assert np.isnan(_career_rates(rows)["career_solo_tackle_share"])


def test_quarter_shares_are_fractions_of_own_career_plays():
    """Shares, not counts, so the feature measures usage SHAPE and cannot be
    read as career length. Overtime periods stay in the denominator."""
    period = pd.Series([1, 1, 2, 3, 4, 4, 4, 5])
    total = len(period)
    q1 = (period == 1).sum() / total
    q4 = (period == 4).sum() / total
    assert q1 == pytest.approx(2 / 8)
    assert q4 == pytest.approx(3 / 8)
    assert q1 + q4 < 1.0


def test_q2_q3_shares_are_never_emitted():
    """Q1-Q4 shares sum to ~1, so emitting all four would be collinear by
    construction; Q2/Q3 also carry little starter-vs-backup contrast. This holds
    whether or not the quarter family is enabled."""
    assert "career_q2_play_share" not in C.ATTRIBUTE_MERGE_COLS
    assert "career_q3_play_share" not in C.ATTRIBUTE_MERGE_COLS


ITEM5_CAREER_FEATURES = ("career_rec_long_max", "career_rush_long_max",
                         "career_int_return_yards_per_int",
                         "career_solo_tackle_share",
                         "career_q1_play_share", "career_q4_play_share")


def test_item5_career_features_are_reverted_from_the_model_surface():
    """All three families scored negative on a 3-seed placebo-anchored A/B
    (long_max -0.0128 at -4.2 SE, def_ball -0.0019, quarter -0.0026), so step 03
    still computes them but no head consumes them.

    Re-enabling requires BOTH this list and step 8's _build_feature_cols. Wiring
    only one produces an experiment in which every arm is identical — the first
    run of this A/B reported Δρ = 0.0000 on all 33 heads for exactly that reason,
    which is why this test names both halves."""
    for col in ITEM5_CAREER_FEATURES:
        assert col not in C.ATTRIBUTE_MERGE_COLS, (
            f"{col} is back in ATTRIBUTE_MERGE_COLS — if that is intentional, "
            "the step 8 _build_feature_cols gate must be restored too, and the "
            "A/B re-run")


def test_step3_retains_the_derivation_code_for_the_reverted_features():
    """The revert keeps the step-03 derivation CODE so a re-test is a config
    change rather than an archaeology exercise — same convention as the
    career_max_conf_tier binaries.

    Note what this does NOT claim: step 03 filters its output to
    ATTRIBUTE_MERGE_COLS + a few debug extras (see its Output section), so with
    the six names removed from that list the values are computed in memory and
    then dropped at write time. player_attributes.parquet does not contain them.
    There are three gates in total, and a re-test has to clear all three:
      1. the derivation in step 03            (retained — this test)
      2. config.ATTRIBUTE_MERGE_COLS          (controls what is WRITTEN + merged)
      3. step 8 _build_feature_cols           (controls what a head may USE)
    """
    src = (Path(__file__).parent.parent / "pipeline" / "03_player_attributes.py").read_text()
    for col in ITEM5_CAREER_FEATURES:
        assert col in src, f"step 03 lost the derivation code for {col}"


# ── the wind item that was dropped ─────────────────────────────────────────────

def test_wind_decomposition_features_are_not_claimed():
    """wind_parallel / wind_perpendicular were declared in PLAY_CONTEXT_FEATURES
    but never existed in player_plays (venue_orientation.parquet was never
    built). 04p silently drops absent columns, so the config asserted a surface
    the model never had. Keep them out until 01e can actually produce
    orientation — see the config comment for why the Overpass route is dead."""
    assert "wind_parallel" not in C.PLAY_CONTEXT_FEATURES
    assert "wind_perpendicular" not in C.PLAY_CONTEXT_FEATURES
    assert "wind_speed" in C.PLAY_CONTEXT_FEATURES, "raw wind_speed does exist and stays"
