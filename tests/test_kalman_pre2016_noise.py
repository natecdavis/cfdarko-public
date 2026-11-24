"""T4 — Kalman pre-2016 noise direction invariant.

Locks in the C3 finding: pre-2016 games (snaps_are_real=False) must produce
a *less informative* update than post-2016 games (real snaps), reflected as
a strictly higher posterior variance for the same observation.

Algebra (kalman.py:462-466 + 211):
    proxy_sv = REF_SNAPS / KALMAN_PRE2016_NOISE_MULTIPLIER
    eff_r    = obs_noise * (REF_SNAPS / sv)
    pre-2016 sv  = proxy_sv  → eff_r = obs_noise * MULTIPLIER
    post-2016 sv = REF_SNAPS → eff_r = obs_noise * 1

The test is parametrized over a few stats so a future config tweak that
inverts one stat's direction (e.g. by flipping multiplier semantics) gets
caught immediately.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.models.kalman import run_kalman_all_players


@pytest.mark.parametrize("stat", ["pass_yards", "rush_yards", "tackles"])
def test_pre2016_posterior_var_strictly_higher_than_post2016(stat: str):
    ref_snaps = config.KALMAN_REF_SNAPS
    mult = config.KALMAN_PRE2016_NOISE_MULTIPLIER
    proxy_sv = ref_snaps / mult

    rows = pd.DataFrame({
        "player_id":      ["pre2016", "post2016"],
        "season":         [2014, 2018],
        "game_date":      pd.to_datetime(["2014-09-01", "2018-09-01"]),
        "team":           ["TEAM_A", "TEAM_A"],
        "snaps":          [proxy_sv, ref_snaps],
        "snaps_are_real": [False, True],
        f"adjusted_{stat}": [50.0, 50.0],   # identical observation
    })

    decay = {stat: 1.0}
    out = run_kalman_all_players(
        adjusted_games=rows, decay_betas=decay, stats=[stat],
        school_col="team", age_col=None,
    )
    var_col = f"kalman_var_{stat}"
    pre  = out[out.player_id == "pre2016"][var_col].iloc[0]
    post = out[out.player_id == "post2016"][var_col].iloc[0]
    assert pre > post, (
        f"{stat}: pre-2016 posterior var ({pre:.4f}) must exceed "
        f"post-2016 posterior var ({post:.4f}) — multiplier direction inverted"
    )
    # Sanity: the ratio should be on the order of the multiplier (within a
    # factor of ~3), not a tiny epsilon. Catches the "barely different" failure
    # mode where direction is right but magnitude is a no-op.
    ratio = pre / post
    assert ratio > 1.5, (
        f"{stat}: pre/post posterior variance ratio = {ratio:.3f} is too "
        f"close to 1; expected ratio scaled by ~MULTIPLIER={mult}"
    )


def test_post2016_real_snaps_match_full_observation_strength():
    """A post-2016 game with snaps == REF_SNAPS should produce exactly the
    same posterior as a game where the snap normalization is a no-op (sv/ref=1).
    Guards against accidentally double-applying the proxy snap rewrite to real-snap rows."""
    ref_snaps = config.KALMAN_REF_SNAPS
    rows = pd.DataFrame({
        "player_id":      ["a", "b"],
        "season":         [2018, 2020],
        "game_date":      pd.to_datetime(["2018-09-01", "2020-09-01"]),
        "team":           ["X", "X"],
        "snaps":          [ref_snaps, ref_snaps],
        "snaps_are_real": [True, True],
        "adjusted_pass_yards": [50.0, 50.0],
    })
    out = run_kalman_all_players(
        adjusted_games=rows, decay_betas={"pass_yards": 1.0},
        stats=["pass_yards"], school_col="team", age_col=None,
    )
    a = out[out.player_id == "a"]["kalman_var_pass_yards"].iloc[0]
    b = out[out.player_id == "b"]["kalman_var_pass_yards"].iloc[0]
    assert np.isclose(a, b, atol=1e-9), \
        f"two real-snap games at REF_SNAPS produced different variance: a={a}, b={b}"
