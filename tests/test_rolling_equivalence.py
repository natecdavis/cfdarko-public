"""Equivalence of the vectorized rolling primitive against the original
per-row reference implementation, on adversarial synthetic data: NaN stats,
NaN/zero/absent weights, duplicate dates, same-day target/obs collisions,
entities missing from the panel, NaN target dates, and multi-year spans."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.rolling import (
    compute_rolling_entity_strength,
    _compute_rolling_entity_strength_reference,
)


def _synthetic(seed=0, n_panel=800, n_targets=400, with_weights=True):
    rng = np.random.default_rng(seed)
    ents = [f"team_{i}" for i in range(12)]
    panel = pd.DataFrame({
        "team": rng.choice(ents, n_panel),
        "game_date": pd.Timestamp("2010-01-01")
                     + pd.to_timedelta(rng.integers(0, 4000, n_panel), unit="D"),
        "stat_a": np.where(rng.random(n_panel) < 0.15, np.nan,
                           rng.normal(0, 1, n_panel)),
        "stat_b": np.where(rng.random(n_panel) < 0.4, np.nan,
                           rng.normal(5, 2, n_panel)),
    })
    if with_weights:
        panel["w"] = np.where(rng.random(n_panel) < 0.1, np.nan,
                              rng.uniform(0, 3, n_panel))
        panel.loc[panel.sample(frac=0.05, random_state=seed).index, "w"] = 0.0
    # Duplicate dates within entities
    panel = pd.concat([panel, panel.head(30)], ignore_index=True)

    targets = pd.DataFrame({
        "opponent": rng.choice(ents + ["ghost_team"], n_targets),
        "game_date": pd.Timestamp("2010-01-01")
                     + pd.to_timedelta(rng.integers(-50, 4100, n_targets), unit="D"),
        "game_id": np.arange(n_targets),
    })
    # Some targets share exact dates with panel rows (strictly-prior edge)
    shared = panel.sample(20, random_state=seed)
    targets.loc[targets.index[:20], "game_date"] = shared["game_date"].values
    targets.loc[targets.index[:20], "opponent"] = shared["team"].values
    # NaN target dates
    targets.loc[targets.index[-5:], "game_date"] = pd.NaT
    return panel, targets


def _compare(panel, targets, weight_col):
    kwargs = dict(
        panel_entity_col="team", target_entity_col="opponent",
        date_col="game_date", stat_cols=["stat_a", "stat_b"],
        halflife_days=45.0, weight_col=weight_col,
        out_prefix="roll", keep_target_cols=["game_id"],
    )
    new = compute_rolling_entity_strength(panel, targets, **kwargs)
    ref = _compute_rolling_entity_strength_reference(panel, targets, **kwargs)
    assert list(new.columns) == list(ref.columns)
    assert len(new) == len(ref) == len(targets)
    for col in ("roll_stat_a", "roll_stat_b"):
        n, r = new[col].to_numpy(), ref[col].to_numpy()
        both_nan = np.isnan(n) & np.isnan(r)
        assert (np.isnan(n) == np.isnan(r)).all(), f"NaN mask mismatch in {col}"
        np.testing.assert_allclose(n[~both_nan], r[~both_nan],
                                   rtol=1e-9, atol=1e-12, err_msg=col)
    assert (new["game_id"].to_numpy() == ref["game_id"].to_numpy()).all()


def test_equivalence_with_weights():
    panel, targets = _synthetic(seed=1)
    _compare(panel, targets, weight_col="w")


def test_equivalence_without_weights():
    panel, targets = _synthetic(seed=2, with_weights=False)
    _compare(panel, targets, weight_col=None)


def test_equivalence_short_halflife_long_span():
    """Short halflife over a many-year span: the regime where naive
    prefix-exponential approaches overflow. The recurrence must stay finite
    and match the reference."""
    panel, targets = _synthetic(seed=3)
    kwargs = dict(
        panel_entity_col="team", target_entity_col="opponent",
        date_col="game_date", stat_cols=["stat_a"],
        halflife_days=5.0, weight_col="w",
        out_prefix="roll", keep_target_cols=["game_id"],
    )
    new = compute_rolling_entity_strength(panel, targets, **kwargs)
    ref = _compute_rolling_entity_strength_reference(panel, targets, **kwargs)
    n, r = new["roll_stat_a"].to_numpy(), ref["roll_stat_a"].to_numpy()
    both_val = ~np.isnan(n) & ~np.isnan(r)
    np.testing.assert_allclose(n[both_val], r[both_val], rtol=1e-7, atol=1e-12)
    # NaN-mask may differ ONLY where the reference's absolute weights
    # underflowed to exactly 0 (obs older than ~745 halflives); the
    # recurrence keeps the well-defined ratio there. Allow new-value-where-
    # ref-NaN, never the reverse.
    assert not (np.isnan(n) & ~np.isnan(r)).any()


def test_empty_inputs():
    panel, targets = _synthetic(seed=4)
    assert compute_rolling_entity_strength(
        panel.iloc[0:0], targets, "team", "opponent", "game_date",
        ["stat_a"], 45.0).empty
    assert compute_rolling_entity_strength(
        panel, targets.iloc[0:0], "team", "opponent", "game_date",
        ["stat_a"], 45.0).empty
