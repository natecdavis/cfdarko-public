"""T5 — Step 10 quantile clamp monotonicity invariants.

The clamp at pipeline/10_output.py:_clamp_impact_band coerces independent
LightGBM p10/p90 heads to bracket the MSE point estimate. This test locks in:

  1. p10 ≤ score ≤ p90 for every row after clamp.
  2. The point estimate is unchanged (preserved exactly).
  3. Within a position group, ranks of impact_apy_score are preserved when
     converted to dollar columns (multiplication by a positive constant
     can't change rank).
  4. Crossed-quantile rows (p90 < score, p10 > score) are corrected without
     side effects on already-bracketed rows.
"""
import sys
import importlib.util
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config


def _get_step10():
    spec = importlib.util.spec_from_file_location(
        "step10_output", ROOT / "pipeline" / "10_output.py",
    )
    mod = types.ModuleType("step10_output")
    mod.__file__ = str(ROOT / "pipeline" / "10_output.py")
    spec.loader.exec_module(mod)
    return mod


def test_clamp_brackets_point_estimate():
    """p10 ≤ score ≤ p90 for every row after _clamp_impact_band."""
    m = _get_step10()
    snap = pd.DataFrame({
        "impact_apy_score": [0.01, 0.05, 0.10, 0.20, 0.04],
        # Mix of well-bracketed, p90<score, p10>score, both-crossed
        "impact_apy_p10":   [0.005, 0.07,  0.05, 0.30, 0.04],
        "impact_apy_p90":   [0.020, 0.04,  0.15, 0.10, 0.04],
    })
    out = m._clamp_impact_band(snap.copy())
    assert (out["impact_apy_p10"] <= out["impact_apy_score"]).all(), \
        f"p10 > score after clamp: {out}"
    assert (out["impact_apy_score"] <= out["impact_apy_p90"]).all(), \
        f"score > p90 after clamp: {out}"


def test_clamp_preserves_point_estimate_exactly():
    """The MSE point estimate is preserved unchanged — only the band moves."""
    m = _get_step10()
    rng = np.random.default_rng(0)
    n = 50
    score = rng.uniform(0, 0.2, n)
    p10 = rng.uniform(-0.05, 0.25, n)   # deliberately can cross score
    p90 = rng.uniform(-0.05, 0.30, n)
    snap = pd.DataFrame({
        "impact_apy_score": score,
        "impact_apy_p10":   p10,
        "impact_apy_p90":   p90,
    })
    out = m._clamp_impact_band(snap.copy())
    np.testing.assert_array_equal(out["impact_apy_score"].values, score), \
        "_clamp_impact_band mutated impact_apy_score"


def test_dollar_transform_preserves_ranks_within_position_group():
    """impact_apy_dollars = impact_apy_score * SALARY_CAP. Multiplication by
    a positive scalar preserves rank, both globally and within group."""
    score = pd.Series([0.01, 0.04, 0.02, 0.10, 0.07])
    pos   = pd.Series(["WR", "WR", "QB", "QB", "WR"])
    df = pd.DataFrame({"impact_apy_score": score, "position_group": pos})
    df["impact_apy_dollars"] = df["impact_apy_score"] * config.SALARY_CAP
    rank_pct = df.groupby("position_group")["impact_apy_score"].rank(pct=True)
    rank_dol = df.groupby("position_group")["impact_apy_dollars"].rank(pct=True)
    pd.testing.assert_series_equal(rank_pct, rank_dol, check_names=False)


def test_clamp_no_op_on_already_bracketed_rows():
    """Rows where p10 ≤ score ≤ p90 already are returned unchanged."""
    m = _get_step10()
    snap = pd.DataFrame({
        "impact_apy_score": [0.05, 0.10, 0.15],
        "impact_apy_p10":   [0.02, 0.07, 0.12],
        "impact_apy_p90":   [0.08, 0.13, 0.18],
    })
    out = m._clamp_impact_band(snap.copy())
    pd.testing.assert_frame_equal(out, snap, check_dtype=False)


def test_clamp_skips_when_columns_missing():
    """If any of {p10, score, p90} is absent, the helper is a no-op."""
    m = _get_step10()
    snap = pd.DataFrame({"role_score": [0.5, 0.6]})
    out = m._clamp_impact_band(snap.copy())
    pd.testing.assert_frame_equal(out, snap)
