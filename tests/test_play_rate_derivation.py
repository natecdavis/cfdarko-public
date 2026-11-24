"""
Unit tests for the post-aggregation rate-stat derivation in step 5p.

Verifies that derived rates (completion_pct, catch_pct, fg_pct, ...) are
correctly computed from numerator/denominator residual columns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

import numpy as np
import pandas as pd
import pytest


# Load step 5p by file path (filename starts with digit)
_spec = importlib.util.spec_from_file_location(
    "_step_05p",
    str(Path(__file__).parent.parent / "pipeline" / "05p_play_residual_aggregation.py")
)
step_05p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step_05p)


def test_rate_derivation_completion_pct():
    """play_total_resid_completion / play_n_credited_completion → rate."""
    df = pd.DataFrame({
        "athlete_id": ["A", "B"],
        "game_id":    [1, 1],
        "adjusted_play_total_resid_completion": [10.0, 0.0],
        "play_n_credited_completion":            [4,    1],
    })
    out = step_05p._derive_rate_residuals(df)
    # rate = 10/4 = 2.5 for A; 0/1 = 0.0 for B
    assert "adjusted_play_rate_play_completion_pct" in out.columns
    assert out["adjusted_play_rate_play_completion_pct"].tolist() == [2.5, 0.0]


def test_rate_derivation_handles_zero_denominator():
    df = pd.DataFrame({
        "athlete_id": ["A"],
        "game_id":    [1],
        "adjusted_play_total_resid_completion": [10.0],
        "play_n_credited_completion":            [0],
    })
    out = step_05p._derive_rate_residuals(df)
    # Zero denominator → NaN (not div-by-zero crash)
    assert pd.isna(out["adjusted_play_rate_play_completion_pct"].iloc[0])


def test_rate_derivation_skips_missing_columns():
    """If a rate stat's source columns aren't present, no new col is created."""
    df = pd.DataFrame({"athlete_id": ["A"], "game_id": [1]})
    out = step_05p._derive_rate_residuals(df)
    # No play_total_resid_completion → no completion_pct
    for c in out.columns:
        assert "completion_pct" not in c
