"""
Unit tests for the new step-0 / step-3 / step-6 validators added by the
audit harness. All tests use synthetic data — no I/O against the real
data/ tree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import validation as V


# ── check_step3_pff_features ─────────────────────────────────────────────────

def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_check_step3_pff_features_happy_path(tmp_path, monkeypatch):
    """All declared raw PFF columns present → no errors."""
    from config import PFF_GRADE_FEATURES_BY_GROUP
    raw_names = sorted({
        f for fl in PFF_GRADE_FEATURES_BY_GROUP.values() for f in fl
        if not V._is_pff_engineered_name(f)
    })
    df = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in raw_names})
    df["player_id"] = ["A", "B", "C"]
    _write_parquet(tmp_path / "pff_season_grades.parquet", df)
    monkeypatch.setattr(V, "PROCESSED_DIR", tmp_path)
    results = V.check_step3_pff_features()
    failed = [r for r in results if not r.passed and r.severity == "ERROR"]
    assert not failed, f"unexpected ERROR-level fails: {failed}"


def test_check_step3_pff_features_missing_raw_column(tmp_path, monkeypatch):
    """Drop one raw PFF column → ERROR fires."""
    from config import PFF_GRADE_FEATURES_BY_GROUP
    raw_names = sorted({
        f for fl in PFF_GRADE_FEATURES_BY_GROUP.values() for f in fl
        if not V._is_pff_engineered_name(f)
    })
    raw_names.remove("pff_grades_offense")   # drop one
    df = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in raw_names})
    df["player_id"] = ["A", "B", "C"]
    _write_parquet(tmp_path / "pff_season_grades.parquet", df)
    monkeypatch.setattr(V, "PROCESSED_DIR", tmp_path)
    results = V.check_step3_pff_features()
    fails = [r for r in results
             if not r.passed and r.severity == "ERROR"
             and r.name == "pff_raw_features_present"]
    assert fails, "expected pff_raw_features_present to fire"


def test_is_pff_engineered_name_classifier():
    """The shared classifier should correctly bucket engineered vs raw names."""
    assert V._is_pff_engineered_name("pff_sacks_per_pr_snap")
    assert V._is_pff_engineered_name("pff_man_zone_yprr_delta")
    assert V._is_pff_engineered_name("pff_avoided_tackles_per_attempt")
    assert V._is_pff_engineered_name("pff_true_pass_set_pressures_allowed_rate")
    # Raw counterparts should NOT be classified as engineered:
    assert not V._is_pff_engineered_name("pff_sacks")
    assert not V._is_pff_engineered_name("pff_grades_offense")
    assert not V._is_pff_engineered_name("pff_man_yprr")
    assert not V._is_pff_engineered_name("pff_true_pass_set_pressures_allowed")


# ── check_step3 (tightened age + ATTRIBUTE_MERGE_COLS) ───────────────────────

def test_check_step3_attribute_merge_cols_must_be_present(tmp_path, monkeypatch):
    """ATTRIBUTE_MERGE_COLS missing from player_attributes triggers ERROR."""
    from config import ATTRIBUTE_MERGE_COLS
    # Build a minimal attributes parquet missing one merge col.
    cols = list(ATTRIBUTE_MERGE_COLS)
    cols.remove("career_team_off_ppa_mean")
    df = pd.DataFrame({c: ["x"] * 5_001 for c in cols})
    df["player_id"] = list(range(5_001))
    df["height_in"] = 72.0
    df["weight_lbs"] = 200.0
    df["birth_year"] = 2003
    df["last_college_season_year"] = 2024
    _write_parquet(tmp_path / "player_attributes.parquet", df)
    monkeypatch.setattr(V, "PROCESSED_DIR", tmp_path)
    results = V.check_step3()
    fails = [r for r in results
             if not r.passed and r.name == "attribute_merge_cols_present"]
    assert fails, "expected attribute_merge_cols_present to fire"


def test_check_step3_age_uses_config_constants(tmp_path, monkeypatch):
    """The tightened age check should use MIN/MAX_BRIDGE_DERIVED_AGE."""
    from config import MAX_BRIDGE_DERIVED_AGE, ATTRIBUTE_MERGE_COLS
    rows = 5_005
    df = pd.DataFrame({
        "player_id": list(range(rows)),
        "position_group": ["QB"] * rows,
        "height_in": [72] * rows,
        "weight_lbs": [200] * rows,
        "birth_year": [2003] * rows,
        # 30 rows just over the upper bound — must fire too_old.
        "last_college_season_year":
            [2003 + MAX_BRIDGE_DERIVED_AGE + 1] * 30
            + [2024] * (rows - 30),
    })
    for c in ATTRIBUTE_MERGE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    _write_parquet(tmp_path / "player_attributes.parquet", df)
    monkeypatch.setattr(V, "PROCESSED_DIR", tmp_path)
    results = V.check_step3()
    err = [r for r in results if r.name == "implausible_age_too_old"][0]
    assert not err.passed
    assert "30" in err.message


# ── check_step6 n_seasons ────────────────────────────────────────────────────

def test_check_step6_n_seasons_zero_fires(tmp_path, monkeypatch):
    """A snapshot with n_seasons=0 anywhere should fail the new check."""
    df = pd.DataFrame({
        "player_id": list(range(600)),
        "n_games": [3] * 600,
        "n_seasons": [1] * 599 + [0],  # one zero
        "kalman_mean_pass_yards": [10.0] * 600,
    })
    _write_parquet(tmp_path / "player_skill_snapshot.parquet", df)
    monkeypatch.setattr(V, "PROCESSED_DIR", tmp_path)
    results = V.check_step6()
    fails = [r for r in results if r.name == "n_seasons_nonzero"]
    assert fails and not fails[0].passed
    assert "1 snapshot rows have n_seasons=0" in fails[0].message


def test_check_step6_n_seasons_all_positive(tmp_path, monkeypatch):
    """Snapshot with n_seasons ≥ 1 everywhere should pass."""
    df = pd.DataFrame({
        "player_id": list(range(600)),
        "n_games": [3] * 600,
        "n_seasons": [1] * 600,
        "kalman_mean_pass_yards": [10.0] * 600,
    })
    _write_parquet(tmp_path / "player_skill_snapshot.parquet", df)
    monkeypatch.setattr(V, "PROCESSED_DIR", tmp_path)
    results = V.check_step6()
    fails = [r for r in results
             if not r.passed and r.name == "n_seasons_nonzero"]
    assert not fails
