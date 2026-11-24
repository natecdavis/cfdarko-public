"""
Unit tests for pipeline/10_output.py helper functions.

Tests score bounds, percentile rank computation, best comp selection,
and output ordering.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import types
import importlib.util
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

import config as C


# Module-level fake models (must be at module level to be picklable)
class _FakeRoleModel:
    def predict(self, X):
        return np.array([-0.5] * len(X))


class _FakeImpactModel:
    def predict(self, X):
        return np.array([-1.0] * len(X))


# Load output module, mocking external scraper imports
_out_mod = None

def _get_out():
    global _out_mod
    if _out_mod is not None:
        return _out_mod

    mod_path = Path(__file__).parent.parent / "pipeline" / "10_output.py"
    spec = importlib.util.spec_from_file_location("output_10", mod_path)
    _out_mod = types.ModuleType("output_10")
    _out_mod.__file__ = str(mod_path)  # required so Path(__file__) inside module works

    pff_stub = types.ModuleType("src.scrapers.pff")
    pff_stub.load_pff_career_features = lambda *a, **kw: pd.DataFrame()
    sys.modules.setdefault("src.scrapers.pff", pff_stub)

    spec.loader.exec_module(_out_mod)
    return _out_mod


# ── _compute_role_impact_scores ───────────────────────────────────────────────

def _make_snapshot(n=20, pos_group="QB"):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "position_group": [pos_group] * n,
        "kalman_mean_pass_yards": rng.normal(0, 2, n),
        "ewma_pass_yards": rng.normal(0, 2, n),
        "height_in": rng.uniform(70, 78, n),
        "weight_lbs": rng.uniform(190, 240, n),
        "conference_tier": rng.choice([1, 2, 3], n),
    })


def test_compute_role_impact_missing_models_gives_nan(tmp_path, monkeypatch):
    """No model pickle files → impact_score stays NaN, no crash."""
    m = _get_out()
    # Point to a tmp dir with no pickle files
    monkeypatch.setattr(m, "PROCESSED_DIR", tmp_path)

    snapshot = _make_snapshot()
    result = m._compute_role_impact_scores(snapshot)

    assert "impact_score" in result.columns
    assert result["impact_score"].isna().all()


def test_stale_role_model_is_ignored(tmp_path, monkeypatch):
    """The role heads were dropped 2026-10-31: a leftover role pickle on disk
    must not produce a role_score column."""
    m = _get_out()

    models_dir = tmp_path / "draft_models"
    models_dir.mkdir(parents=True)

    feature_cols = ["kalman_mean_pass_yards", "ewma_pass_yards"]
    bundle = {"model": _FakeRoleModel(), "feature_cols": feature_cols}
    with open(models_dir / "QB_role_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    monkeypatch.setattr(m, "PROCESSED_DIR", tmp_path)

    result = m._compute_role_impact_scores(_make_snapshot())
    assert not any("role_score" in c for c in result.columns)


def test_compute_impact_score_nonneg(tmp_path, monkeypatch):
    """impact_score must be ≥ 0 after clipping."""
    m = _get_out()

    models_dir = tmp_path / "draft_models"
    models_dir.mkdir(parents=True)

    feature_cols = ["kalman_mean_pass_yards"]
    bundle = {"model": _FakeImpactModel(), "feature_cols": feature_cols}
    with open(models_dir / "QB_impact_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    monkeypatch.setattr(m, "PROCESSED_DIR", tmp_path)

    snapshot = _make_snapshot()
    result = m._compute_role_impact_scores(snapshot)

    non_nan = result["impact_score"].dropna()
    assert (non_nan >= 0).all(), f"Negative impact_score found: {non_nan[non_nan < 0]}"


# ── _add_percentile_ranks ─────────────────────────────────────────────────────

def _make_scored_snapshot(n=30):
    rng = np.random.default_rng(1)
    groups = (["QB"] * 10 + ["WR"] * 10 + ["RB"] * 10)
    return pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "position_group": groups,
        "role_score": rng.uniform(0, 1, n),
        "impact_score": rng.uniform(0, 0.05, n),
        "kalman_mean_pass_yards": rng.normal(0, 3, n),
        "kalman_mean_rec_yards": rng.normal(0, 2, n),
    })


def test_percentile_ranks_in_01():
    m = _get_out()
    df = _make_scored_snapshot()
    result = m._add_percentile_ranks(df)
    pct_cols = [c for c in result.columns if c.startswith("pct_")]
    assert len(pct_cols) > 0, "No pct_ columns generated"
    for col in pct_cols:
        vals = result[col].dropna()
        assert (vals >= 0).all() and (vals <= 1).all(), (
            f"{col} has values out of [0,1]: min={vals.min()}, max={vals.max()}"
        )


def test_percentile_ranks_per_position_group():
    """Each position group should have its best player at percentile = 1.0."""
    m = _get_out()
    df = _make_scored_snapshot()
    result = m._add_percentile_ranks(df)

    for pos_group in ["QB", "WR", "RB"]:
        grp = result[result["position_group"] == pos_group]
        if "pct_role_score" in result.columns:
            max_pct = grp["pct_role_score"].max()
            assert abs(max_pct - 1.0) < 1e-6, (
                f"{pos_group}: max pct_role_score={max_pct}, expected ~1.0"
            )


def test_percentile_ranks_no_position_group():
    """Missing position_group → returns df unchanged, no crash."""
    m = _get_out()
    df = pd.DataFrame({
        "player_id": ["P1", "P2"],
        "role_score": [0.5, 0.8],
    })
    result = m._add_percentile_ranks(df)
    assert "pct_role_score" not in result.columns


# ── _add_best_comp ────────────────────────────────────────────────────────────

def _make_comps_df(n_players=5, include_bridge=True):
    """Build a comps DataFrame with both bridge and all tiers."""
    rows = []
    for pid in [f"P{i}" for i in range(n_players)]:
        for tier in (["all", "bridge"] if include_bridge else ["all"]):
            for rank in range(1, 4):
                rows.append({
                    "player_id": pid,
                    "comp_tier": tier,
                    "comp_rank": rank,
                    "comp_player_name": f"Comp{tier}{rank}",
                    "comp_school": "OldU",
                    "comp_role_target": float(rank % 2),
                    "comp_impact_target": rank * 0.01,
                    "comp_career_w_av": rank * 10.0,
                    "comp_first_nfl_season": 2010 + rank,
                })
    return pd.DataFrame(rows)


def test_add_best_comp_prefers_bridge_tier():
    """When bridge-tier comp exists, best comp should come from bridge tier."""
    m = _get_out()
    df = pd.DataFrame({"player_id": [f"P{i}" for i in range(5)]})
    comps = _make_comps_df(n_players=5, include_bridge=True)
    result = m._add_best_comp(df, comps)

    assert "best_comp_name" in result.columns
    # Bridge-tier rank-1 comp name is "Compbridge1"
    for _, row in result.iterrows():
        if pd.notna(row.get("best_comp_name")):
            assert "bridge" in str(row["best_comp_name"]).lower(), (
                f"Expected bridge-tier comp, got {row['best_comp_name']!r}"
            )


def test_add_best_comp_falls_back_to_all_tier():
    """No bridge-tier comps → falls back to all-tier rank-1."""
    m = _get_out()
    df = pd.DataFrame({"player_id": [f"P{i}" for i in range(5)]})
    comps = _make_comps_df(n_players=5, include_bridge=False)
    result = m._add_best_comp(df, comps)

    for _, row in result.iterrows():
        if pd.notna(row.get("best_comp_name")):
            assert "all" in str(row["best_comp_name"]).lower()


def test_add_best_comp_empty_comps():
    """Empty comps DataFrame → best_comp columns set to None, no crash."""
    m = _get_out()
    df = pd.DataFrame({"player_id": ["P1", "P2"]})
    result = m._add_best_comp(df, pd.DataFrame())
    assert "best_comp_name" in result.columns
    assert result["best_comp_name"].isna().all()


# ── Output ordering ───────────────────────────────────────────────────────────

def test_output_sorted_by_impact_desc():
    """Verify that sorting by impact_score descending works (NaN at bottom)."""
    rng = np.random.default_rng(2)
    n = 20
    df = pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "impact_score": np.where(rng.random(n) > 0.2, rng.uniform(0, 0.05, n), np.nan),
    })
    sorted_df = df.sort_values("impact_score", ascending=False, na_position="last")

    non_nan = sorted_df["impact_score"].dropna()
    # Monotone non-increasing
    for i in range(len(non_nan) - 1):
        assert non_nan.iloc[i] >= non_nan.iloc[i + 1]

    # NaN values all at the end
    nan_positions = sorted_df["impact_score"].isna()
    if nan_positions.any():
        last_non_nan = (~nan_positions).cumsum().argmax()
        first_nan = nan_positions.idxmax()
        assert sorted_df.index.get_loc(first_nan) > sorted_df.index.get_loc(
            sorted_df.index[last_non_nan]
        )


def test_no_duplicate_player_id_in_output():
    """Final output DataFrame should have unique player_id."""
    m = _get_out()
    rng = np.random.default_rng(3)
    n = 30
    df = pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "position_group": rng.choice(["QB", "WR", "RB"], n),
        "role_score": rng.uniform(0, 1, n),
        "impact_score": rng.uniform(0, 0.05, n),
    })
    # _add_percentile_ranks shouldn't introduce duplicates
    result = m._add_percentile_ranks(df)
    assert result["player_id"].nunique() == len(result), (
        "Duplicate player_id in output after _add_percentile_ranks"
    )
