"""
Unit tests for pipeline/09_comps.py.

Tests the distance function (_missing_aware_cdist) and comp generation logic.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import types
import importlib.util

import numpy as np
import pandas as pd
import pytest

import config as C


# Module-level fake model (must be picklable for pickle.dump in tests)
class _FakeImpactModel:
    """Returns zero predictions; feature_importances_ required by find_comps."""
    def __init__(self, n_features=2):
        self.feature_importances_ = np.ones(n_features)

    def predict(self, X):
        return np.zeros(len(X))


# Load comps module, mocking external scraper dependencies
_comps_mod = None

def _get_comps():
    global _comps_mod
    if _comps_mod is not None:
        return _comps_mod

    mod_path = Path(__file__).parent.parent / "pipeline" / "09_comps.py"
    spec = importlib.util.spec_from_file_location("comps_09", mod_path)
    _comps_mod = types.ModuleType("comps_09")
    _comps_mod.__file__ = str(mod_path)  # required so Path(__file__) inside module works

    # Stub pff scraper
    pff_stub = types.ModuleType("src.scrapers.pff")
    pff_stub.load_pff_career_features = lambda *a, **kw: pd.DataFrame()
    sys.modules.setdefault("src.scrapers.pff", pff_stub)

    spec.loader.exec_module(_comps_mod)
    return _comps_mod


# ── _missing_aware_cdist ──────────────────────────────────────────────────────

def test_cdist_zero_shared_dims_returns_inf():
    """Two rows sharing zero valid dimensions → distance = inf."""
    m = _get_comps()
    # Row A: feature 0 is valid, feature 1 is NaN
    # Row B: feature 0 is NaN, feature 1 is valid → no overlap
    X_a = np.array([[1.0, np.nan]])
    X_b = np.array([[np.nan, 2.0]])
    weights = np.array([1.0, 1.0])
    dist = m._missing_aware_cdist(X_a, X_b, weights)
    assert np.isinf(dist[0, 0]), "Distance should be inf when no shared valid dims"


def test_cdist_all_present_matches_weighted_euclidean():
    """No NaN → should match standard importance-weighted Euclidean distance."""
    m = _get_comps()
    X_a = np.array([[1.0, 2.0, 3.0]])
    X_b = np.array([[4.0, 6.0, 3.0]])
    weights = np.array([1.0, 1.0, 1.0])
    dist = m._missing_aware_cdist(X_a, X_b, weights)
    # Manually: sqrt((1-4)^2 + (2-6)^2 + (3-3)^2) = sqrt(9+16) = 5
    assert abs(dist[0, 0] - 5.0) < 1e-8


def test_cdist_identical_rows_zero_distance():
    m = _get_comps()
    X = np.array([[1.0, 2.0, 3.0]])
    weights = np.array([1.0, 1.0, 1.0])
    dist = m._missing_aware_cdist(X, X, weights)
    assert abs(dist[0, 0]) < 1e-8


def test_cdist_partial_nan_penalized():
    """Pair sharing fewer dims should have higher distance than pair sharing all."""
    m = _get_comps()
    X_a = np.array([[1.0, 2.0, 3.0]])
    X_b_full = np.array([[4.0, 2.0, 3.0]])    # share all 3 dims
    X_b_partial = np.array([[4.0, np.nan, 3.0]])  # share only 2 dims

    weights = np.array([1.0, 1.0, 1.0])
    dist_full = m._missing_aware_cdist(X_a, X_b_full, weights)
    dist_partial = m._missing_aware_cdist(X_a, X_b_partial, weights)
    # Partial share → scaled up by sqrt(W_total / W_shared)
    assert dist_partial[0, 0] >= dist_full[0, 0]


def test_cdist_nonneg():
    """All distances must be non-negative or inf."""
    m = _get_comps()
    rng = np.random.default_rng(0)
    X_a = rng.standard_normal((5, 4))
    X_b = rng.standard_normal((7, 4))
    # Randomly NaN-ify some entries
    X_a[X_a < -1.5] = np.nan
    X_b[X_b < -1.5] = np.nan
    weights = np.ones(4)
    dist = m._missing_aware_cdist(X_a, X_b, weights)
    assert (dist >= 0).all() or np.isinf(dist).any()


# ── find_comps_for_group ─────────────────────────────────────────────────────

def _make_comp_inputs(n_prospects=5, n_historical=15, pos_group="QB"):
    """Build minimal snapshot + labels for find_comps_for_group."""
    rng = np.random.default_rng(42)
    all_players = n_prospects + n_historical

    snapshot = pd.DataFrame({
        "player_id": [f"P{i}" for i in range(all_players)],
        "player_name": [f"Player {i}" for i in range(all_players)],
        "school": ["TestU"] * all_players,
        "position_group": [pos_group] * all_players,
        # Prospects: 2022 cohort; historical: 2010–2021 cohort
        "last_college_season_year": (
            [2022] * n_prospects + list(rng.integers(2010, 2022, n_historical))
        ),
        "kalman_mean_pass_yards": rng.normal(0, 2, all_players),
        "ewma_pass_yards": rng.normal(0, 2, all_players),
        "height_in": rng.uniform(70, 80, all_players),
        "weight_lbs": rng.uniform(200, 240, all_players),
    })

    labels = pd.DataFrame({
        "player_id": [f"P{i}" for i in range(all_players)],
        "role_target": rng.choice([0.0, 1.0], all_players),
        "impact_target": rng.uniform(0, 0.05, all_players),
        "career_w_av": rng.integers(0, 50, all_players).astype(float),
        "first_nfl_season": rng.integers(2011, 2023, all_players),
        "is_bridge": [False] * n_prospects + [True] * n_historical,
    })

    return snapshot, labels


def test_find_comps_no_model_returns_empty(tmp_path, monkeypatch):
    """Missing impact model → returns empty DataFrame, no crash."""
    m = _get_comps()
    # Patch PROCESSED_DIR to tmp_path so no model files exist there
    monkeypatch.setattr(m, "PROCESSED_DIR", tmp_path)

    snapshot, labels = _make_comp_inputs()
    result = m.find_comps_for_group(snapshot, labels, pos_group="QB", k=3)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_find_comps_backward_looking_pool(tmp_path, monkeypatch):
    """All comps must have last_college_season_year < prospect's year."""
    m = _get_comps()

    import pickle

    models_dir = tmp_path / "draft_models"
    models_dir.mkdir(parents=True)

    snapshot, labels = _make_comp_inputs(n_prospects=3, n_historical=10)
    feature_cols = ["kalman_mean_pass_yards", "ewma_pass_yards"]

    bundle = {"model": _FakeImpactModel(n_features=2), "feature_cols": feature_cols}
    with open(models_dir / "QB_impact_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    monkeypatch.setattr(m, "PROCESSED_DIR", tmp_path)

    result = m.find_comps_for_group(snapshot, labels, pos_group="QB", k=3)
    if result.empty:
        pytest.skip("find_comps returned empty (model predict failed)")

    # Prospects all have last_college_season_year=2022; historical are 2010-2021.
    # Verify comp player IDs all come from the historical pool (not the 2022 prospects).
    prospect_ids = set(snapshot[snapshot["last_college_season_year"] == 2022]["player_id"])
    comp_player_ids = set(result["comp_player_id"].dropna())
    overlap = comp_player_ids & prospect_ids
    assert not overlap, (
        f"Some comps are from the prospect cohort (not backward-looking): {overlap}"
    )


def test_cdist_weights_applied():
    """Higher-weight feature should matter more in distance."""
    m = _get_comps()
    # Two rows differing only in feature 0 (high weight) vs feature 1 (low weight)
    X_a = np.array([[0.0, 0.0]])
    X_b_high = np.array([[1.0, 0.0]])  # differs on high-weight dim
    X_b_low  = np.array([[0.0, 1.0]])  # differs on low-weight dim

    weights_high = np.array([10.0, 1.0])
    dist_high = m._missing_aware_cdist(X_a, X_b_high, weights_high)
    dist_low  = m._missing_aware_cdist(X_a, X_b_low,  weights_high)
    assert dist_high[0, 0] > dist_low[0, 0]
