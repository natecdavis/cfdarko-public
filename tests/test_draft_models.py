"""
Unit tests for pipeline/08_draft_models.py helper functions.

Tests derived feature computation, feature column building, score bounds,
and training set temporal integrity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import types
import importlib.util
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

import config as C

# Load the draft_models module, mocking external scraper imports
_dm_mod = None

def _get_dm():
    global _dm_mod
    if _dm_mod is not None:
        return _dm_mod

    mod_path = Path(__file__).parent.parent / "pipeline" / "08_draft_models.py"
    spec = importlib.util.spec_from_file_location("draft_models_08", mod_path)
    _dm_mod = types.ModuleType("draft_models_08")
    _dm_mod.__file__ = str(mod_path)  # required so Path(__file__) inside module works

    # Stub pff scraper
    pff_stub = types.ModuleType("src.scrapers.pff")
    pff_stub.load_pff_career_features = lambda *a, **kw: pd.DataFrame()
    sys.modules.setdefault("src.scrapers.pff", pff_stub)

    spec.loader.exec_module(_dm_mod)
    return _dm_mod


# ── _add_derived_features ──────────────────────────────────────────────────────

def test_derived_features_dl_creates_pass_rush_composite():
    """_add_derived_features must create pass_rush_composite for DL."""
    m = _get_dm()
    df = pd.DataFrame({
        "player_id": ["P1", "P2"],
        "kalman_mean_sacks": [1.0, 0.5],
        "kalman_mean_tfl": [2.0, 1.0],
        "kalman_mean_qb_hurries": [3.0, 1.0],
    })
    result = m._add_derived_features(df, "DL")
    assert "pass_rush_composite" in result.columns
    # P1: 1*2 + 2*1 + 3*0.5 = 2 + 2 + 1.5 = 5.5
    assert abs(result.loc[0, "pass_rush_composite"] - 5.5) < 1e-10


def test_derived_features_lb_creates_pass_rush_composite():
    m = _get_dm()
    df = pd.DataFrame({
        "kalman_mean_sacks": [0.0],
        "kalman_mean_tfl": [1.0],
        "kalman_mean_qb_hurries": [2.0],
    })
    result = m._add_derived_features(df, "LB")
    assert "pass_rush_composite" in result.columns


def test_derived_features_qb_no_composite():
    """QB group should NOT get pass_rush_composite."""
    m = _get_dm()
    df = pd.DataFrame({
        "kalman_mean_pass_yards": [100.0],
        "kalman_mean_rush_yards": [20.0],
    })
    result = m._add_derived_features(df, "QB")
    assert "pass_rush_composite" not in result.columns


def test_derived_features_missing_cols_fills_zero():
    """Missing kalman_mean_* columns → filled with 0, not NaN."""
    m = _get_dm()
    df = pd.DataFrame({"player_id": ["P1"]})
    result = m._add_derived_features(df, "DL")
    assert "pass_rush_composite" in result.columns
    assert result["pass_rush_composite"].iloc[0] == 0.0


def test_derived_features_does_not_mutate_input():
    """The function must not modify the original DataFrame."""
    m = _get_dm()
    df = pd.DataFrame({
        "kalman_mean_sacks": [1.0],
        "kalman_mean_tfl": [1.0],
        "kalman_mean_qb_hurries": [1.0],
    })
    original_cols = list(df.columns)
    m._add_derived_features(df, "DL")
    assert list(df.columns) == original_cols


# ── _build_feature_cols ────────────────────────────────────────────────────────

def test_build_feature_cols_nonempty_for_qb():
    """QB snapshot with kalman_mean columns → non-empty feature list."""
    m = _get_dm()
    df = pd.DataFrame({
        "player_id": ["P1"],
        "kalman_mean_pass_yards": [5.0],
        "ewma_pass_yards": [4.0],
        "kalman_var_pass_yards": [1.0],
        "height_in": [74.0],
        "weight_lbs": [215.0],
        "conference_tier": [1],
        "n_games": [30],
    })
    cols = m._build_feature_cols(df, "QB")
    assert len(cols) > 0, "QB feature cols should not be empty"


def test_build_feature_cols_only_present_columns():
    """Feature list must only contain columns that exist in the DataFrame."""
    m = _get_dm()
    df = pd.DataFrame({
        "player_id": ["P1"],
        "kalman_mean_pass_yards": [5.0],
        # No other columns
    })
    cols = m._build_feature_cols(df, "QB")
    for col in cols:
        assert col in df.columns, f"Feature col {col!r} not in DataFrame"


def test_build_feature_cols_for_dl_includes_pass_rush():
    """For DL, pass_rush_composite should be included if present."""
    m = _get_dm()
    df = pd.DataFrame({
        "player_id": ["P1"],
        "kalman_mean_sacks": [1.0],
        "kalman_mean_tfl": [0.5],
        "kalman_mean_qb_hurries": [0.3],
        "pass_rush_composite": [2.8],
    })
    cols = m._build_feature_cols(df, "DL")
    assert "pass_rush_composite" in cols


# ── OL sub-position dummies ───────────────────────────────────────────────────

def test_derived_features_ol_tackle_dummy():
    """OT position → is_ol_tackle=1, guard/center=0."""
    m = _get_dm()
    df = pd.DataFrame({"position": ["OT", "OG", "C"]})
    result = m._add_derived_features(df, "OL")
    assert result.loc[0, "is_ol_tackle"] == 1.0
    assert result.loc[0, "is_ol_guard"]  == 0.0
    assert result.loc[0, "is_ol_center"] == 0.0


def test_derived_features_ol_guard_dummy():
    """OG/IOL/G positions → is_ol_guard=1."""
    m = _get_dm()
    for pos in ["OG", "IOL", "G"]:
        df = pd.DataFrame({"position": [pos]})
        result = m._add_derived_features(df, "OL")
        assert result.loc[0, "is_ol_guard"] == 1.0, f"Expected is_ol_guard=1 for position={pos!r}"
        assert result.loc[0, "is_ol_tackle"] == 0.0
        assert result.loc[0, "is_ol_center"] == 0.0


def test_derived_features_ol_no_position_no_crash():
    """OL df without position column → no crash, no is_ol_* dummies."""
    m = _get_dm()
    df = pd.DataFrame({"player_id": ["P1", "P2"]})
    result = m._add_derived_features(df, "OL")
    assert "is_ol_tackle" not in result.columns


# ── Snap-role composition features ────────────────────────────────────────────

def test_derived_features_dl_snap_pct_in_01():
    """DL pff_pass_rush_snap_pct must be in [0, 1]."""
    m = _get_dm()
    df = pd.DataFrame({
        "pff_snap_counts_defense_mean":    [200.0, 100.0, 0.0],
        "pff_snap_counts_pass_rush_mean":  [ 80.0,  60.0, 50.0],
    })
    result = m._add_derived_features(df, "DL")
    assert "pff_pass_rush_snap_pct" in result.columns
    vals = result["pff_pass_rush_snap_pct"].dropna()
    assert (vals >= 0).all() and (vals <= 1).all()


def test_derived_features_lb_has_coverage_pct_dl_does_not():
    """LB gets pff_coverage_snap_pct; DL does not."""
    m = _get_dm()
    df = pd.DataFrame({
        "pff_snap_counts_defense_mean":   [200.0],
        "pff_snap_counts_pass_rush_mean": [ 40.0],
        "pff_snap_counts_coverage_mean":  [120.0],
    })
    lb_result = m._add_derived_features(df.copy(), "LB")
    dl_result = m._add_derived_features(df.copy(), "DL")
    assert "pff_coverage_snap_pct" in lb_result.columns
    assert "pff_coverage_snap_pct" not in dl_result.columns


# ── Temporal integrity guards ──────────────────────────────────────────────────

def test_final_train_cutoff_is_two_years_before_end():
    """_FINAL_TRAIN_CUTOFF must be END_YEAR - 2."""
    m = _get_dm()
    assert m._FINAL_TRAIN_CUTOFF == C.END_YEAR - 2


def test_expanding_window_folds_no_leakage():
    """Validation windows must all be after their training end year."""
    m = _get_dm()
    folds = m._expanding_window_folds()
    for train_end, val_start, val_end in folds:
        assert val_start > train_end, (
            f"Val start {val_start} not after train end {train_end}"
        )
        assert val_end >= val_start


def test_expanding_window_folds_nonempty():
    """There must be at least one valid fold given our date range."""
    m = _get_dm()
    folds = m._expanding_window_folds()
    assert len(folds) > 0, "No valid expanding-window CV folds found"


def test_expanding_window_folds_sorted():
    """Folds should be in chronological order."""
    m = _get_dm()
    folds = m._expanding_window_folds()
    for i in range(len(folds) - 1):
        assert folds[i][0] <= folds[i+1][0], "Folds not in chronological order"


# ── Hyperparameter tuning helpers ──────────────────────────────────────────────

def test_load_tuned_params_returns_none_when_missing(tmp_path, monkeypatch):
    """No tuned_params.json → returns None."""
    m = _get_dm()
    monkeypatch.setattr(m, "TUNED_PARAMS_PATH", tmp_path / "missing.json")
    assert m._load_tuned_params("QB", "role") is None


def test_load_tuned_params_returns_dict_when_present(tmp_path, monkeypatch):
    """tuned_params.json with QB/role → returns the dict for that key."""
    import json as _json
    m = _get_dm()
    p = tmp_path / "tuned_params.json"
    p.write_text(_json.dumps({"QB/role": {"learning_rate": 0.07, "max_depth": 5}}))
    monkeypatch.setattr(m, "TUNED_PARAMS_PATH", p)
    out = m._load_tuned_params("QB", "role")
    assert out == {"learning_rate": 0.07, "max_depth": 5}
    # Missing key → None.
    assert m._load_tuned_params("DB", "role") is None


def test_sample_hyperparams_returns_valid_lgbm_params():
    """Sampled dict has all 8 LGBM keys with values in their declared ranges."""
    m = _get_dm()
    rng = np.random.default_rng(0)
    p = m._sample_hyperparams(rng)
    assert set(p.keys()) == {
        "learning_rate", "max_depth", "num_leaves", "min_child_samples",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
    }
    assert 0.01 <= p["learning_rate"] <= 0.1
    assert 3 <= p["max_depth"] <= 8
    assert 8 <= p["num_leaves"] <= 63
    assert 5 <= p["min_child_samples"] <= 49
    assert 0.6 <= p["subsample"] <= 1.0
    assert 0.6 <= p["colsample_bytree"] <= 1.0
    assert 0.001 <= p["reg_alpha"] <= 1.0
    assert 0.001 <= p["reg_lambda"] <= 10.0


def test_sample_hyperparams_deterministic_given_seed():
    """Same seed → same sample. Cheap reproducibility guard."""
    m = _get_dm()
    a = m._sample_hyperparams(np.random.default_rng(123))
    b = m._sample_hyperparams(np.random.default_rng(123))
    assert a == b


def _make_synthetic_cv_df(n_per_year: int = 80, seed: int = 0) -> pd.DataFrame:
    """Build a small cohort frame spanning enough years for at least one CV fold."""
    rng = np.random.default_rng(seed)
    years = list(range(C.START_YEAR, C.END_YEAR - 1))
    rows = []
    for yr in years:
        for _ in range(n_per_year):
            x = float(rng.normal())
            target = 1.0 if x + rng.normal(0, 0.5) > 0 else 0.0
            rows.append({"x1": x, "x2": float(rng.normal()),
                         "last_college_season_year": yr,
                         "target": target})
    return pd.DataFrame(rows)


def test_evaluate_params_auc_returns_negative_score():
    """AUC branch returns -AUC so the tuner's lower-is-better logic still works."""
    m = _get_dm()
    df = _make_synthetic_cv_df()
    folds = m._expanding_window_folds()
    fd = m._build_fold_data(df, ["x1", "x2"], "target", folds)
    score, _ = m._evaluate_params(
        {"n_estimators": 50}, fd, ["x1", "x2"], metric="auc"
    )
    assert score is not None
    # -AUC: should be in [-1, -0.5] for any reasonable signal
    assert -1.0 <= score <= 0.0


def test_evaluate_params_mae_returns_positive_score():
    """MAE branch returns mean abs error (≥ 0)."""
    m = _get_dm()
    df = _make_synthetic_cv_df()
    folds = m._expanding_window_folds()
    fd = m._build_fold_data(df, ["x1", "x2"], "target", folds)
    score, _ = m._evaluate_params(
        {"n_estimators": 50}, fd, ["x1", "x2"], metric="mae"
    )
    assert score is not None
    assert score >= 0.0


def test_evaluate_params_unknown_metric_raises():
    m = _get_dm()
    df = _make_synthetic_cv_df()
    with pytest.raises(ValueError):
        m._evaluate_params(
            {}, m._build_fold_data(df, ["x1"], "target",
                                   m._expanding_window_folds()),
            ["x1"], metric="bogus")


def test_evaluate_params_logloss_returns_positive_score():
    """BCE branch returns a non-negative finite float."""
    m = _get_dm()
    df = _make_synthetic_cv_df()
    folds = m._expanding_window_folds()
    fd = m._build_fold_data(df, ["x1", "x2"], "target", folds)
    score, _ = m._evaluate_params(
        {"n_estimators": 50}, fd, ["x1", "x2"], metric="logloss"
    )
    assert score is not None
    assert np.isfinite(score)
    assert score >= 0.0


def test_evaluate_params_logloss_lower_is_better():
    """A clearly better-fit model should produce a lower log loss than an underfit one."""
    m = _get_dm()
    df = _make_synthetic_cv_df()
    folds = m._expanding_window_folds()
    fd = m._build_fold_data(df, ["x1", "x2"], "target", folds)
    good, _ = m._evaluate_params(
        {"n_estimators": 200, "learning_rate": 0.05},
        fd, ["x1", "x2"], metric="logloss",
    )
    bad, _ = m._evaluate_params(
        {"n_estimators": 1, "learning_rate": 0.001},
        fd, ["x1", "x2"], metric="logloss",
    )
    assert good is not None and bad is not None
    assert good < bad, f"underfit logloss {bad:.4f} should exceed well-fit logloss {good:.4f}"


# ── Multi-seed Optuna ──────────────────────────────────────────────────────────

def test_tune_hyperparams_multi_seed_picks_lowest_cv():
    """With n_seeds > 1, tune_hyperparams must run N independent Optuna studies
    and return the params of the seed with the lowest reported CV score.

    Stub out the Optuna dependency with a fake whose best_trial.value depends
    deterministically on the sampler's seed, so we can verify which seed wins."""
    m = _get_dm()

    # Capture per-seed outcomes that we'll feed into the fake Optuna study.
    # Map sampler seed → (best_value, best_params). The function offsets seeds
    # by 1009 between iterations, so adjacent seeds will land in this map.
    base = 42 + int.from_bytes(__import__("hashlib").sha256(b"QB/role").digest()[:4], "big")
    per_seed_outcome = {
        base + 0 * 1009: (0.20, {"learning_rate": 0.10}),  # seed 0 — worst
        base + 1 * 1009: (0.05, {"learning_rate": 0.05}),  # seed 1 — BEST
        base + 2 * 1009: (0.15, {"learning_rate": 0.07}),  # seed 2 — middle
    }

    class _FakeSampler:
        def __init__(self, seed):
            self.seed = seed

    class _FakeTrial:
        def __init__(self, value, params):
            self.value = value
            self.params = params

    class _FakeStudy:
        def __init__(self, sampler):
            v, p = per_seed_outcome[sampler.seed]
            self.best_trial = _FakeTrial(v, p)
        def optimize(self, *args, **kwargs):
            pass

    fake_optuna = types.SimpleNamespace(
        samplers=types.SimpleNamespace(TPESampler=_FakeSampler),
        create_study=lambda direction, sampler: _FakeStudy(sampler),
        logging=types.SimpleNamespace(
            set_verbosity=lambda *a, **k: None,
            WARNING=30,
        ),
        TrialPruned=type("TrialPruned", (Exception,), {}),
    )

    # tune_hyperparams imports optuna inside the function body, so monkeypatch
    # sys.modules to intercept the import.
    with mock.patch.dict(sys.modules, {"optuna": fake_optuna}):
        # Provide a synthetic snapshot + labels with enough rows to pass the
        # n>=50 guard. Real LightGBM never runs because the fake study skips
        # _evaluate_params entirely (optimize is a no-op).
        snapshot = pd.DataFrame({
            "player_id":                  [f"P{i:03d}" for i in range(80)],
            "position_group":             ["QB"] * 80,
            "last_college_season_year":   sorted(list(range(2006, 2022)) * 5),
            "x1":                          np.linspace(0, 1, 80),
            "x2":                          np.linspace(1, 2, 80),
            "pff_grades_pass_mean":        np.linspace(50, 90, 80),
        })
        labels = pd.DataFrame({
            "player_id":   snapshot["player_id"],
            "draft_year":  [2018] * 80,
            "role_target": [0.0, 1.0] * 40,
            "is_bridge":   [True] * 80,
        })

        # n_seeds=3 — winner should be seed 2 (1-indexed in logs, 1 in our map).
        result = m.tune_hyperparams(
            snapshot, labels, "QB", "role_target", "role",
            n_trials=2, n_seeds=3,
        )

    assert result == {"learning_rate": 0.05}, f"Expected winning seed's params, got {result}"


def test_tune_hyperparams_single_seed_legacy_behavior():
    """n_seeds=1 must reproduce legacy single-seed behavior — exactly one Optuna
    study created."""
    m = _get_dm()
    studies_created: list[int] = []

    class _FakeSampler:
        def __init__(self, seed):
            self.seed = seed

    class _FakeTrial:
        value = 0.1
        params = {"learning_rate": 0.05}

    class _FakeStudy:
        def __init__(self, sampler):
            studies_created.append(sampler.seed)
            self.best_trial = _FakeTrial()
        def optimize(self, *args, **kwargs):
            pass

    fake_optuna = types.SimpleNamespace(
        samplers=types.SimpleNamespace(TPESampler=_FakeSampler),
        create_study=lambda direction, sampler: _FakeStudy(sampler),
        logging=types.SimpleNamespace(set_verbosity=lambda *a, **k: None, WARNING=30),
        TrialPruned=type("TrialPruned", (Exception,), {}),
    )

    snapshot = pd.DataFrame({
        "player_id":                  [f"P{i:03d}" for i in range(80)],
        "position_group":             ["QB"] * 80,
        "last_college_season_year":   sorted(list(range(2006, 2022)) * 5),
        "pff_grades_pass_mean":        np.linspace(50, 90, 80),
    })
    labels = pd.DataFrame({
        "player_id":   snapshot["player_id"],
        "draft_year":  [2018] * 80,
        "role_target": [0.0, 1.0] * 40,
        "is_bridge":   [True] * 80,
    })

    with mock.patch.dict(sys.modules, {"optuna": fake_optuna}):
        m.tune_hyperparams(snapshot, labels, "QB", "role_target", "role",
                           n_trials=2, n_seeds=1)
    assert len(studies_created) == 1


# ── Boruta audit ───────────────────────────────────────────────────────────────

def test_boruta_drop_features_pruned_from_build_feature_cols():
    """Anything in config.BORUTA_DROP_FEATURES must be absent from the result of
    _build_feature_cols, even when the upstream snapshot exposes that column."""
    m = _get_dm()
    drop = list(C.BORUTA_DROP_FEATURES)
    if not drop:
        pytest.skip("BORUTA_DROP_FEATURES is empty — exercised separately by "
                    "test_boruta_drop_features_pruned_with_synthetic_drop")
    # Synthesize a snapshot wide enough to surface every potentially-included
    # column for QB so the drop list has something to bite on.
    cols = ["last_college_season_year", "position_group"] + drop + [
        "kalman_mean_pass_yards",   # something legitimate to keep
    ]
    snap = pd.DataFrame({c: [0.0] for c in cols})
    out = m._build_feature_cols(snap, "QB")
    for f in drop:
        assert f not in out, f"{f} survived BORUTA_DROP_FEATURES filter"


def test_boruta_drop_features_pruned_with_synthetic_drop(monkeypatch):
    """Exercise the drop logic regardless of the live config — inject a fake
    drop set so the filter is verified even when BORUTA_DROP_FEATURES is empty
    (current production state). Locks the contract that future edits to
    _build_feature_cols can't accidentally bypass the filter."""
    m = _get_dm()
    fake_drop = frozenset({"kalman_mean_pass_yards", "ewma_pass_tds"})
    # Patch both the module-level reference (config import) and the live config
    # so _build_feature_cols' filter logic sees the synthetic set.
    monkeypatch.setattr(C, "BORUTA_DROP_FEATURES", fake_drop)
    monkeypatch.setattr(m, "BORUTA_DROP_FEATURES", fake_drop, raising=False)
    cols = ["last_college_season_year", "position_group"] + list(fake_drop) + [
        "kalman_mean_pass_att",   # legitimate — must survive
    ]
    snap = pd.DataFrame({c: [0.0] for c in cols})
    out = m._build_feature_cols(snap, "QB")
    for f in fake_drop:
        assert f not in out, f"{f} survived synthetic BORUTA_DROP_FEATURES filter"
    assert "kalman_mean_pass_att" in out, \
        "non-dropped feature was filtered out — drop logic over-pruned"


# ── MI filter ──────────────────────────────────────────────────────────────────

def test_mi_filter_passthrough_when_no_cache(tmp_path, monkeypatch):
    m = _get_dm()
    monkeypatch.setattr(m, "MI_FEATURES_PATH", tmp_path / "missing.json")
    cols = ["combine_a", "kalman_mean_pff_x", "kalman_mean_pff_y", "hasan_rank"]
    assert m._filter_state_space_to_mi(cols, "QB", "role") == cols


def test_mi_filter_keeps_non_state_space_and_culls_state_space(tmp_path, monkeypatch):
    """Cache present → drop unselected state-space cols, keep all others."""
    import json as _json
    m = _get_dm()
    cache = tmp_path / "mi_features.json"
    cache.write_text(_json.dumps({
        "QB/role": {
            "selected": ["kalman_mean_pff_y"],   # keep only y
            "n_state_space_in": 2, "n_selected": 1,
        }
    }))
    monkeypatch.setattr(m, "MI_FEATURES_PATH", cache)
    cols = ["combine_a", "kalman_mean_pff_x", "kalman_mean_pff_y", "hasan_rank"]
    out = m._filter_state_space_to_mi(cols, "QB", "role")
    # Order preserved; only kalman_mean_pff_x dropped.
    assert out == ["combine_a", "kalman_mean_pff_y", "hasan_rank"]


def test_mi_filter_falls_back_when_no_state_space_match(tmp_path, monkeypatch):
    """Cache lists state-space cols not in input → defensive fall-through."""
    import json as _json
    m = _get_dm()
    cache = tmp_path / "mi_features.json"
    cache.write_text(_json.dumps({
        "QB/role": {"selected": ["kalman_mean_pff_unrelated"], "n_state_space_in": 0, "n_selected": 1}
    }))
    monkeypatch.setattr(m, "MI_FEATURES_PATH", cache)
    cols = ["combine_a", "kalman_mean_pff_x", "kalman_mean_pff_y"]
    # Intersection with cols' state-space subset is empty → fall through to all.
    assert m._filter_state_space_to_mi(cols, "QB", "role") == cols


def test_mi_runner_orders_by_signal():
    """rank_by_mi puts real-signal cols above noise."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.mi_filter import rank_by_mi, select_top_mi
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "real_a":  rng.standard_normal(n),
        "real_b":  rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
    })
    X.loc[0:30, "noise_1"] = np.nan   # imputation path
    y = X["real_a"] * 2.0 + X["real_b"] * -1.5 + 0.1 * rng.standard_normal(n)
    verdict = rank_by_mi(X, y, random_state=0)
    assert verdict["ranking"][0] in ("real_a", "real_b")
    assert verdict["ranking"][1] in ("real_a", "real_b")
    top2, _ = select_top_mi(X, y, n_top=2, random_state=0)
    assert set(top2) == {"real_a", "real_b"}


def test_boruta_runner_separates_signal_from_noise():
    """Sanity: the Boruta runner should confirm features that drive y and
    reject pure noise. Uses a small n=300 synthetic frame (deterministic
    seed) so this stays fast (~2s on this machine)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
    from _boruta_runner import run_boruta, select_features

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "real_a": rng.standard_normal(n),
        "real_b": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
        "noise_3": rng.standard_normal(n),
    })
    # Inject NaN to verify imputation path.
    X.loc[0:20, "noise_1"] = np.nan
    y = X["real_a"] * 2.0 + X["real_b"] * -1.5 + 0.1 * rng.standard_normal(n)
    verdict = run_boruta(X, y, max_iter=20, random_state=0)
    assert "real_a" in verdict["confirmed"]
    assert "real_b" in verdict["confirmed"]
    assert all(c not in verdict["confirmed"] for c in ("noise_1", "noise_2", "noise_3"))
    # select_features preserves input column order
    kept = select_features(verdict)
    assert kept == [c for c in X.columns if c in set(verdict["confirmed"])]


def test_logloss_handles_soft_labels():
    """Targets in {0, 0.5, 0.92, 1.0} should produce a finite, non-negative score
    (no crashes from sklearn's binary log_loss assumption)."""
    m = _get_dm()
    rng = np.random.default_rng(7)
    years = list(range(C.START_YEAR, C.END_YEAR - 1))
    rows = []
    soft_targets = [0.0, 0.5, 0.92, 1.0]
    for yr in years:
        for _ in range(80):
            rows.append({
                "x1": float(rng.normal()),
                "x2": float(rng.normal()),
                "last_college_season_year": yr,
                "target": float(rng.choice(soft_targets)),
            })
    df = pd.DataFrame(rows)
    score, _ = m._evaluate_params(
        {"n_estimators": 50},
        m._build_fold_data(df, ["x1", "x2"], "target",
                           m._expanding_window_folds()),
        ["x1", "x2"], metric="logloss",
    )
    assert score is not None
    assert np.isfinite(score)
    assert score >= 0.0
