"""Tests for the bridge-only Spearman tuning objective.

The objective is IMPLEMENTED but NOT SELECTED. It replaced MAE on 2026-07-26
and was reverted on 2026-07-27 after a full 3-seed retune: it improved the
metric it optimised on 19/19 heads and transferred to none of them. The
mechanism it targeted was real (reg_lambda collapsed 7.435 -> 0.0152, so MAE
really was selecting for prediction shrinkage) but bridge-only rho is computed
on 19-237 rows against MAE's thousands, so Optuna exploits its noise.

These tests keep the implementation correct and pin the reverted wiring, so the
option stays available if the sample-size problem is ever solved.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _p8():
    if "p8_spearman" in sys.modules:
        return sys.modules["p8_spearman"]
    mod_path = Path(__file__).parent.parent / "pipeline" / "08_draft_models.py"
    spec = importlib.util.spec_from_file_location("p8_spearman", mod_path)
    mod = types.ModuleType("p8_spearman")
    mod.__file__ = str(mod_path)
    sys.modules["p8_spearman"] = mod
    spec.loader.exec_module(mod)
    return mod


def _frame(n=200, pos="QB"):
    """Half bridge with graded targets, half non-bridge zeros — the mixture
    shape the real heads train on."""
    rng = np.random.default_rng(0)
    n_b = n // 2
    sig = rng.normal(size=n)
    return pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "position_group": pos,
        "last_college_season_year": rng.choice([2010, 2013, 2016, 2019], n),
        "is_bridge": [True] * n_b + [False] * (n - n_b),
        "feat_a": sig,
        "feat_b": rng.normal(size=n),
        "impact_target": np.concatenate([
            np.clip(sig[:n_b] * 0.01 + 0.02, 0, None),
            np.zeros(n - n_b)]),
    })


class TestFoldDataCarriesBridgeMask:
    def test_returns_six_tuple(self):
        mod = _p8()
        df = _frame()
        fd = mod._build_fold_data(df, ["feat_a", "feat_b"], "impact_target",
                                  [(2013, 2016, 2019)])
        assert fd and len(fd[0]) == 6

    def test_bridge_mask_matches_the_frame(self):
        mod = _p8()
        df = _frame()
        folds = [(2013, 2016, 2019)]
        fd = mod._build_fold_data(df, ["feat_a", "feat_b"], "impact_target", folds)
        yr = df["last_college_season_year"]
        expected = df[(yr >= 2016) & (yr <= 2019)]["is_bridge"].values
        assert list(fd[0][5]) == list(expected)

    def test_missing_is_bridge_defaults_to_all_true(self):
        """Callers without the column must not crash; treating every row as
        bridge is the conservative fallback."""
        mod = _p8()
        df = _frame().drop(columns=["is_bridge"])
        fd = mod._build_fold_data(df, ["feat_a", "feat_b"], "impact_target",
                                  [(2013, 2016, 2019)])
        assert fd[0][5].all()


class TestSpearmanMetric:
    def _fold(self, preds_y_pairs, bridge=None):
        """Build a minimal fold_data entry; the model is trained on the same
        signal it scores so predictions track y."""
        y = np.array(preds_y_pairs, dtype=float)
        n = len(y)
        X = y.reshape(-1, 1) + 0.0
        b = np.ones(n, dtype=bool) if bridge is None else np.asarray(bridge)
        return [(X, y, np.full(n, 2015.0), X, y, b)]

    def test_accepts_spearman(self):
        mod = _p8()
        fd = self._fold(list(np.linspace(0, 1, 60)))
        score, n_best = mod._evaluate_params({"n_estimators": 20}, fd,
                                             ["f"], metric="spearman")
        assert score is not None and n_best is not None

    def test_rejects_unknown_metric(self):
        mod = _p8()
        with pytest.raises(ValueError):
            mod._evaluate_params({}, self._fold([0.0, 1.0] * 30), ["f"],
                                 metric="nonsense")

    def test_returns_negative_rho_so_lower_is_better(self):
        """A model that ranks correctly must score BELOW one that does not,
        because Optuna minimises."""
        mod = _p8()
        y = np.linspace(0, 1, 60)
        good = [(y.reshape(-1, 1), y, np.full(60, 2015.0),
                 y.reshape(-1, 1), y, np.ones(60, dtype=bool))]
        noise = np.random.default_rng(1).normal(size=60)
        bad = [(noise.reshape(-1, 1), y, np.full(60, 2015.0),
                noise.reshape(-1, 1), y, np.ones(60, dtype=bool))]
        s_good, _ = mod._evaluate_params({"n_estimators": 50}, good, ["f"],
                                         metric="spearman")
        s_bad, _ = mod._evaluate_params({"n_estimators": 50}, bad, ["f"],
                                        metric="spearman")
        assert s_good < s_bad
        assert s_good <= 0.0        # -rho for a positively-ranked model

    def test_scores_bridge_rows_only(self):
        """Non-bridge rows are zero-padded and carry no ordering, so including
        them would corrupt the correlation. Two folds identical except for
        their non-bridge rows must score the same."""
        mod = _p8()
        y_b = np.linspace(0.01, 0.2, 40)
        y = np.concatenate([y_b, np.zeros(40)])
        X = y.reshape(-1, 1)
        bridge = np.array([True] * 40 + [False] * 40)
        fd_a = [(X, y, np.full(80, 2015.0), X, y, bridge)]
        # Same bridge rows; non-bridge values perturbed (still zero targets).
        X2 = X.copy()
        X2[40:] += 5.0
        fd_b = [(X, y, np.full(80, 2015.0), X2, y, bridge)]
        s_a, _ = mod._evaluate_params({"n_estimators": 30}, fd_a, ["f"],
                                      metric="spearman")
        s_b, _ = mod._evaluate_params({"n_estimators": 30}, fd_b, ["f"],
                                      metric="spearman")
        assert s_a == pytest.approx(s_b, abs=1e-9)

    def test_degenerate_fold_is_skipped_not_scored(self):
        """A fold with constant bridge targets cannot support a correlation.
        It must be skipped, so it neither rewards nor punishes the trial —
        scoring it as 0 would make a useless fold look neutral-good."""
        mod = _p8()
        n = 40
        y = np.concatenate([np.full(20, 0.05), np.zeros(20)])
        X = np.arange(n, dtype=float).reshape(-1, 1)
        bridge = np.array([True] * 20 + [False] * 20)
        fd = [(X, y, np.full(n, 2015.0), X, y, bridge)]
        score, n_best = mod._evaluate_params({"n_estimators": 20}, fd, ["f"],
                                             metric="spearman")
        assert score is None and n_best is None

    def test_too_few_bridge_rows_is_skipped(self):
        mod = _p8()
        n = 30
        y = np.concatenate([np.linspace(0.01, 0.1, 5), np.zeros(25)])
        X = y.reshape(-1, 1)
        bridge = np.array([True] * 5 + [False] * 25)
        fd = [(X, y, np.full(n, 2015.0), X, y, bridge)]
        score, _ = mod._evaluate_params({"n_estimators": 20}, fd, ["f"],
                                        metric="spearman")
        assert score is None


class TestObjectiveWiring:
    def test_impact_heads_select_mae(self):
        """Pins the 2026-07-27 REVERT. The spearman objective is implemented
        and tested here, but selection stays on mae: it improved the optimised
        metric on 19/19 heads and transferred to none of them (holdout rho core
        mean -0.014, cv_mae +30%) — objective overfitting, because bridge-only
        rho is computed on 19-237 rows while mae uses thousands. If this flips
        back, the retune evidence must be redone."""
        src = (Path(__file__).parent.parent / "pipeline"
               / "08_draft_models.py").read_text()
        assert 'metric = "logloss" if model_name == "role" else "mae"' in src

    def test_role_keeps_logloss(self):
        src = (Path(__file__).parent.parent / "pipeline"
               / "08_draft_models.py").read_text()
        assert '"logloss" if model_name == "role"' in src

    def test_global_model_matches(self):
        src = (Path(__file__).parent.parent / "pipeline"
               / "08b_global_draft_model.py").read_text()
        assert 'metric="logloss" if model_name == "role" else "mae"' in src

    def test_objective_is_in_the_cache_fingerprint(self):
        """Params tuned under a different objective are not interchangeable, so
        switching objectives must invalidate the cache rather than silently
        reuse the old params."""
        mod = _p8()
        assert hasattr(mod, "TUNING_OBJECTIVE_TAG")
        src = (Path(__file__).parent.parent / "pipeline"
               / "08_draft_models.py").read_text()
        assert '"tuning_objective":        TUNING_OBJECTIVE_TAG' in src


class TestDegenerateTrialsAreScoredNotSkipped:
    """Regression for the 2026-07-26 retune crash.

    Skipping folds whose PREDICTIONS were constant meant a trial could skip
    every fold, return None, and be pruned. On K/impact most sampled params
    predict a constant, so all 30 trials in a seed were pruned and Optuna's
    `study.best_trial` raised, killing step 8 ~90 minutes in. A constant
    prediction is not unscoreable — it is a model with no ranking ability, so
    it must score rho = 0. Hiding it would also defeat the point of the
    objective, since shrinkage-to-a-constant is the exact pathology it exists
    to penalise.
    """

    def _fold(self, n=200, constant_pred=False):
        # n must clear LightGBM's default min_child_samples=20 comfortably or
        # the tree cannot split and BOTH arms predict a constant, which would
        # make this test vacuously pass/fail on the fixture rather than the
        # behaviour under test.
        y = np.concatenate([np.linspace(0.01, 0.2, n // 2), np.zeros(n - n // 2)])
        b = np.array([True] * (n // 2) + [False] * (n - n // 2))
        X = (np.zeros((n, 1)) if constant_pred else y.reshape(-1, 1))
        return [(X, y, np.full(n, 2015.0), X, y, b)]

    def test_constant_prediction_scores_zero_not_none(self):
        mod = _p8()
        score, n_best = mod._evaluate_params(
            {"n_estimators": 20}, self._fold(constant_pred=True), ["f"],
            metric="spearman")
        assert score is not None
        assert score == pytest.approx(0.0, abs=1e-9)
        assert n_best is not None

    def test_constant_prediction_scores_worse_than_a_ranking_model(self):
        mod = _p8()
        s_const, _ = mod._evaluate_params({"n_estimators": 20},
                                          self._fold(constant_pred=True),
                                          ["f"], metric="spearman")
        s_rank, _ = mod._evaluate_params({"n_estimators": 20},
                                         self._fold(constant_pred=False),
                                         ["f"], metric="spearman")
        assert s_rank < s_const      # lower is better; a ranker must win

    def test_constant_target_is_still_skipped(self):
        """Distinct from a constant prediction: a constant TARGET carries no
        ordering to recover, and that is a property of the data, so skipping
        stays correct."""
        mod = _p8()
        n = 40
        y = np.concatenate([np.full(20, 0.05), np.zeros(20)])
        b = np.array([True] * 20 + [False] * 20)
        X = np.arange(n, dtype=float).reshape(-1, 1)
        score, _ = mod._evaluate_params({"n_estimators": 20},
                                        [(X, y, np.full(n, 2015.0), X, y, b)],
                                        ["f"], metric="spearman")
        assert score is None


class TestRankIncapableHeadFallsBackToMae:
    def test_fallback_is_wired(self):
        """K/impact_wav and P/impact_wav have zero folds able to support a rank
        correlation; without a fallback they would prune every trial and end up
        untuned."""
        src = (Path(__file__).parent.parent / "pipeline"
               / "08_draft_models.py").read_text()
        assert 'falling back to mae for this head' in src
        assert 'n_usable < 2' in src

    def test_all_pruned_study_does_not_crash(self):
        """`study.best_trial` RAISES when every trial is pruned, so a
        truthiness check was never enough."""
        src = (Path(__file__).parent.parent / "pipeline"
               / "08_draft_models.py").read_text()
        assert "bt = study.best_trial" in src
        assert "except (ValueError, RuntimeError):" in src
