"""Tests for the early-cohort filter on the impact heads.

OverTheCap's coverage of small contracts improved over time, so pre-2011 bridge
players carry false "no second contract" zeros — a third of 2004-06 players with
career w_AV >= 30 are labelled washouts. The filter drops those rows from the
APY heads only.

The filter is DISABLED in production (config.IMPACT_LABEL_MIN_DRAFT_YEAR is
None): the 2026-07-26 A/B found that removing the corrupt rows costs more in
training volume than it saves in label noise. These tests therefore enable it
explicitly, so the mechanism stays correct if it is ever re-enabled, and so the
scoping (impact only, bridge only) can never silently drift.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config as C

sys.path.insert(0, str(Path(__file__).parent.parent))


def _p8():
    """Load pipeline/08_draft_models.py as a module (matches the pattern in
    tests/test_draft_models.py)."""
    mod_path = Path(__file__).parent.parent / "pipeline" / "08_draft_models.py"
    if "draft_models_8_cohort" in sys.modules:
        return sys.modules["draft_models_8_cohort"]
    spec = importlib.util.spec_from_file_location("draft_models_8_cohort", mod_path)
    mod = types.ModuleType("draft_models_8_cohort")
    mod.__file__ = str(mod_path)
    sys.modules["draft_models_8_cohort"] = mod
    spec.loader.exec_module(mod)
    return mod


def _frames(n_old=6, n_new=6, n_nonbridge=8, pos="QB"):
    """Snapshot + labels with old bridge rows, new bridge rows, and non-bridge
    rows, so each population's handling can be asserted separately."""
    rows, labels = [], []
    pid = 0

    def add(is_bridge, draft_year, target, lcs):
        nonlocal pid
        p = f"P{pid}"
        pid += 1
        rows.append({"player_id": p, "position_group": pos,
                     "last_college_season_year": lcs, "feat_x": float(pid)})
        labels.append({"player_id": p, "is_bridge": is_bridge,
                       "draft_year": draft_year, "impact_target": target,
                       "role_target": target, "impact_w_av_per_game": target,
                       "nfl_games": 16})

    for _ in range(n_old):
        add(True, 2006, 0.0, 2005)
    for _ in range(n_new):
        add(True, 2015, 0.05, 2014)
    for _ in range(n_nonbridge):
        add(False, np.nan, np.nan, 2014)
    return pd.DataFrame(rows), pd.DataFrame(labels)


@pytest.fixture
def enabled(monkeypatch):
    """Force the filter on regardless of the production default."""
    mod = _p8()
    monkeypatch.setattr(mod, "IMPACT_LABEL_MIN_DRAFT_YEAR", 2011)
    return mod


class TestFilterApplies:
    def test_old_bridge_rows_dropped_from_impact_head(self, enabled):
        mod = enabled
        snap, lab = _frames()
        out = mod._prep_training_df(snap, lab, "QB", "impact_target")
        old_bridge = out[out["is_bridge"].astype(bool)
                         & (out["draft_year"] < 2011)]
        assert len(old_bridge) == 0

    def test_recent_bridge_rows_survive(self, enabled):
        mod = enabled
        snap, lab = _frames()
        out = mod._prep_training_df(snap, lab, "QB", "impact_target")
        assert (out["draft_year"] == 2015).sum() == 6

    def test_non_bridge_rows_are_never_dropped(self):
        """The zero-padded non-bridge universe is load-bearing — the demean and
        role-saturation work both concluded it must stay."""
        mod = _p8()
        snap, lab = _frames()
        out = mod._prep_training_df(snap, lab, "QB", "impact_target")
        assert (~out["is_bridge"].astype(bool)).sum() == 8

    def test_non_bridge_rows_still_zero_padded(self, enabled):
        mod = enabled
        snap, lab = _frames()
        out = mod._prep_training_df(snap, lab, "QB", "impact_target")
        nb = out[~out["is_bridge"].astype(bool)]
        assert (nb["impact_target"] == 0.0).all()


class TestFilterScope:
    def test_role_head_is_untouched(self):
        """Every role arm of the cohort A/B regressed, so the filter must not
        reach role_target."""
        mod = _p8()
        snap, lab = _frames()
        out = mod._prep_training_df(snap, lab, "QB", "role_target")
        old_bridge = out[out["is_bridge"].astype(bool)
                         & (out["draft_year"] < 2011)]
        assert len(old_bridge) == 6

    def test_wav_head_is_untouched(self):
        """impact_w_av_per_game comes from PFR career AV, not from contracts,
        so it does not have the coverage problem."""
        mod = _p8()
        snap, lab = _frames()
        out = mod._prep_training_df(snap, lab, "QB", "impact_w_av_per_game")
        old_bridge = out[out["is_bridge"].astype(bool)
                         & (out["draft_year"] < 2011)]
        assert len(old_bridge) == 6


class TestDisableAndEdges:
    def test_none_disables_the_filter(self, monkeypatch):
        mod = _p8()
        monkeypatch.setattr(mod, "IMPACT_LABEL_MIN_DRAFT_YEAR", None)
        snap, lab = _frames()
        out = mod._prep_training_df(snap, lab, "QB", "impact_target")
        old_bridge = out[out["is_bridge"].astype(bool)
                         & (out["draft_year"] < 2011)]
        assert len(old_bridge) == 6

    def test_missing_draft_year_is_kept(self):
        """A bridge row with an unknown draft year must not be silently
        dropped — absence of a year is not evidence of a bad cohort."""
        mod = _p8()
        snap, lab = _frames(n_old=0, n_new=4, n_nonbridge=2)
        snap.loc[len(snap)] = {"player_id": "PX", "position_group": "QB",
                               "last_college_season_year": 2014, "feat_x": 1.0}
        lab.loc[len(lab)] = {"player_id": "PX", "is_bridge": True,
                             "draft_year": np.nan, "impact_target": 0.03,
                             "role_target": 0.03,
                             "impact_w_av_per_game": 0.03, "nfl_games": 16}
        out = mod._prep_training_df(snap, lab, "QB", "impact_target")
        assert "PX" in set(out["player_id"])

    def test_cutoff_boundary_is_inclusive_of_the_cutoff_year(self, enabled):
        """draft_year == the cutoff is KEPT (the filter is strictly
        less-than)."""
        mod = enabled
        snap, lab = _frames(n_old=0, n_new=0, n_nonbridge=2)
        snap.loc[len(snap)] = {"player_id": "PB", "position_group": "QB",
                               "last_college_season_year": 2010, "feat_x": 1.0}
        lab.loc[len(lab)] = {"player_id": "PB", "is_bridge": True,
                             "draft_year": 2011,
                             "impact_target": 0.03, "role_target": 0.03,
                             "impact_w_av_per_game": 0.03, "nfl_games": 16}
        out = mod._prep_training_df(snap, lab, "QB", "impact_target")
        assert "PB" in set(out["player_id"])

    def test_labels_without_draft_year_column_do_not_crash(self, enabled):
        mod = enabled
        snap, lab = _frames()
        out = mod._prep_training_df(snap, lab.drop(columns=["draft_year"]),
                                    "QB", "impact_target")
        assert len(out) > 0


class TestConfigValue:
    def test_filter_is_disabled_in_production(self):
        """Pins the 2026-07-26 verdict: the filter loses on cv_mae once every
        arm is scored on the same folds. If this fails, someone re-enabled it —
        make sure they re-ran the A/B with the starved-fold guard."""
        assert C.IMPACT_LABEL_MIN_DRAFT_YEAR is None

    def test_any_future_cutoff_stays_in_a_sane_range(self):
        assert C.IMPACT_LABEL_MIN_DRAFT_YEAR is None or \
               2007 <= C.IMPACT_LABEL_MIN_DRAFT_YEAR <= 2013


class TestStarvedFoldGuard:
    """_usable_folds is what turned a +14% mirage into the real answer.

    Dropping pre-2011 bridge rows empties the earliest expanding-window fold's
    bridge training population entirely. A magnitude head with no bridge rows
    predicts ~0 everywhere, which on a mostly-zero target LOWERS MAE while its
    ranking goes to chance — so the fold has to be excluded, not scored.
    """

    def _df(self, n_recent_bridge, train_end=2009):
        """Bridge rows all at last_college_season_year <= train_end."""
        return pd.DataFrame({
            "last_college_season_year": [train_end] * (n_recent_bridge + 5),
            "is_bridge": [True] * n_recent_bridge + [False] * 5,
        })

    def test_starved_fold_is_dropped(self):
        mod = _p8()
        folds = [(2009, 2010, 2012), (2012, 2013, 2015)]
        out = mod._usable_folds(self._df(0), folds, "impact_target", force=True)
        assert (2009, 2010, 2012) not in out

    def test_well_supplied_fold_is_kept(self):
        mod = _p8()
        folds = [(2009, 2010, 2012)]
        n = mod.MIN_BRIDGE_ROWS_PER_FOLD + 10
        out = mod._usable_folds(self._df(n), folds, "impact_target", force=True)
        assert out == folds

    def test_threshold_is_inclusive(self):
        mod = _p8()
        folds = [(2009, 2010, 2012)]
        n = mod.MIN_BRIDGE_ROWS_PER_FOLD
        out = mod._usable_folds(self._df(n), folds, "impact_target", force=True)
        assert out == folds

    def test_no_op_for_non_impact_targets(self):
        """role and impact_wav keep their full training population, so their
        folds must never be dropped."""
        mod = _p8()
        folds = [(2009, 2010, 2012)]
        for tgt in ["role_target", "impact_w_av_per_game"]:
            assert mod._usable_folds(self._df(0), folds, tgt) == folds

    def test_no_op_when_filter_disabled(self, monkeypatch):
        mod = _p8()
        monkeypatch.setattr(mod, "IMPACT_LABEL_MIN_DRAFT_YEAR", None)
        folds = [(2009, 2010, 2012)]
        assert mod._usable_folds(self._df(0), folds, "impact_target") == folds

    def test_missing_is_bridge_column_is_a_no_op(self):
        mod = _p8()
        folds = [(2009, 2010, 2012)]
        df = pd.DataFrame({"last_college_season_year": [2009] * 10})
        assert mod._usable_folds(df, folds, "impact_target", force=True) == folds
