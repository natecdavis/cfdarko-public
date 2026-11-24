"""T1 — Expanding-window leakage invariants.

Locks in the contract that no fold ever has a training row with a year >=
the smallest validation year. Covers both:

  * pipeline/08_draft_models.py — _expanding_window_folds() (cohort-year folds)
  * src/models/contextual.py — _ExpandingSeasonCV (per-season-year folds for
    contextual Ridge / ElasticNet / Adaptive Lasso)

Catches the regression class that originally motivated reverting per-year
contextual coefficient refits (memory: project_contextual_expanding_window).
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


def _get_dm():
    """Load pipeline/08_draft_models.py with its scraper-import stub."""
    pff_stub = types.ModuleType("src.scrapers.pff")
    pff_stub.load_pff_career_features = lambda *a, **kw: pd.DataFrame()
    sys.modules.setdefault("src.scrapers.pff", pff_stub)

    spec = importlib.util.spec_from_file_location(
        "draft_models_08", ROOT / "pipeline" / "08_draft_models.py",
    )
    mod = types.ModuleType("draft_models_08")
    mod.__file__ = str(ROOT / "pipeline" / "08_draft_models.py")
    spec.loader.exec_module(mod)
    return mod


def test_step8_expanding_window_folds_have_no_temporal_leak():
    """Every (train_end, val_start, val_end) tuple must satisfy
    train_end < val_start, and consecutive folds' val windows must not overlap."""
    m = _get_dm()
    folds = m._expanding_window_folds()
    assert folds, "expected at least one fold"
    for train_end, val_start, val_end in folds:
        assert train_end < val_start, \
            f"fold leak: train_end={train_end} >= val_start={val_start}"
        assert val_start <= val_end, \
            f"fold inverted: val_start={val_start} > val_end={val_end}"

    # Non-overlapping val windows
    sorted_folds = sorted(folds, key=lambda f: f[1])
    for prev, cur in zip(sorted_folds, sorted_folds[1:]):
        prev_end = prev[2]
        cur_start = cur[1]
        assert prev_end < cur_start, \
            f"val windows overlap: {prev} and {cur}"


def test_step8_split_holdout_train_strictly_before_holdout():
    """The holdout fold must come strictly after every tuning fold's val window."""
    m = _get_dm()
    folds = m._expanding_window_folds()
    tuning, holdout = m._split_holdout(folds)
    if holdout is None:
        pytest.skip("not enough folds to form a holdout — pipeline misconfig")
    h_train_end, h_val_start, h_val_end = holdout
    for t_train_end, t_val_start, t_val_end in tuning:
        assert t_val_end < h_val_start, \
            f"tuning fold val_end={t_val_end} ≥ holdout val_start={h_val_start}"


def test_contextual_expanding_season_cv_no_leak():
    """For _ExpandingSeasonCV, every train index's season must be < every test
    index's season. This is the core invariant for step 4's per-stat CV."""
    from src.models.contextual import _ExpandingSeasonCV

    rng = np.random.default_rng(0)
    seasons = np.repeat(np.arange(2010, 2025), 50)        # 15 seasons × 50 rows
    rng.shuffle(seasons)
    n = len(seasons)
    X = rng.normal(size=(n, 4))
    y = rng.normal(size=n)

    cv = _ExpandingSeasonCV(min_train_seasons=5, min_test_size=10)
    n_yielded = 0
    for train_idx, test_idx in cv.split(X, y, groups=seasons):
        n_yielded += 1
        train_max = seasons[train_idx].max()
        test_min  = seasons[test_idx].min()
        assert train_max < test_min, (
            f"contextual CV leak: train season {train_max} >= test season {test_min}"
        )
    assert n_yielded > 0, "no folds yielded — guard parameters too tight"
