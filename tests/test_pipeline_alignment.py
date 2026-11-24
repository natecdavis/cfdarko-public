"""
Pipeline alignment guards.

Two tests defend against a regression class where a feature is silently
dropped between Step 8 (training) and Step 10 (scoring), causing LightGBM
to use NaN/missing-branch behavior on real, available data:

  1. Steps 3, 8, 8b, 9, 10 must all use the shared `ATTRIBUTE_MERGE_COLS`
     constant rather than their own inline lists.
  2. Every feature a trained model expects must be present in the latest
     projection (or be a known derived-at-inference column).
"""

import pickle
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_attribute_merge_cols_is_used_in_each_step():
    """Every step that merges player_attributes must import the shared schema."""
    expected = "ATTRIBUTE_MERGE_COLS"
    files = [
        "pipeline/03_player_attributes.py",
        # The attrs merge itself now lives in the shared assembly helper;
        # steps 8/8b/9/10 must still reference the schema (via the helper).
        "pipeline/_snapshot_assembly.py",
        "pipeline/08_draft_models.py",
        "pipeline/08b_global_draft_model.py",
        "pipeline/09_comps.py",
        "pipeline/10_output.py",
    ]
    missing = []
    for f in files:
        text = (PROJECT_ROOT / f).read_text()
        if expected not in text:
            missing.append(f)
    assert not missing, (
        f"These files do not reference {expected} — adding a feature there "
        f"won't propagate without updating each duplicated list: {missing}"
    )


def test_trained_models_features_in_latest_projection():
    """Every feature trained models expect must be in latest.parquet
    (or be a known derived-at-inference feature). Skips if data missing
    so CI without baked artifacts still passes."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from config import DERIVED_AT_INFERENCE

    proj = PROJECT_ROOT / "data/output/projections/latest.parquet"
    models_dir = PROJECT_ROOT / "data/processed/draft_models"
    if not proj.exists() or not list(models_dir.glob("*_model.pkl")):
        pytest.skip("latest.parquet or trained models not present")

    df = pd.read_parquet(proj)
    cols = set(df.columns)
    bad: dict[str, list[str]] = {}
    for pkl in models_dir.glob("*_model.pkl"):
        with open(pkl, "rb") as f:
            bundle = pickle.load(f)
        missing = [c for c in bundle.get("feature_cols", [])
                   if c not in cols and c not in DERIVED_AT_INFERENCE]
        if missing:
            bad[pkl.name] = missing
    assert not bad, (
        f"Trained models reference features missing from output: {bad}. "
        f"Either add the feature to ATTRIBUTE_MERGE_COLS in config.py "
        f"(if it should be merged from player_attributes.parquet) or to "
        f"DERIVED_AT_INFERENCE (if it's computed inline at scoring)."
    )
