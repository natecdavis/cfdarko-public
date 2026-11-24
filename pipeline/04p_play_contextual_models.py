"""
Step 4p: Per-Play Contextual Models (LightGBM)

For each stat_type in PLAY_STATS, fit a LightGBM regressor that predicts
the stat magnitude given play-level + game-level context. Trained only on
credited rows where the play_type makes the stat physically possible
(per PLAY_TYPE_STAT_ELIGIBILITY).

Mirrors step 4 (per-game contextual residualization) but at play grain
with a non-linear learner. Expanding-window CV by season_year prevents
year-effect leakage.

Inputs:
    data/processed/player_plays.parquet         — from step 1f
    data/processed/bridge_players.parquet       — bridge filter

Outputs:
    data/processed/play_contextual_models/{stat}.lgb
    data/processed/play_contextual_models/model_summary_play.json
    data/processed/play_contextual_models/model_summary_play_manifest.json

Usage:
    python pipeline/04p_play_contextual_models.py
    python pipeline/04p_play_contextual_models.py --test     # tiny sample
    python pipeline/04p_play_contextual_models.py --stat rush  # single stat
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALL_PLAY_STAT_TYPES, FG_TRAINING_MAX_YARDS_TO_GOAL, LGBM_PLAY_PARAMS,
    PLAY_CATEGORICAL_FEATURES, PLAY_CONTEXT_FEATURES, PLAY_STAT_COL,
    PLAY_TYPE_STAT_ELIGIBILITY, PROCESSED_DIR,
)
from src.models.contextual import _ExpandingSeasonCV
from src.utils.cache_manifest import (
    file_size_mtime_fingerprint, hash_features, hash_config, write_manifest,
    tag_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLAYER_PLAYS_PATH = PROCESSED_DIR / "player_plays.parquet"
BRIDGE_PATH       = PROCESSED_DIR / "bridge_players.parquet"
MODELS_DIR        = PROCESSED_DIR / "play_contextual_models"
SUMMARY_PATH      = MODELS_DIR / "model_summary_play.json"

# Stats that have so few credited events they aren't worth modeling. Forced
# through the CV-R² gate downstream anyway, but skipping them up front saves
# wall time. (List can be empty — gate decides.)
SKIP_STATS: set[str] = {"Onside Kick", "FG Attempt Blocked", "Field Goal Block",
                         "Punt Block"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Test mode: tiny sample for smoke testing")
    parser.add_argument("--stat", type=str, default=None,
                        help="Fit only this stat_type (Title Case, e.g. 'Rush')")
    return parser.parse_args()


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and coerce feature columns. Categorical → pd.Categorical."""
    feats = [c for c in PLAY_CONTEXT_FEATURES if c in df.columns]
    X = df[feats].copy()
    for cat in PLAY_CATEGORICAL_FEATURES:
        if cat in X.columns:
            X[cat] = X[cat].astype("category")
    return X


def _fit_one_stat(stat_type: str, df: pd.DataFrame, bridge_ids: set[str],
                   test_mode: bool) -> dict | None:
    """Fit LightGBM for one stat. Returns summary row or None if skipped."""
    import lightgbm as lgb

    col = PLAY_STAT_COL[stat_type]
    if col not in df.columns:
        logger.warning("[%s] column %s not in player_plays — skipping",
                       stat_type, col)
        return None

    # Eligibility filter: only train on play_types where this stat is possible
    eligible_play_types = {pt for pt, stats in PLAY_TYPE_STAT_ELIGIBILITY.items()
                            if stat_type in stats}
    if not eligible_play_types:
        logger.info("[%s] no eligible play_types — skipping", stat_type)
        return None
    eligibility_mask = df["play_type"].isin(eligible_play_types)
    df_elig = df[eligibility_mask]

    # FG distance cap: outside ~47-yard range, attempts are conditional on
    # weather/situation rather than skill — sign-flips wind effects in the
    # residual. Train only on attempts at yards_to_goal ≤ FG_TRAINING_MAX_YARDS_TO_GOAL.
    if stat_type.startswith("Field Goal") and "yards_to_goal" in df_elig.columns:
        n_before = len(df_elig)
        df_elig = df_elig[df_elig["yards_to_goal"] <= FG_TRAINING_MAX_YARDS_TO_GOAL]
        logger.info("[%s] FG distance cap (≤%d yards_to_goal): %d → %d rows",
                    stat_type, FG_TRAINING_MAX_YARDS_TO_GOAL, n_before, len(df_elig))

    # Credited filter: only rows where this stat was credited (column non-null)
    credited = df_elig[df_elig[col].notna()]
    if len(credited) < 200:
        logger.info("[%s] only %d credited rows — skipping (min 200)",
                    stat_type, len(credited))
        return None

    # Bridge filter (training only — we'll predict on all in step 5p)
    if "athlete_id" in credited.columns:
        train = credited[credited["athlete_id"].astype(str).isin(bridge_ids)]
    else:
        train = credited
    if len(train) < 200:
        logger.info("[%s] %d eligible rows but only %d on bridge players — skipping",
                    stat_type, len(credited), len(train))
        return None

    if test_mode and len(train) > 5000:
        train = train.sample(n=5000, random_state=42)

    X = _prepare_features(train)
    y = train[col].astype(float).values
    groups = train["season_year"].astype(int).values
    logger.info("[%s] training: %d rows, %d features (%d categorical), seasons=%d-%d",
                stat_type, len(train), X.shape[1],
                sum(c in X.columns for c in PLAY_CATEGORICAL_FEATURES),
                int(groups.min()), int(groups.max()))

    # Expanding-window CV
    cv = _ExpandingSeasonCV(min_train_seasons=3 if test_mode else 5,
                             min_test_size=50)
    cv_scores: list[float] = []
    cat_features = [c for c in PLAY_CATEGORICAL_FEATURES if c in X.columns]

    fold_idx = 0
    for train_idx, test_idx in cv.split(X, y, groups):
        fold_idx += 1
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features,
                              free_raw_data=False)
        dval = lgb.Dataset(X_te, label=y_te, categorical_feature=cat_features,
                            reference=dtrain, free_raw_data=False)
        params = {k: v for k, v in LGBM_PLAY_PARAMS.items()
                   if k not in ("num_boost_round", "early_stopping_round")}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=LGBM_PLAY_PARAMS["num_boost_round"],
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(LGBM_PLAY_PARAMS["early_stopping_round"],
                                       verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
        preds = booster.predict(X_te)
        # R² = 1 - SS_res / SS_tot
        ss_res = float(((y_te - preds) ** 2).sum())
        ss_tot = float(((y_te - y_te.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        cv_scores.append(r2)
        logger.info("  fold %d: %d train / %d test  R²=%.4f",
                    fold_idx, len(train_idx), len(test_idx), r2)

    if not cv_scores:
        logger.warning("[%s] no CV folds completed — skipping", stat_type)
        return None
    cv_r2 = float(np.nanmean(cv_scores))

    # Final fit on ALL training data (same hyperparams; no early-stopping here
    # since there's no held-out fold left)
    dfull = lgb.Dataset(X, label=y, categorical_feature=cat_features,
                         free_raw_data=False)
    params = {k: v for k, v in LGBM_PLAY_PARAMS.items()
               if k not in ("num_boost_round", "early_stopping_round")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        booster = lgb.train(
            params,
            dfull,
            num_boost_round=LGBM_PLAY_PARAMS["num_boost_round"],
            callbacks=[lgb.log_evaluation(0)],
        )

    # Save booster
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    safe = PLAY_STAT_COL[stat_type]
    out_path = MODELS_DIR / f"{safe}.lgb"
    booster.save_model(str(out_path))

    # Train-residual sanity check (μ near 0?)
    train_preds = booster.predict(X)
    train_residual_mean = float((y - train_preds).mean())

    summary = {
        "stat_type":             stat_type,
        "stat_col":              safe,
        "cv_r2":                 cv_r2,
        "cv_r2_folds":           cv_scores,
        "n_train_rows":          int(len(train)),
        "n_features":            int(X.shape[1]),
        "n_categorical":         int(len(cat_features)),
        "train_residual_mean":   train_residual_mean,
        "best_iteration":        int(booster.current_iteration()),
        "passes_r2_gate":        bool(cv_r2 >= 0.01),
    }
    logger.info("[%s] DONE  cv_r2=%.4f  best_iter=%d  resid_μ=%.4f  gate=%s",
                stat_type, cv_r2, booster.current_iteration(),
                train_residual_mean,
                "PASS" if summary["passes_r2_gate"] else "FAIL")
    return summary


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not PLAYER_PLAYS_PATH.exists():
        logger.error("player_plays.parquet not found — run step 1f first")
        sys.exit(1)
    df = pd.read_parquet(PLAYER_PLAYS_PATH)
    logger.info("Loaded %d player-play rows from %s", len(df), PLAYER_PLAYS_PATH)

    # Bridge filter
    if BRIDGE_PATH.exists():
        bridge = pd.read_parquet(BRIDGE_PATH)
        bridge_ids = set(bridge["player_id"].astype(str).tolist())
        logger.info("Bridge players: %d", len(bridge_ids))
    else:
        logger.warning("bridge_players.parquet not found — training on all players")
        bridge_ids = set(df["athlete_id"].astype(str).tolist())

    # Restrict to chosen stat(s)
    stats_to_fit = (
        [args.stat] if args.stat else
        [s for s in ALL_PLAY_STAT_TYPES if s not in SKIP_STATS]
    )
    logger.info("Fitting %d stat models: %s", len(stats_to_fit), stats_to_fit)

    summaries: list[dict] = []
    for stat in stats_to_fit:
        try:
            row = _fit_one_stat(stat, df, bridge_ids, args.test)
        except Exception as exc:
            logger.exception("[%s] FAILED: %s", stat, exc)
            row = {"stat_type": stat, "error": str(exc), "cv_r2": None}
        if row is not None:
            summaries.append(row)

    # Save summary + manifest
    payload = {"stats": summaries}
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("Saved summary → %s", SUMMARY_PATH)

    # Manifest fingerprint: features list, lgbm params, eligibility matrix hash,
    # bridge_players file fingerprint, player_plays file fingerprint
    fp = {
        "play_context_features": hash_features(PLAY_CONTEXT_FEATURES),
        "lgbm_params":            hash_config(LGBM_PLAY_PARAMS),
        "play_type_stat_elig":    hash_config({k: sorted(v) for k, v in
                                                PLAY_TYPE_STAT_ELIGIBILITY.items()}),
        "bridge_file":            file_size_mtime_fingerprint(BRIDGE_PATH),
        "player_plays_file":      file_size_mtime_fingerprint(PLAYER_PLAYS_PATH),
        "n_stats_fitted":         len(summaries),
    }
    write_manifest(
        SUMMARY_PATH,
        fingerprint=fp,
        mode_used="default",
        step_name="04p_play_contextual_models",
    )
    provenance_path = SUMMARY_PATH.parent / "model_summary_play_provenance.json"
    provenance_payload = tag_summary(
        {"n_models": len(summaries)},
        mode_used="default",
        step_name="04p_play_contextual_models",
    )
    provenance_path.write_text(json.dumps(provenance_payload, indent=2))
    logger.info("Provenance → %s (_tuning_mode=default)", provenance_path)
    logger.info("Step 4p complete. Fitted %d models, %d passed R² gate.",
                len(summaries),
                sum(1 for s in summaries if s.get("passes_r2_gate")))


if __name__ == "__main__":
    main()
