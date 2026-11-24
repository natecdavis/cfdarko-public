"""
Step 8b: Cross-Position Global Draft Model

Trains a single LightGBM (per target) across ALL position groups, with
position_group encoded as a categorical feature. Pools structure across
positions — particularly valuable for K/P/TE where per-position training
is starved (n=600-1500). Uses the full union of per-position features,
filling NaN for stats that don't apply to a given position.

Three targets, same as Step 8:
  - role_target          → P(role) at y3 horizon (continuous via trajectory)
  - impact_target        → second-contract APY%cap
  - impact_w_av_per_game → career w_AV / NFL games

Outputs predictions for all bridge players + current prospects to
`data/processed/global_draft_predictions.parquet` with columns:
  player_id, global_role_score, global_impact_apy_score, global_impact_wav_score

Step 10 merges these as additional rank columns alongside the per-position
predictions, so we can see where the global view agrees / disagrees.

Usage:
    python pipeline/08b_global_draft_model.py
    python pipeline/08b_global_draft_model.py --tune --tune-trials 30
"""

import argparse
import hashlib
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))   # so `from _boruta_runner import ...` resolves
from config import (
    PROCESSED_DIR, MODELED_STATS, POSITION_GROUPS, POSITION_GROUP_ORDINAL,
    END_YEAR, START_YEAR,
    LGBM_PARAMS, PFF_GRADE_FEATURES_BY_GROUP, NFL_LG_FEATURES, HASAN_FEATURES,
    PFF_DENOM_STATS,
    COMBINE_FEATURES_BY_GROUP, COMBINE_RAW_BY_GROUP,
    DRAFT_TRAIN_CUTOFF_YEAR, DRAFT_VAL_START_YEAR, ROLE_LABEL_HORIZON_YEARS,
    BORUTA_N_ESTIMATORS, BORUTA_MAX_ITER, BORUTA_PERCENTILE,
    BORUTA_ALPHA, BORUTA_RANDOM_STATE, BORUTA_DROP_FEATURES,
    MI_TOP_N, MI_RANDOM_STATE, MI_N_NEIGHBORS,
    QB_ROLLING_FEATURES, QB_CAREER_BASE_FEATURES, USE_QB_CAREER_FEATURES,
    IMPACT_LABEL_MIN_DRAFT_YEAR,
)
# Shared snapshot assembly (single source of truth across steps 8/8b/8c/9/10;
# merges player_attributes via config.ATTRIBUTE_MERGE_COLS internally).
from _snapshot_assembly import assemble_snapshot
from src.utils.cache_manifest import (
    write_manifest, check_manifest, log_stale_warning, tag_summary,
    hash_features, hash_config, schema_fingerprint,
)
from _boruta_runner import run_boruta
from src.utils.mi_filter import rank_by_mi

STEP_NAME = "08b_global_draft_model"

# Reuse Step 8 helpers (folds, holdout split, derived features, prep df).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_p8", Path(__file__).parent / "08_draft_models.py",
)
_p8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p8)

try:
    import lightgbm as lgb
    _lgbm_available = True
except ImportError:
    _lgbm_available = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GLOBAL_MODELS_DIR = PROCESSED_DIR / "global_draft_models"
GLOBAL_PREDICTIONS_PATH = PROCESSED_DIR / "global_draft_predictions.parquet"
GLOBAL_TUNED_PARAMS_PATH = GLOBAL_MODELS_DIR / "tuned_params.json"
GLOBAL_BORUTA_FEATURES_PATH = GLOBAL_MODELS_DIR / "boruta_features.json"
GLOBAL_MI_FEATURES_PATH     = GLOBAL_MODELS_DIR / "mi_features.json"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

_FINAL_TRAIN_CUTOFF = END_YEAR - 2

# Targets and friendly names.
# The role head was dropped 2026-10-31 (see step 8 HEADS).
TARGETS = [
    ("impact_target",        "impact_apy"),
    ("impact_w_av_per_game", "impact_wav"),
]


# ── MI feature filter (global) ────────────────────────────────────────────────
# Per-target MI ranking on the PFF state-space block within the union feature
# space. Three keys: global/role, global/impact_apy, global/impact_wav. Other
# categories (combine, attrs, hasan, pff_career, gg_*, position_group_ordinal)
# pass through unchanged.

def _global_state_space_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if c.startswith((
        "kalman_mean_pff_", "kalman_var_pff_",
        "ewma_pff_",        "ewma_var_pff_",
        "kalman_yoy_pff_",  "ewma_yoy_pff_",
    ))]


def _global_load_mi_features(model_name: str) -> list[str] | None:
    if not GLOBAL_MI_FEATURES_PATH.exists():
        return None
    try:
        with open(GLOBAL_MI_FEATURES_PATH) as f:
            data = json.load(f)
        block = data.get(f"global/{model_name}")
        if not isinstance(block, dict):
            return None
        selected = list(block.get("selected") or [])
        return selected or None
    except Exception:
        return None


def _global_filter_state_space_to_mi(feature_cols: list[str], model_name: str) -> list[str]:
    cached = _global_load_mi_features(model_name)
    if cached is None:
        return feature_cols
    keep_ss = set(cached)
    ss_in_input = set(_global_state_space_cols(feature_cols))
    selected_ss = ss_in_input & keep_ss
    if not selected_ss and ss_in_input:
        logger.warning("  global/%s: MI cache filtered out all state-space cols; "
                       "falling back to full set", model_name)
        return feature_cols
    out = [c for c in feature_cols
           if c not in ss_in_input or c in selected_ss]
    if len(out) < len(feature_cols):
        logger.info("  global/%s: MI filter %d → %d features (state-space %d → %d)",
                    model_name, len(feature_cols), len(out),
                    len(ss_in_input), len(selected_ss))
    return out


def _run_mi_for_global_heads(df: pd.DataFrame, feature_cols: list[str],
                              top_n: int) -> None:
    """Per-target MI ranking on the global state-space block. Three heads."""
    all_folds = _p8._expanding_window_folds()
    _, holdout_fold = _p8._split_holdout(all_folds)
    mi_train_end = (holdout_fold[0] if holdout_fold is not None
                    else _FINAL_TRAIN_CUTOFF)
    logger.info("MI training window: last_college_season_year ≤ %d", mi_train_end)

    ss_cols = _global_state_space_cols(feature_cols)
    if not ss_cols:
        logger.info("Global: no state-space cols, skipping MI filter")
        return

    yr = df["last_college_season_year"].fillna(0)
    by_key_payload: dict[str, dict] = {}
    by_key_meta: dict[str, dict] = {}

    for target_col, model_name in TARGETS:
        sub = df[(yr <= mi_train_end) & df[target_col].notna()].copy()
        if len(sub) < 100:
            logger.warning("  global/%s: only %d rows ≤ %d — skipping MI",
                           model_name, len(sub), mi_train_end)
            continue
        X = sub[ss_cols]
        y = sub[target_col].astype(float)
        verdict = rank_by_mi(X, y, n_neighbors=MI_N_NEIGHBORS, random_state=MI_RANDOM_STATE)
        ranking = verdict["ranking"]
        scores = verdict["scores"]
        selected = ranking[:top_n]
        logger.info("  global/%s: MI %d → %d state-space cols (top=%.4f, cutoff=%.4f)",
                    model_name, len(ss_cols), len(selected),
                    scores[selected[0]] if selected else 0.0,
                    scores[selected[-1]] if selected else 0.0)
        by_key_payload[f"global/{model_name}"] = {
            "selected":         selected,
            "scores":           {c: round(scores[c], 6) for c in ranking},
            "n_state_space_in": len(ss_cols),
            "n_selected":       len(selected),
            "n_train_rows":     int(len(sub)),
            "mi_train_end":     int(mi_train_end),
            "top_n":            int(top_n),
        }
        by_key_meta[f"global/{model_name}"] = {
            "candidate_cols_hash": hash_features(ss_cols),
            "n_state_space_in":    len(ss_cols),
            "n_selected":          len(selected),
        }

    if not by_key_payload:
        logger.warning("Global MI produced no verdicts — nothing to write")
        return

    GLOBAL_MI_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(by_key_payload)
    payload["_meta"] = {
        "snapshot_n_rows": int(len(df)),
        "top_n":           int(top_n),
        "n_neighbors":     int(MI_N_NEIGHBORS),
        "random_state":    int(MI_RANDOM_STATE),
    }
    with open(GLOBAL_MI_FEATURES_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    fingerprint = {
        "by_key": {k: {
            "candidate_cols_hash": meta["candidate_cols_hash"],
            "n_state_space_in":    int(meta["n_state_space_in"]),
            "n_selected":          int(meta["n_selected"]),
        } for k, meta in by_key_meta.items()},
        "global_context": {
            "snapshot_n_rows":  int(len(df)),
            "mi_top_n":         int(top_n),
            "mi_n_neighbors":   int(MI_N_NEIGHBORS),
            "mi_random_state":  int(MI_RANDOM_STATE),
        },
    }
    write_manifest(
        GLOBAL_MI_FEATURES_PATH, fingerprint=fingerprint,
        mode_used="fresh", step_name=STEP_NAME,
        extra={"n_keys": len(by_key_payload)},
    )
    logger.info("Saved global MI features for %d heads → %s",
                len(by_key_payload), GLOBAL_MI_FEATURES_PATH)


# ── Boruta audit pass (global) ────────────────────────────────────────────────
# Mirrors step 8: writes boruta_features.json under global_draft_models/ for
# `experiments/boruta_summary.py` to consume. Does NOT gate the fit — drop
# decisions live in `config.BORUTA_DROP_FEATURES`.

def _run_boruta_for_global_heads(df: pd.DataFrame, feature_cols: list[str],
                                  max_iter: int) -> None:
    """Run Boruta for each of the 3 global heads (role / impact_apy /
    impact_wav) over the union-of-positions feature space and persist."""
    all_folds = _p8._expanding_window_folds()
    _, holdout_fold = _p8._split_holdout(all_folds)
    boruta_train_end = (holdout_fold[0] if holdout_fold is not None
                        else _FINAL_TRAIN_CUTOFF)
    logger.info("Boruta training window: last_college_season_year ≤ %d",
                boruta_train_end)

    yr = df["last_college_season_year"].fillna(0)
    by_key_payload: dict[str, dict] = {}
    by_key_meta: dict[str, dict] = {}

    for target_col, model_name in TARGETS:
        sub = df[(yr <= boruta_train_end) & df[target_col].notna()].copy()
        n_train = len(sub)
        if n_train < 100:
            logger.warning("  global/%s: only %d rows ≤ %d — skipping Boruta",
                           model_name, n_train, boruta_train_end)
            continue

        X = sub[feature_cols]
        y = sub[target_col].astype(float)
        non_constant = [c for c in feature_cols if X[c].nunique(dropna=True) > 1]
        n_dropped_const = len(feature_cols) - len(non_constant)
        if n_dropped_const:
            logger.info("  global/%s: dropped %d all-constant cols pre-Boruta",
                        model_name, n_dropped_const)

        verdict = run_boruta(
            X[non_constant], y,
            n_estimators=BORUTA_N_ESTIMATORS,
            max_iter=max_iter,
            percentile=BORUTA_PERCENTILE,
            alpha=BORUTA_ALPHA,
            random_state=BORUTA_RANDOM_STATE,
            classifier=False,
        )
        confirmed = verdict["confirmed"]
        tentative = verdict["tentative"]
        n_after = len(confirmed)   # confirmed-only is the audit headline
        logger.info("  global/%s: Boruta confirmed=%d  tentative=%d  rejected=%d  (in=%d)",
                    model_name,
                    len(confirmed), len(tentative), len(verdict["rejected"]),
                    len(feature_cols))

        by_key_payload[f"global/{model_name}"] = {
            "confirmed":         confirmed,
            "tentative":         tentative,
            "rejected_count":    len(verdict["rejected"]),
            "n_features_in":     int(verdict["n_features_in"]) + n_dropped_const,
            "n_features_after":  int(n_after),
            "n_train_rows":      int(n_train),
            "boruta_train_end":  int(boruta_train_end),
            "max_iter":          int(verdict["max_iter"]),
            "percentile":        int(verdict["percentile"]),
            "alpha":             float(verdict["alpha"]),
        }
        by_key_meta[f"global/{model_name}"] = {
            "feature_cols_hash": hash_features(feature_cols),
            "n_features_in":     len(feature_cols),
            "n_features_after":  n_after,
        }

    if not by_key_payload:
        logger.warning("Global Boruta produced no verdicts — nothing to write")
        return

    GLOBAL_BORUTA_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(by_key_payload)
    payload["_meta"] = {
        "snapshot_n_rows":  int(len(df)),
        "boruta_train_end": int(boruta_train_end),
        "max_iter":         int(max_iter),
        "percentile":       int(BORUTA_PERCENTILE),
        "alpha":            float(BORUTA_ALPHA),
    }
    with open(GLOBAL_BORUTA_FEATURES_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    fingerprint = {
        "by_key": {k: {
            "feature_cols_hash": meta["feature_cols_hash"],
            "n_features_in":     int(meta["n_features_in"]),
            "n_features_after":  int(meta["n_features_after"]),
        } for k, meta in by_key_meta.items()},
        "global_context": {
            "snapshot_n_rows":   int(len(df)),
            "boruta_max_iter":   int(BORUTA_MAX_ITER),
            "boruta_percentile": int(BORUTA_PERCENTILE),
            "boruta_alpha":      float(BORUTA_ALPHA),
        },
    }
    write_manifest(
        GLOBAL_BORUTA_FEATURES_PATH, fingerprint=fingerprint,
        mode_used="fresh", step_name=STEP_NAME,
        extra={"n_keys": len(by_key_payload)},
    )
    logger.info("Saved global Boruta features for %d heads → %s",
                len(by_key_payload), GLOBAL_BORUTA_FEATURES_PATH)


def _build_global_feature_cols(snapshot: pd.DataFrame) -> list[str]:
    """Union of all per-position features. Players have NaN for stats that
    don't apply to their position; LightGBM handles NaN natively."""
    cols: list[str] = []
    seen: set[str] = set()

    def _add(c: str):
        if c not in seen and c in snapshot.columns:
            cols.append(c); seen.add(c)

    # Skill features across all position groups (union).
    for pos_group in POSITION_GROUPS.keys():
        for stat in MODELED_STATS.get(pos_group, []):
            for prefix in ("kalman_mean", "kalman_var", "ewma", "ewma_var",
                           "kalman_yoy", "ewma_yoy"):
                _add(f"{prefix}_{stat}")

    # Play-grain skill features (parallel to per-game above; discovered
    # dynamically since play-grain stat names aren't in MODELED_STATS).
    for prefix in (
        "kalman_mean_play_mean_resid_", "kalman_var_play_mean_resid_",
        "ewma_play_mean_resid_",        "ewma_var_play_mean_resid_",
        "kalman_yoy_play_mean_resid_",  "ewma_yoy_play_mean_resid_",
        "kalman_mean_play_total_resid_","kalman_var_play_total_resid_",
        "ewma_play_total_resid_",       "ewma_var_play_total_resid_",
        "kalman_yoy_play_total_resid_", "ewma_yoy_play_total_resid_",
        "kalman_mean_play_raw_mean_",   "kalman_var_play_raw_mean_",
        "ewma_play_raw_mean_",          "ewma_var_play_raw_mean_",
        "kalman_yoy_play_raw_mean_",    "ewma_yoy_play_raw_mean_",
    ):
        for c in snapshot.columns:
            if c.startswith(prefix):
                _add(c)

    # Attributes (shared).
    for c in _p8.ATTRIBUTE_FEATURES:
        _add(c)
    for c in ("age", "games_played_career", "n_games", "n_seasons"):
        _add(c)

    # Derived features (DL/LB/OL/DB position-specific dummies; safe to include
    # globally since they're 0/NaN for non-applicable groups).
    for c in ("pass_rush_composite", "team_sack_share",
              "pff_pass_rush_snap_pct", "pff_coverage_snap_pct",
              "is_ol_tackle", "is_ol_guard", "is_ol_center",
              "pff_corner_snap_pct", "pff_slot_snap_pct", "pff_fs_snap_pct",
              "career_rush_att", "career_touches",
              "pff_career_yards_after_contact",
              "pff_career_avoided_tackles",
              "pff_career_rush_att"):
        _add(c)

    # PFF career features (union).
    for pos_group, pff_cols in PFF_GRADE_FEATURES_BY_GROUP.items():
        for c in pff_cols:
            for suffix in ("_mean", "_peak", "_last"):
                _add(c + suffix)
    # Career-total denominator features (ungated union). LightGBM's missing-
    # direction handles inapplicable cells (e.g. a kicker's pff_routes_total
    # is NaN, never reached during prediction for K rows).
    for denom in PFF_DENOM_STATS:
        _add(f"{denom}_total")

    # Combine z-scores (union) + raw absolute values per audit 2026-05-10
    # (`experiments/00i_combine_zscore_audit.py`). The global model can't do
    # per-position surgery, so it gets BOTH z and raw versions of any swapped
    # metric — the LightGBM can pick whichever is more informative for the
    # position-encoded subset.
    for pos_group, cb_cols in COMBINE_FEATURES_BY_GROUP.items():
        for c in cb_cols:
            _add(c)
    for pos_group, raw_cols in COMBINE_RAW_BY_GROUP.items():
        for c in raw_cols:
            _add(c)
    _add("combine_invited")

    # NFL league averages.
    for c in NFL_LG_FEATURES:
        _add(c)

    # Hasan board rank.
    for c in HASAN_FEATURES:
        _add(c)

    # Teammate-QB career triplets (union). The global model can't position-
    # gate at fit time, so it sees the columns for every player; NaN for
    # non-WR/TE/RB players is handled by LightGBM's missing direction.
    # Slimmed to QB_CAREER_BASE_FEATURES post-Boruta-audit.
    if USE_QB_CAREER_FEATURES:
        for base in QB_CAREER_BASE_FEATURES:
            for suffix in ("_mean", "_peak", "_last"):
                _add(f"{base}{suffix}")

    # Position group as a categorical feature — the whole point of going global.
    _add("position_group_ordinal")

    # Strip the curated Boruta drop list (set in config). Same set is applied
    # in step 8's `_build_feature_cols` — single source of truth.
    if BORUTA_DROP_FEATURES:
        cols = [c for c in cols if c not in BORUTA_DROP_FEATURES]
    return cols


def _attach_position_group_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "position_group" in df.columns:
        df["position_group_ordinal"] = (
            df["position_group"].map(POSITION_GROUP_ORDINAL).astype(float)
        )
    return df


def build_global_training_frame(labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Snapshot ⋈ labels and the global feature list, exactly as main() fits.
    Shared with experiments that refit the global heads."""
    # Shared assembly (attrs / age / sack-share / PFF / QB / OL-fill /
    # combine / hasan / league-averages / derived features) — identical to
    # Step 8's training matrix by construction. 8b then adds its own
    # position_group_ordinal on top.
    snapshot = assemble_snapshot(include_derived=True, logger=logger)
    # Drop the hasan_draft_year bookkeeping column (canonical assembly keeps
    # it; 8b historically dropped it). It is not a feature, but it IS part of
    # the schema_fingerprint that guards the cached global tuned params —
    # keeping 8b's frame schema unchanged avoids a false-stale on that cache.
    snapshot = snapshot.drop(columns=["hasan_draft_year"], errors="ignore")
    snapshot = _attach_position_group_ordinal(snapshot)

    # Merge labels.
    df = snapshot.merge(
        labels[["player_id", "role_target", "impact_target", "impact_w_av_per_game",
                "is_bridge", "draft_year"]],
        on="player_id", how="left",
    )
    df["is_bridge"] = df["is_bridge"].fillna(False)

    # Build feature columns.
    feature_cols = _build_global_feature_cols(df)
    logger.info("Global model feature space: %d columns", len(feature_cols))

    # Drop dominant-value features (e.g. EWMA features stuck at the Bayesian
    # prior for the >99% of players who never had data for that stat). Applied
    # against the full cross-position df since the global model trains on all
    # positions together — the prior-anchored features still pass this filter
    # if at least one position cohort has real variance in them.
    feature_cols = _p8._drop_dominant_value_features(feature_cols, df, max_share=0.99)
    logger.info("Post dominant-value filter: %d columns", len(feature_cols))
    return df, feature_cols


def _train_one_global(df: pd.DataFrame, feature_cols: list[str],
                      target_col: str, model_name: str,
                      tuned_params: dict | None,
                      test_mode: bool = False) -> tuple[object, dict, list[str]]:
    """Train one global LightGBM with expanding-window CV + holdout fold.
    Returns (model, metrics, feature_cols_used). feature_cols_used is the
    MI-filtered subset when a cache is present, else the full input."""
    feature_cols = _global_filter_state_space_to_mi(feature_cols, model_name)
    params = {**LGBM_PARAMS}
    if tuned_params:
        params.update(tuned_params)
    if test_mode:
        params["n_estimators"] = 50

    all_folds = _p8._expanding_window_folds()
    tuning_folds, holdout_fold = _p8._split_holdout(all_folds)
    # Same starvation guard as step 8: the earliest fold's training window is
    # entirely pre-cutoff, so after the early-cohort filter it holds no bridge
    # rows at all and the head collapses to predicting zero.
    df_guard = df.copy()
    if (IMPACT_LABEL_MIN_DRAFT_YEAR is not None
            and target_col == "impact_target"
            and "draft_year" in df_guard.columns):
        stale = (df_guard["is_bridge"].astype(bool)
                 & df_guard["draft_year"].notna()
                 & (df_guard["draft_year"] < IMPACT_LABEL_MIN_DRAFT_YEAR))
        df_guard = df_guard[~stale]
    tuning_folds = _p8._usable_folds(df_guard, tuning_folds, target_col,
                                      "__global__", model_name)

    yr_series = df["last_college_season_year"].fillna(0)
    oof_records: list[pd.DataFrame] = []

    def _fit_eval(train_end: int, val_start: int, val_end: int,
                  collect_oof: bool = False):
        f_train = df[yr_series <= train_end]
        f_val_full = df[(yr_series >= val_start) & (yr_series <= val_end)]
        # Drop rows with NaN target in training (esp. impact_w_av_per_game).
        if target_col in f_train.columns:
            f_train = f_train[f_train[target_col].notna()]
            f_val = f_val_full[f_val_full[target_col].notna()]
        else:
            f_val = f_val_full
        # Mirror step 8's early-cohort filter so the global APY head trains on
        # the same population as the per-position ones (see config rationale).
        if (IMPACT_LABEL_MIN_DRAFT_YEAR is not None
                and target_col == "impact_target"
                and "draft_year" in f_train.columns):
            stale = (f_train["is_bridge"].astype(bool)
                     & f_train["draft_year"].notna()
                     & (f_train["draft_year"] < IMPACT_LABEL_MIN_DRAFT_YEAR))
            f_train = f_train[~stale]
        if len(f_train) < 100 or len(f_val) < 30:
            return None
        X_tr = f_train[feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y_tr = f_train[target_col].astype(float).values
        X_v  = f_val[feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y_v  = f_val[target_col].astype(float).values
        _make = lambda: lgb.LGBMRegressor(**{
            **params,
            **_p8._build_monotone_kwargs(feature_cols, pos_group="__global__",
                                         model_name="__global__")})
        # config.IMPACT_BOARD_ROW_WEIGHT (default 1.0 → None, unweighted)
        w_tr = (_p8._board_row_weights(f_train, "impact")
                if target_col == "impact_target" else None)
        m, best_iter = _p8._fit_fold_honest(
            _make, X_tr, y_tr,
            f_train["last_college_season_year"].fillna(0).values,
            sample_weight=w_tr)
        if collect_oof and len(f_val_full):
            # OOF predictions over the FULL fold-val cohort (including rows
            # whose target is NaN, e.g. non-bridge impact_wav — 8c pads those
            # to the mixture scale when fitting per-position calibrators).
            X_full = f_val_full[feature_cols].to_numpy(dtype=float,
                                                       na_value=np.nan)
            # For impact_apy also fit the fold's p90 quantile head and record
            # its OOF predictions — 8c turns these into per-position CQR
            # offsets so the production p90 band is calibrated to the same
            # scale as the isotonic-calibrated point estimate (without this,
            # raw p90 < calibrated score for ~95% of players and the band
            # clamp collapses Upside onto Impact).
            p90_pred = np.full(len(f_val_full), np.nan)
            if model_name == "impact_apy":
                qm = lgb.LGBMRegressor(**{**params, "n_estimators": best_iter,
                                          "objective": "quantile",
                                          "alpha": 0.9})
                qm.fit(X_tr, y_tr, sample_weight=w_tr)
                p90_pred = np.clip(qm.predict(X_full), 0.0, None)
            oof_records.append(pd.DataFrame({
                "player_id":      f_val_full["player_id"].astype(str).values,
                "position_group": f_val_full.get(
                    "position_group",
                    pd.Series("", index=f_val_full.index)).values,
                "model_name":     model_name,
                "val_start":      val_start,
                "val_end":        val_end,
                "y":              f_val_full[target_col].astype(float).values,
                "is_bridge":      f_val_full["is_bridge"].astype(bool).values,
                "pred":           m.predict(X_full),
                "p90_pred":       p90_pred,
            }))
        return m, m.predict(X_v), y_v, best_iter

    fold_maes, fold_n_bests = [], []
    for tr_end, v_start, v_end in tuning_folds:
        out = _fit_eval(tr_end, v_start, v_end, collect_oof=True)
        if out is None:
            continue
        m, preds, y, best_iter = out
        fold_maes.append(float(np.mean(np.abs(preds - y))))
        fold_n_bests.append(best_iter)

    if not fold_n_bests:
        raise ValueError(f"No valid CV folds for global/{model_name}")

    n_best = max(1, int(round(float(np.mean(fold_n_bests)))))
    metrics = {
        "n_cv_folds": len(fold_maes),
        "cv_mae": float(np.mean(fold_maes)),
        "avg_best_iter": n_best,
    }

    # Holdout fold report.
    if holdout_fold is not None:
        h_tr_end, h_v_start, h_v_end = holdout_fold
        out = _fit_eval(h_tr_end, h_v_start, h_v_end)
        if out is not None:
            _, h_preds, h_y, _ = out
            metrics["holdout_train_end"] = h_tr_end
            metrics["holdout_val_start"] = h_v_start
            metrics["holdout_val_end"] = h_v_end
            metrics["holdout_mae"] = float(np.mean(np.abs(h_preds - h_y)))
            from scipy.stats import spearmanr
            try:
                rho, _ = spearmanr(h_preds, h_y)
                if rho is not None and not np.isnan(rho):
                    metrics["holdout_spearman"] = float(rho)
            except Exception:
                pass
            logger.info("  global/%s: HOLDOUT mae=%.4f spearman=%.4f (n_train=%d, n_val=%d)",
                        model_name, metrics["holdout_mae"],
                        metrics.get("holdout_spearman", float("nan")),
                        int((yr_series <= h_tr_end).sum()),
                        int(((yr_series >= h_v_start) & (yr_series <= h_v_end)).sum()))

    # Final fit on all data ≤ _FINAL_TRAIN_CUTOFF (target non-null).
    full = df[yr_series <= _FINAL_TRAIN_CUTOFF]
    if target_col in full.columns:
        full = full[full[target_col].notna()]
    model = lgb.LGBMRegressor(**{**params, "n_estimators": n_best,
                                  **_p8._build_monotone_kwargs(feature_cols, pos_group="__global__", model_name="__global__")})
    w_full = (_p8._board_row_weights(full, "impact")
              if target_col == "impact_target" else None)
    model.fit(
        full[feature_cols].to_numpy(dtype=float, na_value=np.nan),
        full[target_col].astype(float).values,
        sample_weight=w_full,
        feature_name="auto",
    )
    # Global quantile heads (impact_apy only — the head whose p10/p90 band is
    # user-facing). With impact_apy now globally scored via
    # GLOBAL_SCORE_OVERRIDES, per-position bands around a global point
    # estimate are mismatched; these replace them for overridden positions.
    quantile_models: dict[str, object] = {}
    if model_name == "impact_apy":
        X_full = full[feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y_full = full[target_col].astype(float).values
        # p90 only: the 2026-07-21 --quantile-audit validated the p90 head
        # (pinball -52.6% vs the unconditional baseline) but showed the p10
        # head LOSES to the constant training quantile (+16.8%) — on this
        # zero-inflated target the true p10 is ~0 for nearly everyone, so
        # step 10 floors p10 at 0 instead of fitting pseudo-signal.
        for alpha, suffix in ((0.9, "p90"),):
            qm = lgb.LGBMRegressor(**{**params, "n_estimators": n_best,
                                      "objective": "quantile", "alpha": alpha})
            qm.fit(X_full, y_full, sample_weight=w_full)
            quantile_models[suffix] = qm

    metrics["n_full_train"] = len(full)
    metrics["n_features"] = len(feature_cols)
    logger.info("  global/%s: cv_mae=%.4f best_iter=%d holdout_mae=%s n_full_train=%d n_features=%d",
                model_name, metrics["cv_mae"], n_best,
                f"{metrics.get('holdout_mae', float('nan')):.4f}",
                len(full), len(feature_cols))
    oof = (pd.concat(oof_records, ignore_index=True)
           if oof_records else pd.DataFrame())
    return model, metrics, feature_cols, oof, quantile_models


def _z(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 1e-12 else 1.0)


def _run_blend_ab(df: pd.DataFrame, feature_cols: list[str],
                  holdout: bool = False) -> None:
    """A/B: per-position head alone vs precision-weighted blend with the
    global model, per (position × target).

    Per expanding-window fold (holdout excluded): fit the global model once
    per target and each per-position head (both honest-ES, tuned params),
    predict the position's fold-val rows with both. Blend weight w is chosen
    LEAVE-ONE-FOLD-OUT from a 0..1 grid, so the evaluated fold never picks
    its own weight.

    Metrics: role/impact → MAE on raw-value blend (targets share units and
    populations). impact_wav → bridge-only Spearman on z-scored blend (the
    global head trains bridge-only, so raw values are scale-incompatible
    with the per-position mixture head).
    """
    from scipy.stats import spearmanr
    tuning_folds, holdout_fold = _p8._split_holdout(_p8._expanding_window_folds())
    if holdout:
        # One-time honest check on the untouched latest fold: dump preds only
        # (the LOFO verdict loop needs >=3 folds and is skipped); weights are
        # frozen upstream, chosen from tuning folds alone.
        if holdout_fold is None:
            logger.error("No holdout fold available"); return
        tuning_folds = [holdout_fold]
    W = np.round(np.arange(0.0, 1.01, 0.1), 2)
    tuned_g = {}
    if GLOBAL_TUNED_PARAMS_PATH.exists():
        tuned_g = json.load(open(GLOBAL_TUNED_PARAMS_PATH))
    results = []
    _all_fold_rows: dict[tuple, dict] = {}

    for target_col, model_name in TARGETS:
        g_cols = _global_filter_state_space_to_mi(feature_cols, model_name)
        g_params = {**LGBM_PARAMS, **tuned_g.get(f"global/{model_name}", {})}
        use_rho = model_name == "impact_wav"

        pos_data = {}
        for pos in POSITION_GROUPS.keys():
            d = df[df.get("position_group") == pos].copy()
            if d.empty:
                continue
            # Replicate step-8 target semantics: zero-pad non-bridge, drop
            # bridge rows with deliberately-NaN labels.
            nb = ~d["is_bridge"].astype(bool)
            d.loc[nb, target_col] = d.loc[nb, target_col].fillna(0.0)
            d = d[d[target_col].notna()]
            cols = _p8._build_feature_cols(d, pos)
            if len(d) < 50 or not cols:
                continue
            cols = _p8._filter_state_space_to_mi(cols, pos, model_name)
            cols = _p8._drop_dominant_value_features(cols, d, max_share=0.99)
            p_params = {**LGBM_PARAMS,
                        **(_p8._load_tuned_params(pos, model_name) or {})}
            pos_data[pos] = (d, cols, p_params)

        fold_rows: dict[str, list] = {pos: [] for pos in pos_data}
        _all_fold_rows[(target_col, model_name)] = fold_rows
        for train_end, val_start, val_end in tuning_folds:
            sub = df[df[target_col].notna()]
            yr_sub = sub["last_college_season_year"].fillna(0)
            g_tr = sub[yr_sub <= train_end]
            if len(g_tr) < 100:
                for pos in pos_data:
                    fold_rows[pos].append(None)
                continue
            mkg = lambda: lgb.LGBMRegressor(**{
                **g_params,
                **_p8._build_monotone_kwargs(g_cols, "__global__", "__global__")})
            gm, _ = _p8._fit_fold_honest(
                mkg, g_tr[g_cols].to_numpy(dtype=float, na_value=np.nan),
                g_tr[target_col].astype(float).values,
                g_tr["last_college_season_year"].fillna(0).values)
            for pos, (d, cols, p_params) in pos_data.items():
                yrd = d["last_college_season_year"].fillna(0)
                tr = d[yrd <= train_end]
                va = d[(yrd >= val_start) & (yrd <= val_end)]
                if len(tr) < 30 or len(va) < 10:
                    fold_rows[pos].append(None)
                    continue
                mkp = lambda cols=cols, p_params=p_params, pos=pos: lgb.LGBMRegressor(**{
                    **p_params,
                    **_p8._build_monotone_kwargs(cols, pos, model_name)})
                pm, _ = _p8._fit_fold_honest(
                    mkp, tr[cols].to_numpy(dtype=float, na_value=np.nan),
                    tr[target_col].astype(float).values,
                    tr["last_college_season_year"].fillna(0).values)
                fold_rows[pos].append({
                    "y":      va[target_col].astype(float).values,
                    "pos":    pm.predict(va[cols].to_numpy(dtype=float, na_value=np.nan)),
                    "glob":   gm.predict(va[g_cols].to_numpy(dtype=float, na_value=np.nan)),
                    "bridge": va["is_bridge"].astype(bool).values,
                    "_meta":  (val_start, val_end,
                               va["player_id"].astype(str).values),
                })

        for pos, folds in fold_rows.items():
            valid = [f for f in folds if f]
            if len(valid) < 3:
                continue

            def _loss(rows, w):
                if use_rho:
                    vals = []
                    for r in rows:
                        b = r["bridge"]
                        if b.sum() < 10 or np.std(r["y"][b]) < 1e-12:
                            continue
                        pred = w * _z(r["pos"]) + (1 - w) * _z(r["glob"])
                        rho = spearmanr(pred[b], r["y"][b]).statistic
                        if not np.isnan(rho):
                            vals.append(-float(rho))
                    return float(np.mean(vals)) if vals else float("inf")
                return float(np.mean([
                    np.mean(np.abs(w * r["pos"] + (1 - w) * r["glob"] - r["y"]))
                    for r in rows]))

            per_fold = []
            for i, r in enumerate(valid):
                others = [x for j, x in enumerate(valid) if j != i]
                w_star = min(W, key=lambda w: _loss(others, w))
                per_fold.append((
                    _loss([r], 1.0), _loss([r], 0.0), _loss([r], w_star), w_star))
            per_fold = [p for p in per_fold if np.isfinite(p[0]) and np.isfinite(p[2])]
            if not per_fold:
                continue
            l_pos = float(np.mean([p[0] for p in per_fold]))
            l_glob = float(np.mean([p[1] for p in per_fold]))
            l_blend = float(np.mean([p[2] for p in per_fold]))
            ws = [float(p[3]) for p in per_fold]
            rec = {"pos": pos, "target": model_name, "metric": "neg_rho" if use_rho else "mae",
                   "pos_only": l_pos, "global_only": l_glob, "blend_lofo": l_blend,
                   "delta_vs_pos_pct": (l_blend - l_pos) / abs(l_pos) * 100 if l_pos else float("nan"),
                   "lofo_weights": ws, "n_folds": len(per_fold)}
            results.append(rec)
            logger.info("  %s/%s [%s]: pos=%.5f glob=%.5f blend=%.5f (%+.2f%% vs pos) w=%s",
                        pos, model_name, rec["metric"], l_pos, l_glob, l_blend,
                        rec["delta_vs_pos_pct"], ws)

    out = EXPERIMENTS_DIR / f"blend_ab_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    wins = sum(1 for r in results if r["delta_vs_pos_pct"] < -1e-9)
    logger.info("Blend A/B: blend beats pos-only on %d/%d (pos,target) heads → %s",
                wins, len(results), out)

    # Persist raw fold predictions so metric questions (logloss, Spearman,
    # calibration) can be answered offline without refitting anything.
    dump = []
    for (target_col, model_name), rows_by_pos in _all_fold_rows.items():
        for pos, folds in rows_by_pos.items():
            for r in folds:
                if not r:
                    continue
                v_start, v_end, pids = r["_meta"]
                for i in range(len(r["y"])):
                    dump.append({
                        "target": model_name, "pos": pos,
                        "val_start": v_start, "val_end": v_end,
                        "player_id": pids[i], "y": r["y"][i],
                        "p_pos": r["pos"][i], "p_glob": r["glob"][i],
                        "is_bridge": bool(r["bridge"][i]),
                    })
    preds_path = EXPERIMENTS_DIR / (
        "blend_ab_holdout_preds.parquet" if holdout
        else "blend_ab_fold_preds.parquet")
    pd.DataFrame(dump).to_parquet(preds_path, index=False)
    logger.info("Fold predictions → %s (%d rows)", preds_path, len(dump))


def _pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1.0) * d)))


def _run_quantile_audit(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Validate the global impact_apy p10/p90 quantile heads (which borrow
    the point-estimate's tuned hyperparameters) on expanding-window tuning
    folds: pinball loss vs the unconditional training-quantile baseline,
    plus empirical coverage (share of fold-val outcomes beyond each bound;
    nominal 10% per side)."""
    target_col, model_name = "impact_target", "impact_apy"
    g_cols = _global_filter_state_space_to_mi(feature_cols, model_name)
    tuned_g = {}
    if GLOBAL_TUNED_PARAMS_PATH.exists():
        tuned_g = json.load(open(GLOBAL_TUNED_PARAMS_PATH))
    params = {**LGBM_PARAMS, **tuned_g.get(f"global/{model_name}", {})}
    tuning_folds, _ = _p8._split_holdout(_p8._expanding_window_folds())
    yr = df["last_college_season_year"].fillna(0)

    results = []
    for train_end, val_start, val_end in tuning_folds:
        tr = df[(yr <= train_end) & df[target_col].notna()]
        va = df[(yr >= val_start) & (yr <= val_end) & df[target_col].notna()]
        if len(tr) < 100 or len(va) < 30:
            continue
        X_tr = tr[g_cols].to_numpy(dtype=float, na_value=np.nan)
        y_tr = tr[target_col].astype(float).values
        X_va = va[g_cols].to_numpy(dtype=float, na_value=np.nan)
        y_va = va[target_col].astype(float).values
        # Honest n_estimators for quantile fits: reuse the point head's
        # inner-ES best_iter on this fold.
        _make = lambda: lgb.LGBMRegressor(**{
            **params, **_p8._build_monotone_kwargs(g_cols, "__global__",
                                                   "__global__")})
        _, n_best = _p8._fit_fold_honest(
            _make, X_tr, y_tr, tr["last_college_season_year"].fillna(0).values)
        rec = {"train_end": train_end, "n_val": int(len(va))}
        for alpha, suffix in ((0.1, "p10"), (0.9, "p90")):
            qm = lgb.LGBMRegressor(**{**params, "n_estimators": n_best,
                                      "objective": "quantile", "alpha": alpha})
            qm.fit(X_tr, y_tr)
            q = np.clip(qm.predict(X_va), 0.0, None)
            const = float(np.quantile(y_tr, alpha))
            rec[f"{suffix}_pinball"] = _pinball(y_va, q, alpha)
            rec[f"{suffix}_pinball_const"] = _pinball(
                y_va, np.full_like(y_va, const), alpha)
            rec[f"{suffix}_exceed"] = (float(np.mean(y_va < q)) if alpha == 0.1
                                       else float(np.mean(y_va > q)))
        results.append(rec)
        logger.info("  fold ≤%d: p10 pinball %.6f (const %.6f) below-p10 %.1f%% | "
                    "p90 pinball %.6f (const %.6f) above-p90 %.1f%%",
                    train_end, rec["p10_pinball"], rec["p10_pinball_const"],
                    rec["p10_exceed"] * 100, rec["p90_pinball"],
                    rec["p90_pinball_const"], rec["p90_exceed"] * 100)

    if not results:
        logger.error("No valid folds for quantile audit")
        return
    mean = {k: float(np.mean([r[k] for r in results]))
            for k in results[0] if k != "train_end"}
    out = EXPERIMENTS_DIR / f"quantile_audit_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"folds": results, "mean": mean}, indent=2))
    logger.info("QUANTILE AUDIT mean: p10 pinball %.6f vs const %.6f (%+.1f%%), "
                "coverage below-p10 %.1f%% (nominal 10) | p90 pinball %.6f vs "
                "const %.6f (%+.1f%%), coverage above-p90 %.1f%% (nominal 10) → %s",
                mean["p10_pinball"], mean["p10_pinball_const"],
                (mean["p10_pinball"] / mean["p10_pinball_const"] - 1) * 100,
                mean["p10_exceed"] * 100,
                mean["p90_pinball"], mean["p90_pinball_const"],
                (mean["p90_pinball"] / mean["p90_pinball_const"] - 1) * 100,
                mean["p90_exceed"] * 100, out)


def main():
    if not _lgbm_available:
        logger.error("lightgbm not installed."); sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-trials", type=int, default=30)
    parser.add_argument("--tune-seeds", type=int, default=1,
                        help="Independent Optuna seeds per head; winner is "
                             "the seed with the lowest CV.")
    parser.add_argument("--boruta", action="store_true",
                        help="Run Boruta as an AUDIT pass (writes "
                             "boruta_features.json under global_draft_models/). "
                             "Does not gate the fit — drop candidates are "
                             "promoted manually into config.BORUTA_DROP_FEATURES.")
    parser.add_argument("--boruta-trials", type=int, default=BORUTA_MAX_ITER,
                        help="Boruta max_iter per head (default: BORUTA_MAX_ITER)")
    parser.add_argument("--mi-filter", action="store_true",
                        help="Compute MI ranking on the PFF state-space block "
                             "and cache top-N per target. Subsequent fits filter "
                             "state-space cols to the cached subset.")
    parser.add_argument("--quantile-audit", action="store_true",
                        help="Validate the global impact_apy p10/p90 heads "
                             "(pinball loss + coverage) and exit")
    parser.add_argument("--blend-holdout", action="store_true",
                        help="Dump per-position + global predictions for the "
                             "held-out latest fold only (one-time honest check)")
    parser.add_argument("--blend-ab", action="store_true",
                        help="Run the per-position vs global-blend A/B "
                             "(LOFO weight selection) and exit — no models written")
    parser.add_argument("--mi-top-n", type=int, default=MI_TOP_N,
                        help="Top-N state-space cols to keep per head (default: MI_TOP_N)")
    args = parser.parse_args()

    labels_path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    if not labels_path.exists():
        logger.error("%s missing — run earlier steps", labels_path); sys.exit(1)
    df, feature_cols = build_global_training_frame(pd.read_parquet(labels_path))

    if args.quantile_audit:
        logger.info("Running quantile audit (no models will be written)...")
        _run_quantile_audit(df, feature_cols)
        return

    if args.blend_ab or args.blend_holdout:
        logger.info("Running blend %s (no models will be written)...",
                    "HOLDOUT check" if args.blend_holdout else "A/B")
        _run_blend_ab(df, feature_cols, holdout=args.blend_holdout)
        return

    # MI filter runs FIRST so the cache is on disk by the time tuning + final
    # fit call _global_filter_state_space_to_mi.
    if args.mi_filter:
        logger.info("Running global MI feature filter (top_n=%d)...", args.mi_top_n)
        _run_mi_for_global_heads(df, feature_cols, args.mi_top_n)

    # Boruta is an audit pass — writes boruta_features.json under
    # global_draft_models/ for experiments/boruta_summary.py. Does NOT gate
    # the fit; drop decisions live in config.BORUTA_DROP_FEATURES.
    if args.boruta:
        logger.info("Running global Boruta audit pass (max_iter=%d)...",
                    args.boruta_trials)
        _run_boruta_for_global_heads(df, feature_cols, args.boruta_trials)

    # Fingerprint that captures everything that, if changed, invalidates the
    # cached global hyperparams: feature space, training-data shape,
    # hyperparameter defaults, and the position-group ordering.
    global_fingerprint = {
        "feature_cols_hash":      hash_features(feature_cols),
        "n_features":             len(feature_cols),
        "snapshot_shape":         schema_fingerprint(df, season_col="season_year"),
        "lgbm_defaults_hash":     hash_config(LGBM_PARAMS),
        "position_group_ordinal": sorted(POSITION_GROUPS),
    }

    # Tuning.
    tuned: dict[str, dict] = {}
    if GLOBAL_TUNED_PARAMS_PATH.exists():
        try:
            tuned = json.load(open(GLOBAL_TUNED_PARAMS_PATH))
        except Exception:
            tuned = {}
        # Staleness check: surface diffs so the user sees what changed.
        if tuned:
            result = check_manifest(GLOBAL_TUNED_PARAMS_PATH, fingerprint=global_fingerprint)
            if result.is_stale:
                log_stale_warning(
                    GLOBAL_TUNED_PARAMS_PATH,
                    "python run_pipeline.py --retune-only 8.5",
                    diffs=result.diffs,
                )
    if args.tune:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            optuna = None

        for target_col, model_name in TARGETS:
            sub = df[df[target_col].notna()].copy()
            tuning_folds, _ = _p8._split_holdout(_p8._expanding_window_folds())
            # Filter to MI-selected state-space subset for this target (no-op
            # when no cache). Per-target, since MI rankings differ by target.
            target_feature_cols = _global_filter_state_space_to_mi(feature_cols, model_name)
            seed_h = hashlib.sha256(f"global/{model_name}".encode()).digest()
            base_seed = 42 + int.from_bytes(seed_h[:4], "big")
            if optuna is None:
                continue
            # Fold arrays are trial-invariant — build once per target,
            # reuse across all trials and seeds.
            fold_data = _p8._build_fold_data(sub, target_feature_cols,
                                             target_col, tuning_folds)
            if not fold_data:
                continue
            # Multi-seed: run N independent Optuna studies, pick the seed
            # with the lowest CV. Each seed's best_value is deterministic
            # (LightGBM random_state fixed) so min() across seeds is unbiased.
            seed_results: list[tuple[int, float, dict]] = []
            for seed_idx in range(max(1, args.tune_seeds)):
                model_seed = base_seed + seed_idx * 1009
                sampler = optuna.samplers.TPESampler(seed=model_seed)
                study = optuna.create_study(direction="minimize", sampler=sampler)
                best = {"score": float("inf")}
                seed_label = f"seed {seed_idx+1}/{args.tune_seeds} " if args.tune_seeds > 1 else ""

                def _obj(trial):
                    params = _p8._suggest_optuna_params(trial)
                    # pos/model must be "__global__" so _build_monotone_kwargs
                    # hits the MONOTONE_ENABLED gate and returns {} — the same
                    # unconstrained regime _train_one_global fits in. Metric
                    # mirrors step 8 (logloss for the soft role target; mae
                    # for the impact heads — the spearman objective was tried
                    # on 2026-07-27 and reverted, see 08_draft_models).
                    score, _n = _p8._evaluate_params(
                        params, fold_data, target_feature_cols,
                        metric="logloss" if model_name == "role" else "mae",
                        pos_group="__global__", model_name="__global__",
                    )
                    if score is None:
                        raise optuna.TrialPruned()
                    if score < best["score"]:
                        best["score"] = score
                        logger.info("    %sglobal/%s trial %d: NEW BEST mae=%.4f",
                                    seed_label, model_name, trial.number + 1, score)
                    return score

                study.optimize(_obj, n_trials=args.tune_trials,
                               show_progress_bar=False, catch=(Exception,),
                               n_jobs=-1)
                if study.best_trial:
                    seed_results.append((seed_idx,
                                         float(study.best_trial.value),
                                         dict(study.best_trial.params)))
            if not seed_results:
                continue
            seed_results.sort(key=lambda x: x[1])
            winner_idx, winner_score, winner_params = seed_results[0]
            tuned[f"global/{model_name}"] = winner_params
            if args.tune_seeds > 1:
                scores = [r[1] for r in seed_results]
                logger.info("  global/%s multi-seed (n=%d): best=%.4f (seed %d), "
                            "worst=%.4f, spread=%.4f",
                            model_name, len(seed_results),
                            min(scores), winner_idx + 1, max(scores),
                            max(scores) - min(scores))
            logger.info("  global/%s tuned: %s", model_name, winner_params)
        GLOBAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(GLOBAL_TUNED_PARAMS_PATH, "w") as f:
            json.dump(tuned, f, indent=2)
        write_manifest(
            GLOBAL_TUNED_PARAMS_PATH, fingerprint=global_fingerprint,
            mode_used="fresh", step_name=STEP_NAME,
        )
        logger.info("Saved %d tuned global param sets", len(tuned))

    # Train each global head.
    GLOBAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}
    # Each model carries its own (MI-filtered) feature_cols, so predict
    # has to slice X with the right subset per head.
    models: dict[str, tuple[object, list[str]]] = {}
    oof_frames: list[pd.DataFrame] = []
    for target_col, model_name in TARGETS:
        # Pass the FULL df — _train_one_global drops NaN-target rows at every
        # fit internally, and the OOF collection needs the full fold-val
        # cohort (incl. non-bridge impact_wav rows) for 8c's mixture-scale
        # per-position calibrators.
        tp = tuned.get(f"global/{model_name}")
        try:
            model, metrics, fc_used, oof, quantile_models = _train_one_global(
                df, feature_cols, target_col, model_name, tp,
                test_mode=args.test,
            )
            bundle = {"model": model, "feature_cols": fc_used,
                      "target": target_col}
            if quantile_models:
                bundle["quantile_models"] = quantile_models
            out_path = GLOBAL_MODELS_DIR / f"global_{model_name}_model.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(bundle, f)
            all_metrics[f"global_{model_name}"] = metrics
            models[model_name] = (model, fc_used)
            if not oof.empty:
                oof_frames.append(oof)
            logger.info("  Saved → %s", out_path)
        except ValueError as e:
            logger.warning("  Skipped global/%s: %s", model_name, e)

    # Persist tuning-fold OOF predictions (holdout fold deliberately
    # excluded — it must stay untouched by calibration). Consumed by step
    # 8c's per-position global calibrators and offline blend analyses.
    if oof_frames:
        oof_all = pd.concat(oof_frames, ignore_index=True)
        oof_path = GLOBAL_MODELS_DIR / "global_oof_predictions.parquet"
        oof_all.to_parquet(oof_path, index=False)
        logger.info("Saved %d global OOF predictions → %s", len(oof_all), oof_path)

    # Score all players. Each head uses its own MI-filtered feature_cols.
    pred_records = {"player_id": df["player_id"].values}
    for model_name, (model, fc_used) in models.items():
        X_all = df[fc_used].to_numpy(dtype=float, na_value=np.nan)
        pred_records[f"global_{model_name}_score"] = model.predict(X_all)
    preds_df = pd.DataFrame(pred_records)
    # Clip role to [0,1]; impact heads to [0, ∞).
    if "global_role_score" in preds_df.columns:
        preds_df["global_role_score"] = preds_df["global_role_score"].clip(0.0, 1.0)
    for c in ("global_impact_apy_score", "global_impact_wav_score"):
        if c in preds_df.columns:
            preds_df[c] = preds_df[c].clip(lower=0.0)
    preds_df.to_parquet(GLOBAL_PREDICTIONS_PATH, index=False)
    logger.info("Saved global predictions for %d players → %s",
                len(preds_df), GLOBAL_PREDICTIONS_PATH)

    metrics_path = GLOBAL_MODELS_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved global training metrics → %s", metrics_path)

    # Provenance sidecar (mirrors step 8's pattern).
    metrics_mode = "fresh" if args.tune else "cached"
    provenance_path = GLOBAL_MODELS_DIR / "training_metrics_provenance.json"
    provenance_payload = tag_summary(
        {"n_models": len(all_metrics), "tune_flag": bool(args.tune),
         "tune_trials": int(args.tune_trials) if args.tune else None},
        mode_used=metrics_mode, step_name=STEP_NAME,
    )
    provenance_path.write_text(json.dumps(provenance_payload, indent=2))
    logger.info("Provenance → %s (_tuning_mode=%s)", provenance_path, metrics_mode)

    # Append to experiment log.
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_snapshot = {
        "ROLE_LABEL_HORIZON_YEARS": ROLE_LABEL_HORIZON_YEARS,
        "DRAFT_TRAIN_CUTOFF_YEAR": DRAFT_TRAIN_CUTOFF_YEAR,
        "DRAFT_VAL_START_YEAR": DRAFT_VAL_START_YEAR,
        "FINAL_TRAIN_CUTOFF": _FINAL_TRAIN_CUTOFF,
        "n_features": len(feature_cols),
        "LGBM_PARAMS_DEFAULTS": LGBM_PARAMS,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_snapshot, sort_keys=True).encode()
    ).hexdigest()[:12]
    exp = {
        "run_id": f"{ts}_global_{config_hash}",
        "timestamp": ts,
        "kind": "global_draft_model",
        "config_hash": config_hash,
        "config": config_snapshot,
        "args": {"tune": bool(args.tune), "tune_trials": int(args.tune_trials)},
        "models": all_metrics,
    }
    exp_path = EXPERIMENTS_DIR / f"{ts}_global_{config_hash}.json"
    with open(exp_path, "w") as f:
        json.dump(exp, f, indent=2)
    logger.info("Saved experiment log → %s", exp_path)
    logger.info("Step 8b complete.")


if __name__ == "__main__":
    main()
