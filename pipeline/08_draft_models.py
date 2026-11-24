"""
Step 8: Draft Models (Role + Impact)

Trains two LightGBM models per position group:
  1. Role Model: continuous role label produced by Step 7 — the trajectory
     model's role probability evaluated at a fixed end-of-y3 NFL snapshot
     for every bridge player. Bounded to [0, 1] but treated as regression.
  2. Impact Model: expected second contract APY as fraction of salary cap
     at signing.

Features:
  - Kalman/EWMA skill snapshot
  - Physical attributes (height, weight, class year, conference tier)
  - Team SP+ offense rating (OL quality proxy)
  - Recruiting composite rating
  - WR/TE: conference-adjusted receiving interaction features
  - DL/LB: pass rush composite + team sack share
  - OL: team SP+ offense as primary signal

Training: bridge players with last_college_season_year ≤ DRAFT_TRAIN_CUTOFF_YEAR
Validation: expanding-window CV across all years (see _expanding_window_folds);
the latest fold is reserved post-hoc for honest reporting.

Output: data/processed/draft_models/{pos_group}_{role|impact}_model.pkl
"""

import argparse
import hashlib
import json
import logging
import pickle
import sys
from datetime import date as dt_date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))   # so `from _boruta_runner import ...` resolves
from config import (
    PROCESSED_DIR, MODELED_STATS, POSITION_GROUPS, PLAY_SKILL_STAT_POSITIONS,
    END_YEAR, START_YEAR,
    LGBM_PARAMS, PFF_GRADE_FEATURES_BY_GROUP, NFL_LG_FEATURES, OPP_CAREER_FEATURES,
    COMBINE_RAW_BY_GROUP, SPEED_SCORE_POSITIONS, RETURN_FEATURE_POSITIONS,
    HASAN_FEATURES,
    DRAFT_TRAIN_CUTOFF_YEAR, DRAFT_VAL_START_YEAR, ROLE_LABEL_HORIZON_YEARS,
    MONOTONE_CONSTRAINTS, MONOTONE_ENABLED,
    PFF_DENOM_STATS, _PFF_DENOM_RELEVANCE,
    BORUTA_N_ESTIMATORS, BORUTA_MAX_ITER, BORUTA_PERCENTILE,
    BORUTA_ALPHA, BORUTA_RANDOM_STATE, BORUTA_DROP_FEATURES,
    MI_TOP_N, MI_RANDOM_STATE, MI_N_NEIGHBORS,
    QB_ROLLING_FEATURES, QB_CAREER_BASE_FEATURES,
    QB_CAREER_POSITION_GROUPS, USE_QB_CAREER_FEATURES,
    IMPACT_LABEL_MIN_DRAFT_YEAR, IMPACT_BOARD_ROW_WEIGHT,
)
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

# Objective used to select hyperparameters, per head family. Ranking replaced
# MAE for the impact heads on 2026-07-26: MAE is unsafe as an OBJECTIVE on
# these zero-inflated targets because shrinking predictions toward zero
# improves it, so it rewards over-regularised params. role keeps logloss — its
# target is a probability and logloss is a proper scoring rule for one.
# This tag is part of the tuned-params fingerprint, so changing the objective
# correctly invalidates caches rather than silently reusing params chosen
# against the old metric.
TUNING_OBJECTIVE_TAG = "role=logloss;impact=spearman"
# Shared snapshot assembly (attrs / age / sack-share / PFF / QB / combine /
# hasan / league-averages / derived features). Single source of truth for the
# scoring matrix across steps 8 / 8b / 8c / 9 / 10.
from _snapshot_assembly import (
    assemble_snapshot,
    add_derived_features as _add_derived_features,
)
from src.utils.cache_manifest import (
    hash_features, hash_config, log_stale_warning, manifest_path_for,
    check_manifest, write_manifest, tag_summary,
)
from _boruta_runner import run_boruta, select_features
from src.utils.mi_filter import rank_by_mi

STEP_NAME = "08_draft_models"

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


ATTRIBUTE_FEATURES = ["height_in", "weight_lbs", "class_year_int",
                       "conference_tier",
                       "team_sp_offense", "team_sp_defense", "team_sp_overall",
                       # Career team-context aggregates from advanced_game_stats
                       "career_team_off_ppa_mean", "career_team_off_success_rate_mean",
                       "career_team_off_explosiveness_mean", "career_team_off_stuff_rate_mean",
                       "career_team_def_ppa_mean", "career_team_def_success_rate_mean",
                       "career_team_def_stuff_rate_mean", "career_team_def_explosiveness_mean",
                       # Team rushing four-factor + havoc career means (step 03)
                       "career_team_line_yards_avg_mean",
                       "career_team_second_level_yards_avg_mean",
                       "career_team_open_field_yards_avg_mean",
                       "career_team_rushing_power_success_mean",
                       "career_team_rushing_stuff_rate_mean",
                       "career_team_havoc_total_mean",
                       "career_team_havoc_front_seven_mean",
                       "career_team_havoc_db_mean",
                       "recruiting_stars", "recruiting_composite",
                       "recruit_rank_national", "recruit_rank_position",
                       "age_at_draft", "n_transfers_career"]

# Final model trains on everything up to (but not including) the current scoring class
_FINAL_TRAIN_CUTOFF = END_YEAR - 2   # e.g. 2024 when END_YEAR=2026

TUNED_PARAMS_PATH          = PROCESSED_DIR / "draft_models" / "tuned_params.json"
BORUTA_FEATURES_PATH       = PROCESSED_DIR / "draft_models" / "boruta_features.json"
MI_FEATURES_PATH           = PROCESSED_DIR / "draft_models" / "mi_features.json"


def _load_tuned_params(pos_group: str, model_name: str) -> dict | None:
    """Load tuned hyperparameters for one (pos × model) from tuned_params.json.
    (Historically a second tuned_params_monotone.json existed for the
    MONOTONE_ENABLED set, but it was always written byte-identical to the
    main file — collapsed to a single file 2026-07-21. Whether monotone
    constraints apply at fit time is decided by `_use_monotone_for`.)
    """
    path = TUNED_PARAMS_PATH
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
            params = data.get(f"{pos_group}/{model_name}")
            # Fallback: impact_wav reuses impact's tuned params if it has none
            # of its own (similar continuous regression target on same features).
            if params is None and model_name == "impact_wav":
                params = data.get(f"{pos_group}/impact")
            return params
    except Exception:
        return None


def _use_monotone_for(pos_group: str, model_name: str) -> bool:
    """Whether to apply MONOTONE_CONSTRAINTS at fit time for this model.
    Single source of truth — MUST agree with `_load_tuned_params` so each
    model uses params + constraints from the same training regime."""
    return (pos_group, model_name) in MONOTONE_ENABLED


def _build_tuned_params_fingerprint(snapshot: pd.DataFrame,
                                     labels: pd.DataFrame,
                                     keys: list[str]) -> dict:
    """Build the per-key + global-context fingerprint for the step-8 cache.

    `by_key` maps `"{pos}/{model}"` → per-key dict (feature-cols hash,
    monotone flag, snapshot row count). `global_context` carries the
    LGBM defaults hash and training-cutoff config that affects all keys.

    Feature cols MUST resolve through `_resolve_fit_features` — the same
    path the load-time staleness check hashes — otherwise the fingerprint
    is written pre-filter but checked post-filter and every head flags
    stale forever.
    """
    by_key: dict[str, dict] = {}
    for key in keys:
        try:
            pos_group, model_name = key.split("/", 1)
        except ValueError:
            continue
        target_col = MODEL_TARGET_COLS.get(model_name)
        if target_col is None:
            continue
        try:
            cols, _df = _resolve_fit_features(snapshot, labels, pos_group,
                                              target_col, model_name)
        except Exception:
            continue
        if not cols:
            continue
        by_key[key] = {
            "feature_cols_hash": hash_features(cols),
            "monotone_for_key":  _use_monotone_for(pos_group, model_name),
            "snapshot_n_rows":   int(len(snapshot)),
        }
    return {
        "by_key": by_key,
        "global_context": {
            "lgbm_defaults_hash":      hash_config(LGBM_PARAMS),
            "draft_train_cutoff_year": DRAFT_TRAIN_CUTOFF_YEAR,
            # Params tuned under a different objective are not interchangeable,
            # so the objective belongs in the fingerprint — otherwise switching
            # it silently reuses params selected against the old metric.
            "tuning_objective":        TUNING_OBJECTIVE_TAG,
        },
    }


def _check_tuned_params_staleness(snapshot: pd.DataFrame,
                                   pos_group: str, model_name: str,
                                   feature_cols: list[str]) -> None:
    """On load, warn if the cached fingerprint for this (pos, model) no longer
    matches the current feature set / monotone flag / global context. Filters
    diffs to entries prefixed `by_key.{pos}/{model}.` so each model's load
    only surfaces its own drift; `global_context.*` diffs are surfaced once.
    """
    path = TUNED_PARAMS_PATH
    if not path.exists():
        return
    key = f"{pos_group}/{model_name}"
    current = {
        "by_key": {key: {
            "feature_cols_hash": hash_features(feature_cols),
            "monotone_for_key":  _use_monotone_for(pos_group, model_name),
            "snapshot_n_rows":   int(len(snapshot)),
        }},
        "global_context": {
            "lgbm_defaults_hash":      hash_config(LGBM_PARAMS),
            "draft_train_cutoff_year": DRAFT_TRAIN_CUTOFF_YEAR,
            # Params tuned under a different objective are not interchangeable,
            # so the objective belongs in the fingerprint — otherwise switching
            # it silently reuses params selected against the old metric.
            "tuning_objective":        TUNING_OBJECTIVE_TAG,
        },
    }
    result = check_manifest(path, fingerprint=current)
    if result.is_stale:
        # Only show diffs that pertain to this specific (pos, model) plus any
        # global_context drift (which affects every model).
        prefix = f"by_key.{key}."
        relevant = [d for d in result.diffs
                    if d.startswith(prefix) or d.startswith("global_context.")]
        log_stale_warning(
            f"{path}  ({key})",
            "python run_pipeline.py --retune-only 8",
            diffs=relevant or result.diffs,
        )


def _write_tuned_params_manifest(snapshot: pd.DataFrame, labels: pd.DataFrame,
                                  tuned: dict, paths: list[Path]) -> None:
    """Write a manifest sidecar for each tuned_params*.json path. Sidecar
    captures the {by_key, global_context} fingerprint so a future fast run
    can detect drift at either granularity."""
    fingerprint = _build_tuned_params_fingerprint(snapshot, labels,
                                                  list(tuned.keys()))
    for p in paths:
        write_manifest(
            p, fingerprint=fingerprint,
            mode_used="fresh", step_name=STEP_NAME,
            extra={"n_keys": len(fingerprint["by_key"])},
        )


# ── Boruta audit cache ────────────────────────────────────────────────────────
# `boruta_features.json` is keyed by f"{pos}/{model}" → per-head verdict block:
#   {confirmed: [...], tentative: [...], n_features_in, n_features_after,
#    boruta_train_end, max_iter, percentile, alpha}
# Plus a `_meta` block (snapshot row count, boruta_train_end). Manifest sidecar
# mirrors the tuned_params staleness pattern.
#
# This file is for AUDITING ONLY — it does NOT gate fit-time feature_cols.
# `experiments/boruta_summary.py` reads it to surface globally-rejected
# features, which the user then promotes into config.BORUTA_DROP_FEATURES
# so they're stripped at the source by `_build_feature_cols`.

def _build_boruta_fingerprint(snapshot: pd.DataFrame,
                              by_key_meta: dict[str, dict]) -> dict:
    """Per-head + global-context fingerprint for the Boruta cache. Mirrors the
    tuned_params shape so drift surfaces at the same granularity."""
    by_key: dict[str, dict] = {}
    for key, meta in by_key_meta.items():
        by_key[key] = {
            "feature_cols_hash": meta["feature_cols_hash"],
            "n_features_in":     int(meta["n_features_in"]),
            "n_features_after":  int(meta["n_features_after"]),
        }
    return {
        "by_key": by_key,
        "global_context": {
            "snapshot_n_rows":  int(len(snapshot)),
            "boruta_max_iter":  int(BORUTA_MAX_ITER),
            "boruta_percentile": int(BORUTA_PERCENTILE),
            "boruta_alpha":     float(BORUTA_ALPHA),
        },
    }


def _save_boruta_features(snapshot: pd.DataFrame,
                          by_key_payload: dict[str, dict],
                          by_key_meta: dict[str, dict],
                          *, max_iter: int, boruta_train_end: int) -> None:
    """Write boruta_features.json plus a manifest sidecar. `by_key_payload`
    is the per-head verdict block (confirmed/tentative/etc.) — what readers
    consume. `by_key_meta` carries the fingerprint inputs for staleness."""
    BORUTA_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(by_key_payload)
    payload["_meta"] = {
        "snapshot_n_rows":  int(len(snapshot)),
        "boruta_train_end": int(boruta_train_end),
        "max_iter":         int(max_iter),
        "percentile":       int(BORUTA_PERCENTILE),
        "alpha":            float(BORUTA_ALPHA),
    }
    with open(BORUTA_FEATURES_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    fingerprint = _build_boruta_fingerprint(snapshot, by_key_meta)
    write_manifest(
        BORUTA_FEATURES_PATH, fingerprint=fingerprint,
        mode_used="fresh", step_name=STEP_NAME,
        extra={"n_keys": len(by_key_payload)},
    )
    logger.info("Saved Boruta features for %d heads → %s",
                len(by_key_payload), BORUTA_FEATURES_PATH)


def _run_boruta_for_all_heads(snapshot: pd.DataFrame, labels: pd.DataFrame,
                              max_iter: int) -> None:
    """Run Boruta for all (pos × target) heads and persist the audit cache.

    Boruta's training matrix is `df[yr <= holdout_train_end]` so the holdout
    fold's val window stays uncontaminated. The verdict cache is read by
    `experiments/boruta_summary.py` to surface globally-rejected features —
    NOT by tune/fit. The user promotes drop candidates into
    `config.BORUTA_DROP_FEATURES`, which `_build_feature_cols` honors.
    """
    all_folds = _expanding_window_folds()
    _, holdout_fold = _split_holdout(all_folds)
    boruta_train_end = (holdout_fold[0] if holdout_fold is not None
                        else _FINAL_TRAIN_CUTOFF)
    logger.info("Boruta training window: last_college_season_year ≤ %d",
                boruta_train_end)

    by_key_payload: dict[str, dict] = {}
    by_key_meta: dict[str, dict] = {}

    for pos_group in POSITION_GROUPS.keys():
        feature_cols_full = _build_feature_cols(snapshot, pos_group)
        if not feature_cols_full:
            logger.warning("  %s: no features available, skipping all Boruta heads",
                           pos_group)
            continue

        for target_col, model_name in HEADS:
            df = _prep_training_df(snapshot, labels, pos_group, target_col)
            yr = df["last_college_season_year"].fillna(0)
            df_train = df[yr <= boruta_train_end]
            n_train = len(df_train)
            if n_train < 50:
                logger.warning("  %s/%s: only %d rows ≤ %d — skipping Boruta",
                               pos_group, model_name, n_train, boruta_train_end)
                continue

            X = df_train[feature_cols_full]
            y = df_train[target_col].fillna(0.0)

            # Drop all-constant columns before Boruta — they have zero
            # variance, RF rejects them with a warning and they'd land in
            # rejected anyway. Cleaner to filter pre-fit.
            non_constant = [c for c in feature_cols_full
                            if X[c].nunique(dropna=True) > 1]
            n_dropped_const = len(feature_cols_full) - len(non_constant)
            if n_dropped_const:
                logger.info("  %s/%s: dropped %d all-constant cols pre-Boruta",
                            pos_group, model_name, n_dropped_const)

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
            logger.info("  %s/%s: Boruta confirmed=%d  tentative=%d  rejected=%d  (in=%d)",
                        pos_group, model_name,
                        len(confirmed), len(tentative), len(verdict["rejected"]),
                        len(feature_cols_full))

            by_key_payload[f"{pos_group}/{model_name}"] = {
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
            by_key_meta[f"{pos_group}/{model_name}"] = {
                "feature_cols_hash": hash_features(feature_cols_full),
                "n_features_in":     len(feature_cols_full),
                "n_features_after":  n_after,
            }

    if by_key_payload:
        _save_boruta_features(
            snapshot, by_key_payload, by_key_meta,
            max_iter=max_iter, boruta_train_end=boruta_train_end,
        )
    else:
        logger.warning("Boruta produced no verdicts — nothing to write")


def _sample_hyperparams(rng: np.random.Generator) -> dict:
    """Random-search fallback when Optuna isn't available."""
    return {
        "learning_rate":     float(10 ** rng.uniform(-2, -1)),
        "max_depth":         int(rng.integers(3, 9)),
        "num_leaves":        int(rng.integers(8, 64)),
        "min_child_samples": int(rng.integers(5, 50)),
        "subsample":         float(rng.uniform(0.6, 1.0)),
        "colsample_bytree":  float(rng.uniform(0.6, 1.0)),
        "reg_alpha":         float(10 ** rng.uniform(-3, 0)),
        "reg_lambda":        float(10 ** rng.uniform(-3, 1)),
    }


def _build_monotone_kwargs(features: list[str],
                            pos_group: str | None = None,
                            model_name: str | None = None) -> dict:
    """Translate config.MONOTONE_CONSTRAINTS into LightGBM kwargs aligned to
    the exact feature_cols order. Returns {} when:
      - the (pos_group, model_name) is not in MONOTONE_ENABLED (when both are
        provided), or
      - no listed feature appears in the model's feature_cols.

    Skipping is a no-op vs. the unconstrained default (LightGBM's
    `monotone_constraints` is only honored if passed). Pattern lifted from
    cdarko/pipeline/09c_nba_draft_models.py:_build_monotone_list.
    """
    if pos_group is not None and model_name is not None:
        if (pos_group, model_name) not in MONOTONE_ENABLED:
            return {}
    constraints = [MONOTONE_CONSTRAINTS.get(f, 0) for f in features]
    if not any(c != 0 for c in constraints):
        return {}
    return {
        "monotone_constraints":        constraints,
        "monotone_constraints_method": "advanced",
    }


def _suggest_optuna_params(trial) -> dict:
    """Optuna search space — wider than random search, log-uniform on
    learning_rate / regularization, includes min_split_gain."""
    return {
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "num_leaves":        trial.suggest_int("num_leaves", 8, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 100.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 1e-4, 1.0, log=True),
    }


def _expanding_window_folds(val_window: int = 3, step: int = 3,
                             min_train_years: int = 6) -> list[tuple[int, int, int]]:
    """
    Return (train_end, val_start, val_end) tuples for expanding-window CV.
    Non-overlapping val windows of `val_window` years, stepping back by `step`
    from _FINAL_TRAIN_CUTOFF. Earliest fold must have >= min_train_years of training data.
    Returned chronologically: oldest fold first, latest fold last.
    """
    folds = []
    val_end = _FINAL_TRAIN_CUTOFF
    while True:
        val_start = val_end - val_window + 1
        train_end = val_start - 1
        if train_end - START_YEAR + 1 < min_train_years:
            break
        folds.append((train_end, val_start, val_end))
        val_end -= step
    return list(reversed(folds))


def _split_holdout(folds: list[tuple[int, int, int]]
                   ) -> tuple[list[tuple[int, int, int]], tuple[int, int, int] | None]:
    """
    Split folds into (tuning_folds, holdout_fold).

    The latest fold is reserved as a one-shot generalization check that no
    inner-CV / hyperparameter-search step ever sees. Returns (folds[:-1], folds[-1]).
    If only one fold exists, it is used as both holdout and (degenerate) tuning.
    """
    if not folds:
        return [], None
    if len(folds) == 1:
        return folds, folds[0]
    return folds[:-1], folds[-1]


# NOTE: `_add_derived_features` now lives in pipeline/_snapshot_assembly.py
# (imported above as an alias) so steps 8 / 8b / 8c / 9 / 10 share one
# implementation. It is still exposed under this name for tests and
# downstream importers (`_p8._add_derived_features`).


def _build_feature_cols(snapshot: pd.DataFrame, pos_group: str) -> list[str]:
    """
    Build feature column list for a given position group.
    Includes Kalman/EWMA skill features, physical attributes, and derived features.
    """
    stats = MODELED_STATS.get(pos_group, [])

    skill_cols = []
    for stat in stats:
        for prefix in ("kalman_mean", "kalman_var", "ewma", "ewma_var",
                        "kalman_yoy", "ewma_yoy"):
            col = f"{prefix}_{stat}"
            if col in snapshot.columns:
                skill_cols.append(col)

    attr_cols = [c for c in ATTRIBUTE_FEATURES if c in snapshot.columns]

    if "age" in snapshot.columns:
        attr_cols.append("age")
    if "games_played_career" in snapshot.columns:
        attr_cols.append("games_played_career")
    for col in ("n_games", "n_seasons"):
        if col in snapshot.columns:
            attr_cols.append(col)

    # Derived features
    derived_cols = []
    if pos_group in ("DL", "EDGE", "LB"):
        for col in ("pass_rush_composite", "team_sack_share", "pff_pass_rush_snap_pct"):
            if col in snapshot.columns:
                derived_cols.append(col)
        if pos_group == "LB" and "pff_coverage_snap_pct" in snapshot.columns:
            derived_cols.append("pff_coverage_snap_pct")
    if pos_group == "OL":
        for col in ("is_ol_tackle", "is_ol_guard", "is_ol_center"):
            if col in snapshot.columns:
                derived_cols.append(col)
    if pos_group == "DB":
        for col in ("pff_corner_snap_pct", "pff_slot_snap_pct", "pff_fs_snap_pct"):
            if col in snapshot.columns:
                derived_cols.append(col)
    # Weight-adjusted speed score (see config.SPEED_SCORE_POSITIONS). The raw
    # combine_wt / combine_forty stay in the surface alongside it.
    if pos_group in SPEED_SCORE_POSITIONS and "combine_speed_score" in snapshot.columns:
        derived_cols.append("combine_speed_score")

    # ── Audit item-5 career detail: BUILT AND REVERTED 2026-07-25 ─────────────
    # Step 03 still computes career_rec_long_max / career_rush_long_max /
    # career_int_return_yards_per_int / career_solo_tackle_share /
    # career_q1_play_share / career_q4_play_share, but no head consumes them —
    # all three families scored negative on a 3-seed placebo-anchored A/B (see
    # the note in config.ATTRIBUTE_MERGE_COLS for the per-family numbers).
    # To re-test, restore the position-gated block from git history at this
    # point AND re-add the columns to config.ATTRIBUTE_MERGE_COLS. BOTH are
    # required: the merge list controls what reaches the snapshot, this function
    # controls what a head may use, and wiring only the merge list produces an
    # experiment where every arm is identical (which is how the first run of
    # this A/B silently reported Δρ = 0.0000 on all 33 heads).
    # Position gating used when it was live, for reference:
    #   career_rec_long_max              WR / TE / RB
    #   career_rush_long_max             RB / QB / WR
    #   career_int_return_yards_per_int  DB / LB / EDGE / DL
    #   career_solo_tackle_share         DB / LB / EDGE / DL
    #   career_q1/q4_play_share          all except K / P

    # Kick/punt return production (computed in step 06). Special-teams value is
    # a live draft-value channel for the fringe of these three groups; it is
    # noise for the positions that never return kicks, so it stays gated.
    if pos_group in RETURN_FEATURE_POSITIONS:
        for col in ("career_return_att", "career_return_yards",
                    "career_return_tds", "career_kr_avg", "career_pr_avg"):
            if col in snapshot.columns:
                derived_cols.append(col)

    # Specialist career detail (step 03). K gets leg strength, P gets
    # placement; each is NaN for the other by construction, so they're gated
    # rather than shared across the two heads.
    if pos_group == "K" and "career_fg_long_max" in snapshot.columns:
        derived_cols.append("career_fg_long_max")
    if pos_group == "P":
        for col in ("career_punt_long_max", "career_punt_in20_rate",
                    "career_punt_tb_rate"):
            if col in snapshot.columns:
                derived_cols.append(col)

    if pos_group == "RB":
        # Tire-wear proxies. Gamelog totals (back to 2004, rough classification);
        # PFF totals (2014+, cleaner classification + literal contact-volume via
        # yards_after_contact). Both populated in pipeline/06_skill_estimation.
        for col in ("career_rush_att", "career_touches",
                    "pff_career_yards_after_contact",
                    "pff_career_avoided_tackles",
                    "pff_career_rush_att"):
            if col in snapshot.columns:
                derived_cols.append(col)

    # PFF grade features: static triplets (mean/peak/last). Gates on column
    # presence so missing-PFF players just don't get the columns (NaN-fill
    # via LightGBM missing-direction).
    pff_cols = []
    for base_col in PFF_GRADE_FEATURES_BY_GROUP.get(pos_group, []):
        for suffix in ("_mean", "_peak", "_last"):
            col = f"{base_col}{suffix}"
            if col in snapshot.columns:
                pff_cols.append(col)
    if "pff_seasons" in snapshot.columns:
        pff_cols.append("pff_seasons")
    # Career-total denominator reliability features. Gated by
    # _PFF_DENOM_RELEVANCE so a QB's feature list doesn't carry
    # pff_snap_counts_coverage_total (kicker doesn't see route totals, etc.).
    for denom in PFF_DENOM_STATS:
        if pos_group not in _PFF_DENOM_RELEVANCE.get(denom, frozenset()):
            continue
        col = f"{denom}_total"
        if col in snapshot.columns:
            pff_cols.append(col)

    era_cols = [c for c in NFL_LG_FEATURES if c in snapshot.columns]

    # Combine features: default to all `_z` cohort-normalized cols, then
    # apply per-position raw-vs-z swaps from `COMBINE_RAW_BY_GROUP` (audit
    # 2026-05-10 — anatomy and a few technique events correlate better as
    # absolute values than cohort-z'd. See experiments/00i_combine_zscore_audit.py
    # for the empirical evidence per (metric × pos)).
    z_cols = [c for c in snapshot.columns
              if c.startswith("combine_") and c.endswith("_z")]
    raw_for_pos = COMBINE_RAW_BY_GROUP.get(pos_group, [])
    z_to_drop = {f"{c}_z" for c in raw_for_pos}
    z_cols = [c for c in z_cols if c not in z_to_drop]
    raw_cols = [c for c in raw_for_pos if c in snapshot.columns]
    combine_cols = ["combine_invited"] + sorted(z_cols) + sorted(raw_cols)
    combine_cols = [c for c in combine_cols if c in snapshot.columns]

    hasan_cols = [c for c in HASAN_FEATURES if c in snapshot.columns]

    # Teammate-QB career triplets — emit only for WR/TE/RB (the positions
    # the feature was designed for; see QB_CAREER_POSITION_GROUPS). Gated by
    # the master USE_QB_CAREER_FEATURES flag so the feature can be turned
    # off cleanly post-Boruta-audit if it under-performs.
    qb_career_cols: list[str] = []
    if USE_QB_CAREER_FEATURES and pos_group in QB_CAREER_POSITION_GROUPS:
        for base in QB_CAREER_BASE_FEATURES:
            for suffix in ("_mean", "_peak", "_last"):
                col = f"{base}{suffix}"
                if col in snapshot.columns:
                    qb_career_cols.append(col)

    # Play-grain skill features (step 4p/5p produce per-play residuals; step 6
    # builds Kalman/EWMA streams on them in parallel with the per-game streams
    # enumerated in `skill_cols` above). The play-grain stat names don't live
    # in MODELED_STATS, so discover them from the snapshot. Zero-variance
    # features (rare-event stats with all-catch-all rows) are pruned by the
    # non-constant filter pre-Boruta.
    play_skill_prefixes = (
        "kalman_mean_play_mean_resid_", "kalman_var_play_mean_resid_",
        "ewma_play_mean_resid_",        "ewma_var_play_mean_resid_",
        "kalman_yoy_play_mean_resid_",  "ewma_yoy_play_mean_resid_",
        "kalman_mean_play_total_resid_","kalman_var_play_total_resid_",
        "ewma_play_total_resid_",       "ewma_var_play_total_resid_",
        "kalman_yoy_play_total_resid_", "ewma_yoy_play_total_resid_",
        # Raw rates for FAIL-gated outcome stats (INT rate, fumble rate, ...)
        "kalman_mean_play_raw_mean_",   "kalman_var_play_raw_mean_",
        "ewma_play_raw_mean_",          "ewma_var_play_raw_mean_",
        "kalman_yoy_play_raw_mean_",    "ewma_yoy_play_raw_mean_",
    )
    play_skill_cols: list[str] = []
    for prefix in play_skill_prefixes:
        for c in snapshot.columns:
            if not c.startswith(prefix):
                continue
            # Gate to positions the stat is a real production signal for. A play
            # stat with no entry in PLAY_SKILL_STAT_POSITIONS is kept (fail
            # open, so a newly-added stat isn't silently dropped everywhere).
            stat = c[len(prefix):]
            relevant = PLAY_SKILL_STAT_POSITIONS.get(stat)
            if relevant is None or pos_group in relevant:
                play_skill_cols.append(c)

    cols = (skill_cols + play_skill_cols + attr_cols + derived_cols + pff_cols
            + qb_career_cols + era_cols + combine_cols + hasan_cols)
    # Drop features confirmed in 0 heads on the latest Boruta audit (see
    # config.BORUTA_DROP_FEATURES). Single source of truth — `_build_feature_cols`
    # is called by both tuning + final fit, so no head sees these.
    if BORUTA_DROP_FEATURES:
        cols = [c for c in cols if c not in BORUTA_DROP_FEATURES]
    return cols


def _state_space_cols(cols: list[str]) -> list[str]:
    """Return the PFF state-space subset of cols (the dilution block)."""
    return [c for c in cols if c.startswith((
        "kalman_mean_pff_", "kalman_var_pff_",
        "ewma_pff_",        "ewma_var_pff_",
        "kalman_yoy_pff_",  "ewma_yoy_pff_",
    ))]


def _load_mi_features(pos_group: str, model_name: str) -> list[str] | None:
    """Return the cached MI-selected state-space cols for one (pos, model),
    or None if no cache exists / the head is missing from the cache.
    Returns the SELECTED cols (top-N of state-space block) — caller
    intersects with the candidate set."""
    if not MI_FEATURES_PATH.exists():
        return None
    try:
        with open(MI_FEATURES_PATH) as f:
            data = json.load(f)
        block = data.get(f"{pos_group}/{model_name}")
        if not isinstance(block, dict):
            return None
        selected = list(block.get("selected") or [])
        return selected or None
    except Exception:
        return None


def _drop_dominant_value_features(feature_cols: list[str],
                                    df: pd.DataFrame,
                                    max_share: float = 0.99) -> list[str]:
    """Drop features where a single value covers >max_share of position-cohort
    rows. Catches Bayesian-prior-initialized EWMA features for stat-irrelevant
    positions: e.g. `ewma_play_total_resid_field_goal_missed` lands at the
    cohort prior for every WR with no FG data, so 99%+ of WRs share one value
    while 3 trick-play WRs create distinct values that LightGBM splits on as
    noise. The `nunique > 1` pre-Boruta filter passes these through.
    """
    if df.empty or not feature_cols:
        return feature_cols
    keep, dropped = [], []
    for c in feature_cols:
        if c not in df.columns:
            keep.append(c); continue
        vc = df[c].value_counts(dropna=True)
        if len(vc) == 0:
            dropped.append(c); continue
        n_nonnull = int(df[c].notna().sum())
        # Denominator is non-null rows: a prior-anchored feature where one
        # value dominates among non-NaN rows is the failure mode we want to
        # catch, regardless of how many rows are NaN.
        if n_nonnull > 0 and vc.iloc[0] / n_nonnull > max_share:
            dropped.append(c)
        else:
            keep.append(c)
    if dropped:
        logger.info("  dropped %d dominant-value features (>%.0f%% share)",
                    len(dropped), 100 * max_share)
    return keep


def _filter_state_space_to_mi(feature_cols: list[str], pos_group: str,
                               model_name: str) -> list[str]:
    """Apply the MI cache to feature_cols. Keeps:
      - all non-state-space cols (combine, attrs, hasan, pff_career, gg_*)
      - the cached top-N state-space cols intersected with the current set

    No-op when no cache exists. Defensively falls through to all state-space
    cols if intersection would be empty (shouldn't happen in practice)."""
    cached = _load_mi_features(pos_group, model_name)
    if cached is None:
        return feature_cols
    keep_ss = set(cached)
    ss_in_input = set(_state_space_cols(feature_cols))
    selected_ss = ss_in_input & keep_ss
    if not selected_ss and ss_in_input:
        logger.warning("  %s/%s: MI cache filtered out all state-space cols; "
                       "falling back to full state-space set",
                       pos_group, model_name)
        return feature_cols
    out = [c for c in feature_cols
           if c not in ss_in_input or c in selected_ss]
    if len(out) < len(feature_cols):
        logger.info("  %s/%s: MI filter %d → %d features (state-space %d → %d)",
                    pos_group, model_name,
                    len(feature_cols), len(out),
                    len(ss_in_input), len(selected_ss))
    return out


def _save_mi_features(snapshot: pd.DataFrame,
                      by_key_payload: dict[str, dict],
                      by_key_meta: dict[str, dict],
                      *, top_n: int) -> None:
    """Write mi_features.json keyed by f'{pos}/{model}' → {selected, scores,
    n_state_space_in, n_selected, n_train_rows}. Manifest sidecar mirrors the
    tuned_params staleness pattern."""
    MI_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(by_key_payload)
    payload["_meta"] = {
        "snapshot_n_rows": int(len(snapshot)),
        "top_n":           int(top_n),
        "n_neighbors":     int(MI_N_NEIGHBORS),
        "random_state":    int(MI_RANDOM_STATE),
    }
    with open(MI_FEATURES_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    fingerprint = {
        "by_key": {k: {
            "candidate_cols_hash": meta["candidate_cols_hash"],
            "n_state_space_in":    int(meta["n_state_space_in"]),
            "n_selected":          int(meta["n_selected"]),
        } for k, meta in by_key_meta.items()},
        "global_context": {
            "snapshot_n_rows":  int(len(snapshot)),
            "mi_top_n":         int(top_n),
            "mi_n_neighbors":   int(MI_N_NEIGHBORS),
            "mi_random_state":  int(MI_RANDOM_STATE),
        },
    }
    write_manifest(
        MI_FEATURES_PATH, fingerprint=fingerprint,
        mode_used="fresh", step_name=STEP_NAME,
        extra={"n_keys": len(by_key_payload)},
    )
    logger.info("Saved MI features for %d heads → %s",
                len(by_key_payload), MI_FEATURES_PATH)


def _run_mi_for_all_heads(snapshot: pd.DataFrame, labels: pd.DataFrame,
                          top_n: int) -> None:
    """For each (pos × target) head, rank the state-space candidate cols by
    mutual information against the target on the training panel and write the
    top-N to the MI cache. Subsequent fits read this cache via
    `_filter_state_space_to_mi`. Other feature categories (combine, attrs,
    hasan, pff_career, game-grain Kalman/EWMA) are NOT filtered — only the
    dilution-prone PFF state-space block is.
    """
    all_folds = _expanding_window_folds()
    _, holdout_fold = _split_holdout(all_folds)
    mi_train_end = (holdout_fold[0] if holdout_fold is not None
                    else _FINAL_TRAIN_CUTOFF)
    logger.info("MI training window: last_college_season_year ≤ %d", mi_train_end)

    by_key_payload: dict[str, dict] = {}
    by_key_meta: dict[str, dict] = {}

    for pos_group in POSITION_GROUPS.keys():
        feature_cols_full = _build_feature_cols(snapshot, pos_group)
        if not feature_cols_full:
            continue
        ss_cols = _state_space_cols(feature_cols_full)
        if not ss_cols:
            logger.info("  %s: no state-space cols, skipping MI filter", pos_group)
            continue

        for target_col, model_name in HEADS:
            df = _prep_training_df(snapshot, labels, pos_group, target_col)
            yr = df["last_college_season_year"].fillna(0)
            df_train = df[yr <= mi_train_end]
            if len(df_train) < 50:
                logger.warning("  %s/%s: only %d rows ≤ %d — skipping MI",
                               pos_group, model_name, len(df_train), mi_train_end)
                continue

            X = df_train[ss_cols]
            y = df_train[target_col].fillna(0.0)
            verdict = rank_by_mi(
                X, y,
                n_neighbors=MI_N_NEIGHBORS,
                random_state=MI_RANDOM_STATE,
            )
            ranking = verdict["ranking"]
            scores = verdict["scores"]
            selected = ranking[:top_n]

            logger.info("  %s/%s: MI %d → %d state-space cols (top score=%.4f, cutoff score=%.4f)",
                        pos_group, model_name, len(ss_cols), len(selected),
                        scores[selected[0]] if selected else 0.0,
                        scores[selected[-1]] if selected else 0.0)

            by_key_payload[f"{pos_group}/{model_name}"] = {
                "selected":         selected,
                "scores":           {c: round(scores[c], 6) for c in ranking},
                "n_state_space_in": len(ss_cols),
                "n_selected":       len(selected),
                "n_train_rows":     int(len(df_train)),
                "mi_train_end":     int(mi_train_end),
                "top_n":            int(top_n),
            }
            by_key_meta[f"{pos_group}/{model_name}"] = {
                "candidate_cols_hash": hash_features(ss_cols),
                "n_state_space_in":    len(ss_cols),
                "n_selected":          len(selected),
            }

    if by_key_payload:
        _save_mi_features(snapshot, by_key_payload, by_key_meta, top_n=top_n)
    else:
        logger.warning("MI produced no verdicts — nothing to write")


def _prep_training_df(snapshot: pd.DataFrame, labels: pd.DataFrame,
                       pos_group: str, target_col: str) -> pd.DataFrame:
    """Merge labels onto snapshot, fill missing target/bridge fields, filter to pos_group."""
    label_cols = ["player_id", target_col, "is_bridge"]
    if "nfl_games" in labels.columns:
        label_cols.append("nfl_games")
    if "draft_year" in labels.columns:
        label_cols.append("draft_year")
    df = snapshot.merge(
        labels[[c for c in label_cols if c in labels.columns]],
        on="player_id", how="left",
    )
    df["is_bridge"] = df["is_bridge"].fillna(False)

    # Drop bridge rows from draft classes whose contract coverage is
    # unreliable. Applies to the APY head only — see the config rationale and
    # the 2026-07-26 cohort A/B. Non-bridge rows are never touched: the
    # zero-padded mixture is load-bearing.
    if (IMPACT_LABEL_MIN_DRAFT_YEAR is not None
            and target_col == "impact_target"
            and "draft_year" in df.columns):
        stale = (df["is_bridge"].astype(bool)
                 & df["draft_year"].notna()
                 & (df["draft_year"] < IMPACT_LABEL_MIN_DRAFT_YEAR))
        if stale.any():
            df = df[~stale]
    # Zero-pad the mixture target for non-bridge rows only (never made the
    # NFL → genuine 0 outcome). Bridge rows with a deliberately-NaN label
    # (impact_w_av_per_game when nfl_games < 4: a per-game rate on that
    # sample is noise, not a zero) drop out of training — matching step 7's
    # intent and the global model's handling, instead of training as busts.
    nonbridge = ~df["is_bridge"].astype(bool)
    df.loc[nonbridge, target_col] = df.loc[nonbridge, target_col].fillna(0.0)
    df = df[df[target_col].notna()]
    if "nfl_games" not in df.columns:
        df["nfl_games"] = 0.0
    else:
        df["nfl_games"] = df["nfl_games"].fillna(0.0)
    if "position_group" in df.columns:
        df = df[df["position_group"] == pos_group]
    return df


# The heads step 8 tunes and trains. The role head was dropped 2026-10-31:
# its target (role_target, the y3 role-trajectory score) is ~saturated, the
# head was seed-unstable, and the role score stopped being an output.
HEADS: list[tuple[str, str]] = [
    ("impact_target",        "impact"),       # APY%cap (primary)
    ("impact_w_av_per_game", "impact_wav"),   # complementary AV/g signal
]


# Maps model_name → the label column that head trains on. Must stay in sync
# with the (target_col, model_name) tuples in main()'s tuning/training loops.
MODEL_TARGET_COLS = {
    "role":       "role_target",
    "impact":     "impact_target",
    "impact_wav": "impact_w_av_per_game",
}


def _resolve_fit_features(snapshot: pd.DataFrame, labels: pd.DataFrame,
                          pos_group: str, target_col: str, model_name: str,
                          ) -> tuple[list[str], pd.DataFrame | None]:
    """Single source of truth for a head's fit-time feature surface:
    candidate cols → MI filter → dominant-value filter (computed on the
    head's training cohort). Tuning, the final fit, and the tuned-params
    fingerprint must all resolve features through here — any path that
    diverges makes Optuna optimize (or the staleness check compare) a
    different surface than the deployed model trains on."""
    feature_cols = _build_feature_cols(snapshot, pos_group)
    if not feature_cols:
        return [], None
    feature_cols = _filter_state_space_to_mi(feature_cols, pos_group, model_name)
    df = _prep_training_df(snapshot, labels, pos_group, target_col)
    feature_cols = _drop_dominant_value_features(feature_cols, df, max_share=0.99)
    return feature_cols, df


def _board_row_weights(frame: pd.DataFrame, model_name: str) -> Optional[np.ndarray]:
    """IMPACT_BOARD_ROW_WEIGHT on impact-head rows with a board rank, else 1.

    None (unweighted) when the weight is 1.0, for other heads, or when the
    frame has no hasan_rank column — keeps the production path byte-identical.
    """
    if (IMPACT_BOARD_ROW_WEIGHT == 1.0 or model_name != "impact"
            or "hasan_rank" not in frame.columns):
        return None
    return np.where(frame["hasan_rank"].notna(), IMPACT_BOARD_ROW_WEIGHT, 1.0)


def _fit_fold_honest(make_model, X_tr: np.ndarray, y_tr: np.ndarray,
                     yrs_tr: np.ndarray, es_rounds: int = 50,
                     sample_weight: Optional[np.ndarray] = None):
    """Fold fit whose early stopping never sees the scoring fold.

    Early-stopping against the fold's own validation set lets the model pick
    its iteration count with knowledge of the answers — cv metrics go
    optimistic, Optuna rewards ES-exploiting params, and 8c's calibrators are
    fit on rosier-than-deployment OOF predictions. Instead: peel the most
    recent season(s) off the TRAINING window as an inner early-stopping slice
    (≥10 rows, leaving ≥30 core rows), select best_iteration there, then
    refit on the full training window at that iteration count. Falls back to
    a plain no-ES fit at the params' n_estimators ceiling when no viable
    temporal split exists (e.g. single-season windows).

    `sample_weight` is optional and defaults to None (unweighted), which is
    the production path — it exists for the cohort A/B's down-weighting arm.
    The weights are sliced alongside the early-stopping split so the inner
    probe and the final refit see consistent weighting.

    Returns (fitted_model_on_full_train, best_iter).
    """
    yrs = np.asarray(yrs_tr, dtype=float)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    uniq = np.unique(yrs)[::-1]  # descending — most recent season first
    es_mask = np.zeros(len(yrs), dtype=bool)
    for y in uniq[:-1]:          # never consume the entire window
        es_mask |= yrs == y
        if es_mask.sum() >= 10:
            break
    if es_mask.sum() >= 10 and (~es_mask).sum() >= 30:
        probe = make_model()
        probe.fit(X_tr[~es_mask], y_tr[~es_mask],
                  sample_weight=None if w is None else w[~es_mask],
                  eval_set=[(X_tr[es_mask], y_tr[es_mask])],
                  eval_sample_weight=None if w is None else [w[es_mask]],
                  callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                             lgb.log_evaluation(period=-1)])
        best_iter = int(probe.best_iteration_
                        or probe.get_params().get("n_estimators", 100))
        final = make_model()
        final.set_params(n_estimators=max(1, best_iter))
        final.fit(X_tr, y_tr, sample_weight=w)
        return final, best_iter
    m = make_model()
    m.fit(X_tr, y_tr, sample_weight=w)
    return m, int(m.get_params().get("n_estimators", 100))


# A fold whose training window retains fewer bridge rows than this cannot teach
# a magnitude head anything: it sees only the zero-padded non-bridge universe,
# collapses to predicting ~0, and — because the target is mostly zeros —
# actually posts a BETTER MAE while its ranking goes to chance. Left unguarded
# this turns the early-cohort filter into a metric artifact and poisons step
# 8c's calibrators with degenerate OOF predictions.
MIN_BRIDGE_ROWS_PER_FOLD = 50


def _usable_folds(df: pd.DataFrame, folds: list[tuple[int, int, int]],
                   target_col: str, pos_group: str | None = None,
                   model_name: str | None = None,
                   force: bool = False) -> list[tuple[int, int, int]]:
    """Drop folds starved of bridge training rows by the early-cohort filter.

    Only relevant when that filter is active and the target is the APY head —
    every other head keeps its full training population, so its folds are
    untouched.
    """
    if not force and (IMPACT_LABEL_MIN_DRAFT_YEAR is None
                      or target_col != "impact_target"):
        return folds
    if "is_bridge" not in df.columns:
        return folds
    yr = df["last_college_season_year"].fillna(0)
    bridge = df["is_bridge"].astype(bool)
    keep, dropped = [], []
    for fold in folds:
        n = int((bridge & (yr <= fold[0])).sum())
        (keep if n >= MIN_BRIDGE_ROWS_PER_FOLD else dropped).append((fold, n))
    if dropped:
        logger.info("  %s/%s: skipping %d fold(s) with <%d bridge training "
                    "rows after the early-cohort filter (%s)",
                    pos_group or "?", model_name or target_col, len(dropped),
                    MIN_BRIDGE_ROWS_PER_FOLD,
                    ", ".join(f"val {f[1]}-{f[2]}: n={n}" for f, n in dropped))
    return [f for f, _ in keep]


def _build_fold_data(df: pd.DataFrame, feature_cols: list[str],
                     target_col: str, folds: list[tuple[int, int, int]],
                     ) -> list[tuple]:
    """Materialize per-fold train/val arrays ONCE. The arrays are
    trial-invariant — only LightGBM params vary across Optuna trials — so
    rebuilding them inside `_evaluate_params` paid the dataframe→numpy
    conversion cost n_trials × n_seeds times per head. Returns a list of
    (X_tr, y_tr, yrs_tr, X_val, y_val, bridge_val) tuples, skipping undersized
    folds. `bridge_val` is needed by the "spearman" objective, which scores
    ranking on BRIDGE rows only — the zero-padded non-bridge majority carries
    no ordering information."""
    yr = df["last_college_season_year"].fillna(0)
    out = []
    for train_end, val_start, val_end in folds:
        f_train = df[yr <= train_end]
        f_val   = df[(yr >= val_start) & (yr <= val_end)]
        if len(f_train) < 30 or len(f_val) < 10:
            continue
        out.append((
            f_train[feature_cols].to_numpy(dtype=float, na_value=np.nan),
            f_train[target_col].fillna(0.0).values,
            f_train["last_college_season_year"].fillna(0).values,
            f_val[feature_cols].to_numpy(dtype=float, na_value=np.nan),
            f_val[target_col].fillna(0.0).values,
            (f_val["is_bridge"].astype(bool).values
             if "is_bridge" in f_val.columns
             else np.ones(len(f_val), dtype=bool)),
        ))
    return out


def _evaluate_params(params: dict, fold_data: list[tuple],
                     feature_cols: list[str],
                     metric: str = "mae",
                     pos_group: str | None = None,
                     model_name: str | None = None,
                     ) -> tuple[float | None, int | None]:
    """
    Run one expanding-window CV pass with given params over precomputed
    fold arrays (see `_build_fold_data`); return (loss, mean_n_best).

    `metric`:
      - "mae"     — mean absolute error, lower=better
      - "auc"     — ROC AUC after binarizing y at 0.5; returns -AUC so lower=better
      - "logloss" — soft binary cross-entropy with y ∈ [0,1]; lower=better.
                    Handles soft labels (trajectory-override probabilities) without
                    discarding signal the way binarized AUC would.
      - "spearman" — bridge-only rank correlation against the realised outcome;
                    returns -rho so lower=better. This is the project's primary
                    evaluation metric (2026-07-26). MAE is unsafe as an OBJECTIVE
                    on these zero-inflated targets: ~93% of rows are zero-padded
                    non-bridge, so shrinking predictions toward zero cuts error
                    on the zero majority faster than it adds error on the nonzero
                    minority. Optimising MAE therefore rewards over-regularised,
                    shrinkage-heavy params — the same pathology that produced two
                    spurious "wins" on 2026-07-26. Ranking is immune to a
                    monotone rescale.
    """
    if metric not in ("mae", "auc", "logloss", "spearman"):
        raise ValueError(f"Unknown metric: {metric}")
    # Pin LightGBM to a single thread during tuning so Optuna n_jobs=-1
    # parallelism (study.optimize) doesn't oversubscribe cores. Final fit
    # in train_model_for_group uses default threading.
    base = {**LGBM_PARAMS, "num_threads": 1, **params}
    fold_scores, fold_n_bests = [], []
    for X_ft, y_ft, yrs_ft, X_fv, y_fv, b_fv in fold_data:
        _make = lambda: lgb.LGBMRegressor(**{
            **base,
            **_build_monotone_kwargs(feature_cols, pos_group, model_name),
        })
        m, best_iter = _fit_fold_honest(_make, X_ft, y_ft, yrs_ft)
        preds = m.predict(X_fv)
        if metric == "mae":
            fold_scores.append(float(np.mean(np.abs(preds - y_fv))))
        elif metric == "auc":
            y_bin = (y_fv > 0.5).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            from sklearn.metrics import roc_auc_score
            fold_scores.append(-float(roc_auc_score(y_bin, preds)))
        elif metric == "spearman":
            # Bridge rows only. A fold is skipped ONLY when it structurally
            # cannot support a rank correlation: too few bridge rows, or a
            # constant TARGET (no ordering to recover). Those are properties of
            # the data, identical for every trial.
            #
            # A constant PREDICTION is different — that is a property of the
            # trial, and it means the model has no ranking ability at all, so
            # it scores rho = 0. Skipping it (as this did originally) let
            # degenerate models escape scoring entirely: on small cohorts like
            # K most sampled params predict a constant, every fold got skipped,
            # every trial returned None and was pruned, and a seed with all 30
            # trials pruned crashed on `study.best_trial`. That is exactly the
            # shrinkage pathology this objective exists to catch, so it must be
            # scored, not hidden.
            yb, pb = y_fv[b_fv], preds[b_fv]
            if len(yb) < 10 or np.std(yb) < 1e-12:
                continue
            if np.std(pb) < 1e-12:
                fold_scores.append(0.0)          # no ranking ability
                fold_n_bests.append(int(best_iter))
                continue
            from scipy.stats import spearmanr
            rho = spearmanr(pb, yb).statistic
            fold_scores.append(0.0 if not np.isfinite(rho) else -float(rho))
        else:  # logloss
            eps = 1e-7
            p = np.clip(preds, eps, 1.0 - eps)
            y = np.clip(y_fv, 0.0, 1.0)
            bce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            fold_scores.append(float(bce))
        fold_n_bests.append(int(best_iter))
    if not fold_n_bests:
        return None, None
    return float(np.mean(fold_scores)), int(round(float(np.mean(fold_n_bests))))


def tune_hyperparams(snapshot: pd.DataFrame, labels: pd.DataFrame,
                     pos_group: str, target_col: str, model_name: str,
                     n_trials: int = 30, seed: int = 42,
                     metric: str | None = None,
                     n_seeds: int = 1) -> dict | None:
    """Hyperparameter search via expanding-window CV (Optuna TPE if available,
    random search fallback). Returns best params or None.

    `metric`: "logloss" for role models (handles soft probability targets),
    "mae" for continuous impact models. Defaults based on `model_name`.

    `n_seeds`: when > 1, runs N independent Optuna studies with different
    seeds and returns the params of the seed whose best trial has the
    lowest CV. The per-seed evaluation is deterministic (LightGBM
    random_state fixed via LGBM_PARAMS), so each seed's `best_value` is
    an honest CV score — taking the min across seeds is unbiased. Cost
    scales linearly: 3 seeds ≈ 3× wall time.
    """
    if not _lgbm_available:
        raise ImportError("lightgbm not installed.")
    if metric is None:
        # NOTE (2026-07-26): the project's PRIMARY evaluation metric is now
        # ranking (bridge-only Spearman on fold-val rows), not cv_mae — MAE is
        # gameable on these zero-inflated targets, where shrinking predictions
        # toward zero cuts error on the zero majority faster than it adds error
        # on the nonzero minority. Two separate artifacts of exactly that shape
        # were caught the same day (starved folds; hurdle gate shrinkage).
        #
        # REVERTED 2026-07-27 to "mae" after a full 3-seed retune. The
        # "spearman" objective is implemented and tested below, and the
        # hypothesis behind it was CONFIRMED — reg_lambda (the L2 penalty that
        # shrinks predictions toward zero) collapsed 7.435 -> 0.0152 the moment
        # MAE stopped being the objective, so MAE really was selecting for
        # prediction shrinkage. It still did not generalise:
        #
        #     tuning-fold rho (optimised)   19/19 heads improved, mean +0.097
        #     holdout rho (out-of-sample)   core mean -0.014, 9/18 improved
        #     cv_mae                        +30%
        #
        # A perfect score on the optimised metric with zero transfer is
        # objective overfitting. MAE is computed on thousands of validation
        # rows; bridge-only Spearman on 19-237, so it is unbiased but noisy,
        # and 30 trials x 3 seeds of Optuna pressure exploits that noise. MAE
        # is biased but low-variance, and for SELECTION that tradeoff wins.
        #
        # Ranking remains the primary metric for ship/revert DECISIONS, where a
        # handful of arms are compared and there is little selection pressure —
        # see the evaluation-preferences memory. Selection and evaluation are
        # different jobs; this reverts only the former.
        metric = "logloss" if model_name == "role" else "mae"
    # Resolve through the shared helper so Optuna searches the exact surface
    # the final fit trains on (incl. the dominant-value filter).
    feature_cols, df = _resolve_fit_features(snapshot, labels, pos_group,
                                             target_col, model_name)
    if not feature_cols or df is None or len(df) < 50:
        return None
    # Stable per-model seed: each (pos_group, model_name) gets a different
    # search trajectory but reproducibly so. Multi-seed runs offset this
    # by a large prime to land on independent TPE trajectories.
    h = hashlib.sha256(f"{pos_group}/{model_name}".encode()).digest()
    base_model_seed = seed + int.from_bytes(h[:4], "big")
    # Hyperparameter search uses tuning folds only — never the holdout fold,
    # so the latest-fold report metric stays an honest generalization check.
    tuning_folds, _ = _split_holdout(_expanding_window_folds())
    # Fold arrays are trial-invariant — build once, reuse across all trials
    # and seeds (removes the dominant constant cost of the tuning path).
    tuning_folds = _usable_folds(df, tuning_folds, target_col,
                                  pos_group, model_name)
    fold_data = _build_fold_data(df, feature_cols, target_col, tuning_folds)
    if fold_data and metric == "spearman":
        # A head whose folds cannot support a rank correlation at all — too few
        # bridge rows, or no target variance among them — would prune every
        # trial and end up untuned. K/impact_wav and P/impact_wav are both in
        # this position (zero usable folds). Fall back to MAE there: it is the
        # metric this objective replaced precisely because it is gameable, but
        # for these tiny heads it is the only signal available, and a gameable
        # score beats no tuning at all. Logged loudly so it stays visible.
        n_usable = sum(
            1 for (_Xt, _yt, _yr, _Xv, y_fv, b_fv) in fold_data
            if int(b_fv.sum()) >= 10 and float(np.std(y_fv[b_fv])) > 1e-12)
        if n_usable < 2:
            logger.warning(
                "  %s/%s: only %d fold(s) can support a rank correlation — "
                "falling back to mae for this head", pos_group, model_name,
                n_usable)
            metric = "mae"
    if not fold_data:
        return None

    try:
        import optuna
    except ImportError:
        optuna = None

    def _run_random_search(model_seed: int) -> tuple[float, dict] | None:
        rng = np.random.default_rng(model_seed)
        best: tuple[float, dict] | None = None
        for trial in range(n_trials):
            params = _sample_hyperparams(rng)
            score, _ = _evaluate_params(params, fold_data, feature_cols,
                                        metric=metric,
                                        pos_group=pos_group, model_name=model_name)
            if score is None:
                continue
            if best is None or score < best[0]:
                best = (score, params)
                display = -score if metric == "auc" else score
                logger.info("    trial %d (random): NEW BEST %s=%.4f",
                            trial + 1, metric, display)
        return best

    def _run_optuna(model_seed: int, seed_label: str) -> tuple[float, dict] | None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=model_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        best_seen = {"score": float("inf")}

        def _objective(trial):
            params = _suggest_optuna_params(trial)
            score, _ = _evaluate_params(params, fold_data, feature_cols,
                                        metric=metric,
                                        pos_group=pos_group, model_name=model_name)
            if score is None:
                raise optuna.TrialPruned()
            if score < best_seen["score"]:
                best_seen["score"] = score
                display = -score if metric == "auc" else score
                logger.info("    %strial %d (optuna): NEW BEST %s=%.4f",
                            seed_label, trial.number + 1, metric, display)
            return score

        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False,
                       catch=(Exception,), n_jobs=-1)
        # `study.best_trial` RAISES when every trial was pruned — it does not
        # return falsy — so this needs a try, not a truthiness check.
        try:
            bt = study.best_trial
        except (ValueError, RuntimeError):
            logger.warning("    %sall %d trials pruned — no usable score for "
                           "this seed", seed_label, n_trials)
            return None
        return (float(bt.value), dict(bt.params))

    # Run N seeds; pick the one with the lowest CV. Each seed's best_value
    # is an honest deterministic CV score, so min() is unbiased.
    seed_results: list[tuple[int, float, dict]] = []
    for seed_idx in range(max(1, n_seeds)):
        # Offset by a large prime so each seed lands on a distinct TPE trajectory.
        model_seed = base_model_seed + seed_idx * 1009
        seed_label = f"seed {seed_idx+1}/{n_seeds} " if n_seeds > 1 else ""
        result = (_run_random_search(model_seed) if optuna is None
                  else _run_optuna(model_seed, seed_label))
        if result is not None:
            seed_results.append((seed_idx, result[0], result[1]))

    if not seed_results:
        return None

    seed_results.sort(key=lambda x: x[1])
    winner_idx, winner_score, winner_params = seed_results[0]
    if n_seeds > 1:
        scores = [r[1] for r in seed_results]
        logger.info("  Multi-seed results (n=%d): "
                    "best=%.4f (seed %d), worst=%.4f, spread=%.4f",
                    len(seed_results), min(scores), winner_idx + 1,
                    max(scores), max(scores) - min(scores))
    return winner_params


def train_model_for_group(
        snapshot: pd.DataFrame,
        labels: pd.DataFrame,
        pos_group: str,
        target_col: str,
        model_name: str,
        test_mode: bool = False,
) -> tuple[object, list[str], dict, dict]:
    """
    Train a LightGBM model for one position group and one target.
    Returns (model, feature_cols, metrics_dict, quantile_models).
    quantile_models is non-empty only for the impact head (p10/p50/p90).
    """
    if not _lgbm_available:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    feature_cols, df = _resolve_fit_features(snapshot, labels, pos_group,
                                             target_col, model_name)
    if not feature_cols:
        raise ValueError(f"No features available for {pos_group}")
    if df is None or len(df) < 50:
        raise ValueError(f"Too few samples for {pos_group}/{model_name}: "
                         f"{0 if df is None else len(df)}")

    effective_target = target_col

    params = {**LGBM_PARAMS}
    tuned = _load_tuned_params(pos_group, model_name)
    params_source = "default"
    if tuned:
        params.update(tuned)
        params_source = "tuned"
        logger.info("  %s/%s: using tuned hyperparameters", pos_group, model_name)
        _check_tuned_params_staleness(snapshot, pos_group, model_name, feature_cols)
    if test_mode:
        params["n_estimators"] = 50

    # Phase 1 — expanding-window CV on TUNING folds only (latest fold reserved
    # as a one-shot holdout). The holdout metric is the honest generalization
    # check that tuning never sees.
    all_folds = _expanding_window_folds()
    tuning_folds, holdout_fold = _split_holdout(all_folds)
    tuning_folds = _usable_folds(df, tuning_folds, effective_target,
                                  pos_group, model_name)
    fold_maes, fold_n_bests = [], []
    fold_aucs: list[float] = []
    fold_loglosses: list[float] = []

    yr_series = df["last_college_season_year"].fillna(0)
    oof_records: list[pd.DataFrame] = []

    def _fit_eval_fold(train_end: int, val_start: int, val_end: int,
                       collect_oof: bool = False):
        f_train = df[yr_series <= train_end]
        f_val = df[(yr_series >= val_start) & (yr_series <= val_end)]
        if len(f_train) < 30 or len(f_val) < 10:
            return None
        X_ft = f_train[feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y_ft = f_train[effective_target].fillna(0.0).values
        X_fv = f_val[feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y_fv = f_val[effective_target].fillna(0.0).values
        _make = lambda: lgb.LGBMRegressor(**{
            **params,
            **_build_monotone_kwargs(feature_cols, pos_group, model_name),
        })
        m, best_iter = _fit_fold_honest(
            _make, X_ft, y_ft,
            f_train["last_college_season_year"].fillna(0).values,
            sample_weight=_board_row_weights(f_train, model_name))
        preds = m.predict(X_fv)
        if collect_oof:
            # Persisted for step 8c's calibrators (tuning folds only — the
            # holdout call below never sets collect_oof). Guarantees the
            # calibrators are fit on THE deployed training procedure's OOF
            # predictions instead of 8c's own re-implementation of it.
            oof_records.append(pd.DataFrame({
                "player_id":  f_val["player_id"].astype(str).values,
                "pos_group":  pos_group,
                "model_name": model_name,
                "val_start":  val_start,
                "val_end":    val_end,
                "y":          y_fv,
                "pred":       preds,
            }))
        return m, preds, y_fv, best_iter

    for fold_train_end, fold_val_start, fold_val_end in tuning_folds:
        out = _fit_eval_fold(fold_train_end, fold_val_start, fold_val_end,
                             collect_oof=True)
        if out is None:
            continue
        m, preds, y_fv, best_iter = out
        fold_maes.append(float(np.mean(np.abs(preds - y_fv))))
        fold_n_bests.append(best_iter)
        if model_name == "role":
            y_bin = (y_fv > 0.5).astype(int)
            if 0 < y_bin.sum() < len(y_bin):
                from sklearn.metrics import roc_auc_score
                fold_aucs.append(float(roc_auc_score(y_bin, preds)))
            eps = 1e-7
            p_clip = np.clip(preds, eps, 1.0 - eps)
            y_clip = np.clip(y_fv, 0.0, 1.0)
            fold_loglosses.append(float(
                -np.mean(y_clip * np.log(p_clip) + (1.0 - y_clip) * np.log(1.0 - p_clip))
            ))

    if not fold_n_bests:
        raise ValueError(f"No valid CV folds for {pos_group}/{model_name}")

    n_best = max(1, int(round(float(np.mean(fold_n_bests)))))
    avg_mae = float(np.mean(fold_maes))
    metrics = {"n_cv_folds": len(fold_maes), "cv_mae": avg_mae,
               "avg_best_iter": n_best, "params_source": params_source}
    if fold_loglosses:
        avg_logloss = float(np.mean(fold_loglosses))
        metrics["cv_logloss"] = avg_logloss
        if fold_aucs:
            avg_auc = float(np.mean(fold_aucs))
            metrics["cv_auc"] = avg_auc
            logger.info("  %s/%s: %d folds cv_mae=%.4f cv_logloss=%.4f cv_auc=%.4f avg_best_iter=%d",
                        pos_group, model_name, len(fold_maes), avg_mae,
                        avg_logloss, avg_auc, n_best)
        else:
            logger.info("  %s/%s: %d folds cv_mae=%.4f cv_logloss=%.4f avg_best_iter=%d",
                        pos_group, model_name, len(fold_maes), avg_mae, avg_logloss, n_best)
    else:
        logger.info("  %s/%s: %d folds cv_mae=%.4f avg_best_iter=%d",
                    pos_group, model_name, len(fold_maes), avg_mae, n_best)

    # Phase 1.5 — held-out latest fold: refit one model on the holdout fold's
    # train window only, evaluate on its val window. This is the honest
    # generalization metric. Tuning never touched these years.
    if holdout_fold is not None and holdout_fold not in tuning_folds:
        h_train_end, h_val_start, h_val_end = holdout_fold
        out = _fit_eval_fold(h_train_end, h_val_start, h_val_end)
        if out is not None:
            _, h_preds, h_y, _ = out
            holdout_mae = float(np.mean(np.abs(h_preds - h_y)))
            metrics["holdout_train_end"] = h_train_end
            metrics["holdout_val_start"] = h_val_start
            metrics["holdout_val_end"] = h_val_end
            metrics["holdout_mae"] = holdout_mae
            extra = ""
            if model_name == "role":
                eps = 1e-7
                p_clip = np.clip(h_preds, eps, 1.0 - eps)
                y_clip = np.clip(h_y, 0.0, 1.0)
                metrics["holdout_logloss"] = float(
                    -np.mean(y_clip * np.log(p_clip) + (1.0 - y_clip) * np.log(1.0 - p_clip))
                )
                y_bin = (h_y > 0.5).astype(int)
                if 0 < y_bin.sum() < len(y_bin):
                    from sklearn.metrics import roc_auc_score
                    metrics["holdout_auc"] = float(roc_auc_score(y_bin, h_preds))
                    extra = f" auc={metrics['holdout_auc']:.4f} logloss={metrics['holdout_logloss']:.4f}"
                else:
                    extra = f" logloss={metrics['holdout_logloss']:.4f}"
            from scipy.stats import spearmanr
            try:
                rho, _ = spearmanr(h_preds, h_y)
                if rho is not None and not np.isnan(rho):
                    metrics["holdout_spearman"] = float(rho)
                    extra += f" spearman={rho:.4f}"
            except Exception:
                pass
            logger.info("  %s/%s: HOLDOUT [%d-%d]: mae=%.4f%s (n_train=%d, n_val=%d)",
                        pos_group, model_name, h_val_start, h_val_end,
                        holdout_mae, extra,
                        int((yr_series <= h_train_end).sum()),
                        int(((yr_series >= h_val_start) & (yr_series <= h_val_end)).sum()))
        else:
            logger.warning("  %s/%s: holdout fold [%d-%d] skipped (insufficient samples)",
                           pos_group, model_name, h_val_start, h_val_end)

    # Phase 2 — final deployed model on all data up to _FINAL_TRAIN_CUTOFF.
    # This is the model used to score current prospects; it sees holdout-
    # window training data. Holdout metrics above measure generalization
    # using a model trained ONLY on data prior to the holdout val window.
    full_train = df[df["last_college_season_year"].fillna(0) <= _FINAL_TRAIN_CUTOFF]
    model = lgb.LGBMRegressor(**{
        **params, "n_estimators": n_best,
        **_build_monotone_kwargs(feature_cols, pos_group, model_name),
    })
    full_w = _board_row_weights(full_train, model_name)
    model.fit(
        full_train[feature_cols].to_numpy(dtype=float, na_value=np.nan),
        full_train[effective_target].fillna(0.0).values,
        sample_weight=full_w,
        feature_name="auto",
    )
    metrics["n_full_train"] = len(full_train)
    logger.info("  %s/%s: final model on %d rows (cutoff=%d)",
                pos_group, model_name, len(full_train), _FINAL_TRAIN_CUTOFF)

    # Quantile companion models for the impact head — produces p10/p50/p90
    # prediction intervals so the frontend can flag high-variance prospects.
    # Cheap: 3 extra fits per position with the same params as the mean model.
    # NOTE: LightGBM's quantile objective is incompatible with monotone_constraints;
    # we deliberately skip them here. The median model (above) carries the
    # monotone shape; quantile heads supply uncertainty bounds around it.
    quantile_models: dict[str, object] = {}
    if model_name == "impact":
        quantile_params = {**params, "n_estimators": n_best,
                            "objective": "quantile", "verbose": -1}
        for alpha, suffix in [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]:
            try:
                qm = lgb.LGBMRegressor(**{**quantile_params, "alpha": alpha})
                qm.fit(
                    full_train[feature_cols].to_numpy(dtype=float, na_value=np.nan),
                    full_train[effective_target].fillna(0.0).values,
                    sample_weight=full_w,
                    feature_name="auto",
                )
                quantile_models[suffix] = qm
            except Exception as exc:
                logger.warning("  %s/%s/%s quantile fit failed: %s",
                               pos_group, model_name, suffix, exc)
        if quantile_models:
            metrics["quantile_alphas"] = list(quantile_models.keys())

    # Top-20 feature importances for the experiment log.
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(feature_cols):
            order = np.argsort(importances)[::-1][:20]
            metrics["feature_importance_top20"] = [
                {"feature": feature_cols[i], "importance": int(importances[i])}
                for i in order
            ]
    except Exception:
        pass

    oof = (pd.concat(oof_records, ignore_index=True)
           if oof_records else pd.DataFrame())
    return model, feature_cols, metrics, quantile_models, oof


def _run_hurdle_ab(snapshot: pd.DataFrame, labels: pd.DataFrame) -> None:
    """A/B: single-stage impact_wav (zero-padded mixture regression) vs a
    two-stage hurdle (role gate × bridge-only conditional magnitude).

    Both arms are scored against the IDENTICAL zero-padded mixture target on
    identical expanding-window fold-val rows (holdout excluded), so the MAE
    comparison is direct. Also reports bridge-only Spearman — ranking within
    plausible NFL players is where the hurdle should differ most.

    Caveats (lower-bounds the hurdle): the conditional head borrows the
    mixture head's tuned params, and the gate uses raw clipped role
    predictions (no isotonic, which isn't fold-refittable here cheaply).
    """
    tuning_folds, _ = _split_holdout(_expanding_window_folds())
    results = []
    for pos_group in POSITION_GROUPS.keys():
        role_cols, role_df = _resolve_fit_features(
            snapshot, labels, pos_group, "role_target", "role")
        wav_cols, wav_df = _resolve_fit_features(
            snapshot, labels, pos_group, "impact_w_av_per_game", "impact_wav")
        if not role_cols or not wav_cols or wav_df is None or len(wav_df) < 50:
            continue
        role_params = {**LGBM_PARAMS, **(_load_tuned_params(pos_group, "role") or {})}
        wav_params = {**LGBM_PARAMS, **(_load_tuned_params(pos_group, "impact_wav") or {})}
        cond_df = wav_df[wav_df["is_bridge"].astype(bool)]

        yr_wav = wav_df["last_college_season_year"].fillna(0)
        yr_role = role_df["last_college_season_year"].fillna(0)
        yr_cond = cond_df["last_college_season_year"].fillna(0)

        f_mix, f_prod, f_rho_mix, f_rho_prod = [], [], [], []
        for train_end, val_start, val_end in tuning_folds:
            v_mask = (yr_wav >= val_start) & (yr_wav <= val_end)
            f_val = wav_df[v_mask]
            if len(f_val) < 10:
                continue
            y_val = f_val["impact_w_av_per_game"].fillna(0.0).values

            def _fit(df_t, cols, target, params, mname):
                tr = df_t[df_t["last_college_season_year"].fillna(0) <= train_end]
                if len(tr) < 30:
                    return None
                mk = lambda: lgb.LGBMRegressor(**{
                    **params, **_build_monotone_kwargs(cols, pos_group, mname)})
                m, _bi = _fit_fold_honest(
                    mk, tr[cols].to_numpy(dtype=float, na_value=np.nan),
                    tr[target].fillna(0.0).values,
                    tr["last_college_season_year"].fillna(0).values)
                return m

            m_mix = _fit(wav_df, wav_cols, "impact_w_av_per_game", wav_params, "impact_wav")
            m_cond = _fit(cond_df, wav_cols, "impact_w_av_per_game", wav_params, "impact_wav")
            m_role = _fit(role_df, role_cols, "role_target", role_params, "role")
            if m_mix is None or m_cond is None or m_role is None:
                continue

            X_wav = f_val[wav_cols].to_numpy(dtype=float, na_value=np.nan)
            X_role = f_val[role_cols].to_numpy(dtype=float, na_value=np.nan)
            p_mix = m_mix.predict(X_wav)
            p_role = np.clip(m_role.predict(X_role), 0.0, 1.0)
            p_cond = np.clip(m_cond.predict(X_wav), 0.0, None)
            p_prod = p_role * p_cond

            f_mix.append(float(np.mean(np.abs(p_mix - y_val))))
            f_prod.append(float(np.mean(np.abs(p_prod - y_val))))
            b_mask = f_val["is_bridge"].astype(bool).values
            if b_mask.sum() >= 10 and np.std(y_val[b_mask]) > 0:
                from scipy.stats import spearmanr
                f_rho_mix.append(float(spearmanr(p_mix[b_mask], y_val[b_mask]).statistic))
                f_rho_prod.append(float(spearmanr(p_prod[b_mask], y_val[b_mask]).statistic))

        if not f_mix:
            continue
        rec = {
            "pos": pos_group, "n_folds": len(f_mix),
            "mae_mixture": float(np.mean(f_mix)),
            "mae_hurdle": float(np.mean(f_prod)),
            "mae_delta_pct": (np.mean(f_prod) - np.mean(f_mix)) / np.mean(f_mix) * 100,
            "rho_mixture": float(np.mean(f_rho_mix)) if f_rho_mix else float("nan"),
            "rho_hurdle": float(np.mean(f_rho_prod)) if f_rho_prod else float("nan"),
        }
        results.append(rec)
        logger.info("  %s: mixture MAE %.5f vs hurdle %.5f (%+.2f%%) | "
                    "bridge ρ %.3f vs %.3f",
                    pos_group, rec["mae_mixture"], rec["mae_hurdle"],
                    rec["mae_delta_pct"], rec["rho_mixture"], rec["rho_hurdle"])

    out = EXPERIMENTS_DIR / f"hurdle_ab_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    wins = sum(1 for r in results if r["mae_delta_pct"] < 0)
    rho_wins = sum(1 for r in results
                   if not np.isnan(r["rho_hurdle"]) and r["rho_hurdle"] > r["rho_mixture"])
    logger.info("Hurdle A/B: MAE wins %d/%d, bridge-Spearman wins %d/%d → %s",
                wins, len(results), rho_wins, len(results), out)


def _run_label_ab(snapshot: pd.DataFrame, labels_a: pd.DataFrame,
                   variants: dict, seeds: int = 2) -> None:
    """A/B: legacy second-contract labels vs replacement-gated labels.

    The problem this has to work around: arm B's target is on a different scale
    (gated and rebased above replacement), so MAE is not comparable between
    arms — the L1 lesson, where a "win" was purely definitional. Both arms are
    therefore scored against `label_apy_raw`, the ungated, un-rebased APY share
    of the defect-fixed contract selection, which is IDENTICAL in both label
    files. Ranking metrics on an identical population and an identical ground
    truth are the only honest comparison available.

    Arms:
      A       legacy labels (current production)
      B       replacement-gated labels
      A/A     legacy labels refit with a different seed — the placebo that sets
              the noise floor. Without it a B-vs-A gap of 0.01 is unreadable.

    Metrics, all bridge-only on identical fold-val rows:
      rho_vs_raw    Spearman against label_apy_raw (PRIMARY)
      decile_lift   mean label_apy_raw in the top prediction decile ÷ overall
      role_logloss  role head, comparable across arms (target stays in [0,1])
    """
    from scipy.stats import spearmanr

    tuning_folds, _ = _split_holdout(_expanding_window_folds())
    arms = [("A_legacy", labels_a, 42)]
    arms += [(name, frame, 42) for name, frame in variants.items()]
    arms.append(("AA_placebo", labels_a, 1337))
    if seeds > 2:
        arms.append(("AA_placebo2", labels_a, 2024))
    variant_names = list(variants)

    results = []
    for pos_group in POSITION_GROUPS.keys():
        # Ground truth comes from arm A's file but is identical in both.
        truth_src = labels_a[["player_id", "label_apy_raw"]].drop_duplicates("player_id")

        per_arm: dict[str, dict] = {}
        for arm_name, lab, seed in arms:
            if "label_apy_raw" not in lab.columns:
                logger.error("labels for %s lack label_apy_raw — regenerate "
                             "with the current step 7", arm_name)
                return
            cols, df = _resolve_fit_features(snapshot, lab, pos_group,
                                             "impact_target", "impact")
            if not cols or df is None or len(df) < 50:
                per_arm = {}
                break
            df = df.merge(truth_src, on="player_id", how="left")
            params = {**LGBM_PARAMS,
                      **(_load_tuned_params(pos_group, "impact") or {}),
                      "random_state": seed}
            yr = df["last_college_season_year"].fillna(0)

            rhos, lifts, maes, val_target_sig = [], [], [], []
            for train_end, val_start, val_end in tuning_folds:
                tr = df[yr <= train_end]
                va = df[(yr >= val_start) & (yr <= val_end)]
                if len(tr) < 30 or len(va) < 10:
                    continue
                mk = lambda: lgb.LGBMRegressor(**{
                    **params,
                    **_build_monotone_kwargs(cols, pos_group, "impact")})
                m, _ = _fit_fold_honest(
                    mk, tr[cols].to_numpy(dtype=float, na_value=np.nan),
                    tr["impact_target"].fillna(0.0).values,
                    tr["last_college_season_year"].fillna(0).values)
                pred = m.predict(va[cols].to_numpy(dtype=float, na_value=np.nan))

                b = va["is_bridge"].astype(bool).values
                truth = va["label_apy_raw"].fillna(0.0).values
                if b.sum() < 10 or np.std(truth[b]) < 1e-12:
                    continue
                r = spearmanr(pred[b], truth[b]).statistic
                if np.isfinite(r):
                    rhos.append(float(r))
                # MAE against the arm's own validation targets. Only
                # comparable across arms when those targets are identical —
                # true when a variant changes training rows only (imputation),
                # false when it re-defines the label everywhere (the gate).
                # The caller checks and suppresses it when unsafe.
                maes.append(float(np.mean(np.abs(pred - va["impact_target"]
                                                 .fillna(0.0).values))))
                val_target_sig.append(
                    float(np.nansum(va["impact_target"].fillna(0.0).values)))
                # Decile lift: does the top-scoring 10% actually earn more?
                pb, tb = pred[b], truth[b]
                k = max(1, int(round(0.10 * len(pb))))
                top = np.argsort(-pb)[:k]
                base = tb.mean()
                if base > 0:
                    lifts.append(float(tb[top].mean() / base))
            if not rhos:
                per_arm = {}
                break
            per_arm[arm_name] = {
                "rho_vs_raw": float(np.mean(rhos)),
                "decile_lift": float(np.mean(lifts)) if lifts else float("nan"),
                "mae": float(np.mean(maes)) if maes else float("nan"),
                "val_target_sig": tuple(np.round(val_target_sig, 6)),
                "n_folds": len(rhos),
            }

        if not per_arm or "A_legacy" not in per_arm:
            continue
        if any(v not in per_arm for v in variant_names):
            continue
        a = per_arm["A_legacy"]
        placebos = [v["rho_vs_raw"] for k, v in per_arm.items()
                    if k.startswith("AA_")]
        noise = (max(abs(p - a["rho_vs_raw"]) for p in placebos)
                 if placebos else float("nan"))
        rec = {"pos": pos_group, "rho_legacy": a["rho_vs_raw"],
               "lift_legacy": a["decile_lift"], "mae_legacy": a["mae"],
               "placebo_noise": noise, "n_folds": a["n_folds"]}
        parts = []
        for v in variant_names:
            # cv_mae is only meaningful if this arm's validation targets match
            # the baseline's; otherwise the two arms are scored on different
            # questions and the comparison is definitional, not empirical.
            same_val = per_arm[v]["val_target_sig"] == a["val_target_sig"]
            rec[f"mae_comparable_{v}"] = bool(same_val)
            if same_val and np.isfinite(a["mae"]) and a["mae"] > 0:
                rec[f"mae_{v}"] = per_arm[v]["mae"]
                rec[f"mae_delta_pct_{v}"] = (
                    (per_arm[v]["mae"] - a["mae"]) / a["mae"] * 100)
            d = per_arm[v]["rho_vs_raw"] - a["rho_vs_raw"]
            rec[f"rho_{v}"] = per_arm[v]["rho_vs_raw"]
            rec[f"delta_{v}"] = d
            rec[f"beats_noise_{v}"] = bool(np.isfinite(noise) and abs(d) > noise)
            rec[f"lift_{v}"] = per_arm[v]["decile_lift"]
            mtxt = ("" if f"mae_delta_pct_{v}" not in rec
                    else " mae%+.2f%%" % rec[f"mae_delta_pct_{v}"])
            parts.append("%s=%.4f (Δ%+.4f)%s%s" % (
                v, per_arm[v]["rho_vs_raw"], d,
                "*" if rec[f"beats_noise_{v}"] else " ", mtxt))
        results.append(rec)
        logger.info("  %-5s legacy=%.4f | %s | placebo ±%.4f",
                    pos_group, a["rho_vs_raw"], "  ".join(parts), noise)

    out = EXPERIMENTS_DIR / f"label_ab_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    for v in variant_names:
        wins = sum(1 for r in results if r[f"delta_{v}"] > 0)
        signif = sum(1 for r in results
                     if r[f"beats_noise_{v}"] and r[f"delta_{v}"] > 0)
        losses = sum(1 for r in results
                     if r[f"beats_noise_{v}"] and r[f"delta_{v}"] < 0)
        logger.info("Label A/B [%s]: beats legacy on %d/%d heads "
                    "(%d above placebo noise, %d significant losses)",
                    v, wins, len(results), signif, losses)
    logger.info("→ %s", out)


def _run_cohort_ab(snapshot: pd.DataFrame, labels: pd.DataFrame,
                    heads: tuple = ("impact", "role")) -> None:
    """A/B: does dropping or down-weighting early bridge cohorts help?

    Motivation. OverTheCap's coverage of SMALL contracts improved enormously
    over time, so the second-contract label does not mean the same thing in
    2005 as in 2018. Measured against career w_AV — an independent outcome
    from PFR — the share of bridge players labelled "no second contract" is:

        career w_AV   2004-06   2007-10   2011-19
        0-5             99.1%     90.4%     11.6%
        5-15            92.6%     61.8%      9.3%
        15-30           77.3%     25.3%      2.1%
        30+             34.0%      2.0%      0.6%

    A third of 2004-06 players with w_AV >= 30 — clear multi-year starters —
    are labelled washouts. That is a coverage artifact, not an outcome.

    This is a clean A/B in a way the label-gate A/B was not: every affected
    class has last_college_season_year <= 2009, while the earliest CV fold
    validates on 2010-2012. So the arms differ ONLY in training rows, the
    validation rows and their targets are byte-identical, and MAE is directly
    comparable between arms with no rescaling caveat.

    Arms: baseline; drop bridge rows drafted <= 2006 (severe era); drop <=
    2010 (all suspect); down-weight <= 2010; plus an A/A placebo.

    NOTE: only BRIDGE rows are ever touched. The zero-padded non-bridge
    universe is load-bearing and is left alone in every arm.
    """
    from scipy.stats import spearmanr

    # The production early-cohort filter must be OFF inside this harness, or
    # the "baseline" arm arrives pre-filtered and every arm reads as a no-op.
    global IMPACT_LABEL_MIN_DRAFT_YEAR
    _saved_cutoff = IMPACT_LABEL_MIN_DRAFT_YEAR
    IMPACT_LABEL_MIN_DRAFT_YEAR = None
    try:
        _run_cohort_ab_inner(snapshot, labels, heads)
    finally:
        IMPACT_LABEL_MIN_DRAFT_YEAR = _saved_cutoff


def _run_cohort_ab_inner(snapshot: pd.DataFrame, labels: pd.DataFrame,
                          heads: tuple) -> None:
    from scipy.stats import spearmanr

    tuning_folds, _ = _split_holdout(_expanding_window_folds())
    dy = labels[["player_id", "draft_year"]].drop_duplicates("player_id")

    ARMS = [
        ("baseline",          None,  None, 42),
        ("drop_pre2007",      2006,  0.0,  42),
        ("drop_pre2011",      2010,  0.0,  42),
        ("downweight_pre2011", 2010, 0.25, 42),
        ("AA_placebo",        None,  None, 1337),
    ]

    results = []
    for model_name in heads:
        target_col = MODEL_TARGET_COLS[model_name]
        logger.info("── cohort A/B: %s head ──", model_name)
        for pos_group in POSITION_GROUPS.keys():
            cols, df = _resolve_fit_features(snapshot, labels, pos_group,
                                             target_col, model_name)
            if not cols or df is None or len(df) < 50:
                continue
            if "draft_year" not in df.columns:
                df = df.merge(dy, on="player_id", how="left")
            params_base = {**LGBM_PARAMS,
                           **(_load_tuned_params(pos_group, model_name) or {})}
            yr = df["last_college_season_year"].fillna(0)
            is_bridge = df["is_bridge"].astype(bool)

            # Every arm must be scored on the SAME folds. Each arm's filter
            # starves a different set of early folds of bridge rows, and a
            # starved fold posts a deceptively good MAE by collapsing to zero,
            # so per-arm fold sets would compare arms on different problems.
            # Use the intersection: the folds usable by the most restrictive
            # arm.
            arm_folds = []
            for _a, cutoff, weight, _s in ARMS:
                if cutoff is None or weight != 0.0:
                    arm_folds.append(set(tuning_folds))
                    continue
                sub = df[~(df["is_bridge"].astype(bool)
                           & (df["draft_year"].fillna(9999) <= cutoff))]
                arm_folds.append(set(_usable_folds(
                    sub, tuning_folds, "impact_target", pos_group, model_name,
                    force=True)
                    if target_col == "impact_target" else tuning_folds))
            common = [f for f in tuning_folds if all(f in s_ for s_ in arm_folds)]
            if not common:
                continue

            per_arm = {}
            for arm, cutoff, weight, seed in ARMS:
                params = {**params_base, "random_state": seed}
                maes, rhos = [], []
                for train_end, val_start, val_end in common:
                    tr_mask = yr <= train_end
                    va = df[(yr >= val_start) & (yr <= val_end)]
                    if tr_mask.sum() < 30 or len(va) < 10:
                        continue
                    tr = df[tr_mask]
                    w = np.ones(len(tr))
                    if cutoff is not None:
                        old = (tr["is_bridge"].astype(bool)
                               & (tr["draft_year"].fillna(9999) <= cutoff)).values
                        if weight == 0.0:
                            tr, w = tr[~old], w[~old]
                        else:
                            w[old] = weight
                        if len(tr) < 30:
                            continue
                    mk = lambda: lgb.LGBMRegressor(**{
                        **params,
                        **_build_monotone_kwargs(cols, pos_group, model_name)})
                    m, _ = _fit_fold_honest(
                        mk, tr[cols].to_numpy(dtype=float, na_value=np.nan),
                        tr[target_col].fillna(0.0).values,
                        tr["last_college_season_year"].fillna(0).values,
                        sample_weight=w)
                    pred = m.predict(va[cols].to_numpy(dtype=float, na_value=np.nan))
                    y = va[target_col].fillna(0.0).values
                    maes.append(float(np.mean(np.abs(pred - y))))
                    b = va["is_bridge"].astype(bool).values
                    if b.sum() >= 10 and np.std(y[b]) > 1e-12:
                        r = spearmanr(pred[b], y[b]).statistic
                        if np.isfinite(r):
                            rhos.append(float(r))
                if not maes:
                    per_arm = {}
                    break
                per_arm[arm] = {"mae": float(np.mean(maes)),
                                "rho": float(np.mean(rhos)) if rhos else float("nan"),
                                "n_folds": len(maes)}
            if not per_arm or "baseline" not in per_arm:
                continue

            base = per_arm["baseline"]
            noise = abs(per_arm["AA_placebo"]["mae"] - base["mae"]) \
                if "AA_placebo" in per_arm else float("nan")
            rec = {"head": model_name, "pos": pos_group,
                   "n_common_folds": len(common),
                   "mae_baseline": base["mae"], "rho_baseline": base["rho"],
                   "placebo_mae_noise": noise, "n_folds": base["n_folds"]}
            parts = []
            for arm, cutoff, weight, seed in ARMS:
                if arm in ("baseline", "AA_placebo") or arm not in per_arm:
                    continue
                d = (per_arm[arm]["mae"] - base["mae"]) / base["mae"] * 100
                rec[f"mae_{arm}"] = per_arm[arm]["mae"]
                rec[f"mae_delta_pct_{arm}"] = d
                rec[f"rho_{arm}"] = per_arm[arm]["rho"]
                rec[f"beats_noise_{arm}"] = bool(
                    np.isfinite(noise) and
                    abs(per_arm[arm]["mae"] - base["mae"]) > noise)
                parts.append("%s %+.2f%%%s" % (
                    arm, d, "*" if rec[f"beats_noise_{arm}"] else " "))
            results.append(rec)
            logger.info("  %-5s mae=%.5f | %s | placebo ±%.5f",
                        pos_group, base["mae"], "  ".join(parts), noise)

    out = EXPERIMENTS_DIR / f"cohort_ab_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    for arm, cutoff, weight, seed in ARMS:
        if arm in ("baseline", "AA_placebo"):
            continue
        key = f"mae_delta_pct_{arm}"
        rows = [r for r in results if key in r]
        for head in heads:
            hr = [r for r in rows if r["head"] == head]
            if not hr:
                continue
            wins = sum(1 for r in hr if r[key] < 0)
            sig_w = sum(1 for r in hr if r[key] < 0 and r[f"beats_noise_{arm}"])
            sig_l = sum(1 for r in hr if r[key] > 0 and r[f"beats_noise_{arm}"])
            logger.info("Cohort A/B [%s / %s]: MAE improves %d/%d "
                        "(%d above placebo noise, %d significant losses), "
                        "mean %+.2f%%", arm, head, wins, len(hr), sig_w, sig_l,
                        float(np.mean([r[key] for r in hr])))
    logger.info("→ %s", out)


def main():
    if not _lgbm_available:
        logger.error("lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Fast test run: recent cohorts only, 50 estimators")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter random search before training")
    parser.add_argument("--tune-trials", type=int, default=30,
                        help="Random search trials per (pos_group, model_name)")
    parser.add_argument("--tune-seeds", type=int, default=1,
                        help="Number of independent Optuna seeds per head; the "
                             "winner is the seed with the lowest CV. Costs N× "
                             "wall time for tuning. Default 1 (legacy).")
    parser.add_argument("--tune-only", type=str, default=None,
                        choices=["impact", "impact_wav"],
                        help="Restrict tuning to a single model name. "
                             "Default (omitted) tunes both heads.")
    parser.add_argument("--boruta", action="store_true",
                        help="Run Boruta as an AUDIT pass (writes "
                             "boruta_features.json). Does not gate the fit — "
                             "drop candidates are promoted manually into "
                             "config.BORUTA_DROP_FEATURES after inspecting "
                             "experiments/boruta_summary.py.")
    parser.add_argument("--boruta-trials", type=int, default=BORUTA_MAX_ITER,
                        help="Boruta max_iter per head (default: BORUTA_MAX_ITER)")
    parser.add_argument("--mi-filter", action="store_true",
                        help="Compute MI ranking on the PFF state-space block "
                             "and cache top-N per (pos × model). Subsequent fits "
                             "filter state-space cols to the cached subset; "
                             "other categories pass through. Cache lives at "
                             "data/processed/draft_models/mi_features.json.")
    parser.add_argument("--hurdle-ab", action="store_true",
                        help="Run the impact_wav hurdle A/B (role gate × "
                             "conditional magnitude vs single-stage mixture) "
                             "and exit — no models written")
    parser.add_argument("--cohort-ab", action="store_true",
                        help="Run the early-cohort A/B (drop / down-weight "
                             "bridge rows from draft classes whose contract "
                             "coverage is unreliable) and exit — no models "
                             "written")
    parser.add_argument("--label-ab", metavar="NAME=PATH", nargs="+",
                        help="Run the label A/B: legacy vs one or more variant "
                             "label files, plus an A/A placebo arm. Each "
                             "argument is NAME=PATH. Exits without writing "
                             "models.")
    parser.add_argument("--mi-top-n", type=int, default=MI_TOP_N,
                        help="Top-N state-space cols to keep per head (default: MI_TOP_N)")
    args = parser.parse_args()

    labels_path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    if not labels_path.exists():
        logger.error("%s not found — run earlier steps", labels_path)
        sys.exit(1)

    # Shared assembly: attrs merge (ATTRIBUTE_MERGE_COLS with overlap
    # handling), age_at_draft, team_sack_share, PFF career, QB career,
    # OL sub-position fill/normalization, combine, hasan (year-sanity
    # nulling), NFL league averages, per-position derived features.
    snapshot = assemble_snapshot(include_derived=True, logger=logger)
    labels = pd.read_parquet(labels_path)

    if args.test:
        if "last_college_season_year" in labels.columns:
            max_year = int(labels["last_college_season_year"].dropna().max())
            labels = labels[labels["last_college_season_year"].fillna(0) >= max_year - 4]
        logger.info("Test mode: %d label rows", len(labels))

    models_dir = PROCESSED_DIR / "draft_models"
    models_dir.mkdir(exist_ok=True)

    if args.hurdle_ab:
        logger.info("Running hurdle A/B (no models will be written)...")
        _run_hurdle_ab(snapshot, labels)
        return

    if args.cohort_ab:
        logger.info("Running early-cohort A/B (no models will be written)...")
        _run_cohort_ab(snapshot, labels)
        return

    if args.label_ab:
        variants = {}
        for spec in args.label_ab:
            if "=" not in spec:
                logger.error("--label-ab expects NAME=PATH, got %r", spec)
                sys.exit(1)
            name, _, path = spec.partition("=")
            p = Path(path)
            if not p.exists():
                logger.error("%s not found — generate it with "
                             "`python3 pipeline/07_nfl_labels.py "
                             "--labels-variant <variant>`", p)
                sys.exit(1)
            variants[name] = pd.read_parquet(p)
        logger.info("Running label A/B: legacy vs %s (no models written)...",
                    ", ".join(variants))
        _run_label_ab(snapshot, labels, variants)
        return

    # MI filter runs FIRST so the cache is on disk by the time tuning + final
    # fit call _filter_state_space_to_mi. MI is deterministic per (pos × target)
    # and gates the actual fit, unlike --boruta which is audit-only.
    if args.mi_filter:
        logger.info("Running MI feature filter (top_n=%d)...", args.mi_top_n)
        _run_mi_for_all_heads(snapshot, labels, args.mi_top_n)

    # Boruta is an audit pass — writes boruta_features.json for
    # experiments/boruta_summary.py to consume. The fit pipeline ignores it;
    # drop decisions live in config.BORUTA_DROP_FEATURES (curated by hand).
    if args.boruta:
        logger.info("Running Boruta audit pass (max_iter=%d)...",
                    args.boruta_trials)
        _run_boruta_for_all_heads(snapshot, labels, args.boruta_trials)

    if args.tune:
        logger.info("Tuning hyperparameters (%d trials per model)...", args.tune_trials)
        # Preserve any existing tuned entries (so partial re-tunes work).
        tuned: dict[str, dict] = {}
        if TUNED_PARAMS_PATH.exists():
            try:
                tuned = json.load(open(TUNED_PARAMS_PATH))
            except Exception:
                tuned = {}
        for pos_group in POSITION_GROUPS.keys():
            for target_col, model_name in HEADS:
                if args.tune_only and model_name != args.tune_only:
                    continue
                logger.info("  Tuning %s/%s%s", pos_group, model_name,
                            f"  (n_seeds={args.tune_seeds})" if args.tune_seeds > 1 else "")
                best = tune_hyperparams(
                    snapshot, labels, pos_group, target_col, model_name,
                    n_trials=args.tune_trials,
                    n_seeds=args.tune_seeds,
                )
                if best:
                    tuned[f"{pos_group}/{model_name}"] = best
                    logger.info("  %s/%s tuned: %s", pos_group, model_name, best)
                else:
                    logger.warning("  %s/%s: no valid tuning result", pos_group, model_name)
        TUNED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TUNED_PARAMS_PATH, "w") as f:
            json.dump(tuned, f, indent=2)
        logger.info("Saved %d tuned param sets → %s", len(tuned), TUNED_PARAMS_PATH)
        _write_tuned_params_manifest(snapshot, labels, tuned, [TUNED_PARAMS_PATH])

    all_metrics: dict[str, dict] = {}
    all_oof_frames: list[pd.DataFrame] = []

    for pos_group in POSITION_GROUPS.keys():
        logger.info("Training draft models for position group: %s", pos_group)

        for target_col, model_name in HEADS:
            try:
                model, feature_cols, metrics, quantile_models, oof = \
                    train_model_for_group(
                        snapshot, labels, pos_group, target_col, model_name,
                        test_mode=args.test,
                    )
                if not oof.empty:
                    all_oof_frames.append(oof)

                bundle = {"model": model, "feature_cols": feature_cols,
                          "pos_group": pos_group, "target": target_col}
                if quantile_models:
                    bundle["quantile_models"] = quantile_models
                out_path = models_dir / f"{pos_group}_{model_name}_model.pkl"
                with open(out_path, "wb") as f:
                    pickle.dump(bundle, f)

                all_metrics[f"{pos_group}_{model_name}"] = metrics
                logger.info("  Saved → %s", out_path)

            except ValueError as e:
                logger.warning("  Skipped %s/%s: %s", pos_group, model_name, e)
            except Exception as e:
                logger.error("  Failed %s/%s: %s", pos_group, model_name, e)

    # Persist tuning-fold OOF predictions for step 8c (holdout excluded).
    if all_oof_frames:
        oof_all = pd.concat(all_oof_frames, ignore_index=True)
        oof_path = models_dir / "oof_predictions.parquet"
        oof_all.to_parquet(oof_path, index=False)
        logger.info("Saved %d OOF predictions → %s", len(oof_all), oof_path)

    metrics_path = models_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved training metrics → %s", metrics_path)

    # Provenance sidecar — training_metrics.json is iterated by validators
    # (src/validation.py) and audit scripts as a flat {key: metrics} dict;
    # don't wrap it. Stamp via a sibling so readers comparing cv_mae across
    # runs can tell whether numbers came from a fresh tune vs cached params.
    metrics_mode = "fresh" if args.tune else "cached"
    provenance_path = models_dir / "training_metrics_provenance.json"
    provenance_payload = tag_summary(
        {"n_models": len(all_metrics), "tune_flag": bool(args.tune),
         "tune_trials": int(args.tune_trials) if args.tune else None},
        mode_used=metrics_mode, step_name=STEP_NAME,
    )
    provenance_path.write_text(json.dumps(provenance_payload, indent=2))
    logger.info("Provenance → %s (_tuning_mode=%s)", provenance_path, metrics_mode)

    # Experiment log: full snapshot of config + per-model metrics + feature
    # importances. One file per Step-8 invocation, indexed by timestamp.
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_snapshot = {
        "ROLE_LABEL_HORIZON_YEARS": ROLE_LABEL_HORIZON_YEARS,
        "DRAFT_TRAIN_CUTOFF_YEAR": DRAFT_TRAIN_CUTOFF_YEAR,
        "DRAFT_VAL_START_YEAR": DRAFT_VAL_START_YEAR,
        "FINAL_TRAIN_CUTOFF": _FINAL_TRAIN_CUTOFF,
        "START_YEAR": START_YEAR,
        "END_YEAR": END_YEAR,
        "LGBM_PARAMS_DEFAULTS": LGBM_PARAMS,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_snapshot, sort_keys=True).encode()
    ).hexdigest()[:12]
    exp = {
        "run_id": f"{ts}_{config_hash}",
        "timestamp": ts,
        "config_hash": config_hash,
        "config": config_snapshot,
        "args": {
            "test": bool(args.test),
            "tune": bool(args.tune),
            "tune_trials": int(args.tune_trials),
        },
        "models": all_metrics,
    }
    exp_path = EXPERIMENTS_DIR / f"{ts}_{config_hash}.json"
    with open(exp_path, "w") as f:
        json.dump(exp, f, indent=2)
    logger.info("Saved experiment log → %s", exp_path)

    logger.info("Step 8 complete.")


if __name__ == "__main__":
    main()
