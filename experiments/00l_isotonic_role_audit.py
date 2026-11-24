"""
00l — Isotonic calibration audit for the role head (per position).

Replicates the cdarko/pipeline/09c2 pattern: stitch OOF predictions across
every expanding-window CV fold, fit IsotonicRegression on them, and report
whether the calibrated outputs improve Brier / log-loss / MAE / bias
relative to the raw LightGBM scores. Platt fallback (positive-coef Ridge
mapping into [0, 1]) when n_positive < PLATT_FALLBACK_THRESHOLD.

The audit answers two questions per position:

  1. In-sample fit: does fitting isotonic on stitched OOF preds reduce
     bias / Brier / log-loss when re-scored on the same OOF preds? This
     is what cdarko reports; it is the "best case" delta.

  2. Honest fit (leave-one-fold-out): for each fold k, fit isotonic on
     the OTHER folds' OOF preds and score fold k. Average the metrics.
     Closer to what a deployed calibrator would deliver on a future
     cohort.

A small (1) AND a small (2) means isotonic is not buying anything for that
head. A meaningful (2) is the threshold for promoting the technique into
step 8 / a new step 8c.

Usage:
    python experiments/00l_isotonic_role_audit.py
    python experiments/00l_isotonic_role_audit.py --positions QB,RB,WR
    python experiments/00l_isotonic_role_audit.py --target impact   # role|impact|impact_wav
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ATTRIBUTE_MERGE_COLS, LGBM_PARAMS, NFL_LG_FEATURES, POSITION_GROUPS,
    PROCESSED_DIR, RAW_DIR,
)
from src.scrapers.pff import load_pff_career_features

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import step 8 helpers (folds, feature build, prep, tuned params).
_spec = importlib.util.spec_from_file_location(
    "_p8", ROOT / "pipeline" / "08_draft_models.py",
)
_p8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p8)

PLATT_FALLBACK_THRESHOLD = 50   # mirrors cdarko/09c2

TARGET_RANGES: dict[str, tuple[float, float | None]] = {
    "role":       (0.0, 1.0),
    "impact":     (0.0, None),   # APY% of cap; clamp at observed max post-fit
    "impact_wav": (0.0, None),
}

TARGET_COL: dict[str, str] = {
    "role":       "role_target",
    "impact":     "impact_target",
    "impact_wav": "impact_w_av_per_game",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isotonic calibration audit (per-pos)")
    p.add_argument("--target", default="role",
                   choices=list(TARGET_COL.keys()),
                   help="Which head to audit (default: role)")
    p.add_argument("--positions", default=None,
                   help="Comma-separated subset (default: all in POSITION_GROUPS)")
    p.add_argument("--out", default=str(ROOT / "experiments" / "isotonic_audit.parquet"),
                   help="Where to write the per-pos diagnostic table")
    return p.parse_args()


# ── Snapshot assembly ─────────────────────────────────────────────────────────
# Mirrors pipeline/08b_global_draft_model.py:main() snapshot build verbatim so
# the audit sees the exact training matrix step 8 would have constructed for
# the same (pos × target) head.

def _assemble_snapshot() -> pd.DataFrame:
    snap_path   = PROCESSED_DIR / "player_skill_snapshot.parquet"
    labels_path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    attrs_path  = PROCESSED_DIR / "player_attributes.parquet"
    for p in [snap_path, labels_path, attrs_path]:
        if not p.exists():
            logger.error("%s missing — run earlier steps", p)
            sys.exit(1)

    snapshot = pd.read_parquet(snap_path)
    attrs    = pd.read_parquet(attrs_path)
    attr_cols = [c for c in ATTRIBUTE_MERGE_COLS if c in attrs.columns]
    overlap = [c for c in attr_cols
               if c not in ("player_id", "position_group")
               and c in snapshot.columns]
    snapshot = snapshot.drop(columns=overlap, errors="ignore")
    snapshot = snapshot.merge(attrs[attr_cols].drop_duplicates("player_id"),
                              on="player_id", how="left",
                              suffixes=("", "_attrs"))
    if "position_group_attrs" in snapshot.columns:
        snapshot["position_group"] = snapshot["position_group"].fillna(
            snapshot["position_group_attrs"]
        )
        snapshot = snapshot.drop(columns=["position_group_attrs"])

    # age_at_draft
    if "last_college_season_year" in snapshot.columns:
        draft_yr = snapshot["last_college_season_year"] + 1
        ages = pd.Series(np.nan, index=snapshot.index)
        if "birth_date" in snapshot.columns:
            has_date = snapshot["birth_date"].notna()
            if has_date.any():
                bd = pd.to_datetime(snapshot.loc[has_date, "birth_date"], errors="coerce")
                may1 = pd.to_datetime(draft_yr[has_date].astype(int).astype(str) + "-05-01")
                ages[has_date] = (may1 - bd).dt.days / 365.25
        if "birth_year" in snapshot.columns:
            has_year = snapshot["birth_year"].notna() & ages.isna()
            ages[has_year] = (draft_yr[has_year] - snapshot.loc[has_year, "birth_year"]).astype(float)
        snapshot["age_at_draft"] = ages

    # Team sack share for DL/LB
    seasons_path = RAW_DIR / "player_seasons.parquet"
    if seasons_path.exists() and "player_id" in snapshot.columns:
        ps = pd.read_parquet(seasons_path,
                             columns=["player_id", "school", "season_year", "sacks"])
        ps["sacks"] = pd.to_numeric(ps["sacks"], errors="coerce").fillna(0)
        team_sacks = (ps.groupby(["school", "season_year"])["sacks"].sum().reset_index()
                      .rename(columns={"sacks": "team_sacks"}))
        ps = ps.merge(team_sacks, on=["school", "season_year"], how="left")
        ps["sack_share"] = np.where(ps["team_sacks"] > 0, ps["sacks"] / ps["team_sacks"], 0.0)
        max_share = (ps.groupby("player_id")["sack_share"].max().reset_index()
                     .rename(columns={"sack_share": "team_sack_share"}))
        snapshot = snapshot.merge(max_share, on="player_id", how="left")

    pff_career = load_pff_career_features(snapshot, PROCESSED_DIR)
    if not pff_career.empty and "player_id" in pff_career.columns:
        snapshot = snapshot.merge(pff_career, on="player_id", how="left")

    combine_path = PROCESSED_DIR / "combine_metrics.parquet"
    if combine_path.exists():
        snapshot = snapshot.merge(pd.read_parquet(combine_path),
                                  on="player_id", how="left")
        if "combine_invited" in snapshot.columns:
            snapshot["combine_invited"] = snapshot["combine_invited"].fillna(0).astype(int)

    hasan_path = PROCESSED_DIR / "hasan_ranks.parquet"
    if hasan_path.exists():
        hasan = pd.read_parquet(hasan_path)
        snapshot = snapshot.merge(
            hasan[["player_id", "draft_year", "hasan_rank"]],
            on="player_id", how="left",
        )
        if "last_college_season_year" in snapshot.columns:
            expected = snapshot["last_college_season_year"] + 1
            year_ok = (snapshot["draft_year"].isna() |
                       ((snapshot["draft_year"] - expected).abs() <= 1))
            snapshot.loc[~year_ok, "hasan_rank"] = np.nan
        snapshot = snapshot.drop(columns=["draft_year"], errors="ignore")

    nfl_lg_path = PROCESSED_DIR / "nfl_league_averages.parquet"
    if nfl_lg_path.exists() and "last_college_season_year" in snapshot.columns:
        nfl_lg = pd.read_parquet(nfl_lg_path,
                                 columns=["season_year"] + NFL_LG_FEATURES)
        snapshot = snapshot.merge(
            nfl_lg.rename(columns={"season_year": "last_college_season_year"}),
            on="last_college_season_year", how="left",
        )
    return snapshot


# ── OOF prediction generation ────────────────────────────────────────────────

def _generate_oof_preds(df: pd.DataFrame, feature_cols: list[str],
                        target_col: str, params: dict,
                        pos_group: str, model_name: str,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run expanding-window CV; return stitched (preds, targets, fold_id).

    Uses ALL folds (matches cdarko/09c2). Each fold's predictions are honest
    OOF for that fold's val window. `fold_id` lets the caller leave one out
    for the honest-fit pass.
    """
    folds = _p8._expanding_window_folds()
    yr_series = df["last_college_season_year"].fillna(0)
    base = {**LGBM_PARAMS, **params}

    monotone = _p8._build_monotone_kwargs(feature_cols, pos_group, model_name)
    preds_list, targets_list, fold_ids = [], [], []

    for k, (train_end, val_start, val_end) in enumerate(folds):
        f_train = df[yr_series <= train_end]
        f_val   = df[(yr_series >= val_start) & (yr_series <= val_end)]
        if len(f_train) < 30 or len(f_val) < 10:
            continue
        X_tr = f_train[feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y_tr = f_train[target_col].fillna(0.0).values
        X_va = f_val[feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y_va = f_val[target_col].fillna(0.0).values

        m = lgb.LGBMRegressor(**{**base, **monotone})
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
        preds = m.predict(X_va)
        preds_list.append(preds)
        targets_list.append(y_va)
        fold_ids.append(np.full(len(y_va), k, dtype=int))

    if not preds_list:
        return np.array([]), np.array([]), np.array([])
    return (np.concatenate(preds_list),
            np.concatenate(targets_list),
            np.concatenate(fold_ids))


# ── Calibrator + metrics ─────────────────────────────────────────────────────

def _fit_calibrator(preds: np.ndarray, targets: np.ndarray,
                    y_min: float, y_max: float | None):
    """Returns (method, predict_fn). Platt fallback under n_positive threshold."""
    n_positive = int((targets > 0).sum())
    if n_positive >= PLATT_FALLBACK_THRESHOLD:
        cal = IsotonicRegression(y_min=y_min, y_max=y_max,
                                 out_of_bounds="clip", increasing=True)
        cal.fit(preds, targets)
        return "isotonic", (lambda x: cal.predict(x))
    cal = Ridge(alpha=1.0, positive=True)
    cal.fit(preds.reshape(-1, 1), targets)

    def _pred(x):
        out = cal.predict(x.reshape(-1, 1))
        if y_min is not None or y_max is not None:
            out = np.clip(out, y_min, y_max)
        return out
    return "platt_ridge", _pred


def _metrics(preds: np.ndarray, targets: np.ndarray,
             is_probability: bool) -> dict:
    out = {
        "n":        int(len(preds)),
        "n_positive": int((targets > 0).sum()),
        "bias":     float(np.mean(preds - targets)),
        "mae":      float(np.mean(np.abs(preds - targets))),
        "brier":    float(np.mean((preds - targets) ** 2)),
    }
    if is_probability:
        eps = 1e-7
        p_clip = np.clip(preds, eps, 1.0 - eps)
        y_clip = np.clip(targets, 0.0, 1.0)
        out["logloss"] = float(
            -np.mean(y_clip * np.log(p_clip) + (1.0 - y_clip) * np.log(1.0 - p_clip))
        )
    else:
        out["logloss"] = None
    return out


def _audit_one(df: pd.DataFrame, pos_group: str, target_key: str) -> dict | None:
    target_col = TARGET_COL[target_key]
    model_name = target_key  # "role" / "impact" / "impact_wav"
    y_min, y_max_template = TARGET_RANGES[target_key]

    feature_cols = _p8._build_feature_cols(df, pos_group)
    if not feature_cols:
        logger.warning("  %s: no features available", pos_group)
        return None

    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    df_pos = _p8._prep_training_df(df, labels, pos_group, target_col)
    df_pos = _p8._add_derived_features(df_pos, pos_group)
    if len(df_pos) < 60:
        logger.warning("  %s/%s: only %d rows — skipping", pos_group, model_name, len(df_pos))
        return None

    tuned = _p8._load_tuned_params(pos_group, model_name) or {}

    preds, targets, fold_ids = _generate_oof_preds(
        df_pos, feature_cols, target_col, tuned, pos_group, model_name,
    )
    if len(preds) == 0:
        logger.warning("  %s/%s: no OOF preds produced", pos_group, model_name)
        return None

    # Clip raw preds to the legal range (matches step 10 inference behavior).
    raw_preds = np.clip(preds,
                         y_min if y_min is not None else None,
                         y_max_template if y_max_template is not None else None)

    is_prob = target_key == "role"
    y_max_data = float(targets.max()) if y_max_template is None else y_max_template

    # ── In-sample (cdarko 09c2 style): fit on all OOF, evaluate on all OOF. ──
    method, predict_fn = _fit_calibrator(raw_preds, targets, y_min, y_max_data)
    cal_preds_in = predict_fn(raw_preds)

    raw_m = _metrics(raw_preds, targets, is_prob)
    cal_in_m = _metrics(cal_preds_in, targets, is_prob)

    # ── Leave-one-fold-out: for each fold, fit on others, evaluate on it. ──
    cal_preds_lofo = np.full_like(raw_preds, np.nan, dtype=float)
    unique_folds = np.unique(fold_ids)
    if len(unique_folds) >= 2:
        for k in unique_folds:
            mask_train = fold_ids != k
            mask_eval  = fold_ids == k
            if mask_train.sum() < PLATT_FALLBACK_THRESHOLD:
                # Tiny training pool — degenerate; copy raw through.
                cal_preds_lofo[mask_eval] = raw_preds[mask_eval]
                continue
            _, pred_fn_k = _fit_calibrator(
                raw_preds[mask_train], targets[mask_train], y_min, y_max_data,
            )
            cal_preds_lofo[mask_eval] = pred_fn_k(raw_preds[mask_eval])
    else:
        cal_preds_lofo = cal_preds_in.copy()

    cal_lofo_m = _metrics(cal_preds_lofo, targets, is_prob)

    # Diagnostic: how flat is the calibrator? (step-function pathology)
    n_unique_cal = int(len(np.unique(np.round(cal_preds_in, 6))))

    row = {
        "position_group":     pos_group,
        "target":              target_key,
        "method":              method,
        "n":                   raw_m["n"],
        "n_positive":          raw_m["n_positive"],
        "n_folds":             int(len(unique_folds)),
        "n_unique_cal":        n_unique_cal,
        # Raw OOF
        "raw_bias":            raw_m["bias"],
        "raw_mae":             raw_m["mae"],
        "raw_brier":           raw_m["brier"],
        "raw_logloss":         raw_m["logloss"],
        # In-sample isotonic (best case)
        "cal_in_bias":         cal_in_m["bias"],
        "cal_in_mae":          cal_in_m["mae"],
        "cal_in_brier":        cal_in_m["brier"],
        "cal_in_logloss":      cal_in_m["logloss"],
        # Leave-one-fold-out (honest)
        "cal_lofo_bias":       cal_lofo_m["bias"],
        "cal_lofo_mae":        cal_lofo_m["mae"],
        "cal_lofo_brier":      cal_lofo_m["brier"],
        "cal_lofo_logloss":    cal_lofo_m["logloss"],
        # Deltas (LoFO − raw; negative = improvement on MAE/brier/logloss)
        "d_lofo_bias":         cal_lofo_m["bias"]   - raw_m["bias"],
        "d_lofo_mae":          cal_lofo_m["mae"]    - raw_m["mae"],
        "d_lofo_brier":        cal_lofo_m["brier"]  - raw_m["brier"],
        "d_lofo_logloss":     (cal_lofo_m["logloss"] - raw_m["logloss"])
                              if is_prob else None,
    }

    logmsg = (
        "  %s/%s [%s]: n=%d n_pos=%d folds=%d  "
        "bias %+.4f→%+.4f  brier %.4f→%.4f  "
        "mae %.4f→%.4f%s"
    )
    if is_prob:
        logmsg += "  logloss %.4f→%.4f"
        logger.info(logmsg, pos_group, model_name, method,
                    row["n"], row["n_positive"], row["n_folds"],
                    row["raw_bias"], row["cal_lofo_bias"],
                    row["raw_brier"], row["cal_lofo_brier"],
                    row["raw_mae"], row["cal_lofo_mae"],
                    "",
                    row["raw_logloss"], row["cal_lofo_logloss"])
    else:
        logger.info(logmsg, pos_group, model_name, method,
                    row["n"], row["n_positive"], row["n_folds"],
                    row["raw_bias"], row["cal_lofo_bias"],
                    row["raw_brier"], row["cal_lofo_brier"],
                    row["raw_mae"], row["cal_lofo_mae"],
                    "")
    return row


def main():
    args = parse_args()
    target_key = args.target
    positions = (args.positions.split(",") if args.positions
                  else list(POSITION_GROUPS.keys()))

    logger.info("Loading snapshot...")
    snapshot = _assemble_snapshot()
    logger.info("Snapshot: %d rows, %d cols", len(snapshot), snapshot.shape[1])
    logger.info("Auditing target=%s across positions=%s", target_key, positions)

    rows: list[dict] = []
    for pos in positions:
        if pos not in POSITION_GROUPS:
            logger.warning("Unknown position %s — skipping", pos)
            continue
        try:
            row = _audit_one(snapshot, pos, target_key)
        except Exception as exc:
            logger.exception("  %s: failed (%s)", pos, exc)
            continue
        if row is not None:
            rows.append(row)

    if not rows:
        logger.error("No audit rows produced.")
        sys.exit(1)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info("Wrote audit to %s", out_path)

    # Headline summary: average LoFO deltas across positions where method=isotonic.
    iso = out_df[out_df["method"] == "isotonic"]
    if len(iso):
        logger.info(
            "Isotonic LoFO deltas — n_pos=%d  "
            "Δbias mean=%+.4f  Δmae mean=%+.4f  Δbrier mean=%+.5f%s",
            len(iso),
            float(iso["d_lofo_bias"].mean()),
            float(iso["d_lofo_mae"].mean()),
            float(iso["d_lofo_brier"].mean()),
            (f"  Δlogloss mean={iso['d_lofo_logloss'].mean():+.5f}"
             if target_key == "role" else ""),
        )
    platt = out_df[out_df["method"] == "platt_ridge"]
    if len(platt):
        logger.info("Platt-fallback positions: %s",
                    ", ".join(platt["position_group"].tolist()))


if __name__ == "__main__":
    main()
