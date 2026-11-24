"""
Step 8c — Post-Hoc Calibration of Draft Scores (role, impact_wav)

After step 8 / 8.5 train per-position LightGBM heads, this step stitches
OOF predictions from every expanding-window CV fold and fits a monotone
calibrator that re-aligns the predicted scale to the empirical target.
The pattern is lifted from cdarko/pipeline/09c2_calibrate_draft_scores.py.

Scope (decided by the audit at experiments/00l_isotonic_role_audit.py):

  - `role`        — calibrated. Mean LoFO MAE −34%, logloss −11% across
                    8/10 positions. The role head reports probability of
                    a second contract; isotonic gives proper probability
                    calibration.
  - `impact_wav`  — calibrated. Mean LoFO MAE −15%, bias halved across
                    6/10 positions.
  - `impact`      — NOT calibrated. Audit showed the head's raw bias is
                    already ~0; isotonic collapses to near-identity. The
                    `impact_apy_score` is also the source for the user-
                    facing dollar columns; leaving it untouched keeps the
                    dollar interpretation aligned with the underlying
                    cap-pct target.

Method selection per (pos × target), based on n_positive in the OOF set:

  - n_positive ≥ ISO_THRESHOLD (100) → IsotonicRegression
  - PLATT_THRESHOLD ≤ n_positive < ISO_THRESHOLD (50–99) → positive-coef Ridge
  - n_positive < PLATT_THRESHOLD (50)                    → no calibrator written
                                                          (step 10 falls through
                                                          to raw model output)

The 50 → 100 isotonic cutoff is a bump over cdarko's default after the
audit showed K (n_positive=67) regressed under isotonic while P (n_pos=48)
won cleanly under Platt — the football positions cluster around the
small-n boundary where step-function isotonic overfits.

Inputs:
    data/processed/player_skill_snapshot.parquet
    data/processed/nfl_draft_labels.parquet
    data/processed/draft_models/{pos}_{role|impact_wav}_model.pkl
    data/processed/draft_models/tuned_params*.json   (via step 8 helpers)

Outputs:
    data/processed/draft_models/{pos}_{role|impact_wav}_calibrator.pkl
        — bundle: {"method": "isotonic" | "platt_ridge", "calibrator": obj}
    data/processed/draft_models/calibration_summary.json
        — per-(pos × target) bias/MAE/Brier/logloss before vs. after.

Usage:
    python pipeline/08c_calibrate_draft_scores.py
    python pipeline/08c_calibrate_draft_scores.py --targets role
    python pipeline/08c_calibrate_draft_scores.py --skip-platt-fallback
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for sibling helpers

from config import (
    LGBM_PARAMS, POSITION_GROUPS, PROCESSED_DIR,
)
# Shared snapshot assembly (single source of truth across steps 8/8b/8c/9/10).
from _snapshot_assembly import assemble_snapshot
from src.utils.cache_manifest import (
    hash_features, write_manifest, tag_summary,
)

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reuse step 8 helpers (folds, feature build, prep_training_df, tuned params,
# monotone kwargs, derived features).
_spec = importlib.util.spec_from_file_location(
    "_p8", Path(__file__).resolve().parent / "08_draft_models.py",
)
_p8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p8)

STEP_NAME = "08c_calibrate_draft_scores"

DRAFT_MODELS_DIR  = PROCESSED_DIR / "draft_models"
SUMMARY_PATH      = DRAFT_MODELS_DIR / "calibration_summary.json"
GLOBAL_MODELS_DIR   = PROCESSED_DIR / "global_draft_models"
GLOBAL_OOF_PATH     = GLOBAL_MODELS_DIR / "global_oof_predictions.parquet"
GLOBAL_SUMMARY_PATH = GLOBAL_MODELS_DIR / "global_calibration_summary.json"

# Per-position calibration of the GLOBAL (8b) heads, fit on 8b's persisted
# tuning-fold OOF predictions. Prerequisite for step 10's
# GLOBAL_SCORE_OVERRIDES: the pooled model's ordering wins the blend A/B,
# but its LEVELS are pooled-calibrated — a per-position monotone map makes
# the swapped/blended scores mean the right thing within each position.
# (model_name, is_prob, y_min, y_max_template, mixture_pad)
# mixture_pad: pad non-bridge NaN targets to 0 so the calibrator maps the
# bridge-only-trained global impact_wav head onto the per-position MIXTURE
# scale — which is what makes partial blends with the per-position mixture
# head scale-compatible.
GLOBAL_CALIBRATION_TARGETS = [
    ("impact_apy", False, 0.0, None, False),
    ("impact_wav", False, 0.0, None, True),
]

# Calibrator selection thresholds (see module docstring for rationale).
ISO_THRESHOLD   = 100   # n_positive ≥ this → isotonic
PLATT_THRESHOLD = 50    # n_positive ≥ this → Platt-Ridge; below → identity

# Isotonic stability guard (added 2026-07-26).
#
# n_positive alone decides isotonic-vs-Platt, with no check on whether the
# resulting map is actually well-behaved. On a long-tailed target a modest
# cohort can produce a step function whose FINAL segment is a cliff, and then
# a hair's-width difference in raw prediction becomes a multiple in output.
#
# EDGE/impact_apy did exactly that: 240 positive OOF rows (comfortably over
# ISO_THRESHOLD) but a map that jumped 0.06922 → 0.16100 across a 0.4% change
# in input. The entire current class landed 0.00015 below that edge and was
# scored at 0.036 — below LB and DB, for a premium position — while historical
# EDGE cleared it and clipped to the 0.161 ceiling.
#
# The automatic Platt-Ridge FALLBACK was tried on 2026-07-27 and REVERTED —
# it was worse than the bug. Platt-Ridge fits a line through a target that is
# ~93% zeros, so it flattens the whole range: QB's calibrated max fell from a
# raw 0.140 to 0.028 and LB's from 0.097 to 0.010, dropping QB from 1st to 8th
# on the board and LB from 6th to 10th.
#
# The trigger is also unstable. Refitting on a new retune's OOF moved the cliff
# off EDGE and onto QB and LB, and EVERY position's top isotonic level is
# supported by exactly ONE observation — so this is a universal property of
# isotonic on a long-tailed, zero-inflated target, not an EDGE anomaly, and
# which head happens to cliff is arbitrary across retunes. A quantile-capped
# y_max was also measured and does not reliably help (QB got worse: 1.88 ->
# 1.95).
#
# What remains is DIAGNOSTIC only: the ratio is computed and logged so the
# instability stays visible, but no method is swapped. The real problem — that
# the cross-position CURRENT-CLASS board is unreliable because predictions are
# compressed ~58% against historical and isotonic's single-observation top
# plateau amplifies that unpredictably — needs a considered fix (most likely in
# presentation, via within-position ranks, since isotonic is monotone and
# within-position ordering is unaffected), not a calibrator hack.
MAX_TOP_STEP_RATIO = 1.75

# Targets to calibrate. Tuple is (target_col, model_name, is_probability,
# y_min, y_max_template). y_max_template=None means clamp to observed
# max at fit time (suppresses extrapolation beyond training range).
CALIBRATION_TARGETS = [
    ("impact_w_av_per_game", "impact_wav", False, 0.0, None),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 8c — draft score calibrator")
    p.add_argument(
        "--targets", default=None,
        help="Comma-separated subset of {impact_wav} (default: all). The role head was dropped 2026-10-31.",
    )
    p.add_argument(
        "--skip-global-heads", action="store_true",
        help="Skip per-position calibration of the global (8b) heads.",
    )
    p.add_argument(
        "--skip-platt-fallback", action="store_true",
        help="Skip the Platt fallback band entirely — write isotonic only "
             "when n_positive ≥ ISO_THRESHOLD; smaller heads get no calibrator.",
    )
    p.add_argument(
        "--test", action="store_true",
        help="No-op flag for run_pipeline.py compatibility (step 8c is "
             "already cheap enough to always run on full data).",
    )
    return p.parse_args()


# ── OOF prediction generation ────────────────────────────────────────────────
# Same recipe as the audit: stitch expanding-window-CV val predictions
# across every fold using each fold's tuned params + monotone constraints
# (early stopping per fold so n_estimators matches step 8's _fit_eval_fold).

_STEP8_OOF_CACHE: dict = {}


def _load_step8_oof(pos_group: str, model_name: str
                    ) -> tuple[np.ndarray, np.ndarray] | None:
    """Read step 8's persisted tuning-fold OOF predictions for one head.
    Returns None when the parquet is absent or has no rows for the head —
    callers then fall back to refitting the CV locally (pre-2026-07 path).
    Using step 8's own OOF guarantees the calibrator is fit on predictions
    from the exact deployed training procedure, and makes 8c nearly free."""
    path = DRAFT_MODELS_DIR / "oof_predictions.parquet"
    if not path.exists():
        return None
    if "df" not in _STEP8_OOF_CACHE:
        try:
            _STEP8_OOF_CACHE["df"] = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Could not read %s (%s) — falling back to refit",
                           path.name, exc)
            _STEP8_OOF_CACHE["df"] = pd.DataFrame()
    oof = _STEP8_OOF_CACHE["df"]
    if oof.empty:
        return None
    sub = oof[(oof["pos_group"] == pos_group) & (oof["model_name"] == model_name)]
    if sub.empty:
        return None
    return sub["pred"].to_numpy(float), sub["y"].to_numpy(float)


def _generate_oof_preds(df: pd.DataFrame, feature_cols: list[str],
                        target_col: str, params: dict,
                        pos_group: str, model_name: str,
                        ) -> tuple[np.ndarray, np.ndarray]:
    cached = _load_step8_oof(pos_group, model_name)
    if cached is not None:
        logger.info("  %s/%s: using step-8 persisted OOF (%d rows)",
                    pos_group, model_name, len(cached[0]))
        return cached
    folds = _p8._expanding_window_folds()
    yr_series = df["last_college_season_year"].fillna(0)
    base = {**LGBM_PARAMS, **params}
    monotone = _p8._build_monotone_kwargs(feature_cols, pos_group, model_name)

    preds_list, targets_list = [], []
    for train_end, val_start, val_end in folds:
        f_train = df[yr_series <= train_end]
        f_val   = df[(yr_series >= val_start) & (yr_series <= val_end)]
        if len(f_train) < 30 or len(f_val) < 10:
            continue
        X_tr = f_train[feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y_tr = f_train[target_col].fillna(0.0).values
        X_va = f_val[feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y_va = f_val[target_col].fillna(0.0).values

        _make = lambda: lgb.LGBMRegressor(**{**base, **monotone})
        m, _ = _p8._fit_fold_honest(
            _make, X_tr, y_tr,
            f_train["last_college_season_year"].fillna(0).values)
        preds_list.append(m.predict(X_va))
        targets_list.append(y_va)

    if not preds_list:
        return np.array([]), np.array([])
    return np.concatenate(preds_list), np.concatenate(targets_list)


# ── Calibrator fit + diagnostic ──────────────────────────────────────────────

def _select_method(n_positive: int, skip_platt: bool) -> str:
    if n_positive >= ISO_THRESHOLD:
        return "isotonic"
    if skip_platt:
        return "identity"
    if n_positive >= PLATT_THRESHOLD:
        return "platt_ridge"
    return "identity"


def _isotonic_top_step_ratio(cal) -> float:
    """Ratio of the fitted map's largest output level to the one below it.

    1.0 means the map plateaus smoothly at the top (healthy — DB and DL both
    do this). A large value means the top of the board is decided by a cliff,
    so two players a rounding error apart in raw prediction get very different
    scores.
    """
    ys = np.unique(np.asarray(cal.y_thresholds_, dtype=float))
    ys = ys[ys > 0]
    if len(ys) < 2:
        return 1.0
    top, below = ys[-1], ys[-2]
    return float(top / below) if below > 0 else float("inf")


def _fit_calibrator(preds: np.ndarray, targets: np.ndarray,
                    method: str, y_min: float, y_max: float,
                    label: str = ""):
    """Returns (fitted_object, effective_method). y_max must be concrete
    (caller resolves None → observed max). For method='identity' returns
    (None, 'identity').

    An isotonic fit whose top step is a cliff (see MAX_TOP_STEP_RATIO) is
    rejected in favour of Platt-Ridge — the instability is at exactly the end
    of the range that sets the top of the board. The EFFECTIVE method is
    returned rather than the requested one because `_apply` dispatches on it
    (Ridge needs a 2-D input) and it is recorded in the pickle and summary; a
    silent swap would mislabel both and crash inference.
    """
    if method == "isotonic":
        cal = IsotonicRegression(y_min=y_min, y_max=y_max,
                                 out_of_bounds="clip", increasing=True)
        cal.fit(preds, targets)
        ratio = _isotonic_top_step_ratio(cal)
        if ratio > MAX_TOP_STEP_RATIO:
            # Diagnostic only — see the note on MAX_TOP_STEP_RATIO for why
            # swapping to Platt-Ridge here was tried and reverted.
            logger.warning(
                "  %s: isotonic top step is %.2fx (> %.2f) — top of this "
                "head's range is decided by a cliff; scores near it are "
                "unstable across retunes",
                label or "calibrator", ratio, MAX_TOP_STEP_RATIO)
        return cal, "isotonic"
    if method == "platt_ridge":
        cal = Ridge(alpha=1.0, positive=True)
        cal.fit(preds.reshape(-1, 1), targets)
        return cal, "platt_ridge"
    return None, "identity"


def _apply(method: str, calibrator, raw: np.ndarray,
           y_min: float, y_max: float) -> np.ndarray:
    if method == "isotonic":
        return np.clip(calibrator.predict(raw), y_min, y_max)
    if method == "platt_ridge":
        return np.clip(calibrator.predict(raw.reshape(-1, 1)), y_min, y_max)
    return np.clip(raw, y_min, y_max)


def _metrics(preds: np.ndarray, targets: np.ndarray,
             is_prob: bool) -> dict:
    out = {
        "bias":  float(np.mean(preds - targets)),
        "mae":   float(np.mean(np.abs(preds - targets))),
        "brier": float(np.mean((preds - targets) ** 2)),
    }
    if is_prob:
        eps = 1e-7
        p = np.clip(preds, eps, 1.0 - eps)
        y = np.clip(targets, 0.0, 1.0)
        out["logloss"] = float(
            -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        )
    else:
        out["logloss"] = None
    return out


def _calibrate_one(snapshot: pd.DataFrame, labels: pd.DataFrame,
                   pos_group: str, target_col: str, model_name: str,
                   is_prob: bool, y_min: float, y_max_template: float | None,
                   skip_platt: bool) -> dict | None:
    feature_cols = _p8._build_feature_cols(snapshot, pos_group)
    if not feature_cols:
        return None
    # Derived features already live on the shared assembled snapshot
    # (assemble_snapshot include_derived=True), matching step 8's training
    # path exactly — no per-head recomputation needed.
    df = _p8._prep_training_df(snapshot, labels, pos_group, target_col)
    if len(df) < 60:
        return None

    tuned = _p8._load_tuned_params(pos_group, model_name) or {}
    raw_preds, targets = _generate_oof_preds(
        df, feature_cols, target_col, tuned, pos_group, model_name,
    )
    if len(raw_preds) == 0:
        return None

    n_positive = int((targets > 0).sum())
    method = _select_method(n_positive, skip_platt)
    y_max = float(targets.max()) if y_max_template is None else y_max_template
    raw_clipped = np.clip(raw_preds, y_min, y_max)

    raw_m = _metrics(raw_clipped, targets, is_prob)

    if method == "identity":
        logger.info(
            "  %s/%s [identity, n=%d n_pos=%d]: no calibrator written "
            "(n_positive < %d)",
            pos_group, model_name, len(raw_preds), n_positive, PLATT_THRESHOLD,
        )
        return {
            "method":     "identity",
            "n":          int(len(raw_preds)),
            "n_positive": n_positive,
            "y_min":      y_min, "y_max": y_max,
            "feature_cols_hash": hash_features(feature_cols),
            "raw":        raw_m,
            "cal":        raw_m,   # identity ⇒ cal = raw
        }

    cal, method = _fit_calibrator(raw_preds, targets, method, y_min, y_max,
                                   label=f"{pos_group}/{model_name}")
    cal_preds = _apply(method, cal, raw_preds, y_min, y_max)
    cal_m = _metrics(cal_preds, targets, is_prob)

    out_path = DRAFT_MODELS_DIR / f"{pos_group}_{model_name}_calibrator.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "method":     method,
            "calibrator": cal,
            "y_min":      y_min,
            "y_max":      y_max,
        }, f)

    if is_prob:
        logger.info(
            "  %s/%s [%s, n=%d n_pos=%d]: "
            "bias %+.4f→%+.4f  mae %.4f→%.4f  brier %.4f→%.4f  "
            "logloss %.4f→%.4f  → %s",
            pos_group, model_name, method, len(raw_preds), n_positive,
            raw_m["bias"], cal_m["bias"], raw_m["mae"], cal_m["mae"],
            raw_m["brier"], cal_m["brier"],
            raw_m["logloss"], cal_m["logloss"], out_path.name,
        )
    else:
        logger.info(
            "  %s/%s [%s, n=%d n_pos=%d]: "
            "bias %+.5f→%+.5f  mae %.5f→%.5f  brier %.6f→%.6f  → %s",
            pos_group, model_name, method, len(raw_preds), n_positive,
            raw_m["bias"], cal_m["bias"], raw_m["mae"], cal_m["mae"],
            raw_m["brier"], cal_m["brier"], out_path.name,
        )

    return {
        "method":     method,
        "n":          int(len(raw_preds)),
        "n_positive": n_positive,
        "y_min":      y_min, "y_max": y_max,
        "feature_cols_hash": hash_features(feature_cols),
        "raw":        raw_m,
        "cal":        cal_m,
    }


def _calibrate_global_heads(skip_platt: bool) -> dict:
    """Fit per-position calibrators for the global (8b) heads from the
    persisted tuning-fold OOF predictions. Writes
    {pos}_{model}_global_calibrator.pkl bundles under global_draft_models/
    (same bundle shape step 10's _apply_calibrator consumes)."""
    if not GLOBAL_OOF_PATH.exists():
        logger.info("No global OOF predictions (%s) — run 08b first; "
                    "skipping global-head calibrators", GLOBAL_OOF_PATH.name)
        return {}
    oof = pd.read_parquet(GLOBAL_OOF_PATH)
    summary: dict[str, dict] = {}
    for model_name, is_prob, y_min, y_max_t, mixture_pad in GLOBAL_CALIBRATION_TARGETS:
        sub_all = oof[oof["model_name"] == model_name]
        if sub_all.empty:
            continue
        for pos_group in POSITION_GROUPS.keys():
            sub = sub_all[sub_all["position_group"] == pos_group].copy()
            if mixture_pad:
                nb = ~sub["is_bridge"].astype(bool)
                sub.loc[nb, "y"] = sub.loc[nb, "y"].fillna(0.0)
            sub = sub[sub["y"].notna()]
            if len(sub) < 60:
                continue
            preds = sub["pred"].to_numpy(float)
            targets = sub["y"].to_numpy(float)
            n_positive = int((targets > 0).sum())
            method = _select_method(n_positive, skip_platt)
            y_max = float(targets.max()) if y_max_t is None else y_max_t
            raw_m = _metrics(np.clip(preds, y_min, y_max), targets, is_prob)
            out_path = GLOBAL_MODELS_DIR / f"{pos_group}_{model_name}_global_calibrator.pkl"
            if method == "identity":
                out_path.unlink(missing_ok=True)  # stale calibrator, if any
                summary[f"{pos_group}/{model_name}"] = {
                    "method": "identity", "n": int(len(sub)),
                    "n_positive": n_positive, "y_min": y_min, "y_max": y_max,
                    "raw": raw_m, "cal": raw_m,
                }
                logger.info("  global %s/%s [identity, n=%d n_pos=%d]: "
                            "no calibrator written",
                            pos_group, model_name, len(sub), n_positive)
                continue
            cal, method = _fit_calibrator(
                preds, targets, method, y_min, y_max,
                label=f"global {pos_group}/{model_name}")
            cal_preds = _apply(method, cal, preds, y_min, y_max)
            cal_m = _metrics(cal_preds, targets, is_prob)
            with open(out_path, "wb") as f:
                pickle.dump({"method": method, "calibrator": cal,
                             "y_min": y_min, "y_max": y_max}, f)
            summary[f"{pos_group}/{model_name}"] = {
                "method": method, "n": int(len(sub)),
                "n_positive": n_positive, "y_min": y_min, "y_max": y_max,
                "raw": raw_m, "cal": cal_m,
            }
            logger.info("  global %s/%s [%s, n=%d n_pos=%d]: "
                        "bias %+.5f→%+.5f  mae %.5f→%.5f  → %s",
                        pos_group, model_name, method, len(sub), n_positive,
                        raw_m["bias"], cal_m["bias"], raw_m["mae"],
                        cal_m["mae"], out_path.name)
    if summary:
        stamped = tag_summary({"by_head": summary},
                              mode_used="fresh", step_name=STEP_NAME)
        GLOBAL_SUMMARY_PATH.write_text(json.dumps(stamped, indent=2))
        logger.info("Global calibration summary → %s", GLOBAL_SUMMARY_PATH)
    _fit_quantile_conformal(oof)
    return summary


def _fit_quantile_conformal(oof: pd.DataFrame) -> None:
    """Conformalized quantile regression (CQR) offsets for the global
    impact_apy p90 head, per position: offset = 90th percentile of the OOF
    residuals (y − p90_pred). Adding the offset at inference makes 'about
    10% of players land above p90' true on held-out data — fixing both the
    audit's conservatism (1.1% exceedance vs nominal 10%) and the band
    collapse against the isotonic-calibrated point estimate."""
    sub = oof[(oof["model_name"] == "impact_apy") & oof["p90_pred"].notna()] \
        if "p90_pred" in oof.columns else pd.DataFrame()
    if sub.empty:
        logger.info("No p90 OOF predictions — re-run 08b to enable CQR offsets")
        return
    offsets: dict[str, float] = {}
    for pos_group in POSITION_GROUPS.keys():
        d = sub[sub["position_group"] == pos_group]
        d = d[d["y"].notna()]
        if len(d) < 100:
            continue
        resid = d["y"].to_numpy(float) - d["p90_pred"].to_numpy(float)
        offset = float(np.quantile(resid, 0.9))
        raw_exceed = float((resid > 0).mean())
        adj_exceed = float((resid > offset).mean())
        offsets[pos_group] = offset
        logger.info("  CQR %s: offset %+.5f  exceedance %.1f%% → %.1f%% "
                    "(nominal 10%%, n=%d)",
                    pos_group, offset, raw_exceed * 100, adj_exceed * 100,
                    len(d))
    if offsets:
        path = GLOBAL_MODELS_DIR / "quantile_conformal.json"
        path.write_text(json.dumps(tag_summary(
            {"impact_apy_p90_offsets": offsets, "alpha": 0.9},
            mode_used="fresh", step_name=STEP_NAME), indent=2))
        logger.info("CQR offsets (%d positions) → %s", len(offsets), path)


# ── Stale-calibrator cleanup ─────────────────────────────────────────────────
# When a head drops below the identity threshold (e.g. a position runs out
# of bridge labels), the previous run may have written a calibrator we no
# longer want step 10 to apply. Remove orphans up front so a re-run can't
# leave dead pickles behind.

def _cleanup_orphans(targets_to_run: set[str]) -> None:
    """Unlink calibrator pickles no head in the current run will visit:
    either the model suffix is not among `targets_to_run` (model dropped
    from CALIBRATION_TARGETS) or the pos prefix is no longer in
    POSITION_GROUPS. Heads that WILL run are left alone — the main loop
    overwrites them (isotonic/platt) or removes them on identity via
    `_maybe_remove_calibrator`. Only invoked on full runs; a `--targets`
    subset run skips cleanup so it can't delete the untargeted heads'
    live calibrators (see main())."""
    if not DRAFT_MODELS_DIR.exists():
        return
    valid_suffixes = {f"_{m}_calibrator.pkl"
                      for (_, m, _, _, _) in CALIBRATION_TARGETS
                      if m in targets_to_run}
    for path in DRAFT_MODELS_DIR.glob("*_calibrator.pkl"):
        suffix = next((s for s in valid_suffixes if path.name.endswith(s)), None)
        pos_group = path.name[:-len(suffix)] if suffix else None
        if suffix is not None and pos_group in POSITION_GROUPS:
            continue   # live head — will be re-written or identity-removed
        path.unlink()
        logger.info("  removed orphan calibrator %s (no head in this run "
                    "writes it)", path.name)


def _maybe_remove_calibrator(pos_group: str, model_name: str, method: str) -> None:
    """Delete an existing calibrator pickle when this head selected identity.
    Prevents a stale calibrator from a prior run from being applied at scoring.
    """
    if method != "identity":
        return
    path = DRAFT_MODELS_DIR / f"{pos_group}_{model_name}_calibrator.pkl"
    if path.exists():
        path.unlink()
        logger.info("  removed stale %s (identity method now)", path.name)


def main():
    args = parse_args()
    DRAFT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.targets:
        wanted = {t.strip() for t in args.targets.split(",") if t.strip()}
        targets = [t for t in CALIBRATION_TARGETS if t[1] in wanted]
        if not targets:
            logger.error("--targets %s did not match any of %s",
                         args.targets, [t[1] for t in CALIBRATION_TARGETS])
            sys.exit(1)
    else:
        targets = CALIBRATION_TARGETS

    logger.info("Step 8c — calibrating: %s (skip_platt=%s)",
                [t[1] for t in targets], args.skip_platt_fallback)

    logger.info("Loading snapshot...")
    labels_path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    if not labels_path.exists():
        logger.error("%s missing — run earlier steps", labels_path)
        sys.exit(1)
    # Shared assembly — 8c sees the exact training matrix step 8 trained on
    # (incl. QB-career features, OL sub-position fill, and derived features
    # its old private copy had drifted away from).
    snapshot = assemble_snapshot(include_derived=True, logger=logger)
    labels = pd.read_parquet(labels_path)
    logger.info("Snapshot: %d rows, %d cols", len(snapshot), snapshot.shape[1])

    if args.targets:
        logger.info("--targets subset run — skipping orphan-calibrator cleanup")
    else:
        _cleanup_orphans({t[1] for t in targets})

    summary: dict[str, dict] = {}
    for target_col, model_name, is_prob, y_min, y_max_t in targets:
        logger.info("=== %s ===", model_name)
        for pos_group in POSITION_GROUPS.keys():
            try:
                row = _calibrate_one(
                    snapshot, labels, pos_group, target_col, model_name,
                    is_prob, y_min, y_max_t,
                    skip_platt=args.skip_platt_fallback,
                )
            except Exception as exc:
                logger.exception("  %s/%s: failed (%s)", pos_group, model_name, exc)
                continue
            if row is None:
                continue
            _maybe_remove_calibrator(pos_group, model_name, row["method"])
            summary[f"{pos_group}/{model_name}"] = row

    if not summary:
        logger.error("No calibrators fit.")
        sys.exit(1)

    # Headline rollup per target.
    for _, model_name, is_prob, _, _ in targets:
        rows = [r for k, r in summary.items() if k.endswith(f"/{model_name}")]
        if not rows:
            continue
        iso = [r for r in rows if r["method"] == "isotonic"]
        platt = [r for r in rows if r["method"] == "platt_ridge"]
        identity = [r for r in rows if r["method"] == "identity"]
        logger.info(
            "%s: iso=%d  platt=%d  identity=%d", model_name,
            len(iso), len(platt), len(identity),
        )
        if iso or platt:
            written = iso + platt
            d_bias = float(np.mean([r["cal"]["bias"] - r["raw"]["bias"] for r in written]))
            d_mae  = float(np.mean([r["cal"]["mae"]  - r["raw"]["mae"]  for r in written]))
            logger.info("  written-Δ mean: bias %+.5f  mae %+.5f", d_bias, d_mae)

    stamped = tag_summary(
        {"thresholds": {"isotonic_min_n_positive": ISO_THRESHOLD,
                         "platt_min_n_positive":   PLATT_THRESHOLD},
         "skip_platt_fallback": bool(args.skip_platt_fallback),
         "by_head": summary},
        mode_used="fresh", step_name=STEP_NAME,
    )
    SUMMARY_PATH.write_text(json.dumps(stamped, indent=2))
    logger.info("Summary → %s", SUMMARY_PATH)

    # Fingerprint sidecar: only the per-head feature-cols hash + n_rows. The
    # source model pickles are the upstream invariant; we use Path.stat()
    # mtime via parquet_metadata_fingerprint for the labels/snapshot, and
    # a content hash of tuned_params*.json (which already has its own
    # manifest) is not duplicated here — staleness surfaces via step 8's
    # manifest when those drift.
    fingerprint = {
        "n_heads":           len(summary),
        "iso_threshold":     ISO_THRESHOLD,
        "platt_threshold":   PLATT_THRESHOLD,
        "snapshot_n_rows":   int(len(snapshot)),
        "by_key": {
            k: {
                "feature_cols_hash": v["feature_cols_hash"],
                "method":            v["method"],
                "n_positive":        v["n_positive"],
            }
            for k, v in summary.items()
        },
    }
    write_manifest(
        SUMMARY_PATH, fingerprint=fingerprint,
        mode_used="fresh", step_name=STEP_NAME,
        extra={"n_iso":      sum(1 for v in summary.values() if v["method"] == "isotonic"),
               "n_platt":    sum(1 for v in summary.values() if v["method"] == "platt_ridge"),
               "n_identity": sum(1 for v in summary.values() if v["method"] == "identity")},
    )

    if not args.skip_global_heads:
        logger.info("=== global heads (per-position calibration of 8b) ===")
        _calibrate_global_heads(skip_platt=args.skip_platt_fallback)

    logger.info("Step 8c complete.")


if __name__ == "__main__":
    main()
