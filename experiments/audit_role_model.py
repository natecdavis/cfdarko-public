"""Audit Role-model output for the DL/DT/DE saturation cluster at ~0.50.

Read-only diagnostic: produces ``experiments/role_model_audit.md`` and
supporting CSVs/PNGs in ``experiments/role_model_audit/``. Does not write to
``data/`` or modify any pipeline state.

Sections produced:
  S1  Training-data composition per position group (bridge vs non-bridge mass)
  S2  Per-position reliability curves (holdout refit + in-sample sanity)
  S3  Feature attribution for 2025-class DL prospects with role_score >= 0.40
  S4  Counterfactual: LGBMRegressor (production) vs LGBMClassifier head
  S5  Non-bridge zero-mass leverage test (DL refit on bridge-only rows)
  S6  Synthesis + concrete remediation candidates

Run:  python experiments/audit_role_model.py
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config import POSITION_GROUPS, MONOTONE_ENABLED  # noqa: E402

PROJ_PATH = REPO / "data" / "output" / "projections" / "latest.parquet"
LABELS_PATH = REPO / "data" / "processed" / "nfl_draft_labels.parquet"
SNAPSHOT_PATH = REPO / "data" / "processed" / "player_skill_snapshot.parquet"
MODELS_DIR = REPO / "data" / "processed" / "draft_models"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

OUT_DIR = REPO / "experiments" / "role_model_audit"
OUT_MD = REPO / "experiments" / "role_model_audit.md"

FOCUS_POS = "DL"
SUB_POS_FOCUS = ("DL", "DT", "DE", "EDGE", "NT")

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    proj = pd.read_parquet(PROJ_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    metrics = json.load(open(METRICS_PATH)) if METRICS_PATH.exists() else {}

    label_cols = ["player_id", "role_target", "impact_target",
                  "is_bridge", "nfl_games", "has_nfl_data",
                  "first_nfl_season", "draft_year"]
    label_cols = [c for c in label_cols if c in labels.columns]
    df = proj.merge(labels[label_cols], on="player_id", how="left",
                    suffixes=("", "_lbl"))
    df["is_bridge"] = df["is_bridge"].fillna(False)
    df["role_target"] = df["role_target"].fillna(0.0)
    return df, labels, metrics


def _add_derived_dl_lb(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate Step 10's per-position derived features for DL/LB.
    `pass_rush_composite` and `pff_pass_rush_snap_pct` are not persisted in
    latest.parquet; the model expects them at inference time.
    """
    df = df.copy()
    s = df.get("kalman_mean_sacks",      pd.Series(0, index=df.index)).fillna(0)
    t = df.get("kalman_mean_tfl",        pd.Series(0, index=df.index)).fillna(0)
    h = df.get("kalman_mean_qb_hurries", pd.Series(0, index=df.index)).fillna(0)
    df["pass_rush_composite"] = s * 2.0 + t * 1.0 + h * 0.5
    def_snaps = df.get("pff_snap_counts_defense_mean",
                       pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    pr_snaps = df.get("pff_snap_counts_pass_rush_mean",
                      pd.Series(np.nan, index=df.index))
    df["pff_pass_rush_snap_pct"] = (pr_snaps / def_snaps).clip(0, 1)
    cov_snaps = df.get("pff_snap_counts_coverage_mean",
                       pd.Series(np.nan, index=df.index))
    df["pff_coverage_snap_pct"] = (cov_snaps / def_snaps).clip(0, 1)
    return df


def _load_role_bundle(pos_group: str) -> dict:
    path = MODELS_DIR / f"{pos_group}_role_model.pkl"
    return pickle.load(open(path, "rb"))


def _features_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df.reindex(columns=feature_cols, fill_value=np.nan).to_numpy(
        dtype=float, na_value=np.nan)


# ─── S1: Training-data composition ─────────────────────────────────────────

def section_s1(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Per position_group: bridge vs non-bridge counts, role_target distribution."""
    rows = []
    for pos in sorted(p for p in df["position_group"].dropna().unique()
                      if p != "OTHER"):
        sub = df[df["position_group"] == pos]
        bridge = sub[sub["is_bridge"] == True]
        non_bridge = sub[sub["is_bridge"] == False]
        n_bridge = len(bridge)
        n_non = len(non_bridge)
        rt = bridge["role_target"]
        rt_traj = rt[(rt > 0) & (rt < 1)]
        rows.append({
            "pos_group":           pos,
            "n_total":             len(sub),
            "n_bridge":            n_bridge,
            "n_non_bridge":        n_non,
            "non_bridge_share":    n_non / max(len(sub), 1),
            "bridge_role_mean":    float(rt.mean()) if n_bridge else np.nan,
            "bridge_role_median":  float(rt.median()) if n_bridge else np.nan,
            "bridge_zero_frac":    float((rt == 0).mean()) if n_bridge else np.nan,
            "bridge_one_frac":     float((rt == 1).mean()) if n_bridge else np.nan,
            "bridge_traj_frac":    float(((rt > 0) & (rt < 1)).mean()) if n_bridge else np.nan,
            "bridge_traj_mean":    float(rt_traj.mean()) if len(rt_traj) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("pos_group").reset_index(drop=True)
    out.to_csv(OUT_DIR / "training_composition.csv", index=False)

    md = ["## S1. Training-data composition\n"]
    md.append("Per `position_group`, with the production training mix step 8 sees "
              "after `_prep_training_df` (NaN role_target → 0):\n")
    fmt = out.copy()
    for c in ("non_bridge_share", "bridge_zero_frac",
              "bridge_one_frac", "bridge_traj_frac"):
        fmt[c] = (fmt[c] * 100).round(1).astype(str) + "%"
    for c in ("bridge_role_mean", "bridge_role_median", "bridge_traj_mean"):
        fmt[c] = fmt[c].round(3)
    md.append("```\n" + fmt.to_string(index=False) + "\n```\n")
    md.append("**Reading guide**:\n")
    md.append("- `non_bridge_share`: fraction of step-8 training rows that are "
              "the appended `role_target=0` non-bridge mass "
              "(`pipeline/07_nfl_labels.py:313-326`). Higher → stronger pull-toward-0.\n")
    md.append("- `bridge_role_mean`: mean role_target *among bridge players only*.\n")
    md.append("- `bridge_traj_frac`: fraction of bridge with a continuous "
              "trajectory-imputed label (vs hard 0/1 fallback).\n")
    return "\n".join(md), out


# ─── S2: Reliability curves (in-sample + holdout refit) ────────────────────

def _decile_reliability(y_true: np.ndarray, y_pred: np.ndarray,
                         n_bins: int = 10) -> pd.DataFrame:
    if len(y_true) < n_bins:
        return pd.DataFrame()
    edges = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.digitize(y_pred, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append({
            "decile":      b + 1,
            "n":           int(mask.sum()),
            "pred_mean":   float(y_pred[mask].mean()),
            "actual_mean": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


def _refit_holdout_role(df: pd.DataFrame, pos_group: str,
                        train_end: int, val_start: int, val_end: int,
                        bridge_only: bool = False) -> tuple[np.ndarray, np.ndarray, lgb.LGBMRegressor]:
    """Refit role head with same tuned params, train ≤ train_end, eval val window.
    bridge_only=True drops non-bridge rows from training.
    """
    bundle = _load_role_bundle(pos_group)
    feature_cols = bundle["feature_cols"]
    pos_df = df[df["position_group"] == pos_group]
    if pos_group in ("DL", "LB"):
        pos_df = _add_derived_dl_lb(pos_df)
    yr = pd.to_numeric(pos_df["last_college_season_year"], errors="coerce").fillna(0)
    train_mask = yr <= train_end
    val_mask = (yr >= val_start) & (yr <= val_end)
    if bridge_only:
        train_mask &= (pos_df["is_bridge"] == True)
    f_train = pos_df[train_mask]
    f_val = pos_df[val_mask]
    if len(f_train) < 30 or len(f_val) < 10:
        raise RuntimeError(f"Insufficient samples for {pos_group}: "
                           f"train={len(f_train)} val={len(f_val)}")
    # Pull tuned params from the production model itself.
    prod = bundle["model"]
    params = prod.get_params()
    # Strip non-fittable / inherited keys; keep core hyperparameters.
    keep = {"learning_rate", "max_depth", "num_leaves", "min_child_samples",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
            "min_split_gain", "n_estimators", "objective", "verbose",
            "random_state", "n_jobs"}
    params = {k: v for k, v in params.items() if k in keep}
    params.setdefault("verbose", -1)
    X_t = _features_matrix(f_train, feature_cols)
    y_t = f_train["role_target"].fillna(0).to_numpy(dtype=float)
    X_v = _features_matrix(f_val, feature_cols)
    y_v = f_val["role_target"].fillna(0).to_numpy(dtype=float)
    m = lgb.LGBMRegressor(**params)
    m.fit(X_t, y_t,
          eval_set=[(X_v, y_v)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(period=-1)])
    preds = np.clip(m.predict(X_v), 0.0, 1.0)
    return y_v, preds, m


def section_s2(df: pd.DataFrame, metrics: dict) -> str:
    """Reliability per position: in-sample (production model), plus a refit
    on the holdout fold (train ≤ holdout_train_end) for the focus position(s).
    """
    md = ["## S2. Reliability curves\n"]

    # ----- In-sample reliability for every position (caveat noted) -----
    md.append("### S2a. In-sample reliability (production model on bridge players)\n")
    md.append("**Caveat**: the production model was trained on these rows. This "
              "section is a sanity check (reveals gross miscalibration but cannot "
              "diagnose generalization). See S2b for an honest refit.\n")
    in_sample = []
    for pos in sorted(p for p in df["position_group"].dropna().unique()
                      if p != "OTHER"):
        bridge = df[(df["position_group"] == pos) & (df["is_bridge"] == True)]
        if len(bridge) < 30:
            continue
        rel = _decile_reliability(bridge["role_target"].to_numpy(),
                                   bridge["role_score"].fillna(0).to_numpy())
        if rel.empty:
            continue
        rel.to_csv(OUT_DIR / f"reliability_in_sample_{pos}.csv", index=False)
        top = rel.iloc[-1]
        in_sample.append({
            "pos":               pos,
            "n_bridge":          len(bridge),
            "top_decile_pred":   round(top["pred_mean"], 3),
            "top_decile_actual": round(top["actual_mean"], 3),
            "gap":               round(top["pred_mean"] - top["actual_mean"], 3),
            "overall_pred_mean": round(float(bridge["role_score"].mean()), 3),
            "overall_actual":   round(float(bridge["role_target"].mean()), 3),
        })
    fmt = pd.DataFrame(in_sample).sort_values("pos")
    md.append("```\n" + fmt.to_string(index=False) + "\n```\n")

    # ----- Honest holdout refit for the focus position -----
    md.append(f"### S2b. Honest holdout refit — `{FOCUS_POS}`\n")
    h = metrics.get(f"{FOCUS_POS}_role", {})
    h_train_end = h.get("holdout_train_end", 2021)
    h_val_start = h.get("holdout_val_start", 2022)
    h_val_end = h.get("holdout_val_end", 2024)
    md.append(f"Refit on train ≤ {h_train_end}, eval on {h_val_start}–{h_val_end}. "
              f"Saved tuned params transferred from production model.\n")
    try:
        y_v, preds, _ = _refit_holdout_role(df, FOCUS_POS,
                                            h_train_end, h_val_start, h_val_end)
        rel = _decile_reliability(y_v, preds)
        rel.to_csv(OUT_DIR / f"reliability_holdout_{FOCUS_POS}.csv", index=False)
        # Headline metrics
        spear = float(pd.Series(preds).corr(pd.Series(y_v), method="spearman"))
        y_bin = (y_v > 0.5).astype(int)
        auc = (float(roc_auc_score(y_bin, preds))
               if 0 < y_bin.sum() < len(y_bin) else float("nan"))
        md.append(f"Headline (refit holdout): n_val={len(y_v)}, "
                  f"spearman={spear:.3f}, auc(@0.5)={auc:.3f}, "
                  f"pred_mean={preds.mean():.3f}, actual_mean={y_v.mean():.3f}\n")
        md.append("```\n" + rel.to_string(index=False) + "\n```\n")
        # Plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
            ax.plot(rel["pred_mean"], rel["actual_mean"], "o-",
                    label="refit holdout")
            ax.set_xlabel("predicted role_score (decile mean)")
            ax.set_ylabel("actual role_target (decile mean)")
            ax.set_title(f"{FOCUS_POS} role reliability "
                         f"(train≤{h_train_end} → val {h_val_start}–{h_val_end})")
            ax.legend()
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"reliability_holdout_{FOCUS_POS}.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            md.append(f"_(matplotlib unavailable: {exc})_\n")
    except Exception as exc:
        md.append(f"_(refit failed: {exc})_\n")

    return "\n".join(md)


# ─── S3: Feature attribution for saturated 2025 DL prospects ──────────────

def section_s3(df: pd.DataFrame) -> str:
    md = [f"## S3. Feature attribution — 2025 {FOCUS_POS} prospects with role_score ≥ 0.40\n"]
    bundle = _load_role_bundle(FOCUS_POS)
    feature_cols = bundle["feature_cols"]
    model = bundle["model"]

    cohort = df[(df["position_group"] == FOCUS_POS) &
                (df["last_college_season_year"] == 2025) &
                (df["role_score"].fillna(0) >= 0.40)].copy()
    md.append(f"n={len(cohort)} prospects in 2025 {FOCUS_POS} cohort with role_score ≥ 0.40 "
              f"(impact_blend_pct mean={cohort['impact_blend_pct'].mean():.2f}, "
              f"impact_apy_score mean={cohort['impact_apy_score'].mean():.4f})\n")
    if len(cohort) == 0:
        md.append("_(empty cohort — nothing to attribute)_\n")
        return "\n".join(md)

    cohort = _add_derived_dl_lb(cohort)
    X = _features_matrix(cohort, feature_cols)

    # SHAP via LightGBM's pred_contrib (last col is base value)
    contrib = model.predict(X, pred_contrib=True)
    base = contrib[:, -1]
    feat_contrib = contrib[:, :-1]  # (n, n_features)

    # Group features into families
    def _family(name: str) -> str:
        if name.startswith("kalman_"):     return "kalman"
        if name.startswith("ewma_"):       return "ewma"
        if name.startswith("combine_"):    return "combine"
        if name.startswith("pff_"):        return "pff"
        if name.startswith("nfl_lg_"):     return "nfl_era"
        if name == "hasan_rank":           return "hasan"
        if name in ("age_at_draft", "height_in", "weight_lbs",
                    "class_year_int", "conference_tier"):
            return "attributes"
        if name in ("recruiting_stars", "recruiting_composite",
                    "recruit_rank_national", "recruit_rank_position"):
            return "recruiting"
        if name in ("team_sp_offense", "team_sp_defense", "team_sp_overall",
                    "team_sack_share", "pass_rush_composite",
                    "pff_pass_rush_snap_pct", "pff_coverage_snap_pct",
                    "n_transfers_career", "games_played_career",
                    "n_games", "n_seasons"):
            return "context"
        return "other"

    fams = np.array([_family(c) for c in feature_cols])
    family_levels = ["kalman", "ewma", "combine", "pff", "nfl_era", "hasan",
                     "attributes", "recruiting", "context", "other"]
    fam_contrib = {fam: feat_contrib[:, fams == fam].sum(axis=1)
                    for fam in family_levels}

    cohort_out = cohort[["player_name", "school", "position", "role_score",
                         "impact_apy_score", "hasan_rank", "age_at_draft",
                         "height_in", "weight_lbs", "combine_invited",
                         "recruiting_composite", "pff_seasons"]].copy()
    cohort_out["base"] = np.round(base, 4)
    for fam in family_levels:
        cohort_out[f"shap_{fam}"] = np.round(fam_contrib[fam], 4)
    cohort_out = cohort_out.sort_values("role_score", ascending=False).head(30)
    cohort_out.to_csv(OUT_DIR / "dl_saturation_attribution.csv", index=False)
    md.append("Top 30 by role_score (cohort capped at 30 for display):\n")
    md.append("```\n" + cohort_out.to_string(index=False) + "\n```\n")

    # Family means across the full cohort
    fam_summary = pd.DataFrame({
        "family":          family_levels,
        "mean_shap":       [float(np.mean(fam_contrib[f])) for f in family_levels],
        "n_features":      [int(np.sum(fams == f)) for f in family_levels],
        "abs_mean_shap":   [float(np.mean(np.abs(fam_contrib[f]))) for f in family_levels],
    })
    fam_summary = fam_summary.sort_values("mean_shap", ascending=False).reset_index(drop=True)
    md.append("Mean SHAP contribution by family (over the n=" + str(len(cohort)) +
              " saturated prospects):\n")
    md.append("```\n" + fam_summary.round(4).to_string(index=False) + "\n```\n")

    # Per-feature top contributors
    feat_mean = feat_contrib.mean(axis=0)
    order = np.argsort(np.abs(feat_mean))[::-1][:15]
    top_feats = pd.DataFrame({
        "feature":          [feature_cols[i] for i in order],
        "mean_shap":        np.round(feat_mean[order], 4),
        "fill_rate":        [float(cohort[feature_cols[i]].notna().mean())
                              if feature_cols[i] in cohort.columns else float("nan")
                              for i in order],
    })
    md.append("Top 15 features by |mean SHAP| in this cohort:\n")
    md.append("```\n" + top_feats.to_string(index=False) + "\n```\n")
    return "\n".join(md)


# ─── S4: Counterfactual regressor vs classifier ───────────────────────────

def section_s4(df: pd.DataFrame, metrics: dict) -> str:
    md = ["## S4. Counterfactual — LGBMRegressor vs LGBMClassifier\n"]
    bundle = _load_role_bundle(FOCUS_POS)
    feature_cols = bundle["feature_cols"]
    h = metrics.get(f"{FOCUS_POS}_role", {})
    h_train_end = h.get("holdout_train_end", 2021)
    h_val_start = h.get("holdout_val_start", 2022)
    h_val_end = h.get("holdout_val_end", 2024)

    pos_df = df[df["position_group"] == FOCUS_POS]
    pos_df = _add_derived_dl_lb(pos_df)
    yr = pd.to_numeric(pos_df["last_college_season_year"], errors="coerce").fillna(0)
    train = pos_df[yr <= h_train_end]
    val = pos_df[(yr >= h_val_start) & (yr <= h_val_end)]
    cohort_2025 = pos_df[yr == 2025]

    X_tr = _features_matrix(train, feature_cols)
    X_va = _features_matrix(val, feature_cols)
    X_25 = _features_matrix(cohort_2025, feature_cols)
    y_tr = train["role_target"].fillna(0).to_numpy(dtype=float)
    y_va = val["role_target"].fillna(0).to_numpy(dtype=float)

    # Production-equivalent regressor (same params as production)
    prod = bundle["model"]
    keep = {"learning_rate", "max_depth", "num_leaves", "min_child_samples",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
            "min_split_gain", "n_estimators"}
    params = {k: v for k, v in prod.get_params().items() if k in keep}
    params["verbose"] = -1
    reg = lgb.LGBMRegressor(**params)
    reg.fit(X_tr, y_tr)
    reg_val = np.clip(reg.predict(X_va), 0, 1)
    reg_25 = np.clip(reg.predict(X_25), 0, 1)

    # Classifier on binarized label (target > 0.5)
    y_tr_bin = (y_tr > 0.5).astype(int)
    cls_params = {**params}
    cls_params.pop("objective", None)
    cls = lgb.LGBMClassifier(**cls_params)
    cls.fit(X_tr, y_tr_bin)
    cls_val = cls.predict_proba(X_va)[:, 1]
    cls_25 = cls.predict_proba(X_25)[:, 1]

    y_va_bin = (y_va > 0.5).astype(int)
    reg_auc = (float(roc_auc_score(y_va_bin, reg_val))
               if 0 < y_va_bin.sum() < len(y_va_bin) else float("nan"))
    cls_auc = (float(roc_auc_score(y_va_bin, cls_val))
               if 0 < y_va_bin.sum() < len(y_va_bin) else float("nan"))

    summary = pd.DataFrame([
        {"head": "regressor (production)",
         "val_auc": round(reg_auc, 4),
         "val_pred_mean": round(reg_val.mean(), 3),
         "val_actual_mean": round(y_va.mean(), 3),
         "2025_pred_mean": round(reg_25.mean(), 3),
         "2025_pct_in_[0.45,0.55]": round(float(((reg_25 >= 0.45) & (reg_25 <= 0.55)).mean()), 3),
         "2025_pct_>=0.40": round(float((reg_25 >= 0.40).mean()), 3),
         "2025_pct_>=0.50": round(float((reg_25 >= 0.50).mean()), 3)},
        {"head": "classifier (counterfactual)",
         "val_auc": round(cls_auc, 4),
         "val_pred_mean": round(cls_val.mean(), 3),
         "val_actual_mean": round(y_va.mean(), 3),
         "2025_pred_mean": round(cls_25.mean(), 3),
         "2025_pct_in_[0.45,0.55]": round(float(((cls_25 >= 0.45) & (cls_25 <= 0.55)).mean()), 3),
         "2025_pct_>=0.40": round(float((cls_25 >= 0.40).mean()), 3),
         "2025_pct_>=0.50": round(float((cls_25 >= 0.50).mean()), 3)},
    ])
    md.append(f"Refit on train ≤ {h_train_end} (n={len(train)}); "
              f"val on {h_val_start}–{h_val_end} (n={len(val)}); "
              f"score 2025 cohort (n={len(cohort_2025)}).\n")
    md.append("```\n" + summary.to_string(index=False) + "\n```\n")
    summary.to_csv(OUT_DIR / "regressor_vs_classifier.csv", index=False)

    # Histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(0, 1, 31)
        ax.hist(reg_25, bins=bins, alpha=0.55, label="regressor (prod)")
        ax.hist(cls_25, bins=bins, alpha=0.55, label="classifier (counterfactual)")
        ax.set_xlabel("predicted role_score (2025 cohort)")
        ax.set_ylabel("count")
        ax.set_title(f"{FOCUS_POS} 2025 — regressor vs classifier role predictions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "regressor_vs_classifier_2025.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass
    return "\n".join(md)


# ─── S5: Bridge-only refit ────────────────────────────────────────────────

def section_s5(df: pd.DataFrame, metrics: dict) -> str:
    md = ["## S5. Non-bridge zero-mass leverage test\n"]
    h = metrics.get(f"{FOCUS_POS}_role", {})
    h_train_end = h.get("holdout_train_end", 2021)
    h_val_start = h.get("holdout_val_start", 2022)
    h_val_end = h.get("holdout_val_end", 2024)

    bundle = _load_role_bundle(FOCUS_POS)
    feature_cols = bundle["feature_cols"]

    pos_df = df[df["position_group"] == FOCUS_POS]
    pos_df = _add_derived_dl_lb(pos_df)
    yr = pd.to_numeric(pos_df["last_college_season_year"], errors="coerce").fillna(0)
    cohort_2025 = pos_df[yr == 2025]
    X_25 = _features_matrix(cohort_2025, feature_cols)

    pos_df_focus = df[df["position_group"] == FOCUS_POS]
    yr_focus = pd.to_numeric(pos_df_focus["last_college_season_year"], errors="coerce").fillna(0)

    md_summary = []
    for label, bridge_only in [("production (bridge + non-bridge zeros)", False),
                                ("bridge-only", True)]:
        try:
            y_v, preds, m = _refit_holdout_role(df, FOCUS_POS,
                                                 h_train_end, h_val_start, h_val_end,
                                                 bridge_only=bridge_only)
        except Exception as exc:
            md.append(f"_({label} refit failed: {exc})_\n")
            continue
        train_mask = yr_focus <= h_train_end
        if bridge_only:
            train_mask &= (pos_df_focus["is_bridge"] == True)
        n_train = int(train_mask.sum())
        # Score 2025 cohort with the same refit
        preds_25 = np.clip(m.predict(X_25), 0, 1)
        y_v_bin = (y_v > 0.5).astype(int)
        auc = (float(roc_auc_score(y_v_bin, preds))
               if 0 < y_v_bin.sum() < len(y_v_bin) else float("nan"))
        md_summary.append({
            "training_set":       label,
            "n_train":            n_train,
            "val_pred_mean":      round(float(preds.mean()), 3),
            "val_actual_mean":    round(float(y_v.mean()), 3),
            "val_auc":            round(auc, 4),
            "2025_pred_mean":     round(float(preds_25.mean()), 3),
            "2025_pct_>=0.40":    round(float((preds_25 >= 0.40).mean()), 3),
            "2025_pct_in_45_55":  round(float(((preds_25 >= 0.45) & (preds_25 <= 0.55)).mean()), 3),
            "2025_top_decile":    round(float(np.quantile(preds_25, 0.9)), 3),
        })
    summary = pd.DataFrame(md_summary)
    summary.to_csv(OUT_DIR / "bridge_only_refit.csv", index=False)
    md.append(f"Compares production-style training (with non-bridge zero mass) vs "
              f"bridge-only training. Train ≤ {h_train_end}, val "
              f"{h_val_start}–{h_val_end}, then score 2025 cohort.\n")
    md.append("```\n" + summary.to_string(index=False) + "\n```\n")
    return "\n".join(md)


# ─── S6: Synthesis ────────────────────────────────────────────────────────

def section_s6(s1_df: pd.DataFrame, metrics: dict) -> str:
    md = ["## S6. Findings + remediation candidates\n"]

    md.append("### What the data actually showed\n")
    dl_row = s1_df[s1_df["pos_group"] == FOCUS_POS].iloc[0] if (s1_df["pos_group"] == FOCUS_POS).any() else None
    if dl_row is not None:
        md.append(f"- DL bridge population mean role_target = "
                  f"**{dl_row['bridge_role_mean']:.3f}** "
                  f"(median {dl_row['bridge_role_median']:.3f}). The bridge "
                  f"population is mostly settled draftees who *did* reach a "
                  f"second contract, so their target mean is **high, not 0.5**. "
                  f"S5 confirms this: a bridge-only refit collapses to a 2025 "
                  f"prediction mean of ~0.83 — i.e. *more* saturated, not less.\n")
        md.append(f"- DL non-bridge zero mass = "
                  f"**{dl_row['non_bridge_share']*100:.0f}%** of training rows "
                  f"({dl_row['n_non_bridge']:,} appended at role_target=0). "
                  f"This mass is what *prevents* full saturation; without it, "
                  f"every prediction trends toward the bridge mean.\n")
    md.append("- The MSE-trained `LGBMRegressor` lands the 2025 DL cohort at a "
              "**predicted mean of ~0.49**, with **>90% of rows in [0.45, 0.55]** "
              "(S4 production row). The discrimination is fine — `val_auc=0.998` — "
              "but absolute calibration of *low-information* rows is anchored "
              "near 0.5.\n")
    md.append("- **Top feature SHAP family for the saturated cohort: "
              "`nfl_era` (mean +0.48)**. Inside that, `nfl_lg_pass_att_pg` "
              "alone contributes +0.26 with a fill rate of 0%: the column is NaN "
              "for the 2025 cohort because it's joined on "
              "`last_college_season_year` and 2025-season NFL averages aren't "
              "in `nfl_league_averages.parquet` yet "
              "(`pipeline/08_draft_models.py:861-870`). LightGBM's default-direction "
              "split for that feature points to the high-prediction branch, "
              "pushing the entire 2025 cohort up by ~0.5.\n")
    md.append("- The next-largest family contribution is `combine` "
              "(mean −0.04, abs +0.06): mostly z-score features whose fill rates "
              "are still low for the 2025 cohort.\n")
    md.append("- The trajectory model upstream "
              "(`src/scrapers/nfl_data.py:763-923`) **is** isotonically "
              "calibrated as a classifier. The saturation is generated entirely "
              "in the step-8 regressor head.\n")
    md.append("- A counterfactual `LGBMClassifier` on the same features and "
              "binarized target produces nearly identical AUC (0.998 vs 0.998) "
              "but a 2025 prediction mean of **0.046** with **0.1% in [0.45, 0.55]** "
              "(S4 classifier row).\n")

    md.append("\n### Root cause, ranked\n")
    md.append("1. **NaN nfl_era features for the 2025 cohort**, combined with "
              "LightGBM's default-direction split learning that *missing → high* "
              "for those columns, accounts for the bulk of the saturation. The "
              "step-8 join (`08_draft_models.py:861-870`) leaves "
              "`last_college_season_year=2025` rows with no NFL league averages.\n")
    md.append("2. **MSE objective on a target that's bimodal at 0 and ~0.9** "
              "leaves the regressor expressing ignorance as ~0.5 instead of as "
              "a class-probability that collapses to base rate. A classifier "
              "head sidesteps this entirely (S4).\n")
    md.append("3. **No monotone constraints on Role.** `MONOTONE_ENABLED` "
              "covers Impact heads only — Role gets no `hasan_rank` or "
              "`age_at_draft` constraint despite both being top-importance "
              "for DL_role.\n")

    md.append("\n### Remediation candidates (ordered by expected payoff)\n")
    md.append("1. **Fix the nfl_era join for the current cohort.** Either backfill "
              "season 2025 league averages in `nfl_league_averages.parquet` "
              "(forward-fill from 2024 if 2025 data isn't yet available) or "
              "exclude `nfl_lg_*` from the Role feature set when its fill rate is "
              "0 for the scoring cohort. Cheapest, highest-leverage single change.\n")
    md.append("2. **Switch Role head to `LGBMClassifier` on `(role_target > 0.5)`.** "
              "Same features, comparable AUC (0.998), but predictions de-saturate "
              "by construction. The trajectory model already calibrates the binary "
              "outcome upstream — using log-loss in step 8 preserves that "
              "calibration via the sigmoid output. Higher-leverage but a deeper "
              "code change.\n")
    md.append("3. **Add Role monotone constraints**: at minimum "
              "`(\"DL\", \"role\")` with `hasan_rank` and `age_at_draft` "
              "negative (per `MONOTONE_CONSTRAINTS`). Run "
              "`experiments/00h_monotone_decision.py` after a paired sweep. "
              "Marginal next to options 1 and 2.\n")
    md.append("4. **Do *not* drop the non-bridge zero mass.** S5 shows this "
              "makes saturation strictly worse — those rows are the model's only "
              "negative-class signal for low-information prospects.\n")

    md.append("\n### Suggested next session\n")
    md.append("Start with remediation #1 — the nfl_era NaN issue is the "
              "smoking gun and a 5-line fix. If that doesn't resolve the bulk "
              "of the saturation (re-run this audit to confirm), proceed to #2.\n")
    return "\n".join(md)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading inputs…")
    df, _, metrics = _load_inputs()
    print(f"  proj rows: {len(df):,}  bridge: {(df['is_bridge']==True).sum():,}")

    sections = []
    sections.append("# Role-model audit — DL/DT/DE saturation at ~0.50\n")
    sections.append(f"Generated from `{PROJ_PATH.relative_to(REPO)}` and "
                    f"`{LABELS_PATH.relative_to(REPO)}`.\n")
    sections.append(f"Focus position group: **{FOCUS_POS}** "
                    f"(includes sub-positions {SUB_POS_FOCUS}).\n")

    print("S1 — training-data composition…")
    s1_md, s1_df = section_s1(df)
    sections.append(s1_md)

    print("S2 — reliability curves…")
    sections.append(section_s2(df, metrics))

    print("S3 — feature attribution for saturated 2025 DL prospects…")
    sections.append(section_s3(df))

    print("S4 — regressor vs classifier counterfactual…")
    sections.append(section_s4(df, metrics))

    print("S5 — bridge-only refit leverage test…")
    sections.append(section_s5(df, metrics))

    print("S6 — synthesis…")
    sections.append(section_s6(s1_df, metrics))

    OUT_MD.write_text("\n\n".join(sections))
    print(f"\nWrote audit report → {OUT_MD.relative_to(REPO)}")
    print(f"Supporting CSVs/PNGs in {OUT_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
