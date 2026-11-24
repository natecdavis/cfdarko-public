"""
Step 10: Final Output & Rankings

Joins all data sources into a final projections file:
  - Skill snapshot (Kalman/EWMA features)
  - Draft model scores (impact_score; the role heads were dropped 2026-10-31)
  - Player attributes (position, height, weight)
  - Historical comps (best comp name + outcome)
  - Percentile ranks within position group

Output: data/output/projections/projections_{date}.parquet
        data/output/projections/latest.parquet
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))   # for sibling helpers
from config import (
    PROCESSED_DIR, OUTPUT_DIR, RAW_DIR, POSITION_GROUPS, MODELED_STATS,
    DERIVED_AT_INFERENCE, SALARY_CAP, GLOBAL_SCORE_OVERRIDES,
    IMPACT_TARGET_SOURCE, END_YEAR, VALUE_USAGE_SEASON_OFFSETS,
)
# Shared snapshot assembly (single source of truth across steps 8/8b/8c/9/10;
# merges player_attributes via config.ATTRIBUTE_MERGE_COLS internally). Step
# 10 assembles WITHOUT the derived-feature pass — those are computed at
# inference time per position on a local group_df copy (DERIVED_AT_INFERENCE).
from _snapshot_assembly import assemble_snapshot
from src.models.merged_board import MergedBoard
from src.models.draft_pool import draft_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


_SHAP_COHORT_YEAR_OFFSET = 2  # SHAP for last_college_season_year >= END_YEAR - this


def _compute_shap_explanations(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Compute per-prospect SHAP top contributors for the impact_apy head.

    Only runs for the current scoring cohort (recent classes) — full-population
    SHAP is wasteful since the frontend only renders recent prospects. Stores
    results as a JSON-encoded `shap_top_features_impact` column with shape
    `[{"feature": str, "shap": float, "value": float}, ...]` ordered by
    |shap| descending.
    """
    try:
        import shap
    except ImportError:
        logger.info("shap not installed — skipping per-prospect explanations")
        return snapshot
    from config import END_YEAR

    snapshot = snapshot.copy()
    snapshot["shap_top_features_impact"] = None

    models_dir = PROCESSED_DIR / "draft_models"
    if "last_college_season_year" not in snapshot.columns:
        return snapshot

    cohort_mask = (snapshot["last_college_season_year"]
                   >= (END_YEAR - _SHAP_COHORT_YEAR_OFFSET))
    if not cohort_mask.any():
        return snapshot

    n_explained = 0
    _global_bundle_cache: dict = {}
    for pos_group in POSITION_GROUPS.keys():
        # Explanations must come from the model that produced the score:
        # positions whose impact_apy is globally overridden (w < 0.5) get
        # SHAP from the global impact_apy model.
        w_override = GLOBAL_SCORE_OVERRIDES.get((pos_group, "impact_apy"))
        use_global = w_override is not None and w_override < 0.5
        impact_path = (PROCESSED_DIR / "global_draft_models" / "global_impact_apy_model.pkl"
                       if use_global
                       else models_dir / f"{pos_group}_impact_model.pkl")
        if not impact_path.exists():
            continue
        if "position_group" not in snapshot.columns:
            continue
        pos_mask = (snapshot["position_group"] == pos_group) & cohort_mask
        if not pos_mask.any():
            continue
        try:
            if use_global and _global_bundle_cache:
                bundle = _global_bundle_cache
            else:
                with open(impact_path, "rb") as f:
                    bundle = pickle.load(f)
                if use_global:
                    _global_bundle_cache = bundle
            model = bundle["model"]
            feature_cols = bundle["feature_cols"]
            X = (snapshot.loc[pos_mask]
                 .reindex(columns=feature_cols, fill_value=np.nan)
                 .to_numpy(dtype=float, na_value=np.nan))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            # Top 5 by absolute value per row.
            top_lists: list[str] = []
            for row_i in range(shap_values.shape[0]):
                row_shap = shap_values[row_i]
                row_x    = X[row_i]
                # Sort descending by |shap|.
                order = np.argsort(np.abs(row_shap))[::-1][:5]
                top = []
                for j in order:
                    sv = float(row_shap[j])
                    fv = float(row_x[j]) if not np.isnan(row_x[j]) else None
                    top.append({"feature": feature_cols[j],
                                "shap":    round(sv, 6),
                                "value":   None if fv is None else round(fv, 4)})
                top_lists.append(json.dumps(top))
            snapshot.loc[pos_mask, "shap_top_features_impact"] = top_lists
            n_explained += pos_mask.sum()
        except Exception as exc:
            logger.warning("SHAP failed for %s: %s", pos_group, exc)

    logger.info("Computed SHAP explanations for %d current-cohort prospects",
                int(n_explained))
    return snapshot


def _audit_model_feature_alignment(snapshot: pd.DataFrame, models_dir: Path) -> None:
    """Warn when any trained model expects a feature absent from the snapshot.

    Excludes features in `DERIVED_AT_INFERENCE` since `_compute_role_impact_scores`
    adds those per-position via a local `group_df` copy. Catches the silent
    NaN-at-inference bug class — historically Step 10's `attr_cols` had
    drifted from Step 8's training schema, scoring current prospects with
    LightGBM's missing-branch behavior on real, available features.
    """
    if not models_dir.exists():
        return
    cols = set(snapshot.columns)
    n_warned = 0
    for pkl in sorted(models_dir.glob("*_model.pkl")):
        try:
            with open(pkl, "rb") as f:
                bundle = pickle.load(f)
            missing = [c for c in bundle.get("feature_cols", [])
                       if c not in cols and c not in DERIVED_AT_INFERENCE]
            if missing:
                logger.warning(
                    "Model %s expects %d features missing from snapshot: %s",
                    pkl.name, len(missing),
                    ", ".join(missing[:8]) + ("..." if len(missing) > 8 else ""),
                )
                n_warned += 1
        except Exception as exc:
            logger.warning("Could not inspect %s for feature alignment: %s",
                           pkl.name, exc)
    if n_warned == 0:
        logger.info("Model feature alignment: all training features present in snapshot")


_CALIBRATOR_CACHE: dict[Path, dict] = {}


def _apply_calibrator(cal_path: Path, preds: np.ndarray) -> np.ndarray:
    """Apply a step 8c calibrator bundle to raw LightGBM predictions.

    Bundle shape (from pipeline/08c_calibrate_draft_scores.py):
        {"method": "isotonic" | "platt_ridge",
         "calibrator": <fitted sklearn obj>,
         "y_min": float, "y_max": float}

    Per-pos pickles are loaded once and cached for the lifetime of the
    process; step 10 calls this in a tight per-position loop.
    """
    bundle = _CALIBRATOR_CACHE.get(cal_path)
    if bundle is None:
        try:
            with open(cal_path, "rb") as f:
                bundle = pickle.load(f)
            _CALIBRATOR_CACHE[cal_path] = bundle
        except Exception as exc:
            logger.warning("Failed to load calibrator %s: %s — using raw preds", cal_path, exc)
            return preds
    method = bundle.get("method", "identity")
    cal = bundle.get("calibrator")
    y_min = bundle.get("y_min", 0.0)
    y_max = bundle.get("y_max")
    if method == "isotonic":
        out = cal.predict(preds)
    elif method == "platt_ridge":
        out = cal.predict(preds.reshape(-1, 1))
    else:
        out = preds
    if y_max is not None:
        out = np.clip(out, y_min, y_max)
    return out


def _clamp_impact_band(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Clamp impact_apy_p10/p90 so the band brackets the MSE point estimate:
    p10 ≤ score ≤ p90. The point estimate is preserved unchanged."""
    if not {"impact_apy_p10", "impact_apy_score", "impact_apy_p90"}.issubset(snapshot.columns):
        return snapshot
    score = snapshot["impact_apy_score"]
    p10 = snapshot["impact_apy_p10"]
    p90 = snapshot["impact_apy_p90"]
    snapshot["impact_apy_p10"] = np.minimum(np.minimum(p10, p90), score)
    snapshot["impact_apy_p90"] = np.maximum(np.maximum(p10, p90), score)
    return snapshot


def _compute_role_impact_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Apply draft models to all players to generate role/impact scores.

    Three impact heads are scored when their model files exist:
      - impact_apy_score (from {pos}_impact_model.pkl, target = APY%cap)
      - impact_wav_score (from {pos}_impact_wav_model.pkl, target = w_AV/game)
      - impact_score    = impact_apy_score (raw APY % of cap; primary, has
                          natural cross-position interpretation)
      - impact_blend_pct = mean of within-(position-group) percentile ranks
                          of the two heads (secondary; only sensible WITHIN a
                          position group, not for cross-position ranking).
    """
    models_dir = PROCESSED_DIR / "draft_models"
    snapshot = snapshot.copy()
    snapshot["impact_apy_score"] = np.nan
    snapshot["impact_apy_p10"] = np.nan
    snapshot["impact_apy_p90"] = np.nan
    snapshot["impact_wav_score"] = np.nan
    snapshot["impact_score"] = np.nan
    snapshot["impact_blend_pct"] = np.nan
    # Raw LightGBM outputs preserved alongside the calibrated columns so
    # ranking within an isotonic flat region falls back to the model's
    # underlying ordering. The user-facing score is still the calibrated
    # one; raw_* is only consulted as a tiebreaker.
    snapshot["raw_impact_wav_score"] = np.nan

    for pos_group in POSITION_GROUPS.keys():
        impact_path = models_dir / f"{pos_group}_impact_model.pkl"
        impact_wav_path = models_dir / f"{pos_group}_impact_wav_model.pkl"

        if "position_group" in snapshot.columns:
            mask = snapshot["position_group"] == pos_group
        else:
            mask = pd.Series(True, index=snapshot.index)

        if not mask.any():
            continue

        group_df = snapshot[mask]

        # Apply derived features so inference matches training feature set
        if pos_group in ("DL", "EDGE", "LB"):
            group_df = group_df.copy()
            s = group_df.get("kalman_mean_sacks", pd.Series(0, index=group_df.index)).fillna(0)
            t = group_df.get("kalman_mean_tfl", pd.Series(0, index=group_df.index)).fillna(0)
            h = group_df.get("kalman_mean_qb_hurries", pd.Series(0, index=group_df.index)).fillna(0)
            group_df["pass_rush_composite"] = s * 2.0 + t * 1.0 + h * 0.5
            def_snaps = group_df.get("pff_snap_counts_defense_mean",
                                     pd.Series(np.nan, index=group_df.index)).replace(0, np.nan)
            pr_snaps  = group_df.get("pff_snap_counts_pass_rush_mean",
                                     pd.Series(np.nan, index=group_df.index))
            group_df["pff_pass_rush_snap_pct"] = (pr_snaps / def_snaps).clip(0, 1)
            if pos_group == "LB":
                cov_snaps = group_df.get("pff_snap_counts_coverage_mean",
                                         pd.Series(np.nan, index=group_df.index))
                group_df["pff_coverage_snap_pct"] = (cov_snaps / def_snaps).clip(0, 1)

        if pos_group == "OL" and "position" in group_df.columns:
            group_df = group_df.copy()
            group_df["is_ol_tackle"] = group_df["position"].isin(["OT", "T"]).astype(float)
            group_df["is_ol_guard"]  = group_df["position"].isin(["OG", "IOL", "G"]).astype(float)
            group_df["is_ol_center"] = (group_df["position"] == "C").astype(float)

        if pos_group == "DB":
            group_df = group_df.copy()
            def_snaps = group_df.get("pff_snap_counts_defense_mean",
                                     pd.Series(np.nan, index=group_df.index)).replace(0, np.nan)
            for snap_col, pct_col in [
                ("pff_snap_counts_corner_mean", "pff_corner_snap_pct"),
                ("pff_snap_counts_slot_mean",   "pff_slot_snap_pct"),
                ("pff_snap_counts_fs_mean",     "pff_fs_snap_pct"),
            ]:
                group_df[pct_col] = (group_df.get(snap_col, pd.Series(np.nan, index=group_df.index))
                                     / def_snaps).clip(0, 1)

        for model_path, score_col, calibrator_key in [
            (impact_path,      "impact_apy_score",  None),         # not calibrated; drives dollar columns
            (impact_wav_path,  "impact_wav_score",  "impact_wav"),
        ]:
            if not model_path.exists():
                logger.debug("No model at %s", model_path)
                continue
            try:
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                model = bundle["model"]
                feature_cols = bundle["feature_cols"]

                if not any(c in group_df.columns for c in feature_cols):
                    continue

                X = group_df.reindex(columns=feature_cols, fill_value=np.nan).to_numpy(dtype=float, na_value=np.nan)
                preds = model.predict(X)

                # Post-hoc isotonic / Platt-Ridge calibration (step 8c output).
                # Falls through to raw when no calibrator exists for this head
                # (impact_apy is never calibrated; small-n heads may have
                # written no pickle when n_positive < PLATT_THRESHOLD).
                if calibrator_key is not None:
                    raw_preds = np.clip(preds, 0.0, None)
                    cal_path = models_dir / f"{pos_group}_{calibrator_key}_calibrator.pkl"
                    if cal_path.exists():
                        preds = _apply_calibrator(cal_path, preds)
                    raw_col = f"raw_{score_col}"
                    if raw_col in snapshot.columns:
                        snapshot.loc[mask, raw_col] = raw_preds

                preds = np.clip(preds, 0.0, None)

                snapshot.loc[mask, score_col] = preds

                # Apply quantile companion models for impact_apy → p10/p90.
                if score_col == "impact_apy_score" and "quantile_models" in bundle:
                    qm = bundle["quantile_models"]
                    for suffix, target_col in [("p10", "impact_apy_p10"),
                                                ("p90", "impact_apy_p90")]:
                        m = qm.get(suffix)
                        if m is None:
                            continue
                        try:
                            qpreds = np.clip(m.predict(X), 0.0, None)
                            snapshot.loc[mask, target_col] = qpreds
                        except Exception as exc:
                            logger.error("Quantile %s scoring failed for %s: %s",
                                         suffix, pos_group, exc)
            except Exception as exc:
                logger.error("Failed scoring %s/%s: %s", pos_group, score_col, exc)

    # Quantile band consistency: p10/p90 are independent LightGBM heads with
    # no constraint relative to the MSE point estimate; on feature-sparse rows
    # they can cross (e.g. p90 < score). Preserve the well-tuned point estimate
    # and clamp p10/p90 so the band always brackets it: p10 ≤ score ≤ p90.
    snapshot = _clamp_impact_band(snapshot)

    # impact_score: raw APY%cap from the impact_apy head — has natural
    # cross-position meaning (a kicker projecting at 0.005 of cap is clearly
    # below a WR projecting at 0.04). For within-position ranking that uses
    # both heads, see `impact_blend_pct` below. For human-readable cap-dollar
    # values, see `impact_apy_dollars` (and `_p10`/`_p90` companions).
    snapshot["impact_score"] = snapshot["impact_apy_score"]

    # Cap-dollar conversions for human readability. Same signal as the
    # apy_cap_pct columns; just multiplied by the current SALARY_CAP.
    for src, dst in [("impact_apy_score", "impact_apy_dollars"),
                     ("impact_apy_p10",   "impact_apy_dollars_p10"),
                     ("impact_apy_p90",   "impact_apy_dollars_p90")]:
        if src in snapshot.columns:
            snapshot[dst] = snapshot[src] * SALARY_CAP

    # Secondary view: percentile-rank blend across the two impact heads.
    # Only meaningful for ranking within a position group.
    if "position_group" in snapshot.columns:
        apy_pct = snapshot.groupby("position_group")["impact_apy_score"].rank(pct=True)
        wav_pct = snapshot.groupby("position_group")["impact_wav_score"].rank(pct=True)
        both = apy_pct.notna() & wav_pct.notna()
        snapshot["impact_blend_pct"] = np.where(
            both,
            (apy_pct + wav_pct) / 2.0,
            apy_pct.fillna(wav_pct),
        )
        snapshot["impact_blend_pct"] = pd.to_numeric(snapshot["impact_blend_pct"],
                                                     errors="coerce")

    return snapshot


def _load_replacement_levels() -> dict[str, float]:
    """Return {position_group: replacement_apy_pct} from the step-7 artifact.

    Empty dict if the artifact is missing — callers degrade to no adjustment
    rather than failing, so an older processed/ directory still runs.
    """
    path = PROCESSED_DIR / "replacement_levels.parquet"
    if not path.exists():
        logger.warning("replacement_levels.parquet missing — run step 7 to "
                       "generate it; skipping replacement columns")
        return {}
    lv = pd.read_parquet(path)
    pos = lv[lv["scope"] == "position"]
    return dict(zip(pos["key"].astype(str),
                    pos["replacement_apy_pct"].astype(float)))


def _merged_board_engine(snapshot: pd.DataFrame, labels_path: Path):
    """Off-board engine for the merged board: (OOF frame for the calibrator,
    per-row score for the snapshot, engine name).

    Prefers step 8d's dedicated off-board model (seed-bagged P(NFL) ×
    E[usage | NFL] with PFF side features; 00zi, 2026-11-01). Falls back to
    the deployed 8b impact model when 8d's outputs are missing. The OOF and
    the snapshot score must come from the SAME model: the calibrator maps
    that model's within-class percentile to usage.
    """
    from src.models.offboard_model import pid
    ob_oof = PROCESSED_DIR / "offboard_models" / "offboard_oof_predictions.parquet"
    ob_pred = PROCESSED_DIR / "offboard_predictions.parquet"
    keys = snapshot[["player_id", "last_college_season_year", "hasan_rank"]].drop_duplicates("player_id")
    if ob_oof.exists() and ob_pred.exists():
        oof = (pd.read_parquet(ob_oof).drop_duplicates("player_id", keep="last")
               .rename(columns={"score": "pred"}))
        pred = pd.read_parquet(ob_pred).drop_duplicates("player_id")
        score = pd.Series(pid(snapshot["player_id"])).map(
            pred.set_index("player_id")["offboard_score"])
        for c in ("offboard_p_nfl", "offboard_usage_if_nfl", "offboard_score"):
            snapshot[c] = pd.Series(pid(snapshot["player_id"])).map(
                pred.set_index("player_id")[c]).to_numpy()
        return oof[["player_id", "pos_group", "last_college_season_year", "hasan_rank",
                    "nfl", "usage", "pred"]], score.to_numpy(dtype=float), "offboard_8d"

    logger.warning("Merged board: step 8d outputs missing — falling back to the "
                   "deployed 8b impact model as the off-board engine")
    oof = pd.read_parquet(PROCESSED_DIR / "global_draft_models" / "global_oof_predictions.parquet")
    oof = (oof[oof["model_name"] == "impact_apy"]
           .drop_duplicates("player_id", keep="last")[["player_id", "position_group", "pred"]]
           .rename(columns={"position_group": "pos_group"}))
    lab = pd.read_parquet(labels_path, columns=["player_id", "is_bridge", "usage_y34"])
    lab = lab.dropna(subset=["player_id"]).drop_duplicates("player_id")
    oof = oof.merge(keys, on="player_id", how="inner").merge(lab, on="player_id", how="left")
    oof["nfl"] = oof["is_bridge"].fillna(False).astype(bool)
    oof["usage"] = np.where(oof["nfl"], oof["usage_y34"].fillna(0.0), 0.0)
    raw = snapshot["raw_impact_score"] if "raw_impact_score" in snapshot.columns else None
    score = raw.fillna(snapshot["impact_score"]) if raw is not None else snapshot["impact_score"]
    return oof, score.to_numpy(dtype=float), "deployed_8b"


def _add_draft_pool(snapshot: pd.DataFrame) -> pd.DataFrame:
    """draft_pool: True for players in the newest draft's pool (left school
    and three years out of high school, or drafted / signed / on the board
    for that draft — including players who sat out the last season), False
    for others in that class, NaN when the pool can't be determined (no
    next-season rosters yet) and for older classes."""
    snapshot["draft_pool"] = pd.array([pd.NA] * len(snapshot), dtype="boolean")
    last = snapshot["last_college_season_year"]
    if last.notna().sum() == 0:
        return snapshot
    draft_year = int(last.max()) + 1
    pool = draft_pool(snapshot[["player_id", "last_college_season_year"]], draft_year,
                      RAW_DIR, PROCESSED_DIR)
    if pool is None:
        return snapshot
    in_scope = (last == draft_year - 1) | pool
    snapshot["draft_pool"] = pool.astype("boolean").where(in_scope, pd.NA)
    return snapshot


def _rookie_value_target(oof: pd.DataFrame) -> pd.DataFrame | None:
    """Add `rv` (rookie-deal value, 0 if never NFL) to the engine OOF from
    step 7.5's label; drop NFL rows without a value. None if no label."""
    from src.models.offboard_model import pid
    path = PROCESSED_DIR / "rookie_value_labels.parquet"
    if not path.exists():
        logger.warning("Merged board: %s missing (run step 7.5) — value board skipped", path)
        return None
    L = pd.read_parquet(path, columns=["player_id", "value_y14"]).dropna(subset=["player_id"])
    rv = L.assign(pid=pid(L["player_id"])).drop_duplicates("pid").set_index("pid")["value_y14"]
    oof = oof.copy()
    oof["rv"] = np.where(oof["nfl"], pd.Series(pid(oof["player_id"])).map(rv).to_numpy(), 0.0)
    miss = oof["rv"].isna()
    logger.info("Merged board (value): dropping %d NFL rows without a rookie value", int(miss.sum()))
    return oof[~miss]


def _partial_outcome_rows(snapshot: pd.DataFrame) -> pd.DataFrame | None:
    """Board players of the three most recent drafted classes with projected
    four-year rookie value (src/models/partial_outcomes.py, 00zv), added to
    the value board's board curve."""
    from src.models.partial_outcomes import board_value_frame, project_partial
    path = PROCESSED_DIR / "rookie_value_labels.parquet"
    if not path.exists():
        return None
    L = pd.read_parquet(path, columns=["player_id"] + [f"value_y{k}" for k in range(1, 5)])
    rows, diag = project_partial(board_value_frame(snapshot, L), END_YEAR)
    logger.info("Merged board (value): partial outcomes, projected board rows by draft year %s", diag)
    return rows


def _merged_rank(snapshot: pd.DataFrame, col: str, score: pd.Series) -> np.ndarray:
    """Rank within class by `col` (desc). Ties (capped off-board players share
    the cap; flat isotonic segments tie board players) break by board rank,
    then by the engine's raw score, then by player_id (deterministic: row
    order changes when the output is re-sorted)."""
    order = (pd.DataFrame({"cls": snapshot["last_college_season_year"].to_numpy(),
                           "u": snapshot[col].to_numpy(dtype=float),
                           "h": snapshot["hasan_rank"].to_numpy(dtype=float),
                           "s": np.asarray(score, dtype=float),
                           "pid": snapshot["player_id"].astype(str).to_numpy()})
             .reset_index(drop=True))
    order = order[order["u"].notna()].sort_values(
        ["cls", "u", "h", "s", "pid"], ascending=[True, False, True, False, True],
        na_position="last", kind="mergesort")
    rank = np.full(len(snapshot), np.nan)
    rank[order.index.to_numpy()] = order.groupby("cls").cumcount().to_numpy() + 1
    return rank


def _add_merged_board(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Merged board (src/models/merged_board.py): one list per class — board
    players by their board-rank curve, off-board players by the off-board
    engine's percentile (step 8d's model; see _merged_board_engine), capped
    at the off-board top-1% mean.

    Two currencies, same recipe:
      merged_value   expected rookie-deal value, Σ y1-y4 veteran-market price
                     of the role at the NFL position, % of cap (step 7.5's
                     label). merged_value_dollars = x SALARY_CAP (4-year $).
                     merged_rank ranks by this (experiments/00zs, 2026-11-01).
      merged_usage   expected y3-4 snap share (the original build), with
                     merged_usage_rank, kept for reference.
    merged_source / merged_engine describe both. At a given pick the rookie
    cost is set by the slot, so ranking available players by merged_value is
    ranking them by expected surplus at that pick (scripts/board_at_pick.py).

    Calibrators are fit on the engine's OOF predictions for classes whose
    outcome seasons have been played (y3-4 for usage, y1-4 for value: both
    end at last college season END_YEAR - 5). The value board's board curve
    also uses the three most recent drafted classes with projected four-year
    values (partial outcomes, experiments/00zv).
    """
    labels_path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    for col in ("merged_usage", "merged_usage_rank", "merged_value", "merged_value_dollars",
                "merged_rank"):
        snapshot[col] = np.nan
    snapshot["merged_source"] = None
    if not labels_path.exists():
        logger.warning("Merged board skipped: missing %s", labels_path)
        return snapshot
    oof, score, engine = _merged_board_engine(snapshot, labels_path)
    snapshot["merged_engine"] = engine

    # y3-4 = draft_year + VALUE_USAGE_SEASON_OFFSETS; the last played NFL
    # season is END_YEAR - 1 and draft_year = last college season + 1.
    fit_last_usage = END_YEAR - 2 - max(VALUE_USAGE_SEASON_OFFSETS)
    fit_last_value = END_YEAR - 5          # y4 = draft_year + 3 <= END_YEAR - 1
    boards = [("usage", MergedBoard.fit(oof, fit_last_usage, "usage", "merged_usage"))]
    oof_v = _rookie_value_target(oof)
    if oof_v is not None:
        extra = _partial_outcome_rows(snapshot)
        boards.append(("value", MergedBoard.fit(oof_v, fit_last_value, "rv", "merged_value",
                                                extra_board=extra)))

    snap = snapshot[["position_group", "last_college_season_year", "hasan_rank"]].rename(
        columns={"position_group": "pos_group"})
    snap["_score"] = score
    # Current class: score only the real draft pool when it can be
    # determined (src/models/draft_pool.py); every other class is scored
    # whole, as the calibrators were fit.
    pool = snapshot["draft_pool"] if "draft_pool" in snapshot.columns else None
    use_pool = pool is not None and pool.notna().any()
    if use_pool:
        cur = snapshot["last_college_season_year"].max()
        in_pool = pool.fillna(False).astype(bool)
        mask = (snapshot["last_college_season_year"] != cur) | in_pool
        logger.info("Merged board: current class scored within the %d-player draft pool",
                    int(in_pool.sum()))
    for name, mb in boards:
        pred = (mb.predict(snap, "_score", score_mask=mask.to_numpy(), ref_sizing=False)
                if use_pool else mb.predict(snap, "_score"))
        snapshot[mb.out_col] = pred[mb.out_col].astype(float).to_numpy()
        snapshot["merged_source"] = pred["merged_source"].to_numpy()
        d = mb.diagnostics
        logger.info("Merged board (%s): fit seasons %s, %d board (%d projected) + %d off-board rows (%d NFL); "
                    "caps %s (pooled %.4f); current class %s sized to %s",
                    name, d["fit_seasons"], d["n_board"], d["n_board_projected"], d["n_offboard"],
                    d["n_offboard_nfl"],
                    d["cap"], d["cap_pooled"], d.get("current_class"), d.get("n_ref"))
    logger.info("Merged board engine: %s; sources: %s", engine,
                snapshot["merged_source"].value_counts().to_dict())
    snapshot["merged_usage_rank"] = _merged_rank(snapshot, "merged_usage", snap["_score"])
    if oof_v is not None:
        snapshot["merged_value_dollars"] = snapshot["merged_value"] * SALARY_CAP
        snapshot["merged_rank"] = _merged_rank(snapshot, "merged_value", snap["_score"])
    else:
        snapshot["merged_rank"] = snapshot["merged_usage_rank"]
    return snapshot


def _add_replacement_columns(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Add value-above-replacement views of the impact_apy score.

    Runs AFTER `_apply_global_score_overrides` so it sees the final blended and
    calibrated `impact_apy_score`, and BEFORE `_add_percentile_ranks` so the
    percentile machinery picks up the new columns.

    `impact_score` itself is deliberately NOT changed: it is the raw APY%cap
    projection and is the source of the `impact_apy_dollars` columns, so
    rebasing it would break that identity and the frontend contract. The
    above-replacement view is additive.

    Note on magnitude: replacement cost is close to the league minimum at every
    position (≈0.0038 of cap) EXCEPT QB (≈0.0069) — even backup QBs command
    real money. So this subtraction is a modest, QB-focused correction, not a
    board-reshuffling one. That is an empirical finding about the contract
    market, not a limitation of the implementation: the marginal NFL player
    earns the minimum nearly everywhere, so APY%cap already approximates value
    above replacement for the other ten position groups.
    """
    levels = _load_replacement_levels()
    if not levels or "impact_apy_score" not in snapshot.columns:
        return snapshot
    if "position_group" not in snapshot.columns:
        return snapshot

    repl = snapshot["position_group"].astype(str).map(levels)
    snapshot["replacement_apy_pct"] = repl

    score = pd.to_numeric(snapshot["impact_apy_score"], errors="coerce")
    # Deliberately NOT floored at zero. The obvious "surplus value can't be
    # negative" clip was measured and rejected: because the impact heads train
    # on a zero-padded mixture, 94.8% of OOF predictions (and 28.6% of BRIDGE
    # predictions) sit below the replacement line, so clipping would collapse
    # almost the entire board into one tie and destroy within-position
    # ordering. A negative value reads as "projects below replacement level",
    # which is both meaningful and order-preserving. Consumers wanting the
    # floored version can clip at read time.
    ior = score - repl
    snapshot["impact_over_replacement"] = ior
    snapshot["impact_over_replacement_dollars"] = ior * SALARY_CAP

    # Cross-position ordinal on the replacement-adjusted scale. Ties are
    # inherited from step 8c's isotonic step function (subtracting a
    # per-position constant preserves them), so break them with the
    # pre-calibration model output the same way _add_percentile_ranks does.
    raw = pd.to_numeric(snapshot.get("raw_impact_score"), errors="coerce")
    composite = ior + (raw.fillna(0.0) * 1e-9 if raw is not None else 0.0)
    snapshot["ior_rank"] = composite.rank(ascending=False, method="min",
                                          na_option="keep")

    n = int(ior.notna().sum())
    covered = sorted(set(snapshot["position_group"].dropna().astype(str))
                     & set(levels))
    logger.info("Replacement-adjusted impact for %d players across %d position "
                "groups (%s)", n, len(covered), ", ".join(covered))
    missing = sorted(set(snapshot["position_group"].dropna().astype(str))
                     - set(levels))
    if missing:
        logger.info("  no replacement level for: %s (left NaN)",
                    ", ".join(missing))
    return snapshot


_GLOBAL_APY_BUNDLE: dict = {}


def _predict_global_quantiles(snapshot: pd.DataFrame, mask: pd.Series,
                              pos_group: str) -> dict[str, np.ndarray]:
    """p10/p90 from the global impact_apy bundle's quantile heads for the
    masked rows, with p90 transported through the position's isotonic
    calibrator so band and point estimate share a scale. Returns {} when the
    bundle has no quantile models — callers then keep the per-position band,
    which the clamp keeps bracketing the point estimate."""
    if not _GLOBAL_APY_BUNDLE:
        gdir = PROCESSED_DIR / "global_draft_models"
        try:
            with open(gdir / "global_impact_apy_model.pkl", "rb") as f:
                _GLOBAL_APY_BUNDLE.update(pickle.load(f))
        except Exception as exc:
            logger.warning("Global impact_apy bundle unavailable (%s) — "
                           "keeping per-position quantile bands", exc)
            _GLOBAL_APY_BUNDLE["_missing"] = True

    qm = _GLOBAL_APY_BUNDLE.get("quantile_models") or {}
    fc = _GLOBAL_APY_BUNDLE.get("feature_cols")
    if not qm or not fc:
        return {}
    X = (snapshot.loc[mask]
         .reindex(columns=fc, fill_value=np.nan)
         .to_numpy(dtype=float, na_value=np.nan))
    out = {}
    for suffix, m in qm.items():
        try:
            vals = m.predict(X)
            if suffix == "p90":
                # Transport the raw quantile prediction through the SAME
                # per-position isotonic map as the point estimate. Monotone
                # map ⇒ the band survives exactly where the quantile head
                # said upside exists (raw p90 > raw point, ~15-20% of
                # players — legitimate under a zero-inflated outcome: most
                # players' true q90 IS below their mean). Without transport
                # the calibrated point outruns the raw p90 and the clamp
                # collapses the band for ~95% of players. (Marginal CQR
                # offsets were tried first and correctly computed to 0 —
                # whole-population exceedance is the wrong target here; see
                # 8c's quantile_band_health.json for top-decile coverage.)
                cal_path = (PROCESSED_DIR / "global_draft_models"
                            / f"{pos_group}_impact_apy_global_calibrator.pkl")
                if cal_path.exists():
                    vals = _apply_calibrator(cal_path, vals)
            out[suffix] = np.clip(vals, 0.0, None)
        except Exception as exc:
            logger.warning("Global quantile %s failed: %s", suffix, exc)
    return out


def _apply_global_score_overrides(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Blend per-position scores with global-model scores per
    config.GLOBAL_SCORE_OVERRIDES ((pos, score) → weight on the per-position
    score; 0.0 = take the global score wholesale). Only rows with a non-null
    global score are touched; derived columns (impact_score, dollar
    conversions, quantile-band clamp, impact_blend_pct) are refreshed.
    """
    if not GLOBAL_SCORE_OVERRIDES or "position_group" not in snapshot.columns:
        return snapshot
    col_map = {
        "impact_apy": ("impact_apy_score", "global_impact_apy_score"),
        "impact_wav": ("impact_wav_score", "global_impact_wav_score"),
    }
    # Provenance: which model family produced each score for each row.
    for score in col_map:
        snapshot[f"{score}_score_source"] = "position"
    touched_apy = touched_wav = False
    for (pos, score), w in GLOBAL_SCORE_OVERRIDES.items():
        pos_col, g_col = col_map.get(score, (None, None))
        if (pos_col is None or pos_col not in snapshot.columns
                or g_col not in snapshot.columns):
            continue
        mask = (snapshot["position_group"] == pos) & snapshot[g_col].notna()
        if not mask.any():
            continue
        # Per-position calibration of the global score (step 8c fits these on
        # 8b's tuning-fold OOF preds). Without it the pooled model's LEVELS
        # aren't position-calibrated, and partial blends would mix scales.
        g_raw = snapshot.loc[mask, g_col].astype(float).to_numpy()
        g_cal_path = (PROCESSED_DIR / "global_draft_models"
                      / f"{pos}_{score}_global_calibrator.pkl")
        calibrated = g_cal_path.exists()
        g_vals = _apply_calibrator(g_cal_path, g_raw) if calibrated else g_raw
        g_series = pd.Series(g_vals, index=snapshot.index[mask])
        pos_vals = snapshot.loc[mask, pos_col].astype(float)
        blended = w * pos_vals + (1.0 - w) * g_series
        if w > 0:  # per-pos NaN with partial weight → fall back to global alone
            blended = blended.fillna(g_series)
        snapshot.loc[mask, pos_col] = blended
        # raw_* tiebreaker: isotonic calibrators are step functions, so
        # calibrated global scores tie within flat regions. Store the
        # pre-calibration blend so _add_percentile_ranks orders flat-region
        # prospects by the model's underlying gradient (same convention as
        # the per-position calibrators).
        raw_col = {"impact_apy": "raw_impact_score",
                   "impact_wav": "raw_impact_wav_score"}[score]
        if raw_col not in snapshot.columns:
            snapshot[raw_col] = np.nan
        raw_blend = w * pos_vals + (1.0 - w) * pd.Series(
            g_raw, index=snapshot.index[mask])
        snapshot.loc[mask, raw_col] = raw_blend
        snapshot.loc[mask, f"{score}_score_source"] = (
            "global" if w == 0.0 else f"blend(w_pos={w:.1f})")

        # Global quantile band for globally-scored impact_apy: the per-pos
        # p10/p90 heads band a point estimate that is no longer theirs.
        if score == "impact_apy" and w == 0.0:
            gq = _predict_global_quantiles(snapshot, mask, pos)
            for suffix, vals in gq.items():
                snapshot.loc[mask, f"impact_apy_{suffix}"] = vals
            # p10 = 0: the quantile audit showed a fitted p10 head loses to
            # the constant training quantile on this zero-inflated target
            # (true 10th-percentile outcome is a bust for nearly everyone).
            snapshot.loc[mask, "impact_apy_p10"] = 0.0
        touched_apy |= score == "impact_apy"
        touched_wav |= score == "impact_wav"
        logger.info("Global score override: %s/%s w_pos=%.2f (%scalibrated) → %d rows",
                    pos, score, w, "" if calibrated else "UN", int(mask.sum()))

    if touched_apy:
        snapshot["impact_score"] = snapshot["impact_apy_score"]
        snapshot = _clamp_impact_band(snapshot)
        for src, dst in [("impact_apy_score", "impact_apy_dollars"),
                         ("impact_apy_p10",   "impact_apy_dollars_p10"),
                         ("impact_apy_p90",   "impact_apy_dollars_p90")]:
            if src in snapshot.columns:
                snapshot[dst] = snapshot[src] * SALARY_CAP
    if (touched_apy or touched_wav) and "impact_wav_score" in snapshot.columns:
        apy_pct = snapshot.groupby("position_group")["impact_apy_score"].rank(pct=True)
        wav_pct = snapshot.groupby("position_group")["impact_wav_score"].rank(pct=True)
        both = apy_pct.notna() & wav_pct.notna()
        snapshot["impact_blend_pct"] = pd.to_numeric(np.where(
            both, (apy_pct + wav_pct) / 2.0, apy_pct.fillna(wav_pct)),
            errors="coerce")
    return snapshot


def _convert_value_scores_to_apy(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Translate value-percentile predictions into APY % of cap.

    Under config.IMPACT_TARGET_SOURCE == "value" the impact heads predict a
    within-position value percentile, not an APY share. Each percentile is read
    off its position's quantile function of best-APY-by-y5 (step 7's
    value_apy_map.parquet), so impact_apy_score and everything derived from it
    (impact_score, dollars, the p10/p90 band, replacement columns) keep their
    APY meaning. The map is monotone, so within-position order is unchanged;
    the percentiles are kept as value_pct_*.

    Known limitation: the heads predict a MEAN percentile, and reading a mean
    through a right-skewed APY curve understates the top of the board.
    """
    if IMPACT_TARGET_SOURCE != "value":
        return snapshot
    map_path = PROCESSED_DIR / "value_apy_map.parquet"
    if not map_path.exists():
        logger.error("IMPACT_TARGET_SOURCE='value' but %s is missing — run step 7; "
                     "impact scores left as percentiles", map_path)
        return snapshot
    amap = pd.read_parquet(map_path)
    pairs = [(src, dst) for src, dst in [("impact_apy_score", "value_pct_score"),
                                         ("impact_apy_p10",   "value_pct_p10"),
                                         ("impact_apy_p90",   "value_pct_p90")]
             if src in snapshot.columns]
    for src, dst in pairs:
        snapshot[dst] = snapshot[src]
        snapshot[src] = np.nan
    for group, m in amap.sort_values("q").groupby("position_group"):
        rows = snapshot["position_group"] == group
        for src, dst in pairs:
            pct = snapshot.loc[rows, dst].astype(float)
            snapshot.loc[rows, src] = np.where(
                pct.notna(), np.interp(pct.clip(0.0, 1.0), m["q"], m["apy_cap_pct"]), np.nan)
    unmapped = snapshot["value_pct_score"].notna() & snapshot["impact_apy_score"].isna()
    if unmapped.any():
        logger.warning("Value → APY: no map for %d scored rows (groups %s)", int(unmapped.sum()),
                       sorted(snapshot.loc[unmapped, "position_group"].dropna().unique()))
    snapshot["impact_score"] = snapshot["impact_apy_score"]
    snapshot = _clamp_impact_band(snapshot)
    for src, dst in [("impact_apy_score", "impact_apy_dollars"),
                     ("impact_apy_p10",   "impact_apy_dollars_p10"),
                     ("impact_apy_p90",   "impact_apy_dollars_p90")]:
        if src in snapshot.columns:
            snapshot[dst] = snapshot[src] * SALARY_CAP
    logger.info("Value → APY: mapped impact scores for %d rows across %d groups",
                int(snapshot["impact_apy_score"].notna().sum()), amap["position_group"].nunique())
    return snapshot


def _add_percentile_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentile rank columns for key stats within position group."""
    rank_cols = ["impact_score"]

    # Add stat-specific rank cols
    for pos_group, stats in MODELED_STATS.items():
        for stat in stats:
            col = f"kalman_mean_{stat}"
            if col in df.columns:
                rank_cols.append(col)

    rank_cols = [c for c in set(rank_cols) if c in df.columns]

    if "position_group" not in df.columns:
        return df

    for col in rank_cols:
        pct_col = f"pct_{col}"
        df[pct_col] = np.nan
        # If the column was post-hoc isotonic-calibrated, raw_<col> carries
        # the pre-calibration LightGBM output; using it as a tiebreaker keeps
        # prospects within an isotonic flat region ordered by the model's
        # underlying gradient instead of collapsing to identical percentiles.
        raw_col = f"raw_{col}"
        has_raw = raw_col in df.columns
        for pos_group, grp in df.groupby("position_group"):
            vals = grp[col]
            n = vals.notna().sum()
            if n == 0:
                continue
            if has_raw and grp[raw_col].notna().any():
                composite = vals.astype(float) + grp[raw_col].astype(float).fillna(0) * 1e-9
                ranks = composite.where(vals.notna()).rank(pct=True, na_option="keep")
            else:
                ranks = vals.rank(pct=True, na_option="keep")
            df.loc[grp.index, pct_col] = ranks

    return df


def _add_best_comp(df: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
    """Add best_comp_* columns. Prefers bridge comp (known NFL outcome); falls back to all-tier rank-1."""
    if comps.empty:
        df["best_comp_name"] = None
        df["best_comp_outcome"] = None
        return df

    comp_cols = ["player_id", "comp_player_name", "comp_school", "comp_role_target",
                 "comp_impact_target", "comp_career_w_av", "comp_first_nfl_season"]
    comp_cols = [c for c in comp_cols if c in comps.columns]

    # Prefer bridge comp (nearest player with a real NFL outcome)
    if "comp_tier" in comps.columns:
        best = comps[(comps["comp_tier"] == "bridge") & (comps["comp_rank"] == 1)][comp_cols].copy()
        if best.empty:
            best = comps[(comps["comp_tier"] == "all") & (comps["comp_rank"] == 1)][comp_cols].copy()
    else:
        best = comps[comps["comp_rank"] == 1][comp_cols].copy()

    best = best.drop_duplicates("player_id").rename(columns={
        "comp_player_name":      "best_comp_name",
        "comp_school":           "best_comp_school",
        "comp_role_target":      "best_comp_role",
        "comp_impact_target":    "best_comp_impact",
        "comp_career_w_av":      "best_comp_av",
        "comp_first_nfl_season": "best_comp_draft_year",
    })

    return df.merge(best, on="player_id", how="left")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Fast test run: cap to 500 players")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "projections").mkdir(exist_ok=True)

    comps_path = PROCESSED_DIR / "player_comps.parquet"

    # Shared assembly (attrs / age / sack-share / PFF / QB / OL-fill /
    # combine / hasan / league-averages). include_derived=False: step 10
    # computes derived features at inference time per position group inside
    # `_compute_role_impact_scores` (see config.DERIVED_AT_INFERENCE), so the
    # output parquet's schema is unchanged by the shared-helper refactor.
    snapshot = assemble_snapshot(include_derived=False, logger=logger)
    logger.info("Loaded snapshot: %d players", len(snapshot))

    if args.test:
        snapshot = snapshot.sample(min(500, len(snapshot)), random_state=42)
        logger.info("Test mode: %d players", len(snapshot))

    # Backfill player_name and school from player_seasons for players missing from attrs
    if "player_id" in snapshot.columns:
        missing_name = snapshot["player_name"].isna() if "player_name" in snapshot.columns \
                       else pd.Series(True, index=snapshot.index)
        if missing_name.any():
            seasons_path = RAW_DIR / "player_seasons.parquet"
            if seasons_path.exists():
                ps_names = pd.read_parquet(
                    seasons_path, columns=["player_id", "player_name", "school", "season_year"]
                )
                ps_latest = (ps_names.sort_values("season_year")
                             .groupby("player_id")[["player_name", "school"]]
                             .last().reset_index())
                snapshot = snapshot.merge(
                    ps_latest.rename(columns={"player_name": "_ps_name", "school": "_ps_school"}),
                    on="player_id", how="left",
                )
                if "player_name" not in snapshot.columns:
                    snapshot["player_name"] = None
                if "school" not in snapshot.columns:
                    snapshot["school"] = None
                snapshot["player_name"] = snapshot["player_name"].fillna(snapshot["_ps_name"])
                snapshot["school"] = snapshot["school"].fillna(snapshot["_ps_school"])
                snapshot = snapshot.drop(columns=["_ps_name", "_ps_school"], errors="ignore")
                filled = missing_name.sum() - snapshot["player_name"].isna().sum()
                logger.info("Backfilled player_name/school for %d players from player_seasons", filled)

    # Audit feature alignment between trained models and the assembled
    # snapshot before scoring — surfaces silent attr_cols drift loudly.
    _audit_model_feature_alignment(snapshot, PROCESSED_DIR / "draft_models")

    # Apply draft models
    logger.info("Computing draft scores...")
    snapshot = _compute_role_impact_scores(snapshot)

    logger.info("Computing per-prospect SHAP explanations (current cohort only)...")
    snapshot = _compute_shap_explanations(snapshot)

    # Merge global cross-position predictions (from Step 8b) as additional
    # rank columns. Lets the frontend show where global pooling agrees /
    # disagrees with per-position scoring without breaking existing fields.
    global_preds_path = PROCESSED_DIR / "global_draft_predictions.parquet"
    if global_preds_path.exists():
        gp = pd.read_parquet(global_preds_path)
        gp_cols = [c for c in gp.columns
                   if c == "player_id" or c.startswith("global_")]
        snapshot = snapshot.merge(gp[gp_cols].drop_duplicates("player_id"),
                                  on="player_id", how="left")
        n_g = snapshot.get("global_impact_apy_score",
                           pd.Series(np.nan, index=snapshot.index)).notna().sum()
        logger.info("Merged global cross-position predictions for %d players", int(n_g))
    else:
        logger.info("No global predictions found — run pipeline/08b_global_draft_model.py")

    # Apply config-gated global score overrides (K/P/TE per the 2026-07-21
    # blend A/B). Must run BEFORE percentile ranks / sorting so every derived
    # view reflects the blended scores.
    snapshot = _apply_global_score_overrides(snapshot)

    # Value-target mode: percentile predictions → APY % of cap. After the
    # overrides (they blend in percentile space), before anything that reads
    # impact_apy_score as an APY share.
    snapshot = _convert_value_scores_to_apy(snapshot)

    # Value-above-replacement views. Must run after the overrides (so it sees
    # the final calibrated impact_apy_score) and before percentile ranks.
    snapshot = _add_replacement_columns(snapshot)

    # Draft pool for the newest class, then the merged board (board curve +
    # capped off-board model, in y3-4 snap share), scored within that pool.
    snapshot = _add_draft_pool(snapshot)
    snapshot = _add_merged_board(snapshot)

    # Load comps
    comps = pd.read_parquet(comps_path) if comps_path.exists() else pd.DataFrame()

    # Add best comp
    snapshot = _add_best_comp(snapshot, comps)

    # Percentile ranks
    snapshot = _add_percentile_ranks(snapshot)

    # Add run date
    snapshot["projection_date"] = date.today().isoformat()

    # Sort by impact score descending
    snapshot = snapshot.sort_values("impact_score", ascending=False, na_position="last")

    # The role score is not an output (2026-10-31) and its heads are no longer
    # trained. Drop any role column that still arrives (e.g. comps'
    # best_comp_role, or a global predictions file from an older 8b run).
    role_cols = [c for c in snapshot.columns
                 if "role_score" in c or c in ("best_comp_role",)]
    snapshot = snapshot.drop(columns=role_cols)

    # Save dated + latest
    today_str = date.today().strftime("%Y%m%d")
    dated_path = OUTPUT_DIR / "projections" / f"projections_{today_str}.parquet"
    latest_path = OUTPUT_DIR / "projections" / "latest.parquet"

    snapshot.to_parquet(dated_path, index=False)
    snapshot.to_parquet(latest_path, index=False)
    logger.info("Saved %d player projections → %s", len(snapshot), dated_path)

    # Quick summary
    if "position_group" in snapshot.columns:
        for pos_group, grp in snapshot.groupby("position_group"):
            has_score = grp["impact_score"].notna().sum()
            logger.info("  %s: %d players, %d with scores", pos_group, len(grp), has_score)

    if "impact_score" in snapshot.columns:
        top10 = snapshot.nlargest(10, "impact_score")[
            ["player_name", "school", "position_group", "impact_score"]
        ] if "player_name" in snapshot.columns else snapshot.nlargest(10, "impact_score")
        logger.info("\nTop 10 by impact_score:\n%s", top10.to_string())

    logger.info("Step 10 complete.")


if __name__ == "__main__":
    main()
