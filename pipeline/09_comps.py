"""
Step 9: Draft Prospect Comps

For each player in the snapshot, finds the K most similar historical players
using importance-weighted nearest-neighbor search in impact model feature space.

Two-tier output per player:
  - comp_tier="all",    comp_rank 1–K: nearest K players from all prior cohorts
  - comp_tier="bridge", comp_rank 1:   nearest bridge player (known NFL outcome)

Backward-looking: comp pool for year Y = all snapshot players with
last_college_season_year < Y. No first_nfl_season cutoff on bridge pool.

Output: data/processed/player_comps.parquet
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))   # for sibling helpers
from config import (
    PROCESSED_DIR, POSITION_GROUPS, N_COMPS,
)
# Shared snapshot assembly (single source of truth across steps 8/8b/8c/9/10;
# merges player_attributes via config.ATTRIBUTE_MERGE_COLS internally).
from _snapshot_assembly import assemble_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_impact_model(pos_group: str) -> tuple[object, list[str]]:
    """Load the impact model bundle for a position group."""
    path = PROCESSED_DIR / "draft_models" / f"{pos_group}_impact_model.pkl"
    if not path.exists():
        return None, []
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"]


def _missing_aware_cdist(
    X_a: np.ndarray,
    X_b: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Vectorized weighted Euclidean distance over shared non-NaN dimensions only.

    For each pair (i, j), computes distance using only features where both
    X_a[i] and X_b[j] are non-NaN, then scales by sqrt(W_total / W_shared)
    so pairs with less feature overlap are penalized relative to well-covered
    pairs. Returns inf for pairs that share no valid dimensions.
    """
    W    = weights.astype(np.float64)
    W_sq = W ** 2

    valid_a = (~np.isnan(X_a)).astype(np.float64)
    valid_b = (~np.isnan(X_b)).astype(np.float64)

    A_w = np.where(valid_a.astype(bool), X_a * W, 0.0)
    B_w = np.where(valid_b.astype(bool), X_b * W, 0.0)

    sum_Wsq_shared = (valid_a * W_sq) @ valid_b.T
    sum_Wsq_total  = W_sq.sum()

    sq_dist = (
        (A_w ** 2) @ valid_b.T
        + valid_a @ (B_w ** 2).T
        - 2.0 * (A_w @ B_w.T)
    )
    sq_dist = np.maximum(sq_dist, 0.0)

    norm_sq = np.where(
        sum_Wsq_shared > 0,
        sq_dist * (sum_Wsq_total / np.maximum(sum_Wsq_shared, 1e-12)),
        np.inf,
    )
    return np.sqrt(norm_sq)


def find_comps_for_group(
        snapshot: pd.DataFrame,
        labels: pd.DataFrame,
        pos_group: str,
        k: int = N_COMPS,
        test_mode: bool = False,
) -> pd.DataFrame:
    """
    Find K nearest historical comps (all-tier) + best bridge comp for each
    player in a position group. Comp pool is backward-looking per prospect year.

    Vectorized by cohort year: for each unique year Y, distances from all
    year-Y players to all year-<Y players are computed in one numpy operation.
    """
    model, feature_cols = _load_impact_model(pos_group)
    if model is None or not feature_cols:
        logger.warning("No impact model for %s — skipping comps", pos_group)
        return pd.DataFrame()

    # Filter to position group
    if "position_group" in snapshot.columns:
        group_snap = snapshot[snapshot["position_group"] == pos_group].copy().reset_index(drop=True)
    else:
        group_snap = snapshot.copy().reset_index(drop=True)

    if group_snap.empty:
        return pd.DataFrame()

    # Use individual-player features only — exclude league averages (nfl_lg_*, lg_*)
    # which are identical for all players in the same draft era and distort comp matching.
    avail_features = [f for f in feature_cols
                      if f in group_snap.columns
                      and not f.startswith("nfl_lg_")
                      and not f.startswith("lg_")]
    if not avail_features:
        logger.warning("No comp features available for %s", pos_group)
        return pd.DataFrame()

    # Log-transform hasan_rank for distance. NFL draft value falls roughly
    # logarithmically with pick number, so a linear rank difference overstates
    # similarity at the top (rank 5 vs 50 is a huge tier change) and understates
    # it at the bottom (rank 240 vs 290 are both UDFA territory). Applied only
    # here — the impact model was trained on raw hasan_rank, so its
    # feature_importances_ still reflect raw-scale predictive power.
    if "hasan_rank" in group_snap.columns:
        hr = group_snap["hasan_rank"].astype(float).clip(lower=1.0)
        group_snap["hasan_rank"] = np.log(hr)

    # NaN-aware scaling: NaN cells stay NaN through the scale step
    X_raw      = group_snap[avail_features].values.astype(float)
    feat_means = np.nanmean(X_raw, axis=0)
    feat_stds  = np.nanstd(X_raw, axis=0)
    feat_stds  = np.where(feat_stds == 0, 1.0, feat_stds)
    X_scaled   = (X_raw - feat_means) / feat_stds

    # Importance weights: sqrt(normalized importance) from impact model
    importances = model.feature_importances_
    feat_to_imp = dict(zip(feature_cols, importances))
    weights = np.array([feat_to_imp.get(f, 0.0) for f in avail_features])
    total = weights.sum()
    weights = np.sqrt(weights / total) if total > 0 else np.ones(len(avail_features))

    # Per-player metadata
    if "last_college_season_year" in group_snap.columns:
        years = group_snap["last_college_season_year"].fillna(-1).astype(float).values
    else:
        years = np.full(len(group_snap), -1.0)

    player_ids = group_snap["player_id"].values
    has_name = "player_name" in group_snap.columns
    has_school = "school" in group_snap.columns

    # Bridge player set (no first_nfl_season cutoff)
    bridge_ids = set(labels[labels["is_bridge"] == True]["player_id"].dropna())
    bridge_mask_all = np.isin(player_ids, list(bridge_ids))

    # Label lookup (vectorized via dict)
    label_cols = ["player_id", "role_target", "impact_target", "impact_apy_target",
                  "value_target", "career_w_av", "first_nfl_season"]
    label_cols = [c for c in label_cols if c in labels.columns]
    lb_df = labels[label_cols].drop_duplicates("player_id").set_index("player_id")
    # comp_impact_target is shown as an APY share. impact_target follows
    # config.IMPACT_TARGET_SOURCE, so read the explicit APY column when present.
    apy_col = "impact_apy_target" if "impact_apy_target" in lb_df.columns else "impact_target"

    def _get_label(pid, col):
        if pid in lb_df.index and col in lb_df.columns:
            v = lb_df.at[pid, col]
            return float(v) if pd.notna(v) else np.nan
        return np.nan

    # Sort group_snap index by year so pool is always a prefix
    sort_idx = np.argsort(years, kind="stable")
    years_sorted = years[sort_idx]
    X_sorted = X_scaled[sort_idx]
    pids_sorted = player_ids[sort_idx]
    bridge_sorted = bridge_mask_all[sort_idx]
    if has_name:
        names_sorted = group_snap["player_name"].values[sort_idx]
    if has_school:
        schools_sorted = group_snap["school"].values[sort_idx]

    unique_years = sorted(set(years_sorted[years_sorted > 0]))

    if test_mode:
        # Only process the two most recent cohort years
        unique_years = unique_years[-2:] if len(unique_years) >= 2 else unique_years

    records = []

    for y in unique_years:
        # Prospects = players in year Y
        prospect_mask = years_sorted == y
        # Comp pool = players from strictly earlier years
        pool_mask = years_sorted < y

        if not pool_mask.any():
            continue

        X_prospects = X_sorted[prospect_mask]    # (n_p, n_f)
        X_pool = X_sorted[pool_mask]             # (n_c, n_f)
        pids_prospects = pids_sorted[prospect_mask]
        pids_pool = pids_sorted[pool_mask]
        bridge_pool = bridge_sorted[pool_mask]

        # Precompute pool metadata once per cohort year
        names_pool = names_sorted[pool_mask] if has_name else None
        schools_pool = schools_sorted[pool_mask] if has_school else None

        # Missing-value-aware distance: shared dims only, normalized by shared weight
        dists = _missing_aware_cdist(X_prospects, X_pool, weights)  # (n_p, n_c)

        # All-tier: top-K per prospect
        k_actual = min(k, dists.shape[1])
        # Guard: argpartition fails on all-inf rows (no shared dims) — handled by
        # finite check in the inner loop via np.isfinite(row_dists[j])
        top_k_idx = np.argpartition(dists, k_actual - 1, axis=1)[:, :k_actual]

        for pi, pid in enumerate(pids_prospects):
            row_dists = dists[pi]
            top_local = top_k_idx[pi]
            top_local_sorted = top_local[np.argsort(row_dists[top_local])]

            for rank, j in enumerate(top_local_sorted):
                if not np.isfinite(row_dists[j]):
                    break
                cpid = pids_pool[j]
                records.append({
                    "player_id":             pid,
                    "comp_tier":             "all",
                    "comp_rank":             rank + 1,
                    "comp_player_id":        cpid,
                    "comp_player_name":      names_pool[j] if names_pool is not None else "",
                    "comp_school":           schools_pool[j] if schools_pool is not None else "",
                    "comp_distance":         float(row_dists[j]),
                    "comp_role_target":      _get_label(cpid, "role_target"),
                    "comp_impact_target":    _get_label(cpid, apy_col),
                    "comp_value_target":     _get_label(cpid, "value_target"),
                    "comp_career_w_av":      _get_label(cpid, "career_w_av"),
                    "comp_first_nfl_season": _get_label(cpid, "first_nfl_season"),
                })

            # Bridge-tier: nearest bridge player
            if bridge_pool.any():
                bridge_dists = row_dists.copy()
                bridge_dists[~bridge_pool] = np.inf
                best_j = int(np.argmin(bridge_dists))
                if np.isfinite(bridge_dists[best_j]):
                    cpid = pids_pool[best_j]
                    records.append({
                        "player_id":             pid,
                        "comp_tier":             "bridge",
                        "comp_rank":             1,
                        "comp_player_id":        cpid,
                        "comp_player_name":      names_pool[best_j] if names_pool is not None else "",
                        "comp_school":           schools_pool[best_j] if schools_pool is not None else "",
                        "comp_distance":         float(bridge_dists[best_j]),
                        "comp_role_target":      _get_label(cpid, "role_target"),
                        "comp_impact_target":    _get_label(cpid, apy_col),
                        "comp_value_target":     _get_label(cpid, "value_target"),
                        "comp_career_w_av":      _get_label(cpid, "career_w_av"),
                        "comp_first_nfl_season": _get_label(cpid, "first_nfl_season"),
                    })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Fast test run: only process most recent 2 cohorts per group")
    args = parser.parse_args()

    labels_path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    if not labels_path.exists():
        logger.error("%s not found — run earlier steps", labels_path)
        sys.exit(1)
    labels = pd.read_parquet(labels_path)

    # Shared assembly — comps weight by impact-model feature importance, so
    # the snapshot must carry the SAME feature surface the models trained on
    # (attrs, age_at_draft, sack share, PFF/QB career, combine, hasan with
    # year-sanity nulling, league averages, per-position derived features).
    snapshot = assemble_snapshot(include_derived=True, logger=logger)

    # Consumer-specific: sentinel fill hasan_rank = 301 (just outside the
    # top-300 boundary), preserving ordinal signal. This gives hasan_rank
    # 100% coverage so it enters distance computation with its full weight.
    if "hasan_rank" in snapshot.columns:
        snapshot["hasan_rank"] = snapshot["hasan_rank"].fillna(301)

    all_comps = []
    for pos_group in POSITION_GROUPS.keys():
        logger.info("Computing comps for: %s", pos_group)
        comps = find_comps_for_group(snapshot, labels, pos_group,
                                      k=N_COMPS, test_mode=args.test)
        if not comps.empty:
            comps["position_group"] = pos_group
            all_comps.append(comps)
            n_prospects = comps["player_id"].nunique()
            n_bridge = (comps["comp_tier"] == "bridge").sum()
            logger.info("  %s: comps for %d players (%d bridge comp rows)",
                        pos_group, n_prospects, n_bridge)

    if not all_comps:
        logger.warning("No comps generated — check that impact model pickles exist")
        pd.DataFrame().to_parquet(PROCESSED_DIR / "player_comps.parquet", index=False)
        return

    comps_df = pd.concat(all_comps, ignore_index=True)
    out_path = PROCESSED_DIR / "player_comps.parquet"
    comps_df.to_parquet(out_path, index=False)
    logger.info("Saved %d comp rows for %d players → %s",
                len(comps_df), comps_df["player_id"].nunique(), out_path)
    logger.info("Step 9 complete.")


if __name__ == "__main__":
    main()
