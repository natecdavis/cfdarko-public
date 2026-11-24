"""
Shared snapshot assembly for the draft-model steps (8 / 8b / 8c / 9 / 10).

Historically each of those five steps re-implemented the same chain of
merges onto player_skill_snapshot.parquet, and the copies drifted (filters
applied in some but not others — the repo's documented root cause of
consistency bugs). This module is now the single source of truth; the
canonical behavior is the one step 8's main() had (the most complete copy).

Assembly chain (in order):
  1. player_skill_snapshot.parquet          (base matrix, one row per player)
  2. player_attributes.parquet              via ATTRIBUTE_MERGE_COLS with
                                            overlap handling; snapshot's own
                                            position_group is preserved and
                                            only null-filled from attrs
  3. age_at_draft                           fractional (May 1) from birth_date,
                                            else integer from birth_year
  4. team_sack_share                        player's career-max share of team
                                            sacks, from raw player_seasons
  5. PFF career features                    load_pff_career_features
  6. Teammate-QB career features            load_qb_career_features, gated on
                                            USE_QB_CAREER_FEATURES
  7. OL sub-position fill + normalization   pff_position_mode fallback for
                                            generic "OL", then T→OT G→OG IOL→OG
  8. combine_metrics.parquet                + combine_invited zero-fill
  9. hasan_ranks.parquet                    with year-sanity nulling (player_id
                                            collisions) and draft_year renamed
                                            to hasan_draft_year
 10. nfl_league_averages.parquet            joined on last_college_season_year
 11. per-position derived features          add_derived_features per group
                                            (skipped when include_derived=False
                                            — step 10 computes these at
                                            inference time on a local copy; see
                                            config.DERIVED_AT_INFERENCE)

Import pattern: every pipeline/ script inserts its own directory on sys.path
(and scripts run directly get it as sys.path[0]), so a plain
`from _snapshot_assembly import assemble_snapshot` resolves everywhere.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ATTRIBUTE_MERGE_COLS, NFL_LG_FEATURES, POSITION_GROUPS,
    PROCESSED_DIR, RAW_DIR, SPEED_SCORE_POSITIONS, USE_QB_CAREER_FEATURES,
)
from src.scrapers.pff import load_pff_career_features
from src.utils.qb_career import load_qb_career_features

_MODULE_LOGGER = logging.getLogger(__name__)


def add_derived_features(df: pd.DataFrame, pos_group: str) -> pd.DataFrame:
    """
    Compute position-specific derived features on a copy of `df`.
    Called before both training (Step 8) and scoring (Step 10).
    (Moved verbatim from pipeline/08_draft_models.py:_add_derived_features.)
    """
    df = df.copy()

    # Pass rush composite and team sack share for DL, EDGE and LB
    if pos_group in ("DL", "EDGE", "LB"):
        sacks_col  = "kalman_mean_sacks"
        tfl_col    = "kalman_mean_tfl"
        hurries_col = "kalman_mean_qb_hurries"
        s = df.get(sacks_col,   pd.Series(0, index=df.index)).fillna(0)
        t = df.get(tfl_col,     pd.Series(0, index=df.index)).fillna(0)
        h = df.get(hurries_col, pd.Series(0, index=df.index)).fillna(0)
        df["pass_rush_composite"] = s * 2.0 + t * 1.0 + h * 0.5

        if "team_sack_share" in df.columns:
            pass   # already computed in assemble_snapshot()
        else:
            df["team_sack_share"] = np.nan

    # OL sub-position dummies (OT/T=tackle, OG/IOL/G=guard, C=center)
    if pos_group == "OL" and "position" in df.columns:
        df["is_ol_tackle"] = df["position"].isin(["OT", "T"]).astype(float)
        df["is_ol_guard"]  = df["position"].isin(["OG", "IOL", "G"]).astype(float)
        df["is_ol_center"] = (df["position"] == "C").astype(float)

    # DL/EDGE/LB snap-role composition: fraction of defense snaps rushing / covering
    if pos_group in ("DL", "EDGE", "LB"):
        def_snaps = df.get("pff_snap_counts_defense_mean",
                           pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        pr_snaps  = df.get("pff_snap_counts_pass_rush_mean",
                           pd.Series(np.nan, index=df.index))
        df["pff_pass_rush_snap_pct"] = (pr_snaps / def_snaps).clip(0, 1)
        if pos_group == "LB":
            cov_snaps = df.get("pff_snap_counts_coverage_mean",
                               pd.Series(np.nan, index=df.index))
            df["pff_coverage_snap_pct"] = (cov_snaps / def_snaps).clip(0, 1)

    # Weight-adjusted speed ("speed score" = wt × 200 / forty⁴). A quartic
    # ratio of two raw combine columns is not something a tree can synthesize
    # from the components at these per-head sample sizes (~250-600 rows with a
    # non-zero label), which is why this is built explicitly rather than left
    # to the model. NaN whenever either component is missing — LightGBM routes
    # that via its missing-direction split.
    if pos_group in SPEED_SCORE_POSITIONS:
        wt    = df.get("combine_wt",    pd.Series(np.nan, index=df.index))
        forty = df.get("combine_forty", pd.Series(np.nan, index=df.index))
        forty = pd.to_numeric(forty, errors="coerce").replace(0, np.nan)
        wt    = pd.to_numeric(wt,    errors="coerce")
        df["combine_speed_score"] = wt * 200.0 / (forty ** 4)

    # DB alignment composition: fraction of defense snaps at corner / slot / free safety
    if pos_group == "DB":
        def_snaps = df.get("pff_snap_counts_defense_mean",
                           pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        for snap_col, pct_col in [
            ("pff_snap_counts_corner_mean", "pff_corner_snap_pct"),
            ("pff_snap_counts_slot_mean",   "pff_slot_snap_pct"),
            ("pff_snap_counts_fs_mean",     "pff_fs_snap_pct"),
        ]:
            df[pct_col] = (df.get(snap_col, pd.Series(np.nan, index=df.index))
                           / def_snaps).clip(0, 1)

    return df


def attach_derived_features(snapshot: pd.DataFrame,
                            logger: logging.Logger | None = None) -> pd.DataFrame:
    """Apply add_derived_features per position group over the full snapshot.

    Assigns per-group unconditionally: groups share derived column names
    (pass_rush_composite exists for DL AND LB), so guarding on "column
    already in snapshot" would leave every group after the first as NaN.
    """
    snapshot = snapshot.copy()
    base_cols = set(snapshot.columns)
    for pos_group in POSITION_GROUPS.keys():
        mask = snapshot.get("position_group",
                            pd.Series("", index=snapshot.index)) == pos_group
        if mask.any():
            derived = add_derived_features(snapshot[mask], pos_group)
            for col in [c for c in derived.columns if c not in base_cols]:
                if col not in snapshot.columns:
                    snapshot[col] = np.nan
                snapshot.loc[mask, col] = derived[col].values
    return snapshot


def assemble_snapshot(*, include_derived: bool = True,
                      logger: logging.Logger | None = None) -> pd.DataFrame:
    """Build the per-player scoring matrix exactly as step 8's training path
    does. Every consumer (8 / 8b / 8c / 9 / 10) calls this and then applies
    its own consumer-specific post-processing locally.

    include_derived: when False, skip the per-position derived-feature pass
    (step 10 computes those at inference time per group; see
    config.DERIVED_AT_INFERENCE).
    """
    logger = logger or _MODULE_LOGGER

    snap_path = PROCESSED_DIR / "player_skill_snapshot.parquet"
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    if not snap_path.exists():
        logger.error("%s not found — run earlier steps", snap_path)
        sys.exit(1)

    snapshot = pd.read_parquet(snap_path)

    # Join physical attributes. Schema lives in config.ATTRIBUTE_MERGE_COLS
    # (shared with Steps 3, 8, 8b, 9, 10). Snapshot already carries some of
    # the same columns from Step 5/6 (e.g. position_group with a three-tier
    # fallback that's more reliable than attrs's roster-based mapping), so
    # we drop overlapping cols from snapshot before merging — except
    # position_group, which we preserve from snapshot and only fill from
    # attrs where the snapshot value is null.
    if attrs_path.exists():
        attrs = pd.read_parquet(attrs_path)
        if "player_id" in snapshot.columns:
            attr_cols = [c for c in ATTRIBUTE_MERGE_COLS if c in attrs.columns]
            overlap = [c for c in attr_cols
                       if c not in ("player_id", "position_group")
                       and c in snapshot.columns]
            snapshot = snapshot.drop(columns=overlap, errors="ignore")
            snapshot = snapshot.merge(
                attrs[attr_cols].drop_duplicates("player_id"),
                on="player_id", how="left",
                suffixes=("", "_attrs"),
            )
            if "position_group_attrs" in snapshot.columns:
                # Prefer the attrs group over both NaN and a snapshot "OTHER":
                # bridge-only players (synthetic ids, no gamelog/roster stats)
                # and roster skeletons with an unmapped raw position land in
                # OTHER here, but step 03 resolved their real group from the
                # draft-position backfill. Without this, step 8 filters them
                # out of every per-position head. Only fills FROM a real attrs
                # group (never overwrites a real snapshot group with OTHER —
                # verified 0 such cases), so it strictly adds information.
                snap_pg = snapshot["position_group"]
                attr_pg = snapshot["position_group_attrs"]
                use_attrs = (snap_pg.isna() | (snap_pg == "OTHER")) & \
                            attr_pg.notna() & (attr_pg != "OTHER")
                snapshot["position_group"] = snap_pg.where(~use_attrs, attr_pg)
                snapshot = snapshot.drop(columns=["position_group_attrs"])

    # age_at_draft: fractional (May 1) if full birth_date known, else integer float
    if "last_college_season_year" in snapshot.columns:
        draft_yr = snapshot["last_college_season_year"] + 1
        ages = pd.Series(np.nan, index=snapshot.index)
        if "birth_date" in snapshot.columns:
            has_date = snapshot["birth_date"].notna()
            if has_date.any():
                bd = pd.to_datetime(snapshot.loc[has_date, "birth_date"], errors="coerce")
                may1 = pd.to_datetime(
                    draft_yr[has_date].astype(int).astype(str) + "-05-01"
                )
                ages[has_date] = (may1 - bd).dt.days / 365.25
        if "birth_year" in snapshot.columns:
            has_year = snapshot["birth_year"].notna() & ages.isna()
            ages[has_year] = (draft_yr[has_year] - snapshot.loc[has_year, "birth_year"]).astype(float)
        snapshot["age_at_draft"] = ages
        n_age = ages.notna().sum()
        logger.info("age_at_draft computed for %d / %d players", n_age, len(snapshot))

    # Compute team sack share from player_seasons (DL/LB)
    seasons_path = RAW_DIR / "player_seasons.parquet"
    if seasons_path.exists() and "player_id" in snapshot.columns:
        player_seasons = pd.read_parquet(
            seasons_path, columns=["player_id", "school", "season_year", "sacks"]
        )
        player_seasons["sacks"] = pd.to_numeric(player_seasons["sacks"], errors="coerce").fillna(0)
        team_sacks = (player_seasons.groupby(["school", "season_year"])["sacks"]
                      .sum().reset_index()
                      .rename(columns={"sacks": "team_sacks"}))
        player_seasons = player_seasons.merge(team_sacks, on=["school", "season_year"], how="left")
        player_seasons["sack_share"] = np.where(
            player_seasons["team_sacks"] > 0,
            player_seasons["sacks"] / player_seasons["team_sacks"],
            0.0,
        )
        max_sack_share = (player_seasons.groupby("player_id")["sack_share"]
                          .max().reset_index()
                          .rename(columns={"sack_share": "team_sack_share"}))
        snapshot = snapshot.merge(max_sack_share, on="player_id", how="left")
        n_filled = snapshot["team_sack_share"].notna().sum()
        logger.info("Team sack share computed for %d players", n_filled)

    # Merge PFF career features (static triplets: mean/peak/last per grade)
    pff_career = load_pff_career_features(snapshot, PROCESSED_DIR)
    if not pff_career.empty and "player_id" in pff_career.columns:
        snapshot = snapshot.merge(pff_career, on="player_id", how="left")
        n_pff = snapshot[[c for c in snapshot.columns if c.startswith("pff_")]].any(axis=1).sum()
        logger.info("PFF career features merged: %d players with PFF data", n_pff)
    else:
        logger.info("No PFF career features available — run pipeline/01c_pff_ingestion.py")

    # Merge teammate-QB career triplets (mean/peak/last per qb_rolling_* col).
    # Activated only for WR/TE/RB in step 8's _build_feature_cols; the merge
    # happens unconditionally so the snapshot carries the columns for all
    # downstream consumers regardless of position.
    if USE_QB_CAREER_FEATURES:
        qb_career = load_qb_career_features(snapshot, PROCESSED_DIR, RAW_DIR)
        if not qb_career.empty and "player_id" in qb_career.columns:
            snapshot = snapshot.merge(qb_career, on="player_id", how="left")
            n_qb_cols = sum(c.startswith("qb_rolling_") for c in qb_career.columns)
            n_players = qb_career["qb_rolling_passer_rating_mean"].notna().sum() \
                if "qb_rolling_passer_rating_mean" in qb_career.columns else 0
            logger.info("Teammate-QB career features merged: %d cols, "
                        "%d players with non-null _mean",
                        n_qb_cols, n_players)
        else:
            logger.info("No teammate-QB career features available — "
                        "rerun pipeline/01_data_ingestion.py")

    # Use PFF position code as fallback for OL players with generic CFBD "OL"
    _PFF_TO_CFBD_POS = {"T": "OT", "G": "OG", "C": "C"}
    if "pff_position_mode" in snapshot.columns and "position" in snapshot.columns:
        ol_generic = (
            (snapshot["position_group"] == "OL") &
            (snapshot["position"].isin(["OL"]) | snapshot["position"].isna())
        )
        if ol_generic.any():
            mapped = snapshot.loc[ol_generic, "pff_position_mode"].map(_PFF_TO_CFBD_POS)
            snapshot.loc[ol_generic & mapped.notna(), "position"] = mapped[mapped.notna()]
            logger.info("Filled OL sub-position from PFF for %d players", mapped.notna().sum())

    # Normalize OL position display: T→OT, G→OG, IOL→OG (CFBD sometimes uses bare codes)
    if "position" in snapshot.columns:
        ol_mask = snapshot["position_group"] == "OL"
        snapshot.loc[ol_mask, "position"] = (
            snapshot.loc[ol_mask, "position"].replace({"T": "OT", "G": "OG", "IOL": "OG"})
        )

    # Merge combine metrics
    combine_path = PROCESSED_DIR / "combine_metrics.parquet"
    if combine_path.exists():
        combine_metrics = pd.read_parquet(combine_path)
        snapshot = snapshot.merge(combine_metrics, on="player_id", how="left")
        snapshot["combine_invited"] = snapshot["combine_invited"].fillna(0).astype(int)
        n_combine = (snapshot["combine_invited"] == 1).sum()
        logger.info("Combine metrics merged: %d players with combine data", n_combine)
    else:
        logger.info("No combine metrics found — run pipeline/00d_combine_ingestion.py")

    # Merge Hasan Consensus Big Board ranks
    hasan_path = PROCESSED_DIR / "hasan_ranks.parquet"
    if hasan_path.exists():
        hasan_ranks = pd.read_parquet(hasan_path)
        snapshot = snapshot.merge(
            hasan_ranks[["player_id", "draft_year", "hasan_rank"]],
            on="player_id", how="left",
        )
        # Guard against player_id collisions in CFBD: null out ranks where the
        # board's draft_year doesn't match the player's expected draft year
        # (last_college_season_year + 1), with ±1 tolerance for redshirts/returns.
        if "last_college_season_year" in snapshot.columns:
            expected = snapshot["last_college_season_year"] + 1
            year_ok = (snapshot["draft_year"].isna() |
                       ((snapshot["draft_year"] - expected).abs() <= 1))
            n_nulled = (~year_ok & snapshot["hasan_rank"].notna()).sum()
            if n_nulled > 0:
                logger.warning("Nulled %d errant Hasan rank joins (player_id collision, "
                               "year mismatch > 1)", n_nulled)
            snapshot.loc[~year_ok, "hasan_rank"] = np.nan
        snapshot = snapshot.rename(columns={"draft_year": "hasan_draft_year"})
        n_hasan = snapshot["hasan_rank"].notna().sum()
        logger.info("Hasan ranks merged: %d players with board rank", n_hasan)
    else:
        logger.info("No Hasan ranks found — run pipeline/00e_hasan_ingestion.py")

    # Merge NFL era features (league-wide style-of-play by season)
    nfl_lg_path = PROCESSED_DIR / "nfl_league_averages.parquet"
    if nfl_lg_path.exists() and "last_college_season_year" in snapshot.columns:
        nfl_lg = pd.read_parquet(nfl_lg_path,
                                 columns=["season_year"] + NFL_LG_FEATURES)
        snapshot = snapshot.merge(
            nfl_lg.rename(columns={"season_year": "last_college_season_year"}),
            on="last_college_season_year", how="left",
        )
        n_matched = snapshot[NFL_LG_FEATURES[0]].notna().sum()
        logger.info("Joined NFL league averages: %d/%d rows matched", n_matched, len(snapshot))
    else:
        logger.warning("nfl_league_averages.parquet not found — run pipeline/00c_nfl_league_averages.py")

    # Add all derived features to snapshot before training/scoring.
    if include_derived:
        snapshot = attach_derived_features(snapshot, logger=logger)

    return snapshot
