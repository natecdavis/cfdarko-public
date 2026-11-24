"""
Step 3: Player Attributes

Assembles a player-level attributes table with:
  - Position (from CFBD roster) and position group
  - Height (inches), weight (lbs)
  - Class year (FR=1, SO=2, JR=3, SR=4, GR=5)
  - Birth year and age at last college season (from nfl_data_py, bridge players only)

Rosters are cached to data/raw/rosters.parquet after the first API fetch.
Use --refresh-rosters to force a fresh fetch.

Output: data/processed/player_attributes.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    START_YEAR, END_YEAR, RAW_DIR, PROCESSED_DIR,
    POSITION_TO_GROUP, CONFERENCE_TIERS, DEFAULT_CONFERENCE_TIER,
    KNOWN_SEASON_EXCLUSIONS, BIRTH_YEAR_OVERRIDES,
    ATTRIBUTE_MERGE_COLS, MIN_BRIDGE_DERIVED_AGE, MAX_BRIDGE_DERIVED_AGE,
    ALTITUDE_DIFF_BUCKET_FT,
)
from src.scrapers.cfbd import fetch_rosters, fetch_recruiting_data
from src.scrapers.nfl_data import load_birth_dates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--refresh-rosters", action="store_true",
                        help="Re-fetch rosters from CFBD API even if cache exists")
    parser.add_argument("--refresh-recruiting", action="store_true",
                        help="Re-fetch recruiting data from CFBD API even if cache exists")
    return parser.parse_args()


def _height_to_inches(height_str) -> float:
    """Convert PFR-style height string (e.g. '6-3' or '75') to total inches."""
    if pd.isna(height_str):
        return np.nan
    s = str(height_str).strip()
    if "-" in s:
        parts = s.split("-")
        try:
            return int(parts[0]) * 12 + int(parts[1])
        except ValueError:
            return np.nan
    try:
        v = float(s)
        return v if v < 100 else np.nan
    except ValueError:
        return np.nan


def _class_year_to_int(year_val) -> float:
    """Convert class year string to integer (1=FR, 2=SO, 3=JR, 4=SR, 5=GR)."""
    if pd.isna(year_val):
        return np.nan
    mapping = {"fr": 1, "freshman": 1, "so": 2, "sophomore": 2,
               "jr": 3, "junior": 3, "sr": 4, "senior": 4,
               "gr": 5, "graduate": 5, "5th": 5}
    s = str(year_val).lower().strip()
    if s in mapping:
        return float(mapping[s])
    try:
        return float(s)
    except ValueError:
        return np.nan


# Roster / draft position strings that identify an edge rusher (used only as
# the pre-2014 fallback for base-DL players with no PFF coverage). Interior
# DL is labeled "DT"/"NT" and the generic "DL" is left alone.
_EDGE_ROSTER_CODES: frozenset[str] = frozenset({
    "DE", "EDGE", "defensive end", "defensive edge", "edge rusher", "edge",
})


def _reconcile_edge(attrs: pd.DataFrame) -> pd.DataFrame:
    """Reclassify edge rushers from DL/LB into the dedicated EDGE group.

    Rule (priority order), applied only to base DL / LB players:
      1. PFF position code == "ED"  -> EDGE       (authoritative, 2014+)
      2. no PFF, base == DL, roster/draft position in the DE family -> EDGE
    Off-ball LBs with no PFF are left in LB (roster "OLB" is genuinely
    ambiguous). Returns attrs with position_group updated in place.
    """
    if "position_group" not in attrs.columns or "player_id" not in attrs.columns:
        return attrs

    # Per-player PFF position mode (the code the player is listed at most often).
    pff_mode = pd.Series(dtype=object)
    pff_path = PROCESSED_DIR / "pff_season_grades.parquet"
    if pff_path.exists():
        pff = pd.read_parquet(pff_path, columns=["player_id", "pff_position"])
        pff = pff.dropna(subset=["player_id", "pff_position"])
        if len(pff):
            pff_mode = (pff.groupby(pff["player_id"].astype(str))["pff_position"]
                        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan))

    pid = attrs["player_id"].astype(str)
    mode = pid.map(pff_mode)
    base = attrs["position_group"]
    pos_lower = attrs["position"].astype(str).str.strip().str.lower()

    is_front = base.isin(["DL", "LB"])
    pff_edge = is_front & (mode == "ED")
    roster_edge = (base == "DL") & mode.isna() & pos_lower.isin(_EDGE_ROSTER_CODES)
    to_edge = pff_edge | roster_edge

    n_from_pff = int(pff_edge.sum())
    n_from_roster = int((roster_edge & ~pff_edge).sum())
    attrs.loc[to_edge, "position_group"] = "EDGE"
    logger.info("EDGE reconciliation: %d players -> EDGE (%d via PFF ED, "
                "%d via pre-2014 roster-DE fallback)",
                int(to_edge.sum()), n_from_pff, n_from_roster)
    return attrs


def main():
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    start = END_YEAR - 2 if args.test else START_YEAR
    end = END_YEAR

    # CFBD player_id reuse exclusions (manual KNOWN_SEASON_EXCLUSIONS ∪
    # auto-detected via combine-year gap). Computed once; applied at each
    # rosters / player_seasons read site below so the (pid, year) rows
    # belonging to a namesake don't bleed into attrs.
    from src.utils.cfbd_id_reuse import merged_exclusions as _merged_exclusions, exclusion_mask
    from src.utils.fcs_identities import load_fcs_rosters
    _exclusions = _merged_exclusions(KNOWN_SEASON_EXCLUSIONS, RAW_DIR, PROCESSED_DIR, logger)

    # ── Roster data (cached) ────────────────────────────────────────────────────
    rosters_cache = RAW_DIR / "rosters.parquet"
    if rosters_cache.exists() and not args.refresh_rosters:
        logger.info("Loading rosters from cache → %s", rosters_cache)
        rosters = pd.read_parquet(rosters_cache)
        # If test mode, filter to recent years after loading from cache
        if args.test and "season_year" in rosters.columns:
            rosters = rosters[rosters["season_year"] >= start]
    else:
        logger.info("Fetching rosters %d–%d from CFBD API ...", start, end)
        rosters = fetch_rosters(start, end)
        if rosters.empty:
            logger.error("No roster data fetched — check CFBD_API_KEY")
            sys.exit(1)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        rosters.to_parquet(rosters_cache, index=False)
        logger.info("Saved %d roster rows → %s", len(rosters), rosters_cache)

    if rosters.empty:
        logger.error("No roster data available")
        sys.exit(1)

    # FCS identities (step 01c): CFBD has no FCS rosters, so PFF-anchored
    # FCS players join as roster rows with no height / weight / class year.
    # Before the exclusions: a linked CFBD player's FCS seasons can be
    # post-draft ones.
    fcs_rosters = load_fcs_rosters()
    if len(fcs_rosters) and not args.test:
        rosters = pd.concat([rosters, fcs_rosters.reindex(columns=rosters.columns)],
                            ignore_index=True)
        logger.info("Added %d FCS identity roster rows (%d players)",
                    len(fcs_rosters), fcs_rosters["player_id"].nunique())

    # Apply CFBD-id-reuse exclusions to rosters BEFORE any downstream use.
    # The same registry is applied to player_seasons / gamelogs elsewhere,
    # but if rosters skips the filter, the (player_id, season_year) row
    # gets re-added when `player_seasons` is padded with roster-only pairs
    # (~L182), silently undoing the exclusion.
    if (_exclusions and "player_id" in rosters.columns
            and "season_year" in rosters.columns):
        bad_mask = exclusion_mask(rosters, _exclusions)
        n_dropped = int(bad_mask.sum())
        if n_dropped:
            rosters = rosters[~bad_mask].reset_index(drop=True)
            logger.info("Dropped %d roster rows matching CFBD id-reuse exclusions",
                        n_dropped)

    # ── Clean physical attributes ───────────────────────────────────────────────
    if "height_in" not in rosters.columns:
        if "height" in rosters.columns:
            rosters["height_in"] = rosters["height"].apply(_height_to_inches)
        else:
            rosters["height_in"] = np.nan
    if "weight_lbs" not in rosters.columns:
        if "weight" in rosters.columns:
            rosters["weight_lbs"] = pd.to_numeric(rosters["weight"], errors="coerce")
        else:
            rosters["weight_lbs"] = np.nan
    if "class_year_int" not in rosters.columns:
        if "year" in rosters.columns:
            rosters["class_year_int"] = rosters["year"].apply(_class_year_to_int)
        else:
            rosters["class_year_int"] = np.nan
    rosters["class_year_int"] = pd.to_numeric(rosters["class_year_int"], errors="coerce")
    rosters["class_year_int"] = rosters["class_year_int"].where(
        rosters["class_year_int"].between(1, 5), other=np.nan
    )

    # ── Position mapping ────────────────────────────────────────────────────────
    rosters["position"] = rosters["position"].str.upper().str.strip()
    rosters["position_group"] = rosters["position"].map(POSITION_TO_GROUP)

    # Log unmapped positions so gaps are visible
    unmapped = rosters[rosters["position_group"].isna() & rosters["position"].notna()]
    if not unmapped.empty:
        unmapped_counts = unmapped["position"].value_counts()
        logger.info("Unmapped positions (→ OTHER): %s",
                    unmapped_counts.to_dict())
    rosters["position_group"] = rosters["position_group"].fillna("OTHER")

    # ── Last season per player ──────────────────────────────────────────────────
    seasons_path = RAW_DIR / "player_seasons.parquet"
    if seasons_path.exists():
        player_seasons = pd.read_parquet(seasons_path,
                                          columns=["player_id", "player_name",
                                                   "school", "season_year"])
        # Drop known bad (player_id, season_year) pairs before deriving last season / school
        if _exclusions:
            bad_mask = exclusion_mask(player_seasons, _exclusions)
            if bad_mask.any():
                player_seasons = player_seasons[~bad_mask].reset_index(drop=True)
                logger.info("Dropped %d player_seasons rows matching CFBD id-reuse exclusions",
                            int(bad_mask.sum()))
        # Pad with roster-only (player_id, season_year) pairs — covers OL/etc.
        # who don't accumulate stats in CFBD's stat endpoints, AND covers
        # individual seasons missing from player_seasons even when the player
        # has stat-bearing rows in other years (e.g. Spencer Fano: one fluke
        # 2024 stat row but no 2023/2025 stats despite full-time starting at
        # Utah). Per-(pid, year) granularity is required so the missing-year
        # roster rows aren't silently dropped just because the player has
        # any other player_seasons row anywhere.
        ps_keys = set(zip(
            player_seasons["player_id"].astype(str),
            player_seasons["season_year"].astype("Int64"),
        ))
        roster_pairs = rosters[["player_id", "player_name", "school", "season_year"]] \
            .dropna(subset=["player_id", "season_year"]).copy()
        roster_pairs["_key"] = list(zip(
            roster_pairs["player_id"].astype(str),
            roster_pairs["season_year"].astype("Int64"),
        ))
        roster_only = roster_pairs[~roster_pairs["_key"].isin(ps_keys)] \
            .drop(columns=["_key"])
        if len(roster_only):
            player_seasons = pd.concat([player_seasons, roster_only],
                                        ignore_index=True)
            logger.info("Padded player_seasons with %d roster-only player-seasons "
                        "(missing (pid, year) pairs vs. rosters)", len(roster_only))
        last_season = (player_seasons.groupby("player_id")["season_year"]
                       .max().reset_index()
                       .rename(columns={"season_year": "last_college_season_year"}))
        # Last school by season_year — covers FCS transfers and portal moves that
        # postdate the CFBD roster snapshot (which is FBS-only).
        last_school = (player_seasons.sort_values("season_year")
                       .groupby("player_id")["school"]
                       .last()
                       .reset_index()
                       .rename(columns={"school": "last_ps_school"}))

        # n_transfers_career: count of school changes across consecutive
        # seasons. The Kalman filter already injects extra process-noise
        # variance on each transfer event (school_col="school" in Step 6),
        # but the static fact "this player transferred N times" wasn't
        # surfaced to the draft model. Counting sequential changes (rather
        # than nunique - 1) correctly handles A→B→A patterns.
        ps_sorted = player_seasons.sort_values(["player_id", "season_year"]).copy()
        ps_sorted["prev_school"] = ps_sorted.groupby("player_id")["school"].shift(1)
        ps_sorted["is_transfer_event"] = (
            ps_sorted["prev_school"].notna()
            & (ps_sorted["prev_school"] != ps_sorted["school"])
        )
        n_transfers = (ps_sorted.groupby("player_id")["is_transfer_event"]
                       .sum().astype(int).reset_index()
                       .rename(columns={"is_transfer_event": "n_transfers_career"}))
        logger.info("Transfer counts: %d players with ≥1 transfer (max %d)",
                    int((n_transfers["n_transfers_career"] >= 1).sum()),
                    int(n_transfers["n_transfers_career"].max() or 0))
    else:
        last_season = pd.DataFrame()
        last_school = pd.DataFrame()
        n_transfers = pd.DataFrame()

    # ── One row per player (most recent season) ─────────────────────────────────
    if "player_id" in rosters.columns:
        attrs = (rosters.sort_values("season_year")
                 .groupby("player_id", as_index=False)
                 .last())
    else:
        attrs = (rosters.sort_values("season_year")
                 .groupby(["player_name", "school"], as_index=False)
                 .last())

    # Pad attrs with player_seasons / gamelogs skeleton rows for pids that
    # have CFBD coverage but no rosters entry (FCS draftees — CFBD's roster
    # endpoint is FBS-only — and pre-rosters historical players). Without
    # this, ~180 settled bridge players from FCS schools (Garoppolo,
    # Gronkowski, Joe Thomas, Goedert, Tucker Kraft, etc.) train with all-
    # NaN attributes. Skeleton rows have NaN height_in/weight_lbs/class_year_int;
    # `position` is filled by the bridge_players.draft_position backfill below
    # (line ~272) for draftees, and stays NaN→"OTHER" otherwise.
    if "player_id" in attrs.columns:
        roster_pids = set(attrs["player_id"].dropna())
        skeleton_parts: list[pd.DataFrame] = []
        if seasons_path.exists():
            ps_skel = (player_seasons.loc[
                ~player_seasons["player_id"].isin(roster_pids),
                ["player_id", "player_name", "school", "season_year"],
            ])
            if len(ps_skel):
                skeleton_parts.append(ps_skel)
        gl_path = RAW_DIR / "player_gamelogs.parquet"
        if gl_path.exists():
            covered = roster_pids | (
                set(skeleton_parts[0]["player_id"].dropna())
                if skeleton_parts else set()
            )
            gl = pd.read_parquet(
                gl_path,
                columns=["player_id", "player_name", "school", "season_year"],
            )
            if _exclusions:
                gl = gl[~exclusion_mask(gl, _exclusions)]
            gl_only = gl.loc[~gl["player_id"].isin(covered)]
            if len(gl_only):
                skeleton_parts.append(gl_only)
        if skeleton_parts:
            skel = pd.concat(skeleton_parts, ignore_index=True)
            skel = (skel.dropna(subset=["player_id"])
                         .sort_values("season_year")
                         .groupby("player_id", as_index=False)
                         .last())
            if len(skel):
                attrs = pd.concat([attrs, skel], ignore_index=True)
                logger.info(
                    "Padded attrs with %d skeleton rows for pids in "
                    "player_seasons/gamelogs but absent from rosters "
                    "(FCS bridge players + pre-rosters historical)",
                    len(skel),
                )

    if not last_season.empty and "player_id" in attrs.columns:
        attrs = attrs.merge(last_season, on="player_id", how="left")
        # Gamelog-only pids (CFBD's FCS players: box-score rows from games
        # against FBS opponents) aren't in player_seasons or rosters, so
        # last_season misses them; their skeleton row carries their latest
        # gamelog season. Without this they have no draft year and never
        # enter training or scoring (all 119 bridge players linked by step
        # 02's gamelog_only pass were absent from the deployed OOF).
        gl_last = attrs["last_college_season_year"].isna() & attrs["season_year"].notna()
        attrs.loc[gl_last, "last_college_season_year"] = attrs.loc[gl_last, "season_year"]
        logger.info("Last college season from gamelogs for %d gamelog-only players",
                    int(gl_last.sum()))
        if not n_transfers.empty:
            attrs = attrs.merge(n_transfers, on="player_id", how="left")
            attrs["n_transfers_career"] = attrs["n_transfers_career"].fillna(0).astype(int)

    # Update school to match the player's last known school from player_seasons.
    # CFBD roster data is FBS-only, so FCS transfers and late portal moves show
    # their old FBS school. This also fixes the SP+ join below.
    if not last_school.empty and "player_id" in attrs.columns:
        attrs = attrs.merge(last_school, on="player_id", how="left")
        n_updated = (
            attrs["last_ps_school"].notna()
            & attrs["school"].notna()
            & (attrs["school"] != attrs["last_ps_school"])
        ).sum()
        attrs["school"] = attrs["last_ps_school"].fillna(attrs["school"])
        attrs = attrs.drop(columns=["last_ps_school"])
        logger.info("Updated school to last known school for %d players "
                    "(FCS transfers and portal moves)", n_updated)

    # ── Unmappable roster positions → missing ──────────────────────────────────
    # CFBD writes "?" (and a handful of other non-position tokens) when a
    # roster's position is unknown. Those are missing data, not a position, but
    # they are non-null strings — so the draft-position backfill below, which
    # keys on isna(), never fired for them and the player was stranded in
    # OTHER. OTHER is excluded from every per-position model, so these players
    # dropped out of both the training labels and the scored output.
    unknown_pos = attrs["position"].notna() & ~attrs["position"].isin(POSITION_TO_GROUP)
    if unknown_pos.any():
        logger.info("Clearing %d unmappable roster positions before backfill: %s",
                    int(unknown_pos.sum()),
                    attrs.loc[unknown_pos, "position"].value_counts().head(10).to_dict())
        attrs.loc[unknown_pos, "position"] = np.nan

    # ── Bridge player position backfill ────────────────────────────────────────
    bridge_path = PROCESSED_DIR / "bridge_players.parquet"
    if bridge_path.exists():
        bridge_full = pd.read_parquet(bridge_path)
        bridge_full = bridge_full.rename(columns={
            "college_team": "school",
            "position": "draft_position",
        })
        merge_key = "player_id" if ("player_id" in attrs.columns and
                                     "player_id" in bridge_full.columns) else None
        if merge_key:
            # Step 2 guarantees one bridge row per resolved player_id; dedupe
            # defensively anyway so a stale bridge parquet can't fan attrs out.
            attrs = attrs.merge(
                bridge_full.dropna(subset=["player_id"])
                           .drop_duplicates(subset=["player_id"])
                           [["player_id", "draft_position"]],
                on="player_id", how="left")
        elif "player_name" in attrs.columns and "player_name" in bridge_full.columns:
            attrs = attrs.merge(
                bridge_full[["player_name", "draft_position"]].drop_duplicates("player_name"),
                on="player_name", how="left",
            )
        if "draft_position" in attrs.columns:
            mask = attrs["position"].isna() & attrs["draft_position"].notna()
            attrs.loc[mask, "position"] = attrs.loc[mask, "draft_position"]
        attrs = attrs.drop(columns=["draft_position"], errors="ignore")

    attrs["position_group"] = attrs["position"].map(POSITION_TO_GROUP).fillna("OTHER")

    # ── EDGE reconciliation ─────────────────────────────────────────────────────
    # The CFBD roster label alone can't separate edge rushers from interior DL
    # or off-ball LBs (it's often the generic "DL"/"LB"), so ~1,174 PFF-confirmed
    # edges were being modeled in the off-ball LB head while their DL-labeled
    # twins sat in DL — split across two residual cohorts, z-cohorts, and heads.
    # PFF's ED/DI/LB position code is the authority where we have it (2014+);
    # for the pre-2014 cohort with no PFF, a roster/draft "DE"-family label on a
    # base-DL player is a reliable edge signal (college interior is "DT"/"NT").
    # We deliberately do NOT guess on no-PFF LBs (roster "OLB" is genuinely
    # split between edge and off-ball), so a small set of pre-2014 3-4 edge-OLBs
    # stays in LB — the acknowledged residual bias.
    attrs = _reconcile_edge(attrs)

    # ── Birth year and age ──────────────────────────────────────────────────────
    # Source: nfl_data_py (bridge players only — ~99.9% birth_date coverage)
    # Join chain: attrs.player_id → bridge_players.player_id → pfr_id → birth_year
    attrs["birth_year"] = np.nan
    attrs["birth_date"] = None
    attrs["age_at_last_college_season"] = np.nan

    if bridge_path.exists():
        bridge_df = pd.read_parquet(bridge_path,
                                     columns=["player_id", "pfr_id"])
        birth_years = load_birth_dates()   # dict: pfr_id → birth_year (int)

        if birth_years and "player_id" in attrs.columns:
            bridge_df["birth_year"] = bridge_df["pfr_id"].map(birth_years)
            birth_map = (bridge_df.dropna(subset=["birth_year"])
                         .drop_duplicates("player_id")
                         .set_index("player_id")["birth_year"])
            attrs["birth_year"] = attrs["player_id"].map(birth_map)

            # Reject implausible bridge-derived birth_years — cross-
            # generational namesake collisions can mint either an absurdly
            # old age (bridge stamped with a 1981-born NFL veteran's pfr_id)
            # or an absurdly young one (matched to someone born after the
            # player's documented college years). Clearing both lets
            # wikidata / class-year estimation fill the correct value.
            if "last_college_season_year" in attrs.columns:
                implied_age = attrs["last_college_season_year"] - attrs["birth_year"]
                bad_mask = (implied_age > MAX_BRIDGE_DERIVED_AGE) | \
                           (implied_age < MIN_BRIDGE_DERIVED_AGE)
                n_bad = int(bad_mask.sum())
                if n_bad:
                    logger.warning(
                        "Rejected %d bridge-derived birth_year(s) with implied "
                        "age outside [%d, %d] (likely namesake collision)",
                        n_bad, MIN_BRIDGE_DERIVED_AGE, MAX_BRIDGE_DERIVED_AGE,
                    )
                    attrs.loc[bad_mask, "birth_year"] = np.nan

                attrs["age_at_last_college_season"] = (
                    attrs["last_college_season_year"] - attrs["birth_year"]
                )

        resolved = attrs["birth_year"].notna().sum()
        logger.info("Birth year resolved for %d / %d players (bridge/nfl_data_py)", resolved, len(attrs))

    # ── Wikidata birth dates (second priority, non-bridge players only) ────────
    wikidata_path = PROCESSED_DIR / "wikidata_birth_dates.parquet"
    if wikidata_path.exists():
        wikidata = pd.read_parquet(wikidata_path, columns=["player_id", "birth_year", "birth_date"])
        wikidata = wikidata.dropna(subset=["player_id", "birth_year"]).drop_duplicates("player_id")
        wikidata_map_year = wikidata.set_index("player_id")["birth_year"]
        wikidata_map_date = wikidata.set_index("player_id")["birth_date"]
        missing_mask = attrs["birth_year"].isna()
        attrs.loc[missing_mask, "birth_year"] = attrs.loc[missing_mask, "player_id"].map(wikidata_map_year)
        attrs.loc[missing_mask, "birth_date"] = attrs.loc[missing_mask, "player_id"].map(wikidata_map_date)
        n_wiki = (missing_mask & attrs["birth_year"].notna()).sum()
        logger.info("Birth year resolved for %d additional players (Wikidata)", n_wiki)

        # Wikidata matches by name only — namesake collisions slip through
        # (e.g. a current college Monte Harrison gets the 1995 birth year of
        # an unrelated MLB player named Monte Harrison). Re-apply the same
        # symmetric age sanity reject so class-year estimation can fill
        # correctly.
        if "last_college_season_year" in attrs.columns:
            implied_age = attrs["last_college_season_year"] - attrs["birth_year"]
            bad_mask = (implied_age > MAX_BRIDGE_DERIVED_AGE) | \
                       (implied_age < MIN_BRIDGE_DERIVED_AGE)
            n_bad = int(bad_mask.sum())
            if n_bad:
                logger.warning(
                    "Rejected %d wikidata-derived birth_year(s) with implied "
                    "age outside [%d, %d] (likely namesake match)",
                    n_bad, MIN_BRIDGE_DERIVED_AGE, MAX_BRIDGE_DERIVED_AGE,
                )
                attrs.loc[bad_mask, "birth_year"] = np.nan
                attrs.loc[bad_mask, "birth_date"] = None

    # ── Estimate birth_year from class year for non-bridge players ────────────
    # A freshman (class_year=1) in their last college season is typically ~18;
    # a senior (class_year=4) is ~21. Formula: birth_year ≈ last_season_year - class_year - 17
    attrs["birth_year_estimated"] = False
    has_class = (
        attrs["birth_year"].isna()
        & attrs["class_year_int"].notna()
        & attrs["last_college_season_year"].notna()
    )
    if has_class.any():
        attrs.loc[has_class, "birth_year"] = (
            attrs.loc[has_class, "last_college_season_year"]
            - attrs.loc[has_class, "class_year_int"]
            - 17
        ).round()
        attrs.loc[has_class, "birth_year_estimated"] = True
        logger.info("Estimated birth_year for %d additional players via class year", has_class.sum())

    # ── Manual overrides (highest priority) ───────────────────────────────────
    if BIRTH_YEAR_OVERRIDES and "player_id" in attrs.columns:
        for pid, correct_year in BIRTH_YEAR_OVERRIDES.items():
            mask = attrs["player_id"].astype(str) == str(pid)
            if mask.any():
                attrs.loc[mask, "birth_year"] = correct_year
                attrs.loc[mask, "birth_year_estimated"] = False
        logger.info("Applied %d manual birth year override(s)", len(BIRTH_YEAR_OVERRIDES))

    # Recompute age_at_last_college_season for all players (bridge + estimated)
    if "last_college_season_year" in attrs.columns:
        attrs["age_at_last_college_season"] = (
            attrs["last_college_season_year"] - attrs["birth_year"]
        )
        age_valid = attrs["age_at_last_college_season"].dropna()
        if not age_valid.empty:
            logger.info("Birth year total coverage: %d / %d players (mean age=%.1f)",
                        attrs["birth_year"].notna().sum(), len(attrs), age_valid.mean())

    # ── Conference tier ─────────────────────────────────────────────────────────
    attrs["conference_tier"] = DEFAULT_CONFERENCE_TIER
    if seasons_path.exists() and "player_id" in attrs.columns:
        player_seasons_full = pd.read_parquet(
            seasons_path, columns=["player_id", "school", "season_year", "conference"]
        )
        if _exclusions:
            bad_mask = exclusion_mask(player_seasons_full, _exclusions)
            if bad_mask.any():
                player_seasons_full = player_seasons_full[~bad_mask].reset_index(drop=True)
        last_conf = (
            player_seasons_full.sort_values("season_year")
            .groupby("player_id")["conference"]
            .last()
            .reset_index()
        )
        last_conf["conference_tier"] = last_conf["conference"].map(
            lambda c: CONFERENCE_TIERS.get(c, DEFAULT_CONFERENCE_TIER)
        )
        attrs = attrs.merge(
            last_conf[["player_id", "conference_tier"]].rename(
                columns={"conference_tier": "conference_tier_new"}
            ),
            on="player_id", how="left",
        )
        # Roster-only players (OL etc., no rows in player_seasons) miss the
        # primary player_id-keyed lookup. Fall back to a (school, year)-keyed
        # conference map derived from the same source — Texas A&M in 2025
        # carries conference="SEC" for every player_seasons row, so the school
        # itself resolves the tier even when this specific player has no stats.
        school_year_conf = (
            player_seasons_full.dropna(subset=["school", "season_year", "conference"])
            .drop_duplicates(["school", "season_year"])
            [["school", "season_year", "conference"]]
        )
        school_year_conf["conference_tier_school"] = school_year_conf["conference"].map(
            lambda c: CONFERENCE_TIERS.get(c, DEFAULT_CONFERENCE_TIER)
        )
        attrs = attrs.merge(
            school_year_conf[["school", "season_year", "conference_tier_school"]].rename(
                columns={"season_year": "last_college_season_year"}
            ),
            on=["school", "last_college_season_year"], how="left",
        )
        # Third fallback: team_conferences.parquet's school+season+conference.
        # Covers pre-2010 schools (and some FCS programs) that have no
        # player_seasons rows because CFBD's stat-bearing coverage was sparse
        # before ~2010 — player_seasons has 8 rows for 2004 and 65 for 2005.
        #
        # This used to read team_ratings.parquet, but SP+ reports each team's
        # CURRENT conference on historical rows, so 2004 Utah arrived as "Big 12"
        # (tier 1) when it was Mountain West (tier 2). The anachronism bit hardest
        # in precisely the pre-2010 window this fallback exists to serve. There is
        # deliberately NO fallback back to team_ratings: a missing tier defaults
        # to DEFAULT_CONFERENCE_TIER, which is better than a confidently wrong one.
        conf_cache_for_tier = RAW_DIR / "team_conferences.parquet"
        if conf_cache_for_tier.exists():
            tr_conf = pd.read_parquet(
                conf_cache_for_tier, columns=["school", "season_year", "conference"]
            ).dropna(subset=["school", "season_year", "conference"])
            tr_conf = tr_conf.drop_duplicates(["school", "season_year"])
            tr_conf["conference_tier_ratings"] = tr_conf["conference"].map(
                lambda c: CONFERENCE_TIERS.get(c, DEFAULT_CONFERENCE_TIER)
            )
            attrs = attrs.merge(
                tr_conf[["school", "season_year", "conference_tier_ratings"]].rename(
                    columns={"season_year": "last_college_season_year"}
                ),
                on=["school", "last_college_season_year"], how="left",
            )
        else:
            logger.warning("team_conferences.parquet missing — pre-2010 players "
                           "with no player_seasons rows will fall through to "
                           "DEFAULT_CONFERENCE_TIER. Re-run step 1.")
            attrs["conference_tier_ratings"] = np.nan

        attrs["conference_tier"] = (
            attrs["conference_tier_new"]
            .fillna(attrs["conference_tier_school"])
            .fillna(attrs["conference_tier_ratings"])
            .fillna(DEFAULT_CONFERENCE_TIER)
            .astype(int)
        )
        attrs = attrs.drop(columns=["conference_tier_new", "conference_tier_school",
                                     "conference_tier_ratings"])
        tier_counts = attrs["conference_tier"].value_counts().to_dict()
        logger.info("Conference tier distribution: %s", tier_counts)

    # ── Career-max conference tier ─────────────────────────────────────────────
    # `conference_tier` above is the player's FINAL-college-season tier only,
    # which can't distinguish e.g. a four-year P4 starter from a player who
    # peaked in PFF as a freshman at FCS then transferred up to P4. Two
    # career-aggregated binaries address that, matching the conf_tier_power4
    # / conf_tier_g5 mutually-exclusive convention used in PFF_CONTEXT_FEATURES:
    #   - career_max_conf_tier_power4: 1 if the player EVER played at P4
    #   - career_max_conf_tier_g5:     1 if the player's best level was G5
    # Transfer count is already captured by `n_transfers_career` above.
    #
    # Source: union of (player_id, season_year, school) from player_seasons +
    # rosters, joined to (school, season_year) conference from player_seasons
    # AND team_conferences (covers roster-only OL/walk-ons + pre-2010 schools).
    attrs["career_max_conf_tier_power4"] = 0
    attrs["career_max_conf_tier_g5"]     = 0
    if "player_id" in attrs.columns:
        # Collect (player_id, school, season_year) from both sources
        union_rows: list[pd.DataFrame] = []
        if seasons_path.exists():
            union_rows.append(
                player_seasons_full[["player_id", "school", "season_year"]]
                    .dropna(subset=["player_id", "school", "season_year"])
            )
        if "player_id" in rosters.columns:
            union_rows.append(
                rosters[["player_id", "school", "season_year"]]
                    .dropna(subset=["player_id", "school", "season_year"])
            )
        if union_rows:
            career_rows = pd.concat(union_rows, ignore_index=True).drop_duplicates(
                ["player_id", "school", "season_year"]
            )

            # Build (school, season_year) -> conference map from same chain as
            # conference_tier above: player_seasons.conference primary,
            # team_conferences.conference fallback (per-season affiliation — NOT
            # team_ratings, whose conference column is present-day only).
            ps_conf_map = pd.DataFrame()
            if seasons_path.exists():
                ps_conf_map = (player_seasons_full
                               .dropna(subset=["school", "season_year", "conference"])
                               [["school", "season_year", "conference"]]
                               .drop_duplicates(["school", "season_year"]))
            tr_conf_map = pd.DataFrame()
            conf_cache_for_career = RAW_DIR / "team_conferences.parquet"
            if conf_cache_for_career.exists():
                tr_conf_map = (pd.read_parquet(
                    conf_cache_for_career, columns=["school", "season_year", "conference"]
                ).dropna(subset=["school", "season_year", "conference"])
                  .drop_duplicates(["school", "season_year"])
                  .rename(columns={"conference": "_conf_tr"}))

            if len(ps_conf_map):
                career_rows = career_rows.merge(
                    ps_conf_map, on=["school", "season_year"], how="left"
                )
            else:
                career_rows["conference"] = None
            if len(tr_conf_map):
                career_rows = career_rows.merge(
                    tr_conf_map, on=["school", "season_year"], how="left"
                )
                career_rows["conference"] = career_rows["conference"].fillna(
                    career_rows["_conf_tr"]
                )
                career_rows = career_rows.drop(columns=["_conf_tr"])

            career_rows["_tier"] = career_rows["conference"].map(
                lambda c: CONFERENCE_TIERS.get(c, DEFAULT_CONFERENCE_TIER)
                          if pd.notna(c) else DEFAULT_CONFERENCE_TIER
            )
            best_tier = (career_rows.groupby("player_id")["_tier"].min()
                                     .rename("_best_tier").reset_index())
            attrs = attrs.merge(best_tier, on="player_id", how="left")
            attrs["career_max_conf_tier_power4"] = (
                attrs["_best_tier"] == 1
            ).astype(int)
            attrs["career_max_conf_tier_g5"] = (
                attrs["_best_tier"] == 2
            ).astype(int)
            attrs = attrs.drop(columns=["_best_tier"])

            logger.info(
                "Career-max conference: ever P4=%d  best-was-G5=%d  FCS-only=%d (of %d total)",
                int(attrs["career_max_conf_tier_power4"].sum()),
                int(attrs["career_max_conf_tier_g5"].sum()),
                int(((attrs["career_max_conf_tier_power4"] == 0) &
                     (attrs["career_max_conf_tier_g5"] == 0)).sum()),
                len(attrs),
            )

    # ── Team SP+ offense + defense ratings ─────────────────────────────────────
    ratings_cache = RAW_DIR / "team_ratings.parquet"
    attrs["team_sp_offense"] = np.nan
    attrs["team_sp_defense"] = np.nan
    attrs["team_sp_overall"] = np.nan
    if ratings_cache.exists() and "last_college_season_year" in attrs.columns:
        ratings = pd.read_parquet(
            ratings_cache,
            columns=["school", "season_year", "sp_plus", "sp_plus_offense", "sp_plus_defense"],
        )
        ratings = ratings.rename(columns={"season_year": "last_college_season_year"})
        attrs = attrs.merge(
            ratings[["school", "last_college_season_year",
                     "sp_plus", "sp_plus_offense", "sp_plus_defense"]],
            on=["school", "last_college_season_year"], how="left",
        )
        attrs["team_sp_offense"] = attrs["sp_plus_offense"]
        attrs["team_sp_defense"] = attrs["sp_plus_defense"]
        attrs["team_sp_overall"] = attrs["sp_plus"]
        attrs = attrs.drop(columns=["sp_plus", "sp_plus_offense", "sp_plus_defense"],
                           errors="ignore")
        n_filled = attrs["team_sp_offense"].notna().sum()
        logger.info("Team SP+ resolved for %d / %d players (offense, defense, overall)",
                    n_filled, len(attrs))

    # ── Career team-context aggregates from advanced_game_stats ─────────────────
    # Captures the offensive / defensive environment a player developed in,
    # averaged across every season they appeared on a team's roster. Each
    # team-season is summarized to the mean of its game-level advanced
    # metrics; then per-player career means span all (school, season) pairs
    # in their player_seasons history (which step 03 has padded with rosters
    # earlier — so OL and roster-only players are covered too).
    career_team_cols = [
        "career_team_off_ppa_mean",
        "career_team_off_success_rate_mean",
        "career_team_off_explosiveness_mean",
        "career_team_off_stuff_rate_mean",
        "career_team_def_ppa_mean",
        "career_team_def_success_rate_mean",
        "career_team_def_stuff_rate_mean",
        "career_team_def_explosiveness_mean",
        "career_team_line_yards_avg_mean",
        "career_team_second_level_yards_avg_mean",
        "career_team_open_field_yards_avg_mean",
        "career_team_rushing_power_success_mean",
        "career_team_rushing_stuff_rate_mean",
        "career_team_havoc_total_mean",
        "career_team_havoc_front_seven_mean",
        "career_team_havoc_db_mean",
    ]
    for col in career_team_cols:
        attrs[col] = np.nan
    ags_path = RAW_DIR / "advanced_game_stats.parquet"
    if ags_path.exists() and seasons_path.exists() and "player_id" in attrs.columns:
        ags = pd.read_parquet(
            ags_path,
            columns=["game_id", "team", "season_year",
                     "off_ppa", "off_success_rate",
                     "off_explosiveness", "off_stuff_rate",
                     "def_ppa", "def_success_rate", "def_stuff_rate",
                     "def_explosiveness"],
        )
        # Team box scores carry the rushing four-factor / havoc detail that
        # advanced_game_stats doesn't. This is the only production-adjacent
        # career signal available for OL, who have no per-game stats of their
        # own (MODELED_STATS["OL"] is empty) and no PFF coverage before 2014.
        # Joined at (game_id, team) — the same key step 01 uses for havoc.
        # NOTE: this is the CAREER-MEAN channel, sibling to career_team_off_ppa
        # below. The per-game *rolling* variant of these same stats was tested
        # and reverted (2026-05-16); the career channel is a different
        # construction and was never evaluated.
        tbs_cols = ["line_yards_avg", "second_level_yards_avg",
                    "open_field_yards_avg", "rushing_power_success",
                    "rushing_stuff_rate", "havoc_total", "havoc_front_seven",
                    "havoc_db"]
        tbs_path = RAW_DIR / "team_box_scores.parquet"
        if tbs_path.exists():
            tbs = pd.read_parquet(tbs_path)
            present = [c for c in tbs_cols if c in tbs.columns]
            if present:
                ags = ags.merge(tbs[["game_id", "team"] + present],
                                on=["game_id", "team"], how="left")
            tbs_cols = present
        else:
            tbs_cols = []

        # Team-season averages of the game-level advanced stats.
        agg_cols = ["off_ppa", "off_success_rate", "off_explosiveness",
                    "off_stuff_rate", "def_ppa", "def_success_rate",
                    "def_stuff_rate", "def_explosiveness"] + tbs_cols
        team_season = (ags.groupby(["team", "season_year"], as_index=False)
                       [agg_cols].mean())
        team_season = team_season.rename(columns={"team": "school"})
        # Use the same player_seasons frame that's already been padded with
        # rosters above (so OL coverage is intact). Re-read from disk —
        # safer than threading the in-memory frame through the long
        # attrs-construction sequence.
        ps_for_team = pd.read_parquet(
            seasons_path,
            columns=["player_id", "school", "season_year"],
        ).dropna(subset=["player_id", "school", "season_year"])
        # Pad with rosters at (player_id, school, year) granularity to match
        # the `last_college_season_year` correction earlier in this step.
        if rosters_cache.exists():
            rosters_for_team = pd.read_parquet(
                rosters_cache,
                columns=["player_id", "school", "season_year"],
            ).dropna(subset=["player_id", "school", "season_year"])
            ps_keys = set(zip(ps_for_team["player_id"].astype(str),
                              ps_for_team["season_year"].astype("Int64")))
            rosters_for_team["_key"] = list(zip(
                rosters_for_team["player_id"].astype(str),
                rosters_for_team["season_year"].astype("Int64"),
            ))
            ros_only = rosters_for_team[~rosters_for_team["_key"].isin(ps_keys)] \
                .drop(columns=["_key"])
            ps_for_team = pd.concat([ps_for_team, ros_only], ignore_index=True)
        # Join team-season metrics onto each player-season, then per-player mean.
        joined = ps_for_team.merge(team_season, on=["school", "season_year"], how="left")
        per_player = (joined.groupby("player_id", as_index=False)[agg_cols].mean())
        per_player = per_player.rename(columns={
            "off_ppa": "career_team_off_ppa_mean",
            "off_success_rate": "career_team_off_success_rate_mean",
            "off_explosiveness": "career_team_off_explosiveness_mean",
            "off_stuff_rate": "career_team_off_stuff_rate_mean",
            "def_ppa": "career_team_def_ppa_mean",
            "def_success_rate": "career_team_def_success_rate_mean",
            "def_stuff_rate": "career_team_def_stuff_rate_mean",
            "def_explosiveness": "career_team_def_explosiveness_mean",
            **{c: f"career_team_{c}_mean" for c in tbs_cols},
        })
        per_player["player_id"] = per_player["player_id"].astype(str)
        # Drop the placeholder NaN columns and merge fresh — avoids _x/_y suffixes.
        attrs = attrs.drop(columns=career_team_cols, errors="ignore")
        attrs["player_id"] = attrs["player_id"].astype(str)
        attrs = attrs.merge(per_player, on="player_id", how="left")
        n_team_ctx = attrs["career_team_off_ppa_mean"].notna().sum()
        logger.info("Career team-context aggregates resolved for %d / %d players",
                    n_team_ctx, len(attrs))

    # ── Specialist career detail (K leg strength / P placement) ────────────────
    # K and P are the thinnest heads in the model, and PFF only covers 2014+.
    # These season columns reach back to 2005/2006, so they're the only
    # detail signal available for the bridge cohort that predates PFF.
    # Career-max FG distance is a direct leg-strength read; inside-20 and
    # touchback rates are the core placement skill (and are genuine rates, so
    # they need the punt-count denominator rather than raw counts, which would
    # just re-encode career length).
    for col in ("career_fg_long_max", "career_punt_long_max",
                "career_punt_in20_rate", "career_punt_tb_rate"):
        attrs[col] = np.nan
    if seasons_path.exists() and "player_id" in attrs.columns:
        spec_cols = ["kicking_long", "punting_long", "punting_in_20",
                     "punting_tb", "punts"]
        ps_all = pd.read_parquet(seasons_path)
        present = [c for c in spec_cols if c in ps_all.columns]
        if present and "player_id" in ps_all.columns:
            ps_spec = ps_all[["player_id"] + present].dropna(subset=["player_id"])
            for c in present:
                ps_spec[c] = pd.to_numeric(ps_spec[c], errors="coerce")
            # Plausibility guards on the raw CFBD longs. The NCAA field-goal
            # record is 69 yards, but the feed carries at least one 92 (Josh
            # Clarke, Ole Miss 2018, rostered at LB) — a career MAX propagates
            # a single bad row for the player's whole career, so drop the
            # impossible values rather than let them set the maximum. Punts
            # are capped at the length of the field; the real 91/92-yard
            # punts in the data are legitimate and survive.
            for c, cap in (("kicking_long", 70.0), ("punting_long", 100.0)):
                if c in ps_spec.columns:
                    bad = ps_spec[c] > cap
                    if bad.any():
                        logger.info("Nulling %d implausible %s value(s) > %.0f",
                                    int(bad.sum()), c, cap)
                        ps_spec.loc[bad, c] = np.nan
            agg: dict[str, tuple] = {}
            if "kicking_long" in present:
                agg["career_fg_long_max"] = ("kicking_long", "max")
            if "punting_long" in present:
                agg["career_punt_long_max"] = ("punting_long", "max")
            for c in ("punting_in_20", "punting_tb", "punts"):
                if c in present:
                    agg[f"_sum_{c}"] = (c, "sum")
            spec = ps_spec.groupby("player_id", as_index=False).agg(**agg)
            if "_sum_punts" in spec.columns:
                denom = spec["_sum_punts"].replace(0, np.nan)
                if "_sum_punting_in_20" in spec.columns:
                    spec["career_punt_in20_rate"] = spec["_sum_punting_in_20"] / denom
                if "_sum_punting_tb" in spec.columns:
                    spec["career_punt_tb_rate"] = spec["_sum_punting_tb"] / denom
            spec = spec.drop(columns=[c for c in spec.columns
                                      if c.startswith("_sum_")], errors="ignore")
            spec["player_id"] = spec["player_id"].astype(str)
            attrs["player_id"] = attrs["player_id"].astype(str)
            attrs = attrs.drop(columns=[c for c in spec.columns if c != "player_id"],
                               errors="ignore")
            attrs = attrs.merge(spec, on="player_id", how="left")
            logger.info("Specialist career detail: %d players with a FG long, "
                        "%d with a punt-placement rate",
                        int(attrs.get("career_fg_long_max",
                                      pd.Series(dtype=float)).notna().sum()),
                        int(attrs.get("career_punt_in20_rate",
                                      pd.Series(dtype=float)).notna().sum()))

    # ── Audit item-5 career detail (explosiveness / tackle quality / ball skills)
    # Four season-column aggregates the feature-utilization audit found
    # genuinely unused. All are career-level, all NaN (never 0) when the source
    # is absent, so LightGBM routes missing via its own split direction rather
    # than reading "no data" as "zero ability".
    #
    # Coverage was verified per-year before building these, because the audit's
    # framing was wrong on one of them:
    #   career_rec_long_max / career_rush_long_max — `receiving_long` /
    #     `rushing_long` exist for ALL years 2004-2025. The audit filed this as
    #     "for the pre-2014 cohort", which is about where it's most VALUABLE
    #     (pre-PFF, pre-play-layer, when nothing else describes explosiveness),
    #     not where the data lives. Best-covered of the four.
    #   career_int_return_yards_per_int — return yards per interception:
    #     closing speed and ball skills after the catch, for DB/LB. A rate, not
    #     a count, so it doesn't just re-encode how many INTs a long career
    #     accumulated.
    #   career_solo_tackle_share — solo / total tackles. Distinguishes a
    #     defender making stops alone from one arriving second. THINNEST of the
    #     four by far: `tackles_solo` is exactly zero before 2016 (verified —
    #     not sparse, absent), so this is NaN for 12 of the 22 training years and
    #     for every draft class through ~2016. Included because the audit asked
    #     for it, but it is the one most likely to land at the noise floor, the
    #     way hasan_rank (2.8% coverage) did.
    for col in ("career_rec_long_max", "career_rush_long_max",
                "career_int_return_yards_per_int", "career_solo_tackle_share"):
        attrs[col] = np.nan
    if seasons_path.exists() and "player_id" in attrs.columns:
        item5_cols = ["receiving_long", "rushing_long", "interceptions_yds",
                      "interceptions_int", "tackles_solo", "tackles"]
        ps_all5 = pd.read_parquet(seasons_path)
        present5 = [c for c in item5_cols if c in ps_all5.columns]
        if present5 and "player_id" in ps_all5.columns:
            ps5 = ps_all5[["player_id"] + present5].dropna(subset=["player_id"]).copy()
            for c in present5:
                ps5[c] = pd.to_numeric(ps5[c], errors="coerce")
            # Plausibility guards. A career MAX propagates one bad row for the
            # player's whole career, so cap at the length of the field — a 100+
            # yard reception is impossible, and negatives are feed noise.
            for c in ("receiving_long", "rushing_long"):
                if c in ps5.columns:
                    bad = (ps5[c] > 100.0) | (ps5[c] < 0)
                    if bad.any():
                        logger.info("Nulling %d implausible %s value(s)",
                                    int(bad.sum()), c)
                        ps5.loc[bad, c] = np.nan
            agg5: dict[str, tuple] = {}
            if "receiving_long" in present5:
                agg5["career_rec_long_max"] = ("receiving_long", "max")
            if "rushing_long" in present5:
                agg5["career_rush_long_max"] = ("rushing_long", "max")
            for c in ("interceptions_yds", "interceptions_int",
                      "tackles_solo", "tackles"):
                if c in present5:
                    agg5[f"_s5_{c}"] = (c, "sum")
            it5 = ps5.groupby("player_id", as_index=False).agg(**agg5)

            # Rates use career sums over a career denominator. `replace(0, nan)`
            # is what keeps "0 INTs" from becoming a 0-yard return average —
            # a player with no interceptions has no ball-skill measurement,
            # which is missing, not zero.
            if "_s5_interceptions_int" in it5.columns and "_s5_interceptions_yds" in it5.columns:
                denom_int = it5["_s5_interceptions_int"].replace(0, np.nan)
                ret_avg = it5["_s5_interceptions_yds"] / denom_int
                # Raw `interceptions_yds` reaches -100 in the feed (421 negative
                # season rows), which is not a real return — you cannot retreat
                # the length of the field. Small negatives ARE legitimate (tackled
                # behind the catch point), so the window keeps those and only
                # drops the impossible tail: 100 is a pick-six from your own goal
                # line, and -20 is already a generous worst case. Affects 3
                # players, but a career average propagates one bad row forever.
                it5["career_int_return_yards_per_int"] = ret_avg.where(
                    (ret_avg >= -20.0) & (ret_avg <= 100.0))
            if "_s5_tackles" in it5.columns and "_s5_tackles_solo" in it5.columns:
                denom_tkl = it5["_s5_tackles"].replace(0, np.nan)
                share = it5["_s5_tackles_solo"] / denom_tkl
                # tackles_solo ⊆ tackles, so >1 means the feed disagrees with
                # itself; drop rather than clip so it reads as unknown.
                it5["career_solo_tackle_share"] = share.where(share <= 1.0)

            it5 = it5.drop(columns=[c for c in it5.columns
                                    if c.startswith("_s5_")], errors="ignore")
            it5["player_id"] = it5["player_id"].astype(str)
            attrs["player_id"] = attrs["player_id"].astype(str)
            attrs = attrs.drop(columns=[c for c in it5.columns if c != "player_id"],
                               errors="ignore")
            attrs = attrs.merge(it5, on="player_id", how="left")
            logger.info("Item-5 career detail: rec_long %d, rush_long %d, "
                        "int_ret_per_int %d, solo_share %d players",
                        *[int(attrs.get(c, pd.Series(dtype=float)).notna().sum())
                          for c in ("career_rec_long_max", "career_rush_long_max",
                                    "career_int_return_yards_per_int",
                                    "career_solo_tackle_share")])

    # ── Quarter-usage profile (Q1 / Q4 share of career plays) ──────────────────
    # Which quarters a player's snaps come in separates a starter from a
    # mop-up-duty backup with a similar raw snap count: starters are on the field
    # in Q1, deep-bench players accumulate volume in Q4 blowouts. Emitted as
    # shares of the player's own career plays, so it measures usage SHAPE and is
    # independent of career length.
    #
    # Only Q1 and Q4 are emitted, not all four. Q1-Q4 shares sum to ~1, so the
    # full set is collinear by construction, and Q2/Q3 carry almost no
    # starter/backup contrast (everyone plays the middle quarters). Overtime
    # (period 5-7, ~7k rows) folds into the denominator but gets no share column.
    #
    # NOTE: player_plays starts in 2013 (2012 has 697 rows), so these are NaN for
    # every player whose career ended before then — 9 of the 22 training years.
    # Second-sparsest feature in this batch after career_solo_tackle_share.
    attrs["career_q1_play_share"] = np.nan
    attrs["career_q4_play_share"] = np.nan
    plays_path = PROCESSED_DIR / "player_plays.parquet"
    if plays_path.exists() and "player_id" in attrs.columns:
        try:
            pp = pd.read_parquet(plays_path, columns=["athlete_id", "period"])
        except Exception as exc:
            logger.warning("Could not read player_plays for quarter shares: %s", exc)
            pp = pd.DataFrame()
        if not pp.empty:
            pp = pp.dropna(subset=["athlete_id"])
            pp["period"] = pd.to_numeric(pp["period"], errors="coerce")
            pp = pp[pp["period"].notna()]
            grp = pp.groupby("athlete_id")["period"]
            q = pd.DataFrame({
                "total": grp.size(),
                "q1":    grp.apply(lambda s: int((s == 1).sum())),
                "q4":    grp.apply(lambda s: int((s == 4).sum())),
            }).reset_index()
            denom_q = q["total"].replace(0, np.nan)
            q["career_q1_play_share"] = q["q1"] / denom_q
            q["career_q4_play_share"] = q["q4"] / denom_q
            q["player_id"] = q["athlete_id"].astype(str)
            q = q[["player_id", "career_q1_play_share", "career_q4_play_share"]]
            attrs["player_id"] = attrs["player_id"].astype(str)
            attrs = attrs.drop(columns=["career_q1_play_share",
                                        "career_q4_play_share"], errors="ignore")
            attrs = attrs.merge(q, on="player_id", how="left")
            logger.info("Quarter-usage profile: %d players with a Q1/Q4 share",
                        int(attrs["career_q1_play_share"].notna().sum()))

    # ── Career altitude exposure (per-game avg, bucketed) ──────────────────────
    # Per-player mean of the ACTUAL altitudes the player faced across all
    # games (home + away). A BYU player's home stadium is 4553ft but their
    # season averages ~3000ft once sea-level away games count — that's the
    # honest "did they adapt to thin air" signal. The diff-from-home feature
    # in step 4 already handles the within-game altitude effect identity-
    # safely; this attribute supplies the durable career-level adaptation
    # prior to the draft model. Bucketed to ALTITUDE_DIFF_BUCKET_FT (500ft)
    # to dilute the Air Force / Wyoming identity tag.
    attrs["career_avg_elevation_ft"] = np.nan
    venues_path = RAW_DIR / "venues.parquet"
    sched_path  = RAW_DIR / "game_schedule.parquet"
    if (venues_path.exists() and sched_path.exists() and seasons_path.exists()
            and "player_id" in attrs.columns):
        venues_elev = pd.read_parquet(
            venues_path, columns=["venue_id", "elevation_ft"],
        ).dropna(subset=["venue_id"])
        sched = pd.read_parquet(
            sched_path,
            columns=["game_id", "season_year", "venue_id",
                     "home_team", "away_team"],
        ).dropna(subset=["venue_id", "season_year"])
        game_elev = sched.merge(venues_elev, on="venue_id", how="left") \
                          .dropna(subset=["elevation_ft"])

        # Unpivot to (team, season_year, elevation_ft) — every team plays at
        # the game's venue elevation, regardless of home/away/neutral.
        home_rows = game_elev[["home_team", "season_year", "elevation_ft"]] \
                      .rename(columns={"home_team": "school"})
        away_rows = game_elev[["away_team", "season_year", "elevation_ft"]] \
                      .rename(columns={"away_team": "school"})
        team_games = pd.concat([home_rows, away_rows], ignore_index=True) \
                       .dropna(subset=["school", "elevation_ft"])

        team_season_elev = (
            team_games.groupby(["school", "season_year"], as_index=False)
                       ["elevation_ft"].mean()
        )

        # Reuse the padded (player_id, school, season_year) history that
        # career_team_*_mean computed above — OL/walk-on coverage included.
        # Re-derive here (cheap) to avoid threading the variable through.
        ps_for_alt = pd.read_parquet(
            seasons_path,
            columns=["player_id", "school", "season_year"],
        ).dropna(subset=["player_id", "school", "season_year"])
        if rosters_cache.exists():
            rosters_for_alt = pd.read_parquet(
                rosters_cache,
                columns=["player_id", "school", "season_year"],
            ).dropna(subset=["player_id", "school", "season_year"])
            ps_keys = set(zip(ps_for_alt["player_id"].astype(str),
                              ps_for_alt["season_year"].astype("Int64")))
            rosters_for_alt["_key"] = list(zip(
                rosters_for_alt["player_id"].astype(str),
                rosters_for_alt["season_year"].astype("Int64"),
            ))
            ros_only = rosters_for_alt[~rosters_for_alt["_key"].isin(ps_keys)] \
                .drop(columns=["_key"])
            ps_for_alt = pd.concat([ps_for_alt, ros_only], ignore_index=True)

        joined = ps_for_alt.merge(
            team_season_elev, on=["school", "season_year"], how="left",
        )
        per_player_alt = (
            joined.dropna(subset=["elevation_ft"])
                   .groupby("player_id", as_index=False)["elevation_ft"].mean()
                   .rename(columns={"elevation_ft": "career_avg_elevation_ft"})
        )

        # Bucket to ALTITUDE_DIFF_BUCKET_FT for identity dilution
        bucket = float(ALTITUDE_DIFF_BUCKET_FT)
        per_player_alt["career_avg_elevation_ft"] = (
            (per_player_alt["career_avg_elevation_ft"] / bucket).round() * bucket
        )
        per_player_alt["player_id"] = per_player_alt["player_id"].astype(str)

        attrs = attrs.drop(columns=["career_avg_elevation_ft"], errors="ignore")
        attrs["player_id"] = attrs["player_id"].astype(str)
        attrs = attrs.merge(per_player_alt, on="player_id", how="left")
        n_alt = attrs["career_avg_elevation_ft"].notna().sum()
        logger.info("Career altitude exposure resolved for %d / %d players "
                    "(bucketed to %dft)", n_alt, len(attrs), int(bucket))

    # ── Recruiting composite rating ─────────────────────────────────────────────
    recruiting_cache = RAW_DIR / "recruiting.parquet"
    attrs["recruiting_stars"] = np.nan
    attrs["recruiting_composite"] = np.nan
    attrs["recruit_rank_national"] = np.nan
    attrs["recruit_rank_position"] = np.nan

    if not recruiting_cache.exists() or args.refresh_recruiting:
        logger.info("Fetching recruiting data from CFBD (%d–%d)...", start, end)
        try:
            recruiting_df = fetch_recruiting_data(start, end)
            if not recruiting_df.empty:
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                recruiting_df.to_parquet(recruiting_cache, index=False)
                logger.info("Cached %d recruiting rows → %s", len(recruiting_df), recruiting_cache)
        except Exception as exc:
            logger.warning("Could not fetch recruiting data: %s", exc)
            recruiting_df = pd.DataFrame()
    else:
        logger.info("Loading recruiting data from cache → %s", recruiting_cache)
        recruiting_df = pd.read_parquet(recruiting_cache)

    if not recruiting_df.empty and "player_id" in attrs.columns:
        # Best match: player_name + school. Take the highest-rated year per player name+school.
        agg_cols = ["player_name", "school", "stars", "recruiting_composite"]
        if "recruit_rank_national" in recruiting_df.columns:
            agg_cols.append("recruit_rank_national")
        if "recruit_rank_position" in recruiting_df.columns:
            agg_cols.append("recruit_rank_position")
        rec_agg = (
            recruiting_df.sort_values("recruiting_composite", ascending=False)
            .dropna(subset=["player_name", "school"])
            .groupby(["player_name", "school"], as_index=False)
            .first()[agg_cols]
        )
        attrs = attrs.merge(rec_agg, on=["player_name", "school"], how="left",
                            suffixes=("", "_rec"))
        if "stars_rec" in attrs.columns:
            attrs["recruiting_stars"] = attrs["stars_rec"].combine_first(attrs["recruiting_stars"])
            attrs = attrs.drop(columns=["stars_rec"])
        elif "stars" in attrs.columns and "recruiting_stars" in attrs.columns:
            attrs["recruiting_stars"] = attrs["stars"].combine_first(attrs["recruiting_stars"])
            attrs = attrs.drop(columns=["stars"], errors="ignore")
        if "recruiting_composite_rec" in attrs.columns:
            attrs["recruiting_composite"] = attrs["recruiting_composite_rec"].combine_first(attrs["recruiting_composite"])
            attrs = attrs.drop(columns=["recruiting_composite_rec"])
        elif "recruiting_composite" not in attrs.columns:
            attrs["recruiting_composite"] = np.nan
        # Same combine_first pattern for the new rank columns.
        for rank_col in ("recruit_rank_national", "recruit_rank_position"):
            rec_col = f"{rank_col}_rec"
            if rec_col in attrs.columns:
                attrs[rank_col] = attrs[rec_col].combine_first(attrs[rank_col])
                attrs = attrs.drop(columns=[rec_col])

        n_rec = attrs["recruiting_stars"].notna().sum()
        logger.info("Recruiting stars resolved for %d / %d players", n_rec, len(attrs))
        if n_rec > 0:
            logger.info("  Stars dist: %s", attrs["recruiting_stars"].value_counts().to_dict())
        n_rank = attrs["recruit_rank_national"].notna().sum()
        if n_rank > 0:
            logger.info("Recruiting national rank resolved for %d players", n_rank)

    # ── Output ──────────────────────────────────────────────────────────────────
    # Step-3-only extras that don't propagate downstream (just stored for
    # debugging / age estimation provenance).
    _step3_extras = ["season_year", "birth_year_estimated",
                     "age_at_last_college_season"]
    keep_cols = [c for c in (list(ATTRIBUTE_MERGE_COLS) + _step3_extras)
                 if c in attrs.columns]

    attrs = attrs[keep_cols]

    # Invariant: one attribute row per player. Duplicates here mean a player's
    # college career would carry contradictory labels downstream.
    if not attrs["player_id"].is_unique:
        dupes = attrs.loc[attrs["player_id"].duplicated(), "player_id"].unique()
        raise AssertionError(
            f"player_attributes has duplicate player_ids: {dupes[:10]}")

    out_path = PROCESSED_DIR / "player_attributes.parquet"
    attrs.to_parquet(out_path, index=False)
    logger.info("Saved %d player attribute rows → %s", len(attrs), out_path)

    # Position group breakdown
    for grp, cnt in attrs["position_group"].value_counts().items():
        logger.info("  %s: %d players", grp, cnt)

    if "age_at_last_college_season" in attrs.columns:
        age_valid = attrs["age_at_last_college_season"].dropna()
        if not age_valid.empty:
            age_valid_clipped = age_valid[(age_valid >= 17) & (age_valid <= 30)]
            logger.info("Age at last college season: mean=%.1f, range=%d–%d (n=%d)",
                        age_valid_clipped.mean(), int(age_valid_clipped.min()),
                        int(age_valid_clipped.max()), len(age_valid_clipped))

    logger.info("Step 3 complete.")


if __name__ == "__main__":
    main()
