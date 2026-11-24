"""
Step 1: Data Ingestion

Fetches and saves:
  - player_seasons.parquet   — per-player per-season stats (CFBD) [kept for Steps 2–3]
  - team_ratings.parquet     — SP+ ratings per team per season
  - team_conferences.parquet — per-season FBS conference affiliation (historical)
  - team_advanced.parquet    — season-level advanced stats
  - draft_picks.parquet      — NFL draft picks (seeding bridge players)
  - player_gamelogs.parquet  — per-game player stats with rolling opponent strength
                               and snap normalization (for Steps 4–6)

Usage:
    python pipeline/01_data_ingestion.py [--test]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    START_YEAR, END_YEAR, RAW_DIR, PROCESSED_DIR,
    SNAP_COUNTING_STATS, SNAP_PROXY_BY_POSITION, CONSTANT_SNAP_POSITIONS,
    RATE_STAT_ATTEMPT_COL,
    KALMAN_REF_SNAPS, OPP_ROLLING_HALFLIFE_DAYS, OPP_ROLLING_FEATURES,
    OPP_ROLLING_DEMEAN_COLS, POSITION_TO_GROUP,
    SCHOOL_TIMEZONE_OFFSET, DEFAULT_TIMEZONE_OFFSET,
    CONFERENCE_TIERS, DEFAULT_CONFERENCE_TIER,
    KNOWN_SEASON_EXCLUSIONS,
    QB_ROLLING_HALFLIFE_DAYS, QB_ROLLING_MIN_PASS_ATT,
    QB_ROLLING_FEATURES, QB_ROLLING_DEMEAN_COLS,
    TEAM_ROLLING_HALFLIFE_DAYS, TEAM_ROLLING_MIN_PLAYS,
    TEAM_ROLLING_FEATURES, TEAM_ROLLING_DEMEAN_COLS,
    OPP_DEF_ALLOWED_HALFLIFE_DAYS, OPP_DEF_ALLOWED_RAW_STATS,
    OPP_DEF_ALLOWED_FEATURES, OPP_DEF_ALLOWED_DEMEAN_COLS,
    TEAM_SCORING_ENV_RAW_STATS, TEAM_SCORING_ENV_FEATURES,
    TEAM_SCORING_ENV_DEMEAN_COLS,
    OPP_OFF_VOLUME_HALFLIFE_DAYS, OPP_OFF_VOLUME_MIN_PLAYS,
    OPP_OFF_VOLUME_FEATURES, OPP_OFF_VOLUME_DEMEAN_COLS,
    OPP_QB_ROLLING_FEATURES, OPP_QB_ROLLING_DEMEAN_COLS,
    DEF_ROLLING_HALFLIFE_DAYS, DEF_ROLLING_MIN_OPP_PASS_ATT,
    DEF_ROLLING_MIN_OPP_PLAYS, DEF_ROLLING_FEATURES,
    DEF_ROLLING_DEMEAN_COLS,
    STADIUM_SURFACE_OVERRIDES,
    ALTITUDE_DIFF_BUCKET_FT,
    CAPACITY_DIFF_BUCKET,
    TRAVEL_DISTANCE_BUCKET_MI,
    COMPUTE_REVERTED_CONTEXT,
)
from src.scrapers.cfbd import (
    fetch_player_season_stats,
    fetch_team_ratings,
    fetch_team_conferences,
    fetch_team_advanced_stats,
    fetch_draft_picks,
    fetch_player_game_stats,
    fetch_game_schedule,
    fetch_advanced_game_stats,
    fetch_game_weather,
    fetch_advanced_box_scores,
    fetch_venues,
    compute_rolling_opponent_strength,
    compute_qb_rolling_stats,
    compute_team_rolling_context,
    compute_rolling_opp_defense_allowed,
    compute_team_rolling_scoring_env,
    compute_rolling_opp_offense_volume,
    compute_defense_rolling_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Fetch only 2 recent seasons")
    parser.add_argument("--skip-gamelogs", action="store_true",
                        help="Skip game-level data fetch (use existing gamelogs)")
    parser.add_argument("--assemble-only", action="store_true",
                        help="Skip all API fetches; reload existing raw parquets and "
                             "reassemble player_gamelogs.parquet")
    parser.add_argument("--fetch-box-scores", action="store_true",
                        help="Fetch advanced box scores (snap counts + PPA). "
                             "~1 API call per game; ~19K calls for full history. "
                             "Uses cached parquet files if present unless this flag is set.")
    parser.add_argument("--fetch-venues", action="store_true",
                        help="Force re-fetch venues.parquet (one CFBD call, "
                             "~840 rows). Cached by default; pass this flag "
                             "to refresh after stadium upgrades / new venues.")
    return parser.parse_args()


def _compute_snap_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Compute snap proxy per player-game when actual snap counts are unavailable.
    Uses position-specific formulas from SNAP_PROXY_BY_POSITION.
    """
    snaps = pd.Series(np.nan, index=df.index)

    # Specialists get a constant denominator — their own volume stats are the
    # numerators, so using them as the proxy is self-referential (see
    # CONSTANT_SNAP_POSITIONS). Applied before the proxy loop; the constant
    # makes the downstream Kalman observation weight exactly 1.0 per game.
    if "position_group" in df.columns and CONSTANT_SNAP_POSITIONS:
        const_mask = df["position_group"].isin(CONSTANT_SNAP_POSITIONS)
        if const_mask.any():
            snaps.loc[const_mask] = KALMAN_REF_SNAPS

    for pos_group, proxy_cols in SNAP_PROXY_BY_POSITION.items():
        if not proxy_cols:
            continue
        mask = df["position_group"] == pos_group if "position_group" in df.columns \
            else pd.Series(True, index=df.index)
        if not mask.any():
            continue
        available = [c for c in proxy_cols if c in df.columns]
        if available:
            proxy = df.loc[mask, available].fillna(0).sum(axis=1).clip(lower=1)
            snaps.loc[mask] = proxy

    # Fallback for unknowns: use any available counting stat sum
    still_nan = snaps.isna()
    if still_nan.any():
        fallback_cols = [c for c in ["pass_att", "rush_att", "rec", "tackles"]
                         if c in df.columns]
        if fallback_cols:
            snaps.loc[still_nan] = df.loc[still_nan, fallback_cols].fillna(0).sum(axis=1).clip(lower=1)

    return snaps.fillna(1).clip(lower=1)


def _join_box_score_player_data(gamelogs: pd.DataFrame,
                                  player_box_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Join actual snap counts and PPA from advanced box scores to gamelogs.
    Matches on player_name + school + game_id. Adds actual_snaps column
    (used by _normalize_gamelogs to replace the proxy) and PPA columns.
    """
    if player_box_scores.empty:
        return gamelogs

    bs = player_box_scores.rename(columns={"team": "school"})
    data_cols = [c for c in [
        "snaps", "snaps_rushing", "snaps_passing",
        "snaps_q1", "snaps_q2", "snaps_q3", "snaps_q4",
        "ppa_avg", "ppa_avg_rushing", "ppa_avg_passing",
        "ppa_cum", "ppa_cum_rushing", "ppa_cum_passing",
    ] if c in bs.columns]

    bs_join = bs[["game_id", "player_name", "school"] + data_cols].rename(
        columns={"snaps": "actual_snaps"}
    )

    df = gamelogs.merge(bs_join, on=["game_id", "player_name", "school"], how="left")

    n_snaps = df["actual_snaps"].notna().sum() if "actual_snaps" in df.columns else 0
    logger.info("Box score join: %d / %d player-games have actual snap counts", n_snaps, len(df))
    return df


def _drop_opponent_side_phantom_rows(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that credit one player_id to BOTH teams of the same game.

    CFBD occasionally emits a second, statless row for a player on the OPPOSING
    roster of a single game. As of 2026-07-26 there are exactly 5 such
    (player_id, game_id) pairs in 894k rows:

        Kerry Thomas Jr. 3124042   UTSA 35 rows  / Southern Miss 1
        Malachi Jones    4431185   Tulsa 34      / Houston 1
        Evan Silva       4570184   UNLV 13       / Fresno State 1
        Cody Parkey      504119    Auburn 36     / LSU 1
        D'Angelo Roberts 513035    Indiana 44    / Ohio State 1

    The 2026-07-24 audit filed this as "two real people merged into one career",
    but the shape says otherwise: the minority row is a singleton with NO
    production (all-NaN or explicit zeros, carrying only a snap proxy), while the
    majority row holds the real stat line. A genuine namesake would have their own
    multi-game career on the other roster, not one empty appearance. So this is a
    phantom row to delete, not a career to split — which makes the fix far safer
    than the audit assumed.

    Left in place the row would (a) add a ~zero game to the player's Kalman/EWMA
    streams, dragging the estimate down, (b) inflate n_games by one, and (c) worst,
    attribute the game to the wrong school, so every opponent/venue/team-context
    join for that row resolves against the opposing side.

    Rule: keep the row whose school is the player's modal school for that SEASON;
    drop the others. Season-modal rather than career-modal so a mid-career
    transfer is still handled correctly. Ties (a genuine 1-vs-1 with no majority)
    are left ALONE rather than guessed at — the log names them so they can be
    inspected instead of silently resolved the wrong way.
    """
    need = {"player_id", "game_id", "school"}
    if gamelogs.empty or not need.issubset(gamelogs.columns):
        return gamelogs

    key = ["player_id", "game_id"]
    n_schools = gamelogs.groupby(key)["school"].transform("nunique")
    contaminated = n_schools > 1
    if not contaminated.any():
        return gamelogs

    season_col = "season_year" if "season_year" in gamelogs.columns else None
    grp_cols = ["player_id"] + ([season_col] if season_col else [])
    # Row count per (player, season, school) → the player's real school that year.
    counts = (gamelogs.groupby(grp_cols + ["school"]).size()
              .rename("_school_rows").reset_index())
    gl = gamelogs.merge(counts, on=grp_cols + ["school"], how="left")
    gl["_max_rows"] = gl.groupby(key)["_school_rows"].transform("max")

    # Keep uncontaminated rows, plus contaminated rows that sit on the majority
    # school. A tie leaves every row's _school_rows == _max_rows, so nothing is
    # dropped and the pair survives for inspection.
    contaminated_gl = gl.groupby(key)["school"].transform("nunique") > 1
    drop_mask = contaminated_gl & (gl["_school_rows"] < gl["_max_rows"])

    ties = gl[contaminated_gl & (gl["_school_rows"] == gl["_max_rows"])]
    tie_pairs = ties.groupby(key).size()
    tie_pairs = tie_pairs[tie_pairs > 1]
    if len(tie_pairs):
        logger.warning(
            "%d (player_id, game_id) pair(s) credited to two schools with NO "
            "majority — left in place, inspect manually: %s",
            len(tie_pairs), tie_pairs.index.tolist()[:10])

    if drop_mask.any():
        dropped = gl.loc[drop_mask, ["player_id", "school", "game_id"]]
        logger.info("Dropped %d phantom opponent-side gamelog row(s) across %d "
                    "player(s): %s", int(drop_mask.sum()),
                    dropped["player_id"].nunique(),
                    ", ".join(f"{r.player_id}@{r.school}"
                              for r in dropped.head(10).itertuples()))
    out = gl.loc[~drop_mask].drop(columns=["_school_rows", "_max_rows"])
    return out.reset_index(drop=True)


def _normalize_gamelogs(gamelogs: pd.DataFrame,
                         attrs: pd.DataFrame) -> pd.DataFrame:
    """
    1. Join position_group from player attributes.
    2. Compute snap proxy.
    3. Create per-100-snap versions of counting stats (stored as {stat}_p100).
    """
    df = gamelogs.copy()

    # Join position group
    if not attrs.empty and "player_id" in df.columns and "player_id" in attrs.columns:
        pos_map = attrs.set_index("player_id")["position_group"].to_dict()
        df["position_group"] = df["player_id"].map(pos_map)

    if "position_group" not in df.columns or df["position_group"].isna().all():
        if "position" in df.columns:
            df["position_group"] = df["position"].str.upper().map(POSITION_TO_GROUP).fillna("OTHER")
        else:
            df["position_group"] = "OTHER"

    # Compute snap proxy, then replace with actual snaps where available.
    # CFBD `actual_snaps` is a *share* (0–1) of side-specific team plays, so we
    # convert to a snap *count* using box-score team plays (analogous to
    # KenPom's possession formula): plays = team_pass_att + team_rush_att.
    # Offensive players use their own team's plays; defenders use opponent's
    # (looked up via game_id). K/P keep their proxy — actual_snaps only
    # measures regular-down usage and ignores special-teams reps.
    df["snaps"] = _compute_snap_proxy(df)
    df["snaps_are_real"] = False
    # Note: snap-share→count conversion (when actual_snaps is present) and the
    # _p100 computation are deferred until after team_pass_att/team_rush_att
    # are reconstructed from per-player aggregates below.

    # Ensure rate stats are present (may already be computed by fetch_player_game_stats)
    def _safe_div(n_col: str, d_col: str) -> pd.Series:
        n = df[n_col].astype(float) if n_col in df.columns else pd.Series(np.nan, index=df.index)
        d = df[d_col].astype(float) if d_col in df.columns else pd.Series(np.nan, index=df.index)
        return n.where(d > 0) / d.where(d > 0)

    if "completion_pct" not in df.columns:
        df["completion_pct"] = _safe_div("completions", "pass_att")
    if "yards_per_carry" not in df.columns:
        df["yards_per_carry"] = _safe_div("rush_yards", "rush_att")
    if "yards_per_rec" not in df.columns:
        df["yards_per_rec"] = _safe_div("rec_yards", "rec")
    if "fg_pct" not in df.columns:
        df["fg_pct"] = _safe_div("fg_made", "fg_att")
    if "xp_pct" not in df.columns:
        df["xp_pct"] = _safe_div("xp_made", "xp_att")
    if "net_punt_yards" not in df.columns and "punt_yards" in df.columns:
        df["net_punt_yards"] = df["punt_yards"]  # CFBD doesn't split net/gross at game level

    # ── Team-game totals and share features ────────────────────────────────────
    # Reconstructed by summing all player contributions per team-game.
    # Captures role prominence (share of team offense) as a draft-relevant signal.
    if "game_id" in df.columns and "school" in df.columns:
        share_src = {
            "team_rec":       "rec",
            "team_rush_att":  "rush_att",
            "team_pass_att":  "pass_att",
        }
        available_src = {k: v for k, v in share_src.items() if v in df.columns}
        if available_src:
            team_totals = (
                df.groupby(["game_id", "school"])[list(available_src.values())]
                .sum()
                .rename(columns={v: k for k, v in available_src.items()})
                .reset_index()
            )
            df = df.merge(team_totals, on=["game_id", "school"], how="left")
            if "team_rec" in df.columns and "rec" in df.columns:
                df["rec_share"] = (df["rec"] / df["team_rec"].clip(lower=1)).clip(0, 1)
            if "team_rush_att" in df.columns and "rush_att" in df.columns:
                df["rush_share"] = (df["rush_att"] / df["team_rush_att"].clip(lower=1)).clip(0, 1)
            if "team_pass_att" in df.columns and "pass_att" in df.columns:
                df["pass_share"] = (df["pass_att"] / df["team_pass_att"].clip(lower=1)).clip(0, 1)

    # ── Snap share → snap count conversion (when actual_snaps available) ──────
    # CFBD `actual_snaps` is a *share* (0–1) of side-specific team plays.
    # Convert to a count using box-score team plays (analogous to KenPom's
    # possession formula): plays = team_pass_att + team_rush_att.
    # Offensive players use their own team's plays; defenders use opponent's
    # (looked up via game_id). K/P keep their proxy.
    have_team_plays = ({"team_pass_att", "team_rush_att",
                        "game_id", "school"}.issubset(df.columns))
    if "actual_snaps" in df.columns and have_team_plays:
        own_plays = df["team_pass_att"].fillna(0) + df["team_rush_att"].fillna(0)
        # team_pass_att/team_rush_att repeat per player — dedupe by (game,team)
        # before summing across teams to get the per-game total.
        per_team = (df[["game_id", "school", "team_pass_att", "team_rush_att"]]
                    .drop_duplicates(["game_id", "school"]))
        game_total = (per_team.groupby("game_id")[["team_pass_att", "team_rush_att"]]
                      .sum().sum(axis=1))
        opp_plays = df["game_id"].map(game_total).fillna(0) - own_plays

        OFFENSE = {"QB", "RB", "WR", "TE", "OL"}
        DEFENSE = {"DL", "EDGE", "LB", "DB"}

        has_actual = df["actual_snaps"].notna() & (df["actual_snaps"] > 0)
        off_mask = has_actual & df["position_group"].isin(OFFENSE) & (own_plays > 0)
        def_mask = has_actual & df["position_group"].isin(DEFENSE) & (opp_plays > 0)

        df.loc[off_mask, "snaps"] = df.loc[off_mask, "actual_snaps"] * own_plays.loc[off_mask]
        df.loc[def_mask, "snaps"] = df.loc[def_mask, "actual_snaps"] * opp_plays.loc[def_mask]
        df.loc[off_mask | def_mask, "snaps_are_real"] = True
        df["snaps"] = df["snaps"].clip(lower=1.0)
        logger.info("Using actual snaps (share×team_plays) for %d offense + %d defense / %d player-games",
                    off_mask.sum(), def_mask.sum(), len(df))

    # Create per-100-snap columns for counting stats (using final snap counts)
    for stat in SNAP_COUNTING_STATS:
        if stat in df.columns:
            df[f"{stat}_p100"] = df[stat] * 100.0 / df["snaps"]

    return df


def _add_opponent_rolling_stats(gamelogs: pd.DataFrame,
                                  adv_game: pd.DataFrame,
                                  schedule: pd.DataFrame,
                                  halflife_days: int = OPP_ROLLING_HALFLIFE_DAYS,
                                  team_box_scores: pd.DataFrame = None,
                                  ) -> pd.DataFrame:
    """
    Compute and join rolling opponent strength features.

    De-meaning mirrors cdarko: only composite PPA stats are de-meaned within
    each player-season (removes schedule-strength temporal confound). Four-factor
    components (success rate, stuff rate, explosiveness) are left raw because
    they carry legitimate within-season signal.
    """
    if adv_game.empty or schedule.empty:
        logger.warning("No advanced game stats or schedule — skipping rolling opponent stats")
        return gamelogs

    # Merge havoc from team box scores into adv_game so rolling computation picks it up
    if team_box_scores is not None and not team_box_scores.empty:
        havoc_cols = [c for c in ["game_id", "team", "havoc_total",
                                   "havoc_front_seven", "havoc_db"]
                      if c in team_box_scores.columns]
        if len(havoc_cols) > 2:
            adv_game = adv_game.merge(team_box_scores[havoc_cols],
                                      on=["game_id", "team"], how="left")
            logger.info("Merged havoc data into adv_game for rolling computation")

    logger.info("Computing rolling opponent strength (halflife=%d days)...", halflife_days)
    rolling = compute_rolling_opponent_strength(adv_game, schedule, halflife_days)

    if rolling.empty:
        return gamelogs

    rolling = rolling.rename(columns={"team": "school"})
    opp_cols = [c for c in OPP_ROLLING_FEATURES if c in rolling.columns]

    df = gamelogs.merge(
        rolling[["game_id", "school"] + opp_cols],
        on=["game_id", "school"], how="left",
    )

    # De-mean composite PPA features only (cdarko pattern)
    demean_cols = [c for c in OPP_ROLLING_DEMEAN_COLS if c in df.columns]
    if demean_cols and "player_id" in df.columns and "season_year" in df.columns:
        season_means = df.groupby(["player_id", "season_year"])[demean_cols].transform("mean")
        df[demean_cols] = df[demean_cols] - season_means

    logger.info("Joined rolling opponent stats to %d game rows", len(df))
    return df


def _add_qb_rolling_stats(gamelogs: pd.DataFrame,
                          halflife_days: int = QB_ROLLING_HALFLIFE_DAYS,
                          min_pass_att: int = QB_ROLLING_MIN_PASS_ATT,
                          ) -> pd.DataFrame:
    """
    Compute and join teammate-QB rolling quality features. Mirrors
    `_add_opponent_rolling_stats` shape: build a (game_id, school)-keyed
    panel via `compute_qb_rolling_stats`, merge onto the gamelog, then
    within-(player, season) demean the composite/tempo cols listed in
    `QB_ROLLING_DEMEAN_COLS` (same rationale as opp_rolling: removes the
    schedule-strength temporal confound from the composite rating signal).
    """
    if gamelogs.empty or "school" not in gamelogs.columns:
        return gamelogs

    logger.info("Computing teammate-QB rolling quality "
                "(halflife=%d days, min_pass_att=%d)...",
                halflife_days, min_pass_att)
    qb_roll = compute_qb_rolling_stats(
        gamelogs, halflife_days=halflife_days, min_pass_att=min_pass_att,
    )
    if qb_roll.empty:
        logger.warning("compute_qb_rolling_stats returned empty — "
                       "skipping qb_rolling_* join")
        return gamelogs

    qb_cols = [c for c in QB_ROLLING_FEATURES if c in qb_roll.columns]
    df = gamelogs.merge(
        qb_roll[["game_id", "school"] + qb_cols],
        on=["game_id", "school"], how="left",
    )

    # De-mean composite cols within (player, season) — mirrors OPP_ROLLING.
    demean_cols = [c for c in QB_ROLLING_DEMEAN_COLS if c in df.columns]
    if (demean_cols and "player_id" in df.columns
            and "season_year" in df.columns):
        season_means = df.groupby(["player_id", "season_year"]
                                  )[demean_cols].transform("mean")
        df[demean_cols] = df[demean_cols] - season_means

    coverage = df[qb_cols[0]].notna().sum() if qb_cols else 0
    logger.info("Joined qb_rolling_* (%d cols) to %d game rows "
                "(%d / %d non-null in %s)",
                len(qb_cols), len(df), coverage, len(df), qb_cols[0])
    return df


def _add_team_rolling_context(gamelogs: pd.DataFrame,
                               team_box_scores: pd.DataFrame,
                               halflife_days: int = TEAM_ROLLING_HALFLIFE_DAYS,
                               min_plays: int = TEAM_ROLLING_MIN_PLAYS,
                               ) -> pd.DataFrame:
    """
    Compute and join own-team rolling context features (pace, pass rate, OL
    run-block). Same shape as `_add_qb_rolling_stats`: (game_id, school)-keyed
    panel from `compute_team_rolling_context`, merge onto the gamelog, then
    within-(player, season) demean the cols in `TEAM_ROLLING_DEMEAN_COLS`
    (empty by default — pace/pass_rate are scheme signals we want raw).
    """
    if gamelogs.empty or "school" not in gamelogs.columns:
        return gamelogs

    required = {"game_id", "school", "game_date",
                "team_pass_att", "team_rush_att"}
    if not required.issubset(gamelogs.columns):
        logger.warning("_add_team_rolling_context: missing required cols %s "
                       "— skipping team_rolling_* join",
                       required - set(gamelogs.columns))
        return gamelogs

    logger.info("Computing team rolling context "
                "(halflife=%d days, min_plays=%d)...",
                halflife_days, min_plays)
    team_roll = compute_team_rolling_context(
        gamelogs, team_box_scores,
        halflife_days=halflife_days, min_plays=min_plays,
    )
    if team_roll.empty:
        logger.warning("compute_team_rolling_context returned empty — "
                       "skipping team_rolling_* join")
        return gamelogs

    team_cols = [c for c in TEAM_ROLLING_FEATURES if c in team_roll.columns]
    df = gamelogs.merge(
        team_roll[["game_id", "school"] + team_cols],
        on=["game_id", "school"], how="left",
    )

    demean_cols = [c for c in TEAM_ROLLING_DEMEAN_COLS if c in df.columns]
    if (demean_cols and "player_id" in df.columns
            and "season_year" in df.columns):
        season_means = (df.groupby(["player_id", "season_year"])[demean_cols]
                          .transform("mean"))
        df[demean_cols] = df[demean_cols] - season_means

    coverage = df[team_cols[0]].notna().sum() if team_cols else 0
    logger.info("Joined team_rolling_* (%d cols) to %d game rows "
                "(%d / %d non-null in %s)",
                len(team_cols), len(df), coverage, len(df), team_cols[0])
    return df


def _add_opp_defense_allowed(gamelogs: pd.DataFrame,
                              team_box_scores: pd.DataFrame,
                              schedule: pd.DataFrame,
                              halflife_days: int = OPP_DEF_ALLOWED_HALFLIFE_DAYS,
                              ) -> pd.DataFrame:
    """
    Opponent's defense — rolling stats their defense has ALLOWED, derived
    from team_box_scores (offensive stats produced against that defense).

    Joins (game_id, school) keyed panel of `opp_rolling_def_<stat>` columns
    onto gamelogs. Demeaning policy mirrors OPP_ROLLING: composite-rate cols
    can be listed in OPP_DEF_ALLOWED_DEMEAN_COLS; four-factor splits stay raw.
    """
    if (gamelogs.empty or team_box_scores.empty or schedule.empty
            or "school" not in gamelogs.columns):
        return gamelogs

    logger.info("Computing opponent-defense allowed rolling "
                "(halflife=%d days, %d stats)...",
                halflife_days, len(OPP_DEF_ALLOWED_RAW_STATS))
    rolled = compute_rolling_opp_defense_allowed(
        team_box_scores, schedule,
        stat_cols=OPP_DEF_ALLOWED_RAW_STATS,
        halflife_days=halflife_days,
    )
    if rolled.empty:
        logger.warning("compute_rolling_opp_defense_allowed returned empty — "
                       "skipping opp_rolling_def_* join")
        return gamelogs

    opp_def_cols = [c for c in OPP_DEF_ALLOWED_FEATURES if c in rolled.columns]
    df = gamelogs.merge(
        rolled[["game_id", "school"] + opp_def_cols],
        on=["game_id", "school"], how="left",
    )

    demean_cols = [c for c in OPP_DEF_ALLOWED_DEMEAN_COLS if c in df.columns]
    if (demean_cols and "player_id" in df.columns
            and "season_year" in df.columns):
        season_means = (df.groupby(["player_id", "season_year"])[demean_cols]
                          .transform("mean"))
        df[demean_cols] = df[demean_cols] - season_means

    coverage = df[opp_def_cols[0]].notna().sum() if opp_def_cols else 0
    logger.info("Joined opp_rolling_def_* (%d cols) to %d game rows "
                "(%d / %d non-null in %s)",
                len(opp_def_cols), len(df), coverage, len(df),
                opp_def_cols[0] if opp_def_cols else "<none>")
    return df


def _add_defense_rolling_stats(gamelogs: pd.DataFrame,
                                halflife_days: int = DEF_ROLLING_HALFLIFE_DAYS,
                                min_opp_pass_att: int = DEF_ROLLING_MIN_OPP_PASS_ATT,
                                min_opp_plays: int = DEF_ROLLING_MIN_OPP_PLAYS,
                                ) -> pd.DataFrame:
    """
    Own-defense rolling pressure / coverage rate features for defenders.
    Coverage starts ~2018 (CFBD defensive box-score availability); pre-2018
    games carry NaN, handled downstream by Ridge fillna(0).
    """
    if gamelogs.empty or "school" not in gamelogs.columns:
        return gamelogs

    logger.info("Computing defensive teammate rolling "
                "(halflife=%d days, min_opp_pass_att=%d, min_opp_plays=%d)...",
                halflife_days, min_opp_pass_att, min_opp_plays)
    rolled = compute_defense_rolling_stats(
        gamelogs, halflife_days=halflife_days,
        min_opp_pass_att=min_opp_pass_att,
        min_opp_plays=min_opp_plays,
    )
    if rolled.empty:
        logger.warning("compute_defense_rolling_stats returned empty — "
                       "skipping def_rolling_* join")
        return gamelogs

    def_cols = [c for c in DEF_ROLLING_FEATURES if c in rolled.columns]
    df = gamelogs.merge(
        rolled[["game_id", "school"] + def_cols],
        on=["game_id", "school"], how="left",
    )

    demean_cols = [c for c in DEF_ROLLING_DEMEAN_COLS if c in df.columns]
    if (demean_cols and "player_id" in df.columns
            and "season_year" in df.columns):
        season_means = (df.groupby(["player_id", "season_year"])[demean_cols]
                          .transform("mean"))
        df[demean_cols] = df[demean_cols] - season_means

    coverage = df[def_cols[0]].notna().sum() if def_cols else 0
    logger.info("Joined def_rolling_* (%d cols) to %d game rows "
                "(%d / %d non-null in %s — pre-2018 NaN expected)",
                len(def_cols), len(df), coverage, len(df),
                def_cols[0] if def_cols else "<none>")
    return df


def _add_opp_offense_volume(gamelogs: pd.DataFrame,
                             halflife_days: int = OPP_OFF_VOLUME_HALFLIFE_DAYS,
                             min_plays: int = OPP_OFF_VOLUME_MIN_PLAYS,
                             ) -> pd.DataFrame:
    """
    Rolling EWMA of opponent's offensive VOLUME (pass_att, rush_att, plays)
    over the opponent's strictly-prior games. Aimed at defenders — gives
    opportunity-side context that opponent QUALITY metrics can't supply.
    """
    if gamelogs.empty or "school" not in gamelogs.columns:
        return gamelogs
    required = {"game_id", "school", "opponent", "game_date",
                "team_pass_att", "team_rush_att"}
    if not required.issubset(gamelogs.columns):
        logger.warning("_add_opp_offense_volume: missing required cols %s",
                       required - set(gamelogs.columns))
        return gamelogs

    logger.info("Computing opponent-offense volume rolling "
                "(halflife=%d days, min_plays=%d)...",
                halflife_days, min_plays)
    rolled = compute_rolling_opp_offense_volume(
        gamelogs, halflife_days=halflife_days, min_plays=min_plays,
    )
    if rolled.empty:
        logger.warning("compute_rolling_opp_offense_volume returned empty — "
                       "skipping opp_rolling_off_* (volume) join")
        return gamelogs

    vol_cols = [c for c in OPP_OFF_VOLUME_FEATURES if c in rolled.columns]
    df = gamelogs.merge(
        rolled[["game_id", "school"] + vol_cols],
        on=["game_id", "school"], how="left",
    )

    demean_cols = [c for c in OPP_OFF_VOLUME_DEMEAN_COLS if c in df.columns]
    if (demean_cols and "player_id" in df.columns
            and "season_year" in df.columns):
        season_means = (df.groupby(["player_id", "season_year"])[demean_cols]
                          .transform("mean"))
        df[demean_cols] = df[demean_cols] - season_means

    coverage = df[vol_cols[0]].notna().sum() if vol_cols else 0
    logger.info("Joined opp_rolling_off_* volume (%d cols) to %d game rows "
                "(%d / %d non-null in %s)",
                len(vol_cols), len(df), coverage, len(df),
                vol_cols[0] if vol_cols else "<none>")
    return df


def _add_opp_qb_rolling_stats(gamelogs: pd.DataFrame,
                               halflife_days: int = QB_ROLLING_HALFLIFE_DAYS,
                               min_pass_att: int = QB_ROLLING_MIN_PASS_ATT,
                               ) -> pd.DataFrame:
    """
    Mirror of `_add_qb_rolling_stats` but joined on OPPONENT — gives
    defenders rolling context on the QB they're about to face. Reuses
    `compute_qb_rolling_stats` output (per (game_id, school) rolling rate
    stats) and merges via (game_id, opponent) instead of (game_id, school).
    """
    if gamelogs.empty or "opponent" not in gamelogs.columns:
        return gamelogs

    logger.info("Computing opponent-QB rolling quality "
                "(halflife=%d days, min_pass_att=%d)...",
                halflife_days, min_pass_att)
    qb_roll = compute_qb_rolling_stats(
        gamelogs, halflife_days=halflife_days, min_pass_att=min_pass_att,
    )
    if qb_roll.empty:
        logger.warning("compute_qb_rolling_stats returned empty — "
                       "skipping opp_qb_rolling_* join")
        return gamelogs

    rename_map = {"school": "opponent"}
    rename_map.update({
        c: c.replace("qb_rolling_", "opp_qb_rolling_")
        for c in qb_roll.columns if c.startswith("qb_rolling_")
    })
    opp_qb = qb_roll.rename(columns=rename_map)
    opp_qb_cols = [c for c in OPP_QB_ROLLING_FEATURES if c in opp_qb.columns]

    df = gamelogs.merge(
        opp_qb[["game_id", "opponent"] + opp_qb_cols],
        on=["game_id", "opponent"], how="left",
    )

    demean_cols = [c for c in OPP_QB_ROLLING_DEMEAN_COLS if c in df.columns]
    if (demean_cols and "player_id" in df.columns
            and "season_year" in df.columns):
        season_means = (df.groupby(["player_id", "season_year"])[demean_cols]
                          .transform("mean"))
        df[demean_cols] = df[demean_cols] - season_means

    coverage = df[opp_qb_cols[0]].notna().sum() if opp_qb_cols else 0
    logger.info("Joined opp_qb_rolling_* (%d cols) to %d game rows "
                "(%d / %d non-null in %s)",
                len(opp_qb_cols), len(df), coverage, len(df),
                opp_qb_cols[0] if opp_qb_cols else "<none>")
    return df


def _add_team_scoring_env(gamelogs: pd.DataFrame,
                           team_box_scores: pd.DataFrame,
                           halflife_days: int = TEAM_ROLLING_HALFLIFE_DAYS,
                           min_plays: int = TEAM_ROLLING_MIN_PLAYS,
                           ) -> pd.DataFrame:
    """
    Own-team rolling scoring environment (scoring_opportunities,
    points_per_opportunity, avg_starting_field_pos) from team_box_scores.

    Separate from `_add_team_rolling_context` so we can enable / revert
    independently from the previously-reverted pace/pass_rate/OL features.
    """
    if (gamelogs.empty or team_box_scores.empty
            or "school" not in gamelogs.columns):
        return gamelogs

    required = {"game_id", "school", "game_date",
                "team_pass_att", "team_rush_att"}
    if not required.issubset(gamelogs.columns):
        logger.warning("_add_team_scoring_env: missing required cols %s",
                       required - set(gamelogs.columns))
        return gamelogs

    logger.info("Computing team rolling scoring env "
                "(halflife=%d days, %d stats)...",
                halflife_days, len(TEAM_SCORING_ENV_RAW_STATS))
    rolled = compute_team_rolling_scoring_env(
        gamelogs, team_box_scores,
        stat_cols=TEAM_SCORING_ENV_RAW_STATS,
        halflife_days=halflife_days,
        min_plays=min_plays,
    )
    if rolled.empty:
        logger.warning("compute_team_rolling_scoring_env returned empty — "
                       "skipping team_rolling_scoring_* join")
        return gamelogs

    env_cols = [c for c in TEAM_SCORING_ENV_FEATURES if c in rolled.columns]
    df = gamelogs.merge(
        rolled[["game_id", "school"] + env_cols],
        on=["game_id", "school"], how="left",
    )

    demean_cols = [c for c in TEAM_SCORING_ENV_DEMEAN_COLS if c in df.columns]
    if (demean_cols and "player_id" in df.columns
            and "season_year" in df.columns):
        season_means = (df.groupby(["player_id", "season_year"])[demean_cols]
                          .transform("mean"))
        df[demean_cols] = df[demean_cols] - season_means

    coverage = df[env_cols[0]].notna().sum() if env_cols else 0
    logger.info("Joined team_rolling scoring env (%d cols) to %d game rows "
                "(%d / %d non-null in %s)",
                len(env_cols), len(df), coverage, len(df),
                env_cols[0] if env_cols else "<none>")
    return df


def _compute_school_home_venue(schedule: pd.DataFrame,
                                 venues: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (school, season_year) → (home_elevation_ft, home_grass,
    home_latitude, home_longitude) lookup.

    For each (home_team, season_year), takes the modal venue_id among
    non-neutral home games (almost always one venue per school per season;
    handles split-venue / temporary-stadium edge cases by majority vote).
    Then joins venues.parquet for elevation, grass, and coordinates.
    """
    empty = pd.DataFrame(columns=["school", "season_year",
                                    "home_elevation_ft", "home_grass",
                                    "home_latitude", "home_longitude",
                                    "home_capacity"])
    if schedule.empty or venues.empty:
        return empty
    if "is_neutral" not in schedule.columns or "venue_id" not in schedule.columns:
        return empty

    home = schedule[(schedule["is_neutral"] == 0)].dropna(
        subset=["venue_id", "home_team", "season_year"]
    ).copy()
    if home.empty:
        return empty

    modal = (home.groupby(["home_team", "season_year"])["venue_id"]
                  .agg(lambda s: s.value_counts().idxmax())
                  .reset_index()
                  .rename(columns={"home_team": "school",
                                    "venue_id": "home_venue_id"}))

    venue_lookup = venues[["venue_id", "grass", "elevation_ft",
                            "latitude", "longitude", "capacity"]
                          ].drop_duplicates("venue_id")
    modal = modal.merge(
        venue_lookup, left_on="home_venue_id", right_on="venue_id", how="left",
    ).rename(columns={"elevation_ft": "home_elevation_ft",
                       "grass":        "home_grass",
                       "latitude":     "home_latitude",
                       "longitude":    "home_longitude",
                       "capacity":     "home_capacity"})

    return modal[["school", "season_year",
                  "home_elevation_ft", "home_grass",
                  "home_latitude", "home_longitude", "home_capacity"]]


def _haversine_miles(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in miles. NaN-safe (NaN inputs → NaN out)."""
    R = 3958.756  # Earth's mean radius in miles
    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _add_venue_features(gamelogs: pd.DataFrame,
                          schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Emit identity-free venue context: altitude / surface DIFFERENCES from the
    player's school's home stadium.

    Why diffs instead of raw values: raw altitude is a near-bijective tag for
    school identity in the mountain west (Air Force 7218, Wyoming 7220, BYU
    4553, Utah 4724, Colorado 5345). Raw surface clusters by conference (SEC
    grass, MAC turf). The contextual model could learn "elevation_ft=7218 →
    Air Force-typical production" rather than "thin air helps FG distance via
    physics." Encoding as diff-from-home neutralizes that — HOME games get 0
    (no travel shock), AWAY games get a signed shock that's identity-blind
    once bucketed.

    Sources:
      - `data/raw/venues.parquet`  → current-snapshot `grass` + `elevation_ft`
      - `data/raw/game_schedule.parquet` → `venue_id` per game (joined here
        if not already on gamelogs)
      - `config.STADIUM_SURFACE_OVERRIDES` → historical surface changes that
        the CFBD snapshot misses, applied PER (venue_id, season_year) to BOTH
        the game's grass and the home-stadium grass lookup

    Emits two columns on `gamelogs`:
      - `altitude_diff_from_home_ft` : game_elev - home_elev, rounded to
                                       `ALTITUDE_DIFF_BUCKET_FT` (default 500ft)
      - `surface_diff_from_home`     : signed game_grass - home_grass, in
                                       {-1, 0, +1}; NaN if either side unknown
      - `capacity_diff_from_home`    : game venue seats - home venue seats,
                                       rounded to `CAPACITY_DIFF_BUCKET`
                                       (default 10k); crowd-hostility shock
    """
    def _set_defaults(df: pd.DataFrame) -> pd.DataFrame:
        df["altitude_diff_from_home_ft"] = np.nan
        df["surface_diff_from_home"]     = np.nan
        df["travel_distance_mi"]         = np.nan
        df["capacity_diff_from_home"]    = np.nan
        return df

    venues_path = RAW_DIR / "venues.parquet"
    if not venues_path.exists():
        logger.warning("venues.parquet not found — skipping venue feature join "
                       "(run pipeline/01_data_ingestion.py to fetch)")
        return _set_defaults(gamelogs)

    venues = pd.read_parquet(venues_path)
    if venues.empty or "venue_id" not in venues.columns:
        logger.warning("venues.parquet empty or missing venue_id — skipping")
        return _set_defaults(gamelogs)

    # Ensure venue_id is on gamelogs (merge from schedule if absent)
    if "venue_id" not in gamelogs.columns:
        if schedule.empty or "venue_id" not in schedule.columns:
            logger.warning("Neither gamelogs nor schedule carries venue_id — "
                           "skipping venue feature join")
            return _set_defaults(gamelogs)
        gamelogs = gamelogs.merge(
            schedule[["game_id", "venue_id"]].drop_duplicates("game_id"),
            on="game_id", how="left",
        )

    # Game-side venue lookup (includes lat/lon for travel-distance)
    venue_lookup = venues[["venue_id", "grass", "elevation_ft",
                            "latitude", "longitude", "capacity"]
                          ].drop_duplicates("venue_id")
    df = gamelogs.merge(venue_lookup, on="venue_id", how="left")
    # 70 of 840 venues carry capacity 0 (unknown, not a real empty stadium).
    # Treat as missing so the diff is NaN rather than a fake -100k shock.
    if "capacity" in df.columns:
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
        df.loc[df["capacity"] <= 0, "capacity"] = np.nan

    # Apply surface overrides PER (venue_id, season_year) to the game side.
    # Same lookup will be reused for the home-side join below so the player's
    # home stadium and the game venue both honor the same historical override.
    flat_override: dict[tuple[int, int], bool] = {}
    if STADIUM_SURFACE_OVERRIDES:
        for vid, ranges in STADIUM_SURFACE_OVERRIDES.items():
            for start_year, end_year, surface in ranges:
                grass_val = (str(surface).lower() == "grass")
                for yr in range(int(start_year), int(end_year) + 1):
                    flat_override[(int(vid), int(yr))] = grass_val

    if flat_override and "season_year" in df.columns:
        keys = list(zip(df["venue_id"].astype("Int64"),
                         df["season_year"].astype("Int64")))
        override_vals = pd.Series(
            [flat_override.get((int(v), int(y))
                                if pd.notna(v) and pd.notna(y) else None)
             for v, y in keys],
            index=df.index, dtype="object",
        )
        mask = override_vals.notna()
        if mask.any():
            df.loc[mask, "grass"] = override_vals[mask].astype(bool)
            logger.info("Applied STADIUM_SURFACE_OVERRIDES (game side) to %d rows",
                        int(mask.sum()))

    # Home-side lookup (player's school's home stadium per season)
    home_lookup = _compute_school_home_venue(schedule, venues)
    if not home_lookup.empty and flat_override:
        # Apply the same override to the home-stadium grass so a (school,
        # season) whose home venue had a surface change reads the right value.
        # We need home_venue_id on the lookup — recompute briefly:
        home = schedule[(schedule["is_neutral"] == 0)].dropna(
            subset=["venue_id", "home_team", "season_year"]
        )
        modal = (home.groupby(["home_team", "season_year"])["venue_id"]
                      .agg(lambda s: s.value_counts().idxmax())
                      .reset_index()
                      .rename(columns={"home_team": "school",
                                        "venue_id": "home_venue_id"}))
        home_lookup = home_lookup.merge(
            modal, on=["school", "season_year"], how="left",
        )
        keys = list(zip(home_lookup["home_venue_id"].astype("Int64"),
                         home_lookup["season_year"].astype("Int64")))
        override_vals = pd.Series(
            [flat_override.get((int(v), int(y))
                                if pd.notna(v) and pd.notna(y) else None)
             for v, y in keys],
            index=home_lookup.index, dtype="object",
        )
        mask = override_vals.notna()
        if mask.any():
            home_lookup.loc[mask, "home_grass"] = override_vals[mask].astype(bool)
        home_lookup = home_lookup.drop(columns=["home_venue_id"])

    if home_lookup.empty:
        logger.warning("Empty home-venue lookup — venue diff features will be NaN")
        df["altitude_diff_from_home_ft"] = np.nan
        df["surface_diff_from_home"]     = np.nan
        df["travel_distance_mi"]         = np.nan
    else:
        df = df.merge(home_lookup, on=["school", "season_year"], how="left")

        # Altitude diff, rounded to ALTITUDE_DIFF_BUCKET_FT
        bucket = float(ALTITUDE_DIFF_BUCKET_FT)
        raw_diff = df["elevation_ft"] - df["home_elevation_ft"]
        df["altitude_diff_from_home_ft"] = (
            (raw_diff / bucket).round() * bucket
        )

        # Surface diff: signed in {-1, 0, +1}, NaN if either side unknown
        game_g = df["grass"].astype("boolean")
        home_g = df["home_grass"].astype("boolean")
        df["surface_diff_from_home"] = (
            game_g.astype("Int16") - home_g.astype("Int16")
        ).astype("Float32")

        # Travel distance: Haversine miles from player's home stadium to game
        # venue, bucketed to TRAVEL_DISTANCE_BUCKET_MI. HOME games = 0 (same
        # coords); AWAY = bucketed miles. NaN if either side missing lat/lon.
        raw_miles = _haversine_miles(
            df["home_latitude"], df["home_longitude"],
            df["latitude"], df["longitude"],
        )
        mi_bucket = float(TRAVEL_DISTANCE_BUCKET_MI)
        df["travel_distance_mi"] = (raw_miles / mi_bucket).round() * mi_bucket

        # Crowd-hostility shock: game venue capacity minus the player's own home
        # capacity, bucketed. Same diff-vs-raw reasoning as altitude above — raw
        # capacity is close to a school-identity tag (Michigan 107k, Penn State
        # 106k), so a model could learn "capacity=107601 → Michigan-typical
        # production" instead of "hostile road crowd." As a diff, HOME games are
        # 0 and AWAY games carry a signed step-up/step-down that is blind to which
        # stadium it is. Negative = quieter venue than home (an FCS body-bag game
        # in reverse), positive = louder.
        home_cap = pd.to_numeric(df.get("home_capacity"), errors="coerce")
        home_cap = home_cap.where(home_cap > 0)
        cap_bucket = float(CAPACITY_DIFF_BUCKET)
        raw_cap_diff = df["capacity"] - home_cap
        df["capacity_diff_from_home"] = (
            (raw_cap_diff / cap_bucket).round() * cap_bucket
        )

    # Drop intermediate columns (keep venue_id for downstream optional use)
    drop_cols = [c for c in
                 ["grass", "elevation_ft", "latitude", "longitude",
                  "capacity",
                  "home_grass", "home_elevation_ft",
                  "home_latitude", "home_longitude", "home_capacity"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Coverage logging
    alt_pop = int(df["altitude_diff_from_home_ft"].notna().sum())
    surf_pop = int(df["surface_diff_from_home"].notna().sum())
    miles_pop = int(df["travel_distance_mi"].notna().sum())
    nonzero_alt = int((df["altitude_diff_from_home_ft"].fillna(0) != 0).sum())
    nonzero_surf = int((df["surface_diff_from_home"].fillna(0) != 0).sum())
    nonzero_miles = int((df["travel_distance_mi"].fillna(0) != 0).sum())
    cap_pop = int(df["capacity_diff_from_home"].notna().sum())
    nonzero_cap = int((df["capacity_diff_from_home"].fillna(0) != 0).sum())
    logger.info("Joined venue diff features: "
                "altitude_diff %d / %d (%.1f%%, %d nz); "
                "surface_diff %d / %d (%.1f%%, %d nz); "
                "travel_mi %d / %d (%.1f%%, %d nz); "
                "capacity_diff %d / %d (%.1f%%, %d nz)",
                alt_pop, len(df), 100 * alt_pop / len(df), nonzero_alt,
                surf_pop, len(df), 100 * surf_pop / len(df), nonzero_surf,
                miles_pop, len(df), 100 * miles_pop / len(df), nonzero_miles,
                cap_pop, len(df), 100 * cap_pop / len(df), nonzero_cap)
    return df


def _approximate_game_date(season_year: int, week: int) -> "pd.Timestamp":
    """
    Approximate game date from season_year + week when schedule is unavailable.
    College football week 1 ≈ first Saturday of September.
    """
    import datetime
    # Approximate: September 1 + (week - 1) * 7 days
    approx = datetime.date(int(season_year), 9, 1) + datetime.timedelta(days=(int(week) - 1) * 7)
    return pd.Timestamp(approx)


def _add_schedule_features(gamelogs: pd.DataFrame,
                            schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Join game_date, opponent, is_neutral, game_hour from schedule.
    Falls back to week-based approximation when schedule is unavailable.
    """
    df = gamelogs.copy()

    if not schedule.empty:
        sched = schedule[["game_id", "game_date", "game_hour", "home_team",
                           "away_team", "is_neutral"]].copy()

        # Drop duplicates (postseason may have extra entries)
        sched = sched.drop_duplicates("game_id")

        df = df.merge(sched, on="game_id", how="left")

        # Determine opponent from schedule home/away
        if "school" in df.columns and "home_team" in df.columns:
            df["opponent"] = np.where(df["school"] == df["home_team"],
                                      df["away_team"], df["home_team"])

    # Fill game_date from week approximation for rows where schedule join failed
    if "game_date" not in df.columns:
        df["game_date"] = pd.NaT
    df["game_date"] = pd.to_datetime(df["game_date"])
    missing_date = df["game_date"].isna()
    if missing_date.any() and "season_year" in df.columns and "week" in df.columns:
        df.loc[missing_date, "game_date"] = df.loc[missing_date].apply(
            lambda r: _approximate_game_date(r["season_year"], r["week"]), axis=1
        )
        n_approx = missing_date.sum()
        if n_approx > 0:
            logger.info("Approximated game_date from week for %d rows (no schedule data)", n_approx)

    # days_rest: days since player's previous game
    if "player_id" in df.columns and "game_date" in df.columns:
        df = df.sort_values(["player_id", "game_date"])
        df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
        df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days.clip(upper=14).fillna(7)
        df = df.drop(columns=["prev_game_date"])

    # opp_days_rest: days since the opponent's previous game
    if not schedule.empty and "opponent" in df.columns and "game_id" in df.columns:
        sched_dates = schedule[["game_id", "game_date", "home_team", "away_team"]].drop_duplicates("game_id").copy()
        sched_dates["game_date"] = pd.to_datetime(sched_dates["game_date"])
        home_tg = sched_dates[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"})
        away_tg = sched_dates[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"})
        team_games = pd.concat([home_tg, away_tg], ignore_index=True).drop_duplicates(["game_id", "team"])
        team_games = team_games.sort_values(["team", "game_date"])
        team_games["prev_date"] = team_games.groupby("team")["game_date"].shift(1)
        team_games["team_days_rest"] = (
            team_games["game_date"] - team_games["prev_date"]
        ).dt.days.clip(upper=14).fillna(7)
        opp_rest = team_games[["game_id", "team", "team_days_rest"]].rename(
            columns={"team": "opponent", "team_days_rest": "opp_days_rest"}
        )
        df = df.merge(opp_rest, on=["game_id", "opponent"], how="left")
        df["opp_days_rest"] = df["opp_days_rest"].fillna(7)

    # tz_diff: home_team_tz - away_team_tz (hours)
    # Negative = away team traveled west (plays later in body clock — more disruptive)
    # Positive = away team traveled east (plays earlier in body clock)
    # Zero for home team or neutral site
    if "home_team" in df.columns and "away_team" in df.columns and "school" in df.columns:
        home_tz = df["home_team"].map(SCHOOL_TIMEZONE_OFFSET).fillna(DEFAULT_TIMEZONE_OFFSET)
        away_tz = df["away_team"].map(SCHOOL_TIMEZONE_OFFSET).fillna(DEFAULT_TIMEZONE_OFFSET)
        game_tz = home_tz  # game is played in home team's timezone
        player_tz = df["school"].map(SCHOOL_TIMEZONE_OFFSET).fillna(DEFAULT_TIMEZONE_OFFSET)
        df["tz_diff"] = (game_tz - player_tz).where(
            df.get("is_neutral", pd.Series(0, index=df.index)) == 0, other=0
        )
    else:
        df["tz_diff"] = 0

    # body_clock_hour: game_hour (local at venue) - tz_diff = the time the
    # kickoff feels like for the PLAYER. Equals game_hour for home/neutral
    # rows (tz_diff=0). For an away PT-school player at a 7pm ET kickoff:
    # tz_diff = -5 - (-8) = +3 → body_clock = 19 - 3 = 16 (4pm body time).
    # Modulo 24 to handle any wrap-around at the day boundary.
    if "game_hour" in df.columns:
        gh = pd.to_numeric(df["game_hour"], errors="coerce")
        tz = pd.to_numeric(df["tz_diff"], errors="coerce").fillna(0)
        df["body_clock_hour"] = ((gh - tz) % 24).where(gh.notna(), other=np.nan)
    else:
        df["body_clock_hour"] = np.nan

    if "is_neutral" not in df.columns:
        df["is_neutral"] = 0
    if "game_hour" not in df.columns:
        df["game_hour"] = np.nan

    return df


def _compute_cfb_season_averages(
    gamelogs: pd.DataFrame,
    adv_game_stats: pd.DataFrame,
) -> pd.DataFrame:
    """League-wide season averages — analogous to BartTorvik season stats in cdarko."""
    # From advanced game stats (full 2004–2025 coverage)
    adv_cols = [c for c in ["off_ppa", "off_success_rate", "off_explosiveness", "off_stuff_rate",
                             "def_ppa", "def_success_rate", "def_stuff_rate"]
                if c in adv_game_stats.columns]
    if adv_cols and "season_year" in adv_game_stats.columns:
        adv_avg = (adv_game_stats.groupby("season_year")[adv_cols]
                   .mean().reset_index()
                   .rename(columns={c: f"lg_{c}" for c in adv_cols}))
    else:
        adv_avg = pd.DataFrame(columns=["season_year"])

    # From player gamelogs — collapse to one row per team-game, then season average
    gl_avg = pd.DataFrame(columns=["season_year"])
    if ({"school", "game_id", "season_year", "team_pass_att", "team_rush_att"}
            .issubset(gamelogs.columns)):
        tg_key = ["school", "game_id", "season_year"]
        count_cols = [c for c in ["pass_tds", "rush_tds", "interceptions"]
                      if c in gamelogs.columns]
        tg = gamelogs[tg_key + ["team_pass_att", "team_rush_att"] + count_cols].copy()

        agg_d: dict = {"team_pass_att": ("team_pass_att", "first"),
                       "team_rush_att": ("team_rush_att", "first")}
        for c in count_cols:
            agg_d[c] = (c, "sum")
        team_game = tg.groupby(tg_key).agg(**agg_d).reset_index()

        denom = (team_game["team_pass_att"] + team_game["team_rush_att"]).clip(lower=1)
        team_game["pass_rate"] = team_game["team_pass_att"] / denom

        season_agg = {"pass_rate": "mean", "team_pass_att": "mean", "team_rush_att": "mean"}
        season_rename = {"pass_rate": "lg_pass_rate",
                         "team_pass_att": "lg_pass_att",
                         "team_rush_att": "lg_rush_att"}
        for col, lg_name in [("pass_tds", "lg_pass_tds"),
                              ("rush_tds", "lg_rush_tds"),
                              ("interceptions", "lg_interceptions")]:
            if col in team_game.columns:
                season_agg[col] = "mean"
                season_rename[col] = lg_name

        gl_avg = (team_game.groupby("season_year").agg(season_agg)
                  .reset_index().rename(columns=season_rename))

    season_avgs = adv_avg.merge(gl_avg, on="season_year", how="outer")
    return season_avgs.sort_values("season_year").reset_index(drop=True)


def main():
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    start = END_YEAR - 2 if args.test else START_YEAR
    end = END_YEAR

    if args.assemble_only:
        # ── Load all existing raw data without hitting the API ─────────────────
        logger.info("Assemble-only mode: loading existing raw parquets")

        def _require(path: Path, name: str) -> pd.DataFrame:
            if not path.exists():
                logger.error("%s not found at %s — run Step 1 without --assemble-only first",
                             name, path)
                sys.exit(1)
            df = pd.read_parquet(path)
            logger.info("Loaded %d rows from %s", len(df), path.name)
            return df

        gamelogs  = _require(RAW_DIR / "player_gamelogs_raw.parquet", "player_gamelogs_raw")
        schedule  = _require(RAW_DIR / "game_schedule.parquet",       "game_schedule")
        adv_game  = _require(RAW_DIR / "advanced_game_stats.parquet", "advanced_game_stats")
        weather   = pd.read_parquet(RAW_DIR / "game_weather.parquet") \
                    if (RAW_DIR / "game_weather.parquet").exists() else pd.DataFrame()
    else:
        # ── Season-level data (unchanged — needed by Steps 2 and 3) ───────────

        logger.info("Fetching player season stats %d–%d", start, end)
        player_seasons = fetch_player_season_stats(start, end)
        if player_seasons.empty:
            logger.error("No player season stats returned — check CFBD_API_KEY")
            sys.exit(1)
        out = RAW_DIR / "player_seasons.parquet"
        player_seasons.to_parquet(out, index=False)
        logger.info("Saved %d player-seasons → %s", len(player_seasons), out)

        logger.info("Fetching team ratings %d–%d", start, end)
        team_ratings = fetch_team_ratings(start, end)
        if not team_ratings.empty:
            out = RAW_DIR / "team_ratings.parquet"
            team_ratings.to_parquet(out, index=False)
            logger.info("Saved %d team-season ratings → %s", len(team_ratings), out)

        # Per-season conference affiliation. Separate from team_ratings because
        # SP+ reports only the team's CURRENT conference (see
        # fetch_team_conferences docstring); step 3 needs the season's.
        logger.info("Fetching FBS team conferences %d–%d", start, end)
        team_conferences = fetch_team_conferences(start, end)
        if not team_conferences.empty:
            out = RAW_DIR / "team_conferences.parquet"
            team_conferences.to_parquet(out, index=False)
            logger.info("Saved %d team-season conferences → %s",
                        len(team_conferences), out)

        logger.info("Fetching team advanced stats %d–%d", start, end)
        team_advanced = fetch_team_advanced_stats(start, end)
        if not team_advanced.empty:
            out = RAW_DIR / "team_advanced.parquet"
            team_advanced.to_parquet(out, index=False)
            logger.info("Saved %d team advanced rows → %s", len(team_advanced), out)

        logger.info("Fetching draft picks %d–%d", max(start, 1994), end)
        draft_picks = fetch_draft_picks(max(start, 1994), end)
        if not draft_picks.empty:
            out = RAW_DIR / "draft_picks.parquet"
            draft_picks.to_parquet(out, index=False)
            logger.info("Saved %d draft picks → %s", len(draft_picks), out)

        # Venues: ~840-row one-time fetch (no year axis). Joined to games via
        # venue_id from game_schedule for surface/altitude/dome features.
        venues_path = RAW_DIR / "venues.parquet"
        if args.fetch_venues or not venues_path.exists():
            logger.info("Fetching venues from CFBD")
            venues = fetch_venues()
            if not venues.empty:
                venues.to_parquet(venues_path, index=False)
                logger.info("Saved %d venues → %s", len(venues), venues_path)

        if args.skip_gamelogs:
            logger.info("Skipping game-level data (--skip-gamelogs).")
            logger.info("Step 1 complete.")
            return

        # ── Game-level data ────────────────────────────────────────────────────

        logger.info("Fetching player game stats %d–%d", start, end)
        gamelogs = fetch_player_game_stats(start, end)
        if gamelogs.empty:
            logger.warning("No game-level player stats — proceeding without gamelogs")
            logger.info("Step 1 complete.")
            return
        logger.info("Fetched %d player-game rows", len(gamelogs))

        # Save raw gamelogs before assembly so --assemble-only can reload them
        raw_gamelogs_path = RAW_DIR / "player_gamelogs_raw.parquet"
        gamelogs.to_parquet(raw_gamelogs_path, index=False)
        logger.info("Saved raw gamelogs → %s", raw_gamelogs_path)

        logger.info("Fetching game schedule %d–%d", start, end)
        schedule = fetch_game_schedule(start, end)
        if not schedule.empty:
            out = RAW_DIR / "game_schedule.parquet"
            schedule.to_parquet(out, index=False)
            logger.info("Saved %d game schedule rows → %s", len(schedule), out)

        logger.info("Fetching advanced game stats %d–%d", start, end)
        adv_game = fetch_advanced_game_stats(start, end)
        if not adv_game.empty:
            out = RAW_DIR / "advanced_game_stats.parquet"
            adv_game.to_parquet(out, index=False)
            logger.info("Saved %d advanced game stat rows → %s", len(adv_game), out)

        logger.info("Fetching game weather %d–%d", start, end)
        weather = fetch_game_weather(start, end)
        if not weather.empty:
            out = RAW_DIR / "game_weather.parquet"
            weather.to_parquet(out, index=False)
            logger.info("Saved %d weather rows → %s", len(weather), out)

    # ── Advanced box scores (snap counts + PPA + team stats) ──────────────────

    player_box_path = RAW_DIR / "player_box_scores.parquet"
    team_box_path   = RAW_DIR / "team_box_scores.parquet"

    if args.fetch_box_scores or not (player_box_path.exists() and team_box_path.exists()):
        if not schedule.empty and "game_id" in schedule.columns:
            game_ids = schedule["game_id"].dropna().unique().tolist()
            logger.info("Fetching advanced box scores for %d games "
                        "(~%d min at rate limit) ...", len(game_ids),
                        len(game_ids) // 240)
            player_box, team_box = fetch_advanced_box_scores(game_ids)
            if not player_box.empty:
                player_box.to_parquet(player_box_path, index=False)
                logger.info("Saved %d player box score rows → %s",
                            len(player_box), player_box_path)
            if not team_box.empty:
                team_box.to_parquet(team_box_path, index=False)
                logger.info("Saved %d team box score rows → %s",
                            len(team_box), team_box_path)
        else:
            logger.warning("No schedule — cannot fetch box scores (need game_ids)")
            player_box = pd.DataFrame()
            team_box   = pd.DataFrame()
    else:
        logger.info("Loading cached box scores from parquet")
        player_box = pd.read_parquet(player_box_path)
        team_box   = pd.read_parquet(team_box_path)
        logger.info("Loaded %d player / %d team box score rows",
                    len(player_box), len(team_box))

    # ── Assemble player_gamelogs.parquet ───────────────────────────────────────

    # Strip phantom opponent-side rows BEFORE any join. Must run first: every
    # downstream join (schedule, venue, opponent-rolling) keys off `school`, so
    # a row on the wrong side of the game would pull in the wrong opponent,
    # venue and team context.
    gamelogs = _drop_opponent_side_phantom_rows(gamelogs)

    # Join schedule features (game_date, opponent, is_neutral, game_hour, days_rest, tz_diff)
    gamelogs = _add_schedule_features(gamelogs, schedule)

    # Join weather
    if not weather.empty and "game_id" in gamelogs.columns:
        weather_cols = ["game_id", "temperature", "wind_speed", "wind_direction",
                        "dew_point", "precipitation", "snowfall", "pressure",
                        "game_indoors", "humidity"]
        weather_cols = [c for c in weather_cols if c in weather.columns]
        gamelogs = gamelogs.merge(weather[weather_cols], on="game_id", how="left")

    # Join venue features (surface + elevation). Reads venues.parquet directly;
    # uses schedule.venue_id for the join key (merged via _add_schedule_features
    # below if not already present).
    if "game_id" in gamelogs.columns:
        gamelogs = _add_venue_features(gamelogs, schedule)

    # Join actual snap counts and PPA from box scores (adds actual_snaps column)
    gamelogs = _join_box_score_player_data(gamelogs, player_box)

    # Load player attributes for position group join
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    attrs = pd.read_parquet(attrs_path) if attrs_path.exists() else pd.DataFrame()

    # Normalize: compute position_group, snaps (actual where available), per-100-snap columns
    gamelogs = _normalize_gamelogs(gamelogs, attrs)

    # Drop known bad (player_id, season_year) pairs — CFBD occasionally recycles
    # a player_id for a different person with the same name. The exclusion set
    # = hand-curated KNOWN_SEASON_EXCLUSIONS ∪ auto-detected combine-gap cases.
    from src.utils.cfbd_id_reuse import merged_exclusions as _merged_exclusions, exclusion_mask
    _exclusions = _merged_exclusions(KNOWN_SEASON_EXCLUSIONS, RAW_DIR, PROCESSED_DIR, logger)
    if _exclusions and "player_id" in gamelogs.columns and "season_year" in gamelogs.columns:
        bad_mask = exclusion_mask(gamelogs, _exclusions)
        n_dropped = int(bad_mask.sum())
        if n_dropped:
            gamelogs = gamelogs[~bad_mask].reset_index(drop=True)
            logger.info("Dropped %d gamelog rows matching KNOWN_SEASON_EXCLUSIONS", n_dropped)

    # Join per-game conference tier from player_seasons (player_id × season_year → conference).
    # build_context_features() already prefers this column over the string 'conference' fallback,
    # giving transfer players the correct tier for each school they played at.
    _seasons_path = RAW_DIR / "player_seasons.parquet"
    if _seasons_path.exists() and "player_id" in gamelogs.columns and "season_year" in gamelogs.columns:
        ps_conf = (pd.read_parquet(_seasons_path, columns=["player_id", "season_year", "conference"])
                   .drop_duplicates(["player_id", "season_year"]))
        ps_conf["conference_tier"] = ps_conf["conference"].map(
            lambda c: CONFERENCE_TIERS.get(c, DEFAULT_CONFERENCE_TIER)
        ).astype("Int64")
        gamelogs = gamelogs.merge(
            ps_conf[["player_id", "season_year", "conference_tier"]],
            on=["player_id", "season_year"], how="left",
        )
        n_matched = gamelogs["conference_tier"].notna().sum()
        logger.info("Joined per-game conference_tier: %d / %d game rows matched",
                    n_matched, len(gamelogs))

    # Add rolling opponent strength (with havoc from team box scores)
    gamelogs = _add_opponent_rolling_stats(gamelogs, adv_game, schedule,
                                           team_box_scores=team_box)

    # Add teammate-QB rolling quality. For each (game_id, school), pull the
    # team's QBs' decay-weighted rate stats over strictly-prior games at the
    # same school. Residualizes receiver/RB stats for the quality of the QB
    # feeding them — flows through CONTEXT_FEATURES in step 4 (gated to
    # WR/TE/RB via CONTEXT_FEATURE_DROP_BY_POSITION).
    gamelogs = _add_qb_rolling_stats(gamelogs)

    # Add own-team rolling context: pace (plays/game), raw pass rate, and OL
    # run-block four-factor from team_box_scores. Same (game_id, school) shape
    # as qb_rolling; gated per-position via CONTEXT_FEATURE_DROP_BY_POSITION
    # (pace+pass_rate → QB/RB/WR/TE; OL run-block → RB-only).
    # REVERTED family (removed from CONTEXT_FEATURES 2026-05-16) — only
    # computed when COMPUTE_REVERTED_CONTEXT is flipped on for reactivation.
    if COMPUTE_REVERTED_CONTEXT:
        gamelogs = _add_team_rolling_context(gamelogs, team_box)

    # Add opponent-defense ALLOWED rolling features (derived from
    # team_box_scores via inverse-perspective). Run-defense detail
    # (rushing_stuff_rate, line_yards_avg, ...) plus pass-defense detail
    # (success_rate_passing_downs, points_per_opportunity, ...). Gated to
    # offensive skill positions via CONTEXT_FEATURE_DROP_BY_POSITION.
    gamelogs = _add_opp_defense_allowed(gamelogs, team_box, schedule)

    # Add own-team rolling scoring environment from team_box_scores
    # (scoring_opportunities, points_per_opportunity, avg_starting_field_pos).
    # Gated to offensive skill positions + K via CONTEXT_FEATURE_DROP_BY_POSITION.
    gamelogs = _add_team_scoring_env(gamelogs, team_box)

    # The three defender-side families below are REVERTED (removed from
    # CONTEXT_FEATURES 2026-05-17 after the retune A/B regressed) — only
    # computed when COMPUTE_REVERTED_CONTEXT is flipped on for reactivation.
    if COMPUTE_REVERTED_CONTEXT:
        # Add opponent-offense rolling VOLUME (pass_att / rush_att / plays).
        # Aimed at defenders — gives opportunity-side context that opponent
        # QUALITY metrics (opp_rolling_off_ppa et al.) can't supply.
        gamelogs = _add_opp_offense_volume(gamelogs)

        # Add opponent-QB rolling — mirror of teammate qb_rolling but joined on
        # opponent. For each defender's game, rolling QB rate stats of the QB
        # they're about to face. Gated to DL/LB/DB via CONTEXT_FEATURE_DROP_BY_POSITION.
        gamelogs = _add_opp_qb_rolling_stats(gamelogs)

        # Add own-defense teammate rolling (pressure_rate + coverage_rate).
        # Self-reference managed via CONTEXT_FEATURE_DROP_BY_POSITION:
        # pressure_rate is mostly DL/LB → drop from DL (self); coverage_rate is
        # mostly DB → drop from DB (self). LB gets both (partial self acceptable).
        gamelogs = _add_defense_rolling_stats(gamelogs)

    # Season-level league averages (contextual model era features)
    season_avgs = _compute_cfb_season_averages(gamelogs, adv_game)
    season_avgs_path = PROCESSED_DIR / "cfb_season_averages.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    season_avgs.to_parquet(season_avgs_path, index=False)
    logger.info("Saved %d season-average rows → %s", len(season_avgs), season_avgs_path)
    gamelogs = gamelogs.merge(season_avgs, on="season_year", how="left")
    lg_cols = [c for c in season_avgs.columns if c != "season_year"]
    logger.info("Joined %d league-average features to gamelogs", len(lg_cols))

    # Save
    out = RAW_DIR / "player_gamelogs.parquet"
    gamelogs.to_parquet(out, index=False)
    logger.info("Saved %d player-game rows → %s", len(gamelogs), out)

    # Sanity checks
    snap_col_count = sum(1 for c in gamelogs.columns if c.endswith("_p100"))
    logger.info("Per-100-snap columns: %d", snap_col_count)
    if "snaps" in gamelogs.columns:
        logger.info("Median snaps per game: %.1f", gamelogs["snaps"].median())
    if "game_date" in gamelogs.columns:
        logger.info("Date range: %s — %s", gamelogs["game_date"].min(),
                    gamelogs["game_date"].max())

    logger.info("Step 1 complete.")


if __name__ == "__main__":
    main()
