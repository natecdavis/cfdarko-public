"""
Step 1f: Player-Plays Join

Joins plays + play_stats + game context + venue orientation into the
per-(play, athlete) credited grain used by step 4p (per-play LightGBM
contextual residualization).

CATCH-ALL ROWS ARE NOT MATERIALIZED HERE. Step 5p computes the catch-all
expectation directly from rosters.parquet × plays.parquet at residualization
time — keeps this parquet pure-credited and avoids materializing tens of
millions of placeholder rows.

Inputs (data/raw/, data/processed/):
    plays.parquet                      — from step 1d
    play_stats.parquet                 — from step 1d
    game_schedule.parquet              — from step 1
    game_weather.parquet               — from step 1 (or 01b --weather-only)
    venues.parquet                     — from step 1
    venue_orientation.parquet          — from step 1e
    rosters.parquet                    — from step 1
    player_attributes.parquet          — from step 3 (provides position_group)
    adjusted_gamelogs.parquet          — from step 5 (provides demeaned
                                         opp_rolling_*, qb_rolling_*, conf_tier
                                         per (game_id, player_id))

Output:
    data/processed/player_plays.parquet
        one row per (play_id, athlete_id) credited tuple
        cols: identifiers + 38 PLAY_CONTEXT_FEATURES + 26 stat columns

Usage:
    python pipeline/01f_player_plays.py
    python pipeline/01f_player_plays.py --test     # subset for smoke testing
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALL_PLAY_STAT_TYPES, GARBAGE_TIME_RULES, PLAY_STAT_COL, PLAY_START_YEAR,
    PROCESSED_DIR, RAW_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLAYS_PATH        = RAW_DIR / "plays.parquet"
PLAY_STATS_PATH   = RAW_DIR / "play_stats.parquet"
WEATHER_PATH      = RAW_DIR / "game_weather.parquet"
VENUES_PATH       = RAW_DIR / "venues.parquet"
ORIENTATION_PATH  = RAW_DIR / "venue_orientation.parquet"
PA_PATH           = PROCESSED_DIR / "player_attributes.parquet"
ADJ_GAMELOGS_PATH = PROCESSED_DIR / "adjusted_gamelogs.parquet"

OUT_PATH          = PROCESSED_DIR / "player_plays.parquet"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only the latest season")
    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────


def _pivot_play_stats(play_stats: pd.DataFrame) -> pd.DataFrame:
    """Long → wide on stat_type, keyed by (play_id, athlete_id).

    A given (play_id, athlete_id) may have multiple stat_type rows (e.g., a
    receiver who caught a pass for a TD has Reception + Target + Touchdown).
    Pivot makes one row per (play_id, athlete_id) with up to 26 stat columns.
    """
    if play_stats.empty:
        return pd.DataFrame()

    # Rename stat_type → snake_case wide column. NB: stat_type values are
    # Title Case (e.g., "Pass Breakup"); PLAY_STAT_COL maps to snake_case.
    play_stats = play_stats.copy()
    play_stats["stat_col"] = play_stats["stat_type"].map(PLAY_STAT_COL)
    # Drop rows for stat_types we don't model (shouldn't be any, but defensive)
    play_stats = play_stats.dropna(subset=["stat_col"])

    # Pivot. If duplicates per (play_id, athlete_id, stat_col) — unlikely but
    # possible (e.g., split tackles assigned to same player) — sum them.
    wide = play_stats.pivot_table(
        index=["play_id", "athlete_id"],
        columns="stat_col",
        values="stat",
        aggfunc="sum",
    ).reset_index()

    # Carry the metadata columns (team, opponent, season_year, week) from
    # the first long row per (play_id, athlete_id). They're constant within
    # the group by construction.
    meta = (play_stats
            .drop_duplicates(subset=["play_id", "athlete_id"])
            [["play_id", "athlete_id", "team", "opponent", "season_year",
              "week", "game_id", "conference"]])
    wide = wide.merge(meta, on=["play_id", "athlete_id"], how="left")

    return wide


def _wind_decomposition(wind_speed: pd.Series,
                         wind_direction: pd.Series,
                         orientation_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Project wind vector onto field axis.

    wind_direction is the meteorological "from" bearing (0-359°).
    orientation_deg is the field long-axis bearing (0-180°).

    Returns (wind_parallel, wind_perpendicular). Convention:
      - Positive parallel = wind blowing along field axis
      - Magnitude only (sign discarded — we don't model which direction the
        offense is driving on a given play)
    """
    delta_rad = np.radians(wind_direction.astype(float) - orientation_deg.astype(float))
    wind_par = wind_speed.astype(float) * np.cos(delta_rad)
    wind_perp = wind_speed.astype(float) * np.sin(delta_rad)
    return wind_par.abs(), wind_perp.abs()


def _classify_garbage_time(score_diff: pd.Series, period: pd.Series) -> pd.Series:
    """Flag plays in the garbage-time window (no v2 weighting yet)."""
    thresh = GARBAGE_TIME_RULES["score_diff_abs_gte"]
    per_min = GARBAGE_TIME_RULES["period_gte"]
    return ((score_diff.abs() >= thresh) & (period >= per_min)).astype("int8")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──
    if not PLAYS_PATH.exists() or not PLAY_STATS_PATH.exists():
        logger.error("plays.parquet or play_stats.parquet not found — run step 1d first")
        sys.exit(1)

    plays = pd.read_parquet(PLAYS_PATH)
    play_stats = pd.read_parquet(PLAY_STATS_PATH)
    logger.info("Loaded %d plays, %d play_stats rows", len(plays), len(play_stats))

    plays = plays[plays["season_year"] >= PLAY_START_YEAR]
    play_stats = play_stats[play_stats["season_year"] >= PLAY_START_YEAR]

    if args.test:
        latest = int(plays["season_year"].max())
        logger.info("Test mode: restricting to season %d", latest)
        plays      = plays[plays["season_year"] == latest]
        play_stats = play_stats[play_stats["season_year"] == latest]

    # ── Pivot play_stats wide ──
    logger.info("Pivoting play_stats wide on stat_type ...")
    wide = _pivot_play_stats(play_stats)
    logger.info("  %d credited (play_id, athlete_id) rows after pivot", len(wide))

    # ── Join play-level context from plays.parquet ──
    play_cols = [
        "play_id", "drive_id", "game_id", "offense", "defense",
        "period", "clock_seconds", "offense_timeouts", "defense_timeouts",
        "yardline", "yards_to_goal", "down", "distance", "yards_gained",
        "scoring", "play_type", "ppa",
        "offense_score", "defense_score",
    ]
    play_cols = [c for c in play_cols if c in plays.columns]
    # Drop duplicate game_id from the wide pivot (plays.parquet's value is
    # authoritative; play_stats's was carried through the pivot's meta carry)
    if "game_id" in wide.columns:
        wide = wide.drop(columns=["game_id"])
    df = wide.merge(plays[play_cols], on="play_id", how="left")
    logger.info("  after play context join: %d rows", len(df))

    # ── Derived play features ──
    # score_diff is from offense's perspective. For a credited athlete on the
    # OFFENSIVE team this is "we lead by N"; on DEFENSE flip the sign.
    df["score_diff"] = df["offense_score"] - df["defense_score"]
    # athlete_team == offense → offense_perspective; else defense
    df.loc[df["team"] == df["defense"], "score_diff"] *= -1
    df["is_garbage_time"] = _classify_garbage_time(df["score_diff"], df["period"])

    # ── Position_group ──
    if PA_PATH.exists():
        pa = pd.read_parquet(PA_PATH, columns=["player_id",
                                                "position_group", "conference_tier"])
        # athlete_id is the player_id; align dtype
        pa["athlete_id"] = pa["player_id"].astype(str)
        df["athlete_id"] = df["athlete_id"].astype(str)
        # player_attributes is one row per player (its season_year is the
        # player's LAST roster season) — joining on athlete_id alone gives
        # every credited play its player's canonical position/tier; joining
        # on season_year too would only label final-season plays.
        pa = pa.drop_duplicates(subset=["athlete_id"])
        df = df.merge(
            pa[["athlete_id", "position_group", "conference_tier"]],
            on="athlete_id", how="left",
        )
        # Rename conference_tier → conf_tier (matches PLAY_CONTEXT_FEATURES naming)
        df = df.rename(columns={"conference_tier": "conf_tier"})
        logger.info("  position_group coverage: %.1f%% (%d/%d)",
                    100.0 * df["position_group"].notna().mean(),
                    df["position_group"].notna().sum(), len(df))
    else:
        logger.warning("player_attributes.parquet not found — position_group will be NaN")
        df["position_group"] = pd.NA
        df["conf_tier"] = pd.NA

    # ── Weather + wind decomposition ──
    if WEATHER_PATH.exists():
        weather = pd.read_parquet(WEATHER_PATH)
        weather_cols = ["game_id", "temperature", "dew_point", "humidity",
                        "wind_speed", "wind_direction", "precipitation",
                        "snowfall", "pressure", "game_indoors"]
        weather_cols = [c for c in weather_cols if c in weather.columns]
        df = df.merge(weather[weather_cols], on="game_id", how="left")
    else:
        logger.warning("game_weather.parquet not found")

    # Venue orientation
    if ORIENTATION_PATH.exists() and VENUES_PATH.exists():
        venues = pd.read_parquet(VENUES_PATH,
                                  columns=["venue_id", "grass", "elevation_ft"])
        # Bring venue_id in from schedule
        schedule_path = RAW_DIR / "game_schedule.parquet"
        if schedule_path.exists():
            sched = pd.read_parquet(schedule_path, columns=["game_id", "venue_id"])
            df = df.merge(sched, on="game_id", how="left")
            df = df.merge(venues.rename(columns={"elevation_ft": "altitude_ft"}),
                           on="venue_id", how="left")
            orientation = pd.read_parquet(ORIENTATION_PATH,
                                          columns=["venue_id", "orientation_deg"])
            df = df.merge(orientation, on="venue_id", how="left")
            # Compute wind decomposition (NaN when wind/orientation missing)
            if "wind_speed" in df.columns and "wind_direction" in df.columns:
                par, perp = _wind_decomposition(
                    df["wind_speed"].fillna(0.0),
                    df["wind_direction"].fillna(0.0),
                    df["orientation_deg"].fillna(0.0),
                )
                # Mask to NaN where any input was missing
                missing = (df["wind_speed"].isna() | df["wind_direction"].isna()
                           | df["orientation_deg"].isna())
                df["wind_parallel"]      = par.where(~missing)
                df["wind_perpendicular"] = perp.where(~missing)
        else:
            logger.warning("game_schedule.parquet missing — skipping venue join")

    # ── Game-context features from adjusted_gamelogs ──
    # Carry every CONTEXT_FEATURES column present in adj that isn't already
    # in df, so per-play models (step 4p) see the same feature surface as
    # per-game models (step 4). Keep the prefix-based catch for any future
    # opp_rolling_* / qb_rolling_* / team_rolling_* / lg_* columns that
    # haven't yet been promoted into CONTEXT_FEATURES.
    if ADJ_GAMELOGS_PATH.exists():
        from config import CONTEXT_FEATURES
        adj = pd.read_parquet(ADJ_GAMELOGS_PATH)
        prefix_carry = [c for c in adj.columns
                        if c.startswith("opp_rolling_") or c.startswith("qb_rolling_")
                        or c.startswith("team_rolling_") or c.startswith("lg_")]
        context_carry = [c for c in CONTEXT_FEATURES if c in adj.columns]
        carry_cols = list(dict.fromkeys(prefix_carry + context_carry))
        carry_cols = [c for c in carry_cols if c not in df.columns]
        if "player_id" in adj.columns and carry_cols:
            adj["player_id"] = adj["player_id"].astype(str)
            df["athlete_id"] = df["athlete_id"].astype(str)
            keep = ["player_id", "game_id"] + carry_cols
            df = df.merge(
                adj[keep].drop_duplicates(subset=["player_id", "game_id"])
                         .rename(columns={"player_id": "athlete_id"}),
                on=["athlete_id", "game_id"], how="left",
            )
            logger.info("  joined %d rolling cols from adjusted_gamelogs", len(carry_cols))
    else:
        logger.info("adjusted_gamelogs.parquet not found — rolling cols will be NaN")

    # ── Ensure all 26 stat columns are present (Nan for missing stat_types) ──
    for stat_type in ALL_PLAY_STAT_TYPES:
        col = PLAY_STAT_COL[stat_type]
        if col not in df.columns:
            df[col] = np.nan

    # ── Phase 3a (2026-05-23): derived outcome stats ──
    # Each derived column is set to 0 or 1 only on rows where the pre-
    # condition is satisfied (relevant base stat was credited to this athlete
    # on this play); NaN elsewhere. Step 4p's credited filter (col.notna())
    # will then naturally restrict training to the conditional subset.
    # Pre-conditions are mutually exclusive within an athlete-play row.

    # Pass-attempt outcomes (QB-side)
    qb_attempted = (df["completion"].notna() | df["incompletion"].notna()
                    | df["interception_thrown"].notna() | df["sack_taken"].notna())
    df["pass_completion_outcome"] = np.where(
        qb_attempted, df["completion"].notna().astype(float), np.nan)
    df["pass_interception_outcome"] = np.where(
        qb_attempted, df["interception_thrown"].notna().astype(float), np.nan)
    df["pass_sack_outcome"] = np.where(
        qb_attempted, df["sack_taken"].notna().astype(float), np.nan)

    # (Target outcomes removed 2026-05-23 — Target Reception had cv_r2=-0.027
    # on CFBD context features; Target Interception is structurally un-
    # modelable because CFBD credits no receiver-side stat on INT play_types.)

    # TD-by-type (given relevant skill credit, was it a TD?)
    completion_credited = df["completion"].notna()
    rush_credited = df["rush"].notna()
    reception_credited = df["reception"].notna()
    td_credited = df["touchdown"].notna()

    df["passing_touchdown_outcome"] = np.where(
        completion_credited,
        (td_credited & (df["play_type"] == "Passing Touchdown")).astype(float),
        np.nan)
    df["rushing_touchdown_outcome"] = np.where(
        rush_credited,
        (td_credited & (df["play_type"] == "Rushing Touchdown")).astype(float),
        np.nan)
    df["receiving_touchdown_outcome"] = np.where(
        reception_credited,
        (td_credited & (df["play_type"] == "Passing Touchdown")).astype(float),
        np.nan)

    # Fumble-by-type (given relevant skill credit, was there a fumble on this play
    # attributed to this athlete?)
    fumble_credited = df["fumble"].notna()
    df["rushing_fumble_outcome"] = np.where(
        rush_credited, fumble_credited.astype(float), np.nan)
    df["receiving_fumble_outcome"] = np.where(
        reception_credited, fumble_credited.astype(float), np.nan)

    # ── Final schema + dtypes ──
    df["is_garbage_time"] = df["is_garbage_time"].astype("int8")
    if "scoring" in df.columns:
        df["scoring"] = df["scoring"].astype("int8")
    if "grass" in df.columns:
        df["grass"] = df["grass"].astype("boolean")
    if "game_indoors" in df.columns:
        df["game_indoors"] = df["game_indoors"].astype("boolean")

    # ── Save ──
    df.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %d player-play rows × %d cols → %s",
                len(df), df.shape[1], OUT_PATH)

    # ── Coverage report ──
    if "position_group" in df.columns:
        logger.info("Position-group breakdown of credited rows:")
        for pos, n in df["position_group"].value_counts(dropna=False).items():
            logger.info("    %-8s %d", str(pos), int(n))

    # Sanity check: per-season row counts
    logger.info("Per-season credited rows:")
    for yr, n in df["season_year"].value_counts().sort_index().items():
        logger.info("    %d  %d", int(yr), int(n))


if __name__ == "__main__":
    main()
