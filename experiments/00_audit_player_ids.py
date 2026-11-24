"""
Step 00: Audit Player ID Collisions

Detects player_ids that CFBD may have recycled for two different people.
Outputs a ranked list of suspicious players for manual review.

Three independent checks, each producing a confidence score:

  1. Bridge-player reappearance (HIGH confidence)
     A player who was an NFL draft pick / UDFA appears in college gamelogs
     two or more seasons AFTER their last documented college season.
     Signal: NFL draftees don't return to college.

  2. Offensive ↔ defensive profile flip (HIGH confidence)
     The same player_id has clear offensive production (receiving, rushing, or
     passing yards) at one school and clear defensive production (tackles, sacks,
     passes defended) at a different school — with no overlap.
     Signal: players don't flip between skill and defensive roles mid-career.

  3. Long gap + school change (MEDIUM confidence)
     A player_id disappears for 3+ years then reappears at a different school.
     Legitimate for injury years or military service, but combined with other
     signals is strong evidence of a recycled ID.

Usage:
    python experiments/00_audit_player_ids.py [--output PATH]

Outputs:
    data/processed/player_id_audit.parquet   — flagged players with signals
    (also prints a summary table to stdout)

After review, add confirmed collisions to KNOWN_SEASON_EXCLUSIONS in config.py.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, RAW_DIR, KNOWN_SEASON_EXCLUSIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Stats that indicate clear offensive production
OFFENSE_STAT_COLS = ["rec_yards", "rush_yards", "pass_yards", "rec", "pass_att", "rec_tds", "rush_tds", "pass_tds"]
# Stats that indicate clear defensive production
DEFENSE_STAT_COLS = ["tackles", "sacks", "tfl", "passes_defended", "interceptions", "qb_hurries"]

# Thresholds: total across all games to count as "clearly playing that side of the ball"
OFFENSE_THRESHOLD = 50    # at least 50 offensive yards (rec+rush+pass) in a season
DEFENSE_THRESHOLD = 5     # at least 5 defensive plays (tackles+sacks+PD) in a season

# Gap threshold in years to flag as suspicious
GAP_YEARS_THRESHOLD = 3

# After how many seasons post-draft should college appearances be flagged
BRIDGE_REAPPEARANCE_MIN_GAP = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(PROCESSED_DIR / "player_id_audit.parquet"))
    parser.add_argument("--min-signals", type=int, default=1,
                        help="Minimum number of signals required to include in output (default 1)")
    return parser.parse_args()


def _load_gamelogs() -> pd.DataFrame:
    path = RAW_DIR / "player_gamelogs.parquet"
    if not path.exists():
        logger.error("player_gamelogs.parquet not found — run Step 01 first")
        sys.exit(1)
    gl = pd.read_parquet(path)
    gl["game_date"] = pd.to_datetime(gl["game_date"])
    return gl


def _load_bridge_players() -> pd.DataFrame:
    path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    if not path.exists():
        logger.warning("nfl_draft_labels.parquet not found — skipping bridge-reappearance check")
        return pd.DataFrame()
    return pd.read_parquet(path, columns=["player_id", "player_name", "last_college_season_year",
                                          "draft_year", "is_bridge", "is_udfa"])


# ── Check 1: Bridge-player reappearance ───────────────────────────────────────

def check_bridge_reappearance(gamelogs: pd.DataFrame,
                               bridge: pd.DataFrame) -> pd.DataFrame:
    """
    Flag player_ids that appear in gamelog seasons 2+ years after their
    documented last college season, and who were NFL draft picks or UDFAs.
    """
    if bridge.empty:
        return pd.DataFrame()

    bridge_active = bridge[bridge["is_bridge"]].dropna(subset=["player_id", "last_college_season_year"])
    bridge_active["player_id"] = bridge_active["player_id"].astype(str)

    gl = gamelogs.copy()
    gl["player_id"] = gl["player_id"].astype(str)

    # Latest gamelog season per player_id
    latest_gl = (gl.groupby("player_id")["season_year"]
                 .agg(["min", "max"])
                 .rename(columns={"min": "gl_min_year", "max": "gl_max_year"})
                 .reset_index())

    merged = bridge_active.merge(latest_gl, on="player_id", how="inner")
    merged["gap"] = merged["gl_max_year"] - merged["last_college_season_year"]
    flagged = merged[merged["gap"] >= BRIDGE_REAPPEARANCE_MIN_GAP].copy()
    flagged["signal"] = "bridge_reappearance"
    flagged["detail"] = (
        "NFL bridge player (last college " + flagged["last_college_season_year"].astype(int).astype(str)
        + ") has gamelog entries in " + flagged["gl_max_year"].astype(int).astype(str)
        + " (" + flagged["gap"].astype(int).astype(str) + " years later)"
    )
    flagged["confidence"] = "HIGH"

    # Attach the suspicious gamelog school for context
    suspicious_school = (
        gl[gl["player_id"].isin(flagged["player_id"])]
        .sort_values("game_date")
        .groupby("player_id")
        .apply(lambda g: g[g["season_year"] == g["season_year"].max()]["school"].iloc[0]
               if len(g) > 0 else "")
        .reset_index()
        .rename(columns={0: "suspicious_school"})
    )
    flagged = flagged.merge(suspicious_school, on="player_id", how="left")

    return flagged[["player_id", "player_name", "signal", "confidence", "detail",
                    "last_college_season_year", "gl_max_year", "suspicious_school"]]


# ── Check 2: Offensive ↔ defensive profile flip ───────────────────────────────

def check_profile_flip(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """
    Flag player_ids where one school has clear offensive production and a
    DIFFERENT school has clear defensive production (no overlap).
    """
    gl = gamelogs.copy()

    # Per player-school totals across all seasons at that school
    agg: dict = {"season_year": ["min", "max"], "player_name": "first"}
    for col in OFFENSE_STAT_COLS:
        if col in gl.columns:
            agg[col] = "sum"
    for col in DEFENSE_STAT_COLS:
        if col in gl.columns:
            agg[col] = "sum"

    by_school = gl.groupby(["player_id", "school"]).agg(agg).reset_index()
    by_school.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                         for c in by_school.columns]

    # Compute offensive and defensive totals per player-school
    off_cols_present = [f"{c}_sum" for c in OFFENSE_STAT_COLS if f"{c}_sum" in by_school.columns]
    def_cols_present = [f"{c}_sum" for c in DEFENSE_STAT_COLS if f"{c}_sum" in by_school.columns]

    by_school["total_offense"] = by_school[off_cols_present].fillna(0).sum(axis=1) if off_cols_present else 0
    by_school["total_defense"] = by_school[def_cols_present].fillna(0).sum(axis=1) if def_cols_present else 0
    by_school["is_offense"] = by_school["total_offense"] >= OFFENSE_THRESHOLD
    by_school["is_defense"] = by_school["total_defense"] >= DEFENSE_THRESHOLD

    # Group by player_id: flag if any school is offensive and any OTHER school is defensive
    rows = []
    for pid, group in by_school.groupby("player_id"):
        off_schools = set(group.loc[group["is_offense"] & ~group["is_defense"], "school"])
        def_schools = set(group.loc[group["is_defense"] & ~group["is_offense"], "school"])
        if off_schools and def_schools:
            name = group["player_name_first"].iloc[0]
            off_years = group[group["school"].isin(off_schools)][["school", "season_year_min", "season_year_max"]].drop_duplicates()
            def_years = group[group["school"].isin(def_schools)][["school", "season_year_min", "season_year_max"]].drop_duplicates()

            off_str = "; ".join(f"{r['school']} ({int(r['season_year_min'])}–{int(r['season_year_max'])})"
                                for _, r in off_years.iterrows())
            def_str = "; ".join(f"{r['school']} ({int(r['season_year_min'])}–{int(r['season_year_max'])})"
                                for _, r in def_years.iterrows())
            rows.append({
                "player_id": str(pid),
                "player_name": name,
                "signal": "offense_defense_flip",
                "confidence": "HIGH",
                "detail": f"Offense at: {off_str} | Defense at: {def_str}",
                "off_schools": ", ".join(off_schools),
                "def_schools": ", ".join(def_schools),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Check 3: Long gap + school change ─────────────────────────────────────────

def check_long_gap(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """
    Flag player_ids that vanish for 3+ years and reappear at a different school.
    """
    gl = gamelogs.copy()

    # Per player-season: school
    by_season = (gl.drop_duplicates(["player_id", "season_year"])
                 [["player_id", "player_name", "school", "season_year"]]
                 .sort_values(["player_id", "season_year"]))

    rows = []
    for pid, group in by_season.groupby("player_id"):
        group = group.sort_values("season_year").reset_index(drop=True)
        if len(group) < 2:
            continue
        for i in range(1, len(group)):
            prev = group.iloc[i - 1]
            curr = group.iloc[i]
            gap = int(curr["season_year"]) - int(prev["season_year"])
            school_changed = prev["school"] != curr["school"]
            if gap >= GAP_YEARS_THRESHOLD and school_changed:
                rows.append({
                    "player_id": str(pid),
                    "player_name": curr["player_name"],
                    "signal": "long_gap_school_change",
                    "confidence": "MEDIUM",
                    "detail": (
                        f"{prev['school']} ({int(prev['season_year'])}) → "
                        f"{curr['school']} ({int(curr['season_year'])}), "
                        f"gap={gap} years"
                    ),
                    "gap_years": gap,
                    "school_before": prev["school"],
                    "school_after": curr["school"],
                    "year_before": int(prev["season_year"]),
                    "year_after": int(curr["season_year"]),
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Combine and rank ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info("Loading gamelogs...")
    gamelogs = _load_gamelogs()
    logger.info("Loaded %d game rows, %d unique player_ids",
                len(gamelogs), gamelogs["player_id"].nunique())

    logger.info("Loading bridge/draft labels...")
    bridge = _load_bridge_players()

    # ── Run checks ────────────────────────────────────────────────────────────
    logger.info("Check 1: bridge-player reappearance...")
    df1 = check_bridge_reappearance(gamelogs, bridge)
    logger.info("  Flagged: %d", len(df1))

    logger.info("Check 2: offensive ↔ defensive profile flip...")
    df2 = check_profile_flip(gamelogs)
    logger.info("  Flagged: %d", len(df2))

    logger.info("Check 3: long gap + school change...")
    df3 = check_long_gap(gamelogs)
    logger.info("  Flagged: %d (before signal count filter)", len(df3))

    # ── Merge by player_id and count signals ──────────────────────────────────
    all_flags = pd.concat([df1, df2, df3], ignore_index=True)
    if all_flags.empty:
        logger.info("No suspicious player IDs found.")
        return

    all_flags["player_id"] = all_flags["player_id"].astype(str)

    # Pivot to one row per player_id with signal counts and details
    signal_count = all_flags.groupby("player_id")["signal"].nunique().rename("n_signals")
    name_map = (all_flags.groupby("player_id")["player_name"].first())
    conf_map = (all_flags.groupby("player_id")["confidence"]
                .apply(lambda x: "HIGH" if "HIGH" in x.values else "MEDIUM"))

    # Collect all detail strings per player
    details = (all_flags.groupby("player_id")
               .apply(lambda g: " || ".join(f"[{r['signal']}] {r['detail']}"
                                            for _, r in g.iterrows()))
               .rename("all_details"))

    summary = (pd.DataFrame({"player_name": name_map, "n_signals": signal_count,
                              "confidence": conf_map})
               .join(details)
               .reset_index()
               .sort_values(["n_signals", "confidence"], ascending=[False, True]))

    # Mark already-excluded players
    known_ids = {str(k) for k in KNOWN_SEASON_EXCLUSIONS}
    summary["already_excluded"] = summary["player_id"].isin(known_ids)

    # Filter to min_signals
    summary = summary[summary["n_signals"] >= args.min_signals]

    logger.info("\n=== Player ID Audit Results ===")
    logger.info("Players with ≥%d signal(s): %d", args.min_signals, len(summary))
    logger.info("Already in KNOWN_SEASON_EXCLUSIONS: %d", summary["already_excluded"].sum())
    logger.info("NEW flags requiring review: %d",
                (~summary["already_excluded"]).sum())

    # Print table of new flags
    new_flags = summary[~summary["already_excluded"]].copy()
    if not new_flags.empty:
        logger.info("\nNew flags (not yet in KNOWN_SEASON_EXCLUSIONS):")
        for _, row in new_flags.iterrows():
            logger.info(
                "  player_id=%-10s  %-30s  signals=%d  conf=%-6s\n    %s",
                row["player_id"], row["player_name"], row["n_signals"],
                row["confidence"], row["all_details"],
            )
    else:
        logger.info("No new flags — all suspicious players already excluded.")

    if summary["already_excluded"].any():
        known_in = summary[summary["already_excluded"]]
        logger.info("\nAlready excluded (for reference):")
        for _, row in known_in.iterrows():
            logger.info("  ✓ player_id=%-10s  %s", row["player_id"], row["player_name"])

    # Save full results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_path, index=False)
    logger.info("\nSaved full audit → %s", out_path)

    # Also save the raw per-signal rows for deeper inspection
    raw_path = out_path.parent / (out_path.stem + "_raw.parquet")
    all_flags.to_parquet(raw_path, index=False)
    logger.info("Saved per-signal detail → %s", raw_path)

    logger.info(
        "\nTo suppress a confirmed collision, add to KNOWN_SEASON_EXCLUSIONS in config.py:\n"
        "    <player_id (int)>: [<bad_season_year>, ...]"
    )


if __name__ == "__main__":
    main()
