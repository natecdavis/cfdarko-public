"""
Step 00g: Age Audit

Lists every player in player_attributes.parquet whose implied age at last
college season falls outside the plausible window (17–26). For each row,
reports the bridge_players linkage (if any) and the player's actual season
range from player_seasons.parquet — so cross-generational namesake
collisions become obvious.

The structural fix lives in pipeline/02_bridge_players.py (year-proximity
matching) + pipeline/03_player_attributes.py (post-join sanity reject).
This script is the manual escape hatch: any case the structural fix can't
handle (e.g. wikidata-derived bad date, idiosyncratic rosters glitch) gets
a printed BIRTH_YEAR_OVERRIDES suggestion the user can paste into config.py.

Usage:
    python experiments/00g_age_audit.py
    python experiments/00g_age_audit.py --min-age 18 --max-age 25
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, RAW_DIR,
    BIRTH_YEAR_OVERRIDES, MAX_BRIDGE_DERIVED_AGE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-age", type=int, default=17,
                        help="Minimum plausible age (default 17)")
    parser.add_argument("--max-age", type=int, default=MAX_BRIDGE_DERIVED_AGE,
                        help=f"Maximum plausible age (default {MAX_BRIDGE_DERIVED_AGE})")
    return parser.parse_args()


def main():
    args = parse_args()

    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    if not attrs_path.exists():
        logger.error("player_attributes.parquet not found — run Step 03 first")
        sys.exit(1)

    attrs = pd.read_parquet(attrs_path)
    if "age_at_last_college_season" not in attrs.columns:
        attrs["age_at_last_college_season"] = (
            attrs["last_college_season_year"] - attrs["birth_year"]
        )

    age = attrs["age_at_last_college_season"]
    bad = attrs[age.notna() & ((age > args.max_age) | (age < args.min_age))].copy()
    bad["age_int"] = bad["age_at_last_college_season"].astype("Int64")

    logger.info("Total players: %d", len(attrs))
    logger.info("Implausible-age rows: %d (age > %d: %d, age < %d: %d)",
                len(bad),
                args.max_age, int((age > args.max_age).sum()),
                args.min_age, int((age < args.min_age).sum()))

    if bad.empty:
        logger.info("No implausible ages found — clean.")
        return

    # Cross-reference bridge_players for the polluting pfr_id.
    bridge_path = PROCESSED_DIR / "bridge_players.parquet"
    bridge_lookup = {}
    if bridge_path.exists():
        b = pd.read_parquet(bridge_path,
                            columns=["player_id", "player_name", "college_team",
                                     "draft_year", "pfr_id"])
        b = b.dropna(subset=["player_id"])
        bridge_lookup = {
            row.player_id: row for row in b.itertuples(index=False)
        }

    # Cross-reference player_seasons for the actual season range.
    seasons_path = RAW_DIR / "player_seasons.parquet"
    season_range = {}
    if seasons_path.exists():
        ps = pd.read_parquet(seasons_path,
                             columns=["player_id", "season_year", "school"])
        agg = (ps.groupby("player_id")
                 .agg(min_season=("season_year", "min"),
                      max_season=("season_year", "max"))
                 .reset_index())
        season_range = {
            row.player_id: (int(row.min_season), int(row.max_season))
            for row in agg.itertuples(index=False)
        }

    bad = bad.sort_values("age_at_last_college_season", ascending=False)
    known_overrides = {str(k) for k in BIRTH_YEAR_OVERRIDES}

    logger.info("")
    logger.info("=== Implausible-Age Players ===")
    suggestions = []
    for row in bad.itertuples(index=False):
        pid = getattr(row, "player_id", None)
        name = getattr(row, "player_name", "?")
        school = getattr(row, "school", "?")
        by = getattr(row, "birth_year", None)
        lcs = getattr(row, "last_college_season_year", None)
        age_int = getattr(row, "age_int", None)

        bridge_row = bridge_lookup.get(pid)
        bridge_str = ""
        if bridge_row is not None:
            bridge_str = (f"  bridge: pfr_id={bridge_row.pfr_id}, "
                          f"draft_year={bridge_row.draft_year}, "
                          f"as {bridge_row.player_name}/{bridge_row.college_team}")
        actual = season_range.get(pid)
        actual_str = ""
        if actual:
            actual_str = f"  seasons: {actual[0]}–{actual[1]}"

        already = " [override exists]" if str(pid) in known_overrides else ""

        logger.info(
            "  pid=%-10s age=%-3s  birth_year=%-6s  last_college=%-6s  %-30s  %s%s%s%s",
            pid, age_int, int(by) if pd.notna(by) else "?",
            int(lcs) if pd.notna(lcs) else "?", name, school,
            bridge_str, actual_str, already,
        )

        # If we know the actual season range, suggest birth_year ≈ min_season - 18
        if actual and not already:
            suggested_by = actual[0] - 18
            suggestions.append((pid, name, suggested_by))

    if suggestions:
        logger.info("")
        logger.info("=== Suggested BIRTH_YEAR_OVERRIDES additions ===")
        logger.info("(Paste into config.py BIRTH_YEAR_OVERRIDES dict; verify each by hand)")
        for pid, name, by in suggestions:
            logger.info("    %s: %d,  # %s", pid, by, name)


if __name__ == "__main__":
    main()
