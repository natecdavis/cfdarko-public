"""
Step 1b: Fetch Supplemental Data

Fetches and saves:
  - player_box_scores.parquet  — per-player snap counts + PPA per game
  - team_box_scores.parquet    — per-team havoc, rushing, success rates, etc.
  - game_weather.parquet       — per-game weather (requires Patreon API key)

Resumes automatically from existing parquet files on interruption.
Reads game_ids from the existing game_schedule.parquet (run Step 1 first).

Usage:
    python pipeline/01b_fetch_supplemental.py               # both
    python pipeline/01b_fetch_supplemental.py --box-scores-only
    python pipeline/01b_fetch_supplemental.py --weather-only
    python pipeline/01b_fetch_supplemental.py --start-year 2016
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import START_YEAR, END_YEAR, RAW_DIR
from src.scrapers.cfbd import fetch_advanced_box_scores, fetch_game_weather
from src.utils.parquet_checkpoint import write_part, read_with_parts, consolidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year",   type=int, default=END_YEAR)
    parser.add_argument("--box-scores-only", action="store_true")
    parser.add_argument("--weather-only",    action="store_true")
    return parser.parse_args()


CHECKPOINT_SIZE = 500  # Save a part file every N games


def _done_game_ids(path: Path) -> set:
    """Game ids already saved for a store — main parquet plus part files."""
    df = read_with_parts(path, columns=["game_id"])
    if df.empty or "game_id" not in df.columns:
        return set()
    return set(pd.to_numeric(df["game_id"], errors="coerce")
               .dropna().astype(int).unique())


def run_box_scores(start_year: int, end_year: int):
    schedule_path = RAW_DIR / "game_schedule.parquet"
    if not schedule_path.exists():
        logger.error("game_schedule.parquet not found — run Step 1 first")
        return

    schedule = pd.read_parquet(schedule_path)
    if "season_year" in schedule.columns:
        schedule = schedule[
            (schedule["season_year"] >= start_year) &
            (schedule["season_year"] <= end_year)
        ]

    all_game_ids = schedule["game_id"].dropna().unique().tolist()
    logger.info("Games in schedule (%d–%d): %d", start_year, end_year, len(all_game_ids))

    player_box_path = RAW_DIR / "player_box_scores.parquet"
    team_box_path   = RAW_DIR / "team_box_scores.parquet"

    # Resume: skip game_ids already saved. A game only counts as done when it
    # is present in BOTH the player and team stores — resuming on the player
    # store alone would permanently drop team rows for games where a crash
    # landed between the player-chunk write and the team-chunk write.
    player_done = _done_game_ids(player_box_path)
    team_done = _done_game_ids(team_box_path)
    existing_ids: set = player_done & team_done
    if player_done or team_done:
        logger.info("Resuming: %d already saved in both stores "
                    "(player=%d, team=%d), %d remaining",
                    len(existing_ids), len(player_done), len(team_done),
                    len(all_game_ids) - len(existing_ids))

    remaining = [gid for gid in all_game_ids if int(gid) not in existing_ids]

    if not remaining:
        # Fold in any leftover part files from a previous interrupted run
        consolidate(player_box_path)
        consolidate(team_box_path)
        logger.info("All box scores already fetched.")
        return

    logger.info("Fetching %d games in chunks of %d (~%.0f min total) ...",
                len(remaining), CHECKPOINT_SIZE, len(remaining) / 240)

    # Process in chunks, each saved as its own part file — survives quota
    # errors and interruptions without re-reading the growing parquet (O(N^2)).
    total_saved = len(existing_ids)
    for chunk_start in range(0, len(remaining), CHECKPOINT_SIZE):
        chunk = remaining[chunk_start: chunk_start + CHECKPOINT_SIZE]
        player_chunk, team_chunk = fetch_advanced_box_scores(chunk)

        # A re-fetched game may already exist in one store (crash between the
        # two writes) — drop those rows so resuming never duplicates them.
        if not player_chunk.empty:
            gid = pd.to_numeric(player_chunk["game_id"], errors="coerce")
            write_part(player_box_path, player_chunk[~gid.isin(player_done)])
        if not team_chunk.empty:
            gid = pd.to_numeric(team_chunk["game_id"], errors="coerce")
            write_part(team_box_path, team_chunk[~gid.isin(team_done)])

        total_saved += len(chunk)
        logger.info("Checkpoint saved — %d / %d games complete",
                    total_saved, len(all_game_ids))

    consolidate(player_box_path)
    consolidate(team_box_path)
    logger.info("Box score fetch complete.")


def run_weather(start_year: int, end_year: int):
    logger.info("Fetching weather %d–%d ...", start_year, end_year)
    weather = fetch_game_weather(start_year, end_year)
    if not weather.empty:
        out = RAW_DIR / "game_weather.parquet"
        weather.to_parquet(out, index=False)
        logger.info("Saved %d weather rows → %s", len(weather), out)
    else:
        logger.warning("No weather data returned — verify Patreon API key in .env")


def main():
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.weather_only:
        run_weather(args.start_year, args.end_year)
    elif args.box_scores_only:
        run_box_scores(args.start_year, args.end_year)
    else:
        run_box_scores(args.start_year, args.end_year)
        run_weather(args.start_year, args.end_year)


if __name__ == "__main__":
    main()
