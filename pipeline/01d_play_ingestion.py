"""
Step 1d: Play-by-Play Ingestion

Fetches CFBD `PlaysApi.get_plays` (per (year, week)) and `get_play_stats`
(per (year, week, team) — 2000-row cap → must paginate by team).

Resumes automatically from existing parquet files on interruption. The
ingestion is idempotent and safe to re-run.

Usage:
    python pipeline/01d_play_ingestion.py                # full backfill
    python pipeline/01d_play_ingestion.py --test         # 1 week, 1 team
    python pipeline/01d_play_ingestion.py --plays-only
    python pipeline/01d_play_ingestion.py --play-stats-only
    python pipeline/01d_play_ingestion.py --start-year 2020
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import END_YEAR, PLAY_START_YEAR, RAW_DIR
from src.scrapers.cfbd import fetch_plays, fetch_play_stats
from src.utils.parquet_checkpoint import write_part, read_with_parts, consolidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLAYS_PATH        = RAW_DIR / "plays.parquet"
PLAY_STATS_PATH   = RAW_DIR / "play_stats.parquet"
SCHEDULE_PATH     = RAW_DIR / "game_schedule.parquet"

# Save checkpoint every N (year, week) iterations for plays, every N
# (year, week, team) iterations for play_stats. Survives quota errors.
PLAYS_CHECKPOINT_WEEKS = 4
PLAY_STATS_CHECKPOINT_TEAM_WEEKS = 200

MAX_WEEK = 16  # regular season + conference championship weeks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=PLAY_START_YEAR)
    parser.add_argument("--end-year",   type=int, default=END_YEAR)
    parser.add_argument("--test",       action="store_true",
                        help="Single (year, week, team) probe per year")
    parser.add_argument("--plays-only",        action="store_true")
    parser.add_argument("--play-stats-only",   action="store_true")
    return parser.parse_args()


def _completed_year_weeks(path: Path) -> set[tuple[int, int]]:
    """Set of (year, week) pairs already on disk (main parquet + part files)."""
    df = read_with_parts(path, columns=["season_year", "week"])
    if df.empty:
        return set()
    return set(
        df.dropna()
          .astype({"season_year": int, "week": int})
          .itertuples(index=False, name=None)
    )


def _completed_team_weeks(path: Path) -> set[tuple[int, int, str]]:
    """Set of (year, week, team) tuples already on disk (main + part files)."""
    df = read_with_parts(path, columns=["season_year", "week", "team"])
    if df.empty:
        return set()
    return set(
        df.dropna()
          .astype({"season_year": int, "week": int, "team": str})
          .itertuples(index=False, name=None)
    )


def _team_universe(schedule: pd.DataFrame, year: int, week: int) -> list[str]:
    """All home + away teams that played in (year, week)."""
    sub = schedule[(schedule["season_year"] == year) & (schedule["week"] == week)]
    teams = pd.concat([sub["home_team"], sub["away_team"]]).dropna().unique()
    return sorted(teams.tolist())


def run_plays(start_year: int, end_year: int, test: bool):
    completed = _completed_year_weeks(PLAYS_PATH)
    logger.info("Plays already on disk: %d (year, week) pairs", len(completed))

    pending: list[tuple[int, int]] = []
    for year in range(start_year, end_year + 1):
        week_range = [1] if test else range(1, MAX_WEEK + 1)
        for week in week_range:
            if (year, week) not in completed:
                pending.append((year, week))

    if not pending:
        # Fold in any leftover part files from a previous interrupted run
        consolidate(PLAYS_PATH)
        logger.info("All plays already fetched for %d–%d.", start_year, end_year)
        return

    logger.info("Fetching plays for %d (year, week) pairs ...", len(pending))
    buffer: list[pd.DataFrame] = []
    for i, (year, week) in enumerate(pending, 1):
        df = fetch_plays(year, week)
        if not df.empty:
            buffer.append(df)
            logger.info("  year=%d week=%d → %d plays", year, week, len(df))
        else:
            logger.info("  year=%d week=%d → empty", year, week)

        if i % PLAYS_CHECKPOINT_WEEKS == 0 or i == len(pending):
            if buffer:
                write_part(PLAYS_PATH, pd.concat(buffer, ignore_index=True))
                logger.info("  checkpoint: saved through (year=%d, week=%d) [%d/%d]",
                            year, week, i, len(pending))
                buffer = []

    consolidate(PLAYS_PATH)


def run_play_stats(start_year: int, end_year: int, test: bool):
    if not SCHEDULE_PATH.exists():
        logger.error("game_schedule.parquet not found — run Step 1 first")
        return

    schedule = pd.read_parquet(SCHEDULE_PATH)
    schedule = schedule[
        (schedule["season_year"] >= start_year) &
        (schedule["season_year"] <= end_year)
    ]

    completed = _completed_team_weeks(PLAY_STATS_PATH)
    logger.info("Play-stats already on disk: %d (year, week, team) tuples", len(completed))

    pending: list[tuple[int, int, str]] = []
    for year in range(start_year, end_year + 1):
        week_range = [1] if test else range(1, MAX_WEEK + 1)
        for week in week_range:
            teams = _team_universe(schedule, year, week)
            if test and teams:
                teams = teams[:1]
            for team in teams:
                if (year, week, team) not in completed:
                    pending.append((year, week, team))

    if not pending:
        # Fold in any leftover part files from a previous interrupted run
        consolidate(PLAY_STATS_PATH)
        logger.info("All play-stats already fetched for %d–%d.", start_year, end_year)
        return

    logger.info("Fetching play-stats for %d (year, week, team) tuples ...",
                len(pending))

    buffer: list[pd.DataFrame] = []
    for i, (year, week, team) in enumerate(pending, 1):
        df = fetch_play_stats(year, week, team=team)
        if not df.empty:
            buffer.append(df)
        if i % 50 == 0:
            logger.info("  progress: %d/%d (latest: %d wk%d %s)",
                        i, len(pending), year, week, team)

        if i % PLAY_STATS_CHECKPOINT_TEAM_WEEKS == 0 or i == len(pending):
            if buffer:
                write_part(PLAY_STATS_PATH, pd.concat(buffer, ignore_index=True))
                logger.info("  checkpoint: saved through %d wk%d %s [%d/%d]",
                            year, week, team, i, len(pending))
                buffer = []

    consolidate(PLAY_STATS_PATH)


def main():
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.play_stats_only:
        run_play_stats(args.start_year, args.end_year, args.test)
    elif args.plays_only:
        run_plays(args.start_year, args.end_year, args.test)
    else:
        run_plays(args.start_year, args.end_year, args.test)
        run_play_stats(args.start_year, args.end_year, args.test)

    logger.info("Step 1d complete.")


if __name__ == "__main__":
    main()
