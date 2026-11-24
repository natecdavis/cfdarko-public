"""
Step 00c: NFL League-Average Era Features

Computes NFL-wide per-game averages by season (pass rate, completion %, scoring,
rush rate, pace, sack rate). Joined to each player by last_college_season_year
in steps 8 and 10 to give the model an era-context signal.

Mirrors cdarko/pipeline/00c_nba_league_averages.py in structure.

Output: data/processed/nfl_league_averages.parquet
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, NFL_LG_FEATURES, END_YEAR

try:
    import nfl_data_py as nfl
    _nfl_available = True
except ImportError:
    _nfl_available = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

START_SEASON = 1999
# Latest completed NFL season (END_YEAR is the upcoming draft class year).
END_SEASON = END_YEAR - 1


def _fetch_seasonal_safe(seasons: list[int]) -> pd.DataFrame:
    """Fetch seasonal data one year at a time, skipping unavailable seasons."""
    frames = []
    for yr in seasons:
        try:
            df = nfl.import_seasonal_data([yr], s_type="REG")
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.debug("No seasonal data for %d — skipping", yr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_nfl_league_averages(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    seasons = list(range(start, end + 1))

    logger.info("Fetching seasonal player data for %d–%d...", start, end)
    seasonal = _fetch_seasonal_safe(seasons)

    want_cols = ["completions", "attempts", "passing_yards", "passing_tds",
                 "interceptions", "sacks", "carries", "rushing_yards"]
    agg = {c: "sum" for c in want_cols if c in seasonal.columns}
    league_totals = seasonal.groupby("season").agg(agg).reset_index()

    logger.info("Fetching schedules for %d–%d...", start, end)
    schedules = nfl.import_schedules(seasons)
    reg = schedules[schedules["game_type"] == "REG"].copy()

    games_per_season = reg.groupby("season").size().rename("total_games").reset_index()
    pts_per_season = (
        reg.assign(total_pts=reg["home_score"].fillna(0) + reg["away_score"].fillna(0))
        .groupby("season")["total_pts"].sum().rename("total_pts").reset_index()
    )

    lg = league_totals.merge(games_per_season, on="season", how="left")
    lg = lg.merge(pts_per_season, on="season", how="left")

    team_games = lg["total_games"] * 2  # each game has 2 offensive teams

    result = pd.DataFrame({"season_year": lg["season"]})

    if "attempts" in lg.columns:
        result["nfl_lg_pass_att_pg"]    = lg["attempts"] / team_games
    if "completions" in lg.columns and "attempts" in lg.columns:
        result["nfl_lg_completion_pct"] = lg["completions"] / lg["attempts"]
    if "passing_yards" in lg.columns:
        result["nfl_lg_pass_yards_pg"]  = lg["passing_yards"] / team_games
    if "passing_tds" in lg.columns:
        result["nfl_lg_pass_td_pg"]     = lg["passing_tds"] / team_games
    if "carries" in lg.columns:
        result["nfl_lg_rush_att_pg"]    = lg["carries"] / team_games
    if "rushing_yards" in lg.columns:
        result["nfl_lg_rush_yards_pg"]  = lg["rushing_yards"] / team_games
    if "total_pts" in lg.columns:
        result["nfl_lg_pts_pg"]         = lg["total_pts"] / team_games
    if "attempts" in lg.columns and "carries" in lg.columns:
        result["nfl_lg_plays_pg"]       = (lg["attempts"] + lg["carries"]) / team_games
    if "sacks" in lg.columns:
        result["nfl_lg_sacks_pg"]       = lg["sacks"] / team_games
    if "interceptions" in lg.columns:
        result["nfl_lg_turnovers_pg"]   = lg["interceptions"] / team_games

    result = result.sort_values("season_year").reset_index(drop=True)

    # Forward-fill seasons that nflverse hasn't published yet (current-year
    # gap). Step 8 joins league averages on `last_college_season_year`; a
    # 2025 college senior maps to season_year=2025, and missing rows there
    # become NaN nfl_lg_* features at scoring time. LightGBM's
    # default-direction split on those NaNs has been observed to push the
    # 2025 cohort's role_score up by ~0.5 (DL saturation audit, 2026-05-03).
    # Carrying the latest-known season's values forward keeps the era signal
    # neutral until real data lands.
    if not result.empty and result["season_year"].max() < end:
        latest_year = int(result["season_year"].max())
        latest_row = result[result["season_year"] == latest_year].iloc[0]
        fill_seasons = list(range(latest_year + 1, end + 1))
        fill_rows = pd.DataFrame([latest_row.to_dict() for _ in fill_seasons])
        fill_rows["season_year"] = fill_seasons
        fill_rows["is_forward_filled"] = True
        result["is_forward_filled"] = False
        result = pd.concat([result, fill_rows], ignore_index=True)
        result = result.sort_values("season_year").reset_index(drop=True)
        logger.info("Forward-filled %d season(s) %s from %d (nflverse not yet "
                    "published)", len(fill_seasons), fill_seasons, latest_year)

    return result


def main():
    if not _nfl_available:
        logger.error("nfl_data_py not installed. Run: pip install nfl-data-py")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    lg_avgs = fetch_nfl_league_averages()
    logger.info("Computed league averages for %d seasons (%d–%d)",
                len(lg_avgs), int(lg_avgs["season_year"].min()), int(lg_avgs["season_year"].max()))

    out_path = PROCESSED_DIR / "nfl_league_averages.parquet"
    lg_avgs.to_parquet(out_path, index=False)
    logger.info("Saved → %s", out_path)

    for year in [2000, 2005, 2010, 2015, 2020, 2024]:
        row = lg_avgs[lg_avgs["season_year"] == year]
        if row.empty:
            continue
        r = row.iloc[0]
        logger.info(
            "  %d: pass_att=%.1f cmp%%=%.3f pass_yds=%.1f rush_att=%.1f pts=%.1f plays=%.1f",
            year,
            r.get("nfl_lg_pass_att_pg", np.nan),
            r.get("nfl_lg_completion_pct", np.nan),
            r.get("nfl_lg_pass_yards_pg", np.nan),
            r.get("nfl_lg_rush_att_pg", np.nan),
            r.get("nfl_lg_pts_pg", np.nan),
            r.get("nfl_lg_plays_pg", np.nan),
        )

    logger.info("Step 00c complete.")


if __name__ == "__main__":
    main()
