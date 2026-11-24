"""Who was in a given NFL draft's player pool.

For draft year D the output's current class is everyone whose last college
season is D-1 — underclassmen who went back to school included, ~3x a real
draft class. The pool is:

  1. last college season D-1, AND three years out of high school (first
     college season <= D-3 in rosters / stats / FCS rosters, or listed JR or
     older on the D-1 roster), AND not on any FBS roster for season D
     (left school);
  2. OR drafted / signed in D (bridge_players draft_year == D), whatever
     the last college season — this brings in players who sat out D-1;
  3. OR on the D consensus board (hasan_ranks draft_year == D), likewise.

Validated for 2026 (2026-11-01): rule 1 alone captures all 253 drafted
players and all 295 board players whose last season was 2025; the pool is
6,610 players, with position counts close to a historical class.

Limits: rule 1 needs FBS rosters for season D, so it only works after the
next season's rosters exist (late summer). Before that `draft_pool` returns
None and callers keep their old behaviour; a pre-draft version would need
the NFL's list of underclassmen granted special eligibility. An eligible
player who transferred DOWN (FCS / D2 / JUCO) for season D also leaves FBS
rosters and is wrongly included.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ids(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


def next_season_rosters_path(raw_dir: Path, draft_year: int) -> Path:
    return raw_dir / f"rosters_{draft_year}_season.parquet"


def draft_pool(players: pd.DataFrame, draft_year: int, raw_dir: Path,
               processed_dir: Path) -> pd.Series | None:
    """Boolean Series (indexed like `players`) marking the draft pool, or None
    when season-D rosters aren't available.

    `players` needs player_id and last_college_season_year."""
    roster_path = next_season_rosters_path(raw_dir, draft_year)
    if not roster_path.exists():
        logger.warning("Draft pool: %s missing — pool unknown for %d", roster_path, draft_year)
        return None
    pid = _ids(players["player_id"])
    last = players["last_college_season_year"]

    # Season D: who is back in school.
    nxt = pd.read_parquet(roster_path, columns=["player_id"])
    returned = pid.isin(set(_ids(nxt["player_id"])))

    # Three years out of high school.
    frames = []
    for p in (raw_dir / "rosters.parquet", raw_dir / "player_seasons.parquet",
              processed_dir / "fcs_rosters.parquet"):
        if p.exists():
            frames.append(pd.read_parquet(p, columns=["player_id", "season_year"]))
    seasons = pd.concat(frames, ignore_index=True).dropna()
    first = seasons.groupby(_ids(seasons["player_id"]))["season_year"].min()
    by_first = pid.map(first) <= draft_year - 3
    ros = pd.read_parquet(raw_dir / "rosters.parquet")
    if "class_year_int" in ros.columns:
        prev = ros[ros["season_year"] == draft_year - 1].dropna(subset=["player_id"])
        cls = prev.groupby(_ids(prev["player_id"]))["class_year_int"].max()
        by_class = pid.map(cls) >= 3
    else:
        by_class = pd.Series(False, index=players.index)
    eligible = (by_first.fillna(False) | by_class.fillna(False)).astype(bool)

    rule1 = (last == draft_year - 1) & eligible & ~returned

    bp = pd.read_parquet(processed_dir / "bridge_players.parquet", columns=["player_id", "draft_year"])
    signed = set(_ids(bp.loc[bp["draft_year"] == draft_year, "player_id"].dropna()))
    hr = pd.read_parquet(processed_dir / "hasan_ranks.parquet", columns=["player_id", "draft_year"])
    board = set(_ids(hr.loc[hr["draft_year"] == draft_year, "player_id"]))
    rule2, rule3 = pid.isin(signed), pid.isin(board)

    pool = (rule1 | rule2 | rule3).astype(bool)
    pool.index = players.index
    sat_out = pool & (last < draft_year - 1)
    logger.info("Draft pool %d: %d players (last season %d: %d; sat out %d: %d drafted/signed, "
                "%d board-only); %d returned to school", draft_year, int(pool.sum()),
                draft_year - 1, int((pool & (last == draft_year - 1)).sum()), draft_year - 1,
                int((sat_out & rule2).sum()), int((sat_out & rule3 & ~rule2).sum()),
                int(((last == draft_year - 1) & returned).sum()))
    return pool
