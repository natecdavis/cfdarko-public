"""
Step 00g — College all-star game invites (Senior Bowl, Shrine Bowl, NFLPA
Collegiate Bowl).

Sources (raw pages cached under data/raw/allstar/, so reruns don't refetch):
  senior   Wikipedia "{year} Senior Bowl" roster tables, 2014-2026
  shrine   shrinersbowl.com/roster/ — one table of every player since 2011
           (no 2021 game)
  nflpa    collegiate.nflpa.com roster pages via the Wayback Machine: 2014,
           2016, 2018-2023. 2015 and 2017 have no archived roster; the game
           ended after 2023.
Year = the draft year (games are played in January before the draft).

Output data/raw/allstar_rosters.parquet:
    game, year, player_name, position, school, covered
plus data/raw/allstar_coverage.parquet: (game, year) pairs with a full
roster. A player's invite feature for an uncovered (game, year) must be
NaN, not 0.

Then data/processed/allstar_invites.parquet: the rows matched to CFBD ids
against the FULL college universe — never through the bridge, or the
feature leaks the NFL outcome (the combine-join leak, 2026-10-30):
  pass 1  CollegeIdResolver (name + school + final season, as 00d)
  pass 2  a unique id with the same surname and first initial at the same
          school with a season at final season ±1 — catches formal names,
          middle names and nicknames ("Zachary Zenner", "Kyle Lawrence
          Phillips", 'Julius "Trey" Watts')
An id claimed twice in one (game, year) is dropped.

Usage:
    python pipeline/00g_allstar_ingestion.py [--refresh]
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

CACHE = RAW_DIR / "allstar"
OUT = RAW_DIR / "allstar_rosters.parquet"
COVERAGE = RAW_DIR / "allstar_coverage.parquet"
INVITES = PROCESSED_DIR / "allstar_invites.parquet"
HEADERS = {"User-Agent": "Mozilla/5.0 (cfdarko research)"}
YEARS = range(2014, 2027)
NFLPA_PAGES = {
    2014: ("20140112071436", "collegiate.nflpa.com:80/2014rosters/"),
    2016: ("20160118213751", "collegiate.nflpa.com:80/2016-game-rosters"),
    2018: ("2023", "collegiate.nflpa.com/2018-game-roster/"),
    2019: ("2023", "collegiate.nflpa.com/2019-game-roster/"),
    2020: ("2023", "collegiate.nflpa.com/2020-game-roster/"),
    2021: ("2023", "collegiate.nflpa.com/2021-roster/"),
    2022: ("2023", "collegiate.nflpa.com/2022-game-roster/"),
    2023: ("2023", "collegiate.nflpa.com/2023-game-roster/"),
}


def _get(url: str, cache_name: str, refresh: bool) -> str | None:
    path = CACHE / cache_name
    if path.exists() and not refresh:
        return path.read_text()
    r = requests.get(url, headers=HEADERS, timeout=90)
    time.sleep(1.0)
    if r.status_code != 200:
        logger.warning("  %s → HTTP %d", url, r.status_code)
        return None
    path.write_text(r.text)
    return r.text


def _clean_school(s) -> str | None:
    if not isinstance(s, str):
        return None
    s = re.sub(r"\[.*?\]", "", s).strip()
    s = re.sub(r"\s*\((?:DI|DII|DIII|FCS|NAIA)\)\s*$", "", s, flags=re.I)
    s = re.sub(r"^The\s+", "", s)
    s = re.sub(r"^University of (?:the\s+)?", "", s, flags=re.I)
    s = re.sub(r"\s+University$", "", s, flags=re.I)
    return s.strip() or None


def _clean_name(s) -> str | None:
    if not isinstance(s, str):
        return None
    s = re.sub(r"\[.*?\]|\*|†|‡", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Dotted initials: "L. J. Collier" / "J.C. Davis" → "LJ Collier" (CFBD style)
    s = re.sub(r"\b([A-Za-z])\.\s*([A-Za-z])\.(?=\s)", r"\1\2", s)
    s = re.sub(r"\b([A-Za-z])\.(?=\s)", r"\1", s)
    return s.title() if s.isupper() else (s or None)


def _promote_header(t: pd.DataFrame) -> pd.DataFrame:
    """Tables whose header sits in the first data row."""
    if all(str(c).startswith("Unnamed") or isinstance(c, int) for c in t.columns):
        t = t.copy()
        t.columns = [str(c) for c in t.iloc[0]]
        t = t.iloc[1:]
    return t


def _standardize(t: pd.DataFrame) -> pd.DataFrame | None:
    t = _promote_header(t)
    cols = {str(c).strip().lower(): c for c in t.columns}
    def find(*keys):
        for k in keys:
            for low, orig in cols.items():
                if low == k or low.startswith(k):
                    return orig
        return None
    first = find("first name", "first", "player first")
    last = find("last name", "last", "player last")
    name = find("player", "name")
    school = find("college", "school", "last school", "team(s)")
    pos = find("pos", "position")
    if school is None or (name is None and (first is None or last is None)):
        return None
    if first is not None and last is not None:
        nm = t[first].astype(str).str.strip() + " " + t[last].astype(str).str.strip()
    else:
        nm = t[name]
    out = pd.DataFrame({"player_name": nm.map(_clean_name),
                        "position": t[pos].astype(str).str.strip() if pos is not None else None,
                        "school": t[school].map(_clean_school)})
    out = out[out["player_name"].notna() & out["school"].notna()]
    out = out[~out["player_name"].str.lower().isin({"player", "name", "first name last name"})]
    return out


def senior(refresh: bool) -> list[pd.DataFrame]:
    frames = []
    for y in YEARS:
        html = _get(f"https://en.wikipedia.org/wiki/{y}_Senior_Bowl", f"senior_{y}.html", refresh)
        if html is None:
            continue
        parts = [s for t in pd.read_html(io.StringIO(html))
                 if len(t) >= 20 and (s := _standardize(t)) is not None]
        if parts:
            d = pd.concat(parts, ignore_index=True).assign(game="senior", year=y)
            frames.append(d)
            logger.info("  senior %d: %d players", y, len(d))
    return frames


def shrine(refresh: bool) -> list[pd.DataFrame]:
    html = _get("https://shrinersbowl.com/roster/", "shrine_all.html", refresh)
    if html is None:
        return []
    t = next(t for t in pd.read_html(io.StringIO(html)) if "Last Name" in t.columns)
    t = t[t["Year"].between(YEARS.start, YEARS.stop - 1)]
    d = pd.DataFrame({"player_name": (t["First Name"].astype(str) + " " + t["Last Name"].astype(str)).map(_clean_name),
                      "position": t["Pos."].astype(str), "school": t["School"].map(_clean_school),
                      "game": "shrine", "year": t["Year"].astype(int)})
    logger.info("  shrine: %s", d.groupby("year").size().to_dict())
    return [d]


def nflpa(refresh: bool) -> list[pd.DataFrame]:
    frames = []
    for y, (ts, page) in NFLPA_PAGES.items():
        html = _get(f"https://web.archive.org/web/{ts}id_/http://{page}", f"nflpa_{y}.html", refresh)
        if html is None:
            continue
        try:
            tables = pd.read_html(io.StringIO(html))
        except ValueError:
            logger.warning("  nflpa %d: no tables", y)
            continue
        parts = [s for t in tables if len(t) >= 20 and (s := _standardize(t)) is not None]
        if parts:
            d = pd.concat(parts, ignore_index=True).assign(game="nflpa", year=y)
            frames.append(d)
            logger.info("  nflpa %d: %d players", y, len(d))
    return frames


def match_ids(rosters: pd.DataFrame) -> pd.DataFrame:
    from src.utils.college_id_match import CollegeIdResolver, _norm, _norm_school
    d = rosters.assign(_last=rosters["year"] - 1).reset_index(drop=True)
    res = CollegeIdResolver(logger).resolve(d, "player_name", "school", "_last")
    d["player_id"] = res["player_id"].to_numpy()
    d["match_pass"] = res["match_pass"].to_numpy()

    frames = [pd.read_parquet(RAW_DIR / f"{f}.parquet",
                              columns=["player_id", "player_name", "school", "season_year"])
              for f in ("rosters", "player_seasons") if (RAW_DIR / f"{f}.parquet").exists()]
    u = pd.concat(frames, ignore_index=True).dropna(subset=["player_id", "player_name"])
    u["player_id"] = u["player_id"].astype(str)
    toks = u["player_name"].map(lambda x: _norm(re.sub(r'"[^"]*"', " ", x)).split())
    u["sur"] = toks.map(lambda t: t[-1] if t else "")
    u["ini"] = toks.map(lambda t: t[0][0] if t else "")
    u["sc"] = u["school"].map(_norm_school)
    u = u.drop_duplicates(["player_id", "sur", "ini", "sc", "season_year"])
    by_key = {k: g for k, g in u.groupby(["sur", "ini", "sc"])}
    n2 = 0
    for i in d.index[d["player_id"].isna()]:
        t = _norm(re.sub(r'"[^"]*"', " ", d.at[i, "player_name"])).split()
        if not t:
            continue
        g = by_key.get((t[-1], t[0][0], _norm_school(d.at[i, "school"])))
        if g is None:
            continue
        ids = g.loc[(g["season_year"] - d.at[i, "_last"]).abs() <= 1, "player_id"].unique()
        if len(ids) == 1:
            d.at[i, "player_id"], d.at[i, "match_pass"] = ids[0], "surname_initial"
            n2 += 1
    dup = d["player_id"].notna() & d.duplicated(["game", "year", "player_id"], keep=False)
    d.loc[dup, "player_id"] = None
    logger.info("Matched %d / %d invitees (%.1f%%): %s; %d by surname + initial; %d ambiguous dropped",
                int(d["player_id"].notna().sum()), len(d), 100 * d["player_id"].notna().mean(),
                d["match_pass"].value_counts().to_dict(), n2, int(dup.sum()))
    return d.drop(columns=["_last"])


def main():
    ap = argparse.ArgumentParser(description="Step 00g — all-star game rosters")
    ap.add_argument("--refresh", action="store_true", help="Refetch cached pages")
    ap.add_argument("--test", action="store_true", help="No-op for runner compatibility")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = senior(args.refresh) + shrine(args.refresh) + nflpa(args.refresh)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["game", "year", "player_name", "school"])
    out["year"] = out["year"].astype(int)
    cov = out.groupby(["game", "year"]).size().rename("n").reset_index()
    out.to_parquet(OUT, index=False)
    cov.to_parquet(COVERAGE, index=False)
    match_ids(out).to_parquet(INVITES, index=False)
    logger.info("Saved matched invites → %s", INVITES)
    logger.info("Saved %d all-star roster rows → %s", len(out), OUT)
    logger.info("Coverage (players per game-year):\n%s",
                cov.pivot(index="year", columns="game", values="n").to_string())


if __name__ == "__main__":
    main()
