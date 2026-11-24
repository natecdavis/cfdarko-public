"""
CFBD API Scraper (cfbd v5)

Wraps the College Football Data API via the cfbd Python package.
API key stored in .env file as CFBD_API_KEY or set as environment variable.
Free key: https://collegefootballdata.com/key
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_cfbd_available = False
try:
    import cfbd
    _cfbd_available = True
except ImportError:
    logger.warning("cfbd package not installed. Run: pip install cfbd")


def _load_env_file():
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env_file()


def _get_client() -> "cfbd.ApiClient":
    if not _cfbd_available:
        raise ImportError("cfbd package required. Run: pip install cfbd")
    api_key = os.environ.get("CFBD_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "CFBD_API_KEY not set.\n"
            "Create .env in project root: CFBD_API_KEY=your_key\n"
            "Get a free key at https://collegefootballdata.com/key"
        )
    config = cfbd.Configuration()
    config.access_token = api_key
    return cfbd.ApiClient(config)


def _rate_limit(seconds: float = 0.25):
    time.sleep(seconds)


# ── Disk cache for CFBD responses ─────────────────────────────────────────────
# CFBD's free tier caps monthly calls; even the paid tier has limits. Caching
# response bodies keyed on (function name, args, kwargs) means repeated runs
# of Step 1 / Step 3 / Step 4 don't re-burn quota.
#
# Cache hits return the deserialized response (lists of cfbd model objects
# from the SDK). The cache is OFF by default; set CFBD_CACHE_DIR env to enable
# (e.g., "data/raw/cfbd_cache") so test/CI runs don't pollute production.

_cfbd_cache: "Optional[diskcache.Cache]" = None


def _get_cache():
    """Lazy-init diskcache; returns None if disabled or library missing."""
    global _cfbd_cache
    if _cfbd_cache is not None:
        return _cfbd_cache
    cache_dir = os.environ.get("CFBD_CACHE_DIR")
    if not cache_dir:
        return None
    try:
        import diskcache
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _cfbd_cache = diskcache.Cache(cache_dir, size_limit=int(8e9))  # 8 GB cap
        logger.info("CFBD disk cache enabled at %s", cache_dir)
        return _cfbd_cache
    except ImportError:
        logger.warning("CFBD_CACHE_DIR set but diskcache not installed — running uncached")
        return None


def _cache_key(fn_name: str, args: tuple, kwargs: dict) -> str:
    """Stable cache key: function name + sorted kwargs + repr(args)."""
    parts = [fn_name, repr(args)]
    for k in sorted(kwargs):
        parts.append(f"{k}={kwargs[k]!r}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _api_call_with_retry(fn, *args, retries: int = 3, backoff: float = 2.0, **kwargs):
    """Call a CFBD API function with cache-first lookup + retry on 5xx errors.

    Cache: keyed on (fn name, args, kwargs). Empty results are cached (so a
    confirmed-empty year doesn't re-fetch). Errors are NOT cached.
    """
    cache = _get_cache()
    cache_key = None
    if cache is not None:
        cache_key = _cache_key(getattr(fn, "__name__", repr(fn)), args, kwargs)
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    last_exc = None
    for attempt in range(retries):
        try:
            result = fn(*args, **kwargs)
            if cache is not None and cache_key is not None:
                try:
                    cache.set(cache_key, result)
                except Exception:
                    pass  # cache write failure is non-fatal
            return result
        except Exception as exc:
            exc_str = str(exc)
            if "503" in exc_str or "500" in exc_str or "502" in exc_str:
                wait = backoff ** attempt
                logger.warning("Server error (attempt %d/%d), retrying in %.1fs: %s",
                               attempt + 1, retries, wait, exc_str[:80])
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc


# ── Schema-tolerant JSON fetch ─────────────────────────────────────────────────
# The cfbd SDK deserializes every response into pydantic models whose fields are
# often stricter than the API itself. When CFBD returns null for a field the SDK
# declares required, the WHOLE year raises ValidationError and we lose all of it —
# e.g. `TeamSPDefense.rating` is non-optional, 2 of 121 teams have a null defense
# rating in 2004, and that cost the entire 2004 SP+ season (and with it 2004
# `team_sp_*` coverage) on the last full refetch.
#
# `_get_json` asks the SDK for the raw body instead (`_preload_content=False`,
# payload on `.raw_data`) and parses it ourselves, so a null in a field we don't
# even use can't nuke a season. Auth, rate limiting, retry and caching still come
# from the SDK path. Keys are the API's camelCase, NOT the SDK's snake_case.

def _get_json(api_method, cache_name: str, **kwargs) -> list[dict]:
    """Fetch an endpoint as raw JSON, bypassing the SDK's response models.

    `cache_name` (not the bound method's repr, which embeds an object id) is what
    makes the disk-cache key stable across processes.
    """
    def _call(**kw):
        resp = api_method(_preload_content=False, **kw)
        payload = getattr(resp, "raw_data", None)
        if not payload:
            return []
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, list) else [parsed]

    _call.__name__ = cache_name
    return _api_call_with_retry(_call, **kwargs)


def _report_year_coverage(label: str, requested: range,
                          got_years: set, failed: dict[int, str]) -> None:
    """Log per-year coverage after a multi-year fetch.

    Callers write whatever rows they got, so a year lost to an exception used to
    leave no trace beyond one per-year log line scrolling past — the file still
    saved, just quietly missing a season. Failures are surfaced at ERROR with the
    exception text; years that merely came back empty are a WARNING, because for
    the current/upcoming season that is the normal state (SP+ for an unplayed
    year does not exist yet) and must not read as a bug.
    """
    if failed:
        logger.error("%s: %d year(s) FAILED and are missing from the output: %s",
                     label, len(failed),
                     "; ".join(f"{y} ({exc[:120]})" for y, exc in sorted(failed.items())))
    empty = [y for y in requested if y not in got_years and y not in failed]
    if empty:
        logger.warning("%s: %d year(s) returned no rows (expected for seasons not "
                       "yet published): %s", label, len(empty), empty)
    logger.info("%s: %d/%d requested years present",
                label, len(got_years), len(list(requested)))


# ── Player Season Stats ────────────────────────────────────────────────────────

def fetch_player_season_stats(start_year: int, end_year: int,
                               season_type: str = "regular") -> pd.DataFrame:
    """
    Fetch per-season player stats, pivoted to one row per player-season.
    """
    client = _get_client()
    api = cfbd.StatsApi(client)

    all_records = []
    for year in range(start_year, end_year + 1):
        logger.info("Fetching player season stats: year=%d", year)
        try:
            stats = api.get_player_season_stats(year=year)
            for s in stats:
                all_records.append({
                    "player_id":   s.player_id,
                    "player_name": s.player,
                    "school":      s.team,
                    "conference":  s.conference,
                    "position":    s.position,
                    "season_year": s.season,
                    "category":    s.category,
                    "stat_type":   s.stat_type,
                    "stat":        s.stat,
                })
            _rate_limit()
        except Exception as exc:
            logger.error("Failed to fetch %d stats: %s", year, exc)

    if not all_records:
        return pd.DataFrame()

    raw = pd.DataFrame(all_records)
    raw["stat"] = pd.to_numeric(raw["stat"], errors="coerce")

    # Pivot: one row per player-season
    pivoted = raw.pivot_table(
        index=["player_id", "player_name", "school", "conference",
               "position", "season_year"],
        columns=["category", "stat_type"],
        values="stat",
        aggfunc="first",
    )
    pivoted.columns = [f"{cat}_{typ}".lower().replace(" ", "_")
                       for cat, typ in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Normalize column names to cfdarko conventions
    rename_map = {
        "passing_c/att":        "pass_att",   # completions/attempts split elsewhere
        "passing_att":          "pass_att",
        "passing_comp":         "completions",
        "passing_yds":          "pass_yards",
        "passing_td":           "pass_tds",
        "passing_int":          "interceptions",
        "passing_pct":          "completion_pct_raw",
        "passing_ya":           "yards_per_attempt",
        "rushing_car":          "rush_att",
        "rushing_att":          "rush_att",
        "rushing_yds":          "rush_yards",
        "rushing_td":           "rush_tds",
        "rushing_avg":          "yards_per_carry",
        "receiving_rec":        "rec",
        "receiving_yds":        "rec_yards",
        "receiving_td":         "rec_tds",
        "receiving_avg":        "yards_per_rec",
        "fumbles_lost":         "fumbles",
        "fumbles_fum":          "fumbles_total",
        "defensive_solo":       "tackles_solo",
        "defensive_tot":        "tackles",
        "defensive_tfl":        "tfl",
        "defensive_sacks":      "sacks",
        "defensive_pd":         "passes_defended",
        "defensive_int":        "ints",
        "defensive_ff":         "ff",
        "defensive_qbh":        "qb_hurries",
        "kicking_fga":          "fg_att",
        "kicking_fgm":          "fg_made",
        "kicking_xpa":          "xp_att",
        "kicking_xpm":          "xp_made",
        "kicking_pts":          "kicking_pts",
        "punting_yds":          "punt_yards",
        "punting_no":           "punts",
        "punting_avg":          "gross_punt_avg",
        "interceptionsint":     "ints",
    }
    pivoted = pivoted.rename(columns={k: v for k, v in rename_map.items()
                                       if k in pivoted.columns})

    # Derived stats — use where() to avoid division-by-zero without pd.NA casting issues
    def safe_div(num_col: str, den_col: str) -> "pd.Series":
        num = pivoted[num_col].astype(float)
        den = pivoted[den_col].astype(float)
        return num.where(den > 0) / den.where(den > 0)

    if "completions" in pivoted.columns and "pass_att" in pivoted.columns:
        pivoted["completion_pct"] = safe_div("completions", "pass_att")

    if "rec" in pivoted.columns and "rec_yards" in pivoted.columns:
        pivoted["yards_per_rec"] = safe_div("rec_yards", "rec")

    if "rush_att" in pivoted.columns and "rush_yards" in pivoted.columns:
        pivoted["yards_per_carry"] = safe_div("rush_yards", "rush_att")

    if "fg_made" in pivoted.columns and "fg_att" in pivoted.columns:
        pivoted["fg_pct"] = safe_div("fg_made", "fg_att")

    if "xp_made" in pivoted.columns and "xp_att" in pivoted.columns:
        pivoted["xp_pct"] = safe_div("xp_made", "xp_att")

    return pivoted


# ── Team Ratings (SP+) ─────────────────────────────────────────────────────────

def fetch_team_conferences(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch each FBS team's conference AS OF each season.

    Exists because the SP+ endpoint reports a team's CURRENT conference on every
    historical row — it calls 2004 USC "Big Ten" and 2004 Utah "Big 12" — and
    03_player_attributes was using that as its per-(school, season) conference
    fallback for exactly the pre-2010 era where the anachronism is worst. The
    `/teams/fbs` endpoint is year-aware and gives the real affiliation: Utah 2004
    Mountain West → 2011 Pac-12, TCU 2004 Conference USA → 2012 Big 12.

    Every conference name this returns for 2004-2025 is already covered by
    config.CONFERENCE_TIERS, including the retired ones (Pac-10, Big East,
    Western Athletic), so tiering needs no extra mapping.
    """
    api = cfbd.TeamsApi(_get_client())

    records: list[dict] = []
    failed: dict[int, str] = {}
    got_years: set = set()
    years = range(start_year, end_year + 1)

    for year in years:
        logger.info("Fetching FBS team conferences: year=%d", year)
        try:
            teams = _api_call_with_retry(api.get_fbs_teams, year=year)
        except Exception as exc:
            failed[year] = str(exc)
            logger.error("Failed to fetch FBS teams for %d: %s", year, exc)
            continue
        for t in teams:
            school = getattr(t, "school", None)
            conference = getattr(t, "conference", None)
            if not school or not conference:
                continue
            records.append({
                "school":      school,
                "season_year": year,
                "conference":  conference,
            })
        if teams:
            got_years.add(year)
        _rate_limit()

    _report_year_coverage("FBS team conferences", years, got_years, failed)
    return pd.DataFrame(records) if records else pd.DataFrame()


def fetch_team_ratings(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch SP+ ratings per team per season.

    Goes through `_get_json` rather than `api.get_sp`: the SDK's TeamSPDefense
    model requires a non-null `rating`, and a single null (2 teams in 2004) makes
    it throw away the entire season. Reading the raw payload means a null lands in
    one cell instead of deleting a year. Only the three columns below are consumed
    downstream (03_player_attributes → team_sp_offense / _defense / _overall), and
    2004 has all 121 teams for overall+offense, 119 for defense.

    `conference` is emitted as **conference_current**: this endpoint reports the
    team's PRESENT-DAY conference on every historical row (2004 USC comes back
    "Big Ten", 2004 Utah "Big 12"). The name makes that explicit so it is not
    reused as a per-season affiliation again — for that, use
    `fetch_team_conferences`, which is year-aware.
    """
    api = cfbd.RatingsApi(_get_client())

    records: list[dict] = []
    failed: dict[int, str] = {}
    got_years: set = set()
    years = range(start_year, end_year + 1)

    for year in years:
        logger.info("Fetching SP+ ratings: year=%d", year)
        try:
            ratings = _get_json(api.get_sp_with_http_info, "get_sp_json", year=year)
        except Exception as exc:
            failed[year] = str(exc)
            logger.error("Failed to fetch SP+ for %d: %s", year, exc)
            continue
        for r in ratings:
            offense = r.get("offense") or {}
            defense = r.get("defense") or {}
            records.append({
                "school":             r.get("team"),
                # NOT the season's conference — see docstring.
                "conference_current": r.get("conference"),
                "season_year":        r.get("year") or year,
                "sp_plus":            r.get("rating"),
                "sp_plus_offense":    offense.get("rating"),
                "sp_plus_defense":    defense.get("rating"),
            })
        if ratings:
            got_years.add(year)
        _rate_limit()

    _report_year_coverage("SP+ ratings", years, got_years, failed)
    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Team Advanced Stats ────────────────────────────────────────────────────────

def fetch_team_advanced_stats(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch advanced team stats (pass rate, pace proxy) per season.

    `offense.passing_plays` is a nested object (explosiveness / success_rate /
    total_ppa / ppa / rate), NOT a play count — so the original
    `passing_plays / plays` raised TypeError on every single year and this fetch
    never once produced output (data/raw/team_advanced.parquet did not exist).
    The play share is served directly as `.rate`; verified against 2023, where it
    reads 0.135 for Air Force's triple option and 0.628 for Washington State's Air
    Raid, and pass+rush sum to ~1.0.

    The former `passing_plays` / `rushing_plays` count columns are gone: the API
    exposes no such counts, and deriving `rate * plays` would invent a precision
    the source doesn't have. Nothing consumed them (the file never existed).
    """
    api = cfbd.StatsApi(_get_client())

    records: list[dict] = []
    failed: dict[int, str] = {}
    got_years: set = set()
    years = range(start_year, end_year + 1)

    for year in years:
        logger.info("Fetching team advanced stats: year=%d", year)
        try:
            stats = _api_call_with_retry(api.get_advanced_season_stats, year=year)
        except Exception as exc:
            failed[year] = str(exc)
            logger.error("Failed advanced stats %d: %s", year, exc)
            continue
        for s in stats:
            offense = s.offense
            if offense is None:
                continue
            passing = getattr(offense, "passing_plays", None)
            rushing = getattr(offense, "rushing_plays", None)
            records.append({
                "school":       s.team,
                "season_year":  s.season,
                "pass_rate":    getattr(passing, "rate", None),
                "rush_rate":    getattr(rushing, "rate", None),
                "total_plays":  getattr(offense, "plays", None),
                "success_rate": getattr(offense, "success_rate", None),
                "ppa":          getattr(offense, "ppa", None),
            })
        if stats:
            got_years.add(year)
        _rate_limit()

    _report_year_coverage("team advanced stats", years, got_years, failed)
    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Draft Picks ────────────────────────────────────────────────────────────────

def fetch_draft_picks(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch NFL draft picks with college info for bridge player identification."""
    client = _get_client()
    api = cfbd.DraftApi(client)

    records = []
    for year in range(start_year, end_year + 1):
        logger.info("Fetching draft picks: year=%d", year)
        try:
            picks = api.get_draft_picks(year=year)
            for p in picks:
                records.append({
                    "draft_year":    p.year,
                    "player_name":   p.name,
                    "college_team":  p.college_team,
                    "conference":    p.college_conference,
                    "nfl_team":      p.nfl_team,
                    "position":      p.position,
                    "round":         p.round,
                    "pick":          p.pick,
                    "overall_pick":  p.overall,
                    "height":        p.height,
                    "weight":        p.weight,
                    # CFBD athlete id, the same id space as player_seasons /
                    # rosters player_id: step 02 links drafted players on it
                    # before any name matching.
                    "college_athlete_id": p.college_athlete_id,
                })
            _rate_limit()
        except Exception as exc:
            logger.error("Failed draft picks %d: %s", year, exc)

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Player Game Stats ─────────────────────────────────────────────────────────


def _parse_slash_stat(stat_str: str) -> tuple[int, int]:
    """Parse 'made/att' format (e.g. '2/3' → (2, 3)). Returns (0, 0) on failure."""
    try:
        parts = str(stat_str).split("/")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def _extract_game_player_stats(game, season_year: int, week: int) -> list[dict]:
    """
    Parse one game's player stats from the CFBD nested structure.
    Returns list of per-player-game dicts.
    GamePlayerStats only exposes: id (game_id), teams.
    season_year and week are passed in from the calling loop.
    """
    records = []
    game_id = game.id

    if not game.teams:
        return records

    for team_data in game.teams:
        team_name = team_data.team
        is_home = int(getattr(team_data, "home_away", "away") == "home")

        if not team_data.categories:
            continue

        # Accumulate stats per player
        players: dict[str, dict] = {}

        for category in team_data.categories:
            cat = (category.name or "").lower()

            if not category.types:
                continue

            for stat_type in category.types:
                type_name = stat_type.name or ""

                if not stat_type.athletes:
                    continue

                for athlete in stat_type.athletes:
                    pid = str(athlete.id)
                    if pid not in players:
                        players[pid] = {
                            "player_id": pid,
                            "player_name": athlete.name,
                            "school": team_name,
                            "game_id": game_id,
                            "season_year": season_year,
                            "week": week,
                            "is_home": is_home,
                        }
                    row = players[pid]
                    val_str = str(athlete.stat)

                    # Slash-format stats
                    if "/" in val_str:
                        made, att = _parse_slash_stat(val_str)
                        if cat == "passing" and "c" in type_name.lower():
                            row["completions"] = made
                            row["pass_att"] = att
                        elif cat == "kicking" and type_name.upper() == "FG":
                            row["fg_made"] = made
                            row["fg_att"] = att
                        elif cat == "kicking" and type_name.upper() == "XP":
                            row["xp_made"] = made
                            row["xp_att"] = att
                        continue

                    try:
                        val = float(val_str)
                    except (ValueError, TypeError):
                        continue

                    col_map = {
                        ("passing",       "YDS"):    "pass_yards",
                        ("passing",       "TD"):     "pass_tds",
                        ("passing",       "INT"):    "interceptions",
                        ("passing",       "AVG"):    "yards_per_attempt",
                        ("rushing",       "CAR"):    "rush_att",
                        ("rushing",       "YDS"):    "rush_yards",
                        ("rushing",       "TD"):     "rush_tds",
                        ("rushing",       "AVG"):    "yards_per_carry",
                        ("receiving",     "REC"):    "rec",
                        ("receiving",     "YDS"):    "rec_yards",
                        ("receiving",     "TD"):     "rec_tds",
                        ("receiving",     "AVG"):    "yards_per_rec",
                        ("defensive",     "TOT"):    "tackles",
                        ("defensive",     "SOLO"):   "tackles_solo",
                        ("defensive",     "SACKS"):  "sacks",
                        ("defensive",     "TFL"):    "tfl",
                        ("defensive",     "PD"):     "passes_defended",
                        ("defensive",     "QB HUR"): "qb_hurries",
                        ("defensive",     "FF"):     "ff",
                        ("defensive",     "TD"):     "def_tds",
                        ("fumbles",       "FUM"):    "fumbles_total",
                        ("fumbles",       "LOST"):   "fumbles",
                        ("kicking",       "PTS"):    "kicking_pts",
                        ("punting",       "YDS"):    "punt_yards",
                        ("punting",       "NO"):     "punts",
                        ("punting",       "AVG"):    "gross_punt_avg",
                        ("interceptions", "INT"):    "ints",
                        ("interceptions", "YDS"):    "int_yards",
                        ("interceptions", "TD"):     "int_tds",
                        ("kickreturns",   "YDS"):    "kr_yards",
                        ("kickreturns",   "TD"):     "kr_tds",
                        ("puntreturns",   "YDS"):    "pr_yards",
                        ("puntreturns",   "TD"):     "pr_tds",
                    }
                    col = col_map.get((cat, type_name.upper()))
                    if col:
                        row[col] = val

        records.extend(players.values())

    return records


def fetch_player_game_stats(start_year: int, end_year: int,
                             season_type: str = "regular",
                             max_week: int = 16) -> pd.DataFrame:
    """
    Fetch per-game player stats from CFBD by iterating year × week.
    The API requires week, team, or conference — we iterate by week.
    Returns one row per player-game with raw counting stats.
    """
    client = _get_client()
    api = cfbd.GamesApi(client)

    all_records: list[dict] = []
    for year in range(start_year, end_year + 1):
        logger.info("Fetching player game stats: year=%d (weeks 1–%d)", year, max_week)
        year_rows_start = len(all_records)
        for week in range(1, max_week + 1):
            try:
                games = api.get_game_player_stats(year=year, week=week,
                                                   season_type=season_type)
                for game in games:
                    all_records.extend(_extract_game_player_stats(game, year, week))
                _rate_limit(0.15)
            except Exception as exc:
                exc_str = str(exc)
                if "400" not in exc_str and "404" not in exc_str:
                    logger.warning("Week %d/%d game stats: %s", year, week, exc)

        logger.info("  year=%d: %d player-game rows", year, len(all_records) - year_rows_start)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # Coerce stat columns to numeric
    stat_cols = [c for c in df.columns if c not in
                 {"player_id", "player_name", "school", "game_id",
                  "season_year", "week", "is_home"}]
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived rate stats (raw denominators kept for noise scaling)
    def _safe_div(num: str, den: str) -> "pd.Series":
        n = df[num].astype(float) if num in df.columns else pd.Series(float("nan"), index=df.index)
        d = df[den].astype(float) if den in df.columns else pd.Series(float("nan"), index=df.index)
        return n.where(d > 0) / d.where(d > 0)

    if "completions" in df.columns and "pass_att" in df.columns:
        df["completion_pct"] = _safe_div("completions", "pass_att")
    if "rush_yards" in df.columns and "rush_att" in df.columns:
        df["yards_per_carry"] = _safe_div("rush_yards", "rush_att")
    if "rec_yards" in df.columns and "rec" in df.columns:
        df["yards_per_rec"] = _safe_div("rec_yards", "rec")
    if "fg_made" in df.columns and "fg_att" in df.columns:
        df["fg_pct"] = _safe_div("fg_made", "fg_att")
    if "xp_made" in df.columns and "xp_att" in df.columns:
        df["xp_pct"] = _safe_div("xp_made", "xp_att")

    return df


# ── Game Schedule ──────────────────────────────────────────────────────────────


def fetch_game_schedule(start_year: int, end_year: int,
                         season_type: str = "regular") -> pd.DataFrame:
    """
    Fetch game schedule with dates, home/away, and neutral site info.
    Returns one row per game.

    `game_hour` is the LOCAL kickoff hour at the home stadium (0-23),
    derived from CFBD's UTC timestamp + SCHOOL_TIMEZONE_OFFSET on home_team.
    Standard-offset only (no DST correction) — error capped at 1hr during
    Aug-early Nov window where games actually occur.
    """
    from config import SCHOOL_TIMEZONE_OFFSET, DEFAULT_TIMEZONE_OFFSET

    client = _get_client()
    api = cfbd.GamesApi(client)

    records: list[dict] = []
    for year in range(start_year, end_year + 1):
        logger.info("Fetching game schedule: year=%d", year)
        try:
            games = _api_call_with_retry(api.get_games, year=year, season_type=season_type)
            for g in games:
                start_date = getattr(g, "start_date", None) or ""
                try:
                    ts = pd.to_datetime(start_date)
                    game_date = ts.date()
                    utc_hour = int(ts.hour)
                except Exception:
                    game_date = None
                    utc_hour = None

                home_team = g.home_team
                # Convert UTC → local hour at home stadium (standard offset).
                if utc_hour is not None:
                    tz_offset = SCHOOL_TIMEZONE_OFFSET.get(
                        home_team, DEFAULT_TIMEZONE_OFFSET
                    )
                    game_hour = (utc_hour + int(tz_offset)) % 24
                else:
                    game_hour = None

                records.append({
                    "game_id":      g.id,
                    "season_year":  g.season,
                    "week":         g.week,
                    "game_date":    game_date,
                    "game_hour":    game_hour,
                    "home_team":    home_team,
                    "away_team":    g.away_team,
                    "is_neutral":   int(bool(getattr(g, "neutral_site", False))),
                    "venue_id":     getattr(g, "venue_id", None),
                    "venue_name":   getattr(g, "venue", None),
                })
            _rate_limit()
        except Exception as exc:
            logger.error("Failed to fetch schedule for %d: %s", year, exc)

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Advanced Game Stats ────────────────────────────────────────────────────────


def fetch_advanced_game_stats(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch per-game team advanced stats (PPA, success rate, stuff rate).
    Returns one row per team-game with offense and defense metrics.
    Used to compute rolling opponent strength.
    """
    client = _get_client()
    api = cfbd.StatsApi(client)

    records: list[dict] = []
    for year in range(start_year, end_year + 1):
        logger.info("Fetching advanced game stats: year=%d", year)
        try:
            stats = _api_call_with_retry(api.get_advanced_game_stats, year=year)
            for s in stats:
                off = s.offense or object()
                defe = s.defense or object()

                def _attr(obj, name):
                    return getattr(obj, name, None)

                records.append({
                    "game_id":           s.game_id,
                    "season_year":       s.season,
                    "week":              s.week,
                    "team":              s.team,
                    "opponent":          s.opponent,
                    "off_ppa":           _attr(off,  "ppa"),
                    "off_success_rate":  _attr(off,  "success_rate"),
                    "off_explosiveness": _attr(off,  "explosiveness"),
                    "off_stuff_rate":    _attr(off,  "stuff_rate"),
                    "def_ppa":           _attr(defe, "ppa"),
                    "def_success_rate":  _attr(defe, "success_rate"),
                    "def_stuff_rate":    _attr(defe, "stuff_rate"),
                    "def_explosiveness": _attr(defe, "explosiveness"),
                })
            _rate_limit()
        except Exception as exc:
            logger.error("Failed to fetch advanced game stats for %d: %s", year, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ["off_ppa", "off_success_rate", "def_ppa", "def_success_rate",
                "def_stuff_rate", "off_stuff_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Game Weather ───────────────────────────────────────────────────────────────


def fetch_game_weather(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch per-game weather data. Gracefully returns empty DataFrame if
    the API key lacks Patreon access.
    """
    client = _get_client()
    api = cfbd.GamesApi(client)

    records: list[dict] = []
    for year in range(start_year, end_year + 1):
        logger.info("Fetching game weather: year=%d", year)
        try:
            weather_list = api.get_weather(year=year)
            for w in weather_list:
                records.append({
                    "game_id":              w.id,
                    "temperature":          getattr(w, "temperature", None),
                    "wind_speed":           getattr(w, "wind_speed", None),
                    "wind_direction":       getattr(w, "wind_direction", None),
                    "dew_point":            getattr(w, "dew_point", None),
                    "precipitation":        getattr(w, "precipitation", None),
                    "snowfall":             getattr(w, "snowfall", None),
                    "pressure":             getattr(w, "pressure", None),
                    "humidity":             getattr(w, "humidity", None),
                    "game_indoors":         int(bool(getattr(w, "game_indoors", False))),
                    "weather_condition_code": getattr(w, "weather_condition_code", None),
                })
            _rate_limit()
        except Exception as exc:
            logger.warning("Weather fetch failed for %d (may need Patreon): %s", year, exc)

    if not records:
        logger.info("No weather data fetched — fields will be NaN")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ["temperature", "wind_speed", "wind_direction", "dew_point",
                "precipitation", "snowfall", "pressure", "humidity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Advanced Box Scores ────────────────────────────────────────────────────────


def _safe_nested(obj, *attrs):
    """Safely traverse nested attributes, returning None if any is missing."""
    for attr in attrs:
        if obj is None:
            return None
        obj = getattr(obj, attr, None)
    return obj


def fetch_advanced_box_scores(
        game_ids: list,
        retries: int = 3,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Fetch advanced box scores for a list of game IDs (~1 API call per game).

    Returns:
        player_df  — per-player-game: snap counts (total/rushing/passing/Q1-Q4)
                     and PPA (average and cumulative, by play type)
        team_df    — per-team-game: havoc, rushing breakdown, success rates,
                     scoring efficiency, field position, PPA, explosiveness
    """
    client = _get_client()
    api = cfbd.GamesApi(client)

    player_records: list[dict] = []
    team_records: list[dict] = []
    n = len(game_ids)

    for i, game_id in enumerate(game_ids):
        if i % 100 == 0:
            logger.info("Advanced box scores: %d / %d games", i, n)
        try:
            box = _api_call_with_retry(api.get_advanced_box_score,
                                       id=game_id, retries=retries)
        except Exception as exc:
            logger.debug("Box score %s: %s", game_id, exc)
            _rate_limit()
            continue

        # ── Player snap counts ─────────────────────────────────────────────────
        usage_by_key: dict[tuple, dict] = {}
        if box.players and box.players.usage:
            for p in box.players.usage:
                key = (p.player, p.team)
                usage_by_key[key] = {
                    "position":      p.position,
                    "snaps":         p.total,
                    "snaps_rushing": p.rushing,
                    "snaps_passing": p.passing,
                    "snaps_q1":      p.quarter1,
                    "snaps_q2":      p.quarter2,
                    "snaps_q3":      p.quarter3,
                    "snaps_q4":      p.quarter4,
                }

        # ── Player PPA ─────────────────────────────────────────────────────────
        ppa_by_key: dict[tuple, dict] = {}
        if box.players and box.players.ppa:
            for p in box.players.ppa:
                key = (p.player, p.team)
                ppa_by_key[key] = {
                    "position":         p.position,
                    "ppa_avg":          _safe_nested(p.average,    "total"),
                    "ppa_avg_rushing":  _safe_nested(p.average,    "rushing"),
                    "ppa_avg_passing":  _safe_nested(p.average,    "passing"),
                    "ppa_cum":          _safe_nested(p.cumulative, "total"),
                    "ppa_cum_rushing":  _safe_nested(p.cumulative, "rushing"),
                    "ppa_cum_passing":  _safe_nested(p.cumulative, "passing"),
                }

        for key in set(usage_by_key) | set(ppa_by_key):
            player_name, team = key
            u  = usage_by_key.get(key, {})
            pp = ppa_by_key.get(key, {})
            player_records.append({
                "game_id":     game_id,
                "player_name": player_name,
                "team":        team,
                "position":    u.get("position") or pp.get("position"),
                **{k: v for k, v in u.items()  if k != "position"},
                **{k: v for k, v in pp.items() if k != "position"},
            })

        # ── Team advanced stats ────────────────────────────────────────────────
        if box.teams:
            havoc_map  = {h.team: h for h in (box.teams.havoc                or [])}
            rush_map   = {r.team: r for r in (box.teams.rushing              or [])}
            succ_map   = {s.team: s for s in (box.teams.success_rates        or [])}
            score_map  = {s.team: s for s in (box.teams.scoring_opportunities or [])}
            fp_map     = {f.team: f for f in (box.teams.field_position       or [])}
            ppa_map    = {p.team: p for p in (box.teams.ppa                  or [])}
            expl_map   = {e.team: e for e in (box.teams.explosiveness        or [])}

            all_teams = (set(havoc_map) | set(rush_map) | set(succ_map)
                         | set(score_map) | set(fp_map) | set(ppa_map))
            for team in all_teams:
                h  = havoc_map.get(team)
                r  = rush_map.get(team)
                s  = succ_map.get(team)
                sc = score_map.get(team)
                fp = fp_map.get(team)
                p  = ppa_map.get(team)
                ex = expl_map.get(team)
                team_records.append({
                    "game_id": game_id,
                    "team":    team,
                    # Havoc
                    "havoc_total":              _safe_nested(h,  "total"),
                    "havoc_front_seven":        _safe_nested(h,  "front_seven"),
                    "havoc_db":                 _safe_nested(h,  "db"),
                    # Rushing breakdown
                    "line_yards_avg":           _safe_nested(r,  "line_yards_average"),
                    "second_level_yards_avg":   _safe_nested(r,  "second_level_yards_average"),
                    "open_field_yards_avg":     _safe_nested(r,  "open_field_yards_average"),
                    "rushing_power_success":    _safe_nested(r,  "power_success"),
                    "rushing_stuff_rate":       _safe_nested(r,  "stuff_rate"),
                    # Success rates
                    "success_rate_overall":        _safe_nested(s,  "overall",        "total"),
                    "success_rate_standard_downs": _safe_nested(s,  "standard_downs", "total"),
                    "success_rate_passing_downs":  _safe_nested(s,  "passing_downs",  "total"),
                    # Scoring efficiency
                    "scoring_opportunities":    _safe_nested(sc, "opportunities"),
                    "points_per_opportunity":   _safe_nested(sc, "points_per_opportunity"),
                    # Field position
                    "avg_starting_field_pos":   _safe_nested(fp, "average_start"),
                    # Team PPA
                    "team_ppa":         _safe_nested(p,  "overall", "total"),
                    "team_ppa_passing": _safe_nested(p,  "passing", "total"),
                    "team_ppa_rushing": _safe_nested(p,  "rushing", "total"),
                    # Explosiveness
                    "team_explosiveness": _safe_nested(ex, "overall", "total"),
                })

        _rate_limit()

    return (
        pd.DataFrame(player_records) if player_records else pd.DataFrame(),
        pd.DataFrame(team_records)   if team_records   else pd.DataFrame(),
    )


# ── Rolling Opponent Strength ──────────────────────────────────────────────────


def compute_rolling_opponent_strength(
        adv_game_stats: pd.DataFrame,
        schedule: pd.DataFrame,
        halflife_days: int = 45,
) -> pd.DataFrame:
    """
    For each team-game, compute exponential-decay-weighted average of the
    opponent's defensive/offensive PPA over all of the opponent's PRIOR games.

    Mirrors cdarko's compute_rolling_team_strength() approach. Delegates the
    strictly-prior, exp-decay-weighted aggregation to the shared primitive in
    ``src/utils/rolling.py``.

    Args:
        adv_game_stats: DataFrame with [game_id, season_year, week, team,
                        opponent, def_ppa, off_ppa, def_success_rate, def_stuff_rate]
        schedule:       DataFrame with [game_id, game_date, home_team, away_team]
        halflife_days:  Exponential decay halflife in days.

    Returns:
        DataFrame with [game_id, team, opp_rolling_def_ppa, opp_rolling_off_ppa,
                        opp_rolling_def_success_rate, opp_rolling_def_stuff_rate, ...]
    """
    from src.utils.rolling import compute_rolling_entity_strength

    if adv_game_stats.empty or schedule.empty:
        return pd.DataFrame()

    # Join game_date onto advanced stats so panel + targets share the same
    # date column.
    sched_slim = schedule[["game_id", "game_date"]].drop_duplicates("game_id")
    df = adv_game_stats.merge(sched_slim, on="game_id", how="left")

    # Stat list matches the original implementation's order (one row per team
    # per game; "opp_rolling_<stat>" emits the OPPONENT's prior-game rolling
    # value via the panel_entity="team" / target_entity="opponent" lookup).
    stat_cols = [
        "def_ppa", "off_ppa",
        "def_success_rate", "def_stuff_rate", "def_explosiveness",
        "off_success_rate", "off_explosiveness",
        "havoc_total", "havoc_front_seven", "havoc_db",
    ]

    return compute_rolling_entity_strength(
        panel=df,
        targets=df,
        panel_entity_col="team",
        target_entity_col="opponent",
        date_col="game_date",
        stat_cols=stat_cols,
        halflife_days=halflife_days,
        weight_col=None,
        out_prefix="opp_rolling",
        keep_target_cols=["game_id", "team"],
    )


# ── Teammate-QB Rolling Quality ───────────────────────────────────────────────


def compute_qb_rolling_stats(
        gamelogs: pd.DataFrame,
        halflife_days: int = 45,
        min_pass_att: int = 5,
) -> pd.DataFrame:
    """
    For each (game_id, school), compute the team's QBs' decay-weighted rate
    stats over their strictly-prior games at the same school.

    Pipeline:
      1. Filter gamelogs to position_group == "QB"; drop rows with
         pass_att < min_pass_att (garbage-time / snow-game noise).
      2. Per QB-game, compute the 6 rate stats. NCAA passer rating is
         computed inline from the raw counters.
      3. Per (game_id, school): pass_att-weighted blend across all QBs who
         threw → one panel observation per team-game.
      4. Per (game_id, school) target row: exponential-decay-weighted EWMA
         of the team's prior panel rows (strictly prior, weighted by
         sqrt(pass_att) so a 50-attempt blowout doesn't dominate a 15-att
         normal game).

    Args:
        gamelogs:        DataFrame with [player_id, game_id, school, game_date,
                         position_group, pass_att, pass_yards, pass_tds,
                         completions, interceptions, completion_pct,
                         yards_per_attempt, ppa_avg_passing].
        halflife_days:   Exponential decay halflife in days.
        min_pass_att:    Drop QB-game rows below this from the prior panel.

    Returns:
        DataFrame with [game_id, school, qb_rolling_passer_rating,
                        qb_rolling_completion_pct, qb_rolling_ppa_avg_passing,
                        qb_rolling_ypa, qb_rolling_td_rate, qb_rolling_int_rate].
    """
    from src.utils.rolling import compute_rolling_entity_strength

    if gamelogs.empty:
        return pd.DataFrame()

    required = {"player_id", "game_id", "school", "game_date",
                "position_group", "pass_att"}
    if not required.issubset(gamelogs.columns):
        missing = required - set(gamelogs.columns)
        logger.warning("compute_qb_rolling_stats: missing required cols %s",
                       missing)
        return pd.DataFrame()

    optional_numeric = ["pass_yards", "pass_tds", "completions", "interceptions",
                        "completion_pct", "yards_per_attempt", "ppa_avg_passing"]
    keep = list(required) + [c for c in optional_numeric if c in gamelogs.columns]

    # Targets first — we always emit one row per (game_id, school), even if
    # no qualifying QB rows exist. NaN qb_rolling_* is the right output then.
    if "game_date" not in gamelogs.columns:
        return pd.DataFrame()
    targets = (gamelogs[["game_id", "school", "game_date"]]
               .drop_duplicates(["game_id", "school"]))

    out_cols = [
        "qb_rolling_passer_rating", "qb_rolling_completion_pct",
        "qb_rolling_ppa_avg_passing", "qb_rolling_ypa",
        "qb_rolling_td_rate", "qb_rolling_int_rate",
    ]

    def _empty_output() -> pd.DataFrame:
        out = targets[["game_id", "school"]].copy().reset_index(drop=True)
        for col in out_cols:
            out[col] = float("nan")
        return out

    qb = gamelogs.loc[gamelogs["position_group"] == "QB", keep].copy()
    if qb.empty:
        return _empty_output()

    for c in ["pass_att"] + optional_numeric:
        if c in qb.columns:
            qb[c] = pd.to_numeric(qb[c], errors="coerce")
    qb = qb[qb["pass_att"] >= float(min_pass_att)]
    if qb.empty:
        return _empty_output()

    # NCAA passer rating: (8.4·yds + 330·TDs + 100·COMPS - 200·INTs) / pass_att
    pa = qb["pass_att"].clip(lower=1.0)
    if {"pass_yards", "pass_tds", "completions", "interceptions"}.issubset(qb.columns):
        qb["passer_rating"] = (
            8.4 * qb["pass_yards"]
            + 330.0 * qb["pass_tds"]
            + 100.0 * qb["completions"]
            - 200.0 * qb["interceptions"]
        ) / pa
    else:
        qb["passer_rating"] = float("nan")

    qb["td_rate"]  = qb["pass_tds"]      / pa if "pass_tds"      in qb.columns else float("nan")
    qb["int_rate"] = qb["interceptions"] / pa if "interceptions" in qb.columns else float("nan")

    # yards_per_attempt is already in the gamelog (= pass_yards / pass_att); rename to ypa
    # to match config.QB_ROLLING_FEATURES.
    if "yards_per_attempt" in qb.columns:
        qb["ypa"] = qb["yards_per_attempt"]
    elif "pass_yards" in qb.columns:
        qb["ypa"] = qb["pass_yards"] / pa
    else:
        qb["ypa"] = float("nan")

    rate_cols = ["passer_rating", "completion_pct", "ppa_avg_passing",
                 "ypa", "td_rate", "int_rate"]
    for c in rate_cols:
        if c not in qb.columns:
            qb[c] = float("nan")

    # Per-(game_id, school) blend: pass_att-weighted average across QBs.
    # Use a vectorized groupby; values column = pass_att (the weight) plus rates.
    import numpy as np
    def _blend(grp: pd.DataFrame) -> pd.Series:
        w = grp["pass_att"].astype(float).values
        out = {"pass_att": float(np.nansum(w))}
        for c in rate_cols:
            v = grp[c].astype(float).values
            ok = (~np.isnan(v)) & (w > 0) & (~np.isnan(w))
            if ok.any():
                out[c] = float(np.average(v[ok], weights=w[ok]))
            else:
                out[c] = float("nan")
        return pd.Series(out)

    panel = (qb.groupby(["game_id", "school", "game_date"], dropna=False,
                         group_keys=True)
               .apply(_blend, include_groups=False)
               .reset_index())
    panel["__sqrt_pa"] = np.sqrt(panel["pass_att"].clip(lower=0))

    rolled = compute_rolling_entity_strength(
        panel=panel,
        targets=targets,
        panel_entity_col="school",
        target_entity_col="school",
        date_col="game_date",
        stat_cols=rate_cols,
        halflife_days=halflife_days,
        weight_col="__sqrt_pa",
        out_prefix="qb_rolling",
        keep_target_cols=["game_id", "school"],
    )
    return rolled


# ── Team Rolling Context (pace, pass rate, OL run-block) ──────────────────────


def compute_team_rolling_context(
        gamelogs: pd.DataFrame,
        team_box_scores: pd.DataFrame,
        halflife_days: int = 45,
        min_plays: int = 30,
) -> pd.DataFrame:
    """
    For each (game_id, school), compute the team's decay-weighted EWMA of
    own-team pace, pass rate, and OL run-block four-factor over the team's
    strictly-prior games.

    Pipeline (mirrors compute_qb_rolling_stats):
      1. Dedupe gamelogs to one row per (game_id, school) with team_pass_att,
         team_rush_att, game_date. Compute pace = pass_att + rush_att and
         pass_rate = pass_att / pace. Drop rows with pace < min_plays
         (weather/forfeit games whose play counts aren't representative).
      2. Left-join team_box_scores on (game_id, team=school) for the OL
         four-factor (line_yards_avg, second_level_yards_avg,
         open_field_yards_avg, rushing_power_success).
      3. Per (game_id, school) target row: EWMA over the team's prior panel
         rows (strictly prior, weighted by sqrt(plays) so a 90-play game
         doesn't dominate a 65-play normal game).

    Args:
        gamelogs:        DataFrame with [game_id, school, game_date,
                         team_pass_att, team_rush_att]. Must be post-
                         _normalize_gamelogs so team_pass_att/team_rush_att
                         are populated.
        team_box_scores: DataFrame with [game_id, team, line_yards_avg,
                         second_level_yards_avg, open_field_yards_avg,
                         rushing_power_success]. May be empty — OL columns
                         emit NaN if so.
        halflife_days:   Exponential decay halflife in days.
        min_plays:       Drop panel rows with pace below this (drops forfeits
                         + heavy-weather games whose totals aren't team-style
                         signal).

    Returns:
        DataFrame with [game_id, school, team_rolling_pace,
                        team_rolling_pass_rate, team_rolling_line_yards_avg,
                        team_rolling_second_level_yards_avg,
                        team_rolling_open_field_yards_avg,
                        team_rolling_rushing_power_success].
    """
    from src.utils.rolling import compute_rolling_entity_strength

    if gamelogs.empty:
        return pd.DataFrame()

    required = {"game_id", "school", "game_date",
                "team_pass_att", "team_rush_att"}
    if not required.issubset(gamelogs.columns):
        missing = required - set(gamelogs.columns)
        logger.warning("compute_team_rolling_context: missing required cols %s",
                       missing)
        return pd.DataFrame()

    pace_stats   = ["pace", "pass_rate"]
    ol_stats     = ["line_yards_avg", "second_level_yards_avg",
                    "open_field_yards_avg", "rushing_power_success"]
    stat_cols    = pace_stats + ol_stats
    out_cols     = [f"team_rolling_{s}" for s in stat_cols]

    # Targets first — one row per (game_id, school), preserve order.
    targets = (gamelogs[["game_id", "school", "game_date"]]
               .drop_duplicates(["game_id", "school"]))

    def _empty_output() -> pd.DataFrame:
        out = targets[["game_id", "school"]].copy().reset_index(drop=True)
        for col in out_cols:
            out[col] = float("nan")
        return out

    # Build the per-(game_id, school) panel from gamelogs.
    panel = (gamelogs[["game_id", "school", "game_date",
                       "team_pass_att", "team_rush_att"]]
             .drop_duplicates(["game_id", "school"])
             .copy())
    for c in ["team_pass_att", "team_rush_att"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["pace"] = panel["team_pass_att"].fillna(0) + panel["team_rush_att"].fillna(0)
    # NaN pass_rate when no plays — it'll be filtered out by min_plays anyway
    panel["pass_rate"] = panel["team_pass_att"] / panel["pace"].where(panel["pace"] > 0)

    # Drop forfeits / heavy-weather games — their pace isn't representative.
    panel = panel[panel["pace"] >= float(min_plays)].copy()
    if panel.empty:
        return _empty_output()

    # Left-join OL four-factor from team_box_scores. Box-score schema: team col
    # name. Missing OL stats become NaN columns (handled by primitive).
    if team_box_scores is not None and not team_box_scores.empty:
        ol_cols_present = [c for c in ol_stats if c in team_box_scores.columns]
        if ol_cols_present:
            tb = team_box_scores[["game_id", "team", *ol_cols_present]].copy()
            tb = tb.rename(columns={"team": "school"})
            panel = panel.merge(tb, on=["game_id", "school"], how="left")
    for c in ol_stats:
        if c not in panel.columns:
            panel[c] = float("nan")
        else:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    import numpy as np
    panel["__sqrt_plays"] = np.sqrt(panel["pace"].clip(lower=0))

    rolled = compute_rolling_entity_strength(
        panel=panel,
        targets=targets,
        panel_entity_col="school",
        target_entity_col="school",
        date_col="game_date",
        stat_cols=stat_cols,
        halflife_days=halflife_days,
        weight_col="__sqrt_plays",
        out_prefix="team_rolling",
        keep_target_cols=["game_id", "school"],
    )
    return rolled


# ── Defensive Teammate Rolling (own-defense pressure/coverage rates) ─────────


def compute_defense_rolling_stats(
        gamelogs: pd.DataFrame,
        halflife_days: int = 45,
        min_opp_pass_att: int = 10,
        min_opp_plays: int = 30,
) -> pd.DataFrame:
    """
    For each (game_id, school), exp-decay-weighted EWMA over school's
    strictly-prior games of own-defense rate stats:
      - def_rolling_pressure_rate  = (sacks + qb_hurries) / opp_pass_att
      - def_rolling_coverage_rate  = (passes_defended + ints) / opp_pass_att

    Coverage caveat: CFBD defensive box-score stats begin ~2018; earlier
    games emit NaN here (model handles via fillna(0) on the contextual side).

    Self-reference caveats handled at config gating, not in the computation:
      - pressure_rate is mostly DL/LB production → gate OFF DL; LB gets it
        (partial self acceptable for swing position).
      - coverage_rate is mostly DB production → gate OFF DB; LB gets it.

    Pipeline:
      1. Per (game_id, school): aggregate defensive counters via sum across
         the school's player rows. Keep team_pass_att / team_rush_att as
         the school's own offensive volume (used downstream for self-join).
      2. Self-join (game_id, opponent=school) to attach opp_pass_att and
         opp_plays for the rate denominators.
      3. Drop panel rows below min_opp_pass_att / min_opp_plays (forfeit /
         weather games whose rates aren't representative).
      4. EWMA per school's strictly-prior history.
    """
    from src.utils.rolling import compute_rolling_entity_strength

    if gamelogs.empty:
        return pd.DataFrame()

    required = {"game_id", "school", "opponent", "game_date",
                "team_pass_att", "team_rush_att"}
    if not required.issubset(gamelogs.columns):
        logger.warning("compute_defense_rolling_stats: missing %s",
                       required - set(gamelogs.columns))
        return pd.DataFrame()

    out_cols = ["def_rolling_pressure_rate", "def_rolling_coverage_rate"]

    targets = (gamelogs[["game_id", "school", "game_date"]]
               .drop_duplicates(["game_id", "school"]))

    def _empty_output() -> pd.DataFrame:
        out = targets[["game_id", "school"]].copy().reset_index(drop=True)
        for col in out_cols:
            out[col] = float("nan")
        return out

    # ── Stage 1: per (game_id, school) team defensive counters ────────────────
    def_stat_cols = ["sacks", "qb_hurries", "passes_defended", "ints"]
    present_def = [c for c in def_stat_cols if c in gamelogs.columns]
    if not present_def:
        return _empty_output()

    work = gamelogs[["game_id", "school", "game_date", "opponent",
                     "team_pass_att", "team_rush_att", *present_def]].copy()
    for c in present_def + ["team_pass_att", "team_rush_att"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Use min_count=1 so all-NaN groups (e.g. pre-2018 games where CFBD
    # didn't track defensive box stats) emit NaN, not 0. Without this,
    # pandas sums of all-NaN return 0 → pressure_rate = 0 / opp_pass_att,
    # which would falsely encode "perfect-coverage defense" for the
    # untracked years. The rolling EWMA then skips NaN automatically.
    grouped = work.groupby(["game_id", "school"], dropna=False)
    def_sums = grouped[present_def].sum(min_count=1).reset_index()
    aux = grouped[["team_pass_att", "team_rush_att",
                   "game_date", "opponent"]].first().reset_index()
    team = def_sums.merge(aux, on=["game_id", "school"], how="left")

    # ── Stage 2: self-join for opp_pass_att / opp_plays ────────────────────────
    opp_lookup = team[["game_id", "school", "team_pass_att",
                       "team_rush_att"]].copy()
    opp_lookup = opp_lookup.rename(columns={
        "school":         "opponent",
        "team_pass_att":  "opp_pass_att",
        "team_rush_att":  "opp_rush_att",
    })
    panel = team.merge(opp_lookup, on=["game_id", "opponent"], how="left")
    panel["opp_plays"] = (panel["opp_pass_att"].fillna(0)
                          + panel["opp_rush_att"].fillna(0))

    # ── Stage 3: filter forfeit / weather games ───────────────────────────────
    panel = panel[(panel["opp_pass_att"].fillna(0) >= min_opp_pass_att)
                  & (panel["opp_plays"]    >= min_opp_plays)].copy()
    if panel.empty:
        return _empty_output()

    # ── Stage 4: rate stats; rolling EWMA ─────────────────────────────────────
    pa  = panel["opp_pass_att"].astype(float)
    sacks   = panel["sacks"]            if "sacks"           in panel else 0.0
    hurries = panel["qb_hurries"]       if "qb_hurries"      in panel else 0.0
    pds     = panel["passes_defended"]  if "passes_defended" in panel else 0.0
    ints_   = panel["ints"]             if "ints"            in panel else 0.0
    panel["pressure_rate"] = (sacks + hurries) / pa
    panel["coverage_rate"] = (pds + ints_)     / pa

    rolled = compute_rolling_entity_strength(
        panel=panel,
        targets=targets,
        panel_entity_col="school",
        target_entity_col="school",
        date_col="game_date",
        stat_cols=["pressure_rate", "coverage_rate"],
        halflife_days=halflife_days,
        weight_col=None,
        out_prefix="def_rolling",
        keep_target_cols=["game_id", "school"],
    )
    return rolled


# ── Opponent Offense Volume (rolling, from gamelogs) ──────────────────────────


def compute_rolling_opp_offense_volume(
        gamelogs: pd.DataFrame,
        halflife_days: int = 45,
        min_plays: int = 30,
) -> pd.DataFrame:
    """
    For each (game_id, school), exp-decay-weighted EWMA over the OPPONENT's
    strictly-prior games of the opponent's offensive volume (team_pass_att,
    team_rush_att, total plays).

    Use case: gives DEFENDERS opportunity-side context. A DB on a heavy-passing
    matchup sees more dropbacks → more pressure / coverage opportunities;
    that's signal that's invisible to opponent QUALITY metrics
    (opp_rolling_off_*) because quality is a rate, not a volume.

    Pipeline:
      1. Dedupe gamelogs to one row per (game_id, school) carrying
         team_pass_att, team_rush_att, opponent, game_date. Drop forfeit /
         heavy-weather games via min_plays (mirrors compute_team_rolling_context).
      2. For each target (game_id, focal_team, opponent=B, date=D), find
         panel rows where school==B and date<D, exp-decay-EWMA their
         pass_att / rush_att / plays. Output prefix "opp_rolling_off".

    Returns DataFrame with [game_id, school,
        opp_rolling_off_pass_att,
        opp_rolling_off_rush_att,
        opp_rolling_off_plays].
    Empty if required cols missing.
    """
    from src.utils.rolling import compute_rolling_entity_strength

    if gamelogs.empty:
        return pd.DataFrame()

    required = {"game_id", "school", "opponent", "game_date",
                "team_pass_att", "team_rush_att"}
    if not required.issubset(gamelogs.columns):
        logger.warning("compute_rolling_opp_offense_volume: missing %s",
                       required - set(gamelogs.columns))
        return pd.DataFrame()

    out_cols = ["opp_rolling_off_pass_att",
                "opp_rolling_off_rush_att",
                "opp_rolling_off_plays"]

    targets = (gamelogs[["game_id", "school", "opponent", "game_date"]]
               .drop_duplicates(["game_id", "school"]))

    def _empty_output() -> pd.DataFrame:
        out = targets[["game_id", "school"]].copy().reset_index(drop=True)
        for col in out_cols:
            out[col] = float("nan")
        return out

    panel = (gamelogs[["game_id", "school", "game_date",
                       "team_pass_att", "team_rush_att"]]
             .drop_duplicates(["game_id", "school"])
             .copy())
    for c in ["team_pass_att", "team_rush_att"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["plays"]    = panel["team_pass_att"].fillna(0) + panel["team_rush_att"].fillna(0)
    panel["pass_att"] = panel["team_pass_att"]
    panel["rush_att"] = panel["team_rush_att"]

    panel = panel[panel["plays"] >= float(min_plays)].copy()
    if panel.empty:
        return _empty_output()

    rolled = compute_rolling_entity_strength(
        panel=panel,
        targets=targets,
        panel_entity_col="school",
        target_entity_col="opponent",
        date_col="game_date",
        stat_cols=["pass_att", "rush_att", "plays"],
        halflife_days=halflife_days,
        weight_col=None,
        out_prefix="opp_rolling_off",
        keep_target_cols=["game_id", "school"],
    )
    return rolled


# ── Opponent Defense Allowed (rolling, from team_box_scores) ──────────────────


def compute_rolling_opp_defense_allowed(
        team_box_scores: pd.DataFrame,
        schedule: pd.DataFrame,
        stat_cols: list[str],
        halflife_days: int = 45,
) -> pd.DataFrame:
    """
    For each (game_id, school), exp-decay-weighted EWMA over the OPPONENT's
    strictly-prior games of stats their defense allowed.

    team_box_scores is per-(game_id, team) and only stores offensive-side
    stats. To derive "defense allowed", we read the offensive stats from
    games where the defending team was the OPPONENT (i.e. these stats were
    produced AGAINST that defense), and roll those over time.

    Pipeline:
      1. Join schedule onto team_box_scores so each row carries the
         offensive team's opponent (= the defense those stats were produced
         against).
      2. Targets are one row per (game_id, school) keyed by game_date and
         opponent (the defense being rolled against).
      3. Rolling EWMA per opponent: panel_entity_col="opponent" matches the
         target's opponent — for each target (game_id=X, school=F, opp=B,
         date=D), gathers panel rows where opponent==B and date<D and EWMAs
         their offensive stats.

    Returns DataFrame with [game_id, school, opp_rolling_def_<stat>...].
    Empty if either input is empty or schedule lacks home/away_team.
    """
    from src.utils.rolling import compute_rolling_entity_strength

    if team_box_scores.empty or schedule.empty:
        return pd.DataFrame()

    needed_sched = {"game_id", "game_date", "home_team", "away_team"}
    if not needed_sched.issubset(schedule.columns):
        logger.warning(
            "compute_rolling_opp_defense_allowed: schedule missing %s",
            needed_sched - set(schedule.columns),
        )
        return pd.DataFrame()

    import numpy as np

    sched = (schedule[list(needed_sched)]
             .drop_duplicates("game_id")
             .copy())
    sched["game_date"] = pd.to_datetime(sched["game_date"])

    panel = team_box_scores.merge(sched, on="game_id", how="left")
    panel["opponent"] = np.where(
        panel["team"] == panel["home_team"],
        panel["away_team"], panel["home_team"],
    )
    # Drop rows we couldn't pair with a schedule entry (no opponent).
    panel = panel[panel["opponent"].notna() & panel["game_date"].notna()].copy()
    if panel.empty:
        return pd.DataFrame()

    present_stats = [c for c in stat_cols if c in panel.columns]
    for c in present_stats:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

    # Build the panel of school's prior games (each gamelog row is a target).
    # Target keying: (game_id, school) with school's game_date and opponent.
    # We piggyback off panel itself: every (game_id, team) row is also a
    # "for school=team, opponent=opp" target. So just rename + dedupe.
    targets = (panel[["game_id", "team", "game_date", "opponent"]]
               .drop_duplicates(["game_id", "team"])
               .rename(columns={"team": "school"})
               .copy())

    rolled = compute_rolling_entity_strength(
        panel=panel,
        targets=targets,
        panel_entity_col="opponent",
        target_entity_col="opponent",
        date_col="game_date",
        stat_cols=present_stats,
        halflife_days=halflife_days,
        weight_col=None,
        out_prefix="opp_rolling_def",
        keep_target_cols=["game_id", "school"],
    )
    return rolled


# ── Team Rolling Scoring Environment (own offense, from team_box_scores) ──────


def compute_team_rolling_scoring_env(
        gamelogs: pd.DataFrame,
        team_box_scores: pd.DataFrame,
        stat_cols: list[str],
        halflife_days: int = 45,
        min_plays: int = 30,
) -> pd.DataFrame:
    """
    For each (game_id, school), exp-decay-weighted EWMA over the team's
    strictly-prior games of own-offense scoring-environment stats from
    team_box_scores (scoring_opportunities, points_per_opportunity,
    avg_starting_field_pos, etc.).

    Mirrors `compute_team_rolling_context` shape: dedupe gamelogs to one
    row per (game_id, school) with team_pass_att / team_rush_att / date,
    drop forfeit / heavy-weather games via min_plays, left-join the
    requested team_box_scores cols, EWMA over school's prior panel rows
    weighted by sqrt(plays).

    Returns DataFrame with [game_id, school, team_rolling_<stat>...].
    """
    from src.utils.rolling import compute_rolling_entity_strength

    if gamelogs.empty:
        return pd.DataFrame()
    required = {"game_id", "school", "game_date",
                "team_pass_att", "team_rush_att"}
    if not required.issubset(gamelogs.columns):
        logger.warning("compute_team_rolling_scoring_env: missing %s",
                       required - set(gamelogs.columns))
        return pd.DataFrame()

    targets = (gamelogs[["game_id", "school", "game_date"]]
               .drop_duplicates(["game_id", "school"]))

    out_cols = [f"team_rolling_{s}" for s in stat_cols]

    def _empty_output() -> pd.DataFrame:
        out = targets[["game_id", "school"]].copy().reset_index(drop=True)
        for col in out_cols:
            out[col] = float("nan")
        return out

    panel = (gamelogs[["game_id", "school", "game_date",
                       "team_pass_att", "team_rush_att"]]
             .drop_duplicates(["game_id", "school"])
             .copy())
    for c in ["team_pass_att", "team_rush_att"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["pace"] = panel["team_pass_att"].fillna(0) + panel["team_rush_att"].fillna(0)
    panel = panel[panel["pace"] >= float(min_plays)].copy()
    if panel.empty:
        return _empty_output()

    if team_box_scores is not None and not team_box_scores.empty:
        present = [c for c in stat_cols if c in team_box_scores.columns]
        if present:
            tb = team_box_scores[["game_id", "team", *present]].copy()
            tb = tb.rename(columns={"team": "school"})
            panel = panel.merge(tb, on=["game_id", "school"], how="left")
    for c in stat_cols:
        if c not in panel.columns:
            panel[c] = float("nan")
        else:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    import numpy as np
    panel["__sqrt_plays"] = np.sqrt(panel["pace"].clip(lower=0))

    rolled = compute_rolling_entity_strength(
        panel=panel,
        targets=targets,
        panel_entity_col="school",
        target_entity_col="school",
        date_col="game_date",
        stat_cols=stat_cols,
        halflife_days=halflife_days,
        weight_col="__sqrt_plays",
        out_prefix="team_rolling",
        keep_target_cols=["game_id", "school"],
    )
    return rolled


# ── Recruiting ────────────────────────────────────────────────────────────────

def fetch_recruiting_data(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch player recruiting ratings from CFBD (247Sports composite).
    Returns one row per recruit with composite rating and star count.
    """
    client = _get_client()
    recruiting_api = cfbd.RecruitingApi(client)

    def _get(r, *names, default=None):
        for n in names:
            v = getattr(r, n, None)
            if v is not None:
                return v
        return default

    records = []
    for year in range(start_year, end_year + 1):
        try:
            # Method renamed in cfbd 5.x: get_recruiting_players → get_recruits
            results = _api_call_with_retry(recruiting_api.get_recruits, year=year)
            for r in (results or []):
                records.append({
                    "recruit_year":          year,
                    "player_name":           _get(r, "name"),
                    "school":                _get(r, "committed_to", "committedTo"),
                    "position":              _get(r, "position"),
                    "state_province":        _get(r, "state_province", "stateProvince"),
                    "stars":                 _get(r, "stars"),
                    "recruiting_composite":  _get(r, "rating"),
                    "recruit_rank_national": _get(r, "ranking"),
                })
            _rate_limit(0.25)
            logger.debug("Fetched %d recruits for %d", len(results or []), year)
        except Exception as exc:
            logger.warning("Recruiting fetch failed for %d: %s", year, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["recruiting_composite"] = pd.to_numeric(df["recruiting_composite"], errors="coerce")
    df["recruit_rank_national"] = pd.to_numeric(df["recruit_rank_national"], errors="coerce")
    # Position-specific rank within year+position (proxy for "position ranking";
    # CFBD's recruit endpoint doesn't expose 247's per-position rank directly).
    if not df.empty and "position" in df.columns:
        df["recruit_rank_position"] = (
            df.groupby(["recruit_year", "position"])["recruiting_composite"]
              .rank(ascending=False, method="min")
        )
    return df


# ── Rosters ────────────────────────────────────────────────────────────────────

def fetch_rosters(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch FBS team rosters to get position, height, weight, class year.
    Loops over all FBS teams × years.
    """
    client = _get_client()
    teams_api = cfbd.TeamsApi(client)
    roster_api = cfbd.TeamsApi(client)

    # Get list of all FBS teams (only need to do this once)
    try:
        fbs_teams = teams_api.get_fbs_teams()
        team_names = [t.school for t in fbs_teams]
        logger.info("Fetching rosters for %d FBS teams × %d years",
                    len(team_names), end_year - start_year + 1)
    except Exception as exc:
        logger.error("Failed to get FBS team list: %s", exc)
        return pd.DataFrame()

    records = []
    total = len(team_names) * (end_year - start_year + 1)
    done = 0
    for year in range(start_year, end_year + 1):
        for team in team_names:
            try:
                roster = roster_api.get_roster(team=team, year=year)
                for p in roster:
                    records.append({
                        "player_id":   p.id,
                        "player_name": f"{p.first_name} {p.last_name}".strip(),
                        "school":      p.team or team,
                        "season_year": year,
                        "position":    p.position,
                        "height_in":   p.height,    # already in inches
                        "weight_lbs":  p.weight,
                        "class_year_int": p.year,   # 1=FR, 2=SO, 3=JR, 4=SR, 5=GR
                        "jersey":      p.jersey,
                        "home_city":   p.home_city,
                        "home_state":  p.home_state,
                    })
                _rate_limit(0.1)
            except Exception:
                pass
            done += 1
            if done % 500 == 0:
                logger.info("Roster fetch: %d / %d team-years", done, total)

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Venues (stadium metadata: surface, dome, elevation) ───────────────────────


def fetch_venues() -> pd.DataFrame:
    """Fetch all CFBD venues with stadium metadata.

    Returns one row per venue with:
        venue_id, name, city, state, zip, country_code, timezone,
        latitude, longitude, capacity, construction_year,
        grass (bool, nullable), dome (bool), elevation_m (float, nullable),
        elevation_ft (float, derived).

    CFBD returns a current snapshot — surface can change over a stadium's
    lifetime (LSU went grass→turf→grass, MAC stadiums went grass→turf, etc.).
    Historical by-year overrides live in config.STADIUM_SURFACE_OVERRIDES.
    """
    client = _get_client()
    api = cfbd.VenuesApi(client)

    venues = _api_call_with_retry(api.get_venues)
    logger.info("Fetched %d venues from CFBD", len(venues))

    M_TO_FT = 3.280839895

    records: list[dict] = []
    for v in venues:
        d = v.to_dict()
        # CFBD returns elevation as a string (meters); coerce → float.
        elev_m_raw = d.get("elevation")
        try:
            elev_m = float(elev_m_raw) if elev_m_raw is not None else None
        except (TypeError, ValueError):
            elev_m = None
        elev_ft = elev_m * M_TO_FT if elev_m is not None else None

        records.append({
            "venue_id":          d.get("id"),
            "name":              d.get("name"),
            "city":              d.get("city"),
            "state":             d.get("state"),
            "zip":               d.get("zip"),
            "country_code":      d.get("countryCode"),
            "timezone":          d.get("timezone"),
            "latitude":          d.get("latitude"),
            "longitude":         d.get("longitude"),
            "capacity":          d.get("capacity"),
            "construction_year": d.get("constructionYear"),
            "grass":             d.get("grass"),
            "dome":              d.get("dome"),
            "elevation_m":       elev_m,
            "elevation_ft":      elev_ft,
        })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Play-by-Play (per-play context + player-play stats) ───────────────────────


def fetch_plays(year: int, week: int) -> pd.DataFrame:
    """Fetch all plays for a (year, week).

    Returns one row per play with situational context. Use `fetch_play_stats`
    for the long-format player-play stat associations.
    """
    client = _get_client()
    api = cfbd.PlaysApi(client)
    try:
        plays = _api_call_with_retry(api.get_plays, year=year, week=week)
    except Exception as exc:
        logger.warning("get_plays failed for year=%d week=%d: %s", year, week, exc)
        return pd.DataFrame()
    finally:
        _rate_limit()

    records: list[dict] = []
    for p in plays:
        clock = getattr(p, "clock", None)
        if clock is not None:
            clock_seconds = (getattr(clock, "minutes", 0) or 0) * 60 + (getattr(clock, "seconds", 0) or 0)
        else:
            clock_seconds = None
        records.append({
            "play_id":            p.id,
            "drive_id":           getattr(p, "drive_id", None),
            "game_id":            p.game_id,
            "season_year":        year,
            "week":               week,
            "drive_number":       getattr(p, "drive_number", None),
            "play_number":        getattr(p, "play_number", None),
            "offense":            getattr(p, "offense", None),
            "defense":            getattr(p, "defense", None),
            "home":               getattr(p, "home", None),
            "away":               getattr(p, "away", None),
            "offense_score":      getattr(p, "offense_score", None),
            "defense_score":      getattr(p, "defense_score", None),
            "period":             getattr(p, "period", None),
            "clock_seconds":      clock_seconds,
            "offense_timeouts":   getattr(p, "offense_timeouts", None),
            "defense_timeouts":   getattr(p, "defense_timeouts", None),
            "yardline":           getattr(p, "yardline", None),
            "yards_to_goal":      getattr(p, "yards_to_goal", None),
            "down":               getattr(p, "down", None),
            "distance":           getattr(p, "distance", None),
            "yards_gained":       getattr(p, "yards_gained", None),
            "scoring":            int(bool(getattr(p, "scoring", False))),
            "play_type":          getattr(p, "play_type", None),
            "ppa":                getattr(p, "ppa", None),
        })

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in ["play_id", "drive_id", "game_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def fetch_play_stats(year: int, week: int, team: str | None = None) -> pd.DataFrame:
    """Fetch player-play stat associations for a (year, week, team).

    Returns long format: one row per (play_id, athlete_id, stat_type). The
    endpoint is 2000-row capped, so we paginate by team within the week.
    Pass `team=None` to query the whole week (only safe for early-season /
    sparse-week probes).
    """
    client = _get_client()
    api = cfbd.PlaysApi(client)
    kwargs: dict = {"year": year, "week": week}
    if team is not None:
        kwargs["team"] = team
    try:
        rows = _api_call_with_retry(api.get_play_stats, **kwargs)
    except Exception as exc:
        logger.warning("get_play_stats failed for year=%d week=%d team=%s: %s",
                       year, week, team, exc)
        return pd.DataFrame()
    finally:
        _rate_limit()

    if len(rows) >= 2000:
        logger.warning("get_play_stats hit 2000-row cap for year=%d week=%d team=%s — "
                       "results may be truncated", year, week, team)

    records: list[dict] = []
    for r in rows:
        records.append({
            "play_id":       getattr(r, "play_id", None),
            "drive_id":      getattr(r, "drive_id", None),
            "game_id":       getattr(r, "game_id", None),
            "season_year":   getattr(r, "season", None),
            "week":          getattr(r, "week", None),
            "team":          getattr(r, "team", None),
            "conference":    getattr(r, "conference", None),
            "opponent":      getattr(r, "opponent", None),
            "athlete_id":    getattr(r, "athlete_id", None),
            "athlete_name":  getattr(r, "athlete_name", None),
            "stat_type":     getattr(r, "stat_type", None),
            "stat":          getattr(r, "stat", None),
        })

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in ["play_id", "drive_id", "game_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
    return df
