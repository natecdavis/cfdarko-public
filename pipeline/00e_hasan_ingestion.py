"""
Step 00e: Arif Hasan Consensus Big Board Ingestion

Loads Arif Hasan's annual NFL Draft Consensus Big Board from two sources:
  - hasan.csv (root directory): 2014–2024, compiled by the user
  - Google Sheets (Wide Left Substack): 2025, 2026 (fetched at runtime)

Clips each year to rank <= 300. Joins to CFBD player_ids via bridge_players.parquet
using normalized name + draft_year matching, with school as a tiebreaker.

Output: data/processed/hasan_ranks.parquet
  Columns: player_id, draft_year, hasan_rank (1–300; NaN if not on board)
"""

import io
import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, RAW_DIR, ROOT_DIR

try:
    import requests
    _requests_available = True
except ImportError:
    _requests_available = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_RANK = 300

# Google Sheets CSV export URLs for 2025 and 2026
ONLINE_SHEETS = {
    2025: "https://docs.google.com/spreadsheets/d/1IUxTL9PXAmkasscUiGVYovtdKo7tawIuzXdMfDzmqI4/export?format=csv&gid=0",
    2026: "https://docs.google.com/spreadsheets/d/1kMMdFfdPhcIlmSRFWnKVzx3IGm5VODx5oqJRraGms5E/export?format=csv&gid=0",
}

# School name aliases: Hasan spelling → CFBD spelling
SCHOOL_ALIASES: dict[str, str] = {
    "southern california": "usc",
    "louisiana state": "lsu",
    "miami (fl)": "miami",
    "miami (ohio)": "miami (oh)",
    "central florida": "ucf",
    "texas christian": "tcu",
    "brigham young": "byu",
    "pittsburgh": "pitt",
    "mississippi": "ole miss",
    "nevada las vegas": "unlv",
    "middle tennessee state": "middle tennessee",
    "florida international": "fiu",
    "massachusetts": "umass",
    "hawaii": "hawai'i",
}


_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?$")

def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
    name = _SUFFIXES.sub("", name).strip()
    return name


def _normalize_school(school: str) -> str:
    if not isinstance(school, str):
        return ""
    s = _normalize_name(school)
    return SCHOOL_ALIASES.get(s, s)


def _fetch_online_sheet(year: int, url: str) -> pd.DataFrame:
    """Fetch a Wide Left Google Sheet and return a normalized DataFrame."""
    if not _requests_available:
        logger.warning("requests not installed — skipping online fetch for %d", year)
        return pd.DataFrame()

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text), header=None)
    except Exception as exc:
        logger.warning("Could not fetch %d board: %s", year, exc)
        return pd.DataFrame()

    # Find the header row: first row where first cell looks like a rank column
    header_row = None
    for i, row in raw.iterrows():
        first = str(row.iloc[0]).strip().lower()
        if first in ("ovr", "cons", "rank", "#"):
            header_row = i
            break
    if header_row is None:
        logger.warning("Could not identify header row in %d sheet", year)
        return pd.DataFrame()

    raw.columns = raw.iloc[header_row]
    raw = raw.iloc[header_row + 1:].reset_index(drop=True)

    # Normalize column names to find rank and player columns
    col_map: dict[str, str] = {}
    for col in raw.columns:
        c = str(col).strip().lower()
        if c in ("ovr", "cons", "rank", "#", "ovr rank"):
            col_map[col] = "rank"
        elif c in ("player", "name"):
            col_map[col] = "name"
        elif c in ("school", "college"):
            col_map[col] = "school"
        elif c in ("position", "pos"):
            col_map[col] = "pos"

    raw = raw.rename(columns=col_map)

    needed = {"rank", "name", "school"}
    if not needed.issubset(raw.columns):
        logger.warning("Missing required columns in %d sheet (have: %s)", year, list(raw.columns))
        return pd.DataFrame()

    df = raw[["rank", "name", "school"] + (["pos"] if "pos" in raw.columns else [])].copy()
    df["draft_year"] = year
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    return df.dropna(subset=["rank", "name"])


def _load_csv(path: Path) -> pd.DataFrame:
    """Load hasan.csv and return with standardized columns."""
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        c = col.strip().lower()
        if c in ("rank", "#"):
            rename[col] = "rank"
        elif c in ("name", "player"):
            rename[col] = "name"
        elif c in ("pos", "position"):
            rename[col] = "pos"
        elif c in ("school", "college"):
            rename[col] = "school"
        elif c in ("year", "draft_year"):
            rename[col] = "draft_year"
    df = df.rename(columns=rename)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    return df.dropna(subset=["rank", "name", "draft_year"])


def _season_counts() -> tuple[dict, dict]:
    """Stat seasons and roster seasons per CFBD id (tie-break input)."""
    def count(name):
        p = RAW_DIR / f"{name}.parquet"
        if not p.exists():
            return {}
        x = pd.read_parquet(p, columns=["player_id", "season_year"]).dropna()
        return x.groupby(x["player_id"].astype(str))["season_year"].nunique().to_dict()
    return count("player_seasons"), count("rosters")


def _pick_candidate(candidates: list[tuple], school: str, n_stat: dict, n_ros: dict):
    """One board row's player_id among (pid, school, tier) candidates sharing
    its normalized name and draft year. Tier 0 = bridge (NFL-linked) record,
    1 = last college season + 1, 2 = + 2 (sat a season out).

    Order: school match, then tier, then most stat seasons, then most roster
    seasons. Several candidates at one school are usually two CFBD records of
    the SAME player (a stat-less roster copy, or a stats-feed spelling); the
    old dict-by-school tie-break kept whichever came last, which put 33 board
    ranks on the copy (Robert Spears-Jennings, Vernon Hargreaves, Bucky
    Irving...), so the real record trained and scored as off-board."""
    same_school = [c for c in candidates if c[1] == school]
    pool = same_school or candidates
    best = min(pool, key=lambda c: (c[2], -n_stat.get(str(c[0]), 0),
                                    -n_ros.get(str(c[0]), 0), str(c[0])))
    return best[0]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load hasan.csv (2014–2024) ─────────────────────────────────────────────
    csv_path = ROOT_DIR / "hasan.csv"
    if not csv_path.exists():
        logger.error("hasan.csv not found at %s", csv_path)
        sys.exit(1)

    historical = _load_csv(csv_path)
    historical["draft_year"] = historical["draft_year"].astype(int)
    logger.info("Loaded hasan.csv: %d rows, years %d–%d",
                len(historical), int(historical["draft_year"].min()),
                int(historical["draft_year"].max()))

    # ── Fetch 2025 and 2026 online ─────────────────────────────────────────────
    online_frames = []
    csv_years = set(historical["draft_year"].unique())
    for year, url in ONLINE_SHEETS.items():
        if year in csv_years:
            logger.info("Year %d already in hasan.csv — skipping online fetch", year)
            continue
        logger.info("Fetching %d board from Wide Left...", year)
        frame = _fetch_online_sheet(year, url)
        if not frame.empty:
            frame["draft_year"] = year
            online_frames.append(frame)
            logger.info("  %d: %d rows fetched", year, len(frame))

    all_frames = [historical] + online_frames
    board = pd.concat(all_frames, ignore_index=True)

    # ── Clip to MAX_RANK ───────────────────────────────────────────────────────
    before = len(board)
    board = board[board["rank"] <= MAX_RANK].copy()
    logger.info("Clipped to rank <= %d: %d → %d rows", MAX_RANK, before, len(board))

    # ── Normalize name and school for matching ─────────────────────────────────
    board["_norm_name"]   = board["name"].apply(_normalize_name)
    board["_norm_school"] = board["school"].apply(_normalize_school)
    board["draft_year"]   = board["draft_year"].astype(int)

    # ── Load bridge players ────────────────────────────────────────────────────
    bp_path = PROCESSED_DIR / "bridge_players.parquet"
    if not bp_path.exists():
        logger.error("bridge_players.parquet not found — run Step 2 first")
        sys.exit(1)

    bridge = pd.read_parquet(bp_path, columns=["player_id", "player_name", "college_team", "draft_year"])
    bridge = bridge[bridge["player_id"].notna() & bridge["draft_year"].notna()].copy()
    bridge["draft_year"]   = bridge["draft_year"].astype(int)
    bridge["_norm_name"]   = bridge["player_name"].apply(_normalize_name)
    bridge["_norm_school"] = bridge["college_team"].apply(_normalize_school)

    # Also load player_attributes to catch current prospects not yet in bridge_players
    # (draft_year maps to last_college_season_year + 1)
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    attrs_lookup = pd.DataFrame()
    if attrs_path.exists():
        attrs = pd.read_parquet(attrs_path, columns=["player_id", "player_name", "school",
                                                       "last_college_season_year"])
        attrs = attrs[attrs["player_id"].notna() & attrs["last_college_season_year"].notna()].copy()
        attrs["_norm_name"]   = attrs["player_name"].apply(_normalize_name)
        attrs["_norm_school"] = attrs["school"].apply(_normalize_school)
        # Deduplicate: keep latest season per player
        attrs = attrs.sort_values("last_college_season_year").drop_duplicates("player_id", keep="last")
        # Create entries for both +1 and +2 draft years to handle players who
        # sat out a season due to injury (e.g., ranked in draft year N+2 but
        # last_college_season_year is N).
        attrs_plus1 = attrs.copy()
        attrs_plus1["draft_year"] = attrs["last_college_season_year"].astype(int) + 1
        attrs_plus2 = attrs.copy()
        attrs_plus2["draft_year"] = attrs["last_college_season_year"].astype(int) + 2
        attrs_lookup = pd.concat(
            [attrs_plus1[["player_id", "_norm_name", "_norm_school", "draft_year"]].assign(_tier=1),
             attrs_plus2[["player_id", "_norm_name", "_norm_school", "draft_year"]].assign(_tier=2)],
            ignore_index=True,
        )

    # Combine bridge + attrs into a single lookup (bridge takes precedence).
    # Keyed on the name too: the bridge's spelling ("Cam Sutton", from the
    # draft) and CFBD's ("Cameron Sutton", which the board uses) both have
    # to stay indexed for the same player.
    combined = pd.concat([bridge[["player_id", "_norm_name", "_norm_school", "draft_year"]].assign(_tier=0),
                          attrs_lookup], ignore_index=True)
    combined = (combined.sort_values("_tier")
                .drop_duplicates(subset=["player_id", "draft_year", "_norm_name"], keep="first"))
    n_stat, n_ros = _season_counts()

    # ── Primary match: name + draft_year (across bridge + attrs) ────────────
    name_year_map: dict[tuple, list] = {}
    for _, row in combined.iterrows():
        key = (row["_norm_name"], int(row["draft_year"]))
        name_year_map.setdefault(key, []).append(
            (row["player_id"], row["_norm_school"], int(row["_tier"])))

    board["player_id"] = None
    matched_primary = 0
    ambiguous = 0

    for idx, row in board.iterrows():
        key = (row["_norm_name"], int(row["draft_year"]))
        candidates = name_year_map.get(key, [])
        if not candidates:
            continue
        pick = _pick_candidate(candidates, row["_norm_school"], n_stat, n_ros)
        board.at[idx, "player_id"] = pick
        matched_primary += 1
        if len({c[0] for c in candidates}) > 1 and not any(
                c[1] == row["_norm_school"] for c in candidates):
            ambiguous += 1

    logger.info("Primary match (name + year): %d / %d (ambiguous school: %d)",
                matched_primary, len(board), ambiguous)

    # ── Fallback: name + school + year ────────────────────────────────────────
    school_name_year_map: dict[tuple, str] = {}
    for _, row in combined.iterrows():
        key = (row["_norm_name"], row["_norm_school"], int(row["draft_year"]))
        school_name_year_map[key] = row["player_id"]

    unmatched_mask = board["player_id"].isna()
    fallback_count = 0
    for idx, row in board[unmatched_mask].iterrows():
        key = (row["_norm_name"], row["_norm_school"], int(row["draft_year"]))
        pid = school_name_year_map.get(key)
        if pid:
            board.at[idx, "player_id"] = pid
            fallback_count += 1

    logger.info("School+name fallback: %d additional matches", fallback_count)

    total_matched = board["player_id"].notna().sum()
    logger.info("Total matched: %d / %d (%.1f%%)",
                total_matched, len(board), 100 * total_matched / len(board))

    # Log unmatched so we can inspect them
    unmatched = board[board["player_id"].isna()][["name", "school", "draft_year", "rank"]].head(20)
    if not unmatched.empty:
        logger.info("Sample unmatched players:\n%s", unmatched.to_string(index=False))

    # ── Build output ───────────────────────────────────────────────────────────
    out = (board[board["player_id"].notna()]
           [["player_id", "draft_year", "rank"]]
           .rename(columns={"rank": "hasan_rank"})
           .drop_duplicates(subset=["player_id"])
           .reset_index(drop=True))

    out_path = PROCESSED_DIR / "hasan_ranks.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved %d hasan_rank rows → %s", len(out), out_path)

    # Summary by year
    by_year = (board.groupby("draft_year")
               .agg(total=("rank", "count"),
                    matched=("player_id", lambda x: x.notna().sum()))
               .assign(pct=lambda d: (100 * d["matched"] / d["total"]).round(1)))
    logger.info("Match rate by year:\n%s", by_year.to_string())

    logger.info("Step 00e complete.")


if __name__ == "__main__":
    main()
