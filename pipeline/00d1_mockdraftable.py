"""
Step 00d1: Mockdraftable Combine Supplement

Fills the arm length / hand size / wingspan gap that neither nfl_data_py nor
pro-football-reference publish on their combine pages. Source is
mockdraftable.com — every player URL is server-rendered with a
`window.INITIAL_STATE = {...}` blob that carries all measurements.

Strategy: enumerate combine attendees per draft year from the same sources
00d already consumes (nflverse `import_combine_data` + PFR supplement parquets
written by 00d0), generate the player slug, fetch one HTML page per attendee,
parse the embedded JSON, and emit a normalized parquet per year. Step 00d
then left-merges these on (norm_name, norm_school, year) onto the row already
present from nflverse/PFR — additive only, never replaces.

Network behavior:
  - One GET per player; 1.0 s sleep between requests (mockdraftable is small,
    no Cloudflare; polite rate is more than enough).
  - HTML cached to data/raw/mockdraftable/{year}/{slug}.html so re-runs are
    free and we can re-parse without re-fetching.
  - On 404/500, log and try a small set of alternate slugs (e.g. drop "jr",
    drop apostrophes, swap suffixes).

Output:
  data/raw/mockdraftable/{year}/{slug}.html             — raw cache
  data/raw/mockdraftable/mockdraftable_{year}.parquet   — normalized
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

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

SUPPLEMENT_YEARS = list(range(2011, 2027))

MD_BASE = "https://www.mockdraftable.com/player/{slug}"
RATE_LIMIT_SECONDS = 1.0
REQUEST_TIMEOUT = 20

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# mockdraftable measurableKey ids we care about
MD_KEY_HEIGHT = 1
MD_KEY_WEIGHT = 2
MD_KEY_WINGSPAN = 3
MD_KEY_ARM = 4
MD_KEY_HAND = 5

# source=1 in the player record corresponds to combine; pro-day / re-tests are
# higher numbers. We only want combine numbers — pro-day measurements are
# notoriously inflated.
MD_SOURCE_COMBINE = 1

# Plausible bounds on combine measurements (inches). MD has occasional unit
# typos — Jaylon Carlies 2024 was recorded as arm=64.25" (real value ~32.6").
# Out-of-range values get nulled with a warning rather than corrupting the
# downstream cohort z-score (one outlier blew DB×2024 stdev to ~4.7").
# Lower arm bound is 26" so smallest combine attendees (5'5"-5'7" RB/WR like
# Wandale Robinson 27.625", Deuce Vaughn 27.75") aren't dropped.
ARM_BOUNDS = (26.0, 38.0)
HAND_BOUNDS = (7.0, 12.0)
WINGSPAN_BOUNDS = (65.0, 92.0)


def _bounded(value, bounds, label, slug):
    if value is None:
        return None
    lo, hi = bounds
    if lo <= value <= hi:
        return value
    logger.warning("  %s out-of-range for %s: %s (kept as NaN)", label, slug, value)
    return None

OUTPUT_SCHEMA = [
    "season", "player_name", "pos", "school",
    "arm", "hand", "wingspan", "wingspan_imputed",
    "md_slug",
]


def _normalize_name_for_match(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def slug_candidates(name: str) -> list[str]:
    """Generate plausible mockdraftable slugs for a player name.

    Mockdraftable convention: lowercase, ASCII-folded, hyphenated. Suffixes
    like "Jr." are usually dropped on their site (Marvin Harrison Jr. lives
    at /player/marvin-harrison). Try the bare slug first, then variants.
    """
    if not isinstance(name, str) or not name.strip():
        return []
    n = unicodedata.normalize("NFD", name)
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    n = n.lower()
    n = n.replace("'", "").replace(".", "").replace(",", "")
    # Strip trailing suffixes for the alternate
    base = re.sub(r"\s+", " ", n).strip()
    suffixes = (" jr", " sr", " ii", " iii", " iv", " v")
    stripped = base
    for suf in suffixes:
        if stripped.endswith(suf):
            stripped = stripped[: -len(suf)].strip()

    def to_slug(s: str) -> str:
        s = re.sub(r"[^a-z0-9 -]", "", s)
        return re.sub(r"\s+", "-", s.strip())

    out: list[str] = []
    for cand in (base, stripped):
        sl = to_slug(cand)
        if sl and sl not in out:
            out.append(sl)
    return out


def _extract_initial_state(html: str) -> dict | None:
    needle = "window.INITIAL_STATE = "
    i = html.find(needle)
    if i < 0:
        return None
    s = html[i + len(needle):]
    depth = 0
    end = -1
    for j, c in enumerate(s):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end < 0:
        return None
    try:
        return json.loads(s[:end])
    except json.JSONDecodeError:
        return None


def _fetch_player_html(slug: str, cache_path: Path, session: requests.Session) -> tuple[str | None, int]:
    """Returns (html_or_None, http_status). Uses cache when present."""
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8"), 200
    url = MD_BASE.format(slug=slug)
    try:
        r = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        logger.warning("  GET %s failed: %s", slug, e)
        return None, 0
    time.sleep(RATE_LIMIT_SECONDS)
    if r.status_code != 200:
        return None, r.status_code
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(r.text, encoding="utf-8")
    return r.text, 200


def _parse_player(state: dict, expect_year: int | None = None) -> dict | None:
    pid = state.get("selectedPlayerId")
    players = state.get("players", {}) or {}
    if not pid or pid not in players:
        return None
    p = players[pid]
    measurements = p.get("measurements", []) or []
    by_key = {}
    for m in measurements:
        # Prefer combine source; only fall back to non-combine if we have
        # nothing else for that key. This avoids contaminating the dataset
        # with pro-day / re-test numbers.
        k = m.get("measurableKey")
        src = m.get("source")
        val = m.get("measurement")
        if k is None or val is None:
            continue
        existing = by_key.get(k)
        if existing is None:
            by_key[k] = (val, src)
        else:
            ex_src = existing[1]
            if src == MD_SOURCE_COMBINE and ex_src != MD_SOURCE_COMBINE:
                by_key[k] = (val, src)
    slug_for_log = pid
    arm = _bounded(by_key.get(MD_KEY_ARM, (None, None))[0], ARM_BOUNDS, "arm", slug_for_log)
    hand = _bounded(by_key.get(MD_KEY_HAND, (None, None))[0], HAND_BOUNDS, "hand", slug_for_log)
    wing = _bounded(by_key.get(MD_KEY_WINGSPAN, (None, None))[0], WINGSPAN_BOUNDS, "wingspan", slug_for_log)
    ht = by_key.get(MD_KEY_HEIGHT, (None, None))[0]
    wing_imputed = False
    if wing is None and arm is not None and ht is not None:
        # Common scouting heuristic: wingspan ≈ 2 * arm + shoulder width.
        # MD's height is body height (inches), not shoulder width — using it
        # would overestimate. Don't impute when wingspan is missing; mark NaN.
        # (Earlier plan said impute = arm*2 + ht_in; verifying against mock-
        # draftable's own published triples shows wingspan ≈ 2*arm + (16-22),
        # not 2*arm + ht. Imputation would inject ~50" of error. Skip.)
        wing_imputed = False
    primary_pos = (p.get("positions") or {}).get("primary")
    return {
        "player_name": p.get("name"),
        "pos": primary_pos,
        "school": p.get("school"),
        "season": p.get("draft"),
        "arm": float(arm) if arm is not None else np.nan,
        "hand": float(hand) if hand is not None else np.nan,
        "wingspan": float(wing) if wing is not None else np.nan,
        "wingspan_imputed": wing_imputed,
        "md_slug": pid,
    }


def _load_combine_attendees(years: list[int]) -> pd.DataFrame:
    """Build the union of nflverse + PFR-supplement combine rows for the
    requested years. Only player_name + school + season + draft_year are needed.
    """
    parts: list[pd.DataFrame] = []
    if _nfl_available:
        try:
            df = nfl.import_combine_data(years)
            parts.append(df[["season", "player_name", "school", "pos", "draft_year"]].copy())
            logger.info("nflverse combine: %d rows for years %s", len(df), years)
        except Exception as e:
            logger.warning("nflverse combine fetch failed: %s", e)
    pfr_dir = RAW_DIR / "pfr_combine"
    if pfr_dir.exists():
        for f in sorted(pfr_dir.glob("pfr_combine_*.parquet")):
            df = pd.read_parquet(f)
            df = df[df["season"].isin(years)]
            if len(df) == 0:
                continue
            keep = df[["season", "player_name", "school", "pos", "draft_year"]].copy()
            parts.append(keep)
            logger.info("PFR supplement %s: %d rows", f.name, len(keep))
    if not parts:
        return pd.DataFrame(columns=["season", "player_name", "school", "pos", "draft_year"])
    out = pd.concat(parts, ignore_index=True)
    out["_norm_name"] = out["player_name"].map(_normalize_name_for_match)
    out = out.drop_duplicates(subset=["_norm_name", "season"], keep="first")
    return out.drop(columns=["_norm_name"]).reset_index(drop=True)


def fetch_year(year: int, attendees: pd.DataFrame, session: requests.Session) -> pd.DataFrame:
    cache_root = RAW_DIR / "mockdraftable" / str(year)
    cache_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_total = len(attendees)
    n_hit = n_404 = n_other = 0
    for i, (_, row) in enumerate(attendees.iterrows(), 1):
        name = row["player_name"]
        candidates = slug_candidates(name)
        if not candidates:
            continue
        result = None
        last_status = None
        chosen_slug = None
        for slug in candidates:
            cache_path = cache_root / f"{slug}.html"
            html, status = _fetch_player_html(slug, cache_path, session)
            last_status = status
            if html is None:
                continue
            state = _extract_initial_state(html)
            if state is None:
                continue
            parsed = _parse_player(state, expect_year=year)
            if parsed is None:
                continue
            # Verify the matched player actually corresponds to this attendee.
            # MD will happily return a same-named player from a different draft
            # class; skip if the season disagrees by more than 1 year (lets
            # combine-year vs draft-year wiggle pass).
            md_season = parsed.get("season")
            if md_season is not None and abs(int(md_season) - int(row["season"])) > 1:
                continue
            result = parsed
            chosen_slug = slug
            break
        if result is None:
            if last_status == 404:
                n_404 += 1
            else:
                n_other += 1
            continue
        # Override season with the attendee-side year so downstream merge keys
        # are consistent with nflverse/PFR.
        result["season"] = int(row["season"])
        # Carry the attendee-side school in case MD's school name disagrees
        # (we'll match on attendee's school in 00d).
        result["school"] = row.get("school") or result.get("school")
        rows.append(result)
        n_hit += 1
        if i % 50 == 0:
            logger.info("  year %d: %d/%d processed (hit=%d 404=%d other=%d)",
                        year, i, n_total, n_hit, n_404, n_other)
    logger.info("year %d: %d/%d players matched (404=%d, other-fail=%d)",
                year, n_hit, n_total, n_404, n_other)
    if not rows:
        return pd.DataFrame(columns=OUTPUT_SCHEMA)
    df = pd.DataFrame(rows)
    return df[[c for c in OUTPUT_SCHEMA if c in df.columns]]


def main():
    out_dir = RAW_DIR / "mockdraftable"
    out_dir.mkdir(parents=True, exist_ok=True)
    attendees = _load_combine_attendees(SUPPLEMENT_YEARS)
    if attendees.empty:
        logger.error("No combine attendees found for %s — run 00d first or check nflverse.", SUPPLEMENT_YEARS)
        sys.exit(1)
    logger.info("Total attendees across %d years: %d", len(SUPPLEMENT_YEARS), len(attendees))
    session = requests.Session()
    for year in SUPPLEMENT_YEARS:
        year_attendees = attendees[attendees["season"] == year].reset_index(drop=True)
        logger.info("=== year %d: %d attendees ===", year, len(year_attendees))
        df = fetch_year(year, year_attendees, session)
        out_path = out_dir / f"mockdraftable_{year}.parquet"
        df.to_parquet(out_path, index=False)
        n_arm = df["arm"].notna().sum() if len(df) else 0
        n_hand = df["hand"].notna().sum() if len(df) else 0
        n_wing = df["wingspan"].notna().sum() if len(df) else 0
        logger.info("Saved %d rows → %s (arm=%d hand=%d wing=%d)",
                    len(df), out_path, n_arm, n_hand, n_wing)


if __name__ == "__main__":
    main()
