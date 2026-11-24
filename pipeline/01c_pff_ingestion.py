"""
Step 01c: PFF Season-Grade Ingestion

Loads all PFF CSV exports from data/raw/pff/{year}/, normalizes player/school
names, matches players to CFBD player_ids from player_seasons.parquet, and saves
a unified season-grade file.

Output: data/processed/pff_season_grades.parquet
        data/processed/fcs_rosters.parquet

Matching strategy (applied in order, stops at first match):
  1. Exact: normalized_name + normalized_school + season_year
  2. Name+year fallback: normalized_name + season_year (school may differ)
  3. Fuzzy: rapidfuzz token_sort_ratio >= 85, same normalized_school, year ±1
  Unmatched rows kept with player_id = None.
  4. FCS identities: players whose rows all went unmatched and whose last
     season is in a PFF FCS file get a synthetic id (src/utils/fcs_identities)
     and a roster-like row per season in fcs_rosters.parquet.

Usage:
    python pipeline/01c_pff_ingestion.py
    python pipeline/01c_pff_ingestion.py --seasons 2025
    python pipeline/01c_pff_ingestion.py --min-match-rate 0.5   # warn below 50%
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PFF_DATA_DIR, PROCESSED_DIR, RAW_DIR
from src.scrapers.pff import (
    extract_season_grades,
    _normalize_name,
    normalize_pff_school,
)
from src.utils.fcs_identities import (FCS_ROSTERS_PATH, build_fcs_identities, is_fcs_identity,
                                      load_fcs_rosters)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _normalize_school_base(school: str) -> str:
    import re
    if not isinstance(school, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", school.lower())


def _load_cfbd_player_index(
    seasons_path: Path,
    attrs_path: Path | None = None,
    rosters_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build a normalized player lookup DataFrame from player_seasons,
    player_attributes (for positions like OL that have no per-game stats),
    and rosters (which sometimes carries a different display-name spelling
    than player_seasons — e.g. "KC Concepcion" in rosters vs. "Kevin
    Concepcion" in player_seasons; both forms must be indexed so PFF's
    nickname-style normalized names can match).
    """
    frames = []

    if seasons_path.exists():
        ps = pd.read_parquet(seasons_path, columns=["player_id", "player_name", "school", "season_year"])
        frames.append(ps)

    if attrs_path and attrs_path.exists():
        attrs = pd.read_parquet(attrs_path, columns=["player_id", "player_name", "school", "season_year"])
        # player_attributes carries what a previous run's FCS identities put
        # there: synthetic ids, and the FCS seasons of CFBD players they
        # linked to. Matching against either feeds this run's output back
        # into itself (a linked transfer's FCS season would match as "exact").
        prev = load_fcs_rosters(["player_id", "season_year"])
        prev_keys = set(zip(prev["player_id"].astype(str), prev["season_year"].astype(int)))
        from_fcs = [(str(p), int(y)) in prev_keys if pd.notna(y) else False
                    for p, y in zip(attrs["player_id"], attrs["season_year"])]
        attrs = attrs[~is_fcs_identity(attrs["player_id"]).values & ~pd.Series(from_fcs, index=attrs.index)]
        frames.append(attrs)

    if rosters_path and rosters_path.exists():
        ros = pd.read_parquet(rosters_path, columns=["player_id", "player_name", "school", "season_year"])
        frames.append(ros)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["_name_norm"] = combined["player_name"].apply(_normalize_name)
    combined["_school_norm"] = combined["school"].apply(_normalize_school_base)
    # Dedup on (player_id, season_year, name_norm) so that different display-name
    # spellings of the same player across sources (e.g. "Kevin" vs "KC") both
    # remain indexed, while exact within-source duplicates collapse.
    combined = combined.drop_duplicates(
        subset=["player_id", "season_year", "_name_norm"]
    )
    return combined


def match_players(grades: pd.DataFrame, player_seasons: pd.DataFrame) -> pd.DataFrame:
    """
    Match PFF rows to CFBD player_ids. Adds a 'player_id' column.
    Also adds 'match_method' for debugging ('exact', 'name_year', 'fuzzy', 'unmatched').
    """
    # Build lookup indices
    exact_idx: dict[tuple, str] = {}   # (name_norm, school_norm, year) → player_id
    name_year_idx: dict[tuple, list[str]] = {}  # (name_norm, year) → [player_id, ...]
    school_name_idx: dict[tuple, list[tuple]] = {}  # (school_norm, year) → [(name_norm, player_id), ...]

    for _, row in player_seasons.iterrows():
        key = (row["_name_norm"], row["_school_norm"], int(row["season_year"]))
        exact_idx[key] = str(row["player_id"])

        ny_key = (row["_name_norm"], int(row["season_year"]))
        name_year_idx.setdefault(ny_key, []).append(str(row["player_id"]))

        sy_key = (row["_school_norm"], int(row["season_year"]))
        school_name_idx.setdefault(sy_key, []).append((row["_name_norm"], str(row["player_id"])))

    # Attempt fuzzy match if rapidfuzz available
    try:
        from rapidfuzz import fuzz as _fuzz
        _has_fuzz = True
    except ImportError:
        _has_fuzz = False
        logger.warning("rapidfuzz not installed — fuzzy matching disabled. "
                       "Run: pip install rapidfuzz")

    player_ids = []
    methods = []

    for _, row in grades.iterrows():
        name_norm = row.get("player_name_norm", "")
        school_norm = row.get("school_norm", "")
        year = int(row.get("season_year", 0))

        # 1. Exact match
        exact_key = (name_norm, school_norm, year)
        if exact_key in exact_idx:
            player_ids.append(exact_idx[exact_key])
            methods.append("exact")
            continue

        # 2. Name + year (school may differ)
        ny_key = (name_norm, year)
        if ny_key in name_year_idx:
            player_ids.append(name_year_idx[ny_key][0])
            methods.append("name_year")
            continue

        # 3. Fuzzy: same school, year ±1, name ratio >= 85
        if _has_fuzz and name_norm:
            best_pid = None
            best_score = 84
            for dy in (-1, 0, 1):
                candidates = school_name_idx.get((school_norm, year + dy), [])
                for cand_name, cand_pid in candidates:
                    score = _fuzz.token_sort_ratio(name_norm, cand_name)
                    if score > best_score:
                        best_score = score
                        best_pid = cand_pid
            if best_pid is not None:
                player_ids.append(best_pid)
                methods.append("fuzzy")
                continue

        player_ids.append(None)
        methods.append("unmatched")

    grades = grades.copy()
    grades["player_id"] = player_ids
    grades["match_method"] = methods
    return grades


def main():
    parser = argparse.ArgumentParser(description="Ingest PFF season-grade CSVs")
    parser.add_argument("--seasons", nargs="+", type=int, default=None,
                        help="Specific seasons to ingest (default: all found in pff_dir)")
    parser.add_argument("--min-match-rate", type=float, default=0.60,
                        help="Warn if match rate below this threshold (default: 0.60)")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not PFF_DATA_DIR.exists():
        logger.error("PFF data directory not found: %s", PFF_DATA_DIR)
        sys.exit(1)

    # Discover seasons
    if args.seasons:
        seasons = sorted(args.seasons)
    else:
        seasons = sorted([
            int(d.name)
            for d in PFF_DATA_DIR.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])

    if not seasons:
        logger.error("No season directories found in %s", PFF_DATA_DIR)
        sys.exit(1)

    logger.info("Processing seasons: %s", seasons)

    # Load player index for matching (player_seasons + player_attributes + rosters)
    seasons_path = RAW_DIR / "player_seasons.parquet"
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    rosters_path = RAW_DIR / "rosters.parquet"

    if not seasons_path.exists():
        logger.error("player_seasons.parquet not found — run Step 1 first")
        sys.exit(1)

    player_seasons = _load_cfbd_player_index(
        seasons_path,
        attrs_path if attrs_path.exists() else None,
        rosters_path if rosters_path.exists() else None,
    )
    logger.info("Loaded %d player-season rows for matching (including attrs+rosters)", len(player_seasons))

    # Extract grades for each season
    all_grades: list[pd.DataFrame] = []
    for season in seasons:
        logger.info("Extracting season %d...", season)
        df = extract_season_grades(season, PFF_DATA_DIR)
        if df.empty:
            logger.warning("  No grade data for season %d", season)
            continue
        logger.info("  %d player-season rows extracted", len(df))
        all_grades.append(df)

    if not all_grades:
        logger.error("No PFF grade data found across all seasons")
        sys.exit(1)

    combined = pd.concat(all_grades, ignore_index=True)
    logger.info("Total PFF rows before matching: %d", len(combined))

    # Deduplicate within same season: keep row with most non-null grade columns
    pff_cols = [c for c in combined.columns if c.startswith("pff_") and c != "pff_player_id"]
    combined["_n_grades"] = combined[pff_cols].notna().sum(axis=1)
    combined = (combined.sort_values("_n_grades", ascending=False)
                .drop_duplicates(subset=["pff_player_id", "season_year"], keep="first")
                .drop(columns=["_n_grades"]))
    logger.info("After dedup: %d rows", len(combined))

    # Match to CFBD player_ids
    logger.info("Matching PFF players to CFBD player_ids...")
    combined = match_players(combined, player_seasons)

    # Match rate report
    total = len(combined)
    method_counts = combined["match_method"].value_counts()
    matched = total - method_counts.get("unmatched", 0)
    match_rate = matched / total if total > 0 else 0.0

    logger.info("Match results (%d total rows):", total)
    for method, count in method_counts.items():
        logger.info("  %s: %d (%.1f%%)", method, count, 100 * count / total)
    logger.info("Overall match rate: %.1f%%", 100 * match_rate)

    if match_rate < args.min_match_rate:
        logger.warning("Match rate %.1f%% below threshold %.1f%% — "
                       "consider updating PFF_SCHOOL_OVERRIDES in src/scrapers/pff.py",
                       100 * match_rate, 100 * args.min_match_rate)

    combined, fcs_rosters = build_fcs_identities(combined, PFF_DATA_DIR, RAW_DIR, logger)
    fcs_rosters.to_parquet(FCS_ROSTERS_PATH, index=False)
    logger.info("Saved %d FCS identity-seasons → %s", len(fcs_rosters), FCS_ROSTERS_PATH)

    # Log unmatched sample for debugging
    unmatched = combined[combined["match_method"] == "unmatched"]
    if not unmatched.empty:
        sample = unmatched[["player_name_norm", "school_norm", "season_year", "position_group"]].head(20)
        logger.info("Unmatched sample (first 20):\n%s", sample.to_string(index=False))

    # Save
    out_path = PROCESSED_DIR / "pff_season_grades.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info("Saved %d rows → %s", len(combined), out_path)

    # Summary by position group
    if "position_group" in combined.columns:
        for pg, grp in combined.groupby("position_group"):
            counts = grp["match_method"].value_counts()
            n_matched = grp["player_id"].notna().sum()
            logger.info("  %s: %d rows, %d matched (%.0f%%) [%s]",
                        pg, len(grp), n_matched, 100 * n_matched / len(grp),
                        ", ".join(f"{m}:{c}" for m, c in counts.items()))

    logger.info("Step 01c complete.")


if __name__ == "__main__":
    main()
