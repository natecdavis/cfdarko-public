"""
Step 00f: Wikidata Birth Date Ingestion for College Football Players

Queries the Wikidata SPARQL endpoint for birth dates of American football players
born 1995 or later, then matches them to CFBD player_ids via normalized name.

Coverage is best for top prospects and widely-covered players; depth players
without Wikipedia entries will not be matched.

Usage:
    python pipeline/00f_wikidata_birth_dates.py

Output:
    data/processed/wikidata_birth_dates.parquet
      Columns: player_id, birth_year, birth_date (YYYY-MM-DD string)
"""

import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

SPARQL_QUERY = """
SELECT DISTINCT ?playerLabel ?birthDate WHERE {
  ?player wdt:P31 wd:Q5 ;
          wdt:P641 wd:Q41323 ;
          wdt:P569 ?birthDate .
  FILTER(?birthDate >= "1995-01-01"^^xsd:dateTime)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?")


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
    name = _SUFFIXES.sub("", name).strip()
    return re.sub(r"\s+", " ", name)


def fetch_wikidata_birth_dates() -> pd.DataFrame:
    """Query Wikidata SPARQL and return DataFrame with (name_norm, birth_year, birth_date)."""
    try:
        import requests
    except ImportError:
        logger.error("requests not installed — pip install requests")
        return pd.DataFrame()

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "cfDARKO/1.0 (college football analytics; educational use)",
    }
    try:
        resp = requests.get(
            SPARQL_ENDPOINT,
            params={"query": SPARQL_QUERY, "format": "json"},
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Wikidata SPARQL request failed: %s", exc)
        return pd.DataFrame()

    results = resp.json().get("results", {}).get("bindings", [])
    logger.info("Wikidata returned %d rows", len(results))

    rows = []
    for r in results:
        name = r.get("playerLabel", {}).get("value", "")
        birth_raw = r.get("birthDate", {}).get("value", "")
        if not name or not birth_raw:
            continue
        try:
            birth_date = birth_raw[:10]  # YYYY-MM-DD
            birth_year = int(birth_date[:4])
        except (ValueError, IndexError):
            continue
        rows.append({"name_norm": _normalize_name(name),
                     "birth_year": birth_year,
                     "birth_date": birth_date})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Keep one row per normalized name — prefer earliest birth date on duplicates
    df = df.sort_values("birth_date").drop_duplicates("name_norm", keep="first")
    logger.info("Unique normalized names with birth dates: %d", len(df))
    return df


def match_to_player_ids(wikidata_df: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """
    Match Wikidata rows to CFBD player_ids via normalized name.
    Only matches players where birth_year_estimated=True (no real PFR data).
    Returns DataFrame with (player_id, birth_year, birth_date).
    """
    if wikidata_df.empty:
        return pd.DataFrame(columns=["player_id", "birth_year", "birth_date"])

    # Only try to improve estimated birth years
    candidates = attrs[
        attrs.get("birth_year_estimated", pd.Series(True, index=attrs.index)) != False
    ].copy()
    candidates["name_norm"] = candidates["player_name"].apply(_normalize_name)

    # Build a lookup: name_norm → [player_ids] (some names may be ambiguous)
    name_to_pids: dict[str, list] = {}
    for _, row in candidates.iterrows():
        name_to_pids.setdefault(row["name_norm"], []).append(row["player_id"])

    matched = []
    skipped_ambiguous = 0

    for _, wd_row in wikidata_df.iterrows():
        pids = name_to_pids.get(wd_row["name_norm"], [])
        if not pids:
            continue
        if len(pids) > 1:
            # Disambiguate using estimated birth year tolerance ±2
            est_years = candidates[candidates["player_id"].isin(pids)][
                ["player_id", "birth_year", "last_college_season_year"]
            ]
            close = est_years[
                (est_years["birth_year"] - wd_row["birth_year"]).abs() <= 2
            ]
            if len(close) == 1:
                pids = [close.iloc[0]["player_id"]]
            else:
                skipped_ambiguous += 1
                continue
        matched.append({
            "player_id": pids[0],
            "birth_year": wd_row["birth_year"],
            "birth_date": wd_row["birth_date"],
        })

    logger.info("Matched %d players to Wikidata birth dates (%d skipped: ambiguous name)",
                len(matched), skipped_ambiguous)
    return pd.DataFrame(matched) if matched else pd.DataFrame(
        columns=["player_id", "birth_year", "birth_date"]
    )


def main():
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    if not attrs_path.exists():
        logger.error("player_attributes.parquet not found — run Step 03 first")
        sys.exit(1)

    attrs = pd.read_parquet(
        attrs_path,
        columns=["player_id", "player_name", "birth_year", "birth_year_estimated",
                 "last_college_season_year"],
    )
    logger.info("Loaded %d players from player_attributes", len(attrs))

    n_estimated = int(attrs.get("birth_year_estimated", pd.Series()).sum()
                      if "birth_year_estimated" in attrs.columns else 0)
    logger.info("Players with estimated birth_year: %d", n_estimated)

    logger.info("Querying Wikidata SPARQL...")
    wikidata_df = fetch_wikidata_birth_dates()
    if wikidata_df.empty:
        logger.warning("No Wikidata results — exiting")
        sys.exit(0)

    result = match_to_player_ids(wikidata_df, attrs)

    out_path = PROCESSED_DIR / "wikidata_birth_dates.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Saved %d matched birth dates → %s", len(result), out_path)

    # Show a sample
    if not result.empty:
        sample_ids = result["player_id"].head(10).tolist()
        sample = attrs[attrs["player_id"].isin(sample_ids)][["player_id", "player_name", "birth_year"]].copy()
        sample = sample.merge(result[["player_id", "birth_year"]].rename(columns={"birth_year": "birth_year_wiki"}),
                              on="player_id", how="left")
        logger.info("Sample matches:\n%s", sample[["player_name","birth_year","birth_year_wiki"]].to_string(index=False))

    logger.info("Step 00f complete.")


if __name__ == "__main__":
    main()
