"""
Step 1e: Stadium Orientation Scraper

One-time scrape of stadium polygons from OpenStreetMap, used to decompose
wind into parallel-to-field and perpendicular-to-field components.

Strategy:
    1. For each venue in data/raw/venues.parquet (lat/lon known)
    2. Query OSM for a stadium polygon within ~300m
    3. Compute the field-axis bearing as the long axis of the
       minimum-rotated-rectangle of the polygon (compass bearing 0-180°,
       0=N, 90=E)
    4. Cache result to data/raw/venue_orientation.parquet

Manual overrides for unresolved venues live in config.VENUE_ORIENTATION_OVERRIDES.

Idempotent — re-run only fetches venues missing from the cache. The full
backfill is ~840 venues × ~1.5s = ~20 minutes.

Usage:
    python pipeline/01e_venue_orientation.py
    python pipeline/01e_venue_orientation.py --rescrape   # force refresh
"""

import argparse
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, VENUE_ORIENTATION_OVERRIDES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)
# Unbuffered stdout for live progress when piped to a log file
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

# Suppress noisy CRS warnings from osmnx area calls (we only use them for
# argmax, not absolute area).
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")

VENUES_PATH      = RAW_DIR / "venues.parquet"
ORIENTATION_PATH = RAW_DIR / "venue_orientation.parquet"

SEARCH_RADIUS_M  = 300
TAG_CANDIDATES = [
    {"leisure": "stadium"},
    {"building": "stadium"},
    {"sport": "american_football"},
    {"leisure": "pitch"},
]

POLITENESS_SLEEP = 1.0  # seconds between OSM queries (Overpass courtesy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescrape", action="store_true",
                        help="Force re-fetch of all venues (default: only "
                             "fetch venues missing from cache)")
    return parser.parse_args()


def _compass_bearing(p1: tuple, p2: tuple) -> float:
    """Bearing 0-180° between two lon/lat points (compass: 0=N, 90=E).

    Returns the angle of the line, modulo 180° (we don't care about
    direction along the line, just the axis orientation).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    cos_lat = math.cos(math.radians((p1[1] + p2[1]) / 2))
    return math.degrees(math.atan2(dx * cos_lat, dy)) % 180


def _polygon_orientation(geom) -> float:
    """Compass bearing (0-180°) of the long axis of the minimum-rotated rectangle."""
    if geom.geom_type == "MultiPolygon":
        # Pick the largest sub-polygon (typically the main bowl)
        geom = max(geom.geoms, key=lambda g: g.area)
    mrr = geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    edges = []
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        bearing = _compass_bearing(p1, p2)
        edges.append((length, bearing))
    edges.sort(reverse=True)  # longest first
    return edges[0][1]


def fetch_one_venue(venue_id: int, lat: float, lon: float) -> tuple[float, str]:
    """Try each tag combination until one returns a polygon."""
    import osmnx as ox
    for tags in TAG_CANDIDATES:
        try:
            features = ox.features_from_point((lat, lon), tags=tags, dist=SEARCH_RADIUS_M)
        except Exception as exc:
            logger.debug("osmnx failure for venue=%s tags=%s: %s",
                         venue_id, tags, str(exc)[:80])
            continue
        if len(features) == 0:
            continue
        # Pick the largest polygon
        polygons = features[features.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if len(polygons) == 0:
            continue
        idx = polygons.geometry.area.argmax()
        geom = polygons.geometry.iloc[idx]
        try:
            orientation = _polygon_orientation(geom)
        except Exception as exc:
            logger.debug("Orientation calc failed for venue=%s: %s", venue_id, exc)
            continue
        return orientation, f"osm:{list(tags.keys())[0]}={list(tags.values())[0]}"

    return float("nan"), "missing"


def main():
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not VENUES_PATH.exists():
        logger.error("venues.parquet not found at %s — run Step 1 first", VENUES_PATH)
        sys.exit(1)

    venues = pd.read_parquet(VENUES_PATH)
    venues = venues.dropna(subset=["venue_id", "latitude", "longitude"]).copy()
    venues["venue_id"] = venues["venue_id"].astype(int)
    logger.info("Loaded %d venues from %s", len(venues), VENUES_PATH)

    # Restrict to venues actually referenced in our schedule — saves time on
    # ~500 FCS / D2 stadiums we don't model. Falls back to all venues if
    # game_schedule.parquet is missing.
    schedule_path = RAW_DIR / "game_schedule.parquet"
    if schedule_path.exists():
        sched = pd.read_parquet(schedule_path, columns=["venue_id"])
        used_ids = set(sched["venue_id"].dropna().astype(int).unique())
        before = len(venues)
        venues = venues[venues["venue_id"].isin(used_ids)]
        logger.info("Filtered to %d venues referenced in game_schedule (was %d)",
                    len(venues), before)

    # Load existing orientations (if any)
    if ORIENTATION_PATH.exists() and not args.rescrape:
        existing = pd.read_parquet(ORIENTATION_PATH)
        existing_ids = set(existing["venue_id"].astype(int).tolist())
        logger.info("Resuming: %d venues already scraped", len(existing_ids))
        # Re-attempt only venues marked as "missing" — they may have been
        # added to OSM since the last run
        missing_ids = set(existing.loc[existing["orient_source"] == "missing",
                                       "venue_id"].astype(int).tolist())
        already_done = existing_ids - missing_ids
        venues_to_scrape = venues[~venues["venue_id"].isin(already_done)]
    else:
        existing = pd.DataFrame(columns=["venue_id", "orientation_deg", "orient_source"])
        venues_to_scrape = venues

    logger.info("Scraping %d venues from OSM (~%.0f min at %.1fs/venue)",
                len(venues_to_scrape), len(venues_to_scrape) * POLITENESS_SLEEP / 60,
                POLITENESS_SLEEP)

    records: list[dict] = []
    n_found = 0
    n_missing = 0
    for i, row in enumerate(venues_to_scrape.itertuples(index=False), 1):
        vid = int(row.venue_id)
        # Manual override wins
        if vid in VENUE_ORIENTATION_OVERRIDES:
            records.append({
                "venue_id": vid,
                "orientation_deg": float(VENUE_ORIENTATION_OVERRIDES[vid]),
                "orient_source": "manual_override",
            })
            continue

        orientation, source = fetch_one_venue(vid, row.latitude, row.longitude)
        records.append({
            "venue_id": vid,
            "orientation_deg": orientation,
            "orient_source": source,
        })
        if source == "missing":
            n_missing += 1
        else:
            n_found += 1

        if i % 25 == 0:
            logger.info("  progress: %d/%d (found=%d missing=%d)",
                        i, len(venues_to_scrape), n_found, n_missing)

        time.sleep(POLITENESS_SLEEP)

    new_df = pd.DataFrame(records)
    # Merge with existing, dropping rows for venues we re-scraped
    if not existing.empty:
        existing = existing[~existing["venue_id"].astype(int).isin(new_df["venue_id"])]
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df

    out["venue_id"] = out["venue_id"].astype("int32")
    out["orientation_deg"] = pd.to_numeric(out["orientation_deg"], errors="coerce").astype("float32")
    out.to_parquet(ORIENTATION_PATH, index=False)

    n_total = len(out)
    n_known = int(out["orientation_deg"].notna().sum())
    logger.info("Saved %d orientation rows → %s", n_total, ORIENTATION_PATH)
    logger.info("Coverage: %d/%d venues have orientation (%.1f%%)",
                n_known, n_total, 100.0 * n_known / max(n_total, 1))

    if n_total - n_known > 0:
        logger.warning("%d venues unresolved — add to "
                       "config.VENUE_ORIENTATION_OVERRIDES if needed:",
                       n_total - n_known)
        missing = out[out["orientation_deg"].isna()].merge(
            venues[["venue_id", "name", "city", "state"]], on="venue_id", how="left",
        )
        for r in missing.head(10).itertuples(index=False):
            logger.warning("    venue_id=%d  %s (%s, %s)",
                           r.venue_id, getattr(r, "name", ""),
                           getattr(r, "city", ""), getattr(r, "state", ""))


if __name__ == "__main__":
    main()
