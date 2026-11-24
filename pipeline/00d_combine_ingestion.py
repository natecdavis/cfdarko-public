"""
Step 00d: NFL Combine Data Ingestion

Fetches NFL combine metrics via nfl_data_py, joins to CFBD player_ids via
bridge_players.parquet (pfr_id, then name + year) and then the full college
universe (src/utils/college_id_match.py) for attendees who never reached the
NFL, and z-scores each metric within
(position_group × draft_year) cohort to remove year-level athletic drift.

Design choices:
  - NaN = not tested — LightGBM handles NaN natively, no imputation done here.
  - combine_invited = 1 for any player who attended the combine, 0 for others
    (added when merging into the snapshot in steps 8/10, not here).
  - forty_z is negated: lower time → higher z-score (better athlete).
  - Minimum cohort size = 5 testers; falls back to position-only mean/std.

Output: data/processed/combine_metrics.parquet
"""

import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, RAW_DIR

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

START_SEASON = 2000
END_SEASON   = 2026

COMBINE_METRICS = ["forty", "vertical", "broad_jump", "bench", "cone", "shuttle",
                   "ht", "wt", "arm", "hand", "wingspan"]

# forty is negated: lower time = better athleticism; ht/wt/arm/hand/wingspan:
# higher = more, no negation needed.
NEGATE_METRICS = {"forty"}

COMBINE_POS_MAP: dict[str, str] = {
    "QB":   "QB",
    "RB":   "RB", "FB":  "RB",
    "WR":   "WR",
    "TE":   "TE",
    "OT":   "OL", "OG":  "OL", "G":  "OL", "C":   "OL", "OL":  "OL",
    "DE":   "DL", "DT":  "DL", "NT":  "DL", "DL":  "DL", "EDGE": "DL",
    "ILB":  "LB", "OLB": "LB", "MLB": "LB", "LB":  "LB",
    "CB":   "DB", "SS":  "DB", "FS":  "DB", "S":   "DB", "SAF": "DB", "DB":   "DB",
    "K":    "K",
    "P":    "P",
}


def _ht_to_inches(val) -> float:
    """Convert '6-4' or '76' height string to total inches."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if "-" in s:
        parts = s.split("-")
        try:
            return int(parts[0]) * 12 + int(parts[1])
        except ValueError:
            return np.nan
    try:
        v = float(s)
        return v if v < 100 else np.nan
    except ValueError:
        return np.nan


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _zscore_within_cohort(
    df: pd.DataFrame,
    metric: str,
    cohort_cols: list[str],
    min_cohort: int = 5,
) -> pd.Series:
    """
    Z-score a metric within cohort groups. Falls back to position-only
    mean/std for cohorts smaller than min_cohort.
    """
    raw = df[metric].copy().astype(float)
    # `metric` is the prefixed column ("combine_forty"); NEGATE_METRICS holds
    # bare names. Before 2026-10-30 this compared the prefixed name, so forty
    # was never negated and combine_forty_z meant higher = SLOWER.
    if metric.removeprefix("combine_") in NEGATE_METRICS:
        raw = -raw

    z = pd.Series(np.nan, index=df.index, dtype=float)

    # Full cohort z-score
    for key, grp in df.groupby(cohort_cols):
        vals = raw.loc[grp.index]
        n_valid = vals.notna().sum()
        if n_valid >= min_cohort:
            mu, sigma = vals.mean(), vals.std()
            if sigma > 0:
                z.loc[grp.index] = (vals - mu) / sigma

    # Fallback: position-only mean/std for cohorts that were too small
    needs_fallback = z.isna() & raw.notna()
    if needs_fallback.any():
        pos_col = cohort_cols[0]  # position_group is always first
        for pos, grp in df.groupby(pos_col):
            idx = needs_fallback & (df[pos_col] == pos)
            if not idx.any():
                continue
            all_vals = raw.loc[df[pos_col] == pos]
            mu, sigma = all_vals.mean(), all_vals.std()
            if pd.notna(mu) and sigma > 0:
                z.loc[idx] = (raw.loc[idx] - mu) / sigma

    return z


def main():
    if not _nfl_available:
        logger.error("nfl_data_py not installed. Run: pip install nfl-data-py")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Fetch combine data ──────────────────────────────────────────────────────
    logger.info("Fetching combine data %d–%d...", START_SEASON, END_SEASON)
    seasons = list(range(START_SEASON, END_SEASON + 1))
    raw = nfl.import_combine_data(seasons)
    logger.info("Raw combine rows: %d", len(raw))

    # Splice in PFR supplements written by 00d0 (handles years nflverse hasn't
    # published yet). PFR wins for any year present in a supplement.
    supplement_dir = RAW_DIR / "pfr_combine"
    if supplement_dir.exists():
        for f in sorted(supplement_dir.glob("pfr_combine_*.parquet")):
            supp = pd.read_parquet(f)
            years_in_supp = set(supp["season"].unique())
            n_dropped = raw["season"].isin(years_in_supp).sum()
            raw = raw[~raw["season"].isin(years_in_supp)]
            raw = pd.concat([raw, supp], ignore_index=True)
            logger.info("Merged PFR supplement %s: dropped %d nflverse rows in %s, added %d PFR rows",
                        f.name, n_dropped, sorted(years_in_supp), len(supp))

    # Splice in mockdraftable supplements written by 00d1. ADDITIVE only —
    # adds arm / hand / wingspan to rows already present from nflverse/PFR;
    # never drops or replaces rows. Match key is (norm_name, season).
    md_dir = RAW_DIR / "mockdraftable"
    md_parts: list[pd.DataFrame] = []
    if md_dir.exists():
        for f in sorted(md_dir.glob("mockdraftable_*.parquet")):
            md_parts.append(pd.read_parquet(f))
    if md_parts:
        md_supp = pd.concat(md_parts, ignore_index=True)
        md_supp["_norm_name"] = md_supp["player_name"].apply(_normalize_name)
        md_supp["_match_year"] = md_supp["season"].astype(int)
        md_keep = (md_supp[["_norm_name", "_match_year", "arm", "hand", "wingspan"]]
                   .drop_duplicates(subset=["_norm_name", "_match_year"], keep="first"))
        raw["_norm_name"] = raw["player_name"].apply(_normalize_name)
        raw["_match_year"] = raw["season"].fillna(0).astype(int)
        raw = raw.merge(md_keep, on=["_norm_name", "_match_year"], how="left")
        n_arm = raw["arm"].notna().sum()
        logger.info("Merged mockdraftable supplement: %d/%d rows with arm length",
                    n_arm, len(raw))
        raw = raw.drop(columns=["_norm_name", "_match_year"])

    # Rename raw metric columns to prefixed versions
    rename_map = {m: f"combine_{m}" for m in COMBINE_METRICS}
    raw = raw.rename(columns=rename_map)

    # Convert ht from "6-4" format to total inches
    if "combine_ht" in raw.columns:
        raw["combine_ht"] = raw["combine_ht"].apply(_ht_to_inches)

    # Map positions to our POSITION_GROUPS. Some prospects (e.g. Travis Hunter,
    # CB/WR) appear with slash-separated dual positions; resolve to the primary
    # token. Also strip whitespace.
    def _map_pos(p):
        if not isinstance(p, str):
            return None
        for token in p.replace("\\", "/").split("/"):
            mapped = COMBINE_POS_MAP.get(token.strip().upper())
            if mapped is not None:
                return mapped
        return None
    raw["position_group"] = raw["pos"].apply(_map_pos)
    unmapped = raw["position_group"].isna().sum()
    if unmapped:
        logger.info("Dropped %d rows with unmapped positions: %s",
                    unmapped, raw.loc[raw["position_group"].isna(), "pos"].unique().tolist())
    raw = raw[raw["position_group"].notna()].copy()

    # Cohort year: prefer draft_year, fall back to season
    raw["cohort_year"] = raw["draft_year"].fillna(raw["season"]).astype(int)

    # ── Join to CFBD player_ids via bridge_players.pfr_id ──────────────────────
    bp_path = PROCESSED_DIR / "bridge_players.parquet"
    if not bp_path.exists():
        logger.error("bridge_players.parquet not found — run Step 2 first")
        sys.exit(1)

    bridge = pd.read_parquet(bp_path, columns=["player_id", "pfr_id", "player_name",
                                                "college_team", "draft_year"])
    bridge = bridge[bridge["player_id"].notna() & bridge["pfr_id"].notna()].copy()
    pfr_to_id = bridge.set_index("pfr_id")["player_id"].to_dict()

    raw["player_id"] = raw["pfr_id"].map(pfr_to_id)
    matched_pfr = raw["player_id"].notna().sum()
    logger.info("Matched via pfr_id: %d / %d (%.1f%%)",
                matched_pfr, len(raw), 100 * matched_pfr / len(raw))

    # Fallback: normalized name + draft_year against bridge_players
    unmatched = raw["player_id"].isna() & raw["player_name"].notna()
    if unmatched.any():
        bridge["_norm_name"] = bridge["player_name"].apply(_normalize_name)
        bridge["_draft_year"] = bridge["draft_year"].fillna(0).astype(int)
        name_year_map = bridge.set_index(["_norm_name", "_draft_year"])["player_id"].to_dict()

        raw.loc[unmatched, "_norm_name"] = raw.loc[unmatched, "player_name"].apply(_normalize_name)
        raw.loc[unmatched, "_draft_year"] = raw.loc[unmatched, "cohort_year"]

        fallback_ids = raw.loc[unmatched].apply(
            lambda r: name_year_map.get((r["_norm_name"], int(r["_draft_year"]))), axis=1
        )
        raw.loc[unmatched, "player_id"] = fallback_ids.values
        matched_fallback = raw["player_id"].notna().sum() - matched_pfr
        logger.info("Matched via name+year fallback: %d additional", matched_fallback)
        raw = raw.drop(columns=["_norm_name", "_draft_year"], errors="ignore")

    # Pass 3: everyone else against the FULL college universe (name + school
    # + final season). Passes 1-2 go through bridge_players, i.e. only
    # drafted / NFL-roster players, so before 2026-10-30 every combine
    # attendee who never reached the NFL went unmatched: combine_invited and
    # all combine metrics were present for 0.0% of non-NFL players, leaking
    # the outcome into steps 8 / 8b. An id already claimed by passes 1-2 is
    # never reassigned, and an id claimed by two pass-3 rows is dropped
    # (better unmatched than mislabeled).
    from src.utils.college_id_match import CollegeIdResolver
    unmatched = raw["player_id"].isna() & raw["player_name"].notna()
    if unmatched.any():
        res = CollegeIdResolver(logger).resolve(
            raw.loc[unmatched].assign(_last=raw.loc[unmatched, "cohort_year"] - 1),
            "player_name", "school", "_last")
        claimed = set(raw["player_id"].dropna().astype(str))
        res.loc[res["player_id"].isin(claimed), "player_id"] = None
        dup = res["player_id"].notna() & res["player_id"].duplicated(keep=False)
        res.loc[dup, "player_id"] = None
        raw.loc[unmatched, "player_id"] = res["player_id"]
        logger.info("Matched via college universe: %d additional (%s); %d ambiguous dropped",
                    int(res["player_id"].notna().sum()),
                    res.loc[res["player_id"].notna(), "match_pass"].value_counts().to_dict(),
                    int(dup.sum()))

    total_matched = raw["player_id"].notna().sum()
    logger.info("Total matched: %d / %d (%.1f%%)",
                total_matched, len(raw), 100 * total_matched / len(raw))

    # ── Z-score metrics within (position_group × cohort_year) ──────────────────
    logger.info("Computing cohort z-scores...")
    for metric in COMBINE_METRICS:
        raw_col = f"combine_{metric}"
        z_col   = f"combine_{metric}_z"
        if raw_col not in raw.columns:
            continue
        raw[raw_col] = pd.to_numeric(raw[raw_col], errors="coerce")
        raw[z_col] = _zscore_within_cohort(raw, raw_col, ["position_group", "cohort_year"])
        n_valid = raw[z_col].notna().sum()
        logger.info("  %s: %d players with z-score", z_col, n_valid)

    # ── Build output ────────────────────────────────────────────────────────────
    raw["combine_invited"] = 1

    keep_cols = (
        ["player_id", "combine_invited", "cohort_year"]
        + [f"combine_{m}" for m in COMBINE_METRICS if f"combine_{m}" in raw.columns]
        + [f"combine_{m}_z" for m in COMBINE_METRICS if f"combine_{m}_z" in raw.columns]
    )
    out = raw[[c for c in keep_cols if c in raw.columns]].rename(
        columns={"cohort_year": "combine_draft_year"}
    )

    # Deduplicate: keep the row with the most non-null z-scores per player.
    # IMPORTANT: drop_duplicates collapses all NaN-keyed rows to one, which
    # silently destroys unmatched current-year prospects. Split, dedupe only
    # the matched rows, keep all unmatched rows verbatim.
    out["_n_z"] = out[[f"combine_{m}_z" for m in COMBINE_METRICS
                        if f"combine_{m}_z" in out.columns]].notna().sum(axis=1)
    matched = (out[out["player_id"].notna()]
               .sort_values("_n_z", ascending=False)
               .drop_duplicates(subset=["player_id"], keep="first"))
    unmatched = out[out["player_id"].isna()]
    out = (pd.concat([matched, unmatched], ignore_index=True)
             .drop(columns=["_n_z"])
             .reset_index(drop=True))

    # Keep rows without player_id too (future use / current prospects)
    out_path = PROCESSED_DIR / "combine_metrics.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved %d combine rows → %s", len(out), out_path)

    # Sanity summary
    logger.info("Draft years in data: %d–%d",
                int(out["combine_draft_year"].min()),
                int(out["combine_draft_year"].max()))

    matched_out = out["player_id"].notna().sum()
    logger.info("Players with CFBD player_id: %d / %d", matched_out, len(out))

    z_col_sample = "combine_forty_z"
    if z_col_sample in out.columns:
        sample = out[z_col_sample].dropna()
        logger.info("forty_z: mean=%.3f std=%.3f (expect ~0, ~1 within cohorts)",
                    sample.mean(), sample.std())

    logger.info("Step 00d complete.")


if __name__ == "__main__":
    main()
