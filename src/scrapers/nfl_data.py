"""
NFL Data Module

Uses nfl_data_py (backed by nflverse GitHub data) to fetch:
  - NFL draft picks with career AV (w_av) via import_draft_picks()
  - Season-level snap count data via import_snap_counts()
  - Player ID crosswalk via import_ids()
  - UDFA players via import_players()

No web scraping required — all data pulled from public GitHub repos.
"""

import logging
import pickle
import re
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    PROCESSED_DIR,
    RAW_DIR,
    POSITION_TO_GROUP,
    ROLE_SNAP_GAMES_MIN,
    ROLE_SNAP_PCT_THRESHOLD,
    ROLE_SNAP_PCT_THRESHOLD_BY_GROUP,
    TRAJECTORY_FEATURE_WINDOW,
    TRAJECTORY_LONG_TERM_WINDOW,
    TRAJECTORY_TRAIN_CUTOFF_YEAR,
    TRAJECTORY_MAX_YEARS,
    TRAJECTORY_SETTLED_CUTOFF_YEARS,
    TRAJECTORY_IMPACT_VAL_SPLIT,
    POSITION_GROUP_ORDINAL,
    LGBM_PARAMS,
    OTC_POSITION_TO_GROUP,
    REPLACEMENT_MARGIN,
    REPLACEMENT_POS_QUANTILE,
    REPLACEMENT_POS_MIN_YEARS,
    REPLACEMENT_POS_MIN_N,
    REPLACEMENT_CONTRACT_WINDOW_YEARS,
    VET_MIN_MODE_MIN_COUNT,
    VET_MIN_PRORATION_FLOOR_PCT,
    VET_MIN_FALLBACK_PCT,
    TAG_MAX_YEARS,
    TAG_MIN_GUARANTEED_RATIO,
    TAG_MIN_APY_CAP_PCT,
    SECOND_CONTRACT_MAX_YEARS_AFTER_DRAFT,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

_nfl_available = False
try:
    import nfl_data_py as nfl
    _nfl_available = True
except ImportError:
    logger.warning("nfl_data_py not installed. Run: pip install nfl_data_py")

_rapidfuzz_available = False
try:
    from rapidfuzz import fuzz as _rfuzz
    _rapidfuzz_available = True
except ImportError:
    logger.warning("rapidfuzz not installed. Fuzzy name matching disabled. Run: pip install rapidfuzz")

_lgbm_available = False
try:
    import lightgbm as lgb
    _lgbm_available = True
except ImportError:
    pass

_CURRENT_YEAR = datetime.now().year

TRAJECTORY_MODELS_DIR = PROCESSED_DIR / "trajectory_models"

# Snap column to use per position group
_OFFENSE_GROUPS = {"QB", "RB", "WR", "TE", "OL"}
_DEFENSE_GROUPS = {"DL", "EDGE", "LB", "DB"}

SNAP_FEATURE_COLS = [
    "mean_snap_pct_4yr", "peak_snap_pct_4yr",
    "year1_snap_pct", "year4_snap_pct",
    "snap_trend_4yr", "n_active_seasons_4yr",
]

IMPACT_FEATURE_COLS = SNAP_FEATURE_COLS + ["draft_round_norm", "draft_pick_log"]


# ── Helpers ────────────────────────────────────────────────────────────────────

# Trailing generational suffixes that CFBD and PFR disagree about (CFBD often
# stores "Joe Milton III" while PFR stores "Joe Milton" and vice-versa).
_NAME_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, normalize hyphens to spaces, drop punctuation,
    strip generational suffixes, collapse whitespace.

    Hyphens are converted to spaces (not deleted) so compound surnames like
    "Tryon-Shoyinka" tokenize as ["tryon", "shoyinka"] — that lets the
    token-subset pass in match_draft_picks() recognize "Joe Tryon" (CFBD
    college name) ⊂ "Joe Tryon Shoyinka" (PFR post-name-change name)."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower().replace("-", " ")
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = " ".join(name.split())
    return _NAME_SUFFIX_RE.sub("", name).strip()


def _normalize_school(school: str) -> str:
    """Lowercase, expand 'St'/'St.' → 'state', drop parenthetical regional
    qualifiers, strip non-alphanumerics. CFBD uses 'Florida State' while
    nflverse/PFR uses 'Florida St.' for the same school — without expansion,
    these normalize to different keys and Pass 1 fails."""
    if not isinstance(school, str):
        return ""
    s = school.lower()
    # Drop parenthetical qualifiers like "Miami (FL)" → "Miami"
    s = re.sub(r"\([^)]*\)", "", s)
    # "St" / "St." as a SUFFIX → "state". Only match when preceded by
    # whitespace; "St. Paul's" / "St. John's" must not collapse to "State
    # Paul's" / "State John's" (regression vs original normalization).
    s = re.sub(r"(?<=\s)st\.?\b", "state", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _snap_col_for_group(position_group: str) -> str:
    if position_group in _OFFENSE_GROUPS:
        return "offense_pct"
    if position_group in _DEFENSE_GROUPS:
        return "defense_pct"
    return "st_pct"


def _build_id_crosswalk() -> dict[str, str]:
    """
    Build a cfbref_id → pfr_player_id lookup from nfl.import_ids().
    Returns empty dict if import fails or required columns are missing.
    """
    if not _nfl_available:
        return {}
    try:
        ids = nfl.import_ids()
        # column names vary by version: try cfbref_id / cfb_id / college_id
        cfb_col = next(
            (c for c in ["cfbref_id", "cfb_id", "college_id"] if c in ids.columns),
            None,
        )
        pfr_col = next(
            (c for c in ["pfr_id", "pfr_player_id"] if c in ids.columns),
            None,
        )
        if cfb_col is None or pfr_col is None:
            return {}
        sub = ids[ids[cfb_col].notna() & ids[pfr_col].notna()]
        crosswalk = dict(zip(sub[cfb_col].astype(str), sub[pfr_col].astype(str)))
        logger.info("Built ID crosswalk: %d cfbref → pfr entries", len(crosswalk))
        return crosswalk
    except Exception as exc:
        logger.warning("Could not build ID crosswalk: %s", exc)
        return {}


# ── Draft picks + AV ──────────────────────────────────────────────────────────

def load_nfl_draft_picks() -> pd.DataFrame:
    """
    Load all available NFL draft picks with career stats.
    Returns columns: pfr_player_id, pfr_player_name, college, season,
                     w_av, games, seasons_started, position, round, pick
    """
    if not _nfl_available:
        raise ImportError("nfl_data_py required. Run: pip install nfl_data_py")
    df = nfl.import_draft_picks()
    df["_name_norm"] = df["pfr_player_name"].apply(_normalize_name)
    df["_school_norm"] = df["college"].apply(_normalize_school)
    df["w_av"] = pd.to_numeric(df["w_av"], errors="coerce").fillna(0.0)
    df["games"] = pd.to_numeric(df["games"], errors="coerce").fillna(0.0)
    logger.info("Loaded %d NFL draft picks (seasons %d–%d)",
                len(df), df["season"].min(), df["season"].max())
    return df


def match_draft_picks(cfbd_draft: pd.DataFrame,
                      nfl_draft: pd.DataFrame,
                      id_crosswalk: Optional[dict] = None) -> pd.DataFrame:
    """
    Match CFBD draft picks to nfl_data_py draft picks to get pfr_player_id + AV.

    Matching priority:
      P. Draft year + overall pick. Authoritative: on every name-matched
         pick the two sources agree on (year, pick) except where the name
         match was wrong (RG3 vs 6th-round guard Robert Griffin, Ohio State
         vs Southern Miss Michael Thomas, Terry vs Derrick Johnson), and
         every CFBD pick has an nflverse row at its (year, pick).
      0. config.BRIDGE_NAME_OVERRIDES — manual (cfbd_name, college, year)
         → pfr_player_name mapping for nickname swaps where the first name is
         unrelated (Sauce/Ahmad Gardner, Tank/Nathaniel Dell, etc.) and no
         heuristic could safely catch them.
      1. Exact normalized name + school + draft_year
      1b. Token-subset name + school + draft_year  (compound-surname /
          dropped-suffix cases like "Joe Tryon" ↔ "Joe Tryon Shoyinka",
          "Cornellius Tank Carradine" ↔ "Tank Carradine")
      2. Exact normalized name + draft_year
      2b. Token-subset name + draft_year, school-independent, uniqueness-
          guarded (handles cases where school normalization still disagrees,
          e.g. UTSA ↔ Texas-San Antonio, S.F. Austin ↔ Stephen F. Austin)
      3. Exact normalized name + school (year off by 1)
      4. ID crosswalk via cfb_player_id (if crosswalk provided)
      5. Fuzzy name (max(ratio, token_sort_ratio) ≥ 85) + school + year ±1
         (rapidfuzz required; catches nickname / shortened-given-name cases
         like "Delano Hill" ↔ "Lano Hill")
    """
    if id_crosswalk is None:
        id_crosswalk = {}

    # Manual nickname overrides (config.BRIDGE_NAME_OVERRIDES). Imported here
    # rather than at module top so the scraper module stays importable when
    # config.py is unavailable (test scaffolding).
    try:
        from config import BRIDGE_NAME_OVERRIDES as _NAME_OVERRIDES
    except Exception:
        _NAME_OVERRIDES = {}
    # Build a normalized-key index of overrides for fast Pass 0 lookup.
    # Key: (cfbd_name_norm, school_norm, year) → override pfr_name_norm.
    _override_by_key = {
        (_normalize_name(cfbd_name), _normalize_school(school), yr):
            _normalize_name(pfr_name)
        for (cfbd_name, school, yr), pfr_name in _NAME_OVERRIDES.items()
    }

    cfbd = cfbd_draft.copy()
    cfbd["_name_norm"] = cfbd["player_name"].apply(_normalize_name)
    cfbd["_school_norm"] = cfbd["college_team"].apply(_normalize_school)

    by_pick = {}
    if "pick" in nfl_draft.columns:
        for i, (season, pick) in enumerate(zip(nfl_draft["season"], nfl_draft["pick"])):
            if pd.notna(season) and pd.notna(pick):
                by_pick.setdefault((int(season), int(pick)), []).append(i)

    def _lookup(row):
        yr = row["draft_year"]
        nn = row["_name_norm"]
        sn = row["_school_norm"]

        # Pass P: draft year + overall pick
        op = row.get("overall_pick")
        if pd.notna(op) and pd.notna(yr):
            hit = by_pick.get((int(yr), int(op)), [])
            if len(hit) == 1:
                return nfl_draft.iloc[hit[0]], "draft_pick"

        # Pass 0: manual override map (highest priority). Looks up the cfbd
        # row's (name, school, year) and, if present, fetches the override
        # NFL name within the same draft year.
        override_pfr_nn = _override_by_key.get((nn, sn, yr))
        if override_pfr_nn:
            mask0 = ((nfl_draft["_name_norm"] == override_pfr_nn) &
                     (nfl_draft["season"] == yr))
            hits0 = nfl_draft[mask0]
            if len(hits0) >= 1:
                return hits0.iloc[0], "manual_override"

        # Pass 1: name + school + year
        mask = ((nfl_draft["_name_norm"] == nn) &
                (nfl_draft["_school_norm"] == sn) &
                (nfl_draft["season"] == yr))
        hits = nfl_draft[mask]
        if len(hits) == 1:
            return hits.iloc[0], "exact_school_year"

        # Pass 1b: token-subset name + school + year. Either side's tokens are
        # a strict superset of the other's. Covers CFBD↔PFR name drift where
        # one side has an extra token the other lacks (compound surnames,
        # informal-name parentheticals, missed Jr/Sr beyond what the suffix
        # stripper handles).
        nn_tokens = set(nn.split())
        if nn_tokens:
            cand_sy = nfl_draft[
                (nfl_draft["_school_norm"] == sn) &
                (nfl_draft["season"] == yr)
            ]
            for _, cand in cand_sy.iterrows():
                cn = cand["_name_norm"]
                cn_tokens = set(cn.split())
                if not cn_tokens or nn_tokens == cn_tokens:
                    continue   # equality was already covered by Pass 1
                if nn_tokens <= cn_tokens or cn_tokens <= nn_tokens:
                    return cand, "subset_school_year"

        # Pass 2: name + year only
        mask2 = ((nfl_draft["_name_norm"] == nn) &
                 (nfl_draft["season"] == yr))
        hits2 = nfl_draft[mask2]
        if len(hits2) == 1:
            return hits2.iloc[0], "exact_year"

        # Pass 2b: token-subset name + year, school-independent, uniqueness-
        # guarded. Covers the case where school normalization still disagrees
        # after the State/St. fix (e.g. UTSA vs Texas-San Antonio for Tariq
        # "Riq" Woolen). Only accept if exactly one same-year candidate has a
        # token-subset name relationship — otherwise abstain to avoid
        # picking the wrong namesake across schools.
        if nn_tokens:
            cand_y = nfl_draft[nfl_draft["season"] == yr]
            subset_hits = []
            for _, cand in cand_y.iterrows():
                cn = cand["_name_norm"]
                cn_tokens = set(cn.split())
                if not cn_tokens or nn_tokens == cn_tokens:
                    continue
                if nn_tokens <= cn_tokens or cn_tokens <= nn_tokens:
                    subset_hits.append(cand)
            if len(subset_hits) == 1:
                return subset_hits[0], "subset_year"

        # Pass 3: name + school, year ±1
        mask3 = ((nfl_draft["_name_norm"] == nn) &
                 (nfl_draft["_school_norm"] == sn) &
                 (nfl_draft["season"].between(yr - 1, yr + 1)))
        hits3 = nfl_draft[mask3]
        if len(hits3) >= 1:
            return hits3.iloc[0], "year_fuzzy"

        # Pass 4: ID crosswalk via cfb_player_id
        cfb_pid = row.get("cfb_player_id")
        if cfb_pid and str(cfb_pid) in id_crosswalk:
            pfr = id_crosswalk[str(cfb_pid)]
            pfr_rows = nfl_draft[nfl_draft.get("pfr_player_id", pd.Series(dtype=str)) == pfr]
            if len(pfr_rows) >= 1:
                return pfr_rows.iloc[0], "crosswalk"
            # crosswalk hit but no row in draft table — synthetic row
            return pd.Series({"pfr_player_id": pfr, "w_av": 0.0, "games": 0,
                               "cfb_player_id": cfb_pid}), "crosswalk"

        # Pass 5: fuzzy name (requires rapidfuzz). Use max of ratio and
        # token_sort_ratio — token_sort alone misses surname-dominant
        # nickname swaps ("Delano Hill" ↔ "Lano Hill" sorts to "delano hill"
        # vs "hill lano" → 40%) while plain ratio handles them (90%).
        # Two-tier threshold: ≥85 always accepts; 75-85 accepts only when
        # the surname token (last token) matches exactly. That picks up
        # formal-name ↔ nickname pairs at the same school+year (Patrick ↔
        # Pat Surtain II, Christopher ↔ Chris Carson, Cameron ↔ Cam
        # Jurgens, Carnell ↔ Cadillac Williams, etc.) without loosening
        # the bar for everyone.
        if _rapidfuzz_available and nn:
            nn_last = nn.split()[-1] if nn.split() else ""
            best_score, best_row = 0, None
            candidates = nfl_draft[
                nfl_draft["season"].between(yr - 1, yr + 1) &
                (nfl_draft["_school_norm"] == sn)
            ]
            for _, cand in candidates.iterrows():
                cn = cand["_name_norm"]
                score = max(_rfuzz.ratio(nn, cn),
                            _rfuzz.token_sort_ratio(nn, cn))
                cn_tokens = cn.split()
                surname_match = bool(nn_last) and bool(cn_tokens) and \
                                cn_tokens[-1] == nn_last
                if score > best_score and (score >= 85 or
                                            (score >= 75 and surname_match)):
                    best_score = score
                    best_row = cand
            if best_row is not None:
                return best_row, "fuzzy"

        # Pass 5b: fuzzy name + year only (no school), uniqueness-guarded.
        # Catches cases where school normalization disagrees beyond the
        # State/St. fix (UTSA ↔ Texas-San Antonio, Stephen F. Austin ↔
        # S.F. Austin, Sam Houston ↔ Sam Houston St., etc.) Tighter
        # threshold than school-gated Pass 5 (≥85) AND uniqueness — there
        # must be exactly one fuzzy hit in the draft class, since fuzzy
        # without school is otherwise lossy.
        if _rapidfuzz_available and nn:
            cand_y = nfl_draft[nfl_draft["season"] == yr]
            fuzzy_hits = []
            for _, cand in cand_y.iterrows():
                cn = cand["_name_norm"]
                score = max(_rfuzz.ratio(nn, cn),
                            _rfuzz.token_sort_ratio(nn, cn))
                if score >= 85:
                    fuzzy_hits.append(cand)
            if len(fuzzy_hits) == 1:
                return fuzzy_hits[0], "fuzzy_year"

        return None, "unmatched"

    pfr_ids, w_avs, games_list, cfb_ids, qualities = [], [], [], [], []
    for _, row in cfbd.iterrows():
        m, quality = _lookup(row)
        if m is None:
            pfr_ids.append(None)
            w_avs.append(0.0)
            games_list.append(0)
            cfb_ids.append(None)
        else:
            pfr_ids.append(m.get("pfr_player_id"))
            w_avs.append(float(m.get("w_av") or 0.0))
            games_list.append(int(m.get("games") or 0))
            cfb_ids.append(m.get("cfb_player_id"))
        qualities.append(quality)

    cfbd["pfr_id"] = pfr_ids
    cfbd["career_w_av"] = w_avs
    cfbd["nfl_games"] = games_list
    cfbd["cfb_player_id"] = cfb_ids
    cfbd["match_quality"] = qualities
    cfbd = cfbd.drop(columns=["_name_norm", "_school_norm"])

    matched_n = sum(1 for x in pfr_ids if x is not None)
    quality_counts = pd.Series(qualities).value_counts().to_dict()
    logger.info("Matched %d / %d CFBD picks to nfl_data_py records",
                matched_n, len(cfbd))
    logger.info("Match quality breakdown: %s", quality_counts)
    return cfbd


# ── UDFA players ──────────────────────────────────────────────────────────────

def load_udfa_players(snap_data: pd.DataFrame,
                      start_year: int = 2016,
                      end_year: Optional[int] = None,
                      min_seasons_with_snaps: int = 1) -> pd.DataFrame:
    """
    Load UDFA players who made NFL rosters using nfl.import_players().

    Filters to players who logged snap data in at least min_seasons_with_snaps
    regular-season NFL seasons (removes camp bodies who never contributed).

    Returns a DataFrame with the same schema as match_draft_picks() output.
    """
    if not _nfl_available:
        logger.warning("nfl_data_py not available — skipping UDFA players")
        return pd.DataFrame()

    if end_year is None:
        end_year = _CURRENT_YEAR

    try:
        players = nfl.import_players()
    except Exception as exc:
        logger.error("Failed to load NFL players: %s", exc)
        return pd.DataFrame()

    # Identify undrafted: no draft_round and no draft_pick
    round_col = next((c for c in ["draft_round", "round"] if c in players.columns), None)
    # import_players() uses rookie_season for year; draft_year is NaN for UDFAs
    year_col = next((c for c in ["rookie_season", "entry_year", "rookie_year"] if c in players.columns), None)

    if round_col is None or year_col is None:
        logger.warning("import_players() missing expected columns — skipping UDFA")
        return pd.DataFrame()

    udfa = players[
        players[round_col].isna() &
        players[year_col].notna() &
        (players[year_col] >= start_year) &
        (players[year_col] <= end_year)
    ].copy()

    # Filter to players who appeared in snap count data (made a real roster contribution)
    pfr_col = next((c for c in ["pfr_id", "pfr_player_id"] if c in udfa.columns), None)
    if not snap_data.empty and "pfr_player_id" in snap_data.columns and pfr_col:
        players_with_snaps = set(snap_data["pfr_player_id"].dropna().unique())
        udfa = udfa[udfa[pfr_col].notna() & udfa[pfr_col].isin(players_with_snaps)]

        if min_seasons_with_snaps > 1:
            season_counts = snap_data.groupby("pfr_player_id")["season"].nunique()
            qualifying = set(season_counts[season_counts >= min_seasons_with_snaps].index)
            udfa = udfa[udfa[pfr_col].isin(qualifying)]
    elif pfr_col:
        # No snap data: require at least a pfr_id to indicate the player was tracked
        udfa = udfa[udfa[pfr_col].notna()]

    if udfa.empty:
        logger.info("No UDFA players found in %d–%d", start_year, end_year)
        return pd.DataFrame()

    # Normalise to draft-pick schema
    # import_players() columns: display_name, college_name, rookie_season, pfr_id, position
    # NOTE: import_players() has no career AV data; career_w_av is left 0.0 for UDFAs.
    #       The trajectory models provide impact scores from snap data instead.
    pfr_col = pfr_col or next((c for c in ["pfr_id", "pfr_player_id"] if c in udfa.columns), None)
    name_col = next((c for c in ["display_name", "short_name", "full_name"] if c in udfa.columns), None)
    college_col = next((c for c in ["college_name", "college"] if c in udfa.columns), None)
    pos_col = next((c for c in ["position", "pos"] if c in udfa.columns), None)

    udfa_df = pd.DataFrame({
        "player_name":   udfa[name_col].values if name_col else "",
        "college_team":  udfa[college_col].values if college_col else "",
        "draft_year":    udfa[year_col].values,
        "draft_round":   None,
        "draft_pick":    None,
        "position":      udfa[pos_col].values if pos_col else None,
        "pfr_id":        udfa[pfr_col].values if pfr_col else None,
        "career_w_av":   0.0,
        "nfl_games":     0,
        "cfb_player_id": None,
        "is_udfa":       True,
        "match_quality": "udfa",
    })

    udfa_df["_name_norm"] = udfa_df["player_name"].apply(_normalize_name)
    udfa_df["_school_norm"] = udfa_df["college_team"].apply(_normalize_school)

    logger.info("Loaded %d UDFA players with snap data (%d–%d)",
                len(udfa_df), start_year, end_year)
    return udfa_df


def match_udfa_to_cfbd(udfa_df: pd.DataFrame,
                       player_seasons_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve CFBD player_id for UDFA players by matching on name + school.
    Uses same normalization as match_draft_picks() with exact and fuzzy passes.
    """
    if udfa_df.empty:
        return udfa_df

    seasons = player_seasons_df.copy()
    seasons["_name_norm"] = seasons["player_name"].apply(_normalize_name)
    seasons["_school_norm"] = seasons["school"].apply(_normalize_school)

    latest = (seasons.sort_values("season_year")
              .drop_duplicates(["_name_norm", "_school_norm"], keep="last")
              [["_name_norm", "_school_norm", "player_id"]])

    player_ids = []
    for _, row in udfa_df.iterrows():
        nn = row.get("_name_norm", "")
        sn = row.get("_school_norm", "")

        # Exact name + school
        mask = (latest["_name_norm"] == nn) & (latest["_school_norm"] == sn)
        hits = latest[mask]
        if len(hits) >= 1:
            player_ids.append(hits.iloc[0]["player_id"])
            continue

        # Exact name only (school name may differ)
        mask2 = latest["_name_norm"] == nn
        hits2 = latest[mask2]
        if len(hits2) == 1:
            player_ids.append(hits2.iloc[0]["player_id"])
            continue

        # Fuzzy name + same school (rapidfuzz). Same widened metric as
        # match_draft_picks — handles nickname / shortened-name divergences.
        if _rapidfuzz_available and nn:
            best_score, best_pid = 0, None
            candidates = latest[latest["_school_norm"] == sn]
            for _, s_row in candidates.iterrows():
                cn = s_row["_name_norm"]
                score = max(_rfuzz.ratio(nn, cn),
                            _rfuzz.token_sort_ratio(nn, cn))
                if score >= 85 and score > best_score:
                    best_score = score
                    best_pid = s_row["player_id"]
            player_ids.append(best_pid)
        else:
            player_ids.append(None)

    result = udfa_df.copy()
    result["player_id"] = player_ids
    result = result.drop(columns=["_name_norm", "_school_norm"], errors="ignore")

    matched_n = sum(1 for x in player_ids if x is not None)
    logger.info("Resolved CFBD player_id for %d / %d UDFA players",
                matched_n, len(result))
    return result


# ── Snap counts ────────────────────────────────────────────────────────────────

def fetch_seasonal_snap_pcts(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Aggregate game-level snap counts to season-level snap fractions.
    Returns one row per pfr_player_id × season with columns:
      pfr_player_id, season, offense_pct, defense_pct, st_pct, n_games
    """
    if not _nfl_available:
        raise ImportError("nfl_data_py required.")

    years = [y for y in range(start_year, end_year + 1)]
    try:
        raw = nfl.import_snap_counts(years)
    except Exception as exc:
        logger.error("Failed to load snap counts: %s", exc)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    raw = raw[raw["game_type"] == "REG"].copy()

    agg = (raw.groupby(["pfr_player_id", "season"])
           .agg(
               offense_pct=("offense_pct", "mean"),
               defense_pct=("defense_pct", "mean"),
               st_pct=("st_pct", "mean"),
               n_games=("offense_pct", "count"),
           )
           .reset_index())

    logger.info("Loaded snap counts: %d player-seasons (%d–%d)",
                len(agg), start_year, end_year)
    return agg


def reg_season_games(season: int) -> int:
    """Regular-season games per team: 17 from 2021, 16 before."""
    return 17 if int(season) >= 2021 else 16


_SNAP_PCT_COLS = ("offense_pct", "defense_pct", "st_pct")


def season_snap_shares(snap_data: pd.DataFrame) -> pd.DataFrame:
    """Snap pct columns as shares of the TEAM's regular season.

    The cached season rows (fetch_seasonal_snap_pcts) hold the mean pct over
    the games a player appeared in, so a season cut short by injury reads as
    a full one. Scaling by n_games / the team's regular-season games counts
    missed games as 0; capped at 1 (a mid-season trade can mean more
    appearances than the team's games). Returns a copy; unchanged (with a
    warning) if n_games is absent.
    """
    if snap_data.empty or "n_games" not in snap_data.columns:
        if not snap_data.empty:
            logger.warning("snap data has no n_games: snap pct stays per game appeared")
        return snap_data
    out = snap_data.copy()
    frac = out["n_games"] / out["season"].map(reg_season_games)
    for col in _SNAP_PCT_COLS:
        if col in out.columns:
            out[col] = (out[col] * frac).clip(upper=1.0)
    return out


# ── Snap features for trajectory models ───────────────────────────────────────

def fetch_snap_features(bridge_players: pd.DataFrame,
                        snap_data: pd.DataFrame,
                        window: int = TRAJECTORY_FEATURE_WINDOW) -> pd.DataFrame:
    """
    Compute first-N-year snap features for bridge players.

    Returns a DataFrame indexed by pfr_id with SNAP_FEATURE_COLS columns.
    Players without snap data are omitted.
    """
    if snap_data.empty or bridge_players.empty:
        return pd.DataFrame(columns=SNAP_FEATURE_COLS)
    snap_data = season_snap_shares(snap_data)

    records = []
    for _, player in bridge_players.iterrows():
        pfr_id = player.get("pfr_id")
        draft_year = player.get("draft_year")
        pos_group = player.get("position_group", "")

        if not pfr_id or pd.isna(pfr_id) or not draft_year or pd.isna(draft_year):
            continue

        player_snaps = snap_data[snap_data["pfr_player_id"] == pfr_id].copy()
        if player_snaps.empty:
            continue

        snap_col = _snap_col_for_group(str(pos_group))
        if snap_col not in player_snaps.columns:
            snap_col = "offense_pct" if "offense_pct" in player_snaps.columns else "defense_pct"

        early = player_snaps[
            (player_snaps["season"] >= int(draft_year)) &
            (player_snaps["season"] < int(draft_year) + window)
        ].sort_values("season")

        if early.empty:
            continue

        snaps = early[snap_col].fillna(0.0).values
        n_games = early["n_games"].values if "n_games" in early.columns else np.ones(len(snaps))

        records.append({
            "pfr_id":               pfr_id,
            "mean_snap_pct_4yr":    float(np.mean(snaps)),
            "peak_snap_pct_4yr":    float(np.max(snaps)),
            "year1_snap_pct":       float(snaps[0]),
            "year4_snap_pct":       float(snaps[-1]) if len(snaps) >= window else np.nan,
            "snap_trend_4yr":       float(snaps[-1] - snaps[0]) if len(snaps) >= 2 else np.nan,
            "n_active_seasons_4yr": int(np.sum(n_games >= ROLE_SNAP_GAMES_MIN)),
        })

    if not records:
        return pd.DataFrame(columns=["pfr_id"] + SNAP_FEATURE_COLS)

    return pd.DataFrame(records).set_index("pfr_id")


def _snap_features_for_window(pfr_id: str,
                               snap_data: pd.DataFrame,
                               start_season: int,
                               end_season: int,
                               snap_col: str) -> Optional[float]:
    """Mean season snap share for a player in [start_season, end_season]
    (missed games count as 0). Returns None if no data."""
    snap_data = season_snap_shares(snap_data)
    sub = snap_data[
        (snap_data["pfr_player_id"] == pfr_id) &
        (snap_data["season"] >= start_season) &
        (snap_data["season"] <= end_season)
    ]
    if sub.empty or snap_col not in sub.columns:
        return None
    vals = sub[snap_col].dropna().values
    return float(np.mean(vals)) if len(vals) > 0 else None


# ── Trajectory models ──────────────────────────────────────────────────────────

def _trajectory_paths(variant: str = "") -> tuple:
    """Return (role_path, impact_path) for a labels variant.

    A non-empty variant suffixes the filenames so an A/B arm can fit its own
    trajectory models without clobbering the canonical production pickles.
    """
    suffix = f"_{variant}" if variant else ""
    return (TRAJECTORY_MODELS_DIR / f"role_trajectory{suffix}.pkl",
            TRAJECTORY_MODELS_DIR / f"impact_trajectory{suffix}.pkl")


def load_trajectory_models(variant: str = "") -> tuple:
    """
    Load fitted trajectory models from TRAJECTORY_MODELS_DIR.
    Returns (role_model, role_cols, impact_model, impact_cols).
    Returns (None, None, None, None) if no models exist.
    """
    role_model = role_cols = impact_model = impact_cols = None
    if not TRAJECTORY_MODELS_DIR.exists():
        return role_model, role_cols, impact_model, impact_cols

    role_path, impact_path = _trajectory_paths(variant)

    for path, label in [(role_path, "role"), (impact_path, "impact")]:
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if label == "role":
                role_model = data["model"]
                role_cols  = data["feature_cols"]
                logger.info("Loaded role trajectory model (val AUC=%.3f)", data.get("val_auc", float("nan")))
            else:
                impact_model = data["model"]
                impact_cols  = data["feature_cols"]
                logger.info("Loaded impact trajectory model (val R²=%.3f)", data.get("val_r2", float("nan")))
        except Exception as exc:
            logger.warning("Could not load %s: %s", path, exc)

    return role_model, role_cols, impact_model, impact_cols


# ── EPA column per position group ──────────────────────────────────────────────

_EPA_COL: dict[str, str] = {
    "QB": "passing_epa",
    "RB": "rushing_epa",
    "WR": "receiving_epa",
    "TE": "receiving_epa",
}


def fetch_nfl_seasonal_features(
        bridge_players: pd.DataFrame,
        snap_data: pd.DataFrame,
        max_years: int = TRAJECTORY_MAX_YEARS,
        refresh_epa: bool = False,
) -> pd.DataFrame:
    """
    Build wide-format per-player seasonal features for trajectory models.

    For each bridge player, extracts up to max_years seasons of:
      yn_snap_pct, yn_n_games, yn_epa   (n = 1..max_years)

    snap_pct uses the position-appropriate column (offense/defense/st), as a
    share of the team's regular season (season_snap_shares: missed games
    count as 0).
    epa comes from nflverse import_seasonal_data (QB/RB/WR/TE only; NaN otherwise).
    EPA data is cached at data/raw/nfl_epa.parquet; use refresh_epa=True to re-fetch.

    Returns a DataFrame indexed by pfr_id.
    """
    epa_cache = RAW_DIR / "nfl_epa.parquet"

    if not _nfl_available:
        logger.warning("nfl_data_py not available — EPA features will be NaN")
        epa_by_gsis: dict[tuple, float] = {}
        pfr_to_gsis = {}
    else:
        pfr_to_gsis = _build_pfr_to_gsis_crosswalk()
        all_draft_years = bridge_players["draft_year"].dropna().astype(int)
        if all_draft_years.empty:
            epa_by_gsis = {}
        else:
            seasonal = pd.DataFrame()
            if epa_cache.exists() and not refresh_epa:
                logger.info("Loading EPA data from cache → %s", epa_cache)
                seasonal = pd.read_parquet(epa_cache)
            else:
                seasons = list(range(int(all_draft_years.min()), _CURRENT_YEAR + 1))
                try:
                    seasonal = nfl.import_seasonal_data(seasons, s_type="REG")
                    seasonal = seasonal[["player_id", "season",
                                         "passing_epa", "rushing_epa", "receiving_epa"]
                                        ].copy()
                    RAW_DIR.mkdir(parents=True, exist_ok=True)
                    seasonal.to_parquet(epa_cache, index=False)
                    logger.info("Cached %d EPA rows → %s", len(seasonal), epa_cache)
                except Exception as exc:
                    logger.warning("Could not fetch nflverse EPA data: %s", exc)
                    if epa_cache.exists():
                        logger.info("Falling back to cached EPA data → %s", epa_cache)
                        seasonal = pd.read_parquet(epa_cache)

            epa_by_gsis = {}
            for _, row in seasonal.iterrows():
                epa_by_gsis[(str(row["player_id"]), int(row["season"]))] = row
        if not epa_by_gsis:
            pfr_to_gsis = {}

    snap_data = season_snap_shares(snap_data)
    snap_by_pfr = dict(tuple(snap_data.groupby("pfr_player_id"))) if not snap_data.empty else {}
    _empty_snaps = snap_data.iloc[0:0]

    records = []
    for _, player in bridge_players.iterrows():
        pfr_id    = player.get("pfr_id")
        draft_year = player.get("draft_year")
        pos_group  = str(player.get("position_group", ""))

        if not pfr_id or pd.isna(pfr_id) or not draft_year or pd.isna(draft_year):
            continue

        snap_col  = _snap_col_for_group(pos_group)
        epa_col   = _EPA_COL.get(pos_group)
        gsis_id   = pfr_to_gsis.get(pfr_id) if pfr_to_gsis else None

        player_snaps = snap_by_pfr.get(pfr_id, _empty_snaps)

        row: dict = {"pfr_id": pfr_id}
        for n in range(1, max_years + 1):
            season = int(draft_year) + (n - 1)
            prefix = f"y{n}"

            season_snap = player_snaps[player_snaps["season"] == season]
            if not season_snap.empty and snap_col in season_snap.columns:
                row[f"{prefix}_snap_pct"] = float(season_snap[snap_col].iloc[0])
                row[f"{prefix}_n_games"]  = float(season_snap["n_games"].iloc[0])
            else:
                row[f"{prefix}_snap_pct"] = np.nan
                row[f"{prefix}_n_games"]  = np.nan

            epa_val = np.nan
            if epa_col and gsis_id:
                epa_row = epa_by_gsis.get((gsis_id, season))
                if epa_row is not None:
                    epa_val = float(epa_row[epa_col]) if pd.notna(epa_row[epa_col]) else np.nan
            row[f"{prefix}_epa"] = epa_val

        records.append(row)

    if not records:
        cols = ["pfr_id"]
        for n in range(1, max_years + 1):
            cols += [f"y{n}_snap_pct", f"y{n}_n_games", f"y{n}_epa"]
        return pd.DataFrame(columns=cols).set_index("pfr_id")

    return pd.DataFrame(records).set_index("pfr_id")


def build_trajectory_dataset(
        bridge_players: pd.DataFrame,
        seasonal_feats: pd.DataFrame,
        contract_label=None,
        max_years: int = TRAJECTORY_MAX_YEARS,
        use_over_replacement: bool = False,
        value_label: Optional[pd.Series] = None,
        settled_cutoff: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build partial-snapshot training dataset for trajectory models.

    For each settled bridge player (draft_year + TRAJECTORY_SETTLED_CUTOFF_YEARS ≤ current_year),
    generates up to max_years training rows — one per partial window length:
      row for n_years=k has y1..yk populated, yk+1..y4 as NaN.

    Returns a DataFrame with columns:
      y1_snap_pct .. y4_epa, n_years_observed, position_group_ordinal,
      draft_round_norm, draft_pick_log, obs_weight, target_role, target_impact
    and, when `value_label` (pfr_id → value_target) is given, target_value
    (NaN for players it does not cover). Every row also carries draft_year (not
    a model feature) so callers can split validation by real draft class.
    `settled_cutoff` overrides the default last settled draft class.
    """
    if settled_cutoff is None:
        settled_cutoff = _CURRENT_YEAR - TRAJECTORY_SETTLED_CUTOFF_YEARS
    is_udfa = bridge_players.get("is_udfa", pd.Series(False, index=bridge_players.index)).fillna(False)

    # Accept either the DataFrame from compute_second_contract_label or the
    # legacy bare Series. The role target is hard-coupled to whatever counts as
    # a qualifying contract, so a gate change upstream propagates to role here
    # by construction — that is intended, not incidental.
    if isinstance(contract_label, pd.DataFrame):
        col = "label_apy_over_repl" if use_over_replacement else "label_apy_raw"
        impact_lookup = contract_label[col]
        role_lookup = contract_label["qualifies_role"]
    elif contract_label is not None:
        impact_lookup = contract_label
        role_lookup = (contract_label > 0).astype(float)
    else:
        impact_lookup = role_lookup = None

    feat_cols = []
    for n in range(1, max_years + 1):
        feat_cols += [f"y{n}_snap_pct", f"y{n}_n_games", f"y{n}_epa"]

    records = []
    for _, player in bridge_players.iterrows():
        pfr_id     = player.get("pfr_id")
        draft_year = player.get("draft_year")
        pos_group  = str(player.get("position_group", ""))

        if not pfr_id or pd.isna(pfr_id) or not draft_year or pd.isna(draft_year):
            continue
        if int(draft_year) > settled_cutoff:
            continue
        if bool(is_udfa.get(player.name, False)):
            continue
        if pfr_id not in seasonal_feats.index:
            continue

        if impact_lookup is not None and pfr_id in impact_lookup.index:
            target_impact = float(impact_lookup[pfr_id] or 0.0)
            target_role   = float(role_lookup[pfr_id] or 0.0)
        else:
            continue  # skip settled players with no contract label at all

        draft_round = player.get("round") or player.get("draft_round")
        draft_pick  = player.get("overall_pick") or player.get("pick")
        draft_round_norm = float(draft_round) if pd.notna(draft_round) else 8.0
        draft_pick_log   = float(np.log1p(draft_pick)) if pd.notna(draft_pick) else float(np.log1p(300))
        pos_ord = float(POSITION_GROUP_ORDINAL.get(pos_group, -1))

        player_row = seasonal_feats.loc[pfr_id]
        if isinstance(player_row, pd.DataFrame):
            player_row = player_row.iloc[0]

        for n_years in range(1, max_years + 1):
            # Only create this partial row if year n_years has snap data
            val = player_row.get(f"y{n_years}_snap_pct")
            if val is None or pd.isna(val):
                break

            base: dict = {
                "draft_year":             int(draft_year),
                "n_years_observed":       float(n_years),
                "position_group_ordinal": pos_ord,
                "draft_round_norm":       draft_round_norm,
                "draft_pick_log":         draft_pick_log,
                "obs_weight":             float(n_years),
                "target_role":            target_role,
                "target_impact":          target_impact,
            }
            if value_label is not None:
                base["target_value"] = float(value_label.get(pfr_id, np.nan))
            for col in feat_cols:
                yr = int(col[1])  # y1 → 1, y2 → 2, ...
                v = player_row[col]
                base[col] = float(v) if yr <= n_years and pd.notna(v) else np.nan

            records.append(base)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def _fit_trajectory_regressor(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                              val: np.ndarray, n_obs: np.ndarray, params: dict,
                              label: str):
    """Fit a trajectory regressor: early stopping on the `val` rows picks the
    iteration count, then the model is REFIT ON ALL ROWS at that count.

    `val` should hold out whole draft classes at every snapshot length (see
    _class_val_mask). Returns (model, val_r2, r2_by_seasons_observed, n_iter).
    """
    n_iter, val_r2, by_len = params["n_estimators"], float("nan"), {}
    if val.sum() >= 20 and (~val).sum() >= 20:
        probe = lgb.LGBMRegressor(**params)
        probe.fit(X[~val], y[~val], sample_weight=w[~val], eval_set=[(X[val], y[val])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
        n_iter = int(probe.best_iteration_ or n_iter)
        pred = probe.predict(X[val])
        def _r2(m):
            ss = np.sum((y[val][m] - y[val][m].mean()) ** 2)
            return float(1 - np.sum((y[val][m] - pred[m]) ** 2) / ss) if ss > 0 else float("nan")
        val_r2 = _r2(np.ones(int(val.sum()), bool))
        by_len = {int(k): round(_r2(n_obs[val] == k), 3) for k in np.unique(n_obs[val])}
        logger.info("Trajectory %s: %d train / %d val rows, val R²=%.3f "
                    "(by seasons observed: %s), best_iter=%d",
                    label, int((~val).sum()), int(val.sum()), val_r2, by_len, n_iter)
    model = lgb.LGBMRegressor(**{**params, "n_estimators": n_iter})
    model.fit(X, y, sample_weight=w)
    return model, val_r2, by_len, n_iter


def _class_val_mask(draft_year: pd.Series, n_val_classes: int = 2) -> np.ndarray:
    """Hold out the last `n_val_classes` draft classes present, at every
    snapshot length."""
    val_classes = sorted(draft_year.unique())[-n_val_classes:]
    return draft_year.isin(val_classes).to_numpy()


def fit_trajectory_models(
        train_df: pd.DataFrame,
        max_years: int = TRAJECTORY_MAX_YEARS,
        variant: str = "",
        impact_split: str = TRAJECTORY_IMPACT_VAL_SPLIT,
        only: Optional[str] = None,
) -> tuple:
    """
    Train LightGBM role and impact trajectory models on the partial-snapshot dataset.

    impact_split (config.TRAJECTORY_IMPACT_VAL_SPLIT):
      "legacy"   — validation rows inferred from n_years_observed, so it is
                   EVERY 1-2 season row; the model never trains on the short
                   snapshots it later scores for the newest classes.
      "by_class" — hold out the last two draft classes at every snapshot
                   length, then refit on all rows (_fit_trajectory_regressor).
    The role branch is unaffected: its isotonic calibration already refits on
    train + val. `only` ("role" | "impact") fits just one model and returns
    None for the other.

    Role model:   target_role   (binary 0/1, clipped to [0, 1] at inference)
    Impact model: target_impact (second contract APY/cap, floored at 0)

    Both models use sample_weight = obs_weight (n_years_observed).
    Position is encoded as position_group_ordinal (ordinal int feature).

    Returns (role_model, impact_model, role_feature_cols, impact_feature_cols).
    Saves role_trajectory.pkl and impact_trajectory.pkl to TRAJECTORY_MODELS_DIR.
    """
    if not _lgbm_available:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    TRAJECTORY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    snap_epa_cols = []
    for n in range(1, max_years + 1):
        snap_epa_cols += [f"y{n}_snap_pct", f"y{n}_n_games", f"y{n}_epa"]

    role_feature_cols   = snap_epa_cols + ["n_years_observed", "position_group_ordinal"]
    impact_feature_cols = snap_epa_cols + ["n_years_observed", "position_group_ordinal",
                                            "draft_round_norm", "draft_pick_log"]

    # Val split: hold out the last 2 years of settled players
    settled_cutoff = _CURRENT_YEAR - TRAJECTORY_SETTLED_CUTOFF_YEARS
    val_min = settled_cutoff - 1
    val_max = settled_cutoff

    params = {**LGBM_PARAMS, "n_estimators": 300}

    results = []
    for target_col, feature_cols, label in [
        ("target_role",   role_feature_cols,   "role"),
        ("target_impact", impact_feature_cols, "impact"),
    ]:
        if only is not None and label != only:
            results.append((None, []))
            continue
        available_cols = [c for c in feature_cols if c in train_df.columns]
        if (label == "impact" and impact_split == "by_class"
                and available_cols and "draft_year" in train_df.columns):
            model, val_r2, _, _ = _fit_trajectory_regressor(
                train_df[available_cols].to_numpy(dtype=float, na_value=np.nan),
                train_df[target_col].to_numpy(dtype=float),
                train_df["obs_weight"].to_numpy(dtype=float),
                _class_val_mask(train_df["draft_year"]),
                train_df["n_years_observed"].to_numpy(),
                params, "impact [by_class]")
            path = _trajectory_paths(variant)[1]
            with open(path, "wb") as f:
                pickle.dump({"model": model, "feature_cols": available_cols,
                             "val_r2": val_r2, "split": "by_class"}, f)
            logger.info("Saved trajectory impact model → %s", path)
            results.append((model, available_cols))
            continue
        if not available_cols or target_col not in train_df.columns:
            logger.warning("Trajectory %s: missing columns — skipping", label)
            results.append((None, available_cols))
            continue

        X = train_df[available_cols].to_numpy(dtype=float, na_value=np.nan)
        y = train_df[target_col].to_numpy(dtype=float)
        w = train_df["obs_weight"].to_numpy(dtype=float)

        # Approximate val split by n_years_observed = max (most informative rows, highest draft years)
        # Use last 2 draft cohorts as validation
        draft_year_approx = (_CURRENT_YEAR - TRAJECTORY_SETTLED_CUTOFF_YEARS
                             - train_df["n_years_observed"].astype(int) + 1)
        val_mask   = (draft_year_approx >= val_min) & (draft_year_approx <= val_max)
        train_mask = ~val_mask

        X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
        X_val,   y_val            = X[val_mask],   y[val_mask]

        if label == "role":
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)

        # Role uses uniform weights so the survivor-favoring effect of
        # obs_weight=n_years_observed (settled survivors get up to 10x more
        # weight than washouts) doesn't push the classifier to predict ~1
        # for everyone. Impact regressor keeps obs_weight since its target
        # is a continuous APY value where more-observed players are more
        # reliable.
        role_uniform = (label == "role")
        fit_weight = None if role_uniform else w_train

        use_val = val_mask.sum() >= 20
        if use_val:
            fit_kwargs = dict(
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            if fit_weight is not None:
                fit_kwargs["sample_weight"] = fit_weight
            model.fit(X_train, y_train, **fit_kwargs)
        else:
            if fit_weight is not None:
                model.fit(X_train, y_train, sample_weight=fit_weight)
            else:
                model.fit(X_train, y_train)

        if label == "role":
            from sklearn.metrics import roc_auc_score
            if use_val:
                val_proba_raw = model.predict_proba(X_val)[:, 1]
                val_auc = float(roc_auc_score(y_val, val_proba_raw)) if len(np.unique(y_val)) > 1 else float("nan")
                logger.info("Trajectory role: %d train rows, %d val rows, val AUC=%.3f (pre-calibration)",
                            train_mask.sum(), val_mask.sum(), val_auc)

                # Cross-fitted isotonic calibration on the full training set.
                # The val fold alone has biased class composition (mostly
                # early-snapshot rows of recent cohorts), so calibrating
                # against val pinned probabilities to that slice's positive
                # rate. cv=5 refits 5 LightGBMs across folds and averages the
                # calibrators — gives a much wider effective calibration set
                # spanning all snapshot lengths and cohorts. Uses fixed
                # n_estimators=best_iteration (no early stopping inside CV).
                best_iter = int(getattr(model, "best_iteration_", None) or params.get("n_estimators", 300))
                if len(np.unique(y_train)) > 1 and len(y_train) >= 250:
                    from sklearn.calibration import CalibratedClassifierCV
                    base = lgb.LGBMClassifier(**{**params, "n_estimators": best_iter})
                    calibrated = CalibratedClassifierCV(
                        estimator=base, method="isotonic", cv=5,
                    )
                    # Concat train + val so the calibrator fits across the
                    # full settled cohort distribution (same as the deployed
                    # role label population).
                    X_full = np.vstack([X_train, X_val])
                    y_full = np.concatenate([y_train, y_val])
                    calibrated.fit(X_full, y_full)
                    val_proba_cal = calibrated.predict_proba(X_val)[:, 1]
                    val_auc_cal = float(roc_auc_score(y_val, val_proba_cal)) if len(np.unique(y_val)) > 1 else float("nan")
                    logger.info(
                        "Trajectory role: cv=5 isotonic calibration on %d rows. "
                        "Pre  proba mean=%.3f std=%.3f. Post proba mean=%.3f std=%.3f. "
                        "Post val AUC=%.3f.",
                        len(y_full),
                        float(val_proba_raw.mean()), float(val_proba_raw.std()),
                        float(val_proba_cal.mean()),  float(val_proba_cal.std()),
                        val_auc_cal,
                    )
                    model = calibrated
            else:
                val_auc = float("nan")
                logger.info("Trajectory role: %d train rows (no val data)", train_mask.sum())
            path = _trajectory_paths(variant)[0]
            with open(path, "wb") as f:
                pickle.dump({"model": model, "feature_cols": available_cols, "val_auc": val_auc}, f)
        else:
            if use_val:
                val_pred = model.predict(X_val)
                ss_res = np.sum((y_val - val_pred) ** 2)
                ss_tot = np.sum((y_val - y_val.mean()) ** 2)
                val_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                logger.info("Trajectory impact: %d train rows, %d val rows, val R²=%.3f",
                            train_mask.sum(), val_mask.sum(), val_r2)
            else:
                val_r2 = float("nan")
                logger.info("Trajectory impact: %d train rows (no val data)", train_mask.sum())
            path = _trajectory_paths(variant)[1]
            with open(path, "wb") as f:
                pickle.dump({"model": model, "feature_cols": available_cols, "val_r2": val_r2}, f)

        logger.info("Saved trajectory %s model → %s", label, path)

        results.append((model, available_cols))

    role_model,   role_cols   = results[0]
    impact_model, impact_cols = results[1]
    return role_model, impact_model, role_cols, impact_cols


def score_trajectory_models(
        bridge_players: pd.DataFrame,
        seasonal_feats: pd.DataFrame,
        role_model,
        role_cols: list[str],
        impact_model,
        impact_cols: list[str],
        max_years: int = TRAJECTORY_MAX_YEARS,
) -> pd.DataFrame:
    """
    Score all bridge players using fitted trajectory models.

    Uses however many seasons of data are available per player (n_years_observed
    = 1..max_years). Later year columns are NaN for recent players — LightGBM
    handles this natively.

    Returns a DataFrame indexed by pfr_id with columns:
      role_trajectory_score, impact_trajectory_score, n_nfl_seasons_used
    """
    records = []
    for _, player in bridge_players.iterrows():
        pfr_id     = player.get("pfr_id")
        draft_round = player.get("round") or player.get("draft_round")
        draft_pick  = player.get("overall_pick") or player.get("pick")
        pos_group   = str(player.get("position_group", ""))

        role_score = impact_score = np.nan
        n_seasons_used = 0

        if pfr_id and pfr_id in seasonal_feats.index:
            player_row = seasonal_feats.loc[pfr_id]
            if isinstance(player_row, pd.DataFrame):
                player_row = player_row.iloc[0]

            # Count available seasons
            n_seasons_used = sum(
                1 for n in range(1, max_years + 1)
                if pd.notna(player_row.get(f"y{n}_snap_pct"))
            )

            if n_seasons_used > 0:
                draft_round_norm = float(draft_round) if pd.notna(draft_round) else 8.0
                draft_pick_log   = float(np.log1p(draft_pick)) if pd.notna(draft_pick) else float(np.log1p(300))
                pos_ord = float(POSITION_GROUP_ORDINAL.get(pos_group, -1))

                feat_vals: dict = {
                    "n_years_observed":       float(n_seasons_used),
                    "position_group_ordinal": pos_ord,
                    "draft_round_norm":       draft_round_norm,
                    "draft_pick_log":         draft_pick_log,
                }
                for n in range(1, max_years + 1):
                    for stat in ("snap_pct", "n_games", "epa"):
                        col = f"y{n}_{stat}"
                        feat_vals[col] = float(player_row[col]) if pd.notna(player_row.get(col)) else np.nan

                if role_model is not None and role_cols:
                    x = np.array([[feat_vals.get(c, np.nan) for c in role_cols]])
                    role_score = float(role_model.predict_proba(x)[0][1])

                if impact_model is not None and impact_cols:
                    x = np.array([[feat_vals.get(c, np.nan) for c in impact_cols]])
                    impact_score = float(max(0.0, impact_model.predict(x)[0]))

        records.append({
            "pfr_id":                  pfr_id,
            "role_trajectory_score":   role_score,
            "impact_trajectory_score": impact_score,
            "n_nfl_seasons_used":      n_seasons_used,
        })

    if not records:
        return pd.DataFrame(columns=["pfr_id", "role_trajectory_score",
                                     "impact_trajectory_score", "n_nfl_seasons_used"])

    return pd.DataFrame(records).set_index("pfr_id")


def score_trajectory_at_horizon(
        bridge_players: pd.DataFrame,
        seasonal_feats: pd.DataFrame,
        role_model,
        role_cols: list[str],
        impact_model,
        impact_cols: list[str],
        horizon: int,
        max_years: int = TRAJECTORY_MAX_YEARS,
) -> pd.DataFrame:
    """
    Score trajectory models using only the first `horizon` NFL seasons of data
    (y1..y_horizon), blanking later years even if observed.

    Used to produce denoised forward-looking labels for the draft model:
    a fixed horizon ensures every player's label reflects the same career
    stage and is always pre-second-contract-resolution. y3 is the safe choice
    because early extensions for high performers (Jefferson, Burrow patterns)
    are typically signed in the offseason after year 3.

    Returns DataFrame indexed by pfr_id with columns:
      role_traj_y{horizon}_score, impact_traj_y{horizon}_score, n_seasons_at_horizon
    """
    horizon = max(1, min(horizon, max_years))
    role_col = f"role_traj_y{horizon}_score"
    impact_col = f"impact_traj_y{horizon}_score"

    records = []
    for _, player in bridge_players.iterrows():
        pfr_id     = player.get("pfr_id")
        draft_round = player.get("round") or player.get("draft_round")
        draft_pick  = player.get("overall_pick") or player.get("pick")
        pos_group   = str(player.get("position_group", ""))

        role_score = impact_score = np.nan
        n_seasons_used = 0

        if pfr_id and pfr_id in seasonal_feats.index:
            player_row = seasonal_feats.loc[pfr_id]
            if isinstance(player_row, pd.DataFrame):
                player_row = player_row.iloc[0]

            n_seasons_used = sum(
                1 for n in range(1, horizon + 1)
                if pd.notna(player_row.get(f"y{n}_snap_pct"))
            )

            if n_seasons_used > 0:
                draft_round_norm = float(draft_round) if pd.notna(draft_round) else 8.0
                draft_pick_log   = float(np.log1p(draft_pick)) if pd.notna(draft_pick) else float(np.log1p(300))
                pos_ord = float(POSITION_GROUP_ORDINAL.get(pos_group, -1))

                feat_vals: dict = {
                    "n_years_observed":       float(n_seasons_used),
                    "position_group_ordinal": pos_ord,
                    "draft_round_norm":       draft_round_norm,
                    "draft_pick_log":         draft_pick_log,
                }
                for n in range(1, max_years + 1):
                    for stat in ("snap_pct", "n_games", "epa"):
                        col = f"y{n}_{stat}"
                        if n <= horizon:
                            feat_vals[col] = float(player_row[col]) if pd.notna(player_row.get(col)) else np.nan
                        else:
                            feat_vals[col] = np.nan

                if role_model is not None and role_cols:
                    x = np.array([[feat_vals.get(c, np.nan) for c in role_cols]])
                    role_score = float(role_model.predict_proba(x)[0][1])

                if impact_model is not None and impact_cols:
                    x = np.array([[feat_vals.get(c, np.nan) for c in impact_cols]])
                    impact_score = float(max(0.0, impact_model.predict(x)[0]))

        records.append({
            "pfr_id":              pfr_id,
            role_col:              role_score,
            impact_col:            impact_score,
            "n_seasons_at_horizon": n_seasons_used,
        })

    if not records:
        return pd.DataFrame(columns=["pfr_id", role_col, impact_col, "n_seasons_at_horizon"])

    return pd.DataFrame(records).set_index("pfr_id")


# ── Birth dates ───────────────────────────────────────────────────────────────

def load_birth_dates() -> dict[str, int]:
    """
    Return a dict mapping pfr_id → birth_year for all NFL players.
    Uses nfl.import_players() which has ~99.9% birth_date coverage.
    """
    if not _nfl_available:
        return {}
    try:
        players = nfl.import_players()
    except Exception as exc:
        logger.error("Failed to load NFL players for birth dates: %s", exc)
        return {}

    bd_col = next((c for c in ["birth_date", "dob"] if c in players.columns), None)
    pfr_col = next((c for c in ["pfr_id", "pfr_player_id"] if c in players.columns), None)
    if bd_col is None or pfr_col is None:
        logger.warning("import_players() missing birth_date or pfr_id column")
        return {}

    sub = players[[pfr_col, bd_col]].dropna()
    birth_years: dict[str, int] = {}
    for _, row in sub.iterrows():
        try:
            year = pd.to_datetime(row[bd_col]).year
            birth_years[str(row[pfr_col])] = year
        except Exception:
            pass

    logger.info("Loaded birth years for %d NFL players", len(birth_years))
    return birth_years


# ── Contract data (second contract label) ─────────────────────────────────────

def _build_pfr_to_gsis_crosswalk() -> dict[str, str]:
    """Return {pfr_id: gsis_id} from nfl.import_ids()."""
    if not _nfl_available:
        return {}
    try:
        ids = nfl.import_ids()
        pfr_col = next((c for c in ["pfr_id"] if c in ids.columns), None)
        gsis_col = next((c for c in ["gsis_id"] if c in ids.columns), None)
        if pfr_col is None or gsis_col is None:
            return {}
        sub = ids[ids[pfr_col].notna() & ids[gsis_col].notna()]
        return dict(zip(sub[pfr_col].astype(str), sub[gsis_col].astype(str)))
    except Exception as exc:
        logger.warning("Could not build pfr→gsis crosswalk: %s", exc)
        return {}


def fetch_league_contracts(cache_path: Optional[Path] = None,
                            refresh: bool = False) -> pd.DataFrame:
    """
    Fetch the FULL league-wide OverTheCap contract population and persist it.

    Historically contracts were pulled live on every run, inner-joined to bridge
    players immediately, and the `position` column was discarded — which threw
    away exactly the data needed to define a replacement level. This keeps the
    whole population, with position, and caches it to disk.

    Returns one row per contract with columns:
        gsis_id, player, position, position_group, year_signed, years,
        apy_cap_pct, apy, value, guaranteed, is_active, otc_id,
        draft_year, draft_round, implied_cap
    """
    cache_path = cache_path or (RAW_DIR / "nfl_contracts.parquet")
    if cache_path.exists() and not refresh:
        contracts = pd.read_parquet(cache_path)
        logger.info("Loaded %d league contract rows from cache %s",
                    len(contracts), cache_path)
        return contracts

    if not _nfl_available:
        raise ImportError("nfl_data_py required for contract data.")

    contracts = nfl.import_contracts()
    contracts = contracts.drop(columns=["cols"], errors="ignore")
    contracts["gsis_id"] = contracts["gsis_id"].astype(str).where(contracts["gsis_id"].notna())
    for col in ["draft_year", "draft_round", "year_signed", "years",
                "apy_cap_pct", "apy", "value", "guaranteed"]:
        if col in contracts.columns:
            contracts[col] = pd.to_numeric(contracts[col], errors="coerce")

    keep_cols = ["gsis_id", "player", "position", "year_signed", "years",
                 "apy_cap_pct", "apy", "value", "guaranteed", "is_active",
                 "otc_id", "draft_year", "draft_round"]
    keep_cols = [c for c in keep_cols if c in contracts.columns]
    contracts = contracts[keep_cols].dropna(subset=["year_signed", "apy_cap_pct"])

    # Map OTC position codes onto cfDARKO position groups (LS drops out).
    contracts["position_group"] = contracts["position"].map(OTC_POSITION_TO_GROUP)

    # Implied cap = apy / apy_cap_pct. Useful as a sanity check and as the
    # denominator when converting derived dollar tiers back to cap shares.
    with np.errstate(divide="ignore", invalid="ignore"):
        contracts["implied_cap"] = np.where(
            contracts["apy_cap_pct"] > 0,
            contracts["apy"] / contracts["apy_cap_pct"],
            np.nan)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    contracts.to_parquet(cache_path, index=False)
    logger.info("Fetched %d league contract rows (%d players, %d–%d) → %s",
                len(contracts), contracts["gsis_id"].nunique(),
                int(contracts["year_signed"].min()),
                int(contracts["year_signed"].max()), cache_path)
    return contracts


def derive_vet_min_apy_pct(contracts: pd.DataFrame) -> pd.Series:
    """
    Derive the league minimum-salary line, per year, as a fraction of the cap.

    The OTC feed contains the CBA minimum tiers as sharp modes: one-year
    non-rookie deals spike at exactly the tier values (2025: $0.840M and
    $1.17M; 2024: $0.795M and $1.125M). Extracting the LOWEST full-season tier
    per year reproduces the published minimum-salary ladder — verified monotone
    in dollars across 2013–2026 — without hardcoding a table that cannot be
    checked and silently rots each March.

    Prorated deals (practice-squad elevations, partial-season signings) cluster
    an order of magnitude lower and are excluded by VET_MIN_PRORATION_FLOOR_PCT.

    Years too sparse to derive fall back to VET_MIN_FALLBACK_PCT — the line is
    stable at 0.0027–0.0037 of cap across the derivable era, so extrapolating
    backward by cap share is better motivated than reciting old dollar figures.

    Returns pd.Series indexed by year_signed (int) → minimum APY as cap pct.
    """
    df = contracts.dropna(subset=["year_signed", "apy", "apy_cap_pct"]).copy()
    df = df[df["apy_cap_pct"] > 0]
    if df.empty:
        return pd.Series(dtype=float)

    cap_by_year = df.groupby("year_signed")["implied_cap"].median()

    # Minimum-salary deals are overwhelmingly one-year, non-rookie.
    one_yr = df[(df["year_signed"] != df["draft_year"]) & (df["years"] == 1)]

    derived: dict[int, float] = {}
    for year, grp in one_yr.groupby("year_signed"):
        cap = cap_by_year.get(year, np.nan)
        if not np.isfinite(cap) or cap <= 0:
            continue
        counts = grp["apy"].round(3).value_counts()
        tiers = [v for v, n in counts.items()
                 if n >= VET_MIN_MODE_MIN_COUNT
                 and (v / cap) >= VET_MIN_PRORATION_FLOOR_PCT]
        if not tiers:
            continue
        derived[int(year)] = float(min(tiers) / cap)

    if not derived:
        logger.warning("Could not derive any minimum-salary tier from contracts")
        return pd.Series(dtype=float)

    # Backfill sparse early years with the median derived cap share.
    all_years = range(int(df["year_signed"].min()), int(df["year_signed"].max()) + 1)
    median_pct = float(np.median(list(derived.values())))
    out = {}
    for y in all_years:
        out[y] = derived.get(y, median_pct if np.isfinite(median_pct)
                             else VET_MIN_FALLBACK_PCT)

    n_derived = len(derived)
    logger.info("Minimum-salary line: %d/%d years derived from data "
                "(median %.5f of cap), %d backfilled",
                n_derived, len(out), median_pct, len(out) - n_derived)
    return pd.Series(out, name="vet_min_apy_pct").sort_index()


def vet_min_for_year(vet_min: pd.Series, year: float) -> float:
    """Look up the minimum line for a year, falling back to the nearest
    available year and then to the configured constant. Bridge classes reach
    back to 2004 while OTC coverage starts ~2007, so out-of-range lookups are
    expected, not exceptional."""
    if vet_min is None or len(vet_min) == 0 or not np.isfinite(year):
        return VET_MIN_FALLBACK_PCT
    y = int(year)
    if y in vet_min.index:
        return float(vet_min.loc[y])
    idx = np.asarray(vet_min.index, dtype=float)
    return float(vet_min.iloc[int(np.argmin(np.abs(idx - y)))])


def compute_replacement_levels(contracts: pd.DataFrame) -> pd.DataFrame:
    """
    Build both replacement lines and return them as one tidy frame.

    Rows are one of two `scope` kinds:

      scope="global_year"   — the LABEL-layer line: the league minimum-salary
                              share of cap for that year, times
                              REPLACEMENT_MARGIN. Position-independent by
                              design (see the config.py rationale).

      scope="position"      — the OUTPUT-layer line: per position group, the
                              REPLACEMENT_POS_QUANTILE of multi-year veteran
                              contracts over the last
                              REPLACEMENT_CONTRACT_WINDOW_YEARS. Cohorts thinner
                              than REPLACEMENT_POS_MIN_N fall back to the
                              current-year global line.

    Columns: scope, key, replacement_apy_pct, base_apy_pct, n_contracts, source
    """
    vet_min = derive_vet_min_apy_pct(contracts)

    records: list[dict] = []
    for year, pct in vet_min.items():
        records.append({
            "scope": "global_year",
            "key": str(int(year)),
            "base_apy_pct": float(pct),
            "replacement_apy_pct": float(pct) * REPLACEMENT_MARGIN,
            "n_contracts": int((contracts["year_signed"] == year).sum()),
            "source": "min_salary_mode",
        })

    df = contracts.dropna(subset=["year_signed", "apy_cap_pct"]).copy()
    # apy_cap_pct is quantized to 3 decimals in the OTC feed, which is coarse
    # enough to collapse every position onto the same 0.004 at a low quantile.
    # Recompute a continuous cap share from raw APY against the per-year median
    # implied cap. (Dividing by the row's own implied_cap is circular — it was
    # derived from apy_cap_pct.)
    cap_by_year = df[df["apy_cap_pct"] > 0].groupby("year_signed")["implied_cap"].median()
    df["_pct"] = df["apy"] / df["year_signed"].map(cap_by_year)

    max_year = int(df["year_signed"].max())
    window_start = max_year - REPLACEMENT_CONTRACT_WINDOW_YEARS + 1
    recent = df[(df["year_signed"] >= window_start)
                & (df["year_signed"] != df["draft_year"])
                & (df["years"] >= REPLACEMENT_POS_MIN_YEARS)
                & (df["position_group"].notna())
                & (df["_pct"].notna())]

    global_now = vet_min_for_year(vet_min, max_year) * REPLACEMENT_MARGIN
    for pos, grp in recent.groupby("position_group"):
        n = len(grp)
        if n >= REPLACEMENT_POS_MIN_N:
            level = float(grp["_pct"].quantile(REPLACEMENT_POS_QUANTILE))
            source = "position_quantile"
        else:
            level = float(global_now)
            source = "global_fallback"
        records.append({
            "scope": "position",
            "key": str(pos),
            "base_apy_pct": level,
            "replacement_apy_pct": level,
            "n_contracts": int(n),
            "source": source,
        })

    out = pd.DataFrame(records)
    pos_rows = out[out["scope"] == "position"].sort_values(
        "replacement_apy_pct", ascending=False)
    logger.info("Replacement levels — per-position (%d-%d, %dyr+ deals, q%.2f):",
                window_start, max_year, REPLACEMENT_POS_MIN_YEARS,
                REPLACEMENT_POS_QUANTILE)
    for _, r in pos_rows.iterrows():
        logger.info("    %-5s %.5f of cap  (n=%d, %s)",
                    r["key"], r["replacement_apy_pct"], r["n_contracts"], r["source"])
    return out


def fetch_contract_data(bridge_players: pd.DataFrame,
                         contracts: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Link league contract data to bridge players via gsis_id → pfr_id.

    Returns a DataFrame with columns:
        pfr_id, year_signed, years, apy_cap_pct, apy, guaranteed, value,
        draft_year, draft_round
    One row per contract; a player may have multiple rows. `apy` (dollars) is
    carried so guaranteed money can be expressed as a share of the cap.
    """
    if contracts is None:
        contracts = fetch_league_contracts()
    contracts = contracts.copy()

    carry_cols = ["pfr_id", "year_signed", "years", "apy_cap_pct", "apy",
                  "guaranteed", "value", "draft_year", "draft_round"]

    # Build pfr_id → gsis_id crosswalk
    pfr_to_gsis = _build_pfr_to_gsis_crosswalk()
    bridge = bridge_players.copy()
    bridge["_gsis_id"] = bridge["pfr_id"].map(pfr_to_gsis)

    # Primary match: gsis_id join — use only pfr_id + _gsis_id from bridge to avoid column conflicts
    bridge_slim = bridge[["pfr_id", "_gsis_id"]].dropna(subset=["_gsis_id"])
    matched = bridge_slim.merge(
        contracts.rename(columns={"gsis_id": "_gsis_id"}),
        on="_gsis_id", how="inner",
    )[carry_cols]

    # Fallback: normalized name + draft_year ±1 for unmatched bridge players
    unmatched_pfr = set(bridge["pfr_id"].dropna()) - set(matched["pfr_id"])
    if unmatched_pfr and "player_name" in bridge.columns:
        contracts["_name_norm"] = contracts["player"].apply(_normalize_name)
        bridge_unmatched = bridge[bridge["pfr_id"].isin(unmatched_pfr)].copy()
        bridge_unmatched["_name_norm"] = bridge_unmatched["player_name"].apply(_normalize_name)

        fb_records = []
        for _, bp in bridge_unmatched.iterrows():
            nn = bp["_name_norm"]
            dy = bp.get("draft_year")
            if not nn or pd.isna(dy):
                continue
            cands = contracts[
                (contracts["_name_norm"] == nn) &
                (contracts["draft_year"].between(dy - 1, dy + 1))
            ]
            for _, c in cands.iterrows():
                rec = {"pfr_id": bp["pfr_id"]}
                rec.update({col: c.get(col) for col in carry_cols
                            if col != "pfr_id"})
                fb_records.append(rec)

        if fb_records:
            fallback_df = pd.DataFrame(fb_records)
            matched = pd.concat([matched, fallback_df], ignore_index=True)
            logger.info("Contract fallback name match: %d additional players",
                        len(fallback_df["pfr_id"].unique()))

    logger.info("Contract data fetched: %d contract rows for %d bridge players",
                len(matched), matched["pfr_id"].nunique())
    return matched


def compute_second_contract_label(contracts_df: pd.DataFrame,
                                   replacement_gate: bool = False,
                                   vet_min: Optional[pd.Series] = None,
                                   margin: float = REPLACEMENT_MARGIN,
                                   defect_fixes: Optional[bool] = None,
                                   ) -> pd.DataFrame:
    """
    Identify each bridge player's second contract and return it as a frame.

    Two modes:

      replacement_gate=False (legacy) — a qualifying contract is anything that
        is not a rookie deal and not a 5th-year option. The earliest one wins.

      replacement_gate=True — additionally requires the deal to CLEAR the
        replacement line, to not look like a franchise/transition tag, and to
        fall inside the second-contract window. The earliest CLEARING deal
        wins.

    The gate exists because the legacy definition answers "did this player get
    any second NFL paycheck", not "did this player get paid like a
    difference-maker": the median nonzero label is 0.004 of cap — a veteran
    minimum — and 65% sit below 0.01. It also repairs two concrete defects that
    earliest-qualifying selection causes on its own:

      - Franchise tags pre-empt the real deal. Dak Prescott (round 4, so the
        5th-year-option rule never fires) labels as his 0.158 tag rather than
        his 0.219 extension. Tags are excluded structurally — OTC carries no
        tag flag — as one-year deals that are essentially fully guaranteed at a
        high cap share.

      - UDFA year-2 futures/ERFA deals count as "second contracts". Adam
        Thielen labels as a 0.003 futures deal instead of his 0.029 contract.
        These fall below the replacement line and the gate drops them.

    Returns a DataFrame indexed by pfr_id with columns:
        label_apy_raw          the selected deal's apy_cap_pct (0.0 if none)
        label_apy_over_repl    label_apy_raw - replacement line at signing,
                               floored at 0 (0.0 if no qualifying deal)
        qualifies_role         1.0 if a qualifying deal exists, else 0.0
        max_apy_in_window      diagnostic: the best post-rookie apy_cap_pct
                               inside the window, whether or not it qualified
        year_signed            year of the selected deal (NaN if none)
    """
    df = contracts_df.copy()
    for col in ["year_signed", "years", "apy_cap_pct", "draft_year",
                "draft_round", "guaranteed", "value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rookie_mask = df["year_signed"] == df["draft_year"]
    fifth_yr_mask = (
        (df["draft_round"] == 1) &
        (df["years"] == 1) &
        (df["year_signed"] == df["draft_year"] + 4)
    )
    qualifying = df[~rookie_mask & ~fifth_yr_mask].copy()

    all_pfr = set(contracts_df["pfr_id"].dropna())
    empty = pd.DataFrame({
        "label_apy_raw": 0.0, "label_apy_over_repl": 0.0,
        "qualifies_role": 0.0, "max_apy_in_window": 0.0,
        "year_signed": np.nan,
    }, index=sorted(all_pfr))
    if qualifying.empty:
        logger.warning("No qualifying second contracts found")
        return empty

    # Diagnostic: best post-rookie deal, computed before any gating so it is
    # comparable across modes.
    max_in_window = (qualifying.groupby("pfr_id")["apy_cap_pct"].max()
                     .rename("max_apy_in_window"))

    # `defect_fixes` (tag + window exclusion) and `replacement_gate` (the
    # clear-the-line requirement plus the rebase) are separable. They were
    # bundled in the first A/B, which could not tell whether a mixed result
    # meant the defect fixes were bad or the rebase was. Default keeps them
    # bundled for backward compatibility.
    if defect_fixes is None:
        defect_fixes = replacement_gate

    n_pre = len(qualifying)
    n_tag = n_window = n_below = 0
    keep = pd.Series(True, index=qualifying.index)

    if defect_fixes:
        # Window: a deal signed 7+ years after the draft is a third contract or
        # a late-career signing, not the outcome being predicted.
        in_window = (
            (qualifying["year_signed"] >= qualifying["draft_year"] + 1) &
            (qualifying["year_signed"] <=
             qualifying["draft_year"] + SECOND_CONTRACT_MAX_YEARS_AFTER_DRAFT)
        )
        # draft_year is missing for some rows; don't drop those on the window.
        in_window = in_window | qualifying["draft_year"].isna()
        n_window = int((~in_window).sum())

        # Tag heuristic: one-year, essentially fully guaranteed, high cap share.
        gtd_ratio = qualifying.get("guaranteed")
        val = qualifying.get("value")
        if gtd_ratio is not None and val is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(val > 0, gtd_ratio / val, np.nan)
        else:
            ratio = np.full(len(qualifying), np.nan)
        tag_mask = (
            (qualifying["years"] <= TAG_MAX_YEARS) &
            (pd.Series(ratio, index=qualifying.index) >= TAG_MIN_GUARANTEED_RATIO) &
            (qualifying["apy_cap_pct"] >= TAG_MIN_APY_CAP_PCT)
        ).fillna(False)
        n_tag = int(tag_mask.sum())
        # A tag must not simply delete the player. Being franchised is strong
        # evidence of quality, so zeroing a player whose ONLY post-rookie deal
        # is a tag labels a good player as a washout — the first 3-arm A/B
        # showed the tag exclusion losing on its own for exactly this reason.
        # Tags are therefore DE-PREFERRED, not dropped: they lose to any other
        # qualifying deal in selection, and are kept when nothing else exists.
        has_non_tag = (qualifying[in_window & ~tag_mask]
                       .groupby("pfr_id").size().reindex(
                           qualifying["pfr_id"]).notna().values)
        tag_drop = tag_mask.values & has_non_tag
        n_tag = int(tag_drop.sum())
        keep &= in_window & ~pd.Series(tag_drop, index=qualifying.index)

    if replacement_gate:
        # The deal must clear the minimum-salary line for its signing year,
        # times the margin.
        repl = qualifying["year_signed"].map(
            lambda y: vet_min_for_year(vet_min, y) * margin)
        clears = qualifying["apy_cap_pct"] >= repl
        n_below = int((~clears).sum())
        keep &= clears
    else:
        repl = pd.Series(0.0, index=qualifying.index)

    if defect_fixes or replacement_gate:
        qualifying = qualifying[keep].copy()
        qualifying["_repl"] = repl.reindex(qualifying.index)
        logger.info("Contract gate [%s%s]: %d post-rookie → %d qualifying "
                    "(dropped %d out-of-window, %d tag-like, %d below "
                    "replacement; overlapping)",
                    "defect-fixes" if defect_fixes else "",
                    "+replacement" if replacement_gate else "",
                    n_pre, len(qualifying), n_window, n_tag, n_below)
    else:
        qualifying["_repl"] = 0.0

    if qualifying.empty:
        logger.warning("Contract gate rejected every contract")
        out = empty.copy()
        out["max_apy_in_window"] = max_in_window.reindex(out.index).fillna(0.0)
        return out

    # Stable sort: with several deals in the same year, the default quicksort
    # let file row order pick the "earliest" one, so a contract refresh that
    # only shifted rows changed 983 labels (2026-10-29).
    sel = (qualifying.sort_values("year_signed", kind="mergesort")
           .drop_duplicates("pfr_id", keep="first")
           .set_index("pfr_id"))

    out = pd.DataFrame(index=sorted(all_pfr))
    out["label_apy_raw"] = sel["apy_cap_pct"].reindex(out.index).fillna(0.0)
    over = (sel["apy_cap_pct"] - sel["_repl"]).clip(lower=0.0)
    out["label_apy_over_repl"] = over.reindex(out.index).fillna(0.0)
    # Role fires on a POSITIVE-value qualifying deal, not merely on the
    # existence of one: OTC carries some contracts with apy_cap_pct == 0, and
    # legacy semantics (`contract_label[pfr_id] > 0`) treated those as no
    # contract. Under the replacement gate this is equivalent by construction —
    # anything clearing the line is positive — so the definition is safe in
    # both modes.
    out["qualifies_role"] = (out["label_apy_raw"] > 0).astype(float)
    out["max_apy_in_window"] = max_in_window.reindex(out.index).fillna(0.0)
    out["year_signed"] = sel["year_signed"].reindex(out.index)

    pos = out["label_apy_raw"] > 0
    logger.info("Second contract labels%s: %d with contract (mean raw=%.3f%%, "
                "mean over-replacement=%.3f%%), %d washed out (0.0)",
                " [replacement-gated]" if replacement_gate else "",
                int(pos.sum()),
                out.loc[pos, "label_apy_raw"].mean() * 100 if pos.any() else 0.0,
                out.loc[pos, "label_apy_over_repl"].mean() * 100 if pos.any() else 0.0,
                int((~pos).sum()))
    return out


# ── Bridge player NFL labels ───────────────────────────────────────────────────

def compute_nfl_labels(bridge_players: pd.DataFrame,
                       snap_data: pd.DataFrame,
                       snap_years: int = 4,
                       snap_threshold: float = ROLE_SNAP_PCT_THRESHOLD,
                       threshold_by_group: Optional[dict] = None,
                       contract_label=None,
                       use_over_replacement: bool = False) -> pd.DataFrame:
    """
    Compute role_target and impact_target for bridge players.

    role_target  : binary {0, 1} — 1.0 if the player received a qualifying second
                   NFL contract, else 0.0. What counts as "qualifying" is
                   decided upstream in compute_second_contract_label.
    impact_target: the qualifying contract's APY as a fraction of the cap at
                   signing. With use_over_replacement=True this is the
                   above-replacement remainder instead of the raw share.

    `contract_label` accepts either the DataFrame returned by
    compute_second_contract_label or a bare Series of raw APY shares (the
    legacy shape), so callers that predate the rewrite keep working.

    Players without a pfr_id or without contract data default to 0.0 for both.
    snap_years, snap_threshold, and threshold_by_group are retained for backward
    compatibility but are no longer used for role_target computation.
    """
    if contract_label is None:
        logger.warning("contract_label is None — role_target and impact_target will be 0.0 for all players")

    bridge = bridge_players.copy()

    if "pfr_id" not in bridge.columns:
        bridge["role_target"] = 0.0
        bridge["impact_target"] = 0.0
        return bridge

    # Normalize both accepted shapes into (impact_series, role_series).
    impact_src = role_src = None
    if isinstance(contract_label, pd.DataFrame):
        col = "label_apy_over_repl" if use_over_replacement else "label_apy_raw"
        impact_src = contract_label[col]
        role_src = contract_label["qualifies_role"]
    elif contract_label is not None:
        if use_over_replacement:
            raise ValueError("use_over_replacement requires the DataFrame form "
                             "of contract_label")
        impact_src = contract_label
        role_src = (contract_label > 0).astype(float)

    pfr = bridge["pfr_id"]
    valid = pfr.notna() & (pfr.astype(str) != "")

    if impact_src is None:
        bridge["role_target"] = 0.0
        bridge["impact_target"] = 0.0
        return bridge

    impact = pfr.map(impact_src).where(valid)
    role = pfr.map(role_src).where(valid)
    bridge["impact_target"] = pd.to_numeric(impact, errors="coerce").fillna(0.0)
    bridge["role_target"] = pd.to_numeric(role, errors="coerce").fillna(0.0)
    return bridge


# ── Value target (2026-10-29) ──────────────────────────────────────────────────
#
# A composite NFL-value label: the mean of within-(position group × draft
# class) percentiles of three early-career measures — best post-rookie APY,
# best guaranteed money, and usage in NFL seasons 3-4. See config
# IMPACT_TARGET_SOURCE for the referee evidence behind it.

DEPTH_CHART_CACHE = RAW_DIR / "nfl_depth_charts.parquet"
SEASONAL_ROSTERS_CACHE = RAW_DIR / "nfl_seasonal_rosters.parquet"
VALUE_COMPONENT_COLS = ["best_apy_by_y5", "guar_by_y5", "usage_y34"]

_DEPTH_CHART_COLS = ["gsis_id", "season", "week", "game_type", "formation", "depth_team"]
_ROSTER_COLS = ["season", "player_id", "pfr_id", "player_name", "entry_year"]
_NO_USAGE_GROUPS = {"K", "P"}   # snap share is special-teams noise for them


def _fetch_seasons_cached(seasons, cache_path: Path, refresh: bool,
                          fetch_one, cols: list[str], label: str) -> pd.DataFrame:
    """Per-season nflverse pull, cached to one parquet; only missing seasons
    are fetched. `fetch_one(season)` returns the frame or None to skip."""
    seasons = sorted({int(s) for s in seasons})
    cached = pd.DataFrame(columns=cols)
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
    have = set(pd.to_numeric(cached["season"]).astype(int)) if len(cached) else set()
    missing = [s for s in seasons if s not in have]
    if missing:
        if not _nfl_available:
            raise ImportError(f"nfl_data_py required to fetch {label}.")
        frames = [f for f in (fetch_one(s) for s in missing) if f is not None]
        if frames:
            new = pd.concat(frames, ignore_index=True)[cols]
            cached = pd.concat([cached, new], ignore_index=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached.to_parquet(cache_path, index=False)
            logger.info("Cached %s for %d new season(s) → %s (%d rows)",
                        label, len(frames), cache_path, len(cached))
    cached["season"] = pd.to_numeric(cached["season"]).astype(int)
    return cached[cached["season"].isin(seasons)]


def fetch_depth_charts(seasons, cache_path: Optional[Path] = None,
                       refresh: bool = False) -> pd.DataFrame:
    """Weekly nflverse depth charts. Only the 2001-2024 schema (one row per
    player × week × slot, with `depth_team`) is supported: nflverse moved 2025+
    to daily snapshots with a different schema, and those seasons are skipped
    with a warning. The value target only needs depth charts for seasons before
    snap_counts begins (2016), so nothing is lost."""
    def one(season):
        d = nfl.import_depth_charts([season])
        if "depth_team" not in d.columns:
            logger.warning("Depth charts %d use the post-2024 schema — skipped", season)
            return None
        d = d.copy()
        d["season"] = season
        for c in ["gsis_id", "game_type", "formation", "depth_team"]:
            d[c] = d[c].astype(str)
        d["week"] = pd.to_numeric(d["week"], errors="coerce")
        return d
    return _fetch_seasons_cached(seasons, cache_path or DEPTH_CHART_CACHE, refresh,
                                 one, _DEPTH_CHART_COLS, "depth charts")


def compute_starter_share(depth_charts: pd.DataFrame) -> pd.Series:
    """Share of regular-season weeks a player is listed first-team on offense
    or defense. Indexed by (gsis_id, season); players never listed first-team
    are absent (callers treat absence as 0).

    Validated against snap share on the 2016+ overlap (drafted 2014-21, y3-y4):
    pooled rho 0.846; OL .93, QB .91, LB/DB .87, DL .83, WR .81, TE .81, RB .78.
    """
    d = depth_charts[(depth_charts["game_type"] == "REG")
                     & depth_charts["formation"].isin(["Offense", "Defense"])]
    n_weeks = d.groupby("season")["week"].nunique()
    starts = (d[d["depth_team"].astype(str) == "1"]
              .drop_duplicates(["gsis_id", "season", "week"])
              .groupby(["gsis_id", "season"]).size())
    weeks = starts.index.get_level_values("season").map(n_weeks).to_numpy(dtype=float)
    return (starts / weeks).clip(upper=1.0).rename("starter_share")


def fetch_seasonal_rosters(seasons, cache_path: Optional[Path] = None,
                           refresh: bool = False) -> pd.DataFrame:
    """nflverse seasonal rosters, reduced to the identity columns the gsis
    crosswalk needs."""
    def one(season):
        r = nfl.import_seasonal_rosters([season]).copy()
        r["season"] = season
        for c in ["player_id", "pfr_id", "player_name"]:
            r[c] = r[c].astype("string").astype(object).where(r[c].notna())
        r["entry_year"] = pd.to_numeric(r.get("entry_year"), errors="coerce")
        return r
    return _fetch_seasons_cached(seasons, cache_path or SEASONAL_ROSTERS_CACHE, refresh,
                                 one, _ROSTER_COLS, "seasonal rosters")


def build_bridge_gsis_crosswalk(bridge_players: pd.DataFrame,
                                rosters: pd.DataFrame,
                                base: Optional[dict] = None) -> dict[str, str]:
    """pfr_id → gsis_id for bridge players.

    `import_ids()` alone covers only ~77% of bridge players and collapses
    before 2016 — exactly the era where depth charts stand in for snaps. Gaps
    are filled from the rosters' own gsis ids: first by the rosters' pfr_id,
    then by (normalized name, entry year) — exact, then ±1 — accepting only
    keys that identify a single gsis id. Measured: 98.5% coverage in every era.
    """
    xwalk = dict(base if base is not None else _build_pfr_to_gsis_crosswalk())
    r = rosters.dropna(subset=["player_id"])
    for pfr, gsis in zip(r["pfr_id"], r["player_id"]):
        if isinstance(pfr, str) and pfr and pfr not in xwalk:
            xwalk[pfr] = gsis

    r = r.dropna(subset=["entry_year"]).drop_duplicates("player_id")
    keys = pd.Series(list(zip(r["player_name"].map(_normalize_name),
                              r["entry_year"].astype(int))), index=r.index)
    n_ids = keys.groupby(keys).transform("size")
    by_key = dict(zip(keys[n_ids == 1], r.loc[n_ids == 1, "player_id"]))

    filled = 0
    for _, bp in bridge_players.iterrows():
        pfr, dy = bp.get("pfr_id"), bp.get("draft_year")
        if not isinstance(pfr, str) or not pfr or pfr in xwalk or pd.isna(dy):
            continue
        nm = _normalize_name(bp.get("player_name"))
        for y in (int(dy), int(dy) + 1, int(dy) - 1):
            g = by_key.get((nm, y))
            if g is not None:
                xwalk[pfr] = g
                filled += 1
                break
    logger.info("Bridge gsis crosswalk: %d entries (%d filled by name + entry year)",
                len(xwalk), filled)
    return xwalk


def compute_value_components(bridge_players: pd.DataFrame,
                             contracts_df: pd.DataFrame,
                             snap_data: pd.DataFrame,
                             starter_share: pd.Series,
                             pfr_to_gsis: dict,
                             last_year_offset: int,
                             usage_offsets: tuple[int, ...],
                             snap_first_draft_year: int) -> pd.DataFrame:
    """Raw (pre-percentile) value components per bridge player, indexed by pfr_id.

      best_apy_by_y5  best post-rookie APY % of cap, signed draft_year+1 ..
                      draft_year+last_year_offset. The BEST deal, not the
                      earliest: earliest-deal-wins put the legacy label below
                      the player's own best y5 deal for 46% of players.
      guar_by_y5      best guaranteed money % of cap signed in the same window.
      usage_y34       mean usage over seasons draft_year+usage_offsets; a season
                      with no data counts as 0. Snap share (max of offense /
                      defense pct, scaled from games appeared to the team's
                      regular season, so missed games count as 0) from
                      snap_first_draft_year on, depth-chart starter share before. NaN for K/P, and for pre-snap
                      players with no gsis id.
      usage_source    "snaps" | "depth_chart" | None

    Rookie deals (signed in the draft year) and 5th-year options are excluded,
    matching compute_second_contract_label.
    """
    bp = (bridge_players.dropna(subset=["pfr_id", "draft_year"])
          .drop_duplicates("pfr_id").set_index("pfr_id"))
    draft_year = bp["draft_year"].astype(int)

    c = contracts_df.copy()
    for col in ["year_signed", "years", "apy_cap_pct", "apy", "guaranteed", "draft_round"]:
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce")
    c["_dy"] = c["pfr_id"].map(draft_year)
    c = c.dropna(subset=["_dy", "year_signed"])
    fifth_yr = ((c["draft_round"] == 1) & (c["years"] == 1)
                & (c["year_signed"] == c["_dy"] + 4))
    in_window = ((c["year_signed"] >= c["_dy"] + 1)
                 & (c["year_signed"] <= c["_dy"] + last_year_offset))
    c = c[in_window & ~fifth_yr]
    with np.errstate(divide="ignore", invalid="ignore"):
        # guaranteed / implied cap, where implied cap = apy / apy_cap_pct
        c["_guar_cap"] = np.where(c["apy"] > 0,
                                  c["guaranteed"] * c["apy_cap_pct"] / c["apy"], np.nan)

    out = pd.DataFrame(index=bp.index)
    out["best_apy_by_y5"] = c.groupby("pfr_id")["apy_cap_pct"].max().reindex(out.index).fillna(0.0)
    out["guar_by_y5"] = c.groupby("pfr_id")["_guar_cap"].max().reindex(out.index).fillna(0.0)

    snap_by = pd.Series(dtype=float)
    if not snap_data.empty:
        scaled = season_snap_shares(snap_data)
        s = scaled.assign(_share=scaled[["offense_pct", "defense_pct"]].max(axis=1))
        snap_by = s.groupby(["pfr_player_id", "season"])["_share"].max()
    snap_lookup, start_lookup = snap_by.to_dict(), starter_share.to_dict()

    usage, source = [], []
    for pfr in out.index:
        dy, grp = int(draft_year[pfr]), str(bp.at[pfr, "position_group"])
        seasons = [dy + o for o in usage_offsets]
        if grp in _NO_USAGE_GROUPS:
            usage.append(np.nan); source.append(None)
        elif dy >= snap_first_draft_year:
            usage.append(float(np.mean([snap_lookup.get((pfr, y), 0.0) for y in seasons])))
            source.append("snaps")
        elif pfr_to_gsis.get(pfr):
            g = pfr_to_gsis[pfr]
            usage.append(float(np.mean([start_lookup.get((g, y), 0.0) for y in seasons])))
            source.append("depth_chart")
        else:
            usage.append(np.nan); source.append(None)
    out["usage_y34"] = usage
    out["usage_source"] = source
    return out


def compute_value_target(components: pd.DataFrame,
                         group: pd.Series,
                         draft_year: pd.Series) -> pd.Series:
    """Mean of the components' percentiles within (group × draft_year).

    Exact zeros are pinned to percentile 0 rather than the tie-average rank,
    so a player with no deal and no snaps scores 0 — the same as a non-NFL
    player — and the zero-padded mixture the heads train on stays coherent.
    Missing components (K/P usage, unmapped pre-snap usage) drop out of the
    mean. All three inputs share the components' pfr_id index.
    """
    keys = [group.reindex(components.index), draft_year.reindex(components.index)]
    pcts = []
    for col in VALUE_COMPONENT_COLS:
        v = components[col]
        p = v.groupby(keys).rank(pct=True, method="average")
        pcts.append(p.where(v != 0, 0.0).where(v.notna()))
    return pd.concat(pcts, axis=1).mean(axis=1, skipna=True).rename("value_target")


def build_value_apy_map(best_apy: pd.Series, group: pd.Series,
                        n_grid: int = 201) -> pd.DataFrame:
    """Quantile function of best_apy_by_y5 per group, for step 10's
    percentile → APY % of cap translation. Long format:
    position_group, q, apy_cap_pct."""
    q = np.linspace(0.0, 1.0, n_grid)
    rows = []
    for g, vals in best_apy.groupby(group.reindex(best_apy.index)):
        v = vals.dropna().to_numpy(dtype=float)
        if len(v):
            rows.append(pd.DataFrame({"position_group": g, "q": q,
                                      "apy_cap_pct": np.quantile(v, q)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["position_group", "q", "apy_cap_pct"])


def _value_trajectory_path(variant: str = "") -> Path:
    suffix = f"_{variant}" if variant else ""
    return TRAJECTORY_MODELS_DIR / f"value_trajectory{suffix}.pkl"


def load_value_trajectory_model(variant: str = "") -> tuple:
    """(model, feature_cols) or (None, None) if not fitted yet."""
    path = _value_trajectory_path(variant)
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded value trajectory model (val R²=%.3f)", data.get("val_r2", float("nan")))
    return data["model"], data["feature_cols"]


def fit_value_trajectory_model(train_df: pd.DataFrame, max_years: int = TRAJECTORY_MAX_YEARS,
                               variant: str = "", n_val_classes: int = 2) -> tuple:
    """Fit the regressor that imputes value_target for un-settled classes from
    partial NFL snapshots. Same features and weighting as the impact trajectory
    model (draft position included — the legacy behaviour; dropping it is a
    separate, deferred A/B). Rows with NaN target_value are skipped.

    Validation holds out the last `n_val_classes` draft classes at EVERY
    snapshot length, picks the iteration count by early stopping, then refits
    on all rows. (The impact model's split instead infers draft year from
    n_years_observed, so it validates on all 1-2 season rows and never trains on
    them — exactly the rows it later scores for the newest classes.)
    Returns (model, feature_cols).
    """
    if not _lgbm_available:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")
    df = train_df.dropna(subset=["target_value"])
    feature_cols = [f"y{n}_{s}" for n in range(1, max_years + 1)
                    for s in ("snap_pct", "n_games", "epa")]
    feature_cols += ["n_years_observed", "position_group_ordinal",
                     "draft_round_norm", "draft_pick_log"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    model, val_r2, _, _ = _fit_trajectory_regressor(
        df[feature_cols].to_numpy(dtype=float, na_value=np.nan),
        df["target_value"].to_numpy(dtype=float),
        df["obs_weight"].to_numpy(dtype=float),
        _class_val_mask(df["draft_year"], n_val_classes),
        df["n_years_observed"].to_numpy(),
        {**LGBM_PARAMS, "n_estimators": 300}, "value")

    TRAJECTORY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = _value_trajectory_path(variant)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols, "val_r2": val_r2}, f)
    logger.info("Saved trajectory value model → %s (refit on %d rows)", path, len(df))
    return model, feature_cols
