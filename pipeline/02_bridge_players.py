"""
Step 2: Bridge Player Identification

Identifies players who played college football and then played in the NFL.
These are the "bridge" between college and NFL performance — they are the
training set for contextual models and draft models.

Uses CFBD draft_picks.parquet from Step 1 + nfl_data_py (nflverse) to:
  1. Build list of drafted players with college info
  2. Match each to an NFL/PFR player ID via 5-pass matching
     (exact, year-only, year-fuzzy, ID crosswalk, rapidfuzz)
  3. Pull career AV and games played from nfl_data_py draft picks table
  4. Append UDFA players who made NFL rosters (matched via snap data)

Output: data/processed/bridge_players.parquet
"""

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (RAW_DIR, PROCESSED_DIR, POSITION_TO_GROUP, BRIDGE_YEAR_TOLERANCE,
                    BRIDGE_UDFA_LATE_ENTRY_YEARS, BRIDGE_SCHOOL_ALIASES)
from rapidfuzz import fuzz
from src.utils.fcs_identities import load_fcs_rosters, schools_agree
from src.scrapers.nfl_data import (
    load_nfl_draft_picks,
    match_draft_picks,
    load_udfa_players,
    fetch_seasonal_snap_pcts,
    _build_id_crosswalk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SNAP_DATA_START = 2016   # nfl_data_py snap counts available from ~2016
SNAP_DATA_END = 2025


_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?$")

def _norm_base(s: str) -> str:
    """Normalize WITHOUT suffix stripping — 'Robert Griffin III' stays
    distinguishable from 'Robert Griffin' for dup adjudication."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def _norm(s: str) -> str:
    return _SUFFIXES.sub("", _norm_base(s)).strip()


def _school_keys(college_team) -> list[str]:
    """Normalized school keys for a bridge row's college string. nflverse
    lists transfers as one string ('Indiana; Mississippi') and spells some
    schools differently from CFBD (BRIDGE_SCHOOL_ALIASES)."""
    if not isinstance(college_team, str):
        return []
    keys = []
    for part in college_team.split(";"):
        part = part.strip()
        k = _norm(BRIDGE_SCHOOL_ALIASES.get(part, part))
        if k and k not in keys:
            keys.append(k)
    return keys


def _year_distance(min_s: int, max_s: int, target: int) -> int:
    if min_s <= target <= max_s:
        return 0
    return min(abs(min_s - target), abs(max_s - target))


def _merge_cands(cands: list[tuple]) -> list[tuple]:
    """Collapse (pid, min, max) tuples to one per pid with the union range —
    a transfer has one CFBD id but a row under each school."""
    by_pid: dict = {}
    for pid, mn, mx in cands:
        lo, hi = by_pid.get(pid, (mn, mx))
        by_pid[pid] = (min(lo, mn), max(hi, mx))
    return [(pid, mn, mx) for pid, (mn, mx) in by_pid.items()]


def _is_negative_id(pid) -> bool:
    """CFBD's roster endpoint mints negative ids for players without an
    athlete record. Some duplicate a real id's final season(s) (Jeff Okudah:
    -1027003 for 2019 only vs 4241984 for 2017-2019)."""
    try:
        return float(pid) < 0
    except (TypeError, ValueError):
        return False


def _pick_by_year(cands: list[tuple], bridge_year,
                  tolerance: int = BRIDGE_YEAR_TOLERANCE) -> str | None:
    """From (pid, min_season, max_season) tuples, pick the player_id whose
    season range is closest to bridge_year, within tolerance. Ties go to a
    real (non-negative) id, then the longer career, so a negative-id
    final-season fragment never beats the full record it duplicates."""
    if not cands or pd.isna(bridge_year):
        return None
    target = int(bridge_year)
    ranked = sorted(cands, key=lambda c: (_year_distance(c[1], c[2], target),
                                          _is_negative_id(c[0]), -(c[2] - c[1])))
    best = ranked[0]
    if _year_distance(best[1], best[2], target) <= tolerance:
        return best[0]
    return None


def _drop_negative_twins(cands: list[tuple]) -> list[tuple]:
    """(pid, school, min, max) candidates minus negative ids that share a
    school and overlapping/adjacent seasons with a real id in the list."""
    real = [c for c in cands if not _is_negative_id(c[0])]
    return [c for c in cands
            if not _is_negative_id(c[0])
            or not any(r[1] == c[1] and r[2] <= c[3] + 1 and c[2] <= r[3] + 1
                       for r in real)]


_NICKNAMES = {frozenset(p) for p in [
    ("nate", "nathan"), ("nate", "nathaniel"), ("joe", "joseph"), ("jake", "jacob"),
    ("tony", "antonio"), ("tony", "anthony"), ("mo", "maurice"), ("reggie", "reginald"),
    ("bill", "william"), ("bob", "robert"), ("bobby", "robert"), ("jim", "james"),
    ("jimmy", "james"), ("mike", "michael"), ("dick", "richard"), ("ted", "theodore"),
]}


def _first_names_compatible(a: list[str], b: list[str]) -> bool:
    """Whether two names' given names can be the same person. Without this a
    shared common surname carries the fuzzy score (Braxton / Easton Jones,
    Mario / Jarvis Williams)."""
    fa, fb = a[0], b[0]
    if fa == fb or frozenset((fa, fb)) in _NICKNAMES:
        return True
    short, long_ = sorted((fa, fb), key=len)
    if len(short) >= 2 and (long_.startswith(short) or long_.endswith(short)):
        return True          # Chris / Christopher, Dre / Andre
    if fa[0] == fb[0] and fuzz.ratio(fa, fb) >= 65:
        return True          # Jake / Jacob, Yosh / Yosuah, Loc / Lac
    for init, full in ((fa, b), (fb, a)):   # JP / John Paul
        given = full[:-1]                   # initials come from given names only
        if 2 <= len(init) <= len(given) and init == "".join(t[0] for t in given[:len(init)]):
            return True
    return False


def _name_similar(a: str, b: str) -> bool:
    """Nickname / formal-name pairs (Josh / Joshua Holsey, Mo / Maurice
    Hurst): the same two-tier bar as match_draft_picks' fuzzy pass, plus
    compatible given names."""
    ta, tb = a.split(), b.split()
    if not ta or not tb or not _first_names_compatible(ta, tb):
        return False
    score = max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b))
    return score >= 85 or (score >= 75 and ta[-1] == tb[-1])


def _add_position_group(df: pd.DataFrame) -> pd.DataFrame:
    """Derive position_group from position column using POSITION_TO_GROUP mapping."""
    if "position" not in df.columns:
        df["position_group"] = None
        return df
    df["position_group"] = df["position"].map(POSITION_TO_GROUP)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Use minimal data for pipeline testing")
    parser.add_argument("--no-udfa", action="store_true",
                        help="Skip UDFA player loading")
    args, _ = parser.parse_known_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    draft_path = RAW_DIR / "draft_picks.parquet"
    if not draft_path.exists():
        logger.error("draft_picks.parquet not found — run Step 1 first")
        sys.exit(1)

    cfbd_picks = pd.read_parquet(draft_path)
    logger.info("Loaded %d CFBD draft picks", len(cfbd_picks))

    # Deduplicate: one row per player (keep earliest draft)
    cfbd_picks = cfbd_picks.sort_values(["player_name", "draft_year"])
    cfbd_picks = cfbd_picks.drop_duplicates(subset=["player_name", "college_team"])

    # Estimate last college season = draft_year - 1
    cfbd_picks["last_college_season_year"] = cfbd_picks["draft_year"] - 1
    cfbd_picks["is_udfa"] = False

    if args.test:
        cfbd_picks = cfbd_picks.head(50)
        logger.info("[TEST] Using first 50 picks only")

    # Build ID crosswalk (cfbref_id → pfr_player_id) for improved matching
    logger.info("Building ID crosswalk from nfl.import_ids() ...")
    id_crosswalk = _build_id_crosswalk()

    # Load nfl_data_py draft picks (has pfr_player_id + career AV)
    nfl_picks = load_nfl_draft_picks()

    # Match CFBD picks → NFL picks to get pfr_id and career stats
    bridge_drafted = match_draft_picks(cfbd_picks, nfl_picks, id_crosswalk=id_crosswalk)

    matched = bridge_drafted["pfr_id"].notna().sum()
    logger.info("Drafted: %d / %d matched to NFL records", matched, len(bridge_drafted))

    # Log match quality breakdown
    if "match_quality" in bridge_drafted.columns:
        quality_counts = bridge_drafted["match_quality"].value_counts()
        for quality, count in quality_counts.items():
            logger.info("  %s: %d", quality, count)

    # ── UDFA players ────────────────────────────────────────────────────────────
    bridge_udfa = pd.DataFrame()
    if not args.no_udfa:
        logger.info("Loading snap counts for UDFA filtering (%d–%d)...",
                    SNAP_DATA_START, SNAP_DATA_END)
        snap_data = fetch_seasonal_snap_pcts(SNAP_DATA_START, SNAP_DATA_END)

        if not snap_data.empty:
            udfa_raw = load_udfa_players(
                snap_data,
                start_year=SNAP_DATA_START,
                end_year=SNAP_DATA_END,
                min_seasons_with_snaps=1,
            )

            if not udfa_raw.empty:
                # CFBD player_id is resolved below with the drafted rows
                # (year-aware, player_seasons ∪ rosters). The old
                # match_udfa_to_cfbd() ids never survived the column
                # alignment, so it is no longer called.
                # Estimate last college season from entry year
                udfa_raw["last_college_season_year"] = udfa_raw["draft_year"] - 1

                bridge_udfa = udfa_raw
                logger.info("UDFA bridge players: %d", len(bridge_udfa))
        else:
            logger.warning("No snap data available — skipping UDFA players")

    # ── Combine drafted + UDFA ───────────────────────────────────────────────────
    all_frames = [bridge_drafted]
    if not bridge_udfa.empty:
        # Align columns (UDFA may be missing some CFBD-specific fields)
        for col in bridge_drafted.columns:
            if col not in bridge_udfa.columns:
                bridge_udfa[col] = None
        bridge_udfa = bridge_udfa[bridge_drafted.columns]
        all_frames.append(bridge_udfa)

    bridge_players = pd.concat(all_frames, ignore_index=True)
    logger.info("Total bridge players: %d drafted + %d UDFA = %d",
                len(bridge_drafted), len(bridge_udfa), len(bridge_players))

    # ── Resolve CFBD player_id for drafted players ───────────────────────────────
    # Candidate-id source = player_seasons ∪ rosters. CFBD's player_seasons
    # endpoint only emits players with countable individual stats, which
    # excludes OL (and a handful of other roster-only players) entirely. Those
    # players DO have a CFBD player_id in rosters; falling back here lets us
    # resolve them so they propagate through to the snapshot and projections.
    seasons_path = RAW_DIR / "player_seasons.parquet"
    rosters_path = RAW_DIR / "rosters.parquet"
    seasons_frames = []
    if seasons_path.exists():
        seasons_frames.append(pd.read_parquet(
            seasons_path,
            columns=["player_id", "player_name", "school", "season_year"],
        ))
    if rosters_path.exists():
        seasons_frames.append(pd.read_parquet(
            rosters_path,
            columns=["player_id", "player_name", "school", "season_year"],
        ))
    if seasons_frames:
        seasons = pd.concat(seasons_frames, ignore_index=True)
        seasons = seasons.dropna(subset=["player_id", "season_year"])
        # Drop CFBD-id-reuse (player_id, season_year) rows so the candidate
        # season range for a bridge player doesn't extend into a namesake's
        # tenure. Without this, e.g. player_id 4039307's range would be
        # 2016-2025 instead of 2016-2020, biasing year-proximity resolution.
        from config import KNOWN_SEASON_EXCLUSIONS as _KSE, RAW_DIR as _RD, PROCESSED_DIR as _PD
        from src.utils.cfbd_id_reuse import merged_exclusions as _merged_exclusions, exclusion_mask
        _exclusions = _merged_exclusions(_KSE, _RD, _PD, logger, post_draft=False)
        if _exclusions:
            bad_mask = exclusion_mask(seasons, _exclusions)
            if bad_mask.any():
                logger.info("Dropped %d (player_id, season_year) rows from "
                            "bridge candidate pool matching CFBD id-reuse exclusions",
                            int(bad_mask.sum()))
                seasons = seasons[~bad_mask].reset_index(drop=True)
        seasons["_norm_name"]   = seasons["player_name"].apply(_norm)
        seasons["_norm_school"] = seasons["school"].apply(_norm)
        # Dedup on the normalized fields so that different display-name
        # spellings of the same player across sources (e.g. "KC Concepcion"
        # in rosters vs. "Kevin Concepcion" in player_seasons — same
        # player_id 4870653) both remain indexed. Without _norm_name in the
        # key, the player_seasons row wins (first concat) and the rosters
        # spelling is silently dropped, leaving bridge rows with the rosters
        # spelling unmatchable.
        seasons = seasons.drop_duplicates(
            ["player_id", "_norm_school", "season_year", "_norm_name"]
        )

        # Build per-player season range (one row per player_id+name+school).
        # Cross-generational namesakes (e.g. 2003 OSU Will Smith vs. 2024 OSU
        # Will Smith Jr.) share normalized name+school but have distinct
        # player_ids; keeping all candidates lets the resolver pick the one
        # whose season range matches the draft pick's last_college_season_year.
        candidates = (
            seasons.groupby(["_norm_name", "_norm_school", "player_id"])
                   .agg(min_season=("season_year", "min"),
                        max_season=("season_year", "max"))
                   .reset_index()
        )

        # name_school_map: (norm_name, norm_school) -> list[(pid, min, max)]
        # name_only_map:   norm_name -> list[(pid, norm_school, min, max)]
        name_school_map: dict[tuple, list[tuple]] = {}
        name_only_map: dict[str, list[tuple]] = {}
        cand_cols = ["_norm_name", "_norm_school", "player_id", "min_season", "max_season"]
        for nn, ns, pid, mn, mx in candidates[cand_cols].itertuples(index=False, name=None):
            name_school_map.setdefault((nn, ns), []).append((pid, int(mn), int(mx)))
            name_only_map.setdefault(nn, []).append((pid, ns, int(mn), int(mx)))

        # school_map: norm_school -> list[(norm_name, pid, min, max)], for the
        # fuzzy-name pass.
        school_map: dict[str, list[tuple]] = {}
        for nn, ns, pid, mn, mx in candidates[cand_cols].itertuples(index=False, name=None):
            school_map.setdefault(ns, []).append((nn, pid, int(mn), int(mx)))

        # CFBD's draft endpoint carries the college athlete id, which is the
        # player_seasons / rosters player_id. When it is in our tables it is
        # authoritative: name matching picked a negative-id final-season
        # fragment for players like Jeff Okudah and D.J. Moore. Old-era ids
        # (2004-08) live in a different id space and fall through to names.
        pid_by_str = {}
        for p in seasons["player_id"].unique():
            try:
                pid_by_str[str(int(float(p)))] = p
            except (TypeError, ValueError):
                pass
        if "college_athlete_id" not in bridge_players.columns:
            bridge_players["college_athlete_id"] = None
        schools_by_pid = seasons.groupby("player_id")["_norm_school"].agg(set).to_dict()

        def _athlete_pid(aid, college_team):
            """The athlete id's player_id, if its record includes the draft
            school. Some transfers have one CFBD record per school and the id
            points at the earlier one (D.J. James: Oregon record, drafted
            from Auburn); those fall through to name matching."""
            try:
                pid = pid_by_str.get(str(int(float(aid))))
            except (TypeError, ValueError):
                return None
            if pid is None or not schools_by_pid.get(pid, set()) & set(_school_keys(college_team)):
                return None
            return pid

        # Gamelog-only ids: CFBD has no FCS rosters, so an FCS player who
        # recorded a stat against an FBS opponent exists only in the box
        # scores (Andre Roberts, Buster Skrine, Derek Rivers). He is in the
        # universe through step 06's gamelog snapshot but was never linked,
        # so he trained as a player who missed the NFL. Used by the last
        # pass only, so no earlier match can change.
        gl_path = RAW_DIR / "player_gamelogs.parquet"
        gl_seasons = pd.DataFrame(columns=seasons.columns)
        gamelog_map: dict[tuple, list[tuple]] = {}
        if gl_path.exists():
            gl = (pd.read_parquet(gl_path, columns=["player_id", "player_name", "school", "season_year"])
                    .dropna(subset=["player_id", "season_year"]).drop_duplicates())
            known = set(seasons["player_id"].astype(str))
            gl = gl[~gl["player_id"].astype(str).isin(known)]
            if _exclusions:
                gl = gl[~exclusion_mask(gl, _exclusions)]
            gl["_norm_name"] = gl["player_name"].apply(_norm)
            gl["_norm_school"] = gl["school"].apply(_norm)
            gl_seasons = gl
            for (nn, ns, pid), g in gl.groupby(["_norm_name", "_norm_school", "player_id"]):
                gamelog_map.setdefault((nn, ns), []).append(
                    (pid, int(g["season_year"].min()), int(g["season_year"].max())))

        # FCS identities (step 01c): PFF-anchored ids for FCS players CFBD
        # doesn't carry at all. Used by the last passes only, so no earlier
        # match can change. PFF abbreviates FCS schools, so agreement accepts
        # an abbreviation either way (src.utils.fcs_identities.schools_agree).
        fcs = load_fcs_rosters()
        fcs_by_surname: dict[str, list[tuple]] = {}
        fcs_last = (fcs.groupby("player_id")["season_year"].max().astype(int).to_dict()
                    if len(fcs) else {})
        if len(fcs):
            fcs["_norm_name"] = fcs["player_name"].apply(_norm)
            fcs["_norm_school"] = fcs["school"].apply(_norm)
            for (nn, pid), g in fcs.groupby(["_norm_name", "player_id"]):
                if nn:
                    fcs_by_surname.setdefault(nn.split()[-1], []).append(
                        (pid, nn, set(g["school"]) | set(g["pff_school"]),
                         int(g["season_year"].min()), int(g["season_year"].max())))

        needs_resolve = (bridge_players["player_id"].isna()
                         if "player_id" in bridge_players.columns
                         else pd.Series(True, index=bridge_players.index))

        bridge_players["_norm_name"]   = bridge_players["player_name"].apply(_norm)
        bridge_players["_norm_school"] = bridge_players["college_team"].apply(_norm)

        counts = {k: 0 for k in ["athlete_id", "name_school", "school_fuzzy", "name_only",
                                 "fuzzy_name", "udfa_late_entry", "gamelog_only",
                                 "fcs_identity", "fcs_identity_fuzzy"]}
        for idx in bridge_players[needs_resolve].index:
            nn = bridge_players.at[idx, "_norm_name"]
            keys = _school_keys(bridge_players.at[idx, "college_team"])
            bridge_year = bridge_players.at[idx, "last_college_season_year"]
            is_udfa = bool(bridge_players.at[idx, "is_udfa"])
            draft_year = bridge_players.at[idx, "draft_year"]

            def _possible(first_season):
                """A candidate whose CFBD career starts in or after the draft
                year can't be him: the college season follows the April
                draft (Antonio Johnson, 2007 -> a 2007-only Anthony Johnson)."""
                return pd.isna(draft_year) or first_season < int(draft_year)

            def _commit(pid, how):
                bridge_players.at[idx, "player_id"] = pid
                counts[how] += 1

            # Pass 0: CFBD college athlete id (drafted players only)
            pid = _athlete_pid(bridge_players.at[idx, "college_athlete_id"],
                               bridge_players.at[idx, "college_team"])
            if pid is not None:
                _commit(pid, "athlete_id")
                continue

            # Pass 1: normalized name + any of the row's schools, year-proximity
            school_cands = [c for c in _merge_cands(
                [c for k in keys for c in name_school_map.get((nn, k), [])]) if _possible(c[1])]
            pid = _pick_by_year(school_cands, bridge_year)
            if pid:
                _commit(pid, "name_school")
                continue

            # Pass 2: normalized name + school without "state"/"university" suffixes
            stripped = {re.sub(r"\b(state|university|college|tech)\b", "", k).strip()
                        for k in keys} - {""}
            pass2_cands: list[tuple] = []
            for key, cands in name_school_map.items():
                if key[0] != nn:
                    continue
                ks_stripped = re.sub(r"\b(state|university|college|tech)\b", "", key[1]).strip()
                if ks_stripped in stripped:
                    pass2_cands.extend(cands)
            pid = _pick_by_year([c for c in _merge_cands(pass2_cands) if _possible(c[1])],
                                bridge_year)
            if pid:
                _commit(pid, "school_fuzzy")
                continue

            if pd.isna(bridge_year):
                continue
            target = int(bridge_year)

            # Pass 3: normalized name only — only commit if exactly one
            # candidate is within tolerance (after year-proximity filtering).
            # Without this guard, namesakes at different schools collapse onto
            # one player_id (the BC vs. Cincinnati Zion Johnson case). A
            # negative-id fragment of a real id doesn't count as a second
            # candidate. A unique candidate with a season in or after the
            # draft year is a namesake, not him (the college season follows
            # the April draft): Steven Jackson -> Clemson 2004-05, Cam Sutton
            # -> Fresno State 2016-19. It still counts toward uniqueness:
            # dropping it would make the next namesake "unique" (Jason Smith
            # -> Colorado State). The school-agreeing passes above don't need
            # this: there a late season is a stale roster row of the right
            # player, excluded downstream. A linked FCS transfer's seasons
            # count (Michael Joseph: Memphis 2017, then Hampton 2018-19).
            close = _drop_negative_twins([
                c for c in name_only_map.get(nn, [])
                if _year_distance(c[-2], c[-1], target) <= BRIDGE_YEAR_TOLERANCE
            ])
            last = max(close[0][-1], fcs_last.get(close[0][0], close[0][-1])) if close else None
            if (len({c[0] for c in close}) == 1
                    and not (pd.notna(draft_year) and last >= int(draft_year))):
                _commit(close[0][0], "name_only")
                continue

            # Pass 4: nickname / formal-name variant at one of the row's
            # schools, within tolerance; commit only a unique pid.
            fuzzy = _drop_negative_twins([
                (pid, k, mn, mx) for k in keys for cn, pid, mn, mx in school_map.get(k, [])
                if _year_distance(mn, mx, target) <= BRIDGE_YEAR_TOLERANCE
                and _possible(mn) and _name_similar(nn, cn)
            ])
            if nn and len({c[0] for c in fuzzy}) == 1:
                _commit(fuzzy[0][0], "fuzzy_name")
                continue

            # Pass 5: UDFA who entered the NFL years after his last college
            # season. Exact name + school, career ended before entry, unique.
            if is_udfa:
                late = [c for c in school_cands
                        if target - BRIDGE_UDFA_LATE_ENTRY_YEARS <= c[2] < target]
                if len(late) == 1:
                    _commit(late[0][0], "udfa_late_entry")
                    bridge_players.at[idx, "last_college_season_year"] = late[0][2]
                    continue

            # Pass 6: gamelog-only id, exact name + school, within tolerance, unique
            gl_close = {c[0] for k in keys for c in gamelog_map.get((nn, k), [])
                        if _year_distance(c[1], c[2], target) <= BRIDGE_YEAR_TOLERANCE
                        and _possible(c[1])}
            if len(gl_close) == 1:
                _commit(gl_close.pop(), "gamelog_only")
                continue

            # Pass 7: FCS identity at one of the row's schools, within
            # tolerance: exact name, else a nickname / formal-name variant
            # sharing the surname; commit only a unique pid.
            if not nn:
                continue
            fcs_close = [c for c in fcs_by_surname.get(nn.split()[-1], [])
                         if _year_distance(c[3], c[4], target) <= BRIDGE_YEAR_TOLERANCE
                         and _possible(c[3]) and any(schools_agree(keys, sc, sc) for sc in c[2])]
            exact = {c[0] for c in fcs_close if c[1] == nn}
            similar = {c[0] for c in fcs_close if _name_similar(nn, c[1])}
            if len(exact) == 1:
                _commit(exact.pop(), "fcs_identity")
            elif not exact and len(similar) == 1:
                _commit(similar.pop(), "fcs_identity_fuzzy")

        # ── Enforce one bridge row per resolved player_id ────────────────
        # Two NFL careers must never share one college identity: a duplicated
        # pid gives one college career contradictory training labels (the
        # Texas-LB vs. Washington-DB Derrick Johnson case, both → pid 120215).
        # Rank claimants by (athlete id is this pid, school matches pid's
        # schools, exact suffix-preserving name match, year distance, drafted
        # over UDFA);
        # null the losers, or all rows on a full tie (better unmatched than
        # mislabeled).
        dup_counts = bridge_players["player_id"].value_counts()
        dup_pids = dup_counts[dup_counts > 1].index
        if len(dup_pids):
            sub = pd.concat([seasons, gl_seasons, fcs.reindex(columns=seasons.columns)],
                            ignore_index=True)
            sub = sub[sub["player_id"].isin(dup_pids)].copy()
            sub["_base_name"] = sub["player_name"].apply(_norm_base)
            grp = sub.groupby("player_id")
            pid_schools = grp["_norm_school"].agg(set).to_dict()
            pid_names = grp["_base_name"].agg(set).to_dict()
            pid_min = grp["season_year"].min().to_dict()
            pid_max = grp["season_year"].max().to_dict()

            nulled = 0
            for pid in dup_pids:
                idxs = list(bridge_players.index[bridge_players["player_id"] == pid])
                schools = pid_schools.get(pid, set())
                names = pid_names.get(pid, set())
                mn, mx = pid_min.get(pid), pid_max.get(pid)

                def _claim_key(i):
                    row = bridge_players.loc[i]
                    school_ok = bool(set(_school_keys(row["college_team"])) & schools)
                    name_ok = _norm_base(row["player_name"]) in names
                    by = row.get("last_college_season_year")
                    if pd.notna(by) and mn is not None:
                        dist = _year_distance(int(mn), int(mx), int(by))
                    else:
                        dist = 999
                    drafted = not bool(row.get("is_udfa", False))
                    id_ok = _athlete_pid(row.get("college_athlete_id"),
                                         row["college_team"]) == pid
                    return (-int(id_ok), -int(school_ok), -int(name_ok), dist,
                            -int(drafted))

                ranked = sorted(idxs, key=_claim_key)
                if _claim_key(ranked[0]) == _claim_key(ranked[1]):
                    losers = ranked  # full tie — abstain on every claimant
                else:
                    losers = ranked[1:]
                for i in losers:
                    logger.info(
                        "  dup pid %s: unmatching %r (%s, %s)%s", pid,
                        bridge_players.at[i, "player_name"],
                        bridge_players.at[i, "college_team"],
                        bridge_players.at[i, "draft_year"],
                        " [full tie]" if len(losers) == len(idxs) else "")
                    bridge_players.at[i, "player_id"] = None
                nulled += len(losers)
            logger.info("Duplicate player_id resolution: %d pids contested, "
                        "%d bridge rows unmatched", len(dup_pids), nulled)

        bridge_players = bridge_players.drop(columns=["_norm_name", "_norm_school"],
                                              errors="ignore")
        resolved = bridge_players["player_id"].notna().sum()
        logger.info("Resolved CFBD player_id for %d / %d bridge players (%s)",
                    resolved, len(bridge_players),
                    ", ".join(f"{k}={v}" for k, v in counts.items()))

    # Derive position_group
    bridge_players = _add_position_group(bridge_players)

    # Invariant: at most one bridge row (one NFL career) per CFBD player_id.
    resolved_ids = bridge_players["player_id"].dropna()
    if not resolved_ids.is_unique:
        dupes = resolved_ids[resolved_ids.duplicated()].unique()
        raise AssertionError(
            f"bridge_players has duplicate player_ids after resolution: {dupes[:10]}")

    out_path = PROCESSED_DIR / "bridge_players.parquet"
    bridge_players.to_parquet(out_path, index=False)
    logger.info("Saved %d bridge players → %s", len(bridge_players), out_path)

    # Summary
    with_av = bridge_players[bridge_players["career_w_av"] > 0]
    with_pfr = bridge_players[bridge_players["pfr_id"].notna()]
    udfa_count = bridge_players.get("is_udfa", pd.Series(False)).sum() if "is_udfa" in bridge_players.columns else 0
    logger.info("Bridge players with pfr_id: %d / %d", len(with_pfr), len(bridge_players))
    logger.info("Bridge players with w_av > 0: %d / %d", len(with_av), len(bridge_players))
    logger.info("UDFA bridge players: %d", udfa_count)
    logger.info("Step 2 complete.")


if __name__ == "__main__":
    main()
