"""
Resolve external player records (name, school, final college season) to CFBD
player_ids against the FULL college universe (player_seasons ∪ rosters ∪ the
FCS identities of src/utils/fcs_identities.py).

Mirrors the resolver in pipeline/02_bridge_players.py (same normalization,
year-proximity rule and three passes) so a record links whether or not the
player ever reached the NFL. Joining through bridge_players instead links
only drafted / NFL-roster players and leaks the outcome into any feature
built from the join (the combine leak found 2026-10-30).
"""
import re
import unicodedata

import pandas as pd

from config import RAW_DIR, PROCESSED_DIR, BRIDGE_YEAR_TOLERANCE, KNOWN_SEASON_EXCLUSIONS

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?$")
_SCHOOL_SUFFIXES = re.compile(r"\b(state|university|college|tech)\b")


def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return _SUFFIXES.sub("", s).strip()


def _norm_school(s: str) -> str:
    """_norm plus "St." → "state" and parenthetical qualifiers dropped, as in
    src.scrapers.nfl_data._normalize_school: nflverse writes "Arizona St.",
    CFBD "Arizona State"."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\([^)]*\)", "", s.lower())
    s = re.sub(r"(?<=\s)st\.?(?=\s|$)", "state", s)
    return _norm(s)


def _year_distance(min_s: int, max_s: int, target: int) -> int:
    if min_s <= target <= max_s:
        return 0
    return min(abs(min_s - target), abs(max_s - target))


class CollegeIdResolver:
    """Candidate pool built once; `resolve` maps records to player_ids."""

    def __init__(self, logger=None):
        frames = [pd.read_parquet(p, columns=["player_id", "player_name", "school", "season_year"])
                  for p in (RAW_DIR / "player_seasons.parquet", RAW_DIR / "rosters.parquet")
                  if p.exists()]
        # FCS identities (step 01c), under their CFBD school spelling
        from src.utils.fcs_identities import load_fcs_rosters
        frames.append(load_fcs_rosters(["player_id", "player_name", "school", "season_year"]))
        seasons = pd.concat(frames, ignore_index=True).dropna(subset=["player_id", "season_year"])
        from src.utils.cfbd_id_reuse import merged_exclusions, exclusion_mask
        exclusions = merged_exclusions(KNOWN_SEASON_EXCLUSIONS, RAW_DIR, PROCESSED_DIR, logger)
        if exclusions:
            bad = exclusion_mask(seasons, exclusions)
            seasons = seasons[~bad]
        seasons["_n"] = seasons["player_name"].apply(_norm)
        seasons["_s"] = seasons["school"].apply(_norm_school)
        cands = (seasons.drop_duplicates(["player_id", "_s", "season_year", "_n"])
                 .groupby(["_n", "_s", "player_id"])
                 .agg(mn=("season_year", "min"), mx=("season_year", "max")).reset_index())
        self.by_name_school: dict[tuple, list[tuple]] = {}
        self.by_name: dict[str, list[tuple]] = {}
        for n, s, pid, mn, mx in cands.itertuples(index=False, name=None):
            self.by_name_school.setdefault((n, s), []).append((str(pid), int(mn), int(mx)))
            self.by_name.setdefault(n, []).append((str(pid), s, int(mn), int(mx)))

    @staticmethod
    def _closest(cands, target):
        best, best_d = None, None
        for c in cands:
            d = _year_distance(c[-2], c[-1], target)
            if best_d is None or d < best_d:
                best, best_d = c[0], d
        return best if best_d is not None and best_d <= BRIDGE_YEAR_TOLERANCE else None

    def resolve_one(self, name, school, last_season) -> tuple[str | None, str]:
        """(player_id, pass) for one record; pass is 'school', 'school_stripped',
        'name_unique' or 'none'."""
        if pd.isna(last_season):
            return None, "none"
        n, s, y = _norm(name), _norm_school(school), int(last_season)
        pid = self._closest(self.by_name_school.get((n, s), []), y)
        if pid:
            return pid, "school"
        s_stripped = _SCHOOL_SUFFIXES.sub("", s).strip()
        if s_stripped:
            pool = [c for c in self.by_name.get(n, [])
                    if _SCHOOL_SUFFIXES.sub("", c[1]).strip() == s_stripped]
            pid = self._closest([(c[0], c[2], c[3]) for c in pool], y)
            if pid:
                return pid, "school_stripped"
        close = [c for c in self.by_name.get(n, []) if _year_distance(c[-2], c[-1], y) <= BRIDGE_YEAR_TOLERANCE]
        if len({c[0] for c in close}) == 1:
            return close[0][0], "name_unique"
        return None, "none"

    def resolve(self, df: pd.DataFrame, name_col: str, school_col: str,
                year_col: str) -> pd.DataFrame:
        out = [self.resolve_one(n, s, y) for n, s, y in
               df[[name_col, school_col, year_col]].itertuples(index=False, name=None)]
        return pd.DataFrame(out, columns=["player_id", "match_pass"], index=df.index)
