"""
FCS identities: PFF-anchored player ids for FCS players CFBD doesn't carry.

CFBD has no FCS rosters. An FCS player exists in CFBD only if he recorded a
stat against an FBS opponent, so most FCS players (Cole Strange, Trey Lance,
Trevor Penning, Joe Haeg) are absent from the college universe. PFF's FCS
facet files (pipeline/00b, 2014+) cover them. Step 01c gives every PFF player
whose season-grade rows never matched a CFBD id, whose last season is in an
FCS file, a synthetic player_id SYN_BASE + pff_player_id, and writes one
roster-like row per (identity, season) to fcs_rosters.parquet. Steps 02, 03,
06 and the college-id resolver (00d) read it alongside CFBD's rosters, so
the identities flow through the pipeline like CFBD roster-only players:
PFF career features, combine and board joins, NFL labels, scores.

A candidate who only ever played at schools CFBD has a roster for that season
(FBS ones) isn't minted either: PFF's FCS files list FBS players from their
games against FCS opponents, and one of those CFBD didn't match is a name
mismatch far more often than a missing player.

A candidate that is a CFBD player isn't minted: a name variant at the same
school (Josh / Joshua Chase, Morgan State) or a transfer down from FBS (Noah
Spence, Ohio State -> Eastern Kentucky). Its PFF rows and fcs_rosters seasons
go to the CFBD id, so a transfer's final FCS season counts as his last.

Last season >= FCS_START: PFF's FCS coverage before 2014 is 100-400 players
a year, a selected set that would overstate FCS players' NFL rate.

PFF abbreviates FCS school names ("CHATTNOOGA", "N DAK ST"). The display
school is CFBD's spelling, learned by majority vote over PFF rows that did
match a CFBD id, falling back to a unique abbreviation match against CFBD's
school names, then to PFF's own name.

Evaluated in experiments/00ze_fcs_identities.py: the identities don't add
off-board contributors to the top of any class (FCS players reach the NFL at
a third of the CFBD off-board rate), and don't hurt anyone else's ranking.
They're wired in for completeness, so FCS prospects get scored.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from config import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

SYN_BASE = 900_000_000
FCS_START = 2014
FCS_ROSTERS_PATH = PROCESSED_DIR / "fcs_rosters.parquet"
ROSTER_COLS = ["player_id", "player_name", "school", "season_year", "position",
               "pff_player_id", "pff_school"]
# PFF position code -> the CFBD roster code step 03 maps to a group. ED goes
# to DE so step 03's EDGE reconciliation (PFF mode == "ED") moves it to EDGE.
PFF_TO_CFBD_POSITION = {
    "QB": "QB", "HB": "RB", "FB": "FB", "WR": "WR", "TE": "TE",
    "T": "OT", "G": "OG", "C": "C", "DI": "DT", "ED": "DE",
    "LB": "LB", "CB": "CB", "S": "S", "K": "K", "P": "P",
}

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?$")


def _norm(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return _SUFFIXES.sub("", s).strip()


def is_fcs_identity(pid) -> pd.Series:
    """Boolean mask of synthetic FCS ids in a player_id Series."""
    return pd.to_numeric(pd.Series(pid), errors="coerce") >= SYN_BASE


def load_fcs_rosters(columns: list[str] | None = None) -> pd.DataFrame:
    """fcs_rosters.parquet, or an empty frame with the requested columns."""
    cols = columns or ROSTER_COLS
    if not FCS_ROSTERS_PATH.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_parquet(FCS_ROSTERS_PATH, columns=cols)


def abbrev_match(abbr: str, full: str) -> bool:
    """'n dak st' ~ 'north dakota state', 'chattnooga' ~ 'chattanooga': an
    in-order subsequence with the same first letter and at least 40% of the
    full name's length."""
    a, f = _norm(abbr).replace(" ", ""), _norm(full).replace(" ", "")
    if not a or not f or a[0] != f[0] or len(a) < 0.4 * len(f):
        return False
    it = iter(f)
    return all(ch in it for ch in a)


def schools_agree(keys, school, pff_school) -> bool:
    """Whether any normalized school key names this identity's school: its
    CFBD spelling, or an abbreviation either way of it or of PFF's name."""
    names = [n for n in (_norm(school), _norm(pff_school)) if n]
    for k in keys:
        for n in names:
            if k == n or abbrev_match(k, n) or abbrev_match(n, k):
                return True
    return False


def _fcs_raw(pff_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(pff_dir.glob("*/*_fcs.csv")):
        if not f.parent.name.isdigit():
            continue
        d = pd.read_csv(f, usecols=lambda c: c in ("player", "player_id", "team_name"))
        d["season_year"] = int(f.parent.name)
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["player", "pff_player_id", "team_name", "season_year"])
    return (pd.concat(rows, ignore_index=True).dropna(subset=["player_id"])
              .drop_duplicates(["player_id", "season_year"])
              .rename(columns={"player_id": "pff_player_id"}))


def _cfbd_universe(raw_dir: Path) -> pd.DataFrame:
    """(player_id, player_name, school, season_year) over CFBD's whole
    universe, including gamelog-only ids, which is where CFBD's FCS players
    live, minus the id-reuse exclusions."""
    from config import KNOWN_SEASON_EXCLUSIONS
    from src.utils.cfbd_id_reuse import exclusion_mask, merged_exclusions
    cols = ["player_id", "player_name", "school", "season_year"]
    parts = [pd.read_parquet(raw_dir / f, columns=cols)
             for f in ("player_seasons.parquet", "rosters.parquet", "player_gamelogs.parquet")
             if (raw_dir / f).exists()]
    ps = pd.concat(parts, ignore_index=True).dropna(subset=["player_id", "school", "season_year"])
    ps["player_id"] = ps["player_id"].astype(str)
    ps = ps.drop_duplicates(cols)
    exclusions = merged_exclusions(KNOWN_SEASON_EXCLUSIONS, raw_dir, PROCESSED_DIR, logger,
                                   post_draft=False)
    return ps[~exclusion_mask(ps, exclusions)] if exclusions else ps


def _given_names_compatible(a: str, b: str) -> bool:
    """Josh / Joshua, Cam / Cameron, Kiyon / Kyion: equal, a prefix, or the
    same initial and close spelling."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return (len(short) >= 2 and long_.startswith(short)) or (
        a[:1] == b[:1] and fuzz.ratio(a, b) >= 65)


# Position families for the transfer check: offensive skill, line, front
# seven, secondary, specialists, quarterback.
_FAMILY = {"QB": "QB", "RB": "SKILL", "WR": "SKILL", "TE": "SKILL", "OL": "OL",
           "DL": "FRONT", "EDGE": "FRONT", "LB": "FRONT", "DB": "DB", "K": "KP", "P": "KP"}


def _family(position) -> str | None:
    from config import POSITION_TO_GROUP
    if not isinstance(position, str):
        return None
    return _FAMILY.get(POSITION_TO_GROUP.get(position.strip().upper()))


def _cfbd_roster_seasons(raw_dir: Path) -> set[tuple[str, int]]:
    """(normalized school, season) pairs CFBD has a roster for: FBS ones."""
    path = raw_dir / "rosters.parquet"
    if not path.exists():
        return set()
    ros = pd.read_parquet(path, columns=["school", "season_year"]).dropna().drop_duplicates()
    return set(zip(ros["school"].map(_norm), ros["season_year"].astype(int)))


def _cfbd_families(raw_dir: Path) -> dict[str, str]:
    """CFBD player_id -> position family of his most common roster position."""
    path = raw_dir / "rosters.parquet"
    if not path.exists():
        return {}
    ros = pd.read_parquet(path, columns=["player_id", "position"]).dropna()
    mode = ros.groupby(ros["player_id"].astype(str))["position"].agg(lambda s: s.value_counts().index[0])
    return {p: f for p, f in mode.map(_family).items() if f}


def _declared(raw_dir: Path, board_path: Path) -> pd.DataFrame:
    """(nn, sk, year) for every player in CFBD's draft picks or on the board:
    he had left college by that draft year's season."""
    parts = []
    if (raw_dir / "draft_picks.parquet").exists():
        d = pd.read_parquet(raw_dir / "draft_picks.parquet", columns=["player_name", "college_team", "draft_year"])
        parts.append(d.set_axis(["name", "school", "year"], axis=1))
    if board_path.exists():
        h = pd.read_csv(board_path, encoding="utf-8-sig", usecols=["name", "school", "year"])
        parts.append(h)
    if not parts:
        return pd.DataFrame(columns=["nn", "sk", "year"])
    d = pd.concat(parts, ignore_index=True).dropna()
    return pd.DataFrame({"nn": d["name"].map(_norm), "sk": d["school"].map(_norm),
                         "year": d["year"].astype(int)}).drop_duplicates()


def _cfbd_links(rows: pd.DataFrame, cfbd: pd.DataFrame, cfbd_family: dict[str, str],
                declared: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Candidate identity -> (CFBD id, kind) for candidates that are a CFBD
    player. 01c misses them: its fuzzy pass doesn't bridge nicknames, its
    index has no gamelog-only ids, and a transfer's FCS rows are at another
    school.
        variant   same school, seasons overlapping or within 2, and the same
                  surname with a compatible given name (Josh / Joshua Chase,
                  Morgan State) or the same given name with one surname part
                  of the other (Jamon Johnson / Dumas-Johnson, Kentucky)
        transfer  exact full name, CFBD career ending 1-2 seasons before the
                  FCS one starts (Noah Spence, Ohio State -> Eastern Kentucky),
                  the same position family where CFBD's roster has one (a
                  quarter of name-and-timing pairs disagree, mostly common
                  names: Brian Williams, Andrew Jones), and the CFBD player
                  not drafted or on the board by the FCS season (Texas A&M's
                  Tyree Johnson entered the 2022 draft; Arkansas-Pine Bluff's
                  played in 2022)
    A kind commits only a unique CFBD id, variant before transfer, and a CFBD
    id two candidates claim is ambiguous: both stay minted."""
    def keyed(d):
        d = d.assign(nn=d["player_name"].map(_norm), sk=d["school"].map(_norm))
        t = d["nn"].str.split()
        return d.assign(sur=t.str[-1], given=t.str[0])
    a = keyed(rows[["player_id", "player_name", "school", "season_year"]].dropna())
    b = keyed(cfbd)
    a_sk = a.groupby(["player_id", "nn", "sur", "given", "sk"])["season_year"] \
            .agg(first="min", last="max").reset_index()
    b_sk = b.groupby(["player_id", "nn", "sur", "given", "sk"])["season_year"] \
            .agg(mn="min", mx="max").reset_index()
    a_all = a.groupby(["player_id", "nn"])["season_year"].agg(first="min").reset_index()
    b_all = b.groupby(["player_id", "nn"])["season_year"].agg(mn="min", mx="max").reset_index()

    v = a_sk.merge(b_sk, on=["sur", "sk"], suffixes=("", "_c"))
    v = v[pd.Series([_given_names_compatible(x, y) for x, y in zip(v["given"], v["given_c"])],
                    index=v.index, dtype=bool)]
    h = a_sk.merge(b_sk, on=["given", "sk"], suffixes=("", "_c"))
    h = h[[x != y and len(min(x, y, key=len)) >= 4
           and (y.startswith(x) or y.endswith(x) or x.startswith(y) or x.endswith(y))
           for x, y in zip(h["sur"], h["sur_c"])]] if len(h) else h
    v = pd.concat([v, h], ignore_index=True)
    v = v[(v["mn"] <= v["last"] + 2) & (v["first"] <= v["mx"] + 2)]
    t = a_all.merge(b_all, on="nn", suffixes=("", "_c"))
    t = t[(t["mx"] >= t["first"] - 2) & (t["mx"] <= t["first"] - 1)]
    fam = rows.drop_duplicates("player_id").set_index("player_id")["position"].map(_family)
    c_fam = t["player_id_c"].map(cfbd_family)
    t = t[c_fam.isna() | (c_fam == t["player_id"].map(fam))]
    if len(t) and len(declared):
        c_schools = b_sk.groupby("player_id")["sk"].agg(set)
        dec = declared.groupby("nn")[["sk", "year"]].apply(lambda g: list(zip(g["sk"], g["year"])))
        gone = [any(sk in c_schools.get(cid, set()) and y <= first for sk, y in dec.get(nn, []))
                for cid, nn, first in zip(t["player_id_c"], t["nn"], t["first"])]
        t = t[~pd.Series(gone, index=t.index, dtype=bool)]

    links: dict[str, tuple[str, str]] = {}
    for kind, m in (("variant", v), ("transfer", t)):
        n = m.groupby("player_id")["player_id_c"].nunique()
        one = m[m["player_id"].isin(n[n == 1].index) & ~m["player_id"].isin(links)]
        for pid, cid in one.drop_duplicates("player_id")[["player_id", "player_id_c"]].itertuples(index=False):
            links[pid] = (cid, kind)
    claims = pd.Series([c for c, _ in links.values()]).value_counts()
    return {p: (c, k) for p, (c, k) in links.items() if claims[c] == 1}


def _school_display_map(grades: pd.DataFrame, cfbd: pd.DataFrame,
                        raw_names: dict[str, str]) -> dict[str, str]:
    """PFF school_norm -> CFBD school spelling. Majority vote over PFF rows
    matched to a CFBD id (>= 3 votes, >= 60% share), then a unique
    abbreviation match against CFBD's school names, then PFF's name."""
    m = grades.loc[grades["player_id"].notna(), ["player_id", "season_year", "school_norm"]].copy()
    m["player_id"] = m["player_id"].astype(str)
    m = m.merge(cfbd.drop_duplicates(["player_id", "season_year"])
                    [["player_id", "season_year", "school"]], on=["player_id", "season_year"])
    votes = m.groupby(["school_norm", "school"]).size().rename("n").reset_index()
    votes["share"] = votes["n"] / votes.groupby("school_norm")["n"].transform("sum")
    best = votes[(votes["n"] >= 3) & (votes["share"] >= 0.6)]
    out = dict(zip(best["school_norm"], best["school"]))
    taken = set(out.values())
    pool = [s for s in cfbd["school"].unique() if s not in taken]
    for sn, raw in raw_names.items():
        if sn in out:
            continue
        hits = [s for s in pool if abbrev_match(raw, s)]
        out[sn] = hits[0] if len(hits) == 1 else raw
    return out


def build_fcs_identities(grades: pd.DataFrame, pff_dir: Path, raw_dir: Path = RAW_DIR,
                         log: logging.Logger | None = None,
                         board_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign synthetic ids to FCS players with no CFBD match.

    Returns (grades with player_id filled on the candidates' rows,
    match_method "fcs_identity" or "fcs_cfbd_{variant,transfer}", and the
    fcs_rosters frame: one row per candidate-season, under the synthetic or
    linked CFBD id).
    """
    log = log or logger
    fcs = _fcs_raw(pff_dir)
    if fcs.empty:
        log.info("No PFF FCS files under %s — no FCS identities", pff_dir)
        return grades, pd.DataFrame(columns=ROSTER_COLS)

    g = grades.copy()
    fcs_keys = set(zip(fcs["pff_player_id"], fcs["season_year"]))
    g["_is_fcs"] = [k in fcs_keys for k in zip(g["pff_player_id"], g["season_year"])]
    g = g.sort_values(["pff_player_id", "season_year"])
    per = g.groupby("pff_player_id").agg(
        any_matched=("player_id", lambda s: s.notna().any()),
        last=("season_year", "max"),
        last_is_fcs=("_is_fcs", "last"))
    pos = (g.dropna(subset=["pff_position"]).groupby("pff_player_id")["pff_position"]
             .agg(lambda s: s.value_counts().index[0]))
    per["position"] = pos.map(PFF_TO_CFBD_POSITION)
    keep = per[~per["any_matched"] & per["last_is_fcs"] & (per["last"] >= FCS_START)
               & per["position"].notna()]
    keep = keep[keep.index.isin(set(fcs["pff_player_id"]))]

    rows = g[g["pff_player_id"].isin(keep.index)].copy()
    rows["player_id"] = (SYN_BASE + rows["pff_player_id"].astype(int)).astype(str)
    names = fcs.set_index(["pff_player_id", "season_year"])["player"]
    last_name = fcs.sort_values("season_year").drop_duplicates("pff_player_id", keep="last") \
                   .set_index("pff_player_id")["player"]
    rows["player_name"] = [names.get(k, last_name.get(k[0]))
                           for k in zip(rows["pff_player_id"], rows["season_year"])]

    raw_team = fcs.merge(g[["pff_player_id", "season_year", "school_norm"]],
                         on=["pff_player_id", "season_year"])
    raw_names = (raw_team.groupby("school_norm")["team_name"]
                         .agg(lambda s: s.value_counts().index[0]).to_dict())
    cfbd = _cfbd_universe(raw_dir)
    school = _school_display_map(g, cfbd, raw_names)
    rosters = pd.DataFrame({
        "player_id": rows["player_id"].values,
        "player_name": rows["player_name"].values,
        "school": rows["school_norm"].map(school).fillna(rows["school_norm"]).values,
        "season_year": rows["season_year"].astype(int).values,
        "position": rows["pff_player_id"].map(keep["position"]).values,
        "pff_player_id": rows["pff_player_id"].astype(int).values,
        "pff_school": rows["school_norm"].values,
    })

    # Candidates that are a CFBD player take the CFBD id. Their rows stay in
    # fcs_rosters under it, so a transfer's last school and season are the FCS
    # ones; for a same-season variant they duplicate seasons CFBD already has.
    from config import ROOT_DIR
    links = _cfbd_links(rosters, cfbd, _cfbd_families(raw_dir),
                        _declared(raw_dir, board_path or ROOT_DIR / "hasan.csv"))
    to_cfbd = {p: c for p, (c, _) in links.items()}
    rosters["player_id"] = rosters["player_id"].map(to_cfbd).fillna(rosters["player_id"])
    # Under CFBD's spelling of the name: the rows are the CFBD player's latest
    # seasons, so step 03 would otherwise rename him to PFF's variant (Jamon
    # Johnson for Jamon Dumas-Johnson, no "Jr.") and the board join by name
    # would lose him.
    cfbd_name = cfbd.sort_values("season_year").drop_duplicates("player_id", keep="last") \
                    .set_index("player_id")["player_name"]
    linked = rosters["player_id"].isin(to_cfbd.values())
    rosters.loc[linked, "player_name"] = rosters.loc[linked, "player_id"].map(cfbd_name)

    # CFBD carries full FBS rosters, so an unlinked candidate who only ever
    # played at schools CFBD has a roster for that season is a name mismatch
    # far more often than a missing player. Mint only candidates with an FCS
    # season (a school-season CFBD has no roster for); leave the rest unmatched.
    fbs = _cfbd_roster_seasons(raw_dir)
    unl = rosters[~rosters["player_id"].isin(to_cfbd.values()) & is_fcs_identity(rosters["player_id"]).values]
    in_fbs = pd.Series([(_norm(sc), y) in fbs for sc, y in zip(unl["school"], unl["season_year"])],
                       index=unl.index)
    fbs_only = set(in_fbs.groupby(unl["player_id"]).all().loc[lambda x: x].index)
    rosters = rosters[~rosters["player_id"].isin(fbs_only)].reset_index(drop=True)
    minted = {p for p in rosters["player_id"] if p not in to_cfbd.values()}

    out = grades.copy()
    syn = out["pff_player_id"].isin(keep.index) & out["player_id"].isna()
    syn &= (SYN_BASE + out["pff_player_id"].fillna(0).astype(int)).astype(str).isin(minted | set(to_cfbd))
    pid = (SYN_BASE + out.loc[syn, "pff_player_id"].astype(int)).astype(str)
    out.loc[syn, "player_id"] = pid.map(to_cfbd).fillna(pid)
    out.loc[syn, "match_method"] = pid.map({p: f"fcs_cfbd_{k}" for p, (_, k) in links.items()}) \
                                      .fillna("fcs_identity")
    kinds = pd.Series([k for _, k in links.values()]).value_counts().to_dict()
    log.info("FCS candidates that are a CFBD player: %d %s (PFF rows and seasons "
             "attached to the CFBD id); %d left unmatched with only FBS seasons",
             len(links), kinds, len(fbs_only))
    ident = is_fcs_identity(rosters["player_id"]).values
    r = rosters[ident]
    n_named = int((r["school"].map(_norm).str.replace(" ", "") != r["pff_school"]).sum())
    log.info("FCS identities: %d players, %d player-seasons (%d respelled to CFBD's "
             "school name); positions %s", r["player_id"].nunique(), len(r),
             n_named, r.drop_duplicates("player_id")["position"].value_counts().to_dict())
    return out, rosters
