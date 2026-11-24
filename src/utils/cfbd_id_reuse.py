"""
Auto-detect CFBD player_id reuse — same CFBD player_id assigned to two
different humans sharing a name (e.g. Houston WR Marquez Stevenson
4039307 reused for an unrelated Texas Tech 2025 WR Marquez Stevenson).

Signature: a player has a combine record from year C, but rosters or
player_seasons still show them at year ≥ C + GAP_THRESHOLD. That's
physically impossible for one person — the post-combine rows belong to
a namesake.

The auto-detected set complements hand-curated `config.KNOWN_SEASON_EXCLUSIONS`,
which is still needed for namesake collisions that don't involve a
combine record (where this signature can't see them).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# season_year ≥ combine_draft_year + GAP_THRESHOLD → flagged. 2 is conservative:
# a player who declared, withdrew, and played one more year would have gap=0 or 1
# (still in college shortly after a combine year), which we let through.
GAP_THRESHOLD = 2


def detect_cfbd_id_reuse(
    raw_dir: Path,
    processed_dir: Path,
    gap_threshold: int = GAP_THRESHOLD,
) -> dict[str, list[int]]:
    """Return {player_id_str → sorted list of bad season_years} for player_ids
    whose roster / player_seasons activity post-dates their combine record by
    at least `gap_threshold` years.

    Returns empty dict if any of combine_metrics / rosters / player_seasons
    is unavailable — auto-detection silently no-ops on fresh runs before
    those caches exist, leaving the hand-curated registry as the only filter.
    """
    combine_path = processed_dir / "combine_metrics.parquet"
    if not combine_path.exists():
        return {}
    cm = pd.read_parquet(combine_path, columns=["player_id", "combine_draft_year"])
    cm = cm.dropna(subset=["player_id", "combine_draft_year"])
    if cm.empty:
        return {}
    cm["player_id"] = cm["player_id"].astype(str)
    cm = cm.drop_duplicates("player_id")

    frames = []
    for path, cols in [
        (raw_dir / "rosters.parquet",        ["player_id", "season_year"]),
        (raw_dir / "player_seasons.parquet", ["player_id", "season_year"]),
    ]:
        if path.exists():
            frames.append(pd.read_parquet(path, columns=cols).dropna(subset=cols))
    if not frames:
        return {}
    seasons = pd.concat(frames, ignore_index=True)
    seasons["player_id"] = seasons["player_id"].astype(str)
    seasons = seasons.drop_duplicates(["player_id", "season_year"])

    joined = seasons.merge(cm, on="player_id", how="inner")
    bad = joined[joined["season_year"] >= joined["combine_draft_year"] + gap_threshold]
    if bad.empty:
        return {}
    return (bad.groupby("player_id")["season_year"]
              .agg(lambda x: sorted(int(v) for v in x.unique()))
              .to_dict())


def merged_exclusions(
    manual: dict,
    raw_dir: Path,
    processed_dir: Path,
    log: logging.Logger | None = None,
    post_draft: bool = True,
) -> dict[str, list[int]]:
    """Union hand-curated `KNOWN_SEASON_EXCLUSIONS` with auto-detected ones
    (combine-gap id reuse, negative-id twin fragments, and bridge players'
    seasons from their draft year on).

    post_draft=False skips the last: it reads bridge_players.parquet, so
    step 02, which writes it, and step 01c, which runs before it, would
    read their own previous output.

    Keys are normalized to str so callers can do `df["player_id"].astype(str)
    == pid` uniformly. Logs auto-detected entries that aren't already covered
    by the manual list, so each retune surfaces new contamination cases.
    """
    auto = detect_cfbd_id_reuse(raw_dir, processed_dir)
    twins = detect_negative_twins(raw_dir)
    pos_twins = detect_positive_twins(raw_dir)
    out: dict[str, list[int]] = {str(k): sorted(set(int(y) for y in v))
                                  for k, v in manual.items()}
    new_or_extended: dict[str, list[int]] = {}
    for pid, years in auto.items():
        key = str(pid)
        existing = set(out.get(key, []))
        added = sorted(set(years) - existing)
        if added:
            new_or_extended[key] = added
            out[key] = sorted(existing | set(years))
    for pid, years in twins.items():
        out[pid] = sorted(set(out.get(pid, [])) | set(years))
    for pid, years in pos_twins.items():
        out[pid] = sorted(set(out.get(pid, [])) | set(years))
    post = detect_post_draft_seasons(raw_dir, processed_dir) if post_draft else {}
    for pid, years in post.items():
        out[pid] = sorted(set(out.get(pid, [])) | set(years))
    if log and post:
        log.info("Excluding seasons at or after the draft year for %d bridge "
                 "players (stale rosters, transfers' old schools, id reuse)", len(post))
    if log and twins:
        log.info("Excluding %d CFBD negative-id roster fragments that duplicate "
                 "a real id (same name, school, adjacent seasons)", len(twins))
    if log and pos_twins:
        log.info("Excluding %d positive-id CFBD fragments that duplicate another "
                 "id (same name, school, adjacent seasons)", len(pos_twins))
    if log and new_or_extended:
        sample = list(new_or_extended.items())[:10]
        log.info("Auto-detected CFBD player_id reuse for %d player_id(s) "
                 "via combine-year gap (≥%d). First %d: %s",
                 len(new_or_extended), GAP_THRESHOLD, len(sample), sample)
    return out


_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?$")


def _norm(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return _SUFFIXES.sub("", re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()).strip()


def detect_negative_twins(raw_dir: Path) -> dict[str, list[int]]:
    """Return {negative player_id → all its season_years} for CFBD roster-only
    (negative) ids that duplicate a real id: same normalized name and school,
    with overlapping or adjacent seasons. Jeff Okudah has -1027003 for 2019
    only beside 4241984 for 2017-2019. Left in, the fragment is a second,
    one-season copy of the player that never reached the NFL."""
    frames = [pd.read_parquet(p, columns=["player_id", "player_name", "school", "season_year"])
              for p in (raw_dir / "rosters.parquet", raw_dir / "player_seasons.parquet")
              if p.exists()]
    if not frames:
        return {}
    s = pd.concat(frames, ignore_index=True).dropna(subset=["player_id", "season_year"])
    s["player_id"] = s["player_id"].astype(str)
    s["neg"] = s["player_id"].str.startswith("-")
    if not s["neg"].any():
        return {}
    s["_n"] = s["player_name"].map(_norm)
    s["_s"] = s["school"].map(_norm)
    rng = (s[s["_n"] != ""].groupby(["_n", "_s", "player_id", "neg"])["season_year"]
             .agg(["min", "max"]).reset_index())
    pairs = rng[rng["neg"]].merge(rng[~rng["neg"]], on=["_n", "_s"], suffixes=("", "_r"))
    pairs = pairs[(pairs["min_r"] <= pairs["max"] + 1) & (pairs["min"] <= pairs["max_r"] + 1)]
    twins = set(pairs["player_id"])
    seasons = s[s["player_id"].isin(twins)]
    return (seasons.groupby("player_id")["season_year"]
                   .agg(lambda x: sorted(int(v) for v in x.unique()))
                   .to_dict())


_SIDE = {**{p: "off" for p in ("QB", "RB", "FB", "HB", "WR", "TE", "OL", "OT", "OG", "OC",
                                  "C", "G", "T", "IOL")},
         **{p: "def" for p in ("DL", "DT", "DE", "NT", "EDGE", "LB", "ILB", "OLB", "MLB",
                                  "DB", "CB", "S", "FS", "SS")},
         **{p: "st" for p in ("K", "PK", "P", "LS")}}


def detect_positive_twins(raw_dir: Path) -> dict[str, list[int]]:
    """Return {fragment player_id → its season_years} for POSITIVE CFBD ids
    that are a stat-less copy of another positive id.

    A fragment has NO player_seasons stats — only roster rows and/or stray
    box-score rows — and shares its normalized name and school with a
    stat-bearing id whose seasons (roster ∪ stats ∪ box score) contain every
    one of the fragment's seasons, not on the opposite side of the ball.
    Compared pairwise, never chained, so common names can't link different
    people, and no season stats are ever discarded.

    The negative-id rule above misses these. Robert Spears-Jennings is
    4820584 (Oklahoma 2022-25, stats every season) and also 5244828 ("Robert
    Spearsjennings", one 2024 roster row, no position). The board rank went to
    the fragment, so the real record trained and scored as off-board, and the
    fragment as a board player who never reached the NFL (Vernon Hargreaves,
    Bucky Irving, Felix Anudike-Uzomah: 22 board ranks in 2014-26). Zay Jones'
    NFL link sat on a fragment the same way."""
    frames = []
    for name, src in (("rosters", "roster"), ("player_seasons", "stats"),
                      ("player_gamelogs_raw", "box")):
        p = raw_dir / f"{name}.parquet"
        if not p.exists():
            continue
        cols = ["player_id", "player_name", "school", "season_year"]
        if src == "roster":
            cols.append("position")
        try:
            frames.append(pd.read_parquet(p, columns=cols).assign(_src=src))
        except Exception:      # a source without the name/school columns
            continue
    if not frames:
        return {}
    s = pd.concat(frames, ignore_index=True).dropna(subset=["player_id", "season_year"])
    s["player_id"] = s["player_id"].astype(str)
    s = s[~s["player_id"].str.startswith("-")]
    s["season_year"] = s["season_year"].astype(int)
    s["_n"] = s["player_name"].map(_norm)
    s["_s"] = s["school"].map(_norm)
    s = s[s["_n"] != ""]
    if "position" not in s.columns:
        s["position"] = None
    # Stats, seasons and position are properties of the ID, across all its
    # rows: CFBD's stats feed often spells the name differently from its
    # roster ("Felix Anudike" vs "Felix Anudike-Uzomah", "Mar'Keise" vs
    # "Bucky" Irving), and the fragment matches the roster spelling.
    yrs = s.groupby("player_id")["season_year"].agg(frozenset).rename("yrs")
    n_stat = s[s["_src"] == "stats"].groupby("player_id")["season_year"].nunique().rename("n_stat")
    pos = (s.dropna(subset=["position"]).groupby("player_id")["position"]
             .agg(lambda x: x.mode().iloc[0]).rename("pos"))
    ids = pd.concat([yrs, n_stat, pos], axis=1)
    ids["n_stat"] = ids["n_stat"].fillna(0)
    ids["side"] = ids["pos"].map(lambda p: _SIDE.get(str(p).upper()) if isinstance(p, str) else None)
    keys = s[["_n", "_s", "player_id"]].drop_duplicates()

    frag_ids = set(ids.index[ids["n_stat"] == 0])
    kept_ids = set(ids.index[ids["n_stat"] > 0])
    pairs = keys[keys["player_id"].isin(frag_ids)].merge(
        keys[keys["player_id"].isin(kept_ids)], on=["_n", "_s"], suffixes=("", "_k"))
    pairs = pairs.drop_duplicates(["player_id", "player_id_k"])
    if pairs.empty:
        return {}
    f, k = ids.loc[pairs["player_id"]], ids.loc[pairs["player_id_k"]]
    contained = [a <= b for a, b in zip(f["yrs"], k["yrs"])]
    fs, ks = f["side"].to_numpy(), k["side"].to_numpy()
    same_side = [not (a is not None and b is not None and a != b) for a, b in zip(fs, ks)]
    hit = pairs[pd.Series(contained, index=pairs.index) & pd.Series(same_side, index=pairs.index)]
    out = ids.loc[hit["player_id"].unique(), "yrs"]
    return {pid: sorted(yrs) for pid, yrs in out.items()}


def detect_post_draft_seasons(raw_dir: Path, processed_dir: Path) -> dict[str, list[int]]:
    """Return {bridge player_id → its seasons in or after his draft year}.
    The college season is played the fall after the April draft, so a
    drafted or NFL-signed player can't have one: these rows are stale
    rosters (Pooka Williams, Kansas 2021), a transfer's old school still
    listing him (Adam Prentice, Colorado State 2021), or CFBD reusing the
    id for a later namesake (Kyle Williams, Arizona State through 2019).
    Left in, they move his last college season past his draft."""
    bp = processed_dir / "bridge_players.parquet"
    if not bp.exists():
        return {}
    b = pd.read_parquet(bp, columns=["player_id", "draft_year"]).dropna()
    draft = dict(zip(b["player_id"].astype(str), b["draft_year"].astype(int)))
    frames = [pd.read_parquet(p, columns=["player_id", "season_year"])
              for p in (raw_dir / "rosters.parquet", raw_dir / "player_seasons.parquet",
                        raw_dir / "player_gamelogs.parquet", processed_dir / "fcs_rosters.parquet")
              if p.exists()]
    if not frames:
        return {}
    s = pd.concat(frames, ignore_index=True).dropna().drop_duplicates()
    s["player_id"] = s["player_id"].astype(str)
    s = s[s["player_id"].isin(draft)]
    s = s[s["season_year"].astype(int) >= s["player_id"].map(draft)]
    return (s.groupby("player_id")["season_year"]
             .agg(lambda x: sorted(int(v) for v in x.unique())).to_dict())


def exclusion_mask(df: pd.DataFrame, exclusions: dict[str, list[int]]) -> pd.Series:
    """Boolean mask of df rows whose (player_id, season_year) is excluded."""
    if not exclusions or df.empty:
        return pd.Series(False, index=df.index)
    bad = pd.MultiIndex.from_tuples(
        [(str(pid), int(y)) for pid, years in exclusions.items() for y in years])
    keys = pd.MultiIndex.from_arrays(
        [df["player_id"].astype(str), pd.to_numeric(df["season_year"], errors="coerce")])
    return pd.Series(keys.isin(bad), index=df.index)
