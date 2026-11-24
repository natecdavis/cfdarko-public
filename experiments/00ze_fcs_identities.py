"""
FCS identities: score FCS players that CFBD doesn't carry, using PFF.

RESULT (2026-10-30): DON'T WIRE IT IN. The off-board prize is too small for
the model to find. 15,029 FCS identities (8,046 in classes 2014-22), 67 NFL
links (Cole Strange, Trevor Penning, Braxton Jones, Joe Haeg...), 44
combine and 31 board joins. Of the 15 FCS contributors in classes 2014-22,
9 are ON the board, so only 6 of 273 off-board contributors are FCS (<1 per
class).
    bar 1  off-board top 50/class contributors, aug − base +6, placebo −5,
           and ZERO FCS players entered the top 50: the +6 is CFBD players
           reshuffled by the extra training rows, not coverage. Fail.
    bar 2  original population aug − base: auc_ob +0.001, rho_ob_nfl
           −0.006 (placebo +0.000 / +0.016). No harm.
    FCS-only AUC(contributor) 0.85 pre / 1.00 post-draft. The model ranks
    FCS players among themselves, but their P(NFL) sits far below CFBD
    off-board players (FCS NFL rate 0.8% vs 2.4%), so none reach the top.
Side finding: 00zd's pooled model sees no PFF performance features at all
(only pff_seasons survives the shared-feature intersection). Adding the
side grade/snap triplets lifts the CFBD off-board model: auc_ob +0.007,
rho_ob_nfl +0.024, auc_udfa +0.007, rho_udfa_nfl +0.035 (base −
base_nopff; the auc gain exceeds the placebo, the rho gains are marginal).
Top-50 contributors don't move (113 vs 116).

WIRED IN ANYWAY (2026-10-31), for completeness: step 01c now mints these
identities (src/utils/fcs_identities.py) and steps 02/03/06 and 00d read
them. On that pipeline every FCS row already carries its synthetic id, so
_build_identities here finds none; rerun it only against a pre-wiring
pff_season_grades.parquet.

CFBD has no FCS rosters. An FCS player exists in CFBD only if he recorded a
stat against an FBS opponent (12-40 per school-year), so Cole Strange, Trey
Lance, Trevor Penning, Joe Haeg and Derek Rivers aren't in the universe at
all. PFF's FCS facet files (00b, 2014+, ~13k players a year) are the only
full-season FCS source. This prototype gives every PFF FCS player whose
rows never matched a CFBD id a synthetic id, SYN_BASE + pff_player_id, and
asks whether adding them to 00zd's off-board model is worth wiring into
the pipeline.

Identity: pff_player_id whose pff_season_grades rows are ALL unmatched,
whose last season is in an FCS file, last season >= 2014 (PFF's FCS
coverage before 2014 is 100-400 players a year, a selected set), and whose
PFF position maps to a non-K/P group (ED -> EDGE, DI -> DL).

Features, all built the way the pipeline builds them for CFBD players:
    PFF career triplets   src.scrapers.pff.load_pff_career_features on
                          pff_season_grades with the synthetic ids filled in
    combine               00d's raw frame and z-scores (nflverse + PFR +
                          mockdraftable splices); a record joins only on
                          exact name + cohort year + school agreement
    hasan_rank            hasan.csv rows that 00e left unmatched, same rule
    NFL era               nfl_league_averages by last season
then attach_derived_features over the concatenated snapshot. Everything
CFBD-derived (box score, Kalman/EWMA, team context, recruiting, height,
weight, age) is NaN, as it is for a CFBD roster-only player.

Outcomes: nfl_draft_labels rows with no player_id (NFL players the bridge
couldn't link) are linked to a synthetic id on name + last season (±1),
with school agreement required when the name is not unique or not exact.
School agreement: the PFF abbreviation ("CHATTNOOGA", "N DAK ST") maps to a
CFBD school by majority vote over PFF rows matched by name + year, falling
back to an in-order abbreviation test.

Arms (00zd's two-stage usage model, untuned, same folds). 00zd pools only
the features every position head shares, which drops every position-specific
PFF column: its model sees no PFF performance at all, so FCS identities
would have nothing to be scored on. Every arm here adds SIDE_FEATURES, the
offense (offensive groups) or defense grade and snap triplets:
    base_nopff  00zd as committed, read from --base-dir (reference)
    base        pipeline universe + SIDE_FEATURES
    aug         universe + FCS identities + SIDE_FEATURES (seed 42)
    aug_43      placebo: aug with seed 43
    base_post / aug_post   + drafted and draft round (post-draft)
Pre-registered bar (expanded off-board pool, classes 2015-22, FCS players
scored -inf under base because base can't see them):
    1. contributors in the off-board top 50 per class: aug − base must
       exceed |aug_43 − aug|, summed over classes;
    2. on the ORIGINAL population, aug − base on auc_ob and rho_ob_nfl must
       not be worse than the placebo swing (FCS rows mustn't hurt
       training for everyone else).
Plus diagnostics: link counts, FCS-only AUC, and how FCS entrants to the
top 50 contribute vs the players they displace.

Usage:
    00ze_fcs_identities.py --base-dir DIR --out-dir DIR
"""
import argparse
import glob
import importlib
import importlib.util
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from config import PROCESSED_DIR, RAW_DIR, BRIDGE_SCHOOL_ALIASES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("00ze")

SYN_BASE = 900_000_000
FCS_START = 2014
PFF_POS_TO_GROUP = {"QB": "QB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
                    "T": "OL", "G": "OL", "C": "OL", "DI": "DL", "ED": "EDGE",
                    "LB": "LB", "CB": "DB", "S": "DB"}
TOP_K = 50
OFFENSE = {"QB", "RB", "WR", "TE", "OL"}
SIDE_STATS = ["grades", "snap_counts"]
SIDE_AGGS = ["mean", "peak", "last"]
SIDE_FEATURES = [f"pff_side_{s}_{a}" for s in SIDE_STATS for a in SIDE_AGGS] + ["pff_side_snap_counts_total"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02", "pipeline/02_bridge_players.py")
zd = _load("zd", "experiments/00zd_offboard_model.py")
_norm = s02._norm


# ── school agreement ────────────────────────────────────────────────────────

def _subseq(a: str, b: str) -> bool:
    it = iter(b)
    return all(ch in it for ch in a)


def _abbrev_match(abbr: str, full: str) -> bool:
    """'N DAK ST' ~ 'north dakota state', 'CHATTNOOGA' ~ 'chattanooga'."""
    a, f = _norm(abbr).replace(" ", ""), _norm(full).replace(" ", "")
    return bool(a) and bool(f) and a[0] == f[0] and _subseq(a, f) and len(a) >= 0.4 * len(f)


def _learn_school_map(grades: pd.DataFrame) -> dict[str, str]:
    """PFF school_norm → normalized CFBD school, by majority vote over PFF rows
    matched to a CFBD id (any method), using that id's CFBD school that year."""
    ps = pd.concat([pd.read_parquet(RAW_DIR / f, columns=["player_id", "school", "season_year"])
                    for f in ("player_seasons.parquet", "rosters.parquet")])
    ps["player_id"] = ps["player_id"].astype(str)
    ps = ps.drop_duplicates(["player_id", "season_year"])
    m = grades[grades["player_id"].notna()][["player_id", "season_year", "school_norm"]].copy()
    m["player_id"] = m["player_id"].astype(str)
    m = m.merge(ps, on=["player_id", "season_year"])
    m["cfbd"] = m["school"].map(_norm)
    votes = m.groupby(["school_norm", "cfbd"]).size().rename("n").reset_index()
    tot = votes.groupby("school_norm")["n"].transform("sum")
    best = votes[(votes["n"] >= 3) & (votes["n"] / tot >= 0.6)]
    return dict(zip(best["school_norm"], best["cfbd"]))


def _school_keys_nflverse(college) -> set[str]:
    return set(s02._school_keys(college))


def _schools_agree(pff_team: str, pff_cfbd: str | None, other_keys: set[str]) -> bool:
    if not other_keys:
        return False
    if pff_cfbd and pff_cfbd in other_keys:
        return True
    return any(_abbrev_match(pff_team, k) for k in other_keys)


# ── identities ──────────────────────────────────────────────────────────────

def _fcs_raw() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RAW_DIR / "pff" / "*" / "*_fcs.csv")):
        d = pd.read_csv(f, usecols=lambda c: c in ("player", "player_id", "team_name", "position"))
        d["season_year"] = int(Path(f).parent.name)
        rows.append(d)
    return (pd.concat(rows, ignore_index=True).drop_duplicates()
              .rename(columns={"player_id": "pff_player_id"}))


def _build_identities(grades: pd.DataFrame) -> pd.DataFrame:
    fcs = _fcs_raw()
    fcs_keys = set(zip(fcs["pff_player_id"], fcs["season_year"]))
    g = grades.copy()
    g["is_fcs"] = [k in fcs_keys for k in zip(g["pff_player_id"], g["season_year"])]
    per = g.groupby("pff_player_id").agg(
        any_matched=("player_id", lambda s: s.notna().any()),
        last=("season_year", "max"), first=("season_year", "min"),
        n_seasons=("season_year", "nunique"))
    last_rows = g.sort_values("season_year").drop_duplicates("pff_player_id", keep="last")
    per = per.join(last_rows.set_index("pff_player_id")[["is_fcs", "school_norm", "player_name_norm"]])
    pos = (g.dropna(subset=["pff_position"]).groupby("pff_player_id")["pff_position"]
             .agg(lambda s: s.value_counts().index[0]))
    per["pos_group"] = pos.map(PFF_POS_TO_GROUP)
    # display name + team from the raw FCS files (last season)
    raw_last = (fcs.sort_values("season_year").drop_duplicates("pff_player_id", keep="last")
                   .set_index("pff_player_id")[["player", "team_name"]])
    per = per.join(raw_last)
    ids = per[~per["any_matched"] & per["is_fcs"] & (per["last"] >= FCS_START)
              & per["pos_group"].notna() & per["player"].notna()].copy()
    ids["player_id"] = (SYN_BASE + ids.index.astype(int)).astype(str)
    ids["nn"] = ids["player"].map(_norm)
    log.info("FCS identities: %d (of %d PFF ids; %d FCS-final unmatched before filters)",
             len(ids), len(per), int((~per["any_matched"] & per["is_fcs"]).sum()))
    return ids.reset_index()


# ── feature joins ───────────────────────────────────────────────────────────

def _pff_career(grades: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    from src.scrapers.pff import load_pff_career_features
    syn = dict(zip(ids["pff_player_id"], ids["player_id"]))
    g = grades.copy()
    fill = g["player_id"].isna() & g["pff_player_id"].isin(syn)
    g.loc[fill, "player_id"] = g.loc[fill, "pff_player_id"].map(syn)
    # Full frame, not just the FCS rows: the loader shrinks each season toward
    # a position-pooled prior computed from the rows it's given.
    with tempfile.TemporaryDirectory() as tmp:
        g.to_parquet(Path(tmp) / "pff_season_grades.parquet", index=False)
        career = load_pff_career_features(pd.DataFrame({"player_id": g["player_id"].dropna().unique()}), Path(tmp))
    career["player_id"] = career["player_id"].astype(str)
    return career[career["player_id"].isin(set(ids["player_id"]))]


def _combine_raw() -> pd.DataFrame:
    """00d's pre-matching frame: same splices, renames and cohort z-scores."""
    d = _load("s00d", "pipeline/00d_combine_ingestion.py")
    raw = d.nfl.import_combine_data(list(range(d.START_SEASON, d.END_SEASON + 1)))
    supp_dir = RAW_DIR / "pfr_combine"
    for f in sorted(supp_dir.glob("pfr_combine_*.parquet")) if supp_dir.exists() else []:
        supp = pd.read_parquet(f)
        raw = pd.concat([raw[~raw["season"].isin(set(supp["season"]))], supp], ignore_index=True)
    md_dir = RAW_DIR / "mockdraftable"
    parts = [pd.read_parquet(f) for f in sorted(md_dir.glob("mockdraftable_*.parquet"))] if md_dir.exists() else []
    if parts:
        md = pd.concat(parts, ignore_index=True)
        md["_n"], md["_y"] = md["player_name"].apply(d._normalize_name), md["season"].astype(int)
        md = md[["_n", "_y", "arm", "hand", "wingspan"]].drop_duplicates(["_n", "_y"])
        raw["_n"], raw["_y"] = raw["player_name"].apply(d._normalize_name), raw["season"].fillna(0).astype(int)
        raw = raw.merge(md, on=["_n", "_y"], how="left").drop(columns=["_n", "_y"])
    raw = raw.rename(columns={m: f"combine_{m}" for m in d.COMBINE_METRICS})
    if "combine_ht" in raw.columns:
        raw["combine_ht"] = raw["combine_ht"].apply(d._ht_to_inches)

    def _map_pos(p):
        if not isinstance(p, str):
            return None
        for tok in p.replace("\\", "/").split("/"):
            if d.COMBINE_POS_MAP.get(tok.strip().upper()):
                return d.COMBINE_POS_MAP[tok.strip().upper()]
        return None
    raw["position_group"] = raw["pos"].apply(_map_pos)
    raw = raw[raw["position_group"].notna()].copy()
    raw["cohort_year"] = raw["draft_year"].fillna(raw["season"]).astype(int)
    for m in d.COMBINE_METRICS:
        c = f"combine_{m}"
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
            raw[f"{c}_z"] = d._zscore_within_cohort(raw, c, ["position_group", "cohort_year"])
    raw["combine_invited"] = 1
    return raw


def _join_by_name_school(ids, other, name_col, school_col, year_col, school_map, label):
    """ids rows ← `other` rows on exact name + year (= last + 1) + school
    agreement. Returns {synthetic player_id: other index}; ambiguous dropped."""
    o = other.copy()
    o["nn"] = o[name_col].map(_norm)
    cand = ids[["player_id", "nn", "last", "team_name", "school_norm"]].merge(o[["nn", year_col, school_col]].reset_index(), on="nn")
    cand = cand[cand[year_col] == cand["last"] + 1]
    ok = [
        _schools_agree(t, school_map.get(sn), _school_keys_nflverse(sc))
        for t, sn, sc in zip(cand["team_name"], cand["school_norm"], cand[school_col])
    ]
    cand = cand[ok]
    cand = cand[~cand["player_id"].duplicated(keep=False) & ~cand["index"].duplicated(keep=False)]
    log.info("%s joined to %d FCS identities", label, len(cand))
    return dict(zip(cand["player_id"], cand["index"]))


def _synthetic_snapshot(ids, career, combine, hasan, school_map, columns):
    syn = pd.DataFrame({
        "player_id": ids["player_id"].values,
        "player_name": ids["player"].values,
        "school": ids["team_name"].values,
        "position_group": ids["pos_group"].values,
        "last_college_season_year": ids["last"].astype(float).values,
        "season_year": ids["last"].astype(float).values,
        "n_seasons": ids["n_seasons"].values,
        "conference_tier": 3.0,
    })
    syn = syn.merge(career, on="player_id", how="left")

    cmap = _join_by_name_school(ids, combine, "player_name", "school", "cohort_year", school_map, "combine")
    ccols = [c for c in combine.columns if c.startswith("combine_") and c in columns] + ["combine_invited"]
    cj = combine.loc[list(cmap.values()), [c for c in dict.fromkeys(ccols) if c in combine.columns]].copy()
    cj["player_id"] = list(cmap.keys())
    cj["combine_draft_year"] = combine.loc[list(cmap.values()), "cohort_year"].values
    syn = syn.merge(cj, on="player_id", how="left")
    syn["combine_invited"] = syn["combine_invited"].fillna(0).astype(int)

    hmap = _join_by_name_school(ids, hasan, "name", "school", "year", school_map, "hasan board")
    syn["hasan_rank"] = syn["player_id"].map({k: hasan.loc[v, "rank"] for k, v in hmap.items()}).astype(float)
    syn["hasan_draft_year"] = syn["player_id"].map({k: hasan.loc[v, "year"] for k, v in hmap.items()})

    nfl_lg = pd.read_parquet(PROCESSED_DIR / "nfl_league_averages.parquet")
    lg_cols = [c for c in nfl_lg.columns if c != "season_year" and c in columns]
    syn = syn.merge(nfl_lg[["season_year"] + lg_cols].rename(columns={"season_year": "last_college_season_year"}),
                    on="last_college_season_year", how="left")
    return syn.reindex(columns=list(dict.fromkeys(list(columns) + list(syn.columns))))


def _add_side_pff(snap: pd.DataFrame) -> pd.DataFrame:
    """Position-agnostic PFF career features: the offense grade / snaps for
    offensive groups, defense for defensive ones. 00zd pools only features
    every head shares, which excludes all position-specific PFF columns, so
    without these the pooled model sees no PFF performance at all."""
    snap = snap.copy()
    off = snap["position_group"].isin(OFFENSE)
    for s in SIDE_STATS:
        for a in SIDE_AGGS + (["total"] if s == "snap_counts" else []):
            o, d = f"pff_{s}_offense_{a}", f"pff_{s}_defense_{a}"
            if o in snap.columns and d in snap.columns:
                snap[f"pff_side_{s}_{a}"] = np.where(off, snap[o], snap[d])
    return snap


# ── outcome links ───────────────────────────────────────────────────────────

def _link_labels(ids, labels, bridge, school_map):
    """Unlinked NFL label rows → synthetic ids. Returns labels with player_id
    filled for linked rows, and a bridge frame (player_id, round) for them."""
    lab = labels[labels["player_id"].isna() & labels["is_bridge"].fillna(False).astype(bool)].copy()
    b = bridge[["pfr_id", "college_team", "round", "draft_year"]].dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    lab = lab.reset_index().merge(b.drop(columns="draft_year"), on="pfr_id", how="left").set_index("index")
    lab["last"] = lab["last_college_season_year"].fillna(lab["draft_year"] - 1)
    lab["nn"] = lab["player_name"].map(_norm)
    ids = ids.assign(surname=ids["nn"].str.split(" ").str[-1])
    by_last = {y: g for y, g in ids.groupby("last")}
    links, how = {}, {"exact_unique": 0, "exact_school": 0, "fuzzy_school": 0}
    for ix, r in lab.iterrows():
        if pd.isna(r["last"]):
            continue
        near = [by_last[y] for y in (r["last"] - 1, r["last"], r["last"] + 1) if y in by_last]
        if not near:
            continue
        pool = pd.concat(near)
        keys = _school_keys_nflverse(r["college_team"])
        # Name variants differ in the given name (nicknames, initials), so
        # candidates share the surname; school agreement only on those.
        pool = pool[pool["surname"] == r["nn"].split(" ")[-1]] if r["nn"] else pool.iloc[:0]
        if pool.empty:
            continue
        agree = pool.apply(lambda c: _schools_agree(c["team_name"], school_map.get(c["school_norm"]), keys), axis=1)
        exact = pool[pool["nn"] == r["nn"]]
        if len(exact) == 1 and (agree[exact.index].all() or not keys):
            links[ix], kind = exact["player_id"].iloc[0], "exact_unique"
        elif len(exact[agree[exact.index]]) == 1:
            links[ix], kind = exact[agree[exact.index]]["player_id"].iloc[0], "exact_school"
        else:
            fz = pool[agree & pool["nn"].map(lambda n: s02._name_similar(r["nn"], n))]
            if len(fz) != 1:
                continue
            links[ix], kind = fz["player_id"].iloc[0], "fuzzy_school"
        how[kind] += 1
    s = pd.Series(links)
    s = s[~s.duplicated(keep=False)]
    log.info("linked %d of %d unlinked NFL label rows to FCS identities %s",
             len(s), len(lab), how)
    out = labels.copy()
    out.loc[s.index, "player_id"] = s.values
    br = pd.DataFrame({"player_id": s.values, "round": lab.loc[s.index, "round"].values})
    return out, br


# ── evaluation ──────────────────────────────────────────────────────────────

def _is_fcs(pid: pd.Series) -> pd.Series:
    return pd.to_numeric(pid, errors="coerce") >= SYN_BASE


def _eval_frame(base):
    E = base[base["last_college_season_year"].between(*zd.EVAL_YEARS)].drop_duplicates("player_id", keep="last").copy()
    E["cls"] = E["last_college_season_year"].astype(int) + 1
    E["board"] = E["hasan_rank"].notna()
    E["contrib"] = (E["usage"] >= zd.CONTRIB).astype(int)
    E["starter"] = (E["usage"] >= zd.STARTER).astype(int)
    E["fcs"] = _is_fcs(E["player_id"])
    return E


def _topk_contrib(E, col, k=TOP_K):
    ob = E[~E["board"]]
    rk = ob.groupby("cls")[col].rank(ascending=False, method="first")
    t = ob[rk <= k]
    return t.groupby("cls")["contrib"].sum(), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, required=True,
                    help="00zd output dir on the same pipeline state (oof_all_arms.parquet)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step8 = importlib.import_module("08_draft_models")
    grades = pd.read_parquet(PROCESSED_DIR / "pff_season_grades.parquet")
    school_map = _learn_school_map(grades)
    log.info("learned %d PFF school abbreviations", len(school_map))
    ids = _build_identities(grades)

    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    labels_aug, bridge_syn = _link_labels(ids, labels, bridge, school_map)

    from _snapshot_assembly import assemble_snapshot, attach_derived_features
    snap = assemble_snapshot(include_derived=False, logger=step8.logger)
    snap["player_id"] = snap["player_id"].astype(str)
    hasan = pd.read_csv(ROOT / "hasan.csv", encoding="utf-8-sig")
    hasan_ids = pd.read_parquet(PROCESSED_DIR / "hasan_ranks.parquet")
    matched = set(zip(hasan_ids["draft_year"], hasan_ids["hasan_rank"]))
    hasan = hasan[[(y, r) not in matched for y, r in zip(hasan["year"], hasan["rank"])]]
    log.info("hasan rows unmatched by 00e: %d", len(hasan))
    syn = _synthetic_snapshot(ids, _pff_career(grades, ids), _combine_raw(), hasan,
                              school_map, snap.columns)
    snap_aug = attach_derived_features(pd.concat([snap, syn[snap.columns.intersection(syn.columns)]],
                                                 ignore_index=True), logger=step8.logger)
    bridge_aug = pd.concat([bridge, bridge_syn], ignore_index=True)

    folds, _ = step8._split_holdout(step8._expanding_window_folds())
    snap_aug = _add_side_pff(snap_aug)
    oof = {}
    for tag, frame in (("aug", snap_aug), ("base", snap_aug[~_is_fcs(snap_aug["player_id"])])):
        rows, feats = zd._build_rows(step8, snapshot=frame, labels=labels_aug, bridge=bridge_aug,
                                     extra=SIDE_FEATURES)
        n_syn = _is_fcs(rows["player_id"])
        log.info("%s rows %d (FCS %d, FCS NFL %d); features %d", tag, len(rows), int(n_syn.sum()),
                 int(rows.loc[n_syn, "nfl"].sum()), len(feats))
        out, fold_of = zd._run_arms(step8, rows, feats, folds)
        o = rows[zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted"]].assign(val_start=fold_of)
        for a in ("pre", "pre_43", "post"):
            o[a] = out[a]["p1"] * out[a]["m2"]
        o = o.dropna(subset=["val_start", "pre"])
        o["player_id"] = o["player_id"].astype(str)
        o.to_parquet(args.out_dir / f"oof_{tag}.parquet", index=False)
        oof[tag] = o

    E = _eval_frame(oof["aug"]).rename(columns={"pre": "aug", "pre_43": "aug_43", "post": "aug_post"})
    for name, src, col in (("base", oof["base"], "pre"), ("base_post", oof["base"], "post"),
                           ("base_nopff", pd.read_parquet(args.base_dir / "oof_all_arms.parquet"), "pre")):
        s = src.assign(player_id=src["player_id"].astype(str)).drop_duplicates("player_id", keep="last")
        E[name] = E["player_id"].map(s.set_index("player_id")[col]).fillna(-1.0)   # FCS unscored
    fcs = E[E["fcs"]]
    ob = E[~E["board"]]
    print(f"\neval cohort: {len(E)} players; off-board {len(ob)}; FCS identities {len(fcs)} "
          f"(NFL {int(fcs['nfl'].sum())}, contributors {int(fcs['contrib'].sum())}, "
          f"starters {int(fcs['starter'].sum())}, on board {int(fcs['board'].sum())})")
    print(f"off-board contributors: {int(ob['contrib'].sum())}, of which FCS {int(ob.loc[ob['fcs'], 'contrib'].sum())}")

    print(f"\n1. off-board top {TOP_K} per class, expanded pool (FCS scored -1 under base):")
    res = {}
    for a in ("base_nopff", "base", "aug", "aug_43", "base_post", "aug_post"):
        per_cls, t = _topk_contrib(E, a)
        res[a] = per_cls
        print(f"  {a:9s} contributors {int(per_cls.sum()):4d}  rate {t['contrib'].mean():.3f}  "
              f"NFL rate {t['nfl'].mean():.3f}  FCS in top {int(t['fcs'].sum()):3d} "
              f"(contrib {int(t.loc[t['fcs'], 'contrib'].sum())})")
    d_real = (res["aug"] - res["base"])
    d_plac = (res["aug_43"] - res["aug"])
    print(f"  aug − base {int(d_real.sum()):+d} (per class {d_real.astype(int).tolist()}); "
          f"placebo aug_43 − aug {int(d_plac.sum()):+d} (per class {d_plac.astype(int).tolist()})")

    print("\n2. original population (no FCS rows):")
    m = zd._metrics(E[~E["fcs"]], ["base_nopff", "base", "aug", "aug_43"])
    print(pd.DataFrame(m).T.round(4).to_string())
    print(f"  aug − base: auc_ob {m['aug']['auc_ob'] - m['base']['auc_ob']:+.4f}  "
          f"rho_ob_nfl {m['aug']['rho_ob_nfl'] - m['base']['rho_ob_nfl']:+.4f}; "
          f"placebo: auc_ob {m['aug_43']['auc_ob'] - m['aug']['auc_ob']:+.4f}  "
          f"rho_ob_nfl {m['aug_43']['rho_ob_nfl'] - m['aug']['rho_ob_nfl']:+.4f}")
    print(f"  side PFF features alone (base − base_nopff): auc_ob {m['base']['auc_ob'] - m['base_nopff']['auc_ob']:+.4f}  "
          f"rho_ob_nfl {m['base']['rho_ob_nfl'] - m['base_nopff']['rho_ob_nfl']:+.4f}")

    print("\n3. FCS identities only:")
    for a in ("aug", "aug_43", "aug_post"):
        print(f"  {a:9s} AUC(NFL) {np.nanmean([zd._auc(g['nfl'].astype(int).values, g[a].values) for _, g in fcs.groupby('cls')]):.3f}  "
              f"AUC(contrib) {np.nanmean([zd._auc(g['contrib'].values, g[a].values) for _, g in fcs.groupby('cls')]):.3f}")
    _, t_aug = _topk_contrib(E, "aug")
    _, t_base = _topk_contrib(E, "base")
    entered = t_aug[t_aug["fcs"]]
    displaced = t_base[~t_base["player_id"].isin(t_aug["player_id"])]
    print(f"  FCS entrants to the aug top {TOP_K}: {len(entered)}, contributor rate {entered['contrib'].mean():.3f}; "
          f"displaced base players: {len(displaced)}, contributor rate {displaced['contrib'].mean():.3f}")
    show = fcs[fcs["contrib"] == 1].sort_values("aug", ascending=False)
    show = show.assign(rank_in_class=E[~E["board"]].groupby("cls")["aug"].rank(ascending=False).reindex(show.index))
    print("\n  FCS contributors (off-board rank within class under aug):")
    names = dict(zip(ids["player_id"], ids["player"]))
    show["name"] = show["player_id"].map(names)
    print(show[["name", "cls", "pos_group", "usage", "hasan_rank", "rank_in_class"]].head(40).to_string(index=False))
    E.to_parquet(args.out_dir / "eval_frame.parquet", index=False)


if __name__ == "__main__":
    main()
