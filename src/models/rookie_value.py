"""Rookie-deal value and surplus label (production home of experiments/00zo).

value_y = veteran-market price of the player's role at his NFL position x
games played, % of cap; seasons priced on (snap share, PFF grade quality)
at every position (v4, experiments/01e; v3.1 used EPA for QB/RB/WR/TE). cost_y = rookie-deal APY % of cap. surplus_y14 = Σ value − Σ cost
over y1-y4; surplus_y15 adds the round-1 option year. Seasons before 2013
use a depth-chart model. Full definition, validation and history:
experiments/00zo_rookie_surplus_label.py (label), 00zp/00zq (quality).

build_label() writes PROCESSED_DIR/rookie_value_labels.parquet; the
pipeline runs it as step 7.5 (pipeline/07b_rookie_value.py).

Caches (data/raw/rookie_value/, fetched on first run by ensure_caches):
snap counts 2013-15, depth charts 2017-24, nflverse id table, nflverse
seasonal stats (EPA; 2025 not published yet), player names for 2016-24
snap ids.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb
from scipy.stats import spearmanr

from config import PROCESSED_DIR, RAW_DIR, END_YEAR
from src.scrapers.nfl_data import (
    SEASONAL_ROSTERS_CACHE, DEPTH_CHART_CACHE, _build_pfr_to_gsis_crosswalk,
    build_bridge_gsis_crosswalk, reg_season_games, fetch_depth_charts,
)

log = logging.getLogger(__name__)
FIRST_CLASS, LAST_SEASON = 2006, END_YEAR - 1
SNAP_FIRST, DC_FIT = 2013, (2013, 2024)        # snaps 2013..2025; depth charts 2006..2024
CBA_FIRST_CLASS = 2011
CACHE = RAW_DIR / "rookie_value"      # nflverse downloads; fetched by ensure_caches()
ST_MIN = 0.25
PAY_CACHE = CACHE
QUAL_CACHE = CACHE / "nfl_seasonal_stats.parquet"
LABEL_PATH = PROCESSED_DIR / "rookie_value_labels.parquet"
SKILL = ["QB", "RB", "WR", "TE"]
Q_K = {"QB": 150, "RB": 60, "WR": 40, "TE": 40}
PAY_WIDE = (2016, 2024)
PAY_LEVEL = (2023, 2024)   # per-group level recalibration seasons (today's market)
PAY_GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
VAL_CLASSES = (2011, 2020)
PFF_DIR = RAW_DIR / "pff_nfl"
PFF_K = 200   # shrinkage toward the (group x season) snap-weighted mean grade, in snaps
PFF_TABLE = {"QB": ("passing_summary", "grades_offense"), "RB": ("rushing_summary", "grades_offense"),
             "WR": ("receiving_summary", "grades_offense"), "TE": ("receiving_summary", "grades_offense"),
             "OL": ("offense_blocking", "grades_offense"),
             **{g: ("defense_summary", "grades_defense") for g in ["DL", "EDGE", "LB", "DB"]}}
QUALITY_GROUPS = {"none": [], "epa": SKILL, "pff": PAY_GROUPS}
PAY_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)


def ensure_caches() -> None:
    """Fetch the nflverse inputs this label needs if they are missing."""
    import nfl_data_py as nfl
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "snaps_raw_2012_2015.parquet"      # name kept; 2012 is not published
    if not p.exists():
        log.info("fetching nflverse snap counts 2013-2015 → %s", p)
        nfl.import_snap_counts([2013, 2014, 2015]).query("game_type == 'REG'").to_parquet(p, index=False)
    p = CACHE / "depth_charts_2017_2024.parquet"
    if not p.exists():
        log.info("fetching nflverse depth charts 2017-2024 → %s", p)
        fetch_depth_charts(list(range(2017, 2025)), cache_path=p)
    p = CACHE / "nflverse_ids.parquet"
    if not p.exists():
        nfl.import_ids()[["pfr_id", "gsis_id"]].dropna().to_parquet(p, index=False)
    p = CACHE / "snap_names_2016_2024.parquet"
    if not p.exists():
        nfl.import_snap_counts(list(range(2016, 2025)))[["pfr_player_id", "player"]] \
            .drop_duplicates("pfr_player_id").to_parquet(p, index=False)
    if not QUAL_CACHE.exists():
        frames = []
        for y in range(2006, END_YEAR):
            try:
                frames.append(nfl.import_seasonal_data([y], s_type="REG"))
            except Exception as exc:   # nflverse publishes a season months after it ends
                log.warning("nflverse seasonal stats %d unavailable (%s); quality neutral", y, exc)
        pd.concat(frames, ignore_index=True).to_parquet(QUAL_CACHE, index=False)


def quality_table(pos: pd.Series):
    """q[(gsis, season)] = within-(group × season) percentile of shrunk EPA per
    opportunity; neutral[(group, season)] = percentile of a no-touch player."""
    s = pd.read_parquet(QUAL_CACHE)
    s["g"] = s["player_id"].map(pos)
    s = s[s["g"].isin(SKILL)].copy()
    s["opp"] = s[["attempts", "sacks", "carries", "targets"]].fillna(0).sum(axis=1)
    s["epa"] = s[["passing_epa", "rushing_epa", "receiving_epa"]].fillna(0).sum(axis=1)
    tot = s.groupby(["g", "season"])[["epa", "opp"]].sum()
    mu = tot["epa"] / tot["opp"]
    k = s["g"].map(Q_K)
    m = mu.reindex(pd.MultiIndex.from_frame(s[["g", "season"]])).to_numpy()
    s["rate"] = (s["epa"] + k * m) / (s["opp"] + k)
    q, neutral = {}, {}
    for (g, y), G in s.groupby(["g", "season"]):
        ref = np.sort(G.loc[G["opp"] > 0, "rate"].to_numpy())
        pct = lambda x: np.searchsorted(ref, x, side="right") / len(ref)
        q.update(zip(zip(G["player_id"], G["season"]), pct(G["rate"].to_numpy())))
        neutral[(g, y)] = float(pct(mu[(g, y)]))
    for g in SKILL:
        neutral[(g, None)] = float(np.mean([v for (gg, _), v in neutral.items() if gg == g]))
    log.info("quality: %d player-seasons, neutral q by group %s", len(q),
             {g: round(neutral[(g, None)], 3) for g in SKILL})
    return q, neutral


def pff_grade_seasons(years) -> pd.DataFrame:
    """One row per (gsis, season, group-table): PFF season grade and snaps.
    Group tables as 01b; snaps = offense_blocking snap_counts_offense
    (QB: max with passing_snaps) or defense_summary snap_counts_defense.
    PFF id -> gsis via nflverse players.csv. Missing season files are skipped."""
    def read(y, name):
        f = PFF_DIR / str(y) / f"{name}.csv"
        return pd.read_csv(f, low_memory=False) if f.exists() else None
    rows = []
    for y in years:
        ob = read(y, "offense_blocking")
        osn = ob.drop_duplicates("player_id").set_index("player_id")["snap_counts_offense"] \
            if ob is not None else pd.Series(dtype=float)
        tabs = {}
        for g, (name, col) in PFF_TABLE.items():
            if name not in tabs:
                tabs[name] = read(y, name)
            t = tabs[name]
            if t is None:
                continue
            t = t.drop_duplicates("player_id")
            if name == "defense_summary":
                snaps = t["snap_counts_defense"]
            else:
                snaps = t["player_id"].map(osn)
                if name == "passing_summary":
                    snaps = np.fmax(snaps, t["passing_snaps"])
            rows.append(pd.DataFrame({"pff_id": t["player_id"].to_numpy(), "season": y, "group": g,
                                      "grade": t[col].to_numpy(), "snaps": np.asarray(snaps, dtype=float)}))
    s = pd.concat(rows, ignore_index=True)
    pl = pd.read_csv(CACHE / "nflverse_players.csv", low_memory=False, usecols=["gsis_id", "pff_id"]).dropna()
    pl = pl.assign(pff_id=pl["pff_id"].astype(int)).drop_duplicates("pff_id")
    return s.merge(pl, on="pff_id")


def quality_table_pff(pos: pd.Series):
    """v4 quality: q[(gsis, season)] = within-(group x season) percentile of
    the shrunk PFF grade, rate = (grade*snaps + PFF_K*mu)/(snaps + PFF_K),
    mu = snap-weighted mean grade of the group-season. The grade table is the
    player's pricing group (pos). neutral[(group, season)] = percentile of mu."""
    s = pff_grade_seasons(range(FIRST_CLASS, LAST_SEASON + 1))
    s["g"] = s["gsis_id"].map(pos)
    s = s[(s["g"] == s["group"]) & (s["snaps"] > 0) & s["grade"].notna()].copy()
    s = s.sort_values("snaps").drop_duplicates(["gsis_id", "season"], keep="last")
    s["gs"] = s["grade"] * s["snaps"]
    tot = s.groupby(["g", "season"])[["gs", "snaps"]].sum()
    mu = tot["gs"] / tot["snaps"]
    m = mu.reindex(pd.MultiIndex.from_frame(s[["g", "season"]])).to_numpy()
    s["rate"] = (s["gs"] + PFF_K * m) / (s["snaps"] + PFF_K)
    q, neutral = {}, {}
    for (g, y), G in s.groupby(["g", "season"]):
        ref = np.sort(G["rate"].to_numpy())
        pct = lambda x: np.searchsorted(ref, x, side="right") / len(ref)
        q.update(zip(zip(G["gsis_id"], G["season"]), pct(G["rate"].to_numpy())))
        neutral[(g, y)] = float(pct(mu[(g, y)]))
    for g in PAY_GROUPS:
        neutral[(g, None)] = float(np.mean([v for (gg, _), v in neutral.items() if gg == g]))
    log.info("PFF quality: %d player-seasons (%d-%d), neutral q by group %s", len(q),
             int(s["season"].min()), int(s["season"].max()),
             {g: round(neutral[(g, None)], 3) for g in PAY_GROUPS})
    return q, neutral


def xval_fold(gsis, k):
    return int(pd.util.hash_pandas_object(pd.Series([gsis]), index=False).iloc[0] % k)


def pay_curves_wide(snaps, pfr2gsis, c, qv, qgroups, fold=None, k=5):
    """Per-group price(share, q): veteran APY % of cap of the contract in force,
    seasons PAY_WIDE. Skill groups and quality groups (qgroups): monotone
    LightGBM on (share, q) if the group is in qgroups else share only; other
    groups: isotonic on share."""
    quality = bool(qgroups)
    d = snaps.reset_index()
    d = d[d["season"].between(*PAY_WIDE) & (d["share"] > 0)].copy()
    d["gsis"] = d["pfr_player_id"].map(pfr2gsis)
    rookie = c["year_signed"] == c["draft_year"]
    fifth = (c["draft_round"] == 1) & (c["years"] == 1) & (c["year_signed"] == c["draft_year"] + 4)
    vet = c[~rookie & ~fifth & c["position_group"].isin(PAY_GROUPS)].dropna(
        subset=["gsis_id", "year_signed", "years", "apy_cap_pct"])
    if fold is not None:
        h = pd.util.hash_pandas_object(vet["gsis_id"], index=False).to_numpy() % k
        val = vet["draft_year"].between(*VAL_CLASSES).to_numpy() & (h == fold)
        log.info("pay fit: xval fold %d/%d excludes %d contracts of %d players", fold, k,
                 int(val.sum()), vet.loc[val, "gsis_id"].nunique())
        vet = vet[~val]
    m = d.dropna(subset=["gsis"]).merge(vet, left_on="gsis", right_on="gsis_id")
    m = m[(m["year_signed"] <= m["season"]) & (m["season"] < m["year_signed"] + m["years"])]
    m = m.sort_values("year_signed").drop_duplicates(["gsis", "season"], keep="last")
    log.info("pay fit: %d veteran seasons by group %s", len(m), m.groupby("position_group").size().to_dict())
    m["q"] = [qv(gs, g, y) for gs, g, y in zip(m["gsis"], m["position_group"], m["season"])]
    curves, rows = {}, []
    for g, G in m.groupby("position_group"):
        recent = G[G["season"].between(*PAY_LEVEL)]
        if g in SKILL or g in qgroups:
            feats = ["share", "q"] if g in qgroups else ["share"]
            mdl = lgb.LGBMRegressor(**PAY_PARAMS, monotone_constraints=[1] * len(feats)).fit(
                G[feats], G["apy_cap_pct"])
            memo = {}

            def price(sh, qq, mdl=mdl, feats=feats, memo=memo):
                key = (round(sh, 3), round(qq, 2))
                if key not in memo:
                    x = pd.DataFrame([[key[0], key[1]][:len(feats)]], columns=feats)
                    memo[key] = max(float(mdl.predict(x)[0]), 0.0)
                return memo[key]
        else:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0).fit(
                G["share"], G["apy_cap_pct"])

            def price(sh, qq, iso=iso):
                return float(iso.predict([sh])[0])
        fitted = sum(price(a, b) for a, b in zip(recent["share"], recent["q"]))
        scale = float(recent["apy_cap_pct"].sum() / fitted) if fitted > 0 else 1.0

        def scaled(sh, qq, f=price, k=scale):
            return f(sh, qq) * k
        price = curves[g] = scaled
        row = {"group": g, "n": len(G), "level_scale": round(scale, 3)}
        for sh in (0.25, 0.5, 0.9):
            if g in qgroups:
                row.update({f"f({sh},q{qq})": round(price(sh, qq), 4) for qq in (0.1, 0.5, 0.9)})
            else:
                row[f"f({sh})"] = round(price(sh, 0.5), 4)
        rows.append(row)
    print(f"\npay curves (wide {PAY_WIDE}, quality={quality}), APY % of cap:")
    print(pd.DataFrame(rows).set_index("group").to_string())
    return curves




def cap_by_year(c: pd.DataFrame) -> pd.Series:
    big = c[c["apy_cap_pct"] >= 0.05]
    return big.groupby("year_signed")["implied_cap"].median()


def rookie_costs(c, cap, players):
    """Rookie-deal APY % of cap and option APY per pfr_id."""
    c = c.dropna(subset=["gsis_id", "year_signed"]).copy()
    c["pct"] = c["apy"] / c["year_signed"].map(cap)
    rk = (c[c["year_signed"] == c["draft_year"]].sort_values("apy", ascending=False)
          .drop_duplicates("gsis_id").set_index("gsis_id")["pct"])
    opt = c[(c["draft_round"] == 1) & (c["years"] == 1) & (c["year_signed"] == c["draft_year"] + 4)]
    opt = opt.drop_duplicates("gsis_id").set_index("gsis_id")["pct"]
    p = players.copy()
    p["rookie_pct"] = p["gsis"].map(rk)
    p["option_pct"] = p["gsis"].map(opt)
    p["rookie_src"] = np.where(p["rookie_pct"].notna(), "otc", None)
    post = p[(p["draft_year"] >= CBA_FIRST_CLASS) & p["drafted"] & p["rookie_pct"].notna()]
    scale = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
        post["overall_pick"], post["rookie_pct"])
    udfa_post = c[(c["year_signed"] == c["draft_year"]) & (c["draft_year"] >= CBA_FIRST_CLASS)
                  & c["draft_round"].isna() & (c["apy"] > 0)]["pct"].median()
    pre = p["draft_year"] < CBA_FIRST_CLASS
    p.loc[pre & p["drafted"], "rookie_pct"] = scale.predict(p.loc[pre & p["drafted"], "overall_pick"])
    p.loc[pre & ~p["drafted"], "rookie_pct"] = udfa_post
    p.loc[pre, "rookie_src"] = "scale_2011+"
    p.loc[pre, "option_pct"] = np.nan
    for dy, g in p[~pre].groupby("draft_year"):
        d = g[g["drafted"] & g["rookie_pct"].notna()]
        miss = g.index[g["drafted"] & g["rookie_pct"].isna()]
        if len(miss):
            iso = (IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
                d["overall_pick"], d["rookie_pct"]) if len(d) >= 20 else scale)
            p.loc[miss, "rookie_pct"] = iso.predict(p.loc[miss, "overall_pick"])
            p.loc[miss, "rookie_src"] = "slot_fit"
        udfa_med = c[(c["year_signed"] == dy) & (c["draft_year"] == dy)
                     & c["draft_round"].isna() & (c["apy"] > 0)]["pct"].median()
        miss_u = g.index[~g["drafted"] & g["rookie_pct"].isna()]
        p.loc[miss_u, "rookie_pct"] = udfa_med
        p.loc[miss_u, "rookie_src"] = "udfa_median"
    return p


def load_snaps() -> pd.DataFrame:
    """Season snap rows (pct = mean over games appeared), 2013-2025."""
    cur = pd.read_parquet(RAW_DIR / "snap_counts.parquet")
    raw = pd.read_parquet(CACHE / "snaps_raw_2012_2015.parquet")
    old = (raw.groupby(["pfr_player_id", "season"])
           .agg(offense_pct=("offense_pct", "mean"), defense_pct=("defense_pct", "mean"),
                st_pct=("st_pct", "mean"), n_games=("game_id", "nunique")).reset_index())
    old = old[old["season"] < cur["season"].min()]
    return pd.concat([old, cur[old.columns]], ignore_index=True)


def dc_features() -> pd.DataFrame:
    """Per (gsis_id, season): share of REG weeks listed 1st string, 2nd string, at all."""
    d = pd.concat([pd.read_parquet(DEPTH_CHART_CACHE),
                   pd.read_parquet(CACHE / "depth_charts_2017_2024.parquet")], ignore_index=True)
    d = d[(d["game_type"] == "REG") & d["formation"].isin(["Offense", "Defense"])].copy()
    d["depth"] = pd.to_numeric(d["depth_team"], errors="coerce")
    n_weeks = d.groupby("season")["week"].nunique()
    w = d.groupby(["gsis_id", "season", "week"])["depth"].min().reset_index()
    w["s1"], w["s2"], w["s_any"] = (w["depth"] == 1), (w["depth"] == 2), True
    f = w.groupby(["gsis_id", "season"])[["s1", "s2", "s_any"]].sum().astype(float)
    f = f.div(f.index.get_level_values("season").map(n_weeks).to_numpy(), axis=0).clip(upper=1.0)
    return f


DC_FEATS = ["s1", "s2", "s_any"]
DC_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                 subsample=0.8, subsample_freq=1, random_state=42, verbose=-1,
                 monotone_constraints=[1, 1, 1])


def depth_chart_model(p, dc, snap_val, qv=None):
    """Fit season value ~ depth-chart role per position on the overlap seasons,
    validate, and return ({(gsis, season): value} for pre-snap seasons plus a
    per-group value for unlisted players, set of (gsis, season) listed)."""
    rows = []
    for r in p.dropna(subset=["gsis"]).itertuples():
        for y in range(r.draft_year, r.draft_year + 4):
            rows.append((r.pfr_id, r.gsis, r.group, r.draft_year, r.round, y))
    T = pd.DataFrame(rows, columns=["pfr_id", "gsis", "group", "draft_year", "round", "season"])
    T = T.join(dc, on=["gsis", "season"])
    T[["s1", "s2", "s_any"]] = T[["s1", "s2", "s_any"]].fillna(0.0)
    feats, params = list(DC_FEATS), dict(DC_PARAMS)
    if qv is not None:
        T["q"] = [qv(gs, g, y) for gs, g, y in zip(T["gsis"], T["group"], T["season"])]
        feats, params["monotone_constraints"] = feats + ["q"], [1, 1, 1, 1]
    fit = T[T["season"].between(*DC_FIT)].copy()
    fit["y"] = [snap_val(a, g, y) for a, g, y in zip(fit["pfr_id"], fit["group"], fit["season"])]

    # Leave-one-season-out validation.
    fit["oof"] = np.nan
    for g, G in fit.groupby("group"):
        for y in sorted(G["season"].unique()):
            tr, te = G[G["season"] != y], G[G["season"] == y]
            m = lgb.LGBMRegressor(**params).fit(tr[feats], tr["y"])
            fit.loc[te.index, "oof"] = m.predict(te[feats])
    print("\nDepth-chart → season value, leave-one-season-out (seasons 2013-24):")
    print(pd.DataFrame({g: {"n": len(G), "rho": spearmanr(G["oof"], G["y"])[0],
                            "mean_y": G["y"].mean(), "mean_pred": G["oof"].mean()}
                        for g, G in fit.groupby("group")}).T.round(4).to_string())
    cls = fit[fit["draft_year"].between(2013, 2021)]
    sums = cls.groupby(["pfr_id", "group", "draft_year", "round"])[["y", "oof"]].sum().reset_index()
    sums = sums[sums.groupby("pfr_id")["pfr_id"].transform("size") > 0]
    rho = np.mean([spearmanr(G["oof"], G["y"])[0] for _, G in sums.groupby(["group", "draft_year"])
                   if len(G) >= 10])
    sums["rd"] = sums["round"].fillna(8).astype(int).clip(upper=8).map(
        lambda x: f"R{x}" if x < 8 else "UDFA")
    print(f"\nClasses 2013-21, y1-y4 value: depth-chart vs snaps, Spearman within (group × class) {rho:.3f}")
    print(sums.groupby("rd")[["y", "oof"]].mean().rename(columns={"y": "snaps", "oof": "depth_chart"})
          .round(4).T.to_string())

    out = {}
    for g, G in fit.groupby("group"):
        m = lgb.LGBMRegressor(**params).fit(G[feats], G["y"])
        P = T[(T["group"] == g) & (T["season"] < SNAP_FIRST)]
        out.update(dict(zip(zip(P["gsis"], P["season"]), m.predict(P[feats]))))
        z = [0.0, 0.0, 0.0] + ([qv(None, g, None)] if qv is not None else [])
        out[("__zero__", g)] = float(m.predict(pd.DataFrame([z], columns=feats))[0])
    listed = set(dc.index[dc["s_any"] > 0])
    return out, listed


def build_label(out_path: Path = LABEL_PATH, quality: str = "pff",
                xval_fold: int | None = None, xval_k: int = 5) -> pd.DataFrame:
    """Build the rookie-deal value / surplus label and write it to out_path.
    quality: "pff" (v4 default since 01e: PFF-grade quality at every
    position), "epa" (v3.1: EPA quality, QB/RB/WR/TE) or "none" (snaps-only price). xval_fold /
    xval_k: cross-fitted validation builds (see experiments/00zq)."""
    ensure_caches()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(xval_fold=xval_fold, xval_k=xval_k)
    qgroups = QUALITY_GROUPS[quality]
    qname, quality = quality, bool(qgroups)
    repl = pd.read_parquet(PROCESSED_DIR / "replacement_levels.parquet")
    repl = repl[repl["scope"] == "position"]
    repl = dict(zip(repl["key"], repl["replacement_apy_pct"].astype(float)))

    bp = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    lab = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    grp = lab.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["value_group"]
    p = (bp.dropna(subset=["pfr_id", "draft_year"]).drop_duplicates("pfr_id")
         [["pfr_id", "player_id", "player_name", "draft_year", "round", "overall_pick", "is_udfa"]]
         .copy())
    p["draft_year"] = p["draft_year"].astype(int)
    p = p[p["draft_year"] >= FIRST_CLASS].reset_index(drop=True)
    p["group"] = p["pfr_id"].map(grp)
    p = p[p["group"].notna() & ~p["group"].isin(["K", "P"])].reset_index(drop=True)
    p["drafted"] = p["round"].notna() & (p["is_udfa"] != True)
    xw = build_bridge_gsis_crosswalk(bp, pd.read_parquet(SEASONAL_ROSTERS_CACHE),
                                     base=_build_pfr_to_gsis_crosswalk())
    p["gsis"] = p["pfr_id"].map(xw)
    log.info("players %d (drafted %d, UDFA %d); gsis linked %.1f%%", len(p), int(p["drafted"].sum()),
             int((~p["drafted"]).sum()), 100 * p["gsis"].notna().mean())

    c = pd.read_parquet(RAW_DIR / "nfl_contracts.parquet")
    for col in ("year_signed", "years", "apy", "apy_cap_pct", "implied_cap", "draft_year", "draft_round"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    cap = cap_by_year(c)
    log.info("cap by year ($M): %s", cap.loc[2014:].round(1).to_dict())
    p = rookie_costs(c, cap, p)
    log.info("rookie cost source: %s", p["rookie_src"].value_counts(dropna=False).to_dict())

    # Price at the NFL position: most common OTC position_group across the
    # player's contracts (ties → latest), else the college value_group.
    cc = c.dropna(subset=["gsis_id", "position_group"])
    cc = (cc.groupby(["gsis_id", "position_group"])["year_signed"].agg(["size", "max"])
          .reset_index().sort_values(["size", "max"]).drop_duplicates("gsis_id", keep="last"))
    nfl_pos = p["gsis"].map(cc.set_index("gsis_id")["position_group"])
    p["draft_group"] = p["group"]
    p["group"] = nfl_pos.where(nfl_pos.isin(PAY_GROUPS), p["group"])
    moved = p["group"] != p["draft_group"]
    log.info("pricing position: %d of %d players moved from college group to NFL position; %s",
             int(moved.sum()), len(p),
             p[moved].groupby(["draft_group", "group"]).size().sort_values(ascending=False).head(8).to_dict())

    pos = (c.dropna(subset=["gsis_id"]).sort_values("year_signed")
           .drop_duplicates("gsis_id", keep="last").set_index("gsis_id")["position_group"])
    pos = pd.concat([p.dropna(subset=["gsis"]).drop_duplicates("gsis").set_index("gsis")["group"],
                     pos]).groupby(level=0).first()
    q_tab, neutral = ({}, {}) if not quality else \
        quality_table_pff(pos) if qname == "pff" else quality_table(pos)

    def qv(gsis, g, y):
        if g not in qgroups:
            return 0.5
        if gsis is not None and not pd.isna(gsis) and (gsis, y) in q_tab:
            return q_tab[(gsis, y)]
        return neutral.get((g, y), neutral[(g, None)])

    pfr_gsis = dict(zip(p["pfr_id"], p["gsis"]))

    s = load_snaps()
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["gfrac"] = (s["n_games"] / s["season"].map(reg_season_games)).clip(upper=1.0)
    s = s.sort_values("share").groupby(["pfr_player_id", "season"]).agg(
        share=("share", "last"), gfrac=("gfrac", "last"), st=("st_pct", "max"))
    share, gfrac, st = s["share"].to_dict(), s["gfrac"].to_dict(), s["st"].to_dict()
    present = set(s.index)
    if True:   # wide pay fit (the 2023-25 "zf" fit of v1/v2 lives in git history, 4a82749)
        ids = pd.read_parquet(PAY_CACHE / "nflverse_ids.parquet")
        pfr2gsis = {**_build_pfr_to_gsis_crosswalk(), **dict(zip(ids["pfr_id"], ids["gsis_id"]))}
        names_path = CACHE / "snap_names_2016_2024.parquet"
        from src.scrapers.nfl_data import _normalize_name
        nm = pd.read_parquet(names_path)
        cn = c.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").assign(n=lambda d: d["player"].map(_normalize_name))
        uniq = cn.groupby("n")["gsis_id"].agg(lambda g: g.iloc[0] if len(g) == 1 else None).dropna()
        by_name = dict(zip(nm["pfr_player_id"], nm["player"].map(_normalize_name).map(uniq)))
        n0 = len(pfr2gsis)
        pfr2gsis = {**{k: v for k, v in by_name.items() if isinstance(v, str)}, **pfr2gsis}
        log.info("pay fit crosswalk: %d by id + %d by unique name", n0, len(pfr2gsis) - n0)
        curves = pay_curves_wide(s, pfr2gsis, c, qv, qgroups, args.xval_fold, args.xval_k)

    def snap_val(pfr, g, y):
        sh = share.get((pfr, y), 0.0)
        if sh > 0 and g in curves:
            return curves[g](sh, qv(pfr_gsis.get(pfr), g, y)) * gfrac[(pfr, y)]
        if st.get((pfr, y), 0.0) >= ST_MIN:
            return float(repl.get(g, np.nan))
        return 0.0

    dc = dc_features()
    dc_val, dc_present = depth_chart_model(p, dc, snap_val, qv if quality else None)

    def val(r, y):
        if y >= SNAP_FIRST:
            return snap_val(r.pfr_id, r.group, y)
        if pd.isna(r.gsis):
            return np.nan
        return dc_val.get((r.gsis, y), dc_val.get(("__zero__", r.group)))

    def is_present(r, y):
        if y >= SNAP_FIRST:
            return (r.pfr_id, y) in present
        return (r.gsis, y) in dc_present

    recs = []
    for r in p.itertuples():
        dy, rec = r.draft_year, {"pfr_id": r.pfr_id}
        for k in range(1, 6):
            y = dy + k - 1
            observed = FIRST_CLASS <= y <= LAST_SEASON
            if k <= 4:
                v = val(r, y) if observed else np.nan
                charged = (r.drafted and r.round <= 2) or is_present(r, y)
                cst = (r.rookie_pct if charged else 0.0) if observed and not pd.isna(v) else np.nan
            else:   # option year (post-CBA classes only)
                if not (r.drafted and r.round == 1):
                    v, cst = 0.0, 0.0
                elif dy < CBA_FIRST_CLASS:
                    v, cst = np.nan, np.nan
                elif pd.isna(r.option_pct):
                    v, cst = (0.0, 0.0) if observed else (np.nan, np.nan)
                else:
                    v = val(r, y) if observed else np.nan
                    cst = r.option_pct if observed else np.nan
            rec[f"value_y{k}"], rec[f"cost_y{k}"] = v, cst
        rec["value_src"] = ("snaps" if dy >= SNAP_FIRST else
                            "depth_chart" if dy + 3 < SNAP_FIRST else "mixed")
        recs.append(rec)
    L = p.merge(pd.DataFrame(recs), on="pfr_id")
    vy = [f"value_y{k}" for k in range(1, 5)]
    cy = [f"cost_y{k}" for k in range(1, 5)]
    L["value_y14"] = L[vy].sum(axis=1, min_count=4)
    L["cost_y14"] = L[cy].sum(axis=1, min_count=4)
    L["surplus_y14"] = L["value_y14"] - L["cost_y14"]
    L["surplus_y15"] = L["surplus_y14"] + L["value_y5"] - L["cost_y5"]
    L.to_parquet(out_path, index=False)
    log.info("wrote %s", out_path)

    print("\nCoverage by class (bridge players with a complete y1-y4 label):")
    print(L.groupby("draft_year").agg(n=("pfr_id", "size"), labelled=("surplus_y14", lambda x: x.notna().mean()),
                                      src=("value_src", "first")).round(3).T.to_string())
    full = L[L["draft_year"].between(FIRST_CLASS, 2022) & L["surplus_y14"].notna()].copy()
    log.info("complete y1-y4: %d players, classes %d-%d", len(full), FIRST_CLASS, 2022)
    era = pd.cut(full["draft_year"], [2005, 2010, 2015, 2022], labels=["2006-10", "2011-15", "2016-22"])
    dd = full[full["drafted"]].assign(era=era, pick_bin=pd.cut(full["overall_pick"],
                                      [0, 10, 20, 32, 64, 100, 160, 262]))
    print("\nMean surplus_y14 by pick and era (stability check):")
    print(dd.pivot_table(index="pick_bin", columns="era", values="surplus_y14", aggfunc="mean",
                         observed=True).round(4).to_string())
    print(dd.pivot_table(index="pick_bin", columns="era", values="value_y14", aggfunc="mean",
                         observed=True).round(4).add_prefix("value ").to_string())
    pd.set_option("display.width", 200)
    d = full[full["drafted"]].copy()
    bins = [0, 10, 20, 32, 48, 64, 80, 100, 130, 160, 200, 262]
    d["pick_bin"] = pd.cut(d["overall_pick"], bins)
    agg = dict(n=("surplus_y14", "size"), value=("value_y14", "mean"), cost=("cost_y14", "mean"),
               surplus=("surplus_y14", "mean"), surplus_med=("surplus_y14", "median"),
               surplus_y15=("surplus_y15", "mean"), p_pos=("surplus_y14", lambda x: (x > 0).mean()))
    print("\nBy overall pick (classes 2006-22, % of cap summed over y1-y4; y15 adds the option year, 2011-21):")
    print(d.groupby("pick_bin", observed=True).agg(**agg).round(4).to_string())
    u = full[~full["drafted"]]
    print(f"\nUDFAs who reached the NFL: n={len(u)}, value {u.value_y14.mean():.4f}, "
          f"cost {u.cost_y14.mean():.4f}, surplus {u.surplus_y14.mean():.4f}, "
          f"P(surplus>0) {(u.surplus_y14 > 0).mean():.2f}")
    print("\nBy position, drafted rounds 1-3 vs 4-7 (mean surplus_y14):")
    print(d.assign(rd=np.where(d["round"] <= 3, "R1-3", "R4-7"))
          .pivot_table(index="group", columns="rd", values="surplus_y14", aggfunc="mean").round(4).to_string())
    print("\nTop 15 surplus_y14, classes 2006-22:")
    print(full.nlargest(15, "surplus_y14")[["player_name", "draft_year", "overall_pick", "group",
                                            "value_y14", "cost_y14", "surplus_y14"]].round(4).to_string(index=False))
    return L
