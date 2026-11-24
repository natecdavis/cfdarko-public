"""
Does the drafting team's roster state change the realized rookie value of a
pick, beyond pick, board rank and position? (pre-registered 2026-11-02)

Why it should: rookie value prices snap share x games, so realized value
depends on whether the team gives the rookie the field. Teams draft for
need, so part of this is already in the pick; the consensus board rank is
need-agnostic, which is why the board-era check matters.

Cohort: drafted players, classes 2013-22 (prior-season snaps from 2012;
four rookie seasons played), K/P excluded; outcome = production rookie
value (rookie_value_labels.parquet, v3.1): value_y14 (primary), y12, y34.

Team features for draftee i (team T, draft year t, position g = label
draft_group, the college/board group known at pick time):
    room   the players on T's week-1 roster of season t (status not CUT /
           practice squad, not rookies of class t) whose position group is g
           (OTC contract position group, else roster depth-chart position)
    depth  Σ prior-season (t−1) snap share of the room (any team; side =
           offense/defense by g) / league-average team snap share at g
           (≈ starters at g): starter-equivalents already in the room
    top    max prior-season snap share in the room (entrenched starter?)
    invest Σ APY % of cap of the room's contracts in force in season t
    prior  Σ slot value (production slot curve) of T's earlier picks at g
           in the same draft (0 if none) — "before each pick"
Known limitation: week-1 rosters are after camp; a veteran cut because the
rookie beat him lowers depth (biases toward finding an effect). Starters
(share >= 0.5) are rarely camp cuts; `top` is the least exposed feature.

Arms — pooled LightGBM, draft_group categorical, leave-one-class-out,
3 seeds (00zp params); monotone −1 on log pick (and log board rank):
    B0       log pick, draft_group
    B0_alt   same, seeds +100 (seed placebo)
    T1       B0 + depth, top, invest, prior
    T1_perm  T1 with the four team features permuted among draftees of the
             same (class × draft_group) (feature placebo)
Board-era check (classes 2014-22, drafted players with a board rank):
    B0h = log pick + log board rank + group;  T1h = B0h + team features.
PRIMARY: MSE of value_y14, T1 vs B0, all classes. "Team context matters"
if P(T1 better) > 0.5 (paired bootstrap of players within class, 1000)
AND T1 beats T1_perm with P > 0.5. Then a pick-time team adjustment
(board_at_pick --team) is worth a pre-registered build on the board-rank
baseline. Reported: the same for y12 / y34 (does the effect fade?), the
board-era check, per-position effect of depth/top (partial dependence:
predicted value at depth quartiles, top < 0.5 vs >= 0.9), by round, and
the QB "stash" case (top >= 0.9).

BUG FIX (2026-11-02, before any results): prior-season snaps first linked
to gsis only via roster pfr ids (~55% coverage, 0% for OL: OL depth was 0
for 98% of OL draftees). Now nflverse players.csv pfr ids (~90%) + roster
pfr ids + a unique (team, name) match against the season's rosters.

Also (same fix, before results): nflverse has no 2012 snap counts (file
empty), so class 2013 has no prior-season snaps; cohort is classes 2014-22.

RESULT (2026-11-02): no usable team effect (pre-registered rule fails).
2,025 drafted players, classes 2014-22. MSE x1e4 gains (> 0 = arm better):
    T1 vs B0      y14 −0.86 [−2.38, +0.60] P 0.13; y12 +0.00 P 0.50;
                  y34 −0.47 P 0.03; board-era T1h vs B0h −0.82 P 0.17
    T1 vs T1perm  y14 +0.50 P 0.71; y12 +0.29 P 0.85 (features carry some
                  signal: the permuted arm loses more, −1.36 / −0.29)
Directions are sensible in the all-class fit (partial dependence, y14):
entrenched starter (top 0.95 vs 0.3) QB 0.175 → 0.127, WR 0.115 → 0.080,
OL 0.109 → 0.066; deeper room (depth q75 vs q25) lower at every position
by 5-15%. The effect is in years 1-2 and mostly gone by years 3-4 (QB y12
0.099 → 0.061, y34 0.072 → 0.066): talent overrides the depth chart. And
the pick already prices much of it: R1 QBs behind an entrenched starter
realized 0.293 vs 0.390, but B0 (pick only) already predicted 0.275 vs
0.375 — teams with an entrenched QB pick later.
Reading: team context is real but small next to the noise in four-year
value, and largely absorbed by draft position; four extra features cost
more variance than they add on ~2,000 players. Not wired.

Rechecked on the v4 label (01e, 2026-11-02): same conclusion, T1 vs B0 y14
gain −1.09e-4, P 0.09 → no usable team effect. v3.1 outputs kept in
_team_roster_state_20261102_v31.

Usage: 01c_team_roster_state.py [--n-boot 1000]
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.models.rookie_value import LABEL_PATH  # noqa: E402

OUT = PROCESSED_DIR / "_team_roster_state_20261102"
TC = RAW_DIR / "team_context"
CLASSES = (2014, 2022)
SEEDS = [42, 43, 44]
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
OFFENSE = {"QB", "RB", "WR", "TE", "OL"}
TEAM_F = ["depth", "top", "invest", "prior"]
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)
CODE = {"GNB": "GB", "KAN": "KC", "NOR": "NO", "NWE": "NE", "SFO": "SF", "TAM": "TB", "LAR": "LA",
        "LVR": "LV", "SDG": "SD", "STL": "STL", "SL": "STL", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
        "HST": "HOU"}
DEPTH_POS = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "OT": "OL",
             "G": "OL", "OG": "OL", "C": "OL", "OL": "OL", "DE": "EDGE", "OLB": "EDGE", "DT": "DL", "NT": "DL",
             "DL": "DL", "ILB": "LB", "MLB": "LB", "LB": "LB", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB",
             "DB": "DB", "SAF": "DB"}
ROSTER_OK = {"ACT", "RES", "PUP", "SUS", "NWT", "EXE", "RSN", "INA"}


def norm_team(s):
    return s.replace(CODE)


def _nm(s):
    return (s.str.lower().str.replace(r"[^a-z ]", "", regex=True)
            .str.replace(r"\b(jr|sr|ii|iii|iv|v)\b", "", regex=True).str.split().str.join(" "))


def season_shares(y):
    """Per gsis: season snap share on offense and defense (missed games = 0;
    denominator = team games x league-average team snaps per game)."""
    s = pd.read_parquet(TC / f"snap_counts_{y}.parquet")
    s = s[s["game_type"] == "REG"]
    out = {}
    for side in ("offense", "defense"):
        g = s[s[f"{side}_pct"] > 0]
        team_snaps = (g[f"{side}_snaps"] / g[f"{side}_pct"]).groupby([g["game_id"], g["team"]]).median()
        n_games = s.groupby("team")["game_id"].nunique().median()
        denom = n_games * team_snaps.mean()
        out[side] = s.groupby("pfr_player_id")[f"{side}_snaps"].sum() / denom
    return pd.DataFrame(out)


def snap_gsis(y, p2g):
    """pfr_player_id → gsis for season y's snap counts: nflverse players.csv /
    roster pfr ids, then a unique (team, normalized name) match against that
    season's rosters (OL rows carry no pfr id in nflverse rosters)."""
    s = pd.read_parquet(TC / f"snap_counts_{y}.parquet", columns=["pfr_player_id", "player", "team"])
    s = s.drop_duplicates(["pfr_player_id", "team"])
    g = s["pfr_player_id"].map(p2g)
    r = pd.read_parquet(TC / f"roster_weekly_{y}.parquet", columns=["team", "full_name", "gsis_id"]).dropna()
    r = r.assign(team=norm_team(r["team"]), nm=_nm(r["full_name"])).drop_duplicates(["team", "nm", "gsis_id"])
    r = r[~r.duplicated(["team", "nm"], keep=False)].set_index(["team", "nm"])["gsis_id"]
    key = pd.MultiIndex.from_arrays([norm_team(s["team"]), _nm(s["player"])])
    g = g.fillna(pd.Series(r.reindex(key).to_numpy(), index=s.index))
    m = pd.Series(g.to_numpy(), index=s["pfr_player_id"].to_numpy())
    return m[m.notna()].groupby(level=0).first()


def pfr_to_gsis():
    pl = pd.read_csv(RAW_DIR / "rookie_value" / "nflverse_players.csv", low_memory=False,
                     usecols=["gsis_id", "pfr_id"]).dropna()
    m = [pl]
    for y in range(2012, 2026):
        r = pd.read_parquet(TC / f"roster_weekly_{y}.parquet", columns=["pfr_id", "gsis_id"])
        m.append(r.dropna())
    m = pd.concat(m).drop_duplicates("pfr_id")
    return m.set_index("pfr_id")["gsis_id"]


def position_groups():
    c = pd.read_parquet(RAW_DIR / "nfl_contracts.parquet").dropna(subset=["gsis_id", "position_group"])
    return (c.sort_values("year_signed").groupby("gsis_id")["position_group"]
            .agg(lambda s: s.mode().iloc[-1] if len(s.mode()) else s.iloc[-1]))


def contracts_in_force():
    c = pd.read_parquet(RAW_DIR / "nfl_contracts.parquet").dropna(subset=["gsis_id", "year_signed", "years"])
    c = c[c["year_signed"] > 0]
    return c[["gsis_id", "year_signed", "years", "apy_cap_pct"]]


def team_features(dr):
    p2g, otc, con = pfr_to_gsis(), position_groups(), contracts_in_force()
    slot = pd.read_parquet(PROCESSED_DIR / "slot_curve.parquet")
    slot = slot[slot["pick"] != "UDFA"].assign(pick=lambda d: d["pick"].astype(int)).set_index("pick")["value"]
    rows = []
    for t in range(CLASSES[0], CLASSES[1] + 1):
        sh = season_shares(t - 1)
        sg = snap_gsis(t - 1, p2g)
        n_all = len(sh)
        sh.index = sh.index.map(sg)
        print(f"  {t}: prior-season snap rows linked to gsis {sh.index.notna().sum()}/{n_all}")
        sh = sh[sh.index.notna()].groupby(level=0).sum()
        r = pd.read_parquet(TC / f"roster_weekly_{t}.parquet")
        r = r[(r["week"] == 1) & (r["game_type"] == "REG") & r["status"].isin(ROSTER_OK)
              & (r["rookie_year"].fillna(0) < t)].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
        r["team"] = norm_team(r["team"])
        r["g"] = r["gsis_id"].map(otc).fillna(r["depth_chart_position"].map(DEPTH_POS))
        r = r[r["g"].isin(GROUPS)]
        side = np.where(r["g"].isin(OFFENSE), "offense", "defense")
        r["share"] = [sh.at[g, s] if g in sh.index else 0.0 for g, s in zip(r["gsis_id"], side)]
        cc = con[(con["year_signed"] <= t) & (con["year_signed"] + con["years"] - 1 >= t)]
        cc = cc.sort_values("year_signed").drop_duplicates("gsis_id", keep="last").set_index("gsis_id")["apy_cap_pct"]
        r["apy"] = r["gsis_id"].map(cc).fillna(0.0)
        room = r.groupby(["team", "g"]).agg(sum_share=("share", "sum"), top=("share", "max"), invest=("apy", "sum"))
        starters = room["sum_share"].groupby(level="g").mean()
        room["depth"] = room["sum_share"] / room.index.get_level_values("g").map(starters).to_numpy()
        d = dr[dr["draft_year"] == t].copy()
        idx = pd.MultiIndex.from_arrays([d["team"], d["draft_group"]])
        for f in ("depth", "top", "invest"):
            d[f] = room[f].reindex(idx).fillna(0.0).to_numpy()
        d = d.sort_values("overall_pick")
        sv = d["overall_pick"].round().astype(int).clip(1, 262).map(slot)
        d["prior"] = sv.groupby([d["team"], d["draft_group"]]).transform(lambda s: s.cumsum().shift(fill_value=0.0))
        rows.append(d)
    return pd.concat(rows)


def build():
    path = OUT / "cohort.parquet"
    if path.exists():
        return pd.read_parquet(path)
    L = pd.read_parquet(LABEL_PATH)
    L = L[L["drafted"] & L["draft_year"].between(*CLASSES) & L["gsis"].notna()
          & L["draft_group"].isin(GROUPS) & L["value_y14"].notna()].drop_duplicates("gsis").copy()
    P = pd.read_parquet(TC / "draft_picks.parquet").dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    L["team"] = norm_team(L["gsis"].map(P.set_index("gsis_id")["team"]))
    miss = L["team"].isna()
    print(f"cohort {len(L)}; no team from nflverse draft_picks: {int(miss.sum())} (dropped)")
    L = L[~miss]
    from src.models.offboard_model import pid
    proj = pd.read_parquet(ROOT / "data" / "output" / "projections" / "latest.parquet",
                           columns=["player_id", "hasan_rank"]).dropna()
    proj = proj.assign(pid=pid(proj["player_id"])).drop_duplicates("pid").set_index("pid")["hasan_rank"]
    L["hasan_rank"] = pd.Series(pid(L["player_id"]), index=L.index).map(proj)
    d = team_features(L)
    d["y14"] = d["value_y14"]
    d["y12"] = d[["value_y1", "value_y2"]].sum(axis=1)
    d["y34"] = d[["value_y3", "value_y4"]].sum(axis=1)
    d["lp"] = np.log(d["overall_pick"])
    d["lr"] = np.log(d["hasan_rank"])
    OUT.mkdir(parents=True, exist_ok=True)
    d.reset_index(drop=True).to_parquet(path, index=False)
    return d.reset_index(drop=True)


def oof(df, feats, mono, seeds, y):
    X = df[feats + ["draft_group"]].copy()
    X["draft_group"] = pd.Categorical(X["draft_group"], categories=GROUPS)
    pred = np.full(len(df), np.nan)
    for cls in sorted(df["draft_year"].unique()):
        te = (df["draft_year"] == cls).to_numpy()
        ps = [lgb.LGBMRegressor(**PARAMS, random_state=s, monotone_constraints=mono + [0])
              .fit(X[~te], df.loc[~te, y]).predict(X[te]) for s in seeds]
        pred[te] = np.mean(ps, axis=0)
    return pred


def permute(df, rng):
    d = df.copy()
    for _, ix in d.groupby(["draft_year", "draft_group"]).groups.items():
        ix = np.asarray(ix)
        d.loc[ix, TEAM_F] = d.loc[rng.permutation(ix), TEAM_F].to_numpy()
    return d


def compare(df, y, arms, pairs, n_boot, rng):
    se = {a: (df[a] - df[y]) ** 2 for a in arms}
    groups = [g.index.to_numpy() for _, g in df.groupby("draft_year")]
    idxs = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(n_boot)]
    rows = []
    for a, b in pairs:
        diff = (se[b] - se[a]).to_numpy()
        bs = np.array([diff[i].mean() for i in idxs])
        rows.append({"outcome": y, "arm vs base": f"{a} vs {b}", "mse_base_x1e4": 1e4 * se[b].mean(),
                     "gain_x1e4": 1e4 * diff.mean(), "lo95": 1e4 * np.percentile(bs, 2.5),
                     "hi95": 1e4 * np.percentile(bs, 97.5), "P(arm better)": (bs > 0).mean()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(0)
    df = build()
    print(f"cohort {len(df)} drafted players, classes {CLASSES}")
    print("team features by draft group (mean):")
    print(df.groupby("draft_group")[TEAM_F + ["y14"]].mean().round(3).to_string())
    perm = permute(df, np.random.default_rng(1))
    rows = []
    for y in ("y14", "y12", "y34"):
        base_f, tf = ["lp"], ["lp"] + TEAM_F
        mono_b, mono_t = [-1], [-1, 0, 0, 0, 0]
        df[f"B0_{y}"] = oof(df, base_f, mono_b, SEEDS, y)
        df[f"B0alt_{y}"] = oof(df, base_f, mono_b, [s + 100 for s in SEEDS], y)
        df[f"T1_{y}"] = oof(df, tf, mono_t, SEEDS, y)
        df[f"T1perm_{y}"] = oof(perm.assign(**{y: df[y]}), tf, mono_t, SEEDS, y)
        rows += compare(df, y, [f"{a}_{y}" for a in ("B0", "B0alt", "T1", "T1perm")],
                        [(f"T1_{y}", f"B0_{y}"), (f"T1_{y}", f"T1perm_{y}"), (f"T1perm_{y}", f"B0_{y}"),
                         (f"B0alt_{y}", f"B0_{y}")], args.n_boot, rng)
    hb = df[(df["draft_year"] >= 2014) & df["hasan_rank"].notna()].reset_index(drop=True)
    hb["B0h"] = oof(hb, ["lp", "lr"], [-1, -1], SEEDS, "y14")
    hb["T1h"] = oof(hb, ["lp", "lr"] + TEAM_F, [-1, -1, 0, 0, 0, 0], SEEDS, "y14")
    rows += compare(hb, "y14", ["B0h", "T1h"], [("T1h", "B0h")], args.n_boot, rng)
    D = pd.DataFrame(rows)
    print("\nMSE gains (x1e4, > 0 = arm better):")
    print(D.round(4).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)
    df.to_parquet(OUT / "oof.parquet", index=False)
    p = D[D["arm vs base"] == "T1_y14 vs B0_y14"].iloc[0]
    q = D[D["arm vs base"] == "T1_y14 vs T1perm_y14"].iloc[0]
    ok = p["P(arm better)"] > 0.5 and q["P(arm better)"] > 0.5
    print(f"\nDECISION: T1 vs B0 y14 gain {p['gain_x1e4']:+.3f}e-4 [{p['lo95']:+.3f}, {p['hi95']:+.3f}] "
          f"P {p['P(arm better)']:.3f}; vs permuted P {q['P(arm better)']:.3f} → "
          f"{'TEAM CONTEXT MATTERS' if ok else 'no usable team effect'}")

    # effect sizes: refit T1 on all classes, partial dependence by group
    X = df[["lp"] + TEAM_F + ["draft_group"]].copy()
    X["draft_group"] = pd.Categorical(X["draft_group"], categories=GROUPS)
    print("\npartial dependence (mean predicted value_y14, % of cap; all-class fit, 3 seeds):")
    for y in ("y14", "y12", "y34"):
        ms = [lgb.LGBMRegressor(**PARAMS, random_state=s, monotone_constraints=[-1, 0, 0, 0, 0, 0])
              .fit(X, df[y]) for s in SEEDS]
        def pdv(f, v, sub):
            Z = X.loc[sub].copy()
            Z[f] = v
            return np.mean([m.predict(Z).mean() for m in ms])
        res = {}
        for g in GROUPS:
            sub = df["draft_group"] == g
            q25, q75 = df.loc[sub, "depth"].quantile([0.25, 0.75])
            res[g] = {"n": int(sub.sum()), "depth_q25": pdv("depth", q25, sub), "depth_q75": pdv("depth", q75, sub),
                      "top_0.3": pdv("top", 0.3, sub), "top_0.95": pdv("top", 0.95, sub),
                      "prior_0": pdv("prior", 0.0, sub), "prior_0.15": pdv("prior", 0.15, sub)}
        print(f"  outcome {y}:")
        print(pd.DataFrame(res).T.round(4).to_string())
    df["rd"] = np.select([df["round"] == 1, df["round"] <= 3], ["R1", "R2-3"], "R4-7")
    print("\nMSE x1e4 by round, y14:")
    print(df.groupby("rd").apply(lambda g: pd.Series({a: 1e4 * ((g[f"{a}_y14"] - g["y14"]) ** 2).mean()
                                                      for a in ("B0", "T1", "T1perm")}), include_groups=False)
          .round(3).to_string())
    qb = df[df["draft_group"] == "QB"]
    print("\nQBs by incumbent starter (top >= 0.9 = entrenched): realized y12 / y34 / y14 and B0 / T1 predictions")
    print(qb.assign(entrenched=qb["top"] >= 0.9).groupby(["entrenched", "rd"])[
        ["y12", "y34", "y14", "B0_y14", "T1_y14"]].agg(["mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
