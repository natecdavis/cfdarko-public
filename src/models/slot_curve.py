"""Draft-slot surplus curve and per-player excess (production home of
experiments/00zr_slot_curve.py; method, validation and results there).

slot curve(pick) = E[value_y14 | pick] − E[cost_y14 | pick]; excess =
surplus − slot. Value: log-link Poisson GLM on a cubic in log pick, made
non-increasing by isotonic regression; cost: per-pick mean (00zz: both beat
the local-linear kNN out of sample, mostly at the top of round 1). The kNN
smoother is kept for the direct-surplus diagnostic.
Writes slot_curve.parquet (pick 1..262 + "UDFA") and rookie_excess.parquet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor

from config import PROCESSED_DIR, RAW_DIR, END_YEAR
from src.models.rookie_value import LABEL_PATH

log = logging.getLogger(__name__)
CLASSES = (2006, END_YEAR - 4)        # y1-y4 complete (last played season END_YEAR-1)
CLASSES_Y15 = (2011, END_YEAR - 5)    # + option year; 2011 = first CBA rookie scale
PICKS = np.arange(1, 263)
K_GRID = [30, 50, 80, 120, 180, 260, 380]
K_TOL = 0.005   # pick the smoothest k within 0.5% of the best LOCO error
SPECIALISTS = {"Place Kicker", "Punter", "Long Snapper", "K", "P", "LS"}


def kernel_smooth(x, y, grid, k):
    """Local-linear regression in log(pick) with a nearest-neighbour tricube
    window holding k players: wide where picks are sparse in log space (the
    top of round 1), so no point rests on one pick's 17 players."""
    lx, out = np.log(np.asarray(x, float)), np.empty(len(grid))
    y = np.asarray(y, float)
    for i, g in enumerate(np.log(np.asarray(grid, float))):
        d = np.abs(lx - g)
        hk = np.partition(d, min(k, len(d) - 1))[min(k, len(d) - 1)] * 1.0001 + 1e-9
        w = np.clip(1 - (d / hk) ** 3, 0, None) ** 3
        X = np.column_stack([np.ones_like(lx), lx - g])
        WX = X * w[:, None]
        beta = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None)[0]
        out[i] = beta[0]
    return out


def pick_bandwidth(d, col):
    err = {}
    for h in K_GRID:
        e = []
        for c in d["draft_year"].unique():
            tr, te = d[d["draft_year"] != c], d[d["draft_year"] == c]
            pred = kernel_smooth(tr["overall_pick"].to_numpy(), tr[col].to_numpy(),
                                 te["overall_pick"].to_numpy(), h)
            e.append(((te[col].to_numpy() - pred) ** 2).sum())
        err[h] = np.sum(e) / len(d)
    best = min(err.values())
    return max(k for k, e in err.items() if e <= best * (1 + K_TOL)), err


def _logpoly(p):
    lp = np.log(np.asarray(p, float))
    return np.column_stack([lp, lp ** 2, lp ** 3])


def value_glm(x, y):
    """E[value | pick]: log-link Poisson GLM on a cubic in log pick, then
    isotonic non-increasing."""
    m = PoissonRegressor(alpha=0, max_iter=1000).fit(_logpoly(x), np.clip(y, 0, None))
    return IsotonicRegression(increasing=False).fit_transform(PICKS, m.predict(_logpoly(PICKS)))


def per_pick_mean(x, y):
    """Mean by overall pick (rookie cost is set by the slot); gaps interpolated."""
    s = pd.Series(np.asarray(y, float)).groupby(np.round(np.asarray(x, float)).astype(int)).mean()
    return s.reindex(PICKS).interpolate(limit_direction="both").to_numpy()


def curve(d, h, ycol="surplus_y14", vcol="value_y14", ccol="cost_y14"):
    x = d["overall_pick"].to_numpy()
    v = value_glm(x, d[vcol].to_numpy())
    c = per_pick_mean(x, d[ccol].to_numpy())
    direct = kernel_smooth(x, d[ycol].to_numpy(), PICKS, h)
    return v, c, v - c, direct


def boot_curves(d, h, n_boot, rng, **kw):
    classes = d["draft_year"].unique()
    by = {c: d[d["draft_year"] == c] for c in classes}
    out = []
    for _ in range(n_boot):
        s = pd.concat([by[c] for c in rng.choice(classes, len(classes))])
        out.append(curve(s, h, **kw)[2])
    return np.array(out)


def build_slot_curve(label_path: Path = LABEL_PATH, out_dir: Path = PROCESSED_DIR,
                     n_boot: int = 500) -> pd.DataFrame:
    """Write slot_curve.parquet and rookie_excess.parquet to out_dir; return
    the slot curve."""
    out_dir = Path(out_dir)
    D = out_dir
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 200)

    L = pd.read_parquet(label_path)
    dr = L[L["drafted"] & L["draft_year"].between(*CLASSES) & L["surplus_y14"].notna()
           & L["overall_pick"].notna()].copy()
    print(f"drafted, labelled: {len(dr)} (classes {CLASSES})")

    h, err = pick_bandwidth(dr, "value_y14")
    print("neighbours k: LOCO squared error on value (x1e4):",
          {k: round(v * 1e4, 4) for k, v in err.items()}, "→", h)

    v, c, s, direct = curve(dr, h)
    B = boot_curves(dr, h, n_boot, rng)
    lo, hi = np.percentile(B, 5, axis=0), np.percentile(B, 95, axis=0)

    d15 = dr[dr["draft_year"].between(*CLASSES_Y15) & dr["surplus_y15"].notna()].copy()
    d15["value_y15"] = d15["value_y14"] + d15["value_y5"].fillna(0)
    d15["cost_y15"] = d15["cost_y14"] + d15["cost_y5"].fillna(0)
    s15 = curve(d15, h, "surplus_y15", "value_y15", "cost_y15")[2]

    C = pd.DataFrame({"pick": PICKS, "value": v, "cost": c, "surplus": s, "surplus_lo90": lo,
                      "surplus_hi90": hi, "surplus_direct": direct, "surplus_y15": s15})
    und = L[~L["drafted"] & L["draft_year"].between(*CLASSES) & L["surplus_y14"].notna()]
    und15 = L[~L["drafted"] & L["draft_year"].between(*CLASSES_Y15) & L["surplus_y15"].notna()]
    C["pick"] = C["pick"].astype(object)
    C.loc[len(C)] = {"pick": "UDFA", "value": und["value_y14"].mean(), "cost": und["cost_y14"].mean(),
                     "surplus": und["surplus_y14"].mean(), "surplus_y15": und15["surplus_y15"].mean()}
    C["pick"] = C["pick"].astype(str)
    C.to_parquet(D / "slot_curve.parquet", index=False)

    show = [1, 3, 5, 10, 15, 20, 25, 32, 40, 48, 56, 64, 80, 100, 130, 160, 200, 250]
    print("\nslot curve (% of cap over y1-y4; y15 adds the option year):")
    print(C[C["pick"].isin([str(x) for x in show]) | (C["pick"] == "UDFA")].round(4).to_string(index=False))
    peak = int(C.loc[:261, "surplus"].astype(float).idxmax()) + 1
    print(f"peak expected surplus at pick {peak}; "
          f"bootstrap peak picks 5/50/95%: {np.percentile(B.argmax(1) + 1, [5, 50, 95]).round(0)}")

    # late-R1 dip
    def band(dd, a, b):
        return dd.loc[dd["overall_pick"].between(a, b), "surplus_y14"].mean()
    classes = dr["draft_year"].unique()
    by = {k: dr[dr["draft_year"] == k] for k in classes}
    dips = []
    for _ in range(n_boot):
        ss = pd.concat([by[k] for k in rng.choice(classes, len(classes))])
        dips.append(band(ss, 21, 32) - band(ss, 33, 48))
    dips = np.array(dips)
    print(f"\nlate-R1 dip: mean surplus picks 21-32 − 33-48 = {band(dr, 21, 32) - band(dr, 33, 48):+.4f} "
          f"[{np.percentile(dips, 5):+.4f}, {np.percentile(dips, 95):+.4f}] 90%, P(<0) {(dips < 0).mean():.2f}")
    dec = []
    for _ in range(n_boot):
        ss = pd.concat([by[k] for k in rng.choice(classes, len(classes))])
        dec.append(band(ss, 1, 10) - band(ss, 11, 20))
    dec = np.array(dec)
    print(f"top-10 vs 11-20: {band(dr, 1, 10) - band(dr, 11, 20):+.4f} "
          f"[{np.percentile(dec, 5):+.4f}, {np.percentile(dec, 95):+.4f}] 90%, P(<0) {(dec < 0).mean():.2f}")

    # era stability
    eras = {"2006-10": (2006, 2010), "2011-15": (2011, 2015), "2016-22": (2016, 2022)}
    E = pd.DataFrame({e: curve(dr[dr["draft_year"].between(*r)], h)[2] for e, r in eras.items()},
                     index=PICKS)
    print("\nslot curve by era (surplus):")
    print(E.loc[show].round(4).to_string())

    # sensitivity: unlabelled non-specialist picks as surplus 0
    dp = pd.read_parquet(RAW_DIR / "draft_picks.parquet")
    dp = dp[dp["draft_year"].between(*CLASSES) & ~dp["position"].isin(SPECIALISTS)]
    have = set(zip(dr["draft_year"], dr["overall_pick"].astype(int)))
    miss = dp[[(a, int(b)) not in have for a, b in zip(dp["draft_year"], dp["overall_pick"])]]
    print(f"\nsensitivity: {len(miss)} unlabelled non-specialist picks "
          f"(by round {miss['round'].value_counts().sort_index().to_dict()}) set to value 0, cost 0")
    fill = pd.DataFrame({"draft_year": miss["draft_year"], "overall_pick": miss["overall_pick"],
                         "value_y14": 0.0, "cost_y14": 0.0, "surplus_y14": 0.0})
    s_fill = curve(pd.concat([dr, fill]), h)[2]
    print(pd.DataFrame({"base": s, "missing=0": s_fill}, index=PICKS).loc[show].round(4).T.to_string())

    # excess label
    slot = dict(zip(PICKS, s))
    slot15 = dict(zip(PICKS, s15))
    X = L[L["draft_year"].between(*CLASSES)].copy()
    pk = X["overall_pick"].round().astype("Int64")
    X["slot_y14"] = np.where(X["drafted"], pk.map(slot), C.iloc[-1]["surplus"])
    X["excess_y14"] = X["surplus_y14"] - X["slot_y14"]
    X["slot_y15"] = np.where(X["drafted"], pk.map(slot15), C.iloc[-1]["surplus_y15"])
    X["excess_y15"] = X["surplus_y15"] - X["slot_y15"]
    keep = ["pfr_id", "player_id", "player_name", "draft_year", "round", "overall_pick", "drafted",
            "group", "value_y14", "cost_y14", "surplus_y14", "slot_y14", "excess_y14",
            "surplus_y15", "slot_y15", "excess_y15"]
    X[keep].to_parquet(D / "rookie_excess.parquet", index=False)

    xd = X[X["drafted"] & X["excess_y14"].notna()].copy()
    xd["rd"] = np.select([xd["round"] == 1, xd["round"] <= 3], ["R1", "R2-3"], "R4-7")
    print("\nrealized excess by position × round (mean excess_y14, % of cap; n in brackets):")
    t = xd.pivot_table(index="group", columns="rd", values="excess_y14", aggfunc="mean")
    n = xd.pivot_table(index="group", columns="rd", values="excess_y14", aggfunc="size")
    print((t.round(4).astype(str) + " [" + n.astype(str) + "]").to_string())
    allp = xd.groupby("group")["excess_y14"].agg(["mean", "median", "size"])
    allp["P(>0)"] = xd.groupby("group")["excess_y14"].apply(lambda e: (e > 0).mean())
    print("\nall rounds:")
    print(allp.round(4).to_string())
    print("\ntop 10 / bottom 10 realized excess_y14:")
    cols = ["player_name", "draft_year", "overall_pick", "group", "surplus_y14", "slot_y14", "excess_y14"]
    print(xd.nlargest(10, "excess_y14")[cols].round(4).to_string(index=False))
    print(xd.nsmallest(10, "excess_y14")[cols].round(4).to_string(index=False))
    return C
