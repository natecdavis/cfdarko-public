"""
Does the board's rank → rookie value mapping drift over time? (diagnostics,
2026-11-01)

User question: as teams re-learn positional value and the league changes,
the value of "rank 20 at RB" should change, so a board curve pooled over
classes may be stale. Within-position board rank might be more stable.

Three diagnostics (board players only, K/P/OTHER excluded):
 1. Composition: each position's share of the board's top 32 / top 100 /
    top 300 by draft class, 2015-26 (full 300-player boards), with a
    linear trend per position.
 2. Residual drift: realized rookie value (value_y14 v3.1, 0 if never NFL)
    minus the pooled board curve (production recipe: shrunk isotonic in log
    overall rank by position, fit on all classes 2015-22). Per position, the
    slope of the class-mean residual on draft year (class-cluster bootstrap)
    and the early (2015-18) vs late (2019-22) mean residual.
 3. Out-of-time: fit on early classes, predict later ones. Board curves on
       A  overall rank          (production: shrunk isotonic, log rank)
       B  within-position rank  (shrunk isotonic, log within-pos rank)
       C  both                  (monotone LightGBM on log overall rank, log
                                 within-pos rank, position; plus A-LGB, the
                                 same LightGBM on overall rank only, so C's
                                 comparison is like-for-like)
    Splits: (i) train 2015-18 → test 2019-22; (ii) lagged, as in real use
    (a class's outcome settles ~4 years later): test class t trained on
    classes <= t-4, t = 2020-22.
    Metrics per test class, then averaged: value captured by each list's
    top 32 / top 100 among board players (cross-position); cross-position
    Spearman within class; MSE; mean predicted − realized by position (level
    bias, what drift would show). Paired player bootstrap within class
    (1000): P(arm better than A).

RESULT (2026-11-01):
 1. Boards did re-weight positions, modestly. Top-100 share 2015-18 →
    2023-26: DL 15% → 10%, EDGE 9% → 14%, WR 13% → 16%, RB 9% → 6%,
    OL 16% → 18%; top 32: LB 10% → 2%, WR 11% → 17%. Much of DL → EDGE is
    relabelling (college position groups): DL + EDGE in the top 300 is flat
    (~50-68 per class).
 2. No position's residual (realized − pooled curve) has a clear trend
    over 2015-22 (8 classes). Hints: LB, DL, EDGE drift down (P(slope>0)
    0.09-0.13), DB up (0.90). QB sits above the curve in 6 of 8 classes
    (the pooled shrinkage under-prices QBs; see 00zs).
 3. Out-of-time, within-position rank is clearly WORSE than overall rank
    (fixed split: top100 −0.006 [−0.010, −0.002], rho −0.053, MSE worse;
    lagged split the same). Both ranks together tie overall rank alone
    (C − A_lgb: top100 +0.0005 / −0.0011, rho ±0.001). Overall rank carries
    class-depth information within-position rank throws away, and it is
    no less stable. Level bias out-of-time: the production curve
    under-predicts QB by ~0.019 (−17%) and over-predicts LB (+0.011-0.013,
    ~30%) and DL (+0.006-0.011) — the positions boards and the league moved
    away from.
 Reading: keep overall rank. The drift that shows is in position LEVELS
 (LB/DL down, QB under-priced), not in the rank scale; partial outcomes
 for recent classes are the natural way to track it.

No decision rule: diagnostics to decide whether a drift-robust curve is
worth building. Usage: 00zt_board_drift.py [--n-boot 1000]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from src.models.merged_board import ShrunkIso, K_BOARD  # noqa: E402

spec = importlib.util.spec_from_file_location("zs", ROOT / "experiments/00zs_value_currency_board.py")
zs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zs)

OUT = PROCESSED_DIR / "_board_drift_20261101"
POS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
           subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)


def composition():
    P = pd.read_parquet(OUTPUT_DIR / "projections" / "latest.parquet",
                        columns=["player_id", "position_group", "hasan_rank", "last_college_season_year"])
    b = P[P["hasan_rank"].notna() & P["position_group"].isin(POS)].copy()
    b["cls"] = b["last_college_season_year"].astype(int) + 1
    b = b[b["cls"].between(2015, 2026)]
    rows = []
    for k in (32, 100, 300):
        t = (b[b["hasan_rank"] <= k].groupby(["cls", "position_group"]).size()
             .unstack(fill_value=0).reindex(columns=POS, fill_value=0))
        sh = t.div(t.sum(1), axis=0)
        print(f"\nboard top {k}: players per class by position")
        print(t.T.to_string())
        for g in POS:
            slope = np.polyfit(sh.index, sh[g], 1)[0]
            rows.append({"top": k, "pos": g, "share_2015_18": sh.loc[2015:2018, g].mean(),
                         "share_2023_26": sh.loc[2023:2026, g].mean(), "slope_per_yr": slope})
    T = pd.DataFrame(rows)
    print("\nshare of board slots, early vs recent classes, and linear trend (share per year):")
    print(T.pivot(index="pos", columns="top", values=["share_2015_18", "share_2023_26", "slope_per_yr"])
          .round(4).to_string())
    return T


def frame():
    E = zs.load()
    b = E[E["board"] & E["pos_group"].isin(POS)].copy()
    b = b[b["cls"].between(2015, 2022)].reset_index(drop=True)
    b["lr"] = np.log(b["hasan_rank"])
    b["wrank"] = b.groupby(["cls", "pos_group"])["hasan_rank"].rank(method="first")
    b["lw"] = np.log(b["wrank"])
    return b


def residual_drift(b, n_boot, rng):
    cal = ShrunkIso(increasing=False, k=K_BOARD).fit(b["lr"], b["rv"], b["pos_group"])
    b = b.assign(res=b["rv"] - cal.predict(b["lr"], b["pos_group"]))
    m = b.groupby(["pos_group", "cls"])["res"].mean().unstack()
    print("\nmean residual (realized − pooled curve), rookie value % of cap, by class:")
    print(m.round(4).to_string())
    rows = []
    classes = np.array(sorted(b["cls"].unique()))
    for g in POS:
        G = b[b["pos_group"] == g]
        by = {c: G[G["cls"] == c] for c in classes}
        def slope(d):
            mm = d.groupby("cls")["res"].mean()
            return np.polyfit(mm.index, mm.values, 1)[0] if len(mm) > 2 else np.nan
        bs = np.array([slope(pd.concat([by[c] for c in rng.choice(classes, len(classes))]))
                       for _ in range(n_boot)])
        rows.append({"pos": g, "n": len(G), "slope_per_yr": slope(G),
                     "lo90": np.nanpercentile(bs, 5), "hi90": np.nanpercentile(bs, 95),
                     "P(slope>0)": np.nanmean(bs > 0),
                     "early_2015_18": G.loc[G["cls"] <= 2018, "res"].mean(),
                     "late_2019_22": G.loc[G["cls"] >= 2019, "res"].mean()})
    print("\nresidual trend by position (class-cluster bootstrap, 90%):")
    print(pd.DataFrame(rows).set_index("pos").round(4).to_string())


def fit_predict(tr, te):
    out = {}
    a = ShrunkIso(increasing=False, k=K_BOARD).fit(tr["lr"], tr["rv"], tr["pos_group"])
    out["A_overall"] = a.predict(te["lr"], te["pos_group"])
    bb = ShrunkIso(increasing=False, k=K_BOARD).fit(tr["lw"], tr["rv"], tr["pos_group"])
    out["B_within"] = bb.predict(te["lw"], te["pos_group"])
    for name, feats in (("A_lgb", ["lr"]), ("C_both", ["lr", "lw"])):
        X = tr[feats + ["pos_group"]].assign(pos_group=lambda d: d["pos_group"].astype("category"))
        Xt = te[feats + ["pos_group"]].assign(
            pos_group=lambda d: pd.Categorical(d["pos_group"], categories=X["pos_group"].cat.categories))
        mdl = lgb.LGBMRegressor(**LGB, monotone_constraints=[-1] * len(feats) + [0]).fit(X, tr["rv"])
        out[name] = mdl.predict(Xt)
    return out


ARMS = ["A_overall", "B_within", "A_lgb", "C_both"]


def metrics(V, idx=None):
    d = V if idx is None else V.iloc[idx]
    r = {}
    for a in ARMS:
        by = list(d.groupby("cls"))
        r[a] = {"top32": np.mean([g.nlargest(32, a)["rv"].mean() for _, g in by]),
                "top100": np.mean([g.nlargest(100, a)["rv"].mean() for _, g in by]),
                "rho": np.mean([spearmanr(g[a], g["rv"])[0] for _, g in by]),
                "neg_mse": -np.mean((d[a] - d["rv"]) ** 2)}
    return r


def out_of_time(b, n_boot, rng):
    splits = {"fixed 2015-18 → 2019-22": [(range(2015, 2019), c) for c in range(2019, 2023)],
              "lagged (train ≤ t−4), t = 2020-22": [(range(2015, t - 3), t) for t in range(2020, 2023)]}
    for name, sp in splits.items():
        parts = []
        for tr_cls, t in sp:
            tr, te = b[b["cls"].isin(tr_cls)], b[b["cls"] == t].copy()
            for a, p in fit_predict(tr, te).items():
                te[a] = p
            parts.append(te)
        V = pd.concat(parts, ignore_index=True)
        pt = metrics(V)
        groups = [g.index.to_numpy() for _, g in V.groupby("cls")]
        boots = [metrics(V, np.concatenate([rng.choice(g, len(g)) for g in groups])) for _ in range(n_boot)]
        print(f"\n=== out-of-time: {name} ({len(V)} board players) ===")
        print(pd.DataFrame(pt).T.round(4).to_string())
        rows = []
        for a, base in (("B_within", "A_overall"), ("C_both", "A_lgb"), ("C_both", "A_overall"),
                        ("A_lgb", "A_overall")):
            for k in pt[a]:
                ds = np.array([bt[a][k] - bt[base][k] for bt in boots])
                rows.append({"Δ": f"{a} − {base}", "metric": k, "delta": pt[a][k] - pt[base][k],
                             "lo95": np.percentile(ds, 2.5), "hi95": np.percentile(ds, 97.5),
                             "P(>0)": (ds > 0).mean()})
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        bias = V.groupby("pos_group").apply(lambda g: pd.Series(
            {"n": len(g), "actual": g["rv"].mean(), **{a: g[a].mean() - g["rv"].mean() for a in ARMS}}))
        print("level bias (mean predicted − realized) by position:")
        print(bias.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)
    composition().to_csv(OUT / "composition_trends.csv", index=False)
    b = frame()
    print(f"\nboard players with realized rookie value: {len(b)}, classes {b['cls'].min()}-{b['cls'].max()}")
    residual_drift(b, args.n_boot, rng)
    out_of_time(b, args.n_boot, rng)


if __name__ == "__main__":
    main()
