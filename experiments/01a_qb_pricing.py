"""
QB (position-level) pricing in the merged board (pre-registered 2026-11-01).

Problem: the merged board's calibrators (src/models/merged_board.py) shrink
each position's isotonic curve toward the POOLED curve (board: K_BOARD =
100; off-board: K_OFF = 50). In rookie-value units position levels differ a
lot, so the shrinkage erases part of the exchange rate: 00zs found board QBs
predicted 0.088 vs realized 0.123 (−29%), DL/EDGE/LB/RB over-predicted
10-15%; 00zt found QB −17%, LB +30% out of time. 00zs's level-preserving
arm fixed the calibration but failed its RANK primary (board Spearman
−0.012). Rank metrics score a skewed position (QB: few huge hits) by its
typical outcome, the wrong yardstick for an expected-value currency; the
proper score for a mean is squared error.

Arms (applied to both calibrators; the off-board cap is unchanged):
    P0_pooled  production: shrink toward the pooled curve
    P1_level   shrink toward the pooled curve rescaled to the position's
               level: w·iso_g(x) + (1 − w)·lvl_g·pooled(x),
               lvl_g = mean(y_g) / mean(pooled(x_g)) on training rows
    P2_k25     production with lighter shrinkage (K_BOARD 25, K_OFF 12.5)
Data: players in production 8d out-of-fold predictions
(offboard_models/offboard_oof_predictions.parquet, score = engine) with last
college season 2013-2021 (classes 2014-22, four NFL seasons played);
outcome = production rookie value value_y14 (v3.1), 0 if never NFL; NFL
rows without a value dropped; K/P/OTHER excluded.
Validation: leave-one-class-out (fit on the other 8 classes).
PRIMARY: MSE of predicted vs realized rookie value over BOARD players (the
exchange rate lives on the board). Adopt the arm with the lowest MSE among
those with P(better than P0) > 0.5, provided its value captured by the top
32 and top 100 of the merged list per class is not clearly worse (P(better
than P0) >= 0.2 for both). P from a paired bootstrap of players within
class (1000).
Reported: MSE over off-board and all players; board Spearman (continuity
with 00zs); mean predicted − realized by position (board); expanding-window
check (train <= 2015 → 2016-18, train <= 2018 → 2019-21) for the primary;
top-100 position mix.

RESULT (2026-11-01): ADOPT P1_level (level-preserving shrinkage).
44,636 players, classes 2014-22, 2,467 board players, LOCO.
    P1_level − P0   board MSE gain +0.53e-4 [−0.50, +1.58] (−1.4%), P 0.85;
                    top32 +0.0003 (P 0.46), top100 −0.0006 (P 0.37) → pass;
                    board Spearman −0.009 (P 0.00; rank metric, see above)
    P2_k25 − P0     board MSE P 0.63, smaller gain; board Spearman P 0.02
Board level bias (pred − realized), P0 → P1: QB −0.0223 → −0.0003 (−19%
→ 0), RB +0.0048 → 0.0000, LB +0.0031 → −0.0008, DL +0.0027 → +0.0002,
WR −0.0022 → 0.0000; every position within ±0.001 under P1.
Expanding window (the real use): board MSE x1e4 39.03 → 38.06; QB bias
−0.036 → −0.008, LB +0.006 → +0.002, DL +0.006 → +0.004.
Top 100 per class: QB 6.1 → 6.9, RB 5.9 → 4.7, LB 6.9 → 6.2.
Off-board calibration is unchanged in practice (MSE identical).
Wired: ShrunkIso(level=True) for target "rv" in MergedBoard.fit (both
calibrators); the usage board is unchanged.

V4 RECHECK (2026-11-02, v4 label with 2025 PFF grades): same decision,
P1_level. Board MSE gain +0.38e-4, P 0.77 (v3.1: P 0.85); top32 P 0.46,
top100 P 0.35 (guards pass). QB bias −0.021 → −0.0001; every position
within ±0.0011 under P1. v3.1 outputs: _qb_pricing_20261101_v31.

Usage: 01a_qb_pricing.py [--n-boot 1000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR  # noqa: E402
from src.models.merged_board import CAP_Q, K_BOARD, K_OFF, ShrunkIso, offboard_percentile  # noqa: E402
from src.models.offboard_model import pid  # noqa: E402
from src.models.rookie_value import LABEL_PATH  # noqa: E402

OUT = PROCESSED_DIR / "_qb_pricing_20261101"
SEASONS = (2013, 2021)
ARMS = ["P0_pooled", "P1_level", "P2_k25"]


class LevelShrunkIso(ShrunkIso):
    """ShrunkIso, but the pooled curve is rescaled to each position's level."""

    def fit(self, x, y, pos, weight_n=None):
        super().fit(x, y, pos, weight_n)
        x, y, pos = np.asarray(x, float), np.asarray(y, float), np.asarray(pos)
        pp = self.pooled.predict(x)
        self.lvl = {g: (y[pos == g].mean() / pp[pos == g].mean()) if pp[pos == g].mean() > 0 else 1.0
                    for g in np.unique(pos)}
        return self

    def predict(self, x, pos):
        x, pos = np.asarray(x, float), np.asarray(pos)
        base = self.pooled.predict(x)
        out = base.copy()
        for g in np.unique(pos):
            if g in self.per:
                m = pos == g
                out[m] = self.w[g] * self.per[g].predict(x[m]) + (1 - self.w[g]) * self.lvl[g] * base[m]
        return out


SPEC = {"P0_pooled": (ShrunkIso, K_BOARD, K_OFF), "P1_level": (LevelShrunkIso, K_BOARD, K_OFF),
        "P2_k25": (ShrunkIso, K_BOARD / 4, K_OFF / 4)}


def load():
    o = pd.read_parquet(PROCESSED_DIR / "offboard_models" / "offboard_oof_predictions.parquet")
    o = o.drop_duplicates("player_id", keep="last")
    o = o[o["last_college_season_year"].between(*SEASONS) & ~o["pos_group"].isin(["K", "P", "OTHER"])]
    L = pd.read_parquet(LABEL_PATH).assign(pid=lambda d: pid(d["player_id"])).dropna(subset=["pid"])
    rv = L.drop_duplicates("pid").set_index("pid")["value_y14"]
    o["rv"] = np.where(o["nfl"], o["player_id"].map(rv), 0.0)
    n0 = len(o)
    o = o[o["rv"].notna()].reset_index(drop=True)
    print(f"dropped {n0 - len(o)} NFL rows without a rookie value; {len(o)} rows")
    o["cls"] = o["last_college_season_year"].astype(int) + 1
    o["board"] = o["hasan_rank"].notna()
    o["pct"] = np.nan
    ob = ~o["board"]
    o.loc[ob, "pct"] = offboard_percentile(o.loc[ob, "score"], o.loc[ob, "cls"], o.loc[ob, "pos_group"])
    return o


def predict(tr, va):
    bt, ob = tr[tr["board"]], tr[~tr["board"]]
    vb = va["board"].to_numpy()
    n_nfl = ob.groupby("pos_group")["nfl"].sum().to_dict()
    top = ob[ob["pct"] >= CAP_Q]
    pooled = float(top["rv"].mean())
    cap = {g: (len(t) * t["rv"].mean() + K_OFF * pooled) / (len(t) + K_OFF) for g, t in top.groupby("pos_group")}
    capv = va["pos_group"].map(cap).fillna(pooled).to_numpy()
    out = {}
    for a, (Cal, kb, ko) in SPEC.items():
        bc = Cal(increasing=False, k=kb).fit(np.log(bt["hasan_rank"]), bt["rv"], bt["pos_group"])
        oc = Cal(increasing=True, k=ko).fit(ob["pct"], ob["rv"], ob["pos_group"], weight_n=n_nfl)
        p = np.minimum(oc.predict(va["pct"].fillna(0), va["pos_group"]), capv)
        p[vb] = bc.predict(np.log(va.loc[vb, "hasan_rank"]), va.loc[vb, "pos_group"])
        out[a] = p
    return out


def run(E, splits):
    parts = []
    for tr_m, va_m in splits:
        va = E[va_m].copy()
        for a, p in predict(E[tr_m], va).items():
            va[a] = p
        parts.append(va)
    return pd.concat(parts, ignore_index=True)


def metrics(V, idx=None):
    d = V if idx is None else V.iloc[idx]
    b = d[d["board"]]
    by = list(d.groupby("cls"))
    r = {}
    for a in ARMS:
        r[a] = {"neg_mse_board": -np.mean((b[a] - b["rv"]) ** 2),
                "neg_mse_off": -np.mean((d.loc[~d["board"], a] - d.loc[~d["board"], "rv"]) ** 2),
                "neg_mse_all": -np.mean((d[a] - d["rv"]) ** 2),
                "top32": np.mean([g.nlargest(32, a)["rv"].mean() for _, g in by]),
                "top100": np.mean([g.nlargest(100, a)["rv"].mean() for _, g in by]),
                "board_rho": np.mean([spearmanr(g.loc[g["board"], a], g.loc[g["board"], "rv"])[0] for _, g in by])}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(0)
    E = load()
    classes = sorted(E["cls"].unique())
    V = run(E, [((E["cls"] != c).to_numpy(), (E["cls"] == c).to_numpy()) for c in classes])
    V.to_parquet(OUT / "loco_preds.parquet", index=False)
    print(f"LOCO: {len(V)} players, classes {classes[0]}-{classes[-1]}, board {int(V['board'].sum())}")

    pt = metrics(V)
    groups = [g.index.to_numpy() for _, g in V.groupby("cls")]
    boots = [metrics(V, np.concatenate([rng.choice(g, len(g)) for g in groups])) for _ in range(args.n_boot)]
    print(pd.DataFrame(pt).T.round(6).to_string())
    rows = []
    for a in ARMS[1:]:
        for k in pt[a]:
            ds = np.array([bt[a][k] - bt["P0_pooled"][k] for bt in boots])
            rows.append({"arm": a, "metric": k, "delta": pt[a][k] - pt["P0_pooled"][k],
                         "lo95": np.percentile(ds, 2.5), "hi95": np.percentile(ds, 97.5), "P(better)": (ds > 0).mean()})
    D = pd.DataFrame(rows)
    print("\narm − P0_pooled (> 0 = arm better; MSE metrics negated):")
    print(D.round(6).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)
    Dx = D.set_index(["arm", "metric"])
    ok = [a for a in ARMS[1:] if Dx.loc[(a, "neg_mse_board"), "P(better)"] > 0.5
          and Dx.loc[(a, "top32"), "P(better)"] >= 0.2 and Dx.loc[(a, "top100"), "P(better)"] >= 0.2]
    pick = max(ok, key=lambda a: pt[a]["neg_mse_board"]) if ok else "P0_pooled"
    print(f"\nDECISION: {pick}" + ("" if ok else " (no arm passes)"))

    b = V[V["board"]]
    print("\nboard players, mean predicted − realized rookie value by position (LOCO):")
    print(b.groupby("pos_group").apply(lambda g: pd.Series(
        {"n": len(g), "actual": g["rv"].mean(), **{a: g[a].mean() - g["rv"].mean() for a in ARMS}}),
        include_groups=False).round(4).to_string())
    o = V[~V["board"] & V["nfl"]]
    print("\noff-board NFL players + all off-board, mean predicted − realized by position:")
    print(V[~V["board"]].groupby("pos_group").apply(lambda g: pd.Series(
        {"n_nfl": int(g["nfl"].sum()), "actual": g["rv"].mean(), **{a: g[a].mean() - g["rv"].mean() for a in ARMS}}),
        include_groups=False).round(5).to_string())

    exp = run(E, [((E["cls"] <= te).to_numpy(), E["cls"].between(te + 1, te + 3).to_numpy())
                  for te in (2016, 2019)])
    pe = metrics(exp)
    print("\nexpanding-window check (train <= class 2016 → 2017-19, <= 2019 → 2020-22), board MSE x1e4:")
    print({a: round(-1e4 * pe[a]["neg_mse_board"], 3) for a in ARMS})
    eb = exp[exp["board"]]
    print("expanding: board mean predicted − realized, QB / LB / DL:",
          {a: [round(eb.loc[eb.pos_group == g, a].mean() - eb.loc[eb.pos_group == g, "rv"].mean(), 4)
               for g in ("QB", "LB", "DL")] for a in ARMS})
    print("\ntop-100 position mix (players per class):")
    print(pd.DataFrame({a: V.groupby("cls", group_keys=False)[V.columns.tolist()].apply(lambda g: g.nlargest(100, a))
                        ["pos_group"].value_counts() / len(classes) for a in ARMS}).round(1).fillna(0).to_string())


if __name__ == "__main__":
    main()
