"""
Partial outcomes for recent classes in the board curve (pre-registered
2026-11-01).

Problem: rookie value needs four NFL seasons, so at the draft of class t
the newest complete class is t-4 and the board curve is three classes
stale. 00zt found the drift that matters is in position levels (LB/DL
over-predicted, QB under-predicted in later classes). Classes t-3, t-2,
t-1 have 3, 2, 1 played seasons: project their unplayed seasons and fit
the curve on them too.

Projection (fit per test class, only on classes complete at that time,
i.e. <= t-4): for k = 1, 2, 3 played seasons, LightGBM (small) on
remaining value Σ value_y(k+1..4) ~ value_y1..y_k, log hasan_rank,
position. Projected label = Σ played + predicted remaining (>= 0).
Board players only (the board curve); rookie value v3.1, 0 if never NFL.

Arms, board curve = production ShrunkIso (log overall rank, by position):
    lagged   classes 2014 .. t-4 with complete labels (production today)
    partial  + classes t-3, t-2, t-1 with projected labels
    oracle   + classes t-3, t-2, t-1 with their TRUE four-year labels
             (upper bound, not available in real use)
Test classes t = 2019-22 (realized four-year values); class 2014 is the
first board-era class, so lagged trains on 1-4 classes.

PRIMARY: partial − lagged, MSE of predicted vs realized rookie value on the
test classes (the curve's job is the expected-value level). ADOPT partial
outcomes for the production calibrators if P(MSE gain > 0) > 0.5 (paired
player bootstrap within class, 1000) AND value captured in the top 100 is
not clearly worse (P(> 0) >= 0.3). Prior leans toward the arm (more and
more recent data, labels projected only from information available then).
Reported: top 32 / top 100 value captured, cross-position Spearman, level
bias by position (does partial track the LB/DL/QB drift?), projection
accuracy by k (projected vs true four-year value on the recent classes).

RESULT (2026-11-01): ADOPT by the pre-registered rule, with a caveat.
1,144 board players, test classes 2019-22.
    partial − lagged  MSE gain +0.00013 [+0.00003, +0.00024] (−3.5% MSE),
                      P 0.99; top100 +0.0003 (P 0.63); top32 −0.0012
                      (P 0.49); rho −0.003 (P 0.20)
    oracle − partial  MSE −0.00001 (P 0.30): projections are as good as the
                      true labels. Projected vs true four-year value: rho
                      0.82-0.88 with 1 season played, 0.93-0.95 with 2,
                      0.98-0.99 with 3; means unbiased.
Level bias moves the right way for QB (−0.026 → −0.020), WR (−0.007 →
−0.002), OL, DB; LB stays over-predicted (+0.011 → +0.010).
CAVEAT: the gain shrinks as the lagged curve gets more classes. MSE x1e4
lagged → partial: 2019 (2 classes) 38.9 → 36.1, 2020 (3) 48.6 → 46.9,
2021 (4) 28.4 → 27.6, 2022 (5) 31.9 → 31.9. Production has 8 complete
board classes (2014-21), so most of the measured gain is extra data, not
recency, and the expected production gain is near zero. Its value in
production is insurance against faster drift.

WIRED (2026-11-01, at the user's request after shelving): src/models/
partial_outcomes.py + MergedBoard.fit(extra_board=...) in step 10, value
board's board curve only (off-board map untested, unchanged). First run:
857 projected board rows (draft years 2023-25, 1-3 seasons played), means
0.052-0.057 (complete-class board mean ~0.055). Current class vs before:
top-300 rank correlation 0.992; board mean value by position within $1M
(QB +$1.0M, WR/TE +$0.5M); top-100 mix DB 18 → 17, LB 7 → 8.

V4 RECHECK (2026-11-02, v4 label with 2025 PFF grades): same decision,
ADOPT. MSE gain +0.00012 [+0.00002, +0.00023] P 0.99; top100 P 0.62;
projection rho by seasons played as before (0.82-0.88 / 0.94 / 0.98).
v3.1 outputs: _partial_outcomes_20261101_v31.

Usage: 00zv_partial_outcomes.py [--n-boot 1000]
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
from config import PROCESSED_DIR  # noqa: E402
from src.models.merged_board import ShrunkIso, K_BOARD  # noqa: E402
from src.models.offboard_model import pid  # noqa: E402

spec = importlib.util.spec_from_file_location("zs", ROOT / "experiments/00zs_value_currency_board.py")
zs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zs)

OUT = PROCESSED_DIR / "_partial_outcomes_20261101"
POS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
TEST = [2019, 2020, 2021, 2022]
PROJ = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=20,
            subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)
ARMS = ["lagged", "partial", "oracle"]


def load():
    E = zs.load()
    b = E[E["board"] & E["pos_group"].isin(POS)].copy()
    L = pd.read_parquet(PROCESSED_DIR / "rookie_value_labels.parquet")
    L = L.assign(pid=pid(L["player_id"])).dropna(subset=["pid"]).drop_duplicates("pid").set_index("pid")
    for k in range(1, 5):
        v = b["player_id"].map(L[f"value_y{k}"])
        b[f"v{k}"] = np.where(b["nfl"], v, 0.0)
    b = b.dropna(subset=["v1", "v2", "v3", "v4"])
    b["lr"] = np.log(b["hasan_rank"])
    print(f"board players {len(b)}, classes {b['cls'].min()}-{b['cls'].max()}; "
          f"rv = Σv check max |Δ| {np.abs(b[['v1', 'v2', 'v3', 'v4']].sum(1) - b['rv']).max():.2e}")
    return b.reset_index(drop=True)


def project(done, recent, t):
    """Projected four-year value for `recent` classes (k = t − cls seasons
    played), from a model fit on `done` (complete classes)."""
    out = pd.Series(np.nan, index=recent.index)
    for k in (1, 2, 3):
        m = (t - recent["cls"]) == k
        if not m.any():
            continue
        feats = [f"v{j}" for j in range(1, k + 1)] + ["lr", "pos"]
        X = done.assign(pos=pd.Categorical(done["pos_group"], categories=POS))[feats]
        y = done[[f"v{j}" for j in range(k + 1, 5)]].sum(1)
        mdl = lgb.LGBMRegressor(**PROJ).fit(X, y)
        R = recent[m].assign(pos=pd.Categorical(recent.loc[m, "pos_group"], categories=POS))
        played = R[[f"v{j}" for j in range(1, k + 1)]].sum(1)
        out[m] = played + np.clip(mdl.predict(R[feats]), 0, None)
    return out


def curve(tr, y, te):
    return ShrunkIso(increasing=False, k=K_BOARD).fit(tr["lr"], y, tr["pos_group"]).predict(
        te["lr"], te["pos_group"])


def metrics(V, idx=None):
    d = V if idx is None else V.iloc[idx]
    by = list(d.groupby("cls"))
    return {a: {"neg_mse": -np.mean((d[a] - d["rv"]) ** 2),
                "top32": np.mean([g.nlargest(32, a)["rv"].mean() for _, g in by]),
                "top100": np.mean([g.nlargest(100, a)["rv"].mean() for _, g in by]),
                "rho": np.mean([spearmanr(g[a], g["rv"])[0] for _, g in by])} for a in ARMS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)
    b = load()
    parts, proj_rows = [], []
    for t in TEST:
        done = b[b["cls"] <= t - 4]
        recent = b[b["cls"].between(t - 3, t - 1)].copy()
        te = b[b["cls"] == t].copy()
        recent["proj"] = project(done, recent, t)
        for k in (1, 2, 3):
            r = recent[(t - recent["cls"]) == k]
            proj_rows.append({"t": t, "k_played": k, "n": len(r),
                              "rho_proj_vs_true": spearmanr(r["proj"], r["rv"])[0],
                              "mean_proj": r["proj"].mean(), "mean_true": r["rv"].mean()})
        te["lagged"] = curve(done, done["rv"], te)
        both = pd.concat([done, recent])
        te["partial"] = curve(both, pd.concat([done["rv"], recent["proj"]]), te)
        te["oracle"] = curve(both, both["rv"], te)
        print(f"test class {t}: lagged trains on {done['cls'].nunique()} classes ({len(done)}), "
              f"partial/oracle add {len(recent)} from classes {t - 3}-{t - 1}")
        parts.append(te)
    P = pd.DataFrame(proj_rows)
    print("\nprojection accuracy (projected vs true four-year value, recent classes):")
    print(P.round(4).to_string(index=False))
    V = pd.concat(parts, ignore_index=True)
    V.to_parquet(OUT / "partial_oof.parquet", index=False)

    pt = metrics(V)
    groups = [g.index.to_numpy() for _, g in V.groupby("cls")]
    boots = [metrics(V, np.concatenate([rng.choice(g, len(g)) for g in groups])) for _ in range(args.n_boot)]
    print(f"\nboard players scored: {len(V)}, classes {TEST}")
    print(pd.DataFrame(pt).T.round(5).to_string())
    rows = []
    for a, base in (("partial", "lagged"), ("oracle", "lagged"), ("oracle", "partial")):
        for k in pt[a]:
            ds = np.array([bt[a][k] - bt[base][k] for bt in boots])
            rows.append({"Δ": f"{a} − {base}", "metric": k, "delta": pt[a][k] - pt[base][k],
                         "lo95": np.percentile(ds, 2.5), "hi95": np.percentile(ds, 97.5),
                         "P(>0)": (ds > 0).mean()})
    D = pd.DataFrame(rows)
    print(D.round(5).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)
    p = D[(D["Δ"] == "partial − lagged") & (D["metric"] == "neg_mse")].iloc[0]
    q = D[(D["Δ"] == "partial − lagged") & (D["metric"] == "top100")].iloc[0]
    ok = p["P(>0)"] > 0.5 and q["P(>0)"] >= 0.3
    print(f"\nDECISION: partial − lagged MSE gain {p['delta']:+.5f} [{p['lo95']:+.5f}, {p['hi95']:+.5f}] "
          f"P(>0) {p['P(>0)']:.3f}; top100 P(>0) {q['P(>0)']:.3f} → {'ADOPT' if ok else 'KEEP lagged'}")
    print("\nlevel bias (mean predicted − realized) by position:")
    print(V.groupby("pos_group")[["rv"] + ARMS].apply(
        lambda g: pd.Series({"n": len(g), "actual": g["rv"].mean(),
                             **{a: g[a].mean() - g["rv"].mean() for a in ARMS}})).round(4).to_string())
    print("\nby test class, MSE x1e4:")
    print(V.groupby("cls")[["rv"] + ARMS].apply(
        lambda g: pd.Series({a: 1e4 * np.mean((g[a] - g["rv"]) ** 2) for a in ARMS})).round(3).to_string())


if __name__ == "__main__":
    main()
