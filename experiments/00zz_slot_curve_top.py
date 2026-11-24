"""
Slot curve at the top of the draft (pre-registered 2026-11-01).

Production slot curve (00zr / src/models/slot_curve.py): value and cost
each smoothed by local-linear kNN in log pick (k = 380, chosen by LOCO
error over ALL picks), value made non-increasing by isotonic regression.
Concerns: (1) the bandwidth is set by picks 33-262 and may be wrong for the
top, where the curve is steep (raw mean value pick 1 0.41, 2 0.25, 3 0.21,
4 0.17; 17 players each) and boundary-extrapolated; (2) cost is nearly
fixed by slot, yet smoothing overshoots at the boundary (curve pick 1 0.201
vs raw mean 0.185; pick 4 0.153 vs 0.168).

Value arms (all made non-increasing by isotonic regression afterwards):
    V0_knn     production: local-linear kNN, k = 380
    V1_pick    per-pick training mean (no smoothing across picks)
    V2_glm     log-link Poisson GLM on log p, (log p)^2, (log p)^3
    V3_topk    local-linear kNN with k chosen by inner leave-one-class-out
               error on picks 1-64 only (grid 30, 50, 80, 120, 180, 260, 380)
Cost arms:
    C0_knn     production kNN, k = 380 (no isotonic, as in production)
    C1_pick    per-pick training mean
Data: drafted players with rookie value labels, classes 2006-22 (as
production). Validation: leave-one-class-out; each player's squared error
of realized value_y14 (cost_y14) vs the curve at his pick.
PRIMARY: mean squared error over picks 1-32. Value: adopt the arm with the
lowest point MSE among those with P(better than V0) > 0.5; else keep V0.
Cost: adopt C1 if P(better than C0) > 0.5. P from a class-cluster bootstrap
(classes resampled, 2000 reps; class-level curve errors are correlated).
Reported: MSE over picks 33-262 and all picks (an arm must not be clearly
worse there: P(better than V0) >= 0.2 on picks 33-262, else keep V0 for
those picks and splice at pick 32); curves at picks 1-10, 16, 32, 64, 100;
implied slot surplus.

RESULT (2026-11-01): value = V2_glm, cost = C1_pick (both adopted; no splice).
4,146 drafted players, 2006-22. LOCO MSE x1e4, picks 1-32 / 33-262:
V0 110.54 / 30.07, V1 111.14 / 30.00, V2 108.53 / 29.92, V3 110.89 / 30.09;
cost C0 1.514 / 0.294, C1 1.187 / 0.307.
    V2_glm vs V0   1-32 +2.01 [−0.99, +4.88] P 0.90; 33-262 +0.15 P 1.00
    V1, V3 vs V0   ties/worse at the top (P 0.39, 0.42); V3's inner LOCO
                   picks k = 120 for most folds, no better out of sample
    C1 vs C0 cost  1-32 +0.33 P 1.00; 33-262 −0.013 P 0.00 (tiny)
The smoothing concern ran the other way: production UNDER-states the top.
Value at pick 1 0.349 → 0.401 (raw mean 0.407; 12 of 17 first picks were
QBs); pick 2-3 come down (0.284 → 0.261, 0.244 → 0.222). Cost at pick 1
0.201 → 0.185, picks 3-5 up ~0.01-0.016 (boundary overshoot removed).
Slot surplus: pick 1 0.148 → 0.215; picks 2-5 0.085, 0.052, 0.035, 0.032
(from 0.107-0.054); 8-32 within ±0.007. Picks 3-5 are now the draft's
worst surplus slots in round 1 (expensive, not QB-heavy).

V4 RECHECK (2026-11-02, v4 label with 2025 PFF grades): same decision,
value = V2_glm, cost = C1_pick. V2 vs V0 picks 1-32 +1.67e-4 P 0.91,
33-262 P 0.99. Slot surplus pick 1 0.194; picks 3-5 still round 1's lowest
(0.050, 0.034, 0.032). v3.1 outputs: _slot_curve_top_20261101_v31.

Usage: 00zz_slot_curve_top.py [--n-boot 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR  # noqa: E402
from src.models.rookie_value import LABEL_PATH  # noqa: E402
from src.models.slot_curve import CLASSES, K_GRID, PICKS, kernel_smooth  # noqa: E402

OUT = PROCESSED_DIR / "_slot_curve_top_20261101"
K_PROD = 380
V_ARMS = ["V0_knn", "V1_pick", "V2_glm", "V3_topk"]
C_ARMS = ["C0_knn", "C1_pick"]


def iso(v):
    return IsotonicRegression(increasing=False).fit_transform(PICKS, v)


def per_pick(tr, col):
    m = tr.groupby("p")[col].mean().reindex(PICKS)
    n = tr.groupby("p").size().reindex(PICKS).fillna(0)
    m = m.interpolate(limit_direction="both")
    return m.to_numpy(), np.maximum(n.to_numpy(), 1)


def glm(tr, col):
    def X(p):
        lp = np.log(np.asarray(p, float))
        return np.column_stack([lp, lp ** 2, lp ** 3])
    m = PoissonRegressor(alpha=0, max_iter=1000).fit(X(tr["p"]), tr[col].clip(lower=0))
    return m.predict(X(PICKS))


def topk(tr, col):
    err = {}
    for k in K_GRID:
        e, n = 0.0, 0
        for c in tr["draft_year"].unique():
            a, b = tr[tr["draft_year"] != c], tr[(tr["draft_year"] == c) & (tr["p"] <= 64)]
            pred = kernel_smooth(a["p"], a[col], b["p"], k)
            e += ((b[col].to_numpy() - pred) ** 2).sum()
            n += len(b)
        err[k] = e / n
    k = min(err, key=err.get)
    return kernel_smooth(tr["p"], tr[col], PICKS, k), k


def value_curves(tr):
    m, n = per_pick(tr, "value_y14")
    v3, k = topk(tr, "value_y14")
    return {"V0_knn": iso(kernel_smooth(tr["p"], tr["value_y14"], PICKS, K_PROD)),
            "V1_pick": IsotonicRegression(increasing=False).fit_transform(PICKS, m, sample_weight=n),
            "V2_glm": iso(glm(tr, "value_y14")),
            "V3_topk": iso(v3)}, k


def cost_curves(tr):
    return {"C0_knn": kernel_smooth(tr["p"], tr["cost_y14"], PICKS, K_PROD),
            "C1_pick": per_pick(tr, "cost_y14")[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(0)
    L = pd.read_parquet(LABEL_PATH)
    d = L[L["drafted"] & L["draft_year"].between(*CLASSES) & L["surplus_y14"].notna()
          & L["overall_pick"].notna()].copy()
    d["p"] = d["overall_pick"].round().clip(1, 262).astype(int)
    d = d.reset_index(drop=True)
    print(f"drafted, labelled: {len(d)}, classes {CLASSES}")

    E = d[["draft_year", "p", "value_y14", "cost_y14"]].copy()
    ks = {}
    for c in sorted(d["draft_year"].unique()):
        m = (d["draft_year"] == c).to_numpy()
        tr = d[~m]
        vc, ks[c] = value_curves(tr)
        cc = cost_curves(tr)
        idx = d.loc[m, "p"].to_numpy() - 1
        for a, v in vc.items():
            E.loc[m, f"e_{a}"] = (d.loc[m, "value_y14"].to_numpy() - v[idx]) ** 2
        for a, v in cc.items():
            E.loc[m, f"e_{a}"] = (d.loc[m, "cost_y14"].to_numpy() - v[idx]) ** 2
    print("V3 inner-chosen k by held-out class:", ks)
    E.to_parquet(OUT / "loco_errors.parquet", index=False)

    ranges = {"1-32": (1, 32), "33-262": (33, 262), "all": (1, 262)}
    print("\nLOCO MSE x1e4:")
    print(pd.DataFrame({r: {a: 1e4 * E.loc[E["p"].between(*b), f"e_{a}"].mean() for a in V_ARMS + C_ARMS}
                        for r, b in ranges.items()}).round(3).to_string())

    classes = E["draft_year"].unique()
    by = {c: E[E["draft_year"] == c] for c in classes}
    boots = [pd.concat([by[c] for c in rng.choice(classes, len(classes))]) for _ in range(args.n_boot)]
    rows = []
    for a, base in [(v, "V0_knn") for v in V_ARMS[1:]] + [("C1_pick", "C0_knn")]:
        for r, b in ranges.items():
            def gain(x):
                x = x[x["p"].between(*b)]
                return x[f"e_{base}"].mean() - x[f"e_{a}"].mean()
            bs = np.array([gain(x) for x in boots])
            rows.append({"arm": a, "base": base, "picks": r, "gain_x1e4": 1e4 * gain(E),
                         "lo95": 1e4 * np.percentile(bs, 2.5), "hi95": 1e4 * np.percentile(bs, 97.5),
                         "P(arm better)": (bs > 0).mean()})
    D = pd.DataFrame(rows)
    print(D.round(4).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)

    top = D[D["picks"] == "1-32"].set_index("arm")
    rest = D[D["picks"] == "33-262"].set_index("arm")
    ok = [a for a in V_ARMS[1:] if top.loc[a, "P(arm better)"] > 0.5]
    v_pick = min(ok, key=lambda a: -top.loc[a, "gain_x1e4"]) if ok else "V0_knn"
    splice = v_pick != "V0_knn" and rest.loc[v_pick, "P(arm better)"] < 0.2
    c_pick = "C1_pick" if top.loc["C1_pick", "P(arm better)"] > 0.5 else "C0_knn"
    print(f"\nDECISION: value = {v_pick}{' (spliced into V0 after pick 32)' if splice else ''}; cost = {c_pick}")

    vc, kfull = value_curves(d)
    cc = cost_curves(d)
    show = [1, 2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 100]
    raw = d.groupby("p")[["value_y14", "cost_y14"]].mean()
    T = pd.DataFrame({**{a: vc[a] for a in V_ARMS}, **{a: cc[a] for a in C_ARMS}}, index=PICKS).loc[show]
    T["raw_value"], T["raw_cost"] = raw.loc[show, "value_y14"], raw.loc[show, "cost_y14"]
    print(f"\ncurves fitted on all classes (V3 k = {kfull}):")
    print(T.round(4).to_string())
    print("\nimplied slot surplus (value − cost): production vs chosen")
    print(pd.DataFrame({"production": vc["V0_knn"] - cc["C0_knn"], "chosen": vc[v_pick] - cc[c_pick]},
                       index=PICKS).loc[show].round(4).T.to_string())


if __name__ == "__main__":
    main()
