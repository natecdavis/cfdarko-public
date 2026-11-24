"""
Quality-adjusted rookie value: does pricing QB/RB/WR/TE seasons by
(snap share, EPA quality) beat snaps-only pricing? (pre-registered
2026-11-01)

RESULT (2026-11-01): ADOPT. 800 drafted QB/RB/WR/TE, classes 2011-20,
cross-fitted (5 player folds).
    within-group Pearson    epa − wide +0.031 [+0.022, +0.040] P 1.00 (primary)
                            epa − v2   +0.027 [+0.013, +0.040] P 1.00
    within-group Spearman   epa − wide +0.008 [+0.001, +0.014] P 0.99
    paid > 1% (n 289)       Pearson +0.061, Spearman +0.054, P 1.00
    wide − v2 is a tie (Pearson −0.004, P 0.26), so the gain is the quality
    term, not the refit seasons. epa wins in every group on Pearson (QB
    .768→.814, WR .822→.853, TE .841→.867, RB .841→.865).
Biggest markups: Russell Wilson, Matt Ryan, Mahomes, Newton (+0.09..+0.14
of cap over y1-4); markdowns: Sanchez, Bradford, Darnold, Carr.
Production label rebuilt with --pay-fit wide --quality epa on all contracts
(now the 00zo defaults); v2 kept as rookie_surplus_labels_v2.parquet.

Background: 00zp found quality beyond snaps predicts the SIZE of the second
contract. 00zo --quality epa builds the label with a 2-D veteran pay curve
f_pos(share, q) for skill groups (see 00zo docstring, v3 option).

Label arms, built by 00zo cross-fitted: 5 runs per arm with --xval-fold
0..4; each validation player's value comes from the run whose pay curves
excluded all of his own contracts:
    v2     production label (1-D isotonic, 2023-25 contracts) -- reference
           only; it is NOT refit without the validation contracts, so it is
           mildly advantaged.
    wide   --pay-fit wide: 1-D, 2016-24 contracts, LightGBM on share for
           skill groups (matched reference)
    epa    --pay-fit wide --quality epa: 2-D (share, q) for skill groups
Cohort: drafted QB/RB/WR/TE, classes 2011-20, complete value_y14.
Outcome: second-contract APY % of cap (00zp definition; no deal = 0).

PRIMARY: within-group Pearson of value_y14 vs outcome (n-weighted over
QB/RB/WR/TE), epa − wide. Pearson because the label is used in cap dollars:
surplus sums value, so getting magnitudes right in the paid tail matters,
and Spearman over all players is dominated by deal-vs-no-deal (00zp).
Secondary: within-group Spearman, all players and among paid > 1% of cap
(the latter conditions on the outcome; descriptive).
Decision (feedback_bayesian_decisions): paired player bootstrap (1000)
within group. Adopt --quality epa as the production label if P(epa − wide >
0) on the primary > 0.5 AND overall Spearman is not clearly worse (its
P(> 0) >= 0.2). The prior leans toward the arm: 00zp showed the effect on
the same outcome with a model, and the change is principled (the market
prices quality). Cost: a new data dependency with a missing 2025 season.
A placebo arm is not needed: the label build is deterministic (fixed seed),
and the seed-only spread in 00zp was <= 0.002.

Usage: 00zq_epa_quality_label.py [--n-boot 1000]   (after the 00zo builds)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR  # noqa: E402

D = PROCESSED_DIR / "_rookie_surplus_20261101"
FILES = {"v2": "rookie_surplus_labels_v2.parquet", "wide": "labels_wide_xval{k}.parquet",
         "epa": "labels_epa_xval{k}.parquet"}
K = 5


def load_arm(arm):
    """value_y14 by pfr_id; cross-fitted for xval arms (own-fold run)."""
    if "{k}" not in FILES[arm]:
        return pd.read_parquet(D / FILES[arm]).set_index("pfr_id")["value_y14"]
    out = []
    for k in range(K):
        f = pd.read_parquet(D / FILES[arm].format(k=k))
        h = pd.util.hash_pandas_object(f["gsis"].fillna(""), index=False).to_numpy() % K
        out.append(f[h == k].set_index("pfr_id")["value_y14"])
    return pd.concat(out)
SKILL = ["QB", "RB", "WR", "TE"]

spec = importlib.util.spec_from_file_location("zp", ROOT / "experiments/00zp_quality_proxy_check.py")
zp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zp)


def corr(a, b, kind):
    return spearmanr(a, b)[0] if kind == "spearman" else np.corrcoef(a, b)[0, 1]


def within(df, col, kind, idx=None):
    d = df if idx is None else df.iloc[idx]
    num = den = 0.0
    for _, g in d.groupby("group"):
        if len(g) >= 10 and g["y"].nunique() > 1:
            num += len(g) * corr(g[col], g["y"], kind)
            den += len(g)
    return num / den


def boot(df, a, b, kind, n_boot, rng):
    groups = [np.flatnonzero(df["group"].to_numpy() == g) for g in df["group"].unique()]
    pt = within(df, a, kind) - within(df, b, kind)
    ds = np.array([within(df, a, kind, idx) - within(df, b, kind, idx)
                   for idx in (np.concatenate([rng.choice(ix, len(ix)) for ix in groups])
                               for _ in range(n_boot))])
    return {"Δ": f"{a} − {b}", "metric": kind, "n": len(df), "delta": pt,
            "lo95": np.percentile(ds, 2.5), "hi95": np.percentile(ds, 97.5), "P(>0)": (ds > 0).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    L = {k: load_arm(k) for k in FILES}
    base = pd.read_parquet(D / FILES["v2"])
    df = base[base["drafted"] & base["draft_year"].between(*zp.CLASSES) & base["group"].isin(SKILL)
              & base["gsis"].notna()].drop_duplicates("gsis").copy()
    df["y"] = df["gsis"].map(zp.second_contract(df)).fillna(0.0)
    for k, v in L.items():
        df[k] = df["pfr_id"].map(v)
    df = df.dropna(subset=list(L)).reset_index(drop=True)
    print(f"cohort {len(df)} drafted QB/RB/WR/TE, classes {zp.CLASSES}")

    pd.set_option("display.width", 200)
    print("\nper group, value_y14 vs 2nd-contract pay:")
    rows = {}
    for g, G in df.groupby("group"):
        for kind in ("pearson", "spearman"):
            rows[(g, kind)] = {k: corr(G[k], G["y"], kind) for k in L}
    print(pd.DataFrame(rows).T.round(3).to_string())

    tail = df[df["y"] > 0.01].reset_index(drop=True)
    res = []
    for kind in ("pearson", "spearman"):
        for a, b in [("epa", "wide"), ("epa", "v2"), ("wide", "v2")]:
            res.append({**boot(df, a, b, kind, args.n_boot, rng), "scope": "all"})
    for a, b in [("epa", "wide")]:
        for kind in ("pearson", "spearman"):
            res.append({**boot(tail, a, b, kind, args.n_boot, rng), "scope": "paid>1%"})
    R = pd.DataFrame(res)
    print("\nwithin-group, n-weighted, paired bootstrap:")
    print(pd.DataFrame({k: {f"{kind}/{sc}": within(dd, k, kind) for kind in ("pearson", "spearman")
                            for sc, dd in (("all", df), ("paid>1%", tail))} for k in L}).round(4).to_string())
    print(R.round(4).to_string(index=False))

    prim = R[(R["Δ"] == "epa − wide") & (R["metric"] == "pearson") & (R["scope"] == "all")].iloc[0]
    sec = R[(R["Δ"] == "epa − wide") & (R["metric"] == "spearman") & (R["scope"] == "all")].iloc[0]
    adopt = bool(prim["P(>0)"] > 0.5 and sec["P(>0)"] >= 0.2)
    print(f"\nDECISION: primary Δ {prim['delta']:+.4f}, P(>0) {prim['P(>0)']:.3f}; "
          f"spearman P(>0) {sec['P(>0)']:.3f} → {'ADOPT' if adopt else 'KEEP snaps-only'}")

    e = base.copy()
    e["value_y14"] = e["pfr_id"].map(L["epa"])
    e["d"] = e["value_y14"] - e["pfr_id"].map(L["wide"])
    show = ["player_name", "draft_year", "group", "value_y14", "d"]
    print("\nlargest quality markups / markdowns (epa − wide, value_y14):")
    print(e.dropna(subset=["d"]).nlargest(8, "d")[show].round(4).to_string(index=False))
    print(e.dropna(subset=["d"]).nsmallest(8, "d")[show].round(4).to_string(index=False))
    R.to_csv(D / "00zq_deltas.csv", index=False)


if __name__ == "__main__":
    main()
