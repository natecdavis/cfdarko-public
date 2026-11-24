"""
Does on-field quality add to snap-based rookie value in predicting the
second contract? (exploratory check, 2026-11-01, run while PFF NFL grades
are unavailable)

RESULT (2026-11-01): quality matters, and it matters where the money is.
2,333 drafted players, 80% get some deal. Placebo |Δ| <= 0.002 everywhere.
  All players (deal vs no deal dominates; base rho 0.853):
    av  − base +0.032 [+0.027, +0.039]  P(>0) 1.00   (leaky upper bound)
    epa − base −0.011 [−0.029, +0.005]  P(>0) 0.09   (skill groups)
  Among players whose 2nd contract > 1% of cap (n 939; skill 288):
    av  − base +0.114 [+0.086, +0.141]; epa − base +0.078 [+0.043, +0.120]
  > 2% of cap (n 700; skill 216):
    av  − base +0.144; epa − base +0.131 [+0.063, +0.199]
  Partial Spearman given value_y14, paid > 1%: av_per_game 0.39-0.74 every
  group; epa_y34 QB 0.67, WR 0.46, RB 0.39, TE 0.34.
Reading: snaps decide WHETHER a player earns a second deal; quality decides
HOW MUCH. Clean rookie-window EPA reproduces most of the leaky AV gain in
the paid tail, so it is not just leakage. The label's high-dollar end
(where surplus concentrates) is the part a snaps-only price gets wrong.
Caveat: the > threshold cuts condition on the outcome (descriptive only).
Next: PFF NFL grades (all positions, incl. OL/DB where EPA does not exist)
once the site works; test the same way, then a 2-D pay curve.

Question. The rookie-surplus label (00zo) prices a player's rookie-deal
seasons as the veteran-market price of his ROLE (snap share x games). Two
players with the same snaps are worth the same. If quality matters, the
market's own verdict on the rookie deal -- the second contract -- should be
predictable from quality BEYOND snaps. If it is, a quality measure (PFF
grade, eventually) belongs in the value label.

Outcome: second-contract pay = max apy_cap_pct of contracts signed in
draft_year+3 .. draft_year+5, excluding the rookie deal and the round-1
fifth-year option (franchise tags count). No contract = 0 (out of league or
unsigned). Drafted players, classes 2011-20, K/P excluded.

Arms -- pooled LightGBM, group as categorical, monotone +1 on every
value/quality feature, leave-one-class-out, 3 seeds averaged:
    base      value_y1..value_y4 from 00zo (vet price of role x games)
    base_alt  same, seeds +100 (placebo)
    av        base + career_w_av, career_w_av / nfl_games.  LEAKY: career AV
              includes seasons after the second contract is signed, which
              the contract itself causes. Treat as an UPPER BOUND on what a
              quality measure can add: if even this adds little, quality is
              not worth much in the label.
    epa       base + y1-4 and y3-4 total EPA (passing+rushing+receiving,
              nflverse seasonal REG). Clean (rookie-window only), QB/RB/WR/TE
              only; scored on those groups against base on the same rows.
Score: Spearman of OOF prediction vs outcome within group, n-weighted mean.
Uncertainty: paired player bootstrap within group (1000) + placebo spread;
report P(arm better) per feedback_bayesian_decisions. Also reported: partial
Spearman of each quality measure with the outcome given value_y14, by group.

No pre-registered gate (exploratory). Decision use: a large av gain + a
real epa gain => download PFF NFL grades and test them the same way; a
small av gain => quality adjustment is low priority.

Usage: 00zp_quality_proxy_check.py [--n-boot 1000]
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, RAW_DIR  # noqa: E402

OUT = PROCESSED_DIR / "_quality_proxy_20261101"
CLASSES = (2011, 2020)
SEEDS = [42, 43, 44]
SKILL = ["QB", "RB", "WR", "TE"]
VALUE = ["value_y1", "value_y2", "value_y3", "value_y4"]
ARMS = {
    "base": (VALUE, SEEDS),
    "base_alt": (VALUE, [s + 100 for s in SEEDS]),
    "av": (VALUE + ["career_w_av", "av_per_game"], SEEDS),
    "epa": (VALUE + ["epa_y14", "epa_y34"], SEEDS),
}
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)


def second_contract(lab: pd.DataFrame) -> pd.Series:
    c = pd.read_parquet(RAW_DIR / "nfl_contracts.parquet").dropna(subset=["gsis_id"])
    c = c.merge(lab[["gsis", "draft_year", "round"]], left_on="gsis_id", right_on="gsis")
    dy = c["draft_year_y"]
    opt = (c["round"] == 1) & (c["years"] == 1) & (c["year_signed"] == dy + 4)
    ok = c["year_signed"].between(dy + 3, dy + 5) & ~opt
    return c[ok].groupby("gsis")["apy_cap_pct"].max()


def epa_features(lab: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_parquet(OUT / "cache" / "seasonal.parquet")
    s["epa"] = s[["passing_epa", "rushing_epa", "receiving_epa"]].fillna(0).sum(axis=1)
    s = s.merge(lab[["gsis", "draft_year"]], left_on="player_id", right_on="gsis")
    s["y"] = s["season"] - s["draft_year"] + 1
    e14 = s[s["y"].between(1, 4)].groupby("gsis")["epa"].sum()
    e34 = s[s["y"].between(3, 4)].groupby("gsis")["epa"].sum()
    return pd.DataFrame({"epa_y14": e14, "epa_y34": e34})


def build() -> pd.DataFrame:
    lab = pd.read_parquet(PROCESSED_DIR / "_rookie_surplus_20261101" / "rookie_surplus_labels.parquet")
    lab = lab[lab["drafted"] & lab["draft_year"].between(*CLASSES) & lab["gsis"].notna()
              & ~lab["group"].isin(["K", "P"])].drop_duplicates("gsis").copy()
    nl = (pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
          .dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id"))
    lab["career_w_av"] = lab["pfr_id"].map(nl["career_w_av"]).fillna(0.0)
    lab["av_per_game"] = lab["career_w_av"] / lab["pfr_id"].map(nl["nfl_games"]).fillna(0).clip(lower=1)
    lab["y"] = lab["gsis"].map(second_contract(lab)).fillna(0.0)
    lab = lab.join(epa_features(lab), on="gsis")
    skill = lab["group"].isin(SKILL)
    for col in ["epa_y14", "epa_y34"]:
        lab.loc[skill, col] = lab.loc[skill, col].fillna(0.0)   # skill player, no touches = 0
        lab.loc[~skill, col] = np.nan
    lab["value_y14"] = lab[VALUE].sum(axis=1)
    return lab.reset_index(drop=True)


def oof(df: pd.DataFrame, feats, seeds) -> np.ndarray:
    X = df[feats + ["group"]].copy()
    X["group"] = X["group"].astype("category")
    mono = [1] * len(feats) + [0]
    pred = np.full(len(df), np.nan)
    for cls in range(CLASSES[0], CLASSES[1] + 1):
        te = (df["draft_year"] == cls).to_numpy()
        ps = []
        for s in seeds:
            m = lgb.LGBMRegressor(**PARAMS, random_state=s, monotone_constraints=mono)
            m.fit(X[~te], df.loc[~te, "y"])
            ps.append(m.predict(X[te]))
        pred[te] = np.mean(ps, axis=0)
    return pred


def within_rho(df, col, idx=None) -> float:
    d = df if idx is None else df.iloc[idx]
    num = den = 0.0
    for _, g in d.groupby("group"):
        if len(g) < 10 or g["y"].nunique() < 2:
            continue
        num += len(g) * spearmanr(g[col], g["y"])[0]
        den += len(g)
    return num / den


def boot(df, pairs, n_boot, rng):
    groups = [np.flatnonzero(df["group"].to_numpy() == g) for g in df["group"].unique()]
    rows = []
    for a, b in pairs:
        pt = within_rho(df, a) - within_rho(df, b)
        ds = []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(ix, len(ix)) for ix in groups])
            ds.append(within_rho(df, a, idx) - within_rho(df, b, idx))
        ds = np.array(ds)
        rows.append({"Δ": f"{a} − {b}", "n": len(df), "delta": pt, "se": ds.std(),
                     "lo95": np.percentile(ds, 2.5), "hi95": np.percentile(ds, 97.5),
                     "P(>0)": (ds > 0).mean()})
    return pd.DataFrame(rows)


def partial_rho(g, x, ctrl="value_y14", y="y") -> float:
    r = g[[x, ctrl, y]].rank()
    res = lambda a: r[a] - np.polyval(np.polyfit(r[ctrl], r[a], 1), r[ctrl])
    return np.corrcoef(res(x), res(y))[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    df = build()
    print(f"cohort {len(df)} drafted players, classes {CLASSES}; "
          f"share with a 2nd contract {(df['y'] > 0).mean():.2f}")
    print(df.groupby("group").agg(n=("y", "size"), paid=("y", lambda s: (s > 0).mean()),
                                  mean_y=("y", "mean")).round(3).to_string())

    for arm, (feats, seeds) in ARMS.items():
        df[arm] = oof(df, feats, seeds)
    sk = df[df["group"].isin(SKILL)].reset_index(drop=True)

    pd.set_option("display.width", 200)
    print("\nwithin-group Spearman vs 2nd-contract pay (n-weighted):")
    print(pd.DataFrame({"all": {a: within_rho(df, a) for a in ["base", "base_alt", "av"]},
                        "skill": {a: within_rho(sk, a) for a in ARMS}}).round(4).to_string())
    print("\nper group:")
    print(pd.DataFrame({a: df.groupby("group").apply(lambda g: spearmanr(g[a], g["y"])[0])
                        for a in ARMS}).round(3).to_string())

    d1 = boot(df, [("av", "base"), ("base_alt", "base")], args.n_boot, rng)
    d2 = boot(sk, [("epa", "base"), ("av", "base"), ("base_alt", "base")], args.n_boot, rng)
    print("\nΔ, paired bootstrap (all groups):")
    print(d1.round(4).to_string(index=False))
    print("\nΔ, paired bootstrap (QB/RB/WR/TE):")
    print(d2.round(4).to_string(index=False))

    print("\npartial Spearman with 2nd-contract pay given value_y14:")
    pr = {}
    for g, gg in df.groupby("group"):
        row = {"n": len(gg), "value_y14 (raw rho)": spearmanr(gg["value_y14"], gg["y"])[0],
               "career_w_av": partial_rho(gg, "career_w_av"),
               "av_per_game": partial_rho(gg, "av_per_game")}
        if g in SKILL:
            row["epa_y14"] = partial_rho(gg, "epa_y14")
            row["epa_y34"] = partial_rho(gg, "epa_y34")
        pr[g] = row
    print(pd.DataFrame(pr).T.round(3).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "oof_quality_proxy.parquet", index=False)
    pd.concat([d1.assign(scope="all"), d2.assign(scope="skill")]).to_csv(OUT / "deltas.csv", index=False)


if __name__ == "__main__":
    main()
