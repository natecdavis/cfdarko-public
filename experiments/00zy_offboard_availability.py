"""
P(available at pick n) for OFF-board players (pre-registered 2026-11-01).

00zx models board players from big-board rank. Off-board players (no
hasan_rank) supply ~2% of picks in rounds 1-2, 5% to pick 150, 22% of picks
150-200 and 40% after 200, so late-round availability needs them.

Model: S(n) = 1 − p_d · F(n), p_d = P(drafted | pre-draft info), F = CDF of
the pick among drafted off-board players (pooled empirical, training
classes).

Pool: off-board players in the 8d universe whose class (last college
season + 1) is 2015-2022; features from 8d's out-of-fold predictions
(offboard_oof_predictions.parquet: p_nfl, usage_if_nfl), so nothing is
in-sample; combine invitation and measurables from step 10 projections
(known before the draft). Outcome: overall pick from bridge_players
(undrafted ⇔ Y > 262). Off-board draftees missing from the college data
(mostly FCS) are outside the pool and cannot be scored.

Arms (leave-one-class-out over 2015-22):
    A_base   p_d = training base rate by position
    B_logit  logistic (L2, C = 1) on logit p_nfl, log usage_if_nfl,
             combine_invited, position dummies
    C_lgb    LightGBM binary (num_leaves 7, lr 0.03, 300 trees,
             min_child 20, 5 seeds averaged) on p_nfl, usage_if_nfl,
             combine_invited, combine measurable z-scores (NaN allowed),
             position (categorical)
    B_Finv   B_logit with F stratified by combine invitation (reported arm)
Score: integrated Brier per player = mean over n = 1..262 of
(S(n) − 1[Y > n])^2, averaged over the pool.
PRIMARY (sequential, P from paired bootstrap of players within class, 1000):
    B_logit over A_base if P(B better) > 0.5; then C_lgb over the winner if
    P > 0.5; then B_Finv's F over pooled F if P > 0.5.
Reported: log loss and AUC of drafted; the same scores on the "shown"
subset (each class's top 100 off-board players by the 8d OOF score, the
ones the board tool lists); calibration of p_d by decile; F quantiles.

AMENDMENT (2026-11-01, after the first run, before the rank arm's results):
the first run found the 8d OOF p_nfl is badly mis-levelled for last season
2018 (mean 0.41 vs NFL rate 0.035), and production p_nfl for the current
pool averages 0.007 vs 0.02-0.04 historically. A raw-level model will be
mis-levelled in production. Added arm
    B_rank   logistic as B_logit but on level-free features: log rank within
             class (off-board pool) by p_nfl and by 8d score, combine_invited,
             position dummies; F stratified by combine invite.
Rule: use B_rank in production unless clearly worse than B_Finv
(P(B_rank better) >= 0.3), because its robustness to level shifts cannot be
shown by the LOCO test.

RESULT (2026-11-01): production = B_rank (amendment rule), F by invite.
Pool 38,112 off-board players, 2015-22, 263 drafted (0.69%); combine
invitees 501, 18.4% drafted; top-100-by-score subset 21.8% drafted.
    B_logit vs A_base   +0.94e-4 integrated Brier, P 1.00 (AUC 0.96 vs 0.54)
    C_lgb vs B_logit    −0.67e-4, P 0.00 (LightGBM over-fits 263 positives)
    B_Finv vs B_logit   +0.06e-4, P 0.81 → F stratified by combine invite
                        (invitee median pick 196 vs 219)
    B_rank vs B_Finv    +0.29e-4, P 0.90 full pool (logloss .0233 vs .0255,
                        AUC .971 vs .962); top-100 subset −18e-4, P 0.02
                        (over-predicts there: mean 0.21 vs 0.17)
B_Finv's level problem is real: it predicts 213 draftees for class 2019
(36 actual) because the 8d OOF p_nfl for last season 2018 averages 0.41.
B_rank predicts 19-44 per class (actual 17-47); top decile 0.061 vs 0.063.
Side finding: the 8d OOF p_nfl is mis-levelled for 2018 (and high for
2014/2017); within-class rank metrics used elsewhere are unaffected.

Usage: 00zy_offboard_availability.py [--n-boot 1000]
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402

OUT = PROCESSED_DIR / "_offboard_availability_20261101"
N_PICKS = 262
GRID = np.arange(1, N_PICKS + 1)
CLASSES = (2015, 2022)
POS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB", "ST"]
ZS = ["combine_forty_z", "combine_vertical_z", "combine_broad_jump_z", "combine_bench_z",
      "combine_cone_z", "combine_shuttle_z", "combine_ht_z", "combine_wt_z"]
ARMS = ["A_base", "B_logit", "C_lgb", "B_Finv", "B_rank"]
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def load():
    O = pd.read_parquet(PROCESSED_DIR / "offboard_models" / "offboard_oof_predictions.parquet")
    O = O[O["hasan_rank"].isna()].drop_duplicates("player_id", keep="last")
    P = pd.read_parquet(OUTPUT_DIR / "projections" / "latest.parquet",
                        columns=["player_id", "combine_invited"] + ZS).drop_duplicates("player_id")
    d = O.merge(P, on="player_id", how="left")
    d["cls"] = d["last_college_season_year"].astype(int) + 1
    d = d[d["cls"].between(*CLASSES)].copy()
    B = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    dr = B[B["overall_pick"].notna()].drop_duplicates("player_id").set_index("player_id")
    d["Y"] = d["player_id"].map(dr["overall_pick"]).fillna(N_PICKS + 1).clip(upper=N_PICKS + 1)
    d["drafted"] = (d["Y"] <= N_PICKS).astype(int)
    d["pos"] = np.where(d["pos_group"].isin(POS[:-1]), d["pos_group"], "ST")
    d["inv"] = d["combine_invited"].fillna(0).astype(float)
    p = d["p_nfl"].clip(1e-4, 1 - 1e-4)
    d["lp"] = np.log(p / (1 - p))
    d["lu"] = np.log(d["usage_if_nfl"].clip(lower=1e-3))
    d["shown"] = d.groupby("cls")["score"].rank(ascending=False, method="first") <= 100
    d["lr_p"] = np.log(d.groupby("cls")["p_nfl"].rank(ascending=False))
    d["lr_s"] = np.log(d.groupby("cls")["score"].rank(ascending=False))
    return d.reset_index(drop=True)


def ecdf(picks):
    picks = np.sort(np.asarray(picks))
    return np.searchsorted(picks, GRID, side="right") / len(picks)


def _X_logit(d):
    D = (d["pos"].to_numpy()[:, None] == np.array(POS)[None, :]).astype(float)
    return np.column_stack([d[["lp", "lu", "inv"]].to_numpy(), D])


def _X_rank(d):
    D = (d["pos"].to_numpy()[:, None] == np.array(POS)[None, :]).astype(float)
    return np.column_stack([d[["lr_p", "lr_s", "inv"]].to_numpy(), D])


def _X_lgb(d):
    X = d[["p_nfl", "usage_if_nfl", "inv"] + ZS].copy()
    X["pos"] = pd.Categorical(d["pos"], categories=POS)
    return X


def fold(tr, te):
    """Returns {arm: (p_d, F matrix n_te x 262)}."""
    F = ecdf(tr.loc[tr["drafted"] == 1, "Y"])
    Fm = np.tile(F, (len(te), 1))
    rate = tr.groupby("pos")["drafted"].mean()
    p_a = te["pos"].map(rate).fillna(tr["drafted"].mean()).to_numpy()
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(_X_logit(tr), tr["drafted"])
    p_b = lr.predict_proba(_X_logit(te))[:, 1]
    p_c = np.mean([lgb.LGBMClassifier(**LGB, random_state=s).fit(_X_lgb(tr), tr["drafted"])
                   .predict_proba(_X_lgb(te))[:, 1] for s in range(42, 47)], axis=0)
    dt = tr[tr["drafted"] == 1]
    F_inv = {g: ecdf(dt.loc[dt["inv"] == g, "Y"]) if (dt["inv"] == g).sum() >= 10 else F for g in (0.0, 1.0)}
    Fi = np.vstack([F_inv[g] for g in te["inv"].to_numpy()])
    lr2 = LogisticRegression(C=1.0, max_iter=2000).fit(_X_rank(tr), tr["drafted"])
    p_r = lr2.predict_proba(_X_rank(te))[:, 1]
    return {"A_base": (p_a, Fm), "B_logit": (p_b, Fm), "C_lgb": (p_c, Fm), "B_Finv": (p_b, Fi),
            "B_rank": (p_r, Fi)}


def ibrier(p, F, y):
    S = 1 - p[:, None] * F
    return np.mean((S - (y[:, None] > GRID[None, :])) ** 2, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(0)
    d = load()
    print(f"off-board pool {len(d)}, classes {CLASSES}, drafted {d['drafted'].sum()} "
          f"({d['drafted'].mean():.2%}); combine invitees {int(d['inv'].sum())}, of whom drafted "
          f"{d.loc[d['inv'] == 1, 'drafted'].mean():.1%}; shown-set drafted {d.loc[d['shown'], 'drafted'].mean():.1%}")
    print(d.groupby("cls").agg(pool=("Y", "size"), drafted=("drafted", "sum"), inv=("inv", "sum")).T.to_string())

    pd_ = {a: np.empty(len(d)) for a in ARMS}
    ib = {a: np.empty(len(d)) for a in ARMS}
    y = d["Y"].to_numpy()
    for t in range(CLASSES[0], CLASSES[1] + 1):
        m = (d["cls"] == t).to_numpy()
        for a, (p, F) in fold(d[~m], d[m]).items():
            pd_[a][m] = p
            ib[a][m] = ibrier(p, F, y[m])
    sc = d[["player_id", "cls", "pos", "drafted", "shown", "Y"]].copy()
    for a in ARMS:
        sc[f"p_{a}"], sc[f"ib_{a}"] = pd_[a], ib[a]
    sc.to_parquet(OUT / "scores.parquet", index=False)

    def summary(s):
        return pd.DataFrame({a: {"ibrier_x1e4": 1e4 * s[f"ib_{a}"].mean(),
                                 "logloss": log_loss(s["drafted"], np.clip(s[f"p_{a}"], 1e-6, 1 - 1e-6)),
                                 "auc": roc_auc_score(s["drafted"], s[f"p_{a}"]),
                                 "mean_p": s[f"p_{a}"].mean(), "rate": s["drafted"].mean()} for a in ARMS}).T
    print("\nall off-board pool:")
    print(summary(sc).round(5).to_string())
    print("\nshown subset (top 100 off-board by 8d score per class):")
    print(summary(sc[sc["shown"]]).round(5).to_string())

    groups = [g.index.to_numpy() for _, g in sc.groupby("cls")]
    boot_idx = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(args.n_boot)]

    def compare(a, base, mask=None):
        diff = (sc[f"ib_{base}"] - sc[f"ib_{a}"]).to_numpy()  # > 0: a better
        if mask is not None:
            diff = np.where(mask, diff, np.nan)
        bs = np.array([np.nanmean(diff[i]) for i in boot_idx])
        return {"arm vs base": f"{a} vs {base}", "gain_x1e4": 1e4 * np.nanmean(diff),
                "lo95": 1e4 * np.percentile(bs, 2.5), "hi95": 1e4 * np.percentile(bs, 97.5),
                "P(arm better)": (bs > 0).mean()}

    win = "B_logit" if compare("B_logit", "A_base")["P(arm better)"] > 0.5 else "A_base"
    win2 = "C_lgb" if compare("C_lgb", win)["P(arm better)"] > 0.5 else win
    rows = [compare("B_logit", "A_base"), compare("C_lgb", win), compare("C_lgb", "B_logit"),
            compare("B_Finv", "B_logit"), compare("B_rank", "B_Finv")]
    rows_s = [dict(compare(a, b, sc["shown"].to_numpy()), subset="shown")
              for a, b in (("B_logit", "A_base"), ("C_lgb", "B_logit"), ("B_Finv", "B_logit"),
                                ("B_rank", "B_Finv"))]
    D = pd.DataFrame(rows + rows_s)
    print("\nintegrated Brier gains (x1e4, > 0 = arm better):")
    print(D.round(4).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)
    finv = compare("B_Finv", "B_logit")["P(arm better)"] > 0.5
    print(f"\nDECISION: p_d model = {win2}; F = {'stratified by combine invite' if finv else 'pooled'}")
    pr = compare("B_rank", "B_Finv")["P(arm better)"]
    print(f"AMENDMENT: B_rank vs B_Finv P {pr:.3f} → production uses {'B_rank' if pr >= 0.3 else 'B_Finv'}")
    print("p_d vs realized by class (B_Finv / B_rank):")
    print(sc.groupby("cls")[["p_B_Finv", "p_B_rank", "drafted"]].sum().round(1).T.to_string())
    win2 = "B_rank" if pr >= 0.3 else win2

    s = sc.assign(dec=pd.qcut(sc[f"p_{win2}"].rank(method="first"), 10, labels=False))
    print(f"\ncalibration of p_d ({win2}) by decile:")
    print(s.groupby("dec")[[f"p_{win2}", "drafted"]].mean().round(4).T.to_string())
    dt = d[d["drafted"] == 1]
    print("\nF: pick quantiles among drafted off-board players (10/25/50/75/90%):")
    for g, lab in ((None, "all"), (1.0, "combine invitee"), (0.0, "not invited")):
        yy = dt["Y"] if g is None else dt.loc[dt["inv"] == g, "Y"]
        print(f"  {lab:16s} n={len(yy):3d}  " + " ".join(f"{q:.0f}" for q in np.percentile(yy, [10, 25, 50, 75, 90])))


if __name__ == "__main__":
    main()
