"""
Per-player adjustment on top of the board curve, in rookie-deal value
(pre-registered 2026-11-01).

Today a board player's value is E[rookie value | log hasan_rank, position]
(production ShrunkIso curve): every player at the same rank and position
gets the same value, and positions are shrunk toward the pooled curve
(QB under-priced ~17-29%, LB/DL over-priced out of time; 00zs, 00zt).
Earlier attempts to beat the board (00zb, board-row weighting, two-part)
failed 95% gates on usage / APY targets. This retests the idea on the
product currency and under the expected-value decision rule.

Model: residual r = realized rookie value (value_y14 v3.1, 0 if never NFL)
− board curve. Training residuals use an inner leave-one-class-out curve
(each training class scored by a curve fit on the other training classes),
so they look like the test residuals. Prediction = curve (fit on all
training classes) + predicted residual.
Features (player information):
    model scores: deployed 8b impact OOF and 8d off-board OOF score, each as
        a percentile within (class × position) among board players, and
        each minus the board's own within-(class × position) percentile
    the 8d feature set (55 shared features: production, PFF side grades,
        team context, recruiting, age_at_draft, height/weight, combine
        forty/broad z, invite)
    combine extras: vertical, bench, cone, shuttle, arm, hand z; skipped-
        forty and skipped-agility flags for invitees
    structure: log rank, position one-hot, curve value
Arms (board players; eval folds as 00zs: classes 2017-19 trained on
2014-16, classes 2020-22 trained on 2014-19):
    curve       production board curve (baseline)
    level       curve + per-position mean training residual, shrunk
                n/(n+50): a position-level correction only
    ridge       curve + ridge on standardized features (median-imputed, with
                missing flags); alpha by leave-one-class-out CV  (PRIMARY)
    lgb         curve + LightGBM on the same features (small, regularized)
    ridge_perm, lgb_perm  placebos: player-information features permuted
                within (class × position) in train and test; structure kept
Metrics per class, averaged: value captured by the top 32 / top 100 board
players by each list (cross-position); MSE; cross-position Spearman.
PRIMARY: ridge − curve, top-100 value captured. Prior: skeptical (every
earlier board-beating attempt was null), cost small. ADOPT if P(> 0) > 0.7
(paired player bootstrap within class, 1000) AND MSE not clearly worse
(P(MSE gain > 0) >= 0.3). Reported: lgb, level, placebos (arm − perm =
player information beyond structure), calibration by position, the
features ridge leans on.

RESULT (2026-11-01): KEEP curve (pre-registered primary fails), but the
player information is real. 1,706 board players, classes 2017-22.
    ridge − curve   top100 −0.0010 [−0.0032, +0.0014] P 0.21; MSE P 0.18;
                    top32 +0.0072 P 0.95; rho +0.002
    lgb − curve     top100 −0.0005 P 0.37; top32 +0.0062 P 0.86;
                    rho +0.0135 P 0.92; MSE gain P 0.85
    level − curve   top32 +0.0038; top100 −0.0003; rho −0.020 (P 0)
    vs placebos (player features shuffled within class × position):
                    lgb − lgb_perm  rho +0.063, top100 +0.0022 (P 0.96),
                    MSE P 0.99; ridge − ridge_perm rho +0.029 (P 1.00)
Reading: shuffling the player features clearly hurts, so they carry
signal beyond rank and position, but adding them to the board curve barely
moves value captured in the top 100 (≈ 0) and helps a little at the top
32 (≈ +0.006-0.007 of cap per player, ~$2M over 4 years), part of which
the position-level arm gets alone (QB). The board already prices most of
what the features know. Ridge leans on: our model vs the board (8b, 8d
percentile minus board percentile, +), age (−, and missing age −), cone
time (slower −) and skipped agility (−), hand size (+), broad jump (+),
recruiting composite (− given rank). QB stays under-predicted in every arm
(−0.012 to −0.036).

Usage: 00zu_board_player_adjust.py [--n-boot 1000]
"""
import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from config import PROCESSED_DIR  # noqa: E402
from src.models.merged_board import ShrunkIso, K_BOARD  # noqa: E402
from src.models.offboard_model import pid  # noqa: E402

spec = importlib.util.spec_from_file_location("zs", ROOT / "experiments/00zs_value_currency_board.py")
zs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zs)
zd = importlib.import_module("00zd_offboard_model")

OUT = PROCESSED_DIR / "_board_adjust_20261101"
FOLDS = [(2015, 2016, 2018), (2018, 2019, 2021)]
POS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
COMBINE_X = ["combine_vertical_z", "combine_bench_z", "combine_cone_z", "combine_shuttle_z",
             "combine_arm_z", "combine_hand_z"]
ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000]
LGB = dict(n_estimators=250, learning_rate=0.02, num_leaves=4, min_child_samples=30,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
           random_state=42, verbose=-1)
K_LEVEL = 50
ARMS = ["curve", "level", "ridge", "lgb", "ridge_perm", "lgb_perm"]


def load():
    E = zs.load()
    E = zd._deployed(E)
    b = E[E["board"] & E["pos_group"].isin(POS)].copy()
    rows = pd.read_parquet(OUT / "rows_8d.parquet")
    feats = pd.read_csv(OUT / "feats_8d.csv").iloc[:, 0].tolist()
    feats = [f for f in feats if f not in ("pos_code", "board_era", "hasan_rank")]
    rows = rows.drop_duplicates(["player_id", "pos_group"])[["player_id", "pos_group"] + feats]
    b = b.merge(rows, on=["player_id", "pos_group"], how="left")
    cm = pd.read_parquet(PROCESSED_DIR / "combine_metrics.parquet")
    cm = cm.assign(player_id=pid(cm["player_id"])).drop_duplicates("player_id")
    b = b.merge(cm[["player_id", "combine_forty", "combine_cone", "combine_shuttle"] + COMBINE_X],
                on="player_id", how="left")
    inv = b["combine_invited"].fillna(0) > 0
    b["skip_forty"] = (inv & b["combine_forty"].isna()).astype(float)
    b["skip_agility"] = (inv & b["combine_cone"].isna() & b["combine_shuttle"].isna()).astype(float)
    b = b.drop(columns=["combine_forty", "combine_cone", "combine_shuttle"])
    grp = b.groupby(["cls", "pos_group"])
    b["board_pct"] = grp["hasan_rank"].rank(pct=True, ascending=False)
    for s, name in (("dep_impact", "m8b"), ("base", "m8d")):
        b[f"{name}_pct"] = grp[s].rank(pct=True)
        b[f"{name}_minus_board"] = b[f"{name}_pct"] - b["board_pct"]
    b["lr"] = np.log(b["hasan_rank"])
    player = (feats + COMBINE_X + ["skip_forty", "skip_agility", "m8b_pct", "m8b_minus_board",
                                   "m8d_pct", "m8d_minus_board"])
    print(f"board players {len(b)}, classes {b['cls'].min()}-{b['cls'].max()}; "
          f"player features {len(player)}; 8b OOF coverage {b['dep_impact'].notna().mean():.3f}")
    return b.reset_index(drop=True), player


def curve_fit(tr):
    return ShrunkIso(increasing=False, k=K_BOARD).fit(tr["lr"], tr["rv"], tr["pos_group"])


def inner_residuals(tr):
    out = pd.Series(np.nan, index=tr.index)
    cur = pd.Series(np.nan, index=tr.index)
    for c in tr["cls"].unique():
        m = tr["cls"] == c
        cal = curve_fit(tr[~m])
        cur[m] = cal.predict(tr.loc[m, "lr"], tr.loc[m, "pos_group"])
    out[:] = tr["rv"] - cur
    return out, cur


def design(df, player, med, curve):
    X = df[player].astype(float).copy()
    miss = X.isna().astype(float).add_suffix("_na")
    miss = miss.loc[:, [c for c in miss.columns if c.replace("_na", "") in med.index]]
    X = X.fillna(med).fillna(0.0)   # all-missing-in-train columns → constant
    X["lr"], X["curve"] = df["lr"].to_numpy(), np.asarray(curve)
    for g in POS:
        X[f"pos_{g}"] = (df["pos_group"] == g).astype(float).to_numpy()
    return pd.concat([X, miss], axis=1)


def permute(df, cols, seed):
    rng = np.random.default_rng(seed)
    out = df.copy()
    for _, idx in df.groupby(["cls", "pos_group"]).groups.items():
        p = rng.permutation(len(idx))
        out.loc[idx, cols] = df.loc[idx, cols].to_numpy()[p]
    return out


def fit_arms(tr, te, player):
    res_tr, cur_tr = inner_residuals(tr)
    cal = curve_fit(tr)
    cur_te = cal.predict(te["lr"], te["pos_group"])
    out = {"curve": cur_te}
    lv = {g: res_tr[tr["pos_group"] == g].sum() / ((tr["pos_group"] == g).sum() + K_LEVEL) for g in POS}
    out["level"] = cur_te + te["pos_group"].map(lv).to_numpy()
    info = {}
    for tag, (a, b) in {"": (tr, te), "_perm": (permute(tr, player, 1), permute(te, player, 2))}.items():
        med = a[player].astype(float).median()
        Xa, Xb = design(a, player, med, cur_tr), design(b, player, med, cur_te)
        mu, sd = Xa.mean(), Xa.std().replace(0, 1)
        Za, Zb = (Xa - mu) / sd, (Xb - mu) / sd
        # alpha by leave-one-class-out
        err = {}
        for al in ALPHAS:
            e = 0.0
            for c in a["cls"].unique():
                m = (a["cls"] == c).to_numpy()
                r = Ridge(alpha=al).fit(Za[~m], res_tr[~m])
                e += ((r.predict(Za[m]) - res_tr[m]) ** 2).sum()
            err[al] = e
        al = min(err, key=err.get)
        r = Ridge(alpha=al).fit(Za, res_tr)
        out[f"ridge{tag}"] = cur_te + r.predict(Zb)
        La = a[player].astype(float).assign(lr=a["lr"].to_numpy(), curve=np.asarray(cur_tr),
                                            pos=pd.Categorical(a["pos_group"], categories=POS))
        Lb = b[player].astype(float).assign(lr=b["lr"].to_numpy(), curve=np.asarray(cur_te),
                                            pos=pd.Categorical(b["pos_group"], categories=POS))
        g = lgb.LGBMRegressor(**LGB).fit(La, res_tr)
        out[f"lgb{tag}"] = cur_te + g.predict(Lb)
        if tag == "":
            info = {"alpha": al, "coef": pd.Series(r.coef_, index=Za.columns)}
    return out, info


def metrics(V, idx=None):
    d = V if idx is None else V.iloc[idx]
    by = list(d.groupby("cls"))
    r = {}
    for a in ARMS:
        r[a] = {"top32": np.mean([g.nlargest(32, a)["rv"].mean() for _, g in by]),
                "top100": np.mean([g.nlargest(100, a)["rv"].mean() for _, g in by]),
                "neg_mse": -np.mean((d[a] - d["rv"]) ** 2),
                "rho": np.mean([spearmanr(g[a], g["rv"])[0] for _, g in by])}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)
    b, player = load()
    parts, coefs = [], []
    for train_end, v0, v1 in FOLDS:
        tr = b[b["last_college_season_year"] <= train_end].reset_index(drop=True)
        te = b[b["last_college_season_year"].between(v0, v1)].copy().reset_index(drop=True)
        preds, info = fit_arms(tr, te, player)
        for a, p in preds.items():
            te[a] = p
        print(f"fold {v0}: train {len(tr)} (classes {tr['cls'].min()}-{tr['cls'].max()}), "
              f"test {len(te)}; ridge alpha {info['alpha']}")
        coefs.append(info["coef"].rename(v0))
        parts.append(te)
    V = pd.concat(parts, ignore_index=True)
    V.to_parquet(OUT / "adjust_oof.parquet", index=False)

    pt = metrics(V)
    groups = [g.index.to_numpy() for _, g in V.groupby("cls")]
    boots = [metrics(V, np.concatenate([rng.choice(g, len(g)) for g in groups])) for _ in range(args.n_boot)]
    print(f"\nboard players scored: {len(V)}, classes {V['cls'].min()}-{V['cls'].max()}")
    print(pd.DataFrame(pt).T.round(4).to_string())
    rows = []
    for a, base in (("ridge", "curve"), ("lgb", "curve"), ("level", "curve"), ("ridge", "ridge_perm"),
                    ("lgb", "lgb_perm"), ("ridge_perm", "curve"), ("ridge", "level")):
        for k in pt[a]:
            ds = np.array([bt[a][k] - bt[base][k] for bt in boots])
            rows.append({"Δ": f"{a} − {base}", "metric": k, "delta": pt[a][k] - pt[base][k],
                         "lo95": np.percentile(ds, 2.5), "hi95": np.percentile(ds, 97.5),
                         "P(>0)": (ds > 0).mean()})
    D = pd.DataFrame(rows)
    print(D.round(4).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)

    p = D[(D["Δ"] == "ridge − curve") & (D["metric"] == "top100")].iloc[0]
    m = D[(D["Δ"] == "ridge − curve") & (D["metric"] == "neg_mse")].iloc[0]
    ok = p["P(>0)"] > 0.7 and m["P(>0)"] >= 0.3
    print(f"\nDECISION: ridge − curve top100 {p['delta']:+.4f} [{p['lo95']:+.4f}, {p['hi95']:+.4f}] "
          f"P(>0) {p['P(>0)']:.3f}; MSE P(gain>0) {m['P(>0)']:.3f} → {'ADOPT' if ok else 'KEEP curve'}")

    print("\nlevel bias (mean predicted − realized) by position:")
    print(V.groupby("pos_group").apply(lambda g: pd.Series(
        {"n": len(g), "actual": g["rv"].mean(), **{a: g[a].mean() - g["rv"].mean() for a in ARMS[:4]}}))
        .round(4).to_string())
    C = pd.concat(coefs, axis=1)
    C["mean"] = C.mean(axis=1)
    print("\nridge: largest standardized coefficients (mean over folds):")
    print(C.reindex(C["mean"].abs().sort_values(ascending=False).index).head(20).round(5).to_string())


if __name__ == "__main__":
    main()
