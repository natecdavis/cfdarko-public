"""
Off-board model: rank the players with no consensus board rank.

RESULT (2026-10-30): the usage-target two-stage model TIES the deployed
model. Changing the target and the model doesn't help, so the binding
constraint is the data. 37,977 players, 35,640 off-board, 1,050 off-board
NFL players, 247 off-board contributors. Δ vs dep_impact [95%]:
    pre   auc_ob +0.002 [-0.005,+0.009]   rho_ob_nfl -0.006 [-0.069,+0.054]
          auc_udfa +0.000                 rho_udfa_nfl -0.045 [-0.103,+0.021]
    placebo (pre_43 − pre) moved rho_ob_nfl -0.014, rho_udfa_nfl -0.008
    post − pre  rho_ob_nfl +0.064 [-0.006,+0.130]  auc_udfa -0.016 [-0.045,-0.000]
Ranking undrafted NFL players is ~0 for every arm (rho 0.02-0.08). Draft
status helps only by separating day-3 picks from UDFAs. It costs AUC inside
the undrafted pool, which is the one a team signs from. Top-K: pre's top
10 / 25 off-board per class contribute 39% / 31%, vs 36% / 27% for
dep_impact and 29% for hasan 151-250. The gain comes from stage 1 (top-10
NFL rate 96% vs 80%), and the placebo moved top-10 by 4.5 pts, so it is
not established. Scale: pre puts 135 off-board players per class above the
median hasan 251-300 player, too many (12.5% contribute vs the board's
16%). dep_impact puts 45 there, and they contribute at 20%.

RERUN after the bridge fixes (7f44722, 20740be) and a --no-tune rebuild of
steps 03-10: off-board NFL players 1,050 -> 1,156, contributors 247 -> 272.
The added players are small-school and late-linked, so every arm's AUC
drops ~0.005-0.015. The conclusion holds. pre vs dep_impact: auc_ob +0.008,
auc_udfa +0.012, but the placebo moved +0.004 each, and rho_ob_nfl -0.051
(ns). post − pre rho_ob_nfl +0.104 [+0.030, +0.171] (placebo +0.008), but
post − dep_impact only +0.053 (ns), and auc_udfa still -0.012 vs pre.

Why (2026-10-30). For players with a hasan_rank, nothing cfDARKO builds beats
the board ([[project_board_is_ceiling]]). Off-board players are the open
problem, and the deployed 8b global model splits on them in a specific way.
Measured on classes 2014-22:
    AUC for reaching the NFL, off-board        0.966 (impact) / 0.984 (role)
    Spearman with y3-4 snap share, off-board NFL   0.087 / 0.001  (board 0.40)
It finds the 2.4% of off-board players who make a roster but can't rank them
once they're there. The impact label (second-contract APY) can't teach that:
only 57 off-board players in eight classes got a >2%-cap deal, and career
w_AV is 0 for every UDFA. Usage can teach it: y3-4 snap share (usage_y34)
covers every NFL player, UDFAs included.

    score = P(reach NFL)  ×  E[usage_y34 | NFL]   = expected y3-4 snap share
    stage 1  pooled classifier, all rows, y = is_bridge
    stage 2  pooled regressor, NFL rows only, y = usage_y34 (NaN → 0)
Both stages are pooled across positions: the common feature list (every
non-K/P head shares it) + pos_code + board_era. board_era (last college
season >= 2013) separates "off the board" from "before boards existed";
hasan_rank is NaN in both. hasan_rank is monotone in both stages. Every
player is scored, board players included, so an off-board player can be put
next to hasan #250 on one scale. Untuned.

Arms (OOF on the step-8 tuning folds; the holdout fold is never touched):
    pre       pre-draft features only
    pre_43    same, seed 43 (placebo: the noise floor for every Δ)
    post      + drafted flag and draft round (undrafted = 8, monotone).
              Known the night the draft ends, so it can re-rank the
              undrafted pool for UDFA signing
    dep_impact / dep_role   the deployed 8b global OOF (impact_score ships
              from 8b, [[feedback_deployed_baseline]]), on the same rows

Scoring cohort: last college season 2013-2021 (draft classes 2014-22, board
era, y3-4 observable), K/P excluded. Outcomes: contributor (usage_y34 >=
0.25) and starter (>= 0.5). Metrics are computed within draft class or
within (group × fold), never pooled across folds, since the arms' OOF scale
differs by fold:
    auc_ob       AUC for contributor among off-board players, per class
                 across positions, averaged over classes
    auc_ob_pos   same, per (group × class) with >= 5 contributors
    rho_ob_nfl   Spearman with usage_y34 among off-board NFL players, per
                 (group × fold) with >= 20 players
    auc_udfa     AUC for contributor among UNDRAFTED players (board players
                 included): the pool a team signs from after the draft
    rho_udfa_nfl Spearman with usage_y34 among undrafted NFL players
plus a top-K table (off-board top 10/25/50 per class vs actual hasan 151-300
rates) and a crossover count (off-board players scoring above the class
median of hasan 251-300). Paired bootstrap (players resampled within class)
for Δ vs dep_impact and post vs pre. The bootstrap resamples players, not
seeds, so read every Δ against the pre_43 placebo too ([[feedback_ab_placebo_arm]]).

OOF files are written in the 00u format (model_name "impact"), so
`00u_value_target_referee_ab.py --within-class --subset offboard --oof ...`
can referee them against late snap share / late APY.

Usage:
    00zd_offboard_model.py --out-dir DIR [--n-boot 500]
"""
import argparse
import importlib
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from config import PROCESSED_DIR, POSITION_GROUPS, LGBM_PARAMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
GROUPS = [g for g in POSITION_GROUPS if g not in ("K", "P")]
STAGE1_PARAMS = {**LGBM_PARAMS, "learning_rate": 0.05, "n_estimators": 2000,
                 "num_leaves": 15, "min_child_samples": 50}
STAGE2_PARAMS = {**LGBM_PARAMS, "learning_rate": 0.03, "n_estimators": 3000,
                 "num_leaves": 15, "min_child_samples": 20}
DEPLOYED_OOF = PROCESSED_DIR / "global_draft_models" / "global_oof_predictions.parquet"
CONTRIB, STARTER = 0.25, 0.50
EVAL_YEARS = (2013, 2021)            # last college season → draft classes 2014-22
BOARD_ERA_START = 2013               # last college season of the first board class
ARMS = ["pre", "pre_43", "post"]
POST_FEATURES = ["drafted", "draft_round"]
ID_COLS = ["player_id", "pos_group", "last_college_season_year"]


def _pid(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


def _fit(step8, cls, params, tr, cols, y):
    mono = [-1 if c in ("hasan_rank", "draft_round") else 0 for c in cols]
    mk = lambda: cls(**{**params, "monotone_constraints": mono})
    return step8._fit_fold_honest(
        mk, tr[cols].to_numpy(dtype=np.float32, na_value=np.nan), y,
        tr["last_college_season_year"].fillna(0).values)


def _X(df, cols):
    return df[cols].to_numpy(dtype=np.float32, na_value=np.nan)


def _build_rows(step8, snapshot=None, labels=None, bridge=None, extra=()):
    """Pooled rows + feature list. snapshot / labels / bridge default to the
    pipeline's; 00ze passes copies augmented with FCS identities, and `extra`
    snapshot columns to pool beyond the shared head features."""
    if snapshot is None:
        snapshot = step8.assemble_snapshot(include_derived=True, logger=step8.logger)
    if labels is None:
        labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    frames, common = [], None
    for g in GROUPS:
        cols, df = step8._resolve_fit_features(snapshot, labels, g, "impact_target", "impact")
        common = cols if common is None else [c for c in common if c in cols]
        frames.append(df.assign(pos_group=g))
    common = common + [c for c in extra if c not in common]
    keep = ID_COLS + [c for c in common if c not in ID_COLS]
    if "hasan_rank" not in keep:
        keep.append("hasan_rank")
    rows = pd.concat([f[[c for c in keep if c in f.columns]] for f in frames],
                     ignore_index=True)
    rows["player_id"] = _pid(rows["player_id"])

    lab = labels[labels["player_id"].notna()].assign(player_id=lambda d: _pid(d["player_id"]))
    lab = lab.drop_duplicates("player_id")[["player_id", "is_bridge", "usage_y34"]]
    bp = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet") if bridge is None else bridge
    bp = bp[bp["player_id"].notna()].assign(player_id=lambda d: _pid(d["player_id"]))
    bp = bp.drop_duplicates("player_id")[["player_id", "round"]]
    rows = rows.merge(lab, on="player_id", how="left").merge(bp, on="player_id", how="left")

    rows["nfl"] = rows["is_bridge"].fillna(False).astype(bool)
    rows["usage"] = np.where(rows["nfl"], rows["usage_y34"].fillna(0.0), 0.0)
    rows["drafted"] = rows["round"].notna().astype(float)
    rows["draft_round"] = rows["round"].fillna(8.0)
    rows["pos_code"] = rows["pos_group"].map({g: i for i, g in enumerate(GROUPS)})
    rows["board_era"] = (rows["last_college_season_year"] >= BOARD_ERA_START).astype(float)
    feats = [c for c in common if c not in ID_COLS] + ["pos_code", "board_era"]
    logging.info("rows %d, NFL %d, drafted %d; pooled features %d",
                 len(rows), int(rows["nfl"].sum()), int(rows["drafted"].sum()), len(feats))
    return rows, feats


def _run_arms(step8, rows, feats, folds):
    yr = rows["last_college_season_year"].fillna(0)
    fold_of = pd.Series(np.nan, index=rows.index)
    out = {a: {k: pd.Series(np.nan, index=rows.index) for k in ("p1", "m2")} for a in ARMS}
    specs = {"pre": (feats, 42), "pre_43": (feats, 43), "post": (feats + POST_FEATURES, 42)}
    for train_end, val_start, val_end in folds:
        tr_m, va_m = yr <= train_end, yr.between(val_start, val_end)
        fold_of[va_m] = val_start
        tr, va = rows[tr_m], rows[va_m]
        nfl = tr[tr["nfl"]]
        for arm, (cols, seed) in specs.items():
            clf, b1 = _fit(step8, lgb.LGBMClassifier, {**STAGE1_PARAMS, "random_state": seed},
                           tr, cols, tr["nfl"].astype(int).values)
            reg, b2 = _fit(step8, lgb.LGBMRegressor, {**STAGE2_PARAMS, "random_state": seed},
                           nfl, cols, nfl["usage"].values)
            out[arm]["p1"][va.index] = clf.predict_proba(_X(va, cols))[:, 1]
            out[arm]["m2"][va.index] = np.clip(reg.predict(_X(va, cols)), 0, 1)
            logging.info("fold %d %s: stage 1 %d rows / %d trees; stage 2 %d NFL rows / %d trees",
                         val_start, arm, len(tr), b1, len(nfl), b2)
    return out, fold_of


def _deployed(rows_keys):
    dep = pd.read_parquet(DEPLOYED_OOF)
    dep = dep.assign(player_id=_pid(dep["player_id"]))
    out = rows_keys.copy()
    for m, name in [("impact_apy", "dep_impact"), ("role", "dep_role")]:
        d = dep[dep["model_name"] == m].drop_duplicates(["player_id", "val_start"])
        out = out.merge(d[["player_id", "val_start", "pred"]].rename(columns={"pred": name}),
                        on=["player_id", "val_start"], how="left")
    return out


def _auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def _metrics(E, arms):
    """Headline metrics for every arm on frame E (one row per player)."""
    ob, ud = E[~E["board"]], E[E["drafted"] == 0]
    res = {}
    for a in arms:
        r = {}
        r["auc_ob"] = np.nanmean([_auc(g["contrib"].values, g[a].values)
                                  for _, g in ob.groupby("cls")])
        r["auc_ob_pos"] = np.nanmean([_auc(g["contrib"].values, g[a].values)
                                      for _, g in ob.groupby(["pos_group", "cls"])
                                      if g["contrib"].sum() >= 5])
        r["rho_ob_nfl"] = np.nanmean([spearmanr(g[a], g["usage"])[0]
                                      for _, g in ob[ob["nfl"]].groupby(["pos_group", "val_start"])
                                      if len(g) >= 20])
        r["auc_udfa"] = np.nanmean([_auc(g["contrib"].values, g[a].values)
                                    for _, g in ud.groupby("cls")])
        r["rho_udfa_nfl"] = np.nanmean([spearmanr(g[a], g["usage"])[0]
                                        for _, g in ud[ud["nfl"]].groupby(["pos_group", "val_start"])
                                        if len(g) >= 20])
        res[a] = r
    return res


def _topk_table(E, arms):
    rows = []
    for lo, hi in [(151, 200), (201, 250), (251, 300)]:
        t = E[E["hasan_rank"].between(lo, hi)]
        rows.append({"group": f"hasan {lo}-{hi} (actual)", "per_class": len(t) / E["cls"].nunique(),
                     "nfl": t["nfl"].mean(), "contrib": t["contrib"].mean(),
                     "starter": t["starter"].mean()})
    ob = E[~E["board"]]
    for a in arms:
        rk = ob.groupby("cls")[a].rank(ascending=False, method="first")
        for k in (10, 25, 50):
            t = ob[rk <= k]
            rows.append({"group": f"{a} off-board top {k}", "per_class": k, "nfl": t["nfl"].mean(),
                         "contrib": t["contrib"].mean(), "starter": t["starter"].mean()})
        thr = E[E["hasan_rank"].between(251, 300)].groupby("cls")[a].median()
        t = ob[ob[a] > ob["cls"].map(thr)]
        rows.append({"group": f"{a} off-board > median hasan 251-300",
                     "per_class": len(t) / E["cls"].nunique(), "nfl": t["nfl"].mean(),
                     "contrib": t["contrib"].mean(), "starter": t["starter"].mean()})
    return pd.DataFrame(rows).set_index("group").round(3)


def _bootstrap(E, arms, pairs, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    E = E.reset_index(drop=True)
    by_cls = [g.index.to_numpy() for _, g in E.groupby("cls")]
    point = _metrics(E, arms)
    boots = []
    for i in range(n_boot):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in by_cls])
        boots.append(_metrics(E.iloc[ix], arms))
        if (i + 1) % 100 == 0:
            logging.info("bootstrap %d/%d", i + 1, n_boot)
    lines = []
    for a, b in pairs:
        for k in point[a]:
            d = np.array([bt[a][k] - bt[b][k] for bt in boots])
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            lines.append({"Δ": f"{a} − {b}", "metric": k,
                          "delta": round(point[a][k] - point[b][k], 4),
                          "ci": f"[{lo:+.3f}, {hi:+.3f}]"})
    return point, pd.DataFrame(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step8 = importlib.import_module("08_draft_models")
    folds, _ = step8._split_holdout(step8._expanding_window_folds())
    rows, feats = _build_rows(step8)
    out, fold_of = _run_arms(step8, rows, feats, folds)

    base = rows[ID_COLS + ["hasan_rank", "nfl", "usage", "drafted"]].assign(val_start=fold_of)
    for a in ARMS:
        base[f"{a}_p1"], base[f"{a}_m2"] = out[a]["p1"], out[a]["m2"]
        base[a] = out[a]["p1"] * out[a]["m2"]
    base = base.dropna(subset=["val_start", "pre"])
    base["val_start"] = base["val_start"].astype(int)
    base.to_parquet(args.out_dir / "oof_all_arms.parquet", index=False)
    for a in ARMS:
        base.assign(pred=base[a], y=base["usage"], model_name="impact")[
            ["player_id", "pos_group", "val_start", "y", "pred", "model_name"]
        ].to_parquet(args.out_dir / f"oof_{a}.parquet", index=False)

    E = _deployed(base)
    E = E[E["last_college_season_year"].between(*EVAL_YEARS)]
    E = E.dropna(subset=["dep_impact", "dep_role"])
    E = E.drop_duplicates("player_id", keep="last")
    E["cls"] = E["last_college_season_year"].astype(int) + 1
    E["board"] = E["hasan_rank"].notna()
    E["contrib"] = (E["usage"] >= CONTRIB).astype(int)
    E["starter"] = (E["usage"] >= STARTER).astype(int)
    ob = E[~E["board"]]
    logging.info("scoring cohort %d players (classes %d-%d): off-board %d, off-board NFL %d, "
                 "off-board contributors %d; undrafted %d", len(E), E["cls"].min(),
                 E["cls"].max(), len(ob), int(ob["nfl"].sum()), int(ob["contrib"].sum()),
                 int((E["drafted"] == 0).sum()))

    arms = ARMS + ["dep_impact", "dep_role"]
    pairs = [("pre", "dep_impact"), ("pre_43", "pre"), ("post", "pre"), ("post", "dep_impact")]
    point, deltas = _bootstrap(E, arms, pairs, args.n_boot)

    print("\nheadline metrics (within class / within group × fold):")
    print(pd.DataFrame(point).T.round(3).to_string())
    print("\nΔ with paired bootstrap 95% CI (players resampled within class; "
          "read against the pre_43 − pre placebo row):")
    print(deltas.to_string(index=False))
    print("\ntop-K vs the bottom of the board (rates, classes 2014-22):")
    print(_topk_table(E, arms).to_string())
    print("\nstage diagnostics, off-board: AUC(p1 → NFL) / Spearman(m2, usage | NFL), per class mean")
    for a in ARMS:
        a1 = np.nanmean([_auc(g["nfl"].astype(int).values, g[f"{a}_p1"].values)
                         for _, g in ob.groupby("cls")])
        r2 = np.nanmean([spearmanr(g[f"{a}_m2"], g["usage"])[0]
                         for _, g in ob[ob["nfl"]].groupby(["pos_group", "val_start"]) if len(g) >= 20])
        print(f"  {a:7s} stage 1 AUC {a1:.3f}   stage 2 rho {r2:.3f}")
    deltas.to_csv(args.out_dir / "deltas.csv", index=False)


if __name__ == "__main__":
    main()
