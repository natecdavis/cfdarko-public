"""
Two-part impact model vs the production single-stage head.

CORRECTION (2026-10-30, after the holdout run): "production" below is the
step-8 PER-POSITION impact head, but it is NOT what ships.
config.GLOBAL_SCORE_OVERRIDES replaces impact_apy for all 11 positions with
the pooled step-8b global model (per-position isotonic in 8c). Against the
global model's OOF (global_draft_models/global_oof_predictions.parquet, July
labels) on the same rows, tp_common TIES: within-class referees -0.004 /
-0.012 / -0.016 (all players), +0.017 / +0.038 / +0.010 (board), every
interval includes 0; AUC for y > 0 0.961 vs global 0.990; Spearman among
y > 0 0.365 vs 0.351. The pooling gain found here is the one 8b already
delivers. No switch. Everything below stands as a comparison with the
per-position heads only.

RESULT (2026-10-30): the POOLED two-part model beats production; untuned.
00u --within-class, Δ vs production [95% bootstrap], value / late_snap / late_apy:
    all       tp_common  +0.032 [+0.001,+0.062]  +0.083 [+0.033,+0.130] 9/9  +0.058 [+0.001,+0.114] 8/9
              prod_43    -0.009                  -0.022                      -0.031
    board     tp_common  +0.087                  +0.175                      +0.136   (all CIs exclude 0)
    offboard  tp_common  +0.022                  -0.045                      -0.059   (all CIs include 0)
tp_union is close behind (+0.030 / +0.069 / +0.047); its seed placebo moves
0.005-0.013. tp_pergroup (same split, per-position stage 2) LOSES, so pooling
stage 2 is the active ingredient. Full task per (group × fold): AUC for y > 0
0.961 vs production 0.949 (placebo 0.942); Spearman among y > 0 0.365 vs
0.301. Caveats: (1) --within-class was added AFTER the default pooled scoring
showed tp losing on value (-0.06). The cause was verified (the two-part OOF
scale differs by fold: stage 2 stops at 1-2 trees in fold 2010, ~200 later),
and tp wins inside each of folds 2013 / 2016 / 2019 on the default scoring
too, but it is still a post-hoc scoring choice. (2) Off-board late referees
lean negative (wide CIs). (3) Spearman over ALL rows favors production
(0.634 vs 0.413), an artifact of tied zeros; see the full-task note below.

Why (experiments/00x, 00y, 2026-10-30): the production impact head is one
regressor per position on a zero-padded target (~97% of rows never reach the
NFL). Made-NFL-or-not dominates its loss, and the features that serve that
job add noise to the ranking of NFL players: on board players it loses to
hasan_rank alone within the NFL players, on its own label, in every group.
A board-only model pooled across positions closed that gap, and pooling beat
per-position heads at this sample size. This tests the general version over
ALL players, so nothing has to be spliced:

    score = P(y > 0)  ×  E[y | y > 0]
    stage 1  per-position classifier on the head's full feature list, all rows
    stage 2  conditional magnitude, fit on y > 0 rows only

Arms (OOF on the production tuning folds; production restricted to the same
rows):
    tp_union     stage 2 pooled across positions on the UNION of the heads'
                 feature lists (NaN where a position lacks a feature) +
                 position code
    tp_union_43  same, seed 43 (placebo)
    tp_common    stage 2 pooled on the features every head shares + position
    tp_pergroup  stage 2 per position (isolates pooling from the split)
hasan_rank is monotone in both stages. Neither stage is tuned (production is
Optuna-tuned), which biases the test toward production.

Reading it: 00u --within-class --subset all / board / offboard for the
referees (NFL players only; the arms' OOF scale differs between folds), plus
the full-task check this script prints per (group × fold): AUC for y > 0 and
Spearman among y > 0. A two-part arm that wins the referees but loses the
full task is ranking NFL players at the expense of sorting prospects, and
does not pass.

HOLDOUT CONFIRMATION — pre-registered 2026-10-30, written and committed
before the --holdout run. Fold: production's reserved holdout (train last
college season <= 2021, validate 2022-2024; Optuna and every CV result above
never saw it). Production and its seed-43 placebo are refit on that fold
with their tuned hyperparameters (the script first checks the refit
reproduces the saved production OOF on fold 2019). Primary arm: tp_common.
Metrics, each the mean over position groups: AUC for y > 0, and Spearman
among y > 0. PASS iff on BOTH metrics
    (a) Δ(tp_common − prod) > 0 with its paired 95% bootstrap interval
        (players resampled within group) excluding 0, and
    (b) Δ exceeds |prod_43 − prod| on that metric.
tp_union and tp_pergroup are reported, not decided on. Label caveat: these
are draft classes 2023-25, unsettled, so y for NFL players is the trajectory
estimate from early-career usage, not a realised second contract.

HOLDOUT RESULT (2026-10-30): PASS. 13,971 rows; the production refit
reproduced the saved OOF exactly (max |diff| 0.0).
                 AUC y>0   Δ [95%]                 rho y>0   Δ [95%]
    prod         0.957                             0.472
    prod_43      0.955     -0.002                  0.477     +0.006
    tp_common    0.997     +0.040 [+0.031,+0.049]  0.542     +0.070 [+0.023,+0.119]
    tp_union     0.997     +0.040                  0.541     +0.070
    tp_pergroup  0.989     +0.032                  0.509     +0.038 [-0.010,+0.083]
Read with care: in these classes y > 0 for 98% of NFL players (trajectory
estimates; 90% in the CV years), so the AUC here is close to "reaches the
NFL", an easier target. Stage 1 is shared by every two-part arm, so the seed
placebo only covers stage 2.

Usage:
    00z_two_part_model.py --out-dir DIR [--holdout]
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
PROD_OOF = PROCESSED_DIR / "_board_weight_ab_20261030" / "oof_w1.parquet"
PROD_PLACEBO_OOF = PROCESSED_DIR / "_board_weight_ab_20261030" / "oof_placebo.parquet"
ID_COLS = ["player_id", "pos_group", "impact_target", "last_college_season_year"]


def _fit(step8, cls, params, tr, cols, y):
    mono = [-1 if c == "hasan_rank" else 0 for c in cols]
    mk = lambda: cls(**{**params, "monotone_constraints": mono})
    m, best = step8._fit_fold_honest(
        mk, tr[cols].to_numpy(dtype=np.float32, na_value=np.nan), y,
        tr["last_college_season_year"].fillna(0).values)
    return m, best


def _prod_preds(step8, per_group, train_end, val_start, val_end, seed):
    """Production's impact head refit on one fold, exactly as step 8 does it."""
    out = []
    for g, (cols, df) in per_group.items():
        yr = df["last_college_season_year"].fillna(0)
        tr, va = df[yr <= train_end], df[yr.between(val_start, val_end)]
        if len(tr) < 30 or len(va) < 10:
            continue
        params = {**LGBM_PARAMS, **(step8._load_tuned_params(g, "impact") or {}),
                  "random_state": seed}
        mk = lambda: lgb.LGBMRegressor(**{
            **params, **step8._build_monotone_kwargs(cols, g, "impact")})
        m, _ = step8._fit_fold_honest(
            mk, tr[cols].to_numpy(dtype=float, na_value=np.nan),
            tr["impact_target"].values, tr["last_college_season_year"].fillna(0).values)
        out.append(pd.DataFrame({
            "player_id": va["player_id"].astype(str).values, "pos_group": g,
            "val_start": val_start, "y": va["impact_target"].values,
            "pred": m.predict(va[cols].to_numpy(dtype=float, na_value=np.nan))}))
    return pd.concat(out, ignore_index=True)


def _task_metrics(a):
    auc, rho = [], []
    for _, g in a.groupby("pos_group"):
        pos = g["y"].values > 0
        if pos.sum() < 10 or pos.all():
            continue
        auc.append(roc_auc_score(pos, g["pred"].values))
        rho.append(spearmanr(g["pred"].values[pos], g["y"].values[pos])[0])
    return np.mean(auc), np.nanmean(rho)


def _holdout_report(arms, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    base = arms["prod"][["player_id", "pos_group", "y"]].reset_index(drop=True)
    preds = {n: base.merge(a[["player_id", "pred"]], on="player_id", how="left")["pred"].values
             for n, a in arms.items()}
    groups = [np.where(base["pos_group"].values == g)[0] for g in base["pos_group"].unique()]
    point = {n: _task_metrics(base.assign(pred=p)) for n, p in preds.items()}
    boots = []
    for _ in range(n_boot):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        b = base.iloc[ix]
        boots.append({n: _task_metrics(b.assign(pred=p[ix])) for n, p in preds.items()})
    placebo = [abs(point["prod_43"][k] - point["prod"][k]) for k in range(2)]
    print("\nHOLDOUT (pre-registered bar on tp_common): mean over groups")
    passes = []
    for n in preds:
        line = f"  {n:12s}"
        for k, name in enumerate(["auc_y>0", "rho_y>0"]):
            d = np.array([b[n][k] - b["prod"][k] for b in boots])
            delta = point[n][k] - point["prod"][k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            ok = delta > 0 and lo > 0 and delta > placebo[k]
            if n == "tp_common":
                passes.append(ok)
            line += (f"  {name} {point[n][k]:.3f}  Δ {delta:+.3f} [{lo:+.3f},{hi:+.3f}]"
                     + ("  pass" if ok else ""))
        print(line)
    print(f"  placebo |Δ|: auc {placebo[0]:.3f}, rho {placebo[1]:.3f}")
    print(f"  VERDICT: {'PASS' if all(passes) else 'FAIL'}")


def _X(df, cols):
    return df[cols].to_numpy(dtype=np.float32, na_value=np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--holdout", action="store_true",
                    help="run the pre-registered confirmation on the reserved holdout fold")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step8 = importlib.import_module("08_draft_models")
    snapshot = step8.assemble_snapshot(include_derived=True, logger=step8.logger)
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    folds, holdout = step8._split_holdout(step8._expanding_window_folds())
    if args.holdout:
        folds = [holdout]

    per_group, union, common = {}, [], None
    for g in GROUPS:
        cols, df = step8._resolve_fit_features(snapshot, labels, g, "impact_target", "impact")
        df = df.assign(pos_group=g, impact_target=df["impact_target"].fillna(0.0))
        per_group[g] = (cols, df)
        union += [c for c in cols if c not in union]
        common = set(cols) if common is None else common & set(cols)
    common = [c for c in union if c in common]
    allrows = pd.concat([d[ID_COLS + cols] for cols, d in per_group.values()],
                        ignore_index=True)
    allrows["pos_code"] = allrows["pos_group"].map({g: i for i, g in enumerate(GROUPS)})
    allrows["player_id"] = allrows["player_id"].astype(str)
    logging.info("rows %d, y>0 %d; union features %d, common %d", len(allrows),
                 int((allrows["impact_target"] > 0).sum()), len(union), len(common))

    yr = allrows["last_college_season_year"].fillna(0)
    p1 = pd.Series(np.nan, index=allrows.index)           # stage-1 OOF P(y > 0)
    m2 = {a: pd.Series(np.nan, index=allrows.index)
          for a in ["tp_union", "tp_union_43", "tp_common", "tp_pergroup"]}
    fold_of = pd.Series(np.nan, index=allrows.index)
    for train_end, val_start, val_end in folds:
        tr_m, va_m = yr <= train_end, yr.between(val_start, val_end)
        fold_of[va_m] = val_start
        for g, (cols, _) in per_group.items():
            gm = allrows["pos_group"] == g
            tr, va = allrows[tr_m & gm], allrows[va_m & gm]
            if len(va) == 0 or (tr["impact_target"] > 0).sum() < 10:
                continue
            clf, b1 = _fit(step8, lgb.LGBMClassifier, STAGE1_PARAMS, tr, cols,
                           (tr["impact_target"] > 0).astype(int).values)
            p1[va.index] = clf.predict_proba(_X(va, cols))[:, 1]
            pos = tr[tr["impact_target"] > 0]
            if len(pos) >= 40:
                r, _ = _fit(step8, lgb.LGBMRegressor, STAGE2_PARAMS, pos, cols,
                            pos["impact_target"].values)
                m2["tp_pergroup"][va.index] = r.predict(_X(va, cols))
        tr, va = allrows[tr_m], allrows[va_m]
        pos = tr[tr["impact_target"] > 0]
        for arm, cols, seed in [("tp_union", union + ["pos_code"], 42),
                                ("tp_union_43", union + ["pos_code"], 43),
                                ("tp_common", common + ["pos_code"], 42)]:
            r, b2 = _fit(step8, lgb.LGBMRegressor, {**STAGE2_PARAMS, "random_state": seed},
                         pos, cols, pos["impact_target"].values)
            m2[arm][va.index] = r.predict(_X(va, cols))
            logging.info("fold %d %s: stage-2 rows %d trees %d", val_start, arm, len(pos), b2)

    base = allrows[["player_id", "pos_group", "impact_target"]].assign(
        val_start=fold_of, p1=p1).rename(columns={"impact_target": "y"})
    base = base.dropna(subset=["val_start", "p1"])
    base["val_start"] = base["val_start"].astype(int)
    if args.holdout:
        check = _prod_preds(step8, {"WR": per_group["WR"]}, 2018, 2019, 2021, 42)
        saved = pd.read_parquet(PROD_OOF)
        saved = saved[(saved["model_name"] == "impact") & (saved["val_start"] == 2019)]
        diff = check.merge(saved, on="player_id", suffixes=("", "_saved"))
        gap = float(np.abs(diff["pred"] - diff["pred_saved"]).max())
        logging.info("refit check vs saved production OOF (WR fold 2019): %d rows, max |diff| %.2e",
                     len(diff), gap)
        if gap > 1e-9:
            raise SystemExit("production refit does not reproduce the saved OOF")
        prod_runs = {n: _prod_preds(step8, per_group, *holdout, seed)
                     for n, seed in [("prod", 42), ("prod_43", 43)]}
        prod = prod_runs["prod"]
    else:
        prod = pd.read_parquet(PROD_OOF)
        prod = prod[prod["model_name"] == "impact"]
    keys = base[["player_id", "val_start"]].merge(prod[["player_id", "val_start"]])
    logging.info("rows with stage-1 OOF %d; shared with production %d", len(base), len(keys))

    arms = {}
    for arm, m in m2.items():
        arms[arm] = base.assign(pred=base["p1"] * m.loc[base.index].clip(lower=0))
    arms["tp_stage1"] = base.assign(pred=base["p1"])      # P(y > 0) alone, for reference
    if args.holdout:
        arms.update(prod_runs)
    else:
        for name, path in [("prod", PROD_OOF), ("prod_43", PROD_PLACEBO_OOF)]:
            o = pd.read_parquet(path)
            arms[name] = o[o["model_name"] == "impact"][["player_id", "val_start", "pos_group", "y", "pred"]]
    for name in arms:
        a = arms[name].merge(keys, on=["player_id", "val_start"]).dropna(subset=["pred"])
        a = a.assign(model_name="impact")
        arms[name] = a
        a.to_parquet(args.out_dir / f"{'holdout' if args.holdout else 'oof'}_{name}.parquet",
                     index=False)
    if args.holdout:
        _holdout_report({n: arms[n] for n in ["prod", "prod_43", "tp_common", "tp_union",
                                               "tp_union_43", "tp_pergroup"]})
        return

    # Full-task check, per (group × fold) so fold-to-fold scale never enters:
    # AUC of the score for y > 0 (sorting prospects) and Spearman among y > 0
    # (ranking NFL players). Spearman over ALL rows is NOT reported: with 93%
    # tied zeros it rewards heads that tie the zeros too (production's few-tree
    # heads emit ~19 distinct values per group), not better ranking.
    print("\nfull task, per (group × fold): AUC for y>0 / Spearman among y>0")
    rows = []
    for name, a in arms.items():
        auc, rho = [], []
        for _, g in a.groupby(["pos_group", "val_start"]):
            pos = g["y"] > 0
            if pos.sum() < 10 or pos.all():
                continue
            auc.append(roc_auc_score(pos, g["pred"]))
            rho.append(spearmanr(g.loc[pos, "pred"], g.loc[pos, "y"])[0])
        rows.append({"arm": name, "n": len(a), "auc_y>0": np.mean(auc),
                     "rho_y>0": np.nanmean(rho)})
    print(pd.DataFrame(rows).set_index("arm").round(3).to_string())

if __name__ == "__main__":
    main()
