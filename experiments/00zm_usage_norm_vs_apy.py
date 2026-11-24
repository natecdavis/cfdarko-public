"""
Retuned normalized-usage target vs the deployed APY target, global (8b) model.

STATUS (2026-11-01): STOPPED during the Optuna retune, before any arm was
fit. No result. The user stopped it because the APY-trained 8b head barely
reaches the product: the merged board ranks on-board players by the board
curve and off-board players by step 8d, and 8b is only the fallback engine.
The bar below was never evaluated.

Follows 00zl (normalized usage beats raw usage on every referee, but at
APY-tuned hyperparameters it trailed the deployed APY head on late_apy).
Question: once it has its own hyperparameters, should the shipped impact
model be trained on starter-equivalent y3-4 snap share instead of APY?

Arms
    apy    the deployed head: target impact_target, deployed global/impact_apy
           tuned params. Refit here at seeds 42/43/44 (seed 42 should
           reproduce global_oof_predictions.parquet).
    norm   target usage_norm from 00zl.build_targets (usage_y34 ÷ p80 per
           value_group × usage_source; non-NFL 0; settled classes only),
           rescaled to impact_target's std. Hyperparameters retuned here
           exactly as 8b tunes: _suggest_optuna_params search space, MAE
           objective via _p8._evaluate_params on the 8b tuning folds,
           30 trials × 3 Optuna seeds, lowest-CV seed wins. Fit at 42/43/44.
Same feature set (8b frame), same folds, OOF from the tuning folds.

Not equalized (caveat, stated up front): 04b/04d decay betas and Kalman
parameters were calibrated on impact_target, so the feature surface was
built with APY as its target. That favours apy.

Scoring: bridge players, K/P excluded, settled classes (00u cohort), rows
where every arm has a prediction. Statistics as in 00zl:
    within  Spearman within position group, averaged over groups
    cross   Spearman across all positions within draft class, averaged
Referees: value (APY + guarantees + usage composite, the most power),
late_snap (snaps y5-7; leans norm), late_apy (pay y7-9; leans apy).
Predictions are also ranked within (group × class) and re-scored
(--within-class), because the arms' fold-to-fold scales differ.

Contrast: norm_sS − apy_sS, seed-matched, mean over 3 seeds.
Placebo: same-target seed pairs (apy 43−42, 44−42, 44−43; same for norm).
This measures fit noise only; Optuna noise in norm's retune is not
covered, which is another reason the bar asks for margins beyond it.

PRE-REGISTERED BAR (written before tuning or fitting):
  Recommend switching the shipped target to normalized usage only if ALL:
    1. within value Δ > 0 and > max|placebo| on value;
    2. within late_apy Δ > −max|placebo| (no loss beyond noise on pay);
    3. within late_snap Δ > −max|placebo|;
    4. cross late_apy Δ > −max|placebo| (cross-position ordering is not
       worse on pay).
  Pooled-across-classes scores are primary; --within-class is a
  robustness check, and a disagreement between the two is reported.
  Secondary (reported, not decision): the same table on the off-board
  subset (bridge players without a consensus board rank). Off-board players
  are where the user wants the model to add value.
  If the bar passes, the switch still needs 8c/step-10 work (dollar mapping,
  merged board calibrators), and 04b/04d recalibration should be tested.

Usage:
    00zm_usage_norm_vs_apy.py --tune [--trials 30 --seeds 3]
    00zm_usage_norm_vs_apy.py --fit --arm apy|norm --seed S
    00zm_usage_norm_vs_apy.py --report [--within-class] [--subset all|board|offboard]
"""
import argparse
import hashlib
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
OUT = PROCESSED_DIR / "_usage_norm_vs_apy_20261101"
SEEDS = (42, 43, 44)
REFS = ["value", "late_snap", "late_apy"]
N_BOOT = 300


def _frame(g):
    zl = importlib.import_module("00zl_usage_position_norm")
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    df, feature_cols = g.build_global_training_frame(labels)
    tgt, _ = zl.build_targets(df, labels)
    df = df.join(tgt)
    scale = float(df["impact_target"].std() / df["usage_norm"].std())
    df["usage_target"] = df["usage_norm"] * scale
    return df, feature_cols, scale


def _tune(args):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    g = importlib.import_module("08b_global_draft_model")
    p8 = g._p8
    df, feature_cols, _ = _frame(g)
    sub = df[df["usage_target"].notna()].copy()
    folds, _ = p8._split_holdout(p8._expanding_window_folds())
    fc = g._global_filter_state_space_to_mi(feature_cols, "impact_apy")
    fold_data = p8._build_fold_data(sub, fc, "usage_target", folds)
    base_seed = 42 + int.from_bytes(hashlib.sha256(b"global/impact_apy").digest()[:4], "big")
    results = []
    for k in range(args.seeds):
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=base_seed + k * 1009))

        def _obj(trial):
            score, _ = p8._evaluate_params(p8._suggest_optuna_params(trial), fold_data, fc,
                                           metric="mae", pos_group="__global__",
                                           model_name="__global__")
            if score is None:
                raise optuna.TrialPruned()
            return score

        study.optimize(_obj, n_trials=args.trials, catch=(Exception,), n_jobs=-1)
        results.append((float(study.best_value), dict(study.best_params)))
        logging.info("tune seed %d/%d best mae %.6f %s", k + 1, args.seeds, *results[-1])
    results.sort(key=lambda r: r[0])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"winner": results[0][1], "seeds": results},
              open(OUT / "tuned_norm.json", "w"), indent=1)


def _fit(args):
    g = importlib.import_module("08b_global_draft_model")
    g.LGBM_PARAMS["random_state"] = args.seed
    df, feature_cols, scale = _frame(g)
    if args.arm == "apy":
        tuned = json.loads(g.GLOBAL_TUNED_PARAMS_PATH.read_text())["global/impact_apy"]
        target, s = "impact_target", 1.0
    else:
        tuned = json.load(open(OUT / "tuned_norm.json"))["winner"]
        target, s = "usage_target", scale
    _, metrics, _, oof, _ = g._train_one_global(df, feature_cols, target, "impact_apy", tuned)
    oof["pred"] = oof["pred"] / s
    OUT.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OUT / f"oof_{args.arm}_s{args.seed}.parquet", index=False)
    json.dump(metrics, open(OUT / f"metrics_{args.arm}_s{args.seed}.json", "w"), indent=1)


def _within(df, col, ref):
    r = [spearmanr(s[col], s[ref])[0] for _, sub in df.groupby("value_group")
         for s in [sub[[col, ref]].dropna()] if len(s) >= 10]
    return float(np.mean(r))


def _cross(df, col, ref):
    r = [spearmanr(s[col], s[ref])[0] for _, sub in df.groupby("draft_year")
         for s in [sub[[col, ref]].dropna()] if len(s) >= 30]
    return float(np.mean(r))


STATS = {"within": _within, "cross": _cross}


def _report(args):
    u = importlib.import_module("00u_value_target_referee_ab")
    df = u._cohort().reset_index(drop=True)
    if args.subset != "all":
        board = set(pd.read_parquet(PROCESSED_DIR / "hasan_ranks.parquet")["player_id"])
        on = df["player_id"].isin(board)
        df = df[on if args.subset == "board" else ~on].reset_index(drop=True)
    arms = [f"{a}_s{s}" for a in ("apy", "norm") for s in SEEDS]
    for name in arms:
        o = pd.read_parquet(OUT / f"oof_{name}.parquet").drop_duplicates("player_id", keep="last")
        df[name] = df["player_id"].map(o.set_index("player_id")["pred"])
    dep = pd.read_parquet(PROCESSED_DIR / "global_draft_models/global_oof_predictions.parquet")
    dep = dep[dep["model_name"] == "impact_apy"].drop_duplicates("player_id", keep="last")
    df["deployed"] = df["player_id"].map(dep.set_index("player_id")["pred"])
    df = df[df[arms].notna().all(axis=1)].reset_index(drop=True)
    print(f"subset={args.subset} within_class={args.within_class}: {len(df)} players, "
          f"classes {int(df.draft_year.min())}-{int(df.draft_year.max())}, "
          f"late rows {int(df.late_snap.notna().sum())}")
    ok = df["deployed"].notna()
    print(f"apy_s42 vs deployed OOF: rank corr {spearmanr(df.loc[ok, 'apy_s42'], df.loc[ok, 'deployed'])[0]:.4f}")
    if args.within_class:
        for c in arms + REFS:
            df[c] = df.groupby(["value_group", "draft_year"])[c].rank(pct=True)

    pt = {(st, a, r): STATS[st](df, a, r) for st in STATS for a in arms for r in REFS}
    print(pd.DataFrame({f"{st}|{r}": [round(pt[(st, a, r)], 3) for a in arms]
                        for st in STATS for r in REFS}, index=arms).to_string())
    con = [(f"norm_s{s}", f"apy_s{s}") for s in SEEDS]
    pla = [(f"{a}_s{x}", f"{a}_s{y}") for a in ("apy", "norm")
           for x, y in ((43, 42), (44, 42), (44, 43))]
    rng = np.random.default_rng(0)
    groups = [sub.index.to_numpy() for _, sub in df.groupby("value_group")]
    boots = [np.concatenate([rng.choice(ix, len(ix)) for ix in groups]) for _ in range(N_BOOT)]
    rows = {}
    for st in STATS:
        for r in REFS:
            d = np.mean([pt[(st, a, r)] - pt[(st, b, r)] for a, b in con])
            pl = max(abs(pt[(st, a, r)] - pt[(st, b, r)]) for a, b in pla)
            bd = [np.mean([STATS[st](df.loc[ix], a, r) - STATS[st](df.loc[ix], b, r)
                           for a, b in con]) for ix in boots]
            rows[(st, r)] = d, pl
            print(f"{st:6s} {r:9s} norm−apy {d:+.4f} [{np.percentile(bd, 2.5):+.4f}, "
                  f"{np.percentile(bd, 97.5):+.4f}]  per-seed "
                  + " ".join(f"{pt[(st, a, r)] - pt[(st, b, r)]:+.4f}" for a, b in con)
                  + f"   max|placebo| {pl:.4f}")
    print("\nper-group within Δ (seed mean):")
    pg = {gname: {r: round(np.mean([spearmanr(*sub[[a, r]].dropna().T.values)[0]
                                    - spearmanr(*sub[[b, r]].dropna().T.values)[0]
                                    for a, b in con]), 3) for r in REFS}
          for gname, sub in df.groupby("value_group")}
    print(pd.DataFrame(pg).T.to_string())
    bar = [rows[("within", "value")][0] > rows[("within", "value")][1],
           rows[("within", "late_apy")][0] > -rows[("within", "late_apy")][1],
           rows[("within", "late_snap")][0] > -rows[("within", "late_snap")][1],
           rows[("cross", "late_apy")][0] > -rows[("cross", "late_apy")][1]]
    print(f"\nbar conditions 1-4: {bar} → {'PASS' if all(bar) else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--arm", choices=["apy", "norm"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--within-class", action="store_true")
    ap.add_argument("--subset", choices=["all", "board", "offboard"], default="all")
    args = ap.parse_args()
    if args.tune:
        _tune(args)
    elif args.fit:
        _fit(args)
    elif args.report:
        _report(args)


if __name__ == "__main__":
    main()
