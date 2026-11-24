"""
Confirmation: do PFF side features improve off-board NFL ranking?
(pre-registered 2026-11-01)

Origin. 00zh's PFF arm failed its pre-registered bar on contributor AUC (the
gain didn't beat a single-seed placebo), but lifted rho_ob_nfl — Spearman
with y3-4 snap share among off-board players who reached the NFL — from
0.064 to 0.117 (+0.053 [+0.008, +0.103], placebo +0.005).

DISCLOSURE: rho_ob_nfl is promoted to the primary metric AFTER seeing 00zh,
on the same classes (2015-22). No independent classes exist: later classes
have no observed y3-4 usage. Seed-bagging removes fit noise but not the
selection: a pass here is "survives a noise-controlled re-test", weaker
than an out-of-sample confirmation.

Model: 00zd's two-stage P(NFL) × E[usage | NFL], pooled, untuned, OOF on the
step-8 tuning folds. Each arm is a SEED BAG: stage 1 and stage 2 predictions
are averaged over 5 seeds, then multiplied.
    base      00zd features, seeds 42-46
    base_alt  00zd features, seeds 142-146 (placebo: an independent bag)
    pff       + 00zh's PFF side features, seeds 42-46
    dep_impact deployed 8b OOF (reference)
Scoring: classes 2015-22 (same as 00zh), K/P excluded, off-board players.

PRE-REGISTERED BAR (one primary test, 95% paired bootstrap, 1000 reps,
players resampled within class):
    PASS if pff − base on rho_ob_nfl has its 95% interval above 0 AND
    exceeds |base_alt − base| on rho_ob_nfl.
Secondary (reported, not tested): auc_ob, auc_ob_pos, auc_udfa, rho_udfa_nfl.

Deployment check, run only on a PASS: write 00zf-format OOF for pff and run
00zf's capped merge. Recommend switching the off-board side of the merged
board from the deployed model to the pff bag only if merged_pre_cap −
merged_dep_cap is not significantly negative (95% interval wholly below 0)
on top-M or top-100 usage, AND pff beats dep_impact on rho_ob_nfl. The
switch itself would be a separate production change.

RESULT (2026-11-01): PASS. rho_ob_nfl pff − base +0.0417 [+0.0019, +0.0847];
placebo (base_alt − base) +0.0002. Bag levels: base 0.064, pff 0.105,
dep_impact 0.078. Secondary: auc_ob +0.002 (ns), auc_udfa +0.002 (ns),
rho_udfa_nfl +0.027 (ns). vs dep_impact: rho_ob_nfl +0.027 (ns), auc_ob
+0.012 [+0.004, +0.020], auc_udfa +0.018 [+0.008, +0.027].

Deployment check (00zf, `merged/`; merged_pre_cap = pff bag on the off-board
side, 1000 reps): usage topM +0.0031 [−0.0008, +0.0072], top100 0, top500
+0.0035 [+0.0018, +0.0054], seam150 +0.0080 [+0.0000, +0.0148]. Not
significantly negative on top-M / top-100, and pff beats dep_impact on
rho_ob_nfl → criteria met: RECOMMEND the switch. Not pre-registered but
supportive: value topM +0.0099 [+0.0030, +0.0168], seam150 +0.021; realized
pay topM +0.0003 [+0.0001, +0.0005]. First merged variant to beat the
current merge on top-M on any referee. Lower bound on the primary is close
to 0 and the metric was chosen after 00zh (see DISCLOSURE).

Usage:
    00zi_offboard_pff_confirm.py --out-dir DIR [--n-boot 1000]
"""
import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("00zi")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zh = _load("zh", "experiments/00zh_offboard_visits_pff.py")
zd = zh.zd

SEEDS = [42, 43, 44, 45, 46]
ARM_SPECS = {"base": ([], SEEDS), "base_alt": ([], [s + 100 for s in SEEDS]),
             "pff": (zh.PFF_FEATURES, SEEDS)}
PAIRS = [("pff", "base"), ("base_alt", "base"), ("pff", "dep_impact"),
         ("base", "dep_impact")]


def run_bags(step8, rows, feats, folds):
    yr = rows["last_college_season_year"].fillna(0)
    fold_of = pd.Series(np.nan, index=rows.index)
    score = {a: pd.Series(np.nan, index=rows.index) for a in ARM_SPECS}
    for train_end, val_start, val_end in folds:
        tr_m, va_m = yr <= train_end, yr.between(val_start, val_end)
        fold_of[va_m] = val_start
        tr, va = rows[tr_m], rows[va_m]
        nfl = tr[tr["nfl"]]
        for arm, (extra, seeds) in ARM_SPECS.items():
            cols = feats + extra
            p1s, m2s = [], []
            for seed in seeds:
                clf, _ = zd._fit(step8, lgb.LGBMClassifier,
                                 {**zd.STAGE1_PARAMS, "random_state": seed},
                                 tr, cols, tr["nfl"].astype(int).values)
                reg, _ = zd._fit(step8, lgb.LGBMRegressor,
                                 {**zd.STAGE2_PARAMS, "random_state": seed},
                                 nfl, cols, nfl["usage"].values)
                p1s.append(clf.predict_proba(zd._X(va, cols))[:, 1])
                m2s.append(np.clip(reg.predict(zd._X(va, cols)), 0, 1))
            score[arm][va.index] = np.mean(p1s, axis=0) * np.mean(m2s, axis=0)
            log.info("fold %d %s: %d cols, %d seeds", val_start, arm, len(cols), len(seeds))
    return score, fold_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step8 = importlib.import_module("08_draft_models")
    folds, _ = step8._split_holdout(step8._expanding_window_folds())
    snap = zh.add_side_pff(step8.assemble_snapshot(include_derived=True, logger=step8.logger))
    rows, feats_all = zd._build_rows(step8, snapshot=snap, extra=zh.PFF_FEATURES)
    feats = [c for c in feats_all if c not in zh.PFF_FEATURES]

    score, fold_of = run_bags(step8, rows, feats, folds)
    base = rows[zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted"]].assign(val_start=fold_of)
    for a in ARM_SPECS:
        base[a] = score[a]
    base = base.dropna(subset=["val_start", "base"])
    base["val_start"] = base["val_start"].astype(int)
    base.to_parquet(args.out_dir / "oof_all_arms.parquet", index=False)
    base.assign(pre=base["pff"], pre_43=base["base"])[
        zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted", "val_start", "pre", "pre_43"]
    ].to_parquet(args.out_dir / "oof_00zd_pff.parquet", index=False)

    E = zd._deployed(base).dropna(subset=["dep_impact"]).drop_duplicates("player_id", keep="last")
    E["cls"] = E["last_college_season_year"].astype(int) + 1
    E = E[E["cls"].between(*zh.EVAL_CLASSES)]
    E["board"] = E["hasan_rank"].notna()
    E["contrib"] = (E["usage"] >= zd.CONTRIB).astype(int)
    E["starter"] = (E["usage"] >= zd.STARTER).astype(int)
    ob = E[~E["board"]]
    log.info("scoring cohort %d: off-board %d, off-board NFL %d, contributors %d",
             len(E), len(ob), int(ob["nfl"].sum()), int(ob["contrib"].sum()))

    arms = list(ARM_SPECS) + ["dep_impact"]
    point, deltas = zh.bootstrap(E, arms, PAIRS, args.n_boot, levels=(95.0,))
    pd.set_option("display.width", 200)
    print("\nheadline metrics:")
    print(pd.DataFrame(point).T.round(4).to_string())
    print("\nΔ, paired bootstrap 95%:")
    print(deltas.round(4).to_string(index=False))
    print("\ntop-K vs the bottom of the board:")
    print(zd._topk_table(E, arms).to_string())

    g = lambda a, b: deltas[(deltas["Δ"] == f"{a} − {b}") & (deltas["metric"] == "rho_ob_nfl")].iloc[0]
    d, plac = g("pff", "base"), abs(g("base_alt", "base")["delta"])
    passed = bool(d["lo95"] > 0 and d["delta"] > plac)
    v = {"pass": passed, "d_rho": round(d["delta"], 4),
         "ci95": f"[{d['lo95']:+.4f}, {d['hi95']:+.4f}]", "placebo": round(plac, 4),
         "pff_vs_dep_rho": round(g("pff", "dep_impact")["delta"], 4)}
    print("\nVERDICT:", v)
    deltas.to_csv(args.out_dir / "deltas.csv", index=False)
    pd.Series(v, dtype=object).to_json(args.out_dir / "verdict.json")


if __name__ == "__main__":
    main()
