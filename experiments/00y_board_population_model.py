"""
NOTE (2026-10-30): compared against the step-8 per-position impact head,
which does not ship; the deployed impact_score is the 8b global model
(see 00z CORRECTION). Against it, the board-beyond-model gap is smaller:
+0.17 / +0.25 / +0.17 (vs +0.26 / +0.38 / +0.29), pred~board 0.71.
Option 2b ceiling test: an impact model trained only on board players.

RESULT (2026-10-30): the board-population model reaches the BOARD's level, not
beyond it. 00u --subset board, 1,658 board players, folds val 2016 + 2019:
                 value   late_snap  late_apy
    prod         0.207   0.210      0.184
    prod_43      0.207   0.183      0.154    (placebo, Δ within ±0.03)
    pooled       0.305   0.449      0.347    (Δ +0.10/+0.24/+0.16, 7-9/9 wins)
    offset       0.316   0.422      0.304    (residual trees stop at 2-14)
    board        0.336   0.454      0.328    (-hasan_rank alone)
Own label, bridge only: prod 0.269, pergroup 0.357, pooled 0.455, board 0.505
— pooling across positions beats per-position heads at this n. Partial
Spearman (no null): pooled beyond board +0.04/+0.14/+0.09; board beyond pooled
+0.15/+0.15/+0.12. College features add little beyond the board for board
players, and trees still leave some of the board unused.

Why (experiments/00x, 2026-10-30): on board players the production impact
head ties the consensus board at made-NFL-or-not but loses to it WITHIN the
NFL players on its own training label, in every group and fold. Row weights
and slower learning rates changed nothing: the zero-padded task (~97% of rows
never reach the NFL) is dominated by made-NFL-or-not, and the features that
serve it add noise to the within-NFL ranking. Board players reach the NFL
~75% of the time, so a model fit on them learns mostly within-NFL ranking.
The board stays an input; nothing is blended into the score.

Arms (OOF on board players, CV folds that have board seasons in training:
val 2016-18 and 2019-21):
    pooled      one model across positions: the 46 features every head shares
                + position code; hasan_rank monotone
    pooled_43   same, seed 43 (placebo)
    pergroup    one model per position group on that head's full feature
                list, heavily regularized (100-300 training rows per fold)
    offset      pooled, but fit to the residual from a board baseline: per
                position, isotonic regression of the target on hasan_rank
                (training fold only); prediction = baseline + trees. Same as
                LightGBM init_score under squared loss
    board       -hasan_rank itself (reference, not a candidate)
Production OOF is restricted to the same players for comparison.

Usage:
    00y_board_population_model.py --out-dir DIR
    then 00u_value_target_referee_ab.py --subset board --oof prod=DIR/oof_prod.parquet ...
"""
import argparse
import importlib
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from config import PROCESSED_DIR, POSITION_GROUPS, LGBM_PARAMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
GROUPS = [g for g in POSITION_GROUPS if g not in ("K", "P")]
FOLDS = [(2015, 2016, 2018), (2018, 2019, 2021)]
POOLED_PARAMS = {**LGBM_PARAMS, "learning_rate": 0.03, "n_estimators": 3000,
                 "num_leaves": 15, "min_child_samples": 20}
PERGROUP_PARAMS = {**LGBM_PARAMS, "learning_rate": 0.03, "n_estimators": 3000,
                   "num_leaves": 7, "max_depth": 3, "min_child_samples": 15}


def _fit_predict(step8, tr, va, cols, params, monotone=True):
    mono = [-1 if (c == "hasan_rank" and monotone) else 0 for c in cols]
    mk = lambda: lgb.LGBMRegressor(**{**params, "monotone_constraints": mono})
    m, best = step8._fit_fold_honest(
        mk, tr[cols].to_numpy(dtype=float, na_value=np.nan),
        tr["impact_target"].fillna(0.0).values,
        tr["last_college_season_year"].fillna(0).values)
    return m.predict(va[cols].to_numpy(dtype=float, na_value=np.nan)), best


def _board_baseline(tr, va):
    """Per-position isotonic map hasan_rank -> mean target, fit on the training fold."""
    from sklearn.isotonic import IsotonicRegression
    b_tr, b_va = np.zeros(len(tr)), np.zeros(len(va))
    for g in tr["pos_group"].unique():
        mt, mv = (tr["pos_group"] == g).values, (va["pos_group"] == g).values
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
            tr.loc[mt, "hasan_rank"], tr.loc[mt, "impact_target"].fillna(0.0))
        b_tr[mt] = iso.predict(tr.loc[mt, "hasan_rank"])
        if mv.any():
            b_va[mv] = iso.predict(va.loc[mv, "hasan_rank"])
    return b_tr, b_va


def _oof_frame(va, pred, val_start, val_end):
    return pd.DataFrame({"player_id": va["player_id"].astype(str).values,
                         "pos_group": va["pos_group"].values, "model_name": "impact",
                         "val_start": val_start, "val_end": val_end,
                         "y": va["impact_target"].fillna(0.0).values, "pred": pred})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step8 = importlib.import_module("08_draft_models")
    snapshot = step8.assemble_snapshot(include_derived=True, logger=step8.logger)
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")

    per_group, common = {}, None
    for g in GROUPS:
        cols, df = step8._resolve_fit_features(snapshot, labels, g, "impact_target", "impact")
        df = df[df["hasan_rank"].notna()].copy()
        df["pos_group"] = g
        per_group[g] = (cols, df)
        common = set(cols) if common is None else common & set(cols)
    common = sorted(common)
    pooled = pd.concat([d[["player_id", "pos_group", "impact_target", "is_bridge",
                           "last_college_season_year"] + common] for _, d in per_group.values()],
                       ignore_index=True)
    pooled["pos_code"] = pooled["pos_group"].map({g: i for i, g in enumerate(GROUPS)})
    pooled_cols = common + ["pos_code"]
    logging.info("board rows %d (bridge %.0f%%), pooled features %d", len(pooled),
                 100 * pooled["is_bridge"].astype(bool).mean(), len(pooled_cols))

    arms = {"pooled": [], "pooled_43": [], "offset": [], "offset_43": [],
            "pergroup": [], "board": []}
    yr = pooled["last_college_season_year"].fillna(0)
    for train_end, val_start, val_end in FOLDS:
        tr, va = pooled[yr <= train_end], pooled[yr.between(val_start, val_end)]
        for arm, seed in [("pooled", 42), ("pooled_43", 43)]:
            pred, best = _fit_predict(step8, tr, va, pooled_cols,
                                      {**POOLED_PARAMS, "random_state": seed})
            arms[arm].append(_oof_frame(va, pred, val_start, val_end))
            logging.info("%s fold %d: train %d val %d trees %d", arm, val_start,
                         len(tr), len(va), best)
        b_tr, b_va = _board_baseline(tr, va)
        for arm, seed in [("offset", 42), ("offset_43", 43)]:
            resid = tr.assign(impact_target=tr["impact_target"].fillna(0.0) - b_tr)
            pred, best = _fit_predict(step8, resid, va, pooled_cols,
                                      {**POOLED_PARAMS, "random_state": seed},
                                      monotone=False)
            arms[arm].append(_oof_frame(va, b_va + pred, val_start, val_end))
            logging.info("%s fold %d: trees %d", arm, val_start, best)
        arms["board"].append(_oof_frame(va, -va["hasan_rank"].values, val_start, val_end))
        for g, (cols, df) in per_group.items():
            gy = df["last_college_season_year"].fillna(0)
            gtr, gva = df[gy <= train_end], df[gy.between(val_start, val_end)]
            if len(gtr) < 40 or len(gva) < 10:
                continue
            pred, _ = _fit_predict(step8, gtr, gva, cols, PERGROUP_PARAMS)
            arms["pergroup"].append(_oof_frame(gva, pred, val_start, val_end))

    for arm, frames in arms.items():
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(args.out_dir / f"oof_{arm}.parquet", index=False)
        logging.info("%s: %d OOF rows", arm, len(out))

    # production and its placebo, restricted to exactly the same players / folds
    keys = pd.concat(arms["pooled"])[["player_id", "val_start"]]
    for name, path in [("prod", PROCESSED_DIR / "_board_weight_ab_20261030" / "oof_w1.parquet"),
                       ("prod_43", PROCESSED_DIR / "_board_weight_ab_20261030" / "oof_placebo.parquet")]:
        o = pd.read_parquet(path)
        o = o[o["model_name"] == "impact"].merge(keys, on=["player_id", "val_start"])
        o.to_parquet(args.out_dir / f"oof_{name}.parquet", index=False)
        logging.info("%s: %d OOF rows on the same players", name, len(o))


if __name__ == "__main__":
    main()
