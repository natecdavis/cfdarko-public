"""
NOTE (2026-10-30): compared against the step-8 per-position impact head,
which does not ship; the deployed impact_score is the 8b global model
(see 00z CORRECTION). Against it, the board-beyond-model gap is smaller:
+0.17 / +0.25 / +0.17 (vs +0.26 / +0.38 / +0.29), pred~board 0.71.
Does up-weighting board rows in training make the impact head use the board?

GLOBAL RESULT (2026-10-30, --global, the deployed 8b impact_apy head; ×1
arm corr 0.9986 with the July production OOF). Scored 00u --within-class.
Weights: NO. ×3 / ×10 / ×30 within placebo on all and board players
(all-player Δ value -0.006 / -0.010 / -0.005); off-board ×3 late_apy -0.055
[-0.102, -0.009]; AUC for y > 0 falls 0.989 → 0.976 at ×30; pred~board
FALLS to 0.62 at ×30. Board-beyond-model unchanged at +0.17 / +0.25 / +0.18.
Tree count / learning rate: the tuned head runs to the 500-tree ceiling at
lr 0.0028 (MAE-tuned). Cap 5000 (2,013 trees), lr 0.01, lr 0.03: all-player
Δ +0.009 / +0.011 / +0.012 at best, every interval includes 0 (lr 0.03's own
seed placebo moves up to 0.018); off-board gains +0.03 / +0.04 / +0.07 (lr
0.01 late_apy interval excludes 0) but AUC for y > 0 drops 0.989 → 0.976 —
a trade, not a win. The board gap does not move in any arm.
Within-position board rank (--hasan-pos-rank, rank inside draft year ×
position group): FAILS the bar (all-player gain beyond the seed placebo).
add: all players +0.000 / +0.003 / +0.003; board players +0.004 / +0.013
[+0.001, +0.026] / +0.012, 7-8/9 wins, reproduced at seed 43 (+0.007 /
+0.008 / +0.007); off-board -0.010 / -0.016 / -0.013 (intervals just include
0); AUC unchanged. Board gap barely moves (+0.165 / +0.220 / +0.158).
replace: everything within noise; pred~board falls to 0.61.

RESULT (2026-10-30): NO. Weights ×3 / ×10 / ×30 at production hyperparameters
(no retune) leave the head's use of the board unchanged: pred~board within
group 0.451 → 0.455 / 0.454 / 0.430, and hasan_rank's partial Spearman beyond
the OOF stays +0.26 / +0.38 / +0.29 (value / late_snap / late_apy) in every
arm. 00u referees: all arms within the seed placebo on all / board / offboard
(best: ×30 on board players +0.017 / +0.063 / +0.021, intervals include 0).
The ×1 arm reproduces production OOF exactly (max |diff| 0.0).
Striking side result: on board players, hasan_rank ALONE beats the head
(0.311 / 0.445 / 0.357 vs 0.185 / 0.198 / 0.167) — the head ranks board
players worse than one of its own inputs. Loss share was not the cause.

Diagnostic (2026-10-30, same day): not tree count either. --lr-cap 0.03 /
0.01 raises the few-tree heads (QB/RB/WR/TE/LB/DB, 5-19 trees at lr ~0.15-0.30)
to 64-586 trees; referees all within placebo (board players +0.015 / -0.003 /
+0.011), pred~board 0.465 / 0.459, board-beyond unchanged. hasan_rank is
already the #1 SHAP feature in 6/9 heads (13-37% of |SHAP| on board rows). On
board players the head matches the board at separating bridge from non-bridge
but loses badly WITHIN bridge on its own training label (all 9 groups, e.g. RB
0.10 vs 0.46; latest fold 0.32 vs 0.48). Structural: the zero-padded task is
dominated by made-NFL-or-not, and the combine features that serve it add noise
to the within-NFL ranking.

Motivation (experiments/00w, 2026-10-30): hasan_rank adds +0.24 partial
Spearman beyond the impact head's OOF prediction in every CV fold, whatever
the number of board seasons in training. Board rows are ~2.8% of the
zero-padded training set, so splits on hasan_rank gain little loss reduction
and the head barely uses it. Arms re-weight those rows
(config.IMPACT_BOARD_ROW_WEIGHT); the board stays an input, nothing is blended
into the score.

Each arm runs step 8's own train_model_for_group for the impact head at the
production tuned hyperparameters (no retune) and writes the OOF predictions.
Nothing in draft_models/ is written. Score the arms with
experiments/00u_value_target_referee_ab.py --oof, on --subset all / board /
offboard, and re-run the hasan-beyond-model check per arm.

--global refits the DEPLOYED impact model instead: the step-8b global
impact_apy head (config.GLOBAL_SCORE_OVERRIDES routes every position's
impact_score through it), at its tuned hyperparameters, tuning folds only.

Usage:
    00x_board_row_weight_ab.py --weight W [--seed 42] [--lr-cap LR] --out PATH
    00x_board_row_weight_ab.py --global --weight W [--seed 42] [--n-estimators N] [--lr LR]
        [--hasan-pos-rank add|replace] --out PATH
"""
import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from config import PROCESSED_DIR, POSITION_GROUPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _run_global(args):
    import json
    g = importlib.import_module("08b_global_draft_model")
    g._p8.IMPACT_BOARD_ROW_WEIGHT = args.weight          # read by _board_row_weights
    g.LGBM_PARAMS["random_state"] = args.seed            # the placebo arm's only change
    df, feature_cols = g.build_global_training_frame(
        pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet"))
    if args.hasan_pos_rank:
        # hasan_rank is the overall board rank; a model pooled across
        # positions must learn what rank N means per position. Rank within
        # the board's own position-group cohort for that draft year instead.
        board_year = (pd.read_parquet(PROCESSED_DIR / "hasan_ranks.parquet")
                      .set_index("player_id")["draft_year"])
        df["_board_year"] = df["player_id"].map(board_year)
        on = df["hasan_rank"].notna()
        df["hasan_pos_rank"] = np.nan
        df.loc[on, "hasan_pos_rank"] = (df[on].groupby(["_board_year", "position_group"])
                                        ["hasan_rank"].rank(method="min"))
        feature_cols = feature_cols + ["hasan_pos_rank"]
        if args.hasan_pos_rank == "replace":
            feature_cols = [c for c in feature_cols if c != "hasan_rank"]
        logging.info("hasan_pos_rank (%s): %d rows; e.g. median overall rank of pos-rank 1 = %.0f",
                     args.hasan_pos_rank, int(on.sum()),
                     df.loc[df["hasan_pos_rank"] == 1, "hasan_rank"].median())
    tuned = dict(json.loads(g.GLOBAL_TUNED_PARAMS_PATH.read_text()).get("global/impact_apy"))
    if args.n_estimators is not None:
        tuned["n_estimators"] = args.n_estimators
    if args.lr is not None:
        tuned["learning_rate"] = args.lr
    _, metrics, _, oof, _ = g._train_one_global(df, feature_cols, "impact_target",
                                                "impact_apy", tuned)
    oof = (oof.rename(columns={"position_group": "pos_group"})
              .assign(model_name="impact"))
    oof.to_parquet(args.out, index=False)
    logging.info("global weight=%g seed=%d cv_mae=%.5f best_iter=%d: %d OOF rows → %s",
                 args.weight, args.seed, metrics["cv_mae"], metrics["avg_best_iter"],
                 len(oof), args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lr-cap", type=float, default=None,
                    help="diagnostic: cap the tuned learning_rate (n_estimators "
                         "ceiling raised to 5000; early stopping picks the count)")
    ap.add_argument("--global", dest="global_model", action="store_true",
                    help="refit the deployed 8b global impact_apy head instead")
    ap.add_argument("--n-estimators", type=int, default=None,
                    help="--global diagnostic: override the tree ceiling (early "
                         "stopping still picks the count)")
    ap.add_argument("--lr", type=float, default=None,
                    help="--global diagnostic: override the tuned learning_rate")
    ap.add_argument("--hasan-pos-rank", choices=["add", "replace"], default=None,
                    help="--global: board rank WITHIN (draft year × position group) as "
                         "a feature, added to or replacing the overall hasan_rank")
    args = ap.parse_args()

    if args.global_model:
        _run_global(args)
        return

    step8 = importlib.import_module("08_draft_models")
    step8.IMPACT_BOARD_ROW_WEIGHT = args.weight
    if args.lr_cap is not None:
        _load = step8._load_tuned_params

        def _capped(pos_group, model_name):
            p = dict(_load(pos_group, model_name) or {})
            if p.get("learning_rate", step8.LGBM_PARAMS["learning_rate"]) > args.lr_cap:
                p.update(learning_rate=args.lr_cap, n_estimators=5000)
            return p
        step8._load_tuned_params = _capped
    step8.LGBM_PARAMS["random_state"] = args.seed   # the placebo arm's only change

    snapshot = step8.assemble_snapshot(include_derived=True, logger=step8.logger)
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    oofs = []
    for pos_group in POSITION_GROUPS:
        try:
            _, _, metrics, _, oof = step8.train_model_for_group(
                snapshot, labels, pos_group, "impact_target", "impact")
        except ValueError as exc:
            logging.info("skip %s: %s", pos_group, exc)
            continue
        oofs.append(oof)
        logging.info("%s weight=%g seed=%d cv_mae=%.5f", pos_group, args.weight,
                     args.seed, metrics["cv_mae"])
    out = pd.concat(oofs, ignore_index=True)
    out.to_parquet(args.out, index=False)
    logging.info("%d OOF rows → %s", len(out), args.out)


if __name__ == "__main__":
    main()
