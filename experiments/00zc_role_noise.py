"""
Noise floor of the deployed role score.

RESULT (2026-10-30): the deployed role model is SEED-UNSTABLE; the forty
sign change was ordinary noise. Within-position rank correlation of
final-model predictions:
                            all    board  2025 class  top 5%
    impact  s42 vs s43      0.981  0.997  0.981       0.995
    role    s42 vs s43      0.684  0.879  0.533       0.903
    role    s42 vs s44      0.874  0.886  0.882       0.905
    role    s43 vs s44      0.693  0.891  0.528       0.899
    role    s42 vs s42 forty-negated  0.855  0.923  0.773  0.945
OOF logloss (role_target > 0.5) by seed: 0.0455 / 0.0566 / 0.0443 (forty-
negated 0.0444); seed 43 is worse in every fold. Early stopping picks
31-56 trees depending on the seed. Impact OOF bridge Spearman: 0.319 vs
0.322. Cause: the tuned role params are a high-variance configuration
(learning_rate 0.146, min_child_samples 8, num_leaves 28, max_depth 9,
weak regularization) whose tree count is chosen by early stopping on a
single-season slice. Any past role-score A/B smaller than this seed spread
was not measurable.

Why (2026-10-30): negating one feature (the combine_forty_z sign fix) should
leave a tree model essentially unchanged, yet the deployed role score's
within-position rank correlation with its previous version was 0.73 for the
2025-season class (raw global role model 0.89; impact 0.99). Before trusting
any role-score A/B, measure how much the role model moves with nothing but
the seed changed.

Arms: the step-8b global role head (deployed for every position via
GLOBAL_SCORE_OVERRIDES), refit at its tuned hyperparameters with
LGBM_PARAMS random_state 42 / 43 / 44, plus seed 42 with combine_forty_z
negated (a change trees should ignore). The global impact_apy head at seeds
42 / 43 is the comparison. Reported per pair of arms: within-position rank
correlation of the final models' predictions (all players, board players,
2025-season class, and players scoring in the top 5% of either arm), and
per arm the OOF quality on the tuning folds (logloss and AUC against
role_target > 0.5; Spearman among bridge rows for impact).

Usage:
    00zc_role_noise.py --head role|impact_apy --seed S [--negate-forty] --out PATH
    00zc_role_noise.py --report DIR
"""
import argparse
import importlib
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
TARGET = {"role": "role_target", "impact_apy": "impact_target"}


def _fit(args):
    g = importlib.import_module("08b_global_draft_model")
    g.LGBM_PARAMS["random_state"] = args.seed
    df, feature_cols = g.build_global_training_frame(
        pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet"))
    if args.negate_forty:
        df["combine_forty_z"] = -df["combine_forty_z"]
    tuned = json.loads(g.GLOBAL_TUNED_PARAMS_PATH.read_text()).get(f"global/{args.head}")
    model, metrics, fc, oof, _ = g._train_one_global(df, feature_cols, TARGET[args.head],
                                                     args.head, tuned)
    final = pd.DataFrame({
        "player_id": df["player_id"].astype(str).values,
        "position_group": df["position_group"].values,
        "last_college_season_year": df["last_college_season_year"].values,
        "on_board": df["hasan_rank"].notna().values,
        "pred": model.predict(df[fc].to_numpy(dtype=float, na_value=np.nan))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(args.out, index=False)
    oof.to_parquet(args.out.with_name(args.out.stem + "_oof.parquet"), index=False)
    logging.info("%s seed=%d negate_forty=%s best_iter=%d → %s", args.head, args.seed,
                 args.negate_forty, metrics["avg_best_iter"], args.out)


def _rank_corr(a, b, mask=None):
    d = a.merge(b, on=["player_id", "position_group"], suffixes=("_a", "_b"))
    if mask is not None:
        d = d[mask(d)]
    r = [spearmanr(g["pred_a"], g["pred_b"])[0] for _, g in d.groupby("position_group")
         if len(g) >= 20 and g["pred_a"].nunique() > 1]
    return round(float(np.mean(r)), 3)


def _report(d: Path):
    arms = {p.stem: pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))
            if not p.stem.endswith("_oof")}
    print("final-model predictions, within-position rank correlation:")
    rows = []
    for a, b in itertools.combinations(arms, 2):
        if a.split("_")[0] != b.split("_")[0]:
            continue
        A, B = arms[a], arms[b]
        rows.append({
            "pair": f"{a} vs {b}",
            "all": _rank_corr(A, B),
            "board": _rank_corr(A, B, lambda x: x["on_board_a"]),
            "2025 class": _rank_corr(A, B, lambda x: x["last_college_season_year_a"] == 2025),
            "top 5%": _rank_corr(A, B, lambda x: (x["pred_a"] >= x["pred_a"].quantile(0.95))
                                 | (x["pred_b"] >= x["pred_b"].quantile(0.95)))})
    print(pd.DataFrame(rows).set_index("pair").to_string())

    print("\nOOF quality on the tuning folds:")
    for name in arms:
        o = pd.read_parquet(d / f"{name}_oof.parquet")
        if name.startswith("role"):
            o = o.dropna(subset=["y"])
            yb = (o["y"] > 0.5).astype(int)
            p = o["pred"].clip(1e-6, 1 - 1e-6)
            print(f"  {name:22s} logloss {log_loss(yb, p):.4f}  AUC {roc_auc_score(yb, p):.4f}")
        else:
            b = o[o["is_bridge"].astype(bool)]
            rho = np.mean([spearmanr(g["pred"], g["y"])[0] for _, g in b.groupby("position_group")
                           if len(g) >= 20])
            print(f"  {name:22s} bridge-only Spearman {rho:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=list(TARGET))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negate-forty", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    if args.report:
        _report(args.report)
    else:
        _fit(args)


if __name__ == "__main__":
    main()
