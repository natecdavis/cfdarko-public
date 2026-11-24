"""
L1-vs-L2 objective A/B for the impact heads, at CACHED hyperparameters.

Context (2026-07-22 pipeline audit): `LGBM_PARAMS` sets no `objective`, so
every head trains on LightGBM's default L2 while step 8 tunes and reports MAE.
Switching the impact heads to `regression_l1` aligns the training loss with the
evaluation metric.

This script is the CHEAP pre-check before paying for `--retune-only 8,8.5`.
It runs both objectives through step 8's own `train_model_for_group`, so the
folds, feature resolution, monotone constraints, and holdout split are
identical to production — the only thing that changes is `objective`.

Read the result with two caveats in mind:

  1. The comparison is BIASED AGAINST L1: the cached hyperparameters were
     tuned under L2, so L1 runs on someone else's optimum. A win here is
     therefore meaningful; a small loss is not disqualifying.
  2. cv_mae is PARTLY DEFINITIONAL. L1 optimizes the conditional median and
     the metric is MAE, so some improvement is guaranteed by construction and
     is not evidence of a better model. The load-bearing columns are the
     holdout Spearman / rank metrics, which measure what the draft model is
     actually for. A head that improves MAE while Spearman falls is a
     regression dressed as a win.

The role heads are deliberately excluded — they're the probability-style head
(cross_entropy tested neutral) and L2 is left alone there.

Usage:
    python experiments/00m_l1_objective_ab.py
    python experiments/00m_l1_objective_ab.py --positions DL,WR,OL
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import POSITION_GROUPS, PROCESSED_DIR

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

HEADS = [("impact_target", "impact"), ("impact_w_av_per_game", "impact_wav")]


def _load_step8():
    """Import pipeline/08_draft_models.py under a module name (leading digits
    make it non-importable normally)."""
    path = Path(__file__).resolve().parent.parent / "pipeline" / "08_draft_models.py"
    spec = importlib.util.spec_from_file_location("_p8_ab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(p8, snapshot, labels, pos, target_col, model_name) -> dict | None:
    try:
        out = p8.train_model_for_group(snapshot, labels, pos, target_col, model_name)
        return out[2]                            # (model, feature_cols, metrics, ...)
    except Exception as exc:                     # small heads legitimately fail
        logger.warning("%s/%s skipped: %s", pos, model_name, exc)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default=None,
                    help="Comma-separated subset (default: all).")
    args = ap.parse_args()

    positions = ([p.strip() for p in args.positions.split(",")]
                 if args.positions else list(POSITION_GROUPS.keys()))

    p8 = _load_step8()
    snapshot = p8.assemble_snapshot()
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")

    baseline_params = dict(p8.LGBM_PARAMS)
    baseline_monotone = p8.MONOTONE_ENABLED
    rows = []

    for pos in positions:
        for target_col, model_name in HEADS:
            # ── L2 (current production behaviour) ──
            p8.LGBM_PARAMS = dict(baseline_params)
            m_l2 = _run(p8, snapshot, labels, pos, target_col, model_name)

            # ── L1 ──
            # LightGBM REFUSES monotone_constraints under regression_l1
            # ("Cannot use monotone_constraints in regression_l1 objective").
            # Every MONOTONE_ENABLED head is an impact head, so for those the
            # L1 arm necessarily also drops the hasan_rank / age_at_draft
            # constraints — two changes at once. Those rows are flagged
            # `confounded` and must not be read as a clean objective test.
            was_monotone = (pos, model_name) in baseline_monotone
            p8.LGBM_PARAMS = {**baseline_params, "objective": "regression_l1"}
            if was_monotone:
                p8.MONOTONE_ENABLED = frozenset()
            m_l1 = _run(p8, snapshot, labels, pos, target_col, model_name)
            p8.LGBM_PARAMS = dict(baseline_params)
            p8.MONOTONE_ENABLED = baseline_monotone

            if not m_l2 or not m_l1:
                continue
            rows.append({
                "pos": pos,
                "head": model_name,
                "confounded": "mono-dropped" if was_monotone else "",
                "cv_mae_l2": m_l2.get("cv_mae"),
                "cv_mae_l1": m_l1.get("cv_mae"),
                "rho_l2": m_l2.get("holdout_spearman"),
                "rho_l1": m_l1.get("holdout_spearman"),
                "hmae_l2": m_l2.get("holdout_mae"),
                "hmae_l1": m_l1.get("holdout_mae"),
            })
            print(f"  {pos:3s}/{model_name:10s} "
                  f"cv_mae {rows[-1]['cv_mae_l2']:.4f} → {rows[-1]['cv_mae_l1']:.4f}   "
                  f"rho {str(rows[-1]['rho_l2'])[:6]} → {str(rows[-1]['rho_l1'])[:6]}",
                  flush=True)

    if not rows:
        print("No heads evaluated.")
        return

    df = pd.DataFrame(rows)
    df["cv_mae_pct"] = (df["cv_mae_l1"] - df["cv_mae_l2"]) / df["cv_mae_l2"] * 100
    df["rho_delta"] = df["rho_l1"] - df["rho_l2"]

    out = PROCESSED_DIR / "l1_objective_ab.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("L1 vs L2 at cached (L2-tuned) hyperparameters")
    print("=" * 78)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\ncv_mae   mean {df['cv_mae_pct'].mean():+.2f}%  "
          f"median {df['cv_mae_pct'].median():+.2f}%  "
          f"wins {(df['cv_mae_pct'] < 0).sum()}/{len(df)}")
    rho = df.dropna(subset=["rho_delta"])
    if len(rho):
        print(f"holdout rho  mean {rho['rho_delta'].mean():+.4f}  "
              f"median {rho['rho_delta'].median():+.4f}  "
              f"wins {(rho['rho_delta'] > 0).sum()}/{len(rho)}")

    clean = df[df["confounded"] == ""]
    if len(clean) and len(clean) < len(df):
        print(f"\nCLEAN heads only (no monotone constraint to drop), n={len(clean)}:")
        print(f"  cv_mae mean {clean['cv_mae_pct'].mean():+.2f}%  "
              f"wins {(clean['cv_mae_pct'] < 0).sum()}/{len(clean)}")
        cr = clean.dropna(subset=["rho_delta"])
        if len(cr):
            print(f"  holdout rho mean {cr['rho_delta'].mean():+.4f}  "
                  f"wins {(cr['rho_delta'] > 0).sum()}/{len(cr)}")
        print(f"  ({len(df) - len(clean)} monotone heads excluded — their L1 arm "
              "also lost hasan_rank/age_at_draft constraints.)")
    print("\nRanking (holdout rho) is the load-bearing signal; cv_mae gains are "
          "partly definitional under an MAE-aligned objective.")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
