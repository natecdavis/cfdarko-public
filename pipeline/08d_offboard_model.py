"""
Step 8d — Off-board model (the merged board's engine for unranked players).

Trains the seed-bagged two-stage model of src/models/offboard_model.py
(P(reach NFL) × E[y3-4 snap share | NFL], PFF side features included) and
writes:
  data/processed/offboard_models/offboard_oof_predictions.parquet
      OOF on step 8's tuning folds — what step 10 fits the merged board's
      off-board calibrator and cap on (never in-sample scores).
  data/processed/offboard_predictions.parquet
      final-model scores for every player (current class included):
      player_id, pos_group, offboard_p_nfl, offboard_usage_if_nfl,
      offboard_score.
  data/processed/offboard_models/summary.json

Final fit windows:
  stage 1 (reach NFL)   last college season <= END_YEAR - 2, like step 8b:
                        NFL status is known once the class has been drafted.
  stage 2 (usage | NFL) last college season <= END_YEAR - 2 - max(y3-4
                        offset): usage_y34 counts unplayed seasons as 0, so
                        newer classes would teach "every rookie busts".
OOF folds need no such guard: their training windows end by 2018.

Validated by experiments/00zi_offboard_pff_confirm.py (2026-11-01): vs the
same model without PFF, off-board NFL ranking +0.042 [+0.002, +0.085]; in
the capped merge, top-500 / seam / value top-M improve vs the deployed
model's off-board side.

Usage:
    python pipeline/08d_offboard_model.py [--test]
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, END_YEAR, VALUE_USAGE_SEASON_OFFSETS  # noqa: E402
from src.models.offboard_model import (  # noqa: E402
    ID_COLS, SEEDS, OffboardBag, build_rows, oof_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUT_DIR = PROCESSED_DIR / "offboard_models"
OOF_PATH = OUT_DIR / "offboard_oof_predictions.parquet"
PRED_PATH = PROCESSED_DIR / "offboard_predictions.parquet"
STAGE1_LAST_SEASON = END_YEAR - 2
STAGE2_LAST_SEASON = END_YEAR - 2 - max(VALUE_USAGE_SEASON_OFFSETS)


def main():
    ap = argparse.ArgumentParser(description="Step 8d — off-board model")
    ap.add_argument("--test", action="store_true",
                    help="No-op for run_pipeline.py compatibility")
    ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    step8 = importlib.import_module("08_draft_models")
    folds, _ = step8._split_holdout(step8._expanding_window_folds())
    snapshot = step8.assemble_snapshot(include_derived=True, logger=step8.logger)
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    rows, feats = build_rows(step8, snapshot, labels, bridge)

    oof = oof_predictions(step8, rows, feats, folds)
    oof.to_parquet(OOF_PATH, index=False)
    logger.info("Saved OOF (%d rows) → %s", len(oof), OOF_PATH)

    yr = rows["last_college_season_year"].fillna(0)
    tr1 = rows[yr <= STAGE1_LAST_SEASON]
    tr2 = rows[(yr <= STAGE2_LAST_SEASON) & rows["nfl"]]
    bag = OffboardBag().fit(step8, tr1, tr2, feats)
    pred = bag.predict(rows)
    out = (rows[ID_COLS + ["hasan_rank"]].join(pred)
           .rename(columns={"p_nfl": "offboard_p_nfl", "usage_if_nfl": "offboard_usage_if_nfl",
                            "score": "offboard_score"})
           .drop_duplicates("player_id", keep="last"))
    out.to_parquet(PRED_PATH, index=False)
    logger.info("Final fit: stage 1 %d rows (≤%d), stage 2 %d NFL rows (≤%d); trees %s",
                len(tr1), STAGE1_LAST_SEASON, len(tr2), STAGE2_LAST_SEASON, bag.trees)
    logger.info("Saved predictions (%d players) → %s", len(out), PRED_PATH)

    summary = {
        "seeds": list(SEEDS), "n_features": len(feats), "features": feats,
        "stage1_last_season": STAGE1_LAST_SEASON, "stage2_last_season": STAGE2_LAST_SEASON,
        "n_stage1": int(len(tr1)), "n_stage2": int(len(tr2)), "trees": bag.trees,
        "folds": [list(map(int, f)) for f in folds], "n_oof": int(len(oof)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("=== Step 8d complete ===")


if __name__ == "__main__":
    main()
