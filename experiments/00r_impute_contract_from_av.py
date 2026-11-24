"""
Repair false-zero contract labels by imputing from career AV.

The problem. OverTheCap's coverage of SMALL contracts improved enormously over
time, so pre-2011 bridge players carry false "no second contract" zeros. Held
against career w_AV — an independent PFR outcome that does NOT share the
failure mode — a third of 2004-06 players with w_AV >= 30 are labelled
washouts.

Why imputation rather than the two things already tried and rejected. Dropping
those rows (2026-07-26 cohort A/B) and gating them to zero (the label-gate A/B)
both LOST, and both lost the same way: they shed training volume, and for these
tree-ensemble heads volume beats the label noise it carries. Imputation is the
one repair that keeps every row and fixes the label, so it does not have that
failure mode.

Not leakage: `career_w_av` and second-contract APY are both NFL OUTCOMES, while
every model feature is college-side. Using one outcome to repair another is
label engineering.

Method — conservative, quantile-mapped, and deliberately narrow:

  * Only bridge rows drafted before IMPUTE_ERA_CUTOFF, currently labelled 0,
    whose career w_AV is at least MIN_WAV_TO_IMPUTE. Rows with w_AV == 0 are
    LEFT ALONE: zero production plus zero contract is a consistent story, and
    imputing them would inject the opposite error.
  * Quantile mapping within position: find the player's w_AV percentile in the
    CLEAN-era (2011+) w_AV distribution for his position, then read off that
    percentile of the clean-era APY distribution. Monotone, bounded by observed
    APY values, no regression extrapolation.
  * Everything else keeps its observed label.

Arms are scored fold-matched with the starved-fold guard, per the 2026-07-26
lesson: imputation removes no rows, so no fold is starved, but the harness is
shared so the guard is applied uniformly.

Read-only with respect to production: writes a variant labels file plus a JSON.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, ROOT_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = ROOT_DIR / "experiments"

IMPUTE_ERA_CUTOFF = 2011      # only classes before this are suspect
CLEAN_ERA_START = 2011        # source distribution for the quantile map
MIN_CLEAN_PER_POS = 50        # else fall back to the all-position map


def build_imputed_labels(min_wav: float) -> tuple[pd.DataFrame, dict]:
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    # career_w_av already rides along in the labels file; only position_group
    # has to come from the bridge table.
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    pos = (bridge[["pfr_id", "position_group"]]
           .dropna(subset=["pfr_id"]).drop_duplicates("pfr_id"))

    df = labels.merge(pos, on="pfr_id", how="left")
    is_bridge = df["is_bridge"].fillna(False).astype(bool)

    clean = df[is_bridge & (df["draft_year"] >= CLEAN_ERA_START)
               & df["career_w_av"].notna()]

    # Per-position quantile maps from the clean era, plus a global fallback.
    def _map_for(pos):
        sub = clean[clean["position_group"] == pos]
        if len(sub) < MIN_CLEAN_PER_POS:
            sub = clean
        return (np.sort(sub["career_w_av"].values),
                np.sort(sub["impact_target"].values))

    maps = {pos: _map_for(pos) for pos in clean["position_group"].dropna().unique()}
    global_map = (np.sort(clean["career_w_av"].values),
                  np.sort(clean["impact_target"].values))

    target = (is_bridge
              & (df["draft_year"] < IMPUTE_ERA_CUTOFF)
              & (df["impact_target"] == 0.0)
              & df["career_w_av"].notna()
              & (df["career_w_av"] >= min_wav))

    imputed = df["impact_target"].copy()
    n_by_pos: dict = {}
    for idx in df.index[target]:
        pos = df.at[idx, "position_group"]
        av_sorted, apy_sorted = maps.get(pos, global_map)
        if len(av_sorted) < 10:
            continue
        pct = float(np.searchsorted(av_sorted, df.at[idx, "career_w_av"])
                    / len(av_sorted))
        val = float(np.quantile(apy_sorted, min(max(pct, 0.0), 1.0)))
        imputed.at[idx] = val
        n_by_pos[pos] = n_by_pos.get(pos, 0) + 1

    out = labels.copy()
    out["impact_target"] = imputed.values
    # role_target is derived from the trajectory model, not from this column,
    # and every role arm of the cohort A/B regressed — leave it untouched.
    stats = {
        "min_wav_to_impute": min_wav,
        "n_imputed": int(target.sum()),
        "n_by_position": n_by_pos,
        "mean_imputed_value": float(imputed[target].mean()) if target.any() else 0.0,
        "n_left_as_true_zero": int((is_bridge
                                    & (df["draft_year"] < IMPUTE_ERA_CUTOFF)
                                    & (df["impact_target"] == 0.0)
                                    & (df["career_w_av"].fillna(0) < min_wav)).sum()),
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-wav", type=float, default=10.0,
                    help="Minimum career w_AV for a zero label to be treated "
                         "as false and imputed (default 10)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_df, stats = build_imputed_labels(args.min_wav)
    path = Path(args.out) if args.out else (
        PROCESSED_DIR / f"nfl_draft_labels_imputed_w{int(args.min_wav)}.parquet")
    out_df.to_parquet(path, index=False)

    logger.info("Imputed %d false-zero labels (min w_AV=%.0f), mean value %.4f",
                stats["n_imputed"], args.min_wav, stats["mean_imputed_value"])
    logger.info("  left as true zeros (w_AV below threshold): %d",
                stats["n_left_as_true_zero"])
    logger.info("  by position: %s", stats["n_by_position"])
    logger.info("→ %s", path)

    js = EXPERIMENTS_DIR / f"impute_av_stats_{datetime.now():%Y%m%d_%H%M%S}.json"
    js.write_text(json.dumps(stats, indent=2, default=float))


if __name__ == "__main__":
    main()
