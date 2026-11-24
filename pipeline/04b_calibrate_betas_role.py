"""
Step 4b: Calibrate EWMA Decay Betas Against NFL Impact Outcomes

For each stat, finds β ∈ [0.990, 1.000] that maximises the Pearson r between
the terminal (end-of-college-career) EWMA estimate and the NFL impact target:
    r(EWMA, impact_target)
(Until 2026-10-31 the objective was r(EWMA, role_target); it moved to impact
when the role heads were dropped. The file name keeps "role" for history.)

This is a one-shot calibration script — NOT part of the main pipeline sequence.
Run it after nfl_draft_labels.parquet exists (Step 07), then re-run Steps 06–10
to pick up the improved betas.

Usage:
    python pipeline/04b_calibrate_betas_role.py

Inputs:
    data/processed/adjusted_gamelogs.parquet   (Step 05)
    data/processed/nfl_draft_labels.parquet    (Step 07)

Outputs:
    data/processed/decay_betas_role.json
    data/processed/decay_betas_role_summary.parquet
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DIR, MODELED_STATS
from src.models.decay import optimize_betas_for_role
from src.utils.cache_manifest import (
    write_manifest, parquet_metadata_fingerprint,
)

STEP_NAME = "04b_calibrate_betas_role"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    labels_path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    if not labels_path.exists():
        logger.error(
            "nfl_draft_labels.parquet not found — run Step 07 first.\n"
            "On first setup: run the full pipeline once, then re-run this script "
            "and re-run Steps 06–10."
        )
        sys.exit(1)

    adj_path = PROCESSED_DIR / "adjusted_gamelogs.parquet"
    if not adj_path.exists():
        logger.error("adjusted_gamelogs.parquet not found — run Step 05 first")
        sys.exit(1)

    logger.info("Loading adjusted gamelogs...")
    adjusted = pd.read_parquet(adj_path)
    adjusted["game_date"] = pd.to_datetime(adjusted["game_date"])
    logger.info("Loaded %d rows, %d cols", *adjusted.shape)

    logger.info("Loading NFL draft labels...")
    role_labels = pd.read_parquet(
        labels_path,
        columns=["player_id", "role_target", "impact_target",
                 "last_college_season_year", "is_bridge"],
    )
    role_labels = role_labels[role_labels["is_bridge"]].dropna(
        subset=["role_target", "impact_target", "player_id"])
    logger.info("Bridge players with role labels: %d", len(role_labels))
    logger.info(
        "Cal cohort (last season ≤2020): %d | Val cohort (>2020): %d",
        (role_labels["last_college_season_year"] <= 2020).sum(),
        (role_labels["last_college_season_year"] > 2020).sum(),
    )

    all_stats = sorted({s for stats in MODELED_STATS.values() for s in stats})
    stats_to_run = [s for s in all_stats if f"adjusted_{s}" in adjusted.columns]
    logger.info("Stats to calibrate: %d", len(stats_to_run))

    # Objective switched role → impact 2026-10-31, when the role heads were
    # dropped: the betas should serve the head that ships (impact_score is
    # the impact_target/APY head). The output filename keeps its historical
    # "_role" suffix so step 6's cache priority chain is unchanged.
    logger.info("Calibrating betas (impact-only objective)...")
    betas, summary = optimize_betas_for_role(
        adjusted_games=adjusted,
        role_labels=role_labels,
        stats=stats_to_run,
        cal_season_cutoff=2020,
        bounds=(0.990, 1.0 - 1e-7),
        role_weight=0.0,
        impact_weight=1.0,
    )

    betas_path = PROCESSED_DIR / "decay_betas_role.json"
    betas_path.write_text(json.dumps(betas, indent=2))
    fingerprint = {
        "stats":                    sorted(stats_to_run),
        "cal_season_cutoff":        2020,
        "bounds":                   [0.990, 1.0 - 1e-7],
        "objective":                "impact_target",
        "nfl_labels_file":          parquet_metadata_fingerprint(labels_path),
        "adjusted_gamelogs_file":   parquet_metadata_fingerprint(adj_path),
    }
    write_manifest(
        betas_path, fingerprint=fingerprint,
        mode_used="fresh", step_name=STEP_NAME,
    )
    logger.info("Saved role-calibrated betas → %s", betas_path)

    summary_path = PROCESSED_DIR / "decay_betas_role_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    logger.info("Saved summary → %s", summary_path)

    logger.info("\nBeta calibration summary (sorted by r_cal):\n%s",
                summary.sort_values("r_cal", ascending=False).to_string(index=False))

    logger.info(
        "\n=== Step 4b complete ===\n"
        "Re-run skill estimation to use role-calibrated betas:\n"
        "    python pipeline/06_skill_estimation.py\n"
        "    python pipeline/08_draft_models.py\n"
        "    python pipeline/09_comps.py\n"
        "    python pipeline/10_output.py"
    )


if __name__ == "__main__":
    main()
