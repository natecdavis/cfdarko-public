"""
Pipeline output validator.

Checks the output files from each step for required columns, row counts,
value ranges, and other data quality invariants.

Usage:
  python pipeline/validate.py --all
  python pipeline/validate.py --step 6
  python pipeline/validate.py --step 5 6 7
  python pipeline/validate.py --all --warn-only

Exit codes:
  0 — all ERROR-severity checks passed (warnings may still be logged)
  1 — at least one ERROR-severity check failed (unless --warn-only)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.validation import (
    CheckResult,
    check_step0,
    check_step1,
    check_step2,
    check_step3,
    check_step3_pff_features,
    check_step4,
    check_step5,
    check_step6,
    check_step7,
    check_step8,
    check_step9,
    check_step10,
    run_checks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def _check_step3_combined() -> list:
    """Run base step-3 schema checks plus the PFF-feature presence check."""
    return check_step3() + check_step3_pff_features()


STEP_CHECKS = {
    0: check_step0,
    1: check_step1,
    2: check_step2,
    3: _check_step3_combined,
    4: check_step4,
    5: check_step5,
    6: check_step6,
    7: check_step7,
    8: check_step8,
    9: check_step9,
    10: check_step10,
}

STEP_LABELS = {
    0:  "Pre-stage outputs       → 00* (PFF / combine / nfl_avg / hasan / wikidata)",
    1:  "Data Ingestion          → player_gamelogs.parquet",
    2:  "Bridge Players          → bridge_players.parquet",
    3:  "Player Attributes       → player_attributes.parquet (+ PFF features)",
    4:  "Contextual Models       → contextual_models/*.pkl",
    5:  "Residual Computation    → adjusted_gamelogs.parquet",
    6:  "Skill Estimation        → player_skill_snapshot.parquet",
    7:  "NFL Labels              → nfl_draft_labels.parquet",
    8:  "Draft Models            → draft_models/*.pkl",
    9:  "Comps                   → player_comps.parquet",
    10: "Final Output            → projections/latest.parquet",
}


def main():
    parser = argparse.ArgumentParser(
        description="Validate cfDARKO pipeline step outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="Validate all steps")
    group.add_argument("--step", nargs="+", type=int,
                       choices=range(0, 11), metavar="N",
                       help="Validate specific step(s), e.g. --step 0 5 6 7")
    parser.add_argument("--warn-only", action="store_true",
                        help="Exit 0 even if ERROR checks fail (just log)")
    args = parser.parse_args()

    steps = sorted(STEP_CHECKS.keys()) if args.all else sorted(args.step)

    all_passed = True
    for step in steps:
        label = STEP_LABELS[step]
        logger.info("── Step %d: %s", step, label)
        results = STEP_CHECKS[step]()
        passed = run_checks(results, raise_on_error=False)
        n_total = len(results)
        n_failed = sum(1 for r in results if not r.passed)
        if passed:
            logger.info("   Step %d: all %d checks passed", step, n_total)
        else:
            n_errors = sum(1 for r in results if not r.passed and r.severity == "ERROR")
            n_warns = sum(1 for r in results if not r.passed and r.severity == "WARNING")
            logger.info("   Step %d: %d/%d checks failed (%d errors, %d warnings)",
                        step, n_failed, n_total, n_errors, n_warns)
            if n_errors:
                all_passed = False

    logger.info("─" * 50)
    if all_passed:
        logger.info("All ERROR checks passed.")
    else:
        logger.error("One or more ERROR checks failed.")

    if not all_passed and not args.warn_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
