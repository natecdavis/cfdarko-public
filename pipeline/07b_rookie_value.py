"""
Step 7.5: rookie-deal value label + draft-slot curve.

Builds the label the merged board's value currency is calibrated to
(src/models/rookie_value.py; method in experiments/00zo, 00zp, 00zq) and the
slot curve / excess label (src/models/slot_curve.py; experiments/00zr).

Outputs (data/processed/):
    rookie_value_labels.parquet  per bridge player: value_y1..y5, cost_y1..y5,
                                 value_y14, cost_y14, surplus_y14, surplus_y15,
                                 group (NFL pricing position), draft_group
    slot_curve.parquet           pick 1..262 + "UDFA": expected value, cost,
                                 surplus (90% class-bootstrap band), surplus_y15
    rookie_excess.parquet        per player: surplus − slot curve at his pick

Needs: bridge_players / nfl_draft_labels (step 7), nfl_contracts,
snap_counts, replacement_levels, PFF NFL grades (data/raw/pff_nfl, manual
download via pipeline/00_pff_download.py --league nfl; seasons not yet
downloaded get neutral quality); nflverse downloads are cached under
data/raw/rookie_value/ on first run.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR  # noqa: E402
from src.models.rookie_value import build_label, LABEL_PATH  # noqa: E402
from src.models.slot_curve import build_slot_curve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("07b")

N_BOOT = 200   # slot-curve bands only; the curve itself does not depend on it


def main():
    build_label(LABEL_PATH)
    build_slot_curve(LABEL_PATH, PROCESSED_DIR, n_boot=N_BOOT)
    logger.info("Step 7.5 complete.")


if __name__ == "__main__":
    main()
