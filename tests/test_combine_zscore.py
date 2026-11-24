"""combine z-score direction (forty negation bug, 2026-10-30)."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("combine00d", ROOT / "pipeline" / "00d_combine_ingestion.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def _cohort(col, vals):
    return pd.DataFrame({"position_group": "WR", "cohort_year": 2025, col: vals})


def test_forty_z_is_negated_so_faster_is_higher():
    df = _cohort("combine_forty", [4.30, 4.40, 4.50, 4.60, 4.70])
    z = _m._zscore_within_cohort(df, "combine_forty", ["position_group", "cohort_year"])
    assert z.iloc[0] > z.iloc[-1]


def test_shuttle_z_is_not_negated_so_slower_is_higher():
    df = _cohort("combine_shuttle", [4.0, 4.1, 4.2, 4.3, 4.4])
    z = _m._zscore_within_cohort(df, "combine_shuttle", ["position_group", "cohort_year"])
    assert z.iloc[0] < z.iloc[-1]
