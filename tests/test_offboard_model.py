"""Tests for src/models/offboard_model.py (step 8d's off-board engine)."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from src.models.offboard_model import (  # noqa: E402
    PFF_FEATURES, OffboardBag, add_side_pff, oof_predictions, pid,
)


def _pff_snapshot():
    cols = {}
    for side, v in (("offense", 1.0), ("defense", 2.0)):
        for s in ("grades", "snap_counts"):
            for a in ("mean", "peak", "last"):
                cols[f"pff_{s}_{side}_{a}"] = [v, v]
        cols[f"pff_snap_counts_{side}_total"] = [v, v]
    return pd.DataFrame({"position_group": ["WR", "LB"], **cols})


def test_side_pff_takes_the_players_side_of_the_ball():
    out = add_side_pff(_pff_snapshot())
    for c in PFF_FEATURES:
        assert out.loc[0, c] == 1.0      # WR → offense
        assert out.loc[1, c] == 2.0      # LB → defense


def test_side_pff_missing_columns_raise():
    with pytest.raises(KeyError):
        add_side_pff(pd.DataFrame({"position_group": ["WR"]}))


def test_pid_canonical():
    assert pid(pd.Series(["123.0", 45, "-7", None])).tolist() == ["123", "45", "-7", "<NA>"]


@pytest.fixture(scope="module")
def step8():
    return importlib.import_module("08_draft_models")


def _rows(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    nfl = rng.random(n) < 1 / (1 + np.exp(-(2 * x - 2)))
    return pd.DataFrame({
        "player_id": [str(i) for i in range(n)], "pos_group": "WR",
        "last_college_season_year": rng.integers(2010, 2022, n).astype(float),
        "hasan_rank": np.nan, "x": x, "noise": rng.normal(size=n),
        "nfl": nfl, "usage": np.where(nfl, np.clip(0.3 + 0.2 * x, 0, 1), 0.0)})


def test_bag_learns_signal_and_is_deterministic(step8):
    rows = _rows()
    a = OffboardBag(seeds=(1, 2)).fit(step8, rows, rows[rows["nfl"]], ["x", "noise"]).predict(rows)
    b = OffboardBag(seeds=(1, 2)).fit(step8, rows, rows[rows["nfl"]], ["x", "noise"]).predict(rows)
    np.testing.assert_array_equal(a["score"].to_numpy(), b["score"].to_numpy())
    assert np.corrcoef(a["score"], rows["x"])[0, 1] > 0.5
    assert ((a["score"] >= 0) & (a["score"] <= 1)).all()


def test_oof_never_trains_on_its_validation_window(step8):
    rows = _rows()
    folds = [(2013, 2014, 2016), (2016, 2017, 2019)]
    oof = oof_predictions(step8, rows, ["x", "noise"], folds, seeds=(1,))
    assert set(oof["val_start"]) == {2014, 2017}
    for vs, g in oof.groupby("val_start"):
        lo, hi = {2014: (2014, 2016), 2017: (2017, 2019)}[vs]
        assert g["last_college_season_year"].between(lo, hi).all()
