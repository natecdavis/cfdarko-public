"""Tests for src/models/merged_board.py (the production capped merge)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.merged_board import (  # noqa: E402
    CAP_Q, MergedBoard, ShrunkIso, offboard_percentile,
)


def _oof(seed=0, seasons=range(2013, 2022), n_off=200, n_board=30):
    """Synthetic OOF frame: board usage falls with rank, off-board usage
    rises with the model score; two positions."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in seasons:
        for pos in ("WR", "OL"):
            r = np.arange(1, n_board + 1)
            rows.append(pd.DataFrame({
                "player_id": [f"b{y}{pos}{i}" for i in r], "pos_group": pos,
                "last_college_season_year": y, "hasan_rank": r * 10.0,
                "pred": rng.random(n_board), "nfl": True,
                "usage": np.clip(0.9 - 0.025 * r + rng.normal(0, 0.05, n_board), 0, 1)}))
            p = rng.random(n_off)
            nfl = rng.random(n_off) < p ** 4
            rows.append(pd.DataFrame({
                "player_id": [f"o{y}{pos}{i}" for i in range(n_off)], "pos_group": pos,
                "last_college_season_year": y, "hasan_rank": np.nan, "pred": p,
                "nfl": nfl, "usage": np.where(nfl, np.clip(p * 0.6, 0, 1), 0.0)}))
    return pd.concat(rows, ignore_index=True)


def test_shrunk_iso_weight_moves_toward_pooled():
    x = np.r_[np.linspace(0, 1, 10), np.linspace(0, 1, 1000)]
    y = np.r_[np.ones(10), np.linspace(0, 1, 1000)]
    pos = np.r_[["A"] * 10, ["B"] * 1000]
    iso = ShrunkIso(increasing=True, k=100).fit(x, y, pos)
    assert iso.w["A"] == pytest.approx(10 / 110)
    # A's own curve is flat at 1; shrinkage pulls its low end toward pooled.
    assert iso.predict([0.0], ["A"])[0] < 0.5


def test_offboard_percentile_ordinary_and_reference_sized():
    s = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0, 9.0, 8.0])
    cls = pd.Series([2020] * 5 + [2025] * 2)
    pos = pd.Series(["WR"] * 7)
    pct = offboard_percentile(s, cls, pos)
    assert pct.iloc[0] == 1.0 and pct.iloc[4] == pytest.approx(0.2)
    # The reference size only ever shrinks N: a current class smaller than
    # the reference keeps its own size.
    pct = offboard_percentile(s, cls, pos, n_ref={"WR": 4}, ref_cls=2025)
    assert pct.iloc[5] == 1.0 and pct.iloc[6] == pytest.approx(0.5)
    # A larger current class is cut at the reference size, floored at 0.
    big = pd.Series(np.arange(10, 0, -1.0))
    pct = offboard_percentile(big, pd.Series([2025] * 10), pd.Series(["WR"] * 10),
                              n_ref={"WR": 4}, ref_cls=2025)
    assert pct.iloc[3] == pytest.approx(0.25) and (pct.iloc[4:] == 0).all()


def _snap(oof):
    snap = oof.drop(columns=["usage", "nfl"]).rename(columns={"pred": "score"})
    return snap


def test_board_curve_decreases_and_offboard_is_capped():
    oof = _oof()
    mb = MergedBoard.fit(oof, fit_last_season=2021)
    snap = _snap(oof[oof["last_college_season_year"] == 2021])
    snap = pd.concat([snap, snap.iloc[:1].assign(last_college_season_year=2026)])
    out = mb.predict(snap, "score")
    b = out[out["merged_source"] == "board"].join(snap["hasan_rank"])
    b = b[~b.index.duplicated()]
    wr = b.join(snap["pos_group"].rename("g"))
    wr = wr[wr["g"] == "WR"].sort_values("hasan_rank")
    assert wr["merged_usage"].is_monotonic_decreasing
    off = out[out["merged_source"].isin(["model", "model_capped"])]
    cap = off.join(snap["pos_group"].rename("g")).groupby("g")["merged_usage"].max()
    for g, v in cap.items():
        assert v <= mb.cap[g] + 1e-12
    assert (out["merged_source"] == "model_capped").any()


def test_excluded_groups_and_duplicate_index():
    oof = _oof(seasons=range(2013, 2018), n_off=50, n_board=10)
    mb = MergedBoard.fit(oof, fit_last_season=2017)
    snap = _snap(oof[oof["last_college_season_year"] == 2017]).copy()
    snap.loc[snap.index[:3], "pos_group"] = "K"
    snap.index = [0] * len(snap)          # fully duplicated index
    out = mb.predict(snap, "score")
    assert out.index.equals(snap.index)
    assert out["merged_usage"].iloc[:3].isna().all()
    assert out["merged_usage"].iloc[3:].notna().all()


def test_fit_uses_only_observed_classes():
    oof = _oof()
    late = oof["last_college_season_year"] > 2018
    oof.loc[late, "usage"] = 0.0           # unplayed y3-4 seasons read as 0
    a = MergedBoard.fit(oof, fit_last_season=2018)
    b = MergedBoard.fit(_oof(), fit_last_season=2018)
    x = np.log([10.0, 100.0, 300.0])
    np.testing.assert_allclose(a.board_cal.predict(x, ["WR"] * 3),
                               b.board_cal.predict(x, ["WR"] * 3))
    assert a.diagnostics["fit_seasons"] == [2013, 2018]
    assert 0 < CAP_Q < 1


def test_value_target_and_output_column():
    """The value currency writes its own column, uses level-preserving
    shrinkage (01a), and a rescaled target gives a rescaled board."""
    oof = _oof(seasons=range(2013, 2018), n_off=50, n_board=10)
    oof["rv"] = oof["usage"] * 0.2
    oof1 = oof.assign(rv=oof["usage"])
    mu = MergedBoard.fit(oof, fit_last_season=2017)
    mv = MergedBoard.fit(oof, fit_last_season=2017, target="rv", out_col="merged_value")
    mv1 = MergedBoard.fit(oof1, fit_last_season=2017, target="rv", out_col="merged_value")
    assert mv.diagnostics["target"] == "rv" and mv.diagnostics["level_preserving"]
    assert not mu.diagnostics["level_preserving"]
    snap = _snap(oof[oof["last_college_season_year"] == 2017])
    ov, ov1 = mv.predict(snap, "score"), mv1.predict(snap, "score")
    assert "merged_value" in ov.columns and "merged_usage" not in ov.columns
    np.testing.assert_allclose(ov["merged_value"], 0.2 * ov1["merged_value"], rtol=1e-9, atol=1e-12)


def test_level_preserving_shrinkage_keeps_position_level():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.uniform(0, 5, n)
    pos = np.where(rng.random(n) < 0.05, "QB", "DB")
    y = np.clip((0.3 - 0.05 * x) * np.where(pos == "QB", 2.0, 1.0) + rng.normal(0, 0.02, n), 0, 1)
    plain = ShrunkIso(increasing=False, k=100).fit(x, y, pos)
    lvl = ShrunkIso(increasing=False, k=100, level=True).fit(x, y, pos)
    q = pos == "QB"
    bias_plain = plain.predict(x[q], pos[q]).mean() - y[q].mean()
    bias_lvl = lvl.predict(x[q], pos[q]).mean() - y[q].mean()
    assert bias_plain < -0.03 and abs(bias_lvl) < 0.01
    assert lvl.lvl["QB"] > 1.5 and all(v == 1.0 for v in plain.lvl.values())


def test_extra_board_rows_enter_board_curve_only():
    oof = _oof(seasons=range(2013, 2018), n_off=50, n_board=10)
    oof["rv"] = oof["usage"] * 0.2
    base = MergedBoard.fit(oof, fit_last_season=2017, target="rv", out_col="merged_value")
    xb = oof[oof["hasan_rank"].notna()][["hasan_rank", "pos_group"]].assign(rv=0.5)
    mb = MergedBoard.fit(oof, fit_last_season=2017, target="rv", out_col="merged_value", extra_board=xb)
    assert mb.diagnostics["n_board_projected"] == len(xb)
    assert mb.diagnostics["n_board"] == base.diagnostics["n_board"] + len(xb)
    snap = _snap(oof[oof["last_college_season_year"] == 2017])
    pb, pm = base.predict(snap, "score"), mb.predict(snap, "score")
    on = snap["hasan_rank"].notna().to_numpy()
    assert (pm["merged_value"].to_numpy()[on] > pb["merged_value"].to_numpy()[on]).all()
    np.testing.assert_allclose(pm["merged_value"].to_numpy()[~on], pb["merged_value"].to_numpy()[~on])
