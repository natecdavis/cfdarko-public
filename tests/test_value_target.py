"""
Unit tests for the value target (src/scrapers/nfl_data.py, 2026-10-29).

Covers the component definitions (best deal not earliest, guarantees as a cap
share, rookie/5th-year exclusion, snap vs depth-chart usage, K/P), the
percentile composite (zero pinning, per-group ranking, missing components),
the starter-share and crosswalk helpers, the step-10 map, and step 10's
percentile → APY translation.
"""
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.scrapers.nfl_data import (
    build_bridge_gsis_crosswalk,
    build_value_apy_map,
    compute_starter_share,
    compute_value_components,
    compute_value_target,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def bridge():
    """Five 2018 draftees (one per role below) and one 2010 draftee whose
    usage seasons predate snap data."""
    return pd.DataFrame({
        "pfr_id":         ["early", "star", "bust", "kick", "nogsis", "old"],
        "player_name":    ["Early Guy", "Star Guy", "Bust Guy", "Kick Guy", "No Gsis", "Old Guy"],
        "draft_year":     [2018, 2018, 2018, 2018, 2010, 2010],
        "position_group": ["WR", "WR", "WR", "K", "WR", "WR"],
    })


@pytest.fixture
def contracts():
    """
      early: rookie 2018, a vet-min deal in 2019, then a real deal in 2022
             — the legacy label would take the 2019 deal; the value
             component must take the 2022 one.
      star:  1st-rounder; 5th-year option 2022 (excluded), extension 2021,
             plus a 2023 deal outside the y5 window (excluded).
      bust:  rookie deal only.
      kick:  one modest deal.
    """
    return pd.DataFrame({
        "pfr_id":      ["early", "early", "early", "star", "star", "star", "star", "bust", "kick"],
        "year_signed": [2018,    2019,    2022,    2018,   2021,   2022,   2023,   2018,   2020],
        "years":       [4,       1,       3,       4,      4,      1,      5,      4,      2],
        "apy_cap_pct": [0.004,   0.004,   0.030,   0.020,  0.080,  0.100,  0.150,  0.003,  0.010],
        "apy":         [0.8,     0.8,     6.0,     4.0,    16.0,   20.0,   30.0,   0.6,    2.0],
        "guaranteed":  [0.1,     0.0,     4.0,     8.0,    40.0,   20.0,   90.0,   0.1,    1.0],
        "draft_round": [3,       3,       3,       1,      1,      1,      1,      6,      5],
    })


@pytest.fixture
def snaps():
    return pd.DataFrame({
        "pfr_player_id": ["early", "early", "star", "star", "kick"],
        "season":        [2020,    2021,    2020,   2021,   2020],
        "offense_pct":   [0.40,    0.60,    0.90,   0.95,   0.00],
        "defense_pct":   [0.00,    0.00,    0.00,   0.00,   0.00],
    })


@pytest.fixture
def starter_share():
    # "old" (gsis G-OLD) started all of 2012 and half of 2013.
    return pd.Series({("G-OLD", 2012): 1.0, ("G-OLD", 2013): 0.5})


def _components(bridge, contracts, snaps, starter_share):
    return compute_value_components(
        bridge, contracts, snaps, starter_share, {"old": "G-OLD"},
        last_year_offset=4, usage_offsets=(2, 3), snap_first_draft_year=2014)


# ── Components ────────────────────────────────────────────────────────────────

def test_best_deal_not_earliest(bridge, contracts, snaps, starter_share):
    c = _components(bridge, contracts, snaps, starter_share)
    assert c.at["early", "best_apy_by_y5"] == pytest.approx(0.030)


def test_excludes_fifth_year_option_and_out_of_window(bridge, contracts, snaps, starter_share):
    c = _components(bridge, contracts, snaps, starter_share)
    # 0.100 is the 5th-year option, 0.150 is signed 2023 = draft_year + 5.
    assert c.at["star", "best_apy_by_y5"] == pytest.approx(0.080)


def test_rookie_only_is_zero(bridge, contracts, snaps, starter_share):
    c = _components(bridge, contracts, snaps, starter_share)
    assert c.at["bust", "best_apy_by_y5"] == 0.0
    assert c.at["bust", "guar_by_y5"] == 0.0


def test_guarantee_is_cap_share(bridge, contracts, snaps, starter_share):
    c = _components(bridge, contracts, snaps, starter_share)
    # 40M guaranteed on a 16M-APY deal worth 8% of cap → cap = 200M → 0.20
    assert c.at["star", "guar_by_y5"] == pytest.approx(0.20)


def test_snap_usage_averages_seasons_three_and_four(bridge, contracts, snaps, starter_share):
    c = _components(bridge, contracts, snaps, starter_share)
    assert c.at["early", "usage_y34"] == pytest.approx(0.50)
    assert c.at["early", "usage_source"] == "snaps"
    assert c.at["bust", "usage_y34"] == 0.0   # no snap rows → 0, not NaN


def test_snap_usage_counts_missed_games_as_zero(bridge, contracts, starter_share):
    """offense_pct is a mean over games appeared: 0.90 over 4 of 16 games in
    2020 is a 0.225 season; 0.60 over 17 of 17 in 2021 stays 0.60."""
    snaps = pd.DataFrame({
        "pfr_player_id": ["early", "early"], "season": [2020, 2021],
        "offense_pct": [0.90, 0.60], "defense_pct": [0.0, 0.0], "n_games": [4, 17]})
    c = _components(bridge, contracts, snaps, starter_share)
    assert c.at["early", "usage_y34"] == pytest.approx((0.225 + 0.60) / 2)


def test_snap_usage_share_capped_at_one(bridge, contracts, starter_share):
    # a mid-season trade can mean more appearances than the team's games
    snaps = pd.DataFrame({
        "pfr_player_id": ["early", "early"], "season": [2020, 2021],
        "offense_pct": [1.0, 1.0], "defense_pct": [0.0, 0.0], "n_games": [17, 18]})
    c = _components(bridge, contracts, snaps, starter_share)
    assert c.at["early", "usage_y34"] == pytest.approx(1.0)


def test_pre_snap_class_uses_depth_chart(bridge, contracts, snaps, starter_share):
    c = _components(bridge, contracts, snaps, starter_share)
    assert c.at["old", "usage_y34"] == pytest.approx(0.75)
    assert c.at["old", "usage_source"] == "depth_chart"


def test_pre_snap_without_gsis_is_nan(bridge, contracts, snaps, starter_share):
    c = _components(bridge, contracts, snaps, starter_share)
    assert np.isnan(c.at["nogsis", "usage_y34"])
    assert c.at["nogsis", "usage_source"] is None


def test_kickers_have_no_usage(bridge, contracts, snaps, starter_share):
    c = _components(bridge, contracts, snaps, starter_share)
    assert np.isnan(c.at["kick", "usage_y34"])


# ── Composite ─────────────────────────────────────────────────────────────────

def _target(c, bridge):
    b = bridge.set_index("pfr_id")
    return compute_value_target(c, b["position_group"], b["draft_year"])


def test_composite_in_unit_interval(bridge, contracts, snaps, starter_share):
    v = _target(_components(bridge, contracts, snaps, starter_share), bridge)
    assert v.dropna().between(0, 1).all()


def test_total_washout_pinned_to_zero(bridge, contracts, snaps, starter_share):
    v = _target(_components(bridge, contracts, snaps, starter_share), bridge)
    assert v["bust"] == 0.0


def test_composite_orders_within_group(bridge, contracts, snaps, starter_share):
    v = _target(_components(bridge, contracts, snaps, starter_share), bridge)
    assert v["star"] > v["early"] > v["bust"]
    assert v["star"] == pytest.approx(1.0)


def test_missing_component_drops_out_of_mean(bridge, contracts, snaps, starter_share):
    # The kicker is alone in its (K, 2018) group with one positive deal, so
    # both contract percentiles are 1.0 and the NaN usage is ignored.
    v = _target(_components(bridge, contracts, snaps, starter_share), bridge)
    assert v["kick"] == pytest.approx(1.0)


def test_percentiles_are_per_group_and_class():
    comps = pd.DataFrame({"best_apy_by_y5": [0.01, 0.02, 0.01, 0.02],
                          "guar_by_y5":     [0.01, 0.02, 0.01, 0.02],
                          "usage_y34":      [0.10, 0.20, 0.10, 0.20]},
                         index=["a", "b", "c", "d"])
    group = pd.Series({"a": "WR", "b": "WR", "c": "WR", "d": "WR"})
    year = pd.Series({"a": 2015, "b": 2015, "c": 2016, "d": 2016})
    v = compute_value_target(comps, group, year)
    # same within-class standing → same value, regardless of class
    assert v["a"] == pytest.approx(v["c"])
    assert v["b"] == pytest.approx(v["d"]) == pytest.approx(1.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def test_starter_share_counts_first_team_reg_weeks():
    weeks = list(range(1, 5))
    dc = pd.DataFrame({
        "gsis_id":    ["g1"] * 4 + ["g2"] * 4 + ["g1"],
        "season":     [2010] * 9,
        "week":       weeks + weeks + [19],
        "game_type":  ["REG"] * 8 + ["DIV"],
        "formation":  ["Offense"] * 8 + ["Offense"],
        "depth_team": ["1", "1", "2", "1", "2", "2", "2", "2", "1"],
    })
    share = compute_starter_share(dc)
    assert share[("g1", 2010)] == pytest.approx(0.75)   # playoff week ignored
    assert ("g2", 2010) not in share.index               # never first-team


def test_crosswalk_fills_by_unique_name_and_entry_year():
    bridge = pd.DataFrame({"pfr_id": ["p1", "p2", "p3"],
                           "player_name": ["Ann Lee", "Bo Dee", "Cy Fox"],
                           "draft_year": [2008, 2009, 2009]})
    rosters = pd.DataFrame({
        "season":      [2008, 2009, 2010, 2010],
        "player_id":   ["G1", "G2", "G3", "G4"],
        "pfr_id":      [None, None, None, None],
        "player_name": ["Ann Lee", "Bo Dee", "Cy Fox", "Cy Fox"],
        "entry_year":  [2008, 2010, 2009, 2009],
    })
    x = build_bridge_gsis_crosswalk(bridge, rosters, base={})
    assert x["p1"] == "G1"           # exact name + year
    assert x["p2"] == "G2"           # entry year off by one
    assert "p3" not in x             # two gsis ids share the key → ambiguous


def test_apy_map_is_monotone_quantile_function():
    best = pd.Series([0.0, 0.0, 0.01, 0.05, 0.10], index=list("abcde"))
    group = pd.Series(["WR"] * 5, index=list("abcde"))
    m = build_value_apy_map(best, group, n_grid=11)
    assert m["apy_cap_pct"].is_monotonic_increasing
    assert m["apy_cap_pct"].iloc[0] == 0.0
    assert m["apy_cap_pct"].iloc[-1] == pytest.approx(0.10)


# ── Step 10 translation ───────────────────────────────────────────────────────

def _load_step10():
    path = Path(__file__).parent.parent / "pipeline" / "10_output.py"
    spec = importlib.util.spec_from_file_location("step10_output", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def test_step10_maps_percentiles_to_apy(tmp_path, monkeypatch):
    step10 = _load_step10()
    amap = pd.DataFrame({"position_group": ["WR"] * 3 + ["QB"] * 3,
                         "q": [0.0, 0.5, 1.0] * 2,
                         "apy_cap_pct": [0.0, 0.01, 0.05, 0.0, 0.02, 0.20]})
    amap.to_parquet(tmp_path / "value_apy_map.parquet")
    monkeypatch.setattr(step10, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(step10, "IMPACT_TARGET_SOURCE", "value")
    snap = pd.DataFrame({"position_group": ["WR", "WR", "QB", "EDGE"],
                         "impact_apy_score": [0.5, 1.0, 0.5, 0.5],
                         "impact_apy_p10":   [0.0, 0.5, 0.0, 0.0],
                         "impact_apy_p90":   [0.9, 1.0, 0.9, 0.9]})
    out = step10._convert_value_scores_to_apy(snap)
    assert out["impact_apy_score"].tolist()[:3] == pytest.approx([0.01, 0.05, 0.02])
    assert out["value_pct_score"].tolist() == [0.5, 1.0, 0.5, 0.5]
    assert np.isnan(out.loc[3, "impact_apy_score"])     # group missing from map
    assert (out["impact_score"].iloc[:3] == out["impact_apy_score"].iloc[:3]).all()
    assert (out["impact_apy_p10"] <= out["impact_apy_score"]).iloc[:3].all()
    assert out.loc[0, "impact_apy_dollars"] == pytest.approx(0.01 * step10.SALARY_CAP)


def test_step10_noop_in_apy_mode(monkeypatch):
    step10 = _load_step10()
    monkeypatch.setattr(step10, "IMPACT_TARGET_SOURCE", "apy")
    snap = pd.DataFrame({"position_group": ["WR"], "impact_apy_score": [0.5]})
    out = step10._convert_value_scores_to_apy(snap.copy())
    assert out.equals(snap)
