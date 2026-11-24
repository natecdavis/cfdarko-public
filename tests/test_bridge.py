"""
Unit tests for pipeline/02_bridge_players.py helper functions.

Tests name normalization (_norm) and position group mapping (_add_position_group).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import types
import importlib.util
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

import config as C

# Load the bridge module, mocking external nfl_data imports
_bridge_mod = None

def _get_bridge():
    global _bridge_mod
    if _bridge_mod is not None:
        return _bridge_mod

    mod_path = Path(__file__).parent.parent / "pipeline" / "02_bridge_players.py"
    spec = importlib.util.spec_from_file_location("bridge_players", mod_path)
    _bridge_mod = types.ModuleType("bridge_players")
    _bridge_mod.__file__ = str(mod_path)  # required so Path(__file__) inside the module works

    # Stub out nfl_data imports so we don't hit the network
    nfl_stub = types.ModuleType("src.scrapers.nfl_data")
    nfl_stub.load_nfl_draft_picks = lambda *a, **kw: pd.DataFrame()
    nfl_stub.match_draft_picks = lambda *a, **kw: pd.DataFrame()
    nfl_stub.load_udfa_players = lambda *a, **kw: pd.DataFrame()
    nfl_stub.match_udfa_to_cfbd = lambda *a, **kw: pd.DataFrame()
    nfl_stub.fetch_seasonal_snap_pcts = lambda *a, **kw: pd.DataFrame()
    nfl_stub._build_id_crosswalk = lambda *a, **kw: pd.DataFrame()
    sys.modules.setdefault("src.scrapers.nfl_data", nfl_stub)

    spec.loader.exec_module(_bridge_mod)
    return _bridge_mod


# ── _norm ─────────────────────────────────────────────────────────────────────

def test_norm_lowercase():
    m = _get_bridge()
    assert m._norm("TOM BRADY") == "tom brady"


def test_norm_strips_jr():
    m = _get_bridge()
    assert m._norm("Tom Jones Jr.") == m._norm("Tom Jones")


def test_norm_strips_sr():
    m = _get_bridge()
    assert m._norm("Mike Smith Sr") == m._norm("Mike Smith")


def test_norm_strips_ii():
    m = _get_bridge()
    assert m._norm("John Doe II") == m._norm("John Doe")


def test_norm_strips_iii():
    m = _get_bridge()
    assert m._norm("John Doe III") == m._norm("John Doe")


def test_norm_strips_iv():
    m = _get_bridge()
    assert m._norm("John Doe IV") == m._norm("John Doe")


def test_norm_strips_v():
    m = _get_bridge()
    assert m._norm("John Doe V") == m._norm("John Doe")


def test_norm_accented_chars_removed():
    """Accented characters should be normalized to ASCII equivalents."""
    m = _get_bridge()
    result = m._norm("Álvaro García")
    # Should be ASCII only
    assert result.isascii()
    # And lowercase
    assert result == result.lower()


def test_norm_empty_string():
    m = _get_bridge()
    assert m._norm("") == ""


def test_norm_non_string_returns_empty():
    m = _get_bridge()
    assert m._norm(None) == ""
    assert m._norm(42) == ""


def test_norm_strips_punctuation():
    m = _get_bridge()
    # Non-alphanumeric, non-space chars should be stripped
    result = m._norm("O'Brien")
    assert "'" not in result


def test_norm_idempotent():
    """Applying _norm twice should give the same result as once."""
    m = _get_bridge()
    for name in ["Tom Jones Jr", "Mike Smith III", "Álvaro García"]:
        once = m._norm(name)
        twice = m._norm(once)
        assert once == twice, f"_norm not idempotent for {name!r}"


# ── _add_position_group ────────────────────────────────────────────────────────

def test_add_position_group_known_positions():
    """All positions in POSITION_GROUPS must map correctly."""
    m = _get_bridge()
    for group, positions in C.POSITION_GROUPS.items():
        df = pd.DataFrame({"position": positions})
        result = m._add_position_group(df)
        # All should map to the expected group
        for pos, pg in zip(positions, result["position_group"]):
            assert pg == group, (
                f"Position {pos!r} should map to {group!r}, got {pg!r}"
            )


def test_add_position_group_no_position_column():
    """Missing position column → position_group set to None, no error."""
    m = _get_bridge()
    df = pd.DataFrame({"player_id": [1, 2, 3]})
    result = m._add_position_group(df)
    assert "position_group" in result.columns
    assert result["position_group"].isna().all()


def test_add_position_group_unknown_position_is_nan():
    """Unknown position string → NaN (not a crash)."""
    m = _get_bridge()
    df = pd.DataFrame({"position": ["UNICORN", "QB"]})
    result = m._add_position_group(df)
    assert pd.isna(result.loc[0, "position_group"])
    assert result.loc[1, "position_group"] == "QB"


def test_add_position_group_all_ten_groups_reachable():
    """Each of the 10 position groups should be reachable from POSITION_GROUPS."""
    m = _get_bridge()
    all_positions = []
    for positions in C.POSITION_GROUPS.values():
        all_positions.extend(positions)
    df = pd.DataFrame({"position": all_positions})
    result = m._add_position_group(df)
    mapped_groups = set(result["position_group"].dropna())
    expected_groups = set(C.POSITION_GROUPS.keys())
    assert mapped_groups == expected_groups, (
        f"Missing groups: {expected_groups - mapped_groups}"
    )


# ── bridge resolution helpers ─────────────────────────────────────────────────

def test_school_keys_splits_transfers_and_applies_aliases():
    m = _get_bridge()
    assert m._school_keys("Indiana; Mississippi") == ["indiana", "ole miss"]
    assert m._school_keys("Connecticut") == ["uconn"]
    assert m._school_keys(None) == []


def test_pick_by_year_prefers_real_id_over_negative_fragment():
    m = _get_bridge()
    # Jeff Okudah: a -id record for 2019 only and the real 2017-2019 record
    cands = [("-1027003", 2019, 2019), ("4241984", 2017, 2019)]
    assert m._pick_by_year(cands, 2019) == "4241984"


def test_pick_by_year_still_prefers_closer_years():
    m = _get_bridge()
    assert m._pick_by_year([("-5", 2019, 2019), ("7", 2014, 2016)], 2019) == "-5"
    assert m._pick_by_year([("7", 2010, 2012)], 2019) is None


def test_merge_cands_unions_a_transfer_range():
    m = _get_bridge()
    assert m._merge_cands([("1", 2018, 2020), ("1", 2021, 2021)]) == [("1", 2018, 2021)]


def test_drop_negative_twins_keeps_unrelated_negative_ids():
    m = _get_bridge()
    cands = [("-1", "kentucky", 2018, 2018), ("9", "kentucky", 2016, 2018),
             ("-2", "auburn", 2018, 2018)]
    assert m._drop_negative_twins(cands) == [("9", "kentucky", 2016, 2018),
                                             ("-2", "auburn", 2018, 2018)]


@pytest.mark.parametrize("a,b", [
    ("josh holsey", "joshua holsey"), ("dre hal", "andre hal"),
    ("nate bussey", "nathan bussey"), ("jake hummel", "jacob hummel"),
    ("jp richardson", "john paul richardson"), ("ken walker", "kenneth walker"),
])
def test_name_similar_accepts_nicknames(a, b):
    assert _get_bridge()._name_similar(a, b)


@pytest.mark.parametrize("a,b", [
    ("braxton jones", "easton jones"), ("mario williams", "jarvis williams"),
    ("tay martin", "adam martin"), ("casey bramlet", "corey bramlet"),
    ("jr russell", "jon russell"), ("aj davis", "jay davis"),
])
def test_name_similar_rejects_shared_surname_only(a, b):
    assert not _get_bridge()._name_similar(a, b)
