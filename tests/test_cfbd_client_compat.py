"""
Regression tests for the two cfbd-SDK compatibility failures fixed 2026-07-25.

Both were silent: each logged one per-year error and let the caller save a file
that was simply missing data, so the pipeline kept running on degraded input.

  1. `fetch_team_advanced_stats` computed `passing_plays / plays`, but in cfbd v5
     `offense.passing_plays` is a nested object, not a count → TypeError on every
     year → team_advanced.parquet was never written at all.
  2. `fetch_team_ratings` deserialized through the SDK's TeamSP model, whose
     TeamSPDefense.rating is non-optional. 2004 has 2 teams with a null defense
     rating, so the whole 2004 season raised ValidationError and 2004 team_sp_*
     coverage was lost.

No network: the SDK boundary is stubbed with payloads shaped like the real ones.
"""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

import src.scrapers.cfbd as S


# ── Fixtures shaped like the real API responses ────────────────────────────────

class _FakeNested:
    """Stands in for AdvancedSeasonStatOffensePassingPlays — has `.rate`, and is
    deliberately NOT a number, which is what broke the original division."""
    def __init__(self, rate):
        self.rate = rate


class _FakeOffense:
    def __init__(self, pass_rate, rush_rate, plays):
        self.passing_plays = _FakeNested(pass_rate)
        self.rushing_plays = _FakeNested(rush_rate)
        self.plays = plays
        self.success_rate = 0.45
        self.ppa = 0.26


class _FakeAdvStat:
    def __init__(self, team, season, offense):
        self.team = team
        self.season = season
        self.offense = offense


# Real 2004 shape: `sos`/`secondOrderWins` and all sub-detail are null that far
# back, and 2 of 121 teams have a null defense rating.
_SP_2004 = [
    {"team": "USC", "conference": "Big Ten", "year": 2004, "rating": 32.3,
     "offense": {"rating": 45.2}, "defense": {"rating": 14.3},
     "sos": None, "secondOrderWins": None},
    {"team": "Nullville", "conference": "Big 12", "year": 2004, "rating": 1.0,
     "offense": {"rating": 2.0}, "defense": {"rating": None},
     "sos": None, "secondOrderWins": None},
    {"team": "NoDefenseObj", "conference": None, "year": 2004, "rating": 0.5,
     "offense": None, "defense": None},
]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(S, "_rate_limit", lambda *a, **kw: None)
    monkeypatch.setattr(S, "_get_client", lambda: object())


# ── 1. advanced season stats: pass_rate comes from the nested `.rate` ──────────

def test_pass_rate_read_from_nested_rate_object(monkeypatch):
    """The bug: `passing_plays / plays` on an object raised TypeError and the
    whole fetch returned empty. Guards the attribute path, not just the value."""
    stats = [
        _FakeAdvStat("Air Force", 2023, _FakeOffense(0.1346, 0.8605, 817)),
        _FakeAdvStat("Washington State", 2023, _FakeOffense(0.6280, 0.3630, 843)),
    ]
    fake_api = types.SimpleNamespace(get_advanced_season_stats=lambda **kw: stats)
    monkeypatch.setattr(S.cfbd, "StatsApi", lambda client: fake_api)

    df = S.fetch_team_advanced_stats(2023, 2023)

    assert len(df) == 2
    assert df.loc[df.school == "Air Force", "pass_rate"].iloc[0] == pytest.approx(0.1346)
    assert df.loc[df.school == "Washington State", "pass_rate"].iloc[0] == pytest.approx(0.6280)
    # A run-heavy team must stay run-heavy: catches a pass/rush swap.
    af = df[df.school == "Air Force"].iloc[0]
    assert af.pass_rate < af.rush_rate
    # The count columns the API doesn't actually expose must not reappear.
    assert "passing_plays" not in df.columns
    assert "rushing_plays" not in df.columns


def test_pass_rate_is_a_share_not_a_ratio_of_objects(monkeypatch):
    """pass_rate must be a probability. The old code, had the division worked,
    would have produced a count ratio; a >1 value means someone reintroduced it."""
    stats = [_FakeAdvStat("Alabama", 2023, _FakeOffense(0.4166, 0.5789, 881))]
    fake_api = types.SimpleNamespace(get_advanced_season_stats=lambda **kw: stats)
    monkeypatch.setattr(S.cfbd, "StatsApi", lambda client: fake_api)

    df = S.fetch_team_advanced_stats(2023, 2023)
    row = df.iloc[0]
    assert 0.0 <= row.pass_rate <= 1.0
    assert row.pass_rate + row.rush_rate == pytest.approx(1.0, abs=0.02)


def test_advanced_stats_skips_rows_with_no_offense(monkeypatch):
    stats = [_FakeAdvStat("Ghost", 2023, None),
             _FakeAdvStat("Real", 2023, _FakeOffense(0.5, 0.5, 800))]
    fake_api = types.SimpleNamespace(get_advanced_season_stats=lambda **kw: stats)
    monkeypatch.setattr(S.cfbd, "StatsApi", lambda client: fake_api)

    df = S.fetch_team_advanced_stats(2023, 2023)
    assert list(df.school) == ["Real"]


# ── 2. SP+: a null defense rating must cost one cell, not the season ───────────

def _stub_sp(monkeypatch, payload_by_year: dict):
    """Stub at the SDK's raw-response boundary, the same surface _get_json uses."""
    def _get_sp_with_http_info(**kwargs):
        year = kwargs["year"]
        if year not in payload_by_year:
            raise RuntimeError(f"no payload for {year}")
        body = json.dumps(payload_by_year[year]).encode()
        return types.SimpleNamespace(raw_data=body)

    fake_api = types.SimpleNamespace(get_sp_with_http_info=_get_sp_with_http_info)
    monkeypatch.setattr(S.cfbd, "RatingsApi", lambda client: fake_api)


def test_null_defense_rating_does_not_drop_the_season(monkeypatch):
    """The regression this whole change exists for: 2004 must survive."""
    _stub_sp(monkeypatch, {2004: _SP_2004})

    df = S.fetch_team_ratings(2004, 2004)

    assert len(df) == 3, "a null defense rating must not delete the season"
    assert set(df.season_year) == {2004}
    usc = df[df.school == "USC"].iloc[0]
    assert usc.sp_plus == pytest.approx(32.3)
    assert usc.sp_plus_offense == pytest.approx(45.2)
    assert usc.sp_plus_defense == pytest.approx(14.3)
    # The null lands in exactly one cell.
    assert pd.isna(df[df.school == "Nullville"].iloc[0].sp_plus_defense)
    assert df[df.school == "Nullville"].iloc[0].sp_plus == pytest.approx(1.0)


def test_missing_offense_and_defense_objects_are_tolerated(monkeypatch):
    _stub_sp(monkeypatch, {2004: _SP_2004})
    df = S.fetch_team_ratings(2004, 2004)
    row = df[df.school == "NoDefenseObj"].iloc[0]
    assert pd.isna(row.sp_plus_offense) and pd.isna(row.sp_plus_defense)
    assert row.sp_plus == pytest.approx(0.5)


def test_one_failed_year_does_not_lose_the_others(monkeypatch):
    """A year that throws is reported, but the years that worked still come back."""
    _stub_sp(monkeypatch, {2004: _SP_2004})       # 2005 has no payload → raises
    df = S.fetch_team_ratings(2004, 2005)
    assert set(df.season_year) == {2004}
    assert len(df) == 3


def _messages(caplog, level: str) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelname == level)


def test_failed_year_is_logged_at_error(monkeypatch, caplog):
    """The silent-degradation fix: a lost year must be loud, and must name itself
    in the coverage report rather than only in a per-year line that scrolls past."""
    _stub_sp(monkeypatch, {2004: _SP_2004})
    with caplog.at_level("ERROR"):
        S.fetch_team_ratings(2004, 2005)
    errors = _messages(caplog, "ERROR")
    assert "2005" in errors
    assert "FAILED" in errors


def test_empty_year_warns_but_does_not_error(monkeypatch, caplog):
    """An unplayed season legitimately returns nothing — that must not read as a
    bug, or the real failures get lost in the noise. 2004 has a payload, 2026 is
    empty, and 2005 raises: all three must be reported differently."""
    _stub_sp(monkeypatch, {2004: _SP_2004, 2026: []})
    with caplog.at_level("DEBUG"):
        df = S.fetch_team_ratings(2004, 2026)

    assert set(df.season_year) == {2004}
    assert "2026" in _messages(caplog, "WARNING"), "empty year should warn"

    failed_line = [ln for ln in _messages(caplog, "ERROR").splitlines()
                   if "FAILED" in ln]
    assert failed_line, "years that raised should be reported"
    assert "2026" not in failed_line[0], "an empty year is not a failed year"
    assert "2005" in failed_line[0]


# ── 3. conference affiliation must be per-season, not present-day ──────────────

def test_sp_conference_column_is_named_current(monkeypatch):
    """SP+ reports today's conference on historical rows. The column name has to
    say so, or it gets reused as a per-season affiliation again (which is exactly
    what happened: 03_player_attributes tiered 2004 Utah as Big 12)."""
    _stub_sp(monkeypatch, {2004: _SP_2004})
    df = S.fetch_team_ratings(2004, 2004)
    assert "conference_current" in df.columns
    assert "conference" not in df.columns, (
        "a bare `conference` column invites the anachronism bug back")


def test_fetch_team_conferences_is_per_season(monkeypatch):
    """The replacement source must key conference by (school, season) and return
    the season's affiliation, so a mover gets different values across years."""
    by_year = {
        2004: [("Utah", "Mountain West"), ("USC", "Pac-10"), ("TCU", "Conference USA")],
        2011: [("Utah", "Pac-12"), ("USC", "Pac-12"), ("TCU", "Mountain West")],
    }

    def _get_fbs_teams(**kwargs):
        return [types.SimpleNamespace(school=s, conference=c)
                for s, c in by_year[kwargs["year"]]]

    monkeypatch.setattr(S.cfbd, "TeamsApi",
                        lambda client: types.SimpleNamespace(get_fbs_teams=_get_fbs_teams))

    df = S.fetch_team_conferences(2004, 2011)
    got = {(r.school, r.season_year): r.conference for r in df.itertuples()}
    assert got[("Utah", 2004)] == "Mountain West"
    assert got[("Utah", 2011)] == "Pac-12"
    assert got[("TCU", 2004)] == "Conference USA"
    assert got[("USC", 2004)] == "Pac-10"


def test_historical_conference_names_all_have_a_tier():
    """Pac-10 / Big East / Western Athletic are retired but appear throughout the
    2004-2012 rows. If any lacked a tier it would silently become DEFAULT (FCS),
    turning a BCS program into tier 3."""
    from config import CONFERENCE_TIERS
    for name, expected in [("Pac-10", 1), ("Big East", 1), ("Western Athletic", 2),
                           ("Conference USA", 2), ("Mountain West", 2)]:
        assert CONFERENCE_TIERS.get(name) == expected, f"{name} must map to tier {expected}"


def test_team_conferences_skips_rows_missing_school_or_conference(monkeypatch):
    teams = [types.SimpleNamespace(school="Utah", conference="Mountain West"),
             types.SimpleNamespace(school=None, conference="SEC"),
             types.SimpleNamespace(school="Ghost", conference=None)]
    monkeypatch.setattr(S.cfbd, "TeamsApi",
                        lambda client: types.SimpleNamespace(get_fbs_teams=lambda **kw: teams))
    df = S.fetch_team_conferences(2004, 2004)
    assert list(df.school) == ["Utah"]


# ── _get_json boundary ─────────────────────────────────────────────────────────

def test_get_json_returns_empty_list_on_empty_body(monkeypatch):
    """`.raw_data` is None when the SDK preloaded content; must not explode."""
    method = lambda **kw: types.SimpleNamespace(raw_data=None)
    assert S._get_json(method, "probe", year=2004) == []


def test_get_json_wraps_a_bare_object_in_a_list(monkeypatch):
    method = lambda **kw: types.SimpleNamespace(raw_data=b'{"team": "USC"}')
    assert S._get_json(method, "probe", year=2004) == [{"team": "USC"}]


def test_get_json_requests_the_raw_body(monkeypatch):
    """If _preload_content ever stops being passed, the SDK validates again and
    the 2004 bug comes back — so assert on the call, not just the result."""
    seen = {}

    def method(**kw):
        seen.update(kw)
        return types.SimpleNamespace(raw_data=b"[]")

    S._get_json(method, "probe", year=2004)
    assert seen.get("_preload_content") is False
    assert seen.get("year") == 2004
