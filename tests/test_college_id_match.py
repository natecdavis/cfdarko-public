"""Tests for src/utils/college_id_match.py (combine-join leak fix, 2026-10-30)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.college_id_match import CollegeIdResolver, _norm, _norm_school


def _resolver(rows):
    """Resolver over an in-memory pool of (name, school, player_id, min, max)."""
    r = CollegeIdResolver.__new__(CollegeIdResolver)
    r.by_name_school, r.by_name = {}, {}
    for name, school, pid, mn, mx in rows:
        n, s = _norm(name), _norm_school(school)
        r.by_name_school.setdefault((n, s), []).append((pid, mn, mx))
        r.by_name.setdefault(n, []).append((pid, s, mn, mx))
    return r


def test_school_normalization_matches_nflverse_spellings():
    assert _norm_school("Arizona St.") == _norm_school("Arizona State")
    assert _norm_school("Miami (FL)") == "miami"
    assert _norm_school("St. John's") == "st johns"      # leading "St." is not "State"


def test_name_suffixes_and_accents():
    assert _norm("Joe Milton III") == _norm("Joe Milton")
    assert _norm("José Núñez") == "jose nunez"


def test_school_pass_uses_year_proximity_between_namesakes():
    r = _resolver([("Will Smith", "Ohio State", "old", 2000, 2003),
                   ("Will Smith", "Ohio State", "new", 2021, 2024)])
    assert r.resolve_one("Will Smith", "Ohio St.", 2024) == ("new", "school")
    assert r.resolve_one("Will Smith", "Ohio St.", 2012) == (None, "none")


def test_name_only_pass_requires_a_unique_candidate():
    r = _resolver([("Zion Johnson", "Boston College", "a", 2017, 2021),
                   ("Zion Johnson", "Cincinnati", "b", 2018, 2021)])
    assert r.resolve_one("Zion Johnson", "Unknown U", 2021) == (None, "none")
    r = _resolver([("Zion Johnson", "Boston College", "a", 2017, 2021)])
    assert r.resolve_one("Zion Johnson", "Unknown U", 2021) == ("a", "name_unique")


def test_record_without_final_season_is_unmatched():
    r = _resolver([("A B", "X", "a", 2020, 2021)])
    assert r.resolve_one("A B", "X", float("nan")) == (None, "none")
