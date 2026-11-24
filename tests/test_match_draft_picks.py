"""match_draft_picks: (draft year, overall pick) beats name matching."""

import pandas as pd

from src.scrapers.nfl_data import _normalize_name, _normalize_school, match_draft_picks


def _nfl(rows):
    d = pd.DataFrame(rows, columns=["season", "pick", "pfr_player_id", "pfr_player_name", "college", "w_av", "games"])
    d["_name_norm"] = d["pfr_player_name"].apply(_normalize_name)
    d["_school_norm"] = d["college"].apply(_normalize_school)
    return d


def test_overall_pick_separates_namesakes():
    nfl = _nfl([(2012, 2, "GrifRo01", "Robert Griffin III", "Baylor", 36, 56),
                (2012, 203, "GrifRo02", "Robert Griffin", "Baylor", 1, 3)])
    cfbd = pd.DataFrame({"player_name": ["Robert Griffin III", "Robert Griffin"],
                         "college_team": ["Baylor", "Baylor"],
                         "draft_year": [2012, 2012], "overall_pick": [2, 203]})
    out = match_draft_picks(cfbd, nfl)
    assert out["pfr_id"].tolist() == ["GrifRo01", "GrifRo02"]
    assert set(out["match_quality"]) == {"draft_pick"}


def test_name_matching_when_pick_missing():
    nfl = _nfl([(2016, 47, "ThomMi05", "Michael Thomas", "Ohio St.", 60, 83)])
    cfbd = pd.DataFrame({"player_name": ["Michael Thomas"], "college_team": ["Ohio State"],
                         "draft_year": [2016], "overall_pick": [None]})
    out = match_draft_picks(cfbd, nfl)
    assert out["pfr_id"].tolist() == ["ThomMi05"]
    assert out["match_quality"].tolist() != ["draft_pick"]
