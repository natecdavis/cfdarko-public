"""Tests for src/utils/cfbd_id_reuse.py negative-twin detection and masking."""

import pandas as pd

from src.utils.cfbd_id_reuse import detect_negative_twins, exclusion_mask


def _rosters(tmp_path, rows):
    pd.DataFrame(rows, columns=["player_id", "player_name", "school", "season_year"]) \
      .to_parquet(tmp_path / "rosters.parquet", index=False)


def test_negative_twin_detected(tmp_path):
    _rosters(tmp_path, [
        ("4241984", "Jeff Okudah", "Ohio State", 2017),
        ("4241984", "Jeff Okudah", "Ohio State", 2019),
        ("-1027003", "Jeff Okudah", "Ohio State", 2019),
    ])
    assert detect_negative_twins(tmp_path) == {"-1027003": [2019]}


def test_adjacent_seasons_and_suffix_count_as_twin(tmp_path):
    _rosters(tmp_path, [
        ("100", "Mario Edwards Jr.", "Florida State", 2012),
        ("100", "Mario Edwards Jr.", "Florida State", 2013),
        ("-5", "Mario Edwards", "Florida State", 2014),
    ])
    assert detect_negative_twins(tmp_path) == {"-5": [2014]}


def test_lone_or_distant_negative_ids_kept(tmp_path):
    _rosters(tmp_path, [
        ("-1", "Walk On", "Iowa", 2015),                # no real twin
        ("200", "John Johnson", "Boston College", 2006),
        ("-2", "John Johnson", "Boston College", 2015),  # namesake, years apart
        ("300", "Sam Smith", "Utah", 2015),
        ("-3", "Sam Smith", "Utah State", 2015),         # different school
    ])
    assert detect_negative_twins(tmp_path) == {}


def test_exclusion_mask_matches_id_and_year():
    df = pd.DataFrame({"player_id": [1.0, 1.0, 2.0, "-7"],
                       "season_year": [2019, 2020, 2019, 2014]})
    df["player_id"] = df["player_id"].map(lambda v: v if isinstance(v, str) else str(int(v)))
    mask = exclusion_mask(df, {"1": [2020], "-7": [2014]})
    assert mask.tolist() == [False, True, False, True]
    assert not exclusion_mask(df, {}).any()


def test_post_draft_seasons_detected(tmp_path):
    from src.utils.cfbd_id_reuse import detect_post_draft_seasons
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir(); proc.mkdir()
    pd.DataFrame([("10", "Pooka Williams", "Kansas", y) for y in (2018, 2019, 2020, 2021)]
                 + [("20", "Someone Else", "Kansas", 2021)],
                 columns=["player_id", "player_name", "school", "season_year"]) \
      .to_parquet(raw / "rosters.parquet", index=False)
    pd.DataFrame({"player_id": ["10", None], "draft_year": [2021, 2021]}) \
      .to_parquet(proc / "bridge_players.parquet", index=False)
    assert detect_post_draft_seasons(raw, proc) == {"10": [2021]}
