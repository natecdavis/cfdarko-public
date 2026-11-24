"""Split CFBD identities: positive-id fragments and the board tie-break."""
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from src.utils.cfbd_id_reuse import detect_positive_twins  # noqa: E402


def _write(raw: Path, rosters, seasons, box=()):
    cols = ["player_id", "player_name", "school", "season_year"]
    pd.DataFrame(rosters, columns=cols + ["position"]).to_parquet(raw / "rosters.parquet")
    pd.DataFrame(seasons, columns=cols).to_parquet(raw / "player_seasons.parquet")
    pd.DataFrame(list(box), columns=cols).to_parquet(raw / "player_gamelogs_raw.parquet")


def test_statless_roster_copy_is_a_fragment(tmp_path):
    _write(tmp_path,
           rosters=[(4820584, "Robert Spears-Jennings", "Oklahoma", y, "S") for y in (2022, 2024, 2025)]
           + [(5244828, "Robert Spearsjennings", "Oklahoma", 2024, None)],
           seasons=[(4820584, "Robert Spears-Jennings", "Oklahoma", y) for y in (2022, 2023, 2024, 2025)])
    assert detect_positive_twins(tmp_path) == {"5244828": [2024]}


def test_stats_feed_spelling_still_links(tmp_path):
    # The real record's stats are under "Felix Anudike"; its roster rows carry
    # the full name the fragment shares.
    _write(tmp_path,
           rosters=[(4612387, "Felix Anudike-Uzomah", "Kansas State", y, "DL") for y in (2020, 2021, 2022)]
           + [(4890031, "Felix Anudike-Uzomah", "Kansas State", 2021, None)],
           seasons=[(4612387, "Felix Anudike", "Kansas State", y) for y in (2020, 2021, 2022)])
    assert set(detect_positive_twins(tmp_path)) == {"4890031"}


def test_records_with_stats_are_never_fragments(tmp_path):
    # Akeem Davis-Gaither: both ids have stats → neither is dropped.
    _write(tmp_path,
           rosters=[(i, "Akeem Davis-Gaither", "App State", y, "LB")
                    for i in (3917142, 4283230) for y in (2017, 2018, 2019)],
           seasons=[(3917142, "Akeem Davis", "App State", y) for y in (2016, 2017, 2018, 2019)]
           + [(4283230, "Akeem Davis-Gaither", "App State", 2017)])
    assert detect_positive_twins(tmp_path) == {}


def test_season_outside_the_real_record_is_kept(tmp_path):
    # A stat-less id with a season the stat-bearing id never had may be a
    # different player (or real history): not contained → not a fragment.
    _write(tmp_path,
           rosters=[(1, "John Williams", "Utah", 2012, "WR"), (2, "John Williams", "Utah", 2022, "WR")],
           seasons=[(2, "John Williams", "Utah", y) for y in (2022, 2023)])
    assert detect_positive_twins(tmp_path) == {}


def test_opposite_side_of_the_ball_is_kept(tmp_path):
    _write(tmp_path,
           rosters=[(1, "Chris Smith", "Ohio", 2020, "QB"), (2, "Chris Smith", "Ohio", 2020, "CB")],
           seasons=[(1, "Chris Smith", "Ohio", 2020)])
    assert detect_positive_twins(tmp_path) == {}


def test_board_tie_break():
    e = importlib.import_module("00e_hasan_ingestion")
    n_stat = {"4820584": 4, "5244828": 0, "a": 1, "b": 3}
    n_ros = {"4820584": 3, "5244828": 1}
    # Same school: the +1 record with stats beats the +2 sat-out copy
    cands = [("4820584", "oklahoma", 1), ("5244828", "oklahoma", 2)]
    assert e._pick_candidate(cands[::-1], "oklahoma", n_stat, n_ros) == "4820584"
    # Same tier: more stat seasons wins, whatever the order
    assert e._pick_candidate([("a", "x", 1), ("b", "x", 1)], "x", n_stat, {}) == "b"
    # The NFL-linked (bridge) record beats an attrs-only one at the same school
    assert e._pick_candidate([("b", "x", 1), ("a", "x", 0)], "x", n_stat, {}) == "a"
    # School match beats everything
    assert e._pick_candidate([("b", "y", 0), ("a", "x", 2)], "x", n_stat, {}) == "a"
