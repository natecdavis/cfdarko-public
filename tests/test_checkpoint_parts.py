"""
Unit tests for the part-file checkpoint scheme used by the long-running
scrapers (pipeline/01b_fetch_supplemental.py, pipeline/01d_play_ingestion.py)
and the shared helpers in src/utils/parquet_checkpoint.py.

All fetches are faked — no network calls.

Covers:
  - helper write/read/consolidate roundtrip
  - 01b fresh run (chunked part files, final consolidation, no parts left)
  - 01b resume-key INTERSECTION (game with player rows written but team rows
    missing must be re-fetched, without duplicating the player rows)
  - 01b all-done path still consolidates leftover parts
  - 01d resume after a simulated crash mid-run (parts present, no main file)
  - 01d play-stats resume from part files + final consolidation
"""
import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.utils.parquet_checkpoint import (
    parts_dir, write_part, read_with_parts, consolidate,
)


def _load_pipeline_module(name: str, filename: str):
    mod_path = Path(__file__).parent.parent / "pipeline" / filename
    spec = importlib.util.spec_from_file_location(name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── helper roundtrip ──────────────────────────────────────────────────────────

def test_write_read_consolidate_roundtrip(tmp_path):
    path = tmp_path / "store.parquet"

    # Pre-existing main parquet
    pd.DataFrame({"a": [1, 2]}).to_parquet(path, index=False)

    # Two checkpoint parts
    assert write_part(path, pd.DataFrame({"a": [3]})) is not None
    assert write_part(path, pd.DataFrame({"a": [4, 5]})) is not None
    # Empty chunk writes nothing
    assert write_part(path, pd.DataFrame()) is None

    d = parts_dir(path)
    assert sorted(p.name for p in d.glob("*.parquet")) == \
        ["part_00000.parquet", "part_00001.parquet"]

    # Resume reads see main + all parts
    combined = read_with_parts(path)
    assert sorted(combined["a"].tolist()) == [1, 2, 3, 4, 5]
    assert read_with_parts(path, columns=["a"]).columns.tolist() == ["a"]

    # Consolidation folds everything into the main file and removes parts
    consolidate(path)
    assert not d.exists()
    final = pd.read_parquet(path)
    assert sorted(final["a"].tolist()) == [1, 2, 3, 4, 5]

    # Consolidate is a no-op when no parts exist
    consolidate(path)
    assert sorted(pd.read_parquet(path)["a"].tolist()) == [1, 2, 3, 4, 5]


def test_read_with_parts_missing_everything(tmp_path):
    assert read_with_parts(tmp_path / "nope.parquet").empty


# ── 01b box scores ────────────────────────────────────────────────────────────

def _setup_01b(tmp_path, game_ids, fetch_calls):
    mod = _load_pipeline_module("fetch_supplemental_01b_test",
                                "01b_fetch_supplemental.py")
    mod.RAW_DIR = tmp_path
    pd.DataFrame({
        "game_id": game_ids,
        "season_year": [2022] * len(game_ids),
    }).to_parquet(tmp_path / "game_schedule.parquet", index=False)

    def fake_fetch(chunk):
        fetch_calls.append(sorted(int(g) for g in chunk))
        players = pd.DataFrame({
            "game_id": np.repeat(chunk, 2).astype(int),
            "player_id": [f"P{g}_{i}" for g in chunk for i in range(2)],
            "snaps": 50.0,
        })
        teams = pd.DataFrame({
            "game_id": np.asarray(chunk, dtype=int),
            "team": [f"T{g}" for g in chunk],
            "havoc": 0.1,
        })
        return players, teams

    mod.fetch_advanced_box_scores = fake_fetch
    return mod


def test_01b_fresh_run_consolidates(tmp_path):
    calls = []
    mod = _setup_01b(tmp_path, [1, 2, 3, 4, 5, 6, 7], calls)
    mod.CHECKPOINT_SIZE = 3

    mod.run_box_scores(2022, 2022)

    # 3 chunks of <=3 games
    assert len(calls) == 3
    player = pd.read_parquet(tmp_path / "player_box_scores.parquet")
    team = pd.read_parquet(tmp_path / "team_box_scores.parquet")
    assert sorted(player["game_id"].unique()) == [1, 2, 3, 4, 5, 6, 7]
    assert sorted(team["game_id"].unique()) == [1, 2, 3, 4, 5, 6, 7]
    assert len(player) == 14 and len(team) == 7
    # Part directories cleaned up
    assert not parts_dir(tmp_path / "player_box_scores.parquet").exists()
    assert not parts_dir(tmp_path / "team_box_scores.parquet").exists()


def test_01b_resume_intersection_refetches_half_written_game(tmp_path):
    """Crash between the player-chunk write and team-chunk write: game 3 has
    player rows (in a part file) but no team rows. The done-set must be the
    INTERSECTION of the two stores, so game 3 is re-fetched — and its player
    rows must not be duplicated."""
    calls = []
    mod = _setup_01b(tmp_path, [1, 2, 3, 4], calls)

    # Pre-crash state: games 1, 2 fully consolidated in both stores;
    # game 3 player rows written as a part file, team rows lost in the crash.
    pd.DataFrame({
        "game_id": [1, 1, 2, 2],
        "player_id": ["P1_0", "P1_1", "P2_0", "P2_1"],
        "snaps": 50.0,
    }).to_parquet(tmp_path / "player_box_scores.parquet", index=False)
    pd.DataFrame({
        "game_id": [1, 2], "team": ["T1", "T2"], "havoc": 0.1,
    }).to_parquet(tmp_path / "team_box_scores.parquet", index=False)
    write_part(tmp_path / "player_box_scores.parquet", pd.DataFrame({
        "game_id": [3, 3], "player_id": ["P3_0", "P3_1"], "snaps": 50.0,
    }))

    mod.run_box_scores(2022, 2022)

    # Only games 3 and 4 re-fetched (1, 2 are done in BOTH stores)
    assert calls == [[3, 4]]

    player = pd.read_parquet(tmp_path / "player_box_scores.parquet")
    team = pd.read_parquet(tmp_path / "team_box_scores.parquet")
    # No duplicated player rows for the re-fetched game 3
    counts = player.groupby("game_id").size()
    assert counts.to_dict() == {1: 2, 2: 2, 3: 2, 4: 2}
    assert sorted(team["game_id"].tolist()) == [1, 2, 3, 4]
    assert not parts_dir(tmp_path / "player_box_scores.parquet").exists()
    assert not parts_dir(tmp_path / "team_box_scores.parquet").exists()


def test_01b_all_done_consolidates_leftover_parts(tmp_path):
    """Everything already fetched, but game 2 still sits in part files from an
    interrupted run — the early-return path must consolidate them."""
    calls = []
    mod = _setup_01b(tmp_path, [1, 2], calls)

    pd.DataFrame({"game_id": [1, 1], "player_id": ["P1_0", "P1_1"],
                  "snaps": 50.0}).to_parquet(
        tmp_path / "player_box_scores.parquet", index=False)
    pd.DataFrame({"game_id": [1], "team": ["T1"], "havoc": 0.1}).to_parquet(
        tmp_path / "team_box_scores.parquet", index=False)
    write_part(tmp_path / "player_box_scores.parquet", pd.DataFrame({
        "game_id": [2, 2], "player_id": ["P2_0", "P2_1"], "snaps": 50.0}))
    write_part(tmp_path / "team_box_scores.parquet", pd.DataFrame({
        "game_id": [2], "team": ["T2"], "havoc": 0.1}))

    mod.run_box_scores(2022, 2022)

    assert calls == []  # nothing re-fetched
    player = pd.read_parquet(tmp_path / "player_box_scores.parquet")
    team = pd.read_parquet(tmp_path / "team_box_scores.parquet")
    assert sorted(player["game_id"].unique()) == [1, 2]
    assert sorted(team["game_id"].unique()) == [1, 2]
    assert not parts_dir(tmp_path / "player_box_scores.parquet").exists()
    assert not parts_dir(tmp_path / "team_box_scores.parquet").exists()


# ── 01d plays / play-stats ────────────────────────────────────────────────────

def _setup_01d(tmp_path):
    mod = _load_pipeline_module("play_ingestion_01d_test",
                                "01d_play_ingestion.py")
    mod.RAW_DIR = tmp_path
    mod.PLAYS_PATH = tmp_path / "plays.parquet"
    mod.PLAY_STATS_PATH = tmp_path / "play_stats.parquet"
    mod.SCHEDULE_PATH = tmp_path / "game_schedule.parquet"
    return mod


def _fake_plays(year, week, n=3):
    return pd.DataFrame({
        "play_id": [f"{year}_{week}_{i}" for i in range(n)],
        "season_year": year,
        "week": week,
        "yards_gained": 5,
    })


def test_01d_plays_crash_and_resume(tmp_path):
    """Simulated crash mid-run leaves part files but no main parquet; the
    resumed run must skip already-saved weeks and consolidate everything."""
    mod = _setup_01d(tmp_path)
    mod.MAX_WEEK = 3
    mod.PLAYS_CHECKPOINT_WEEKS = 1  # one part file per (year, week)

    calls = []

    def crashing_fetch(year, week):
        calls.append((year, week))
        if week == 2:
            raise RuntimeError("simulated quota crash")
        return _fake_plays(year, week)

    mod.fetch_plays = crashing_fetch
    with pytest.raises(RuntimeError):
        mod.run_plays(2021, 2021, test=False)

    # Crash state: week 1 saved as a part, no main parquet yet
    assert calls == [(2021, 1), (2021, 2)]
    assert not mod.PLAYS_PATH.exists()
    assert len(list(parts_dir(mod.PLAYS_PATH).glob("*.parquet"))) == 1

    # Resume: week 1 must be seen as done from its part file
    calls.clear()
    mod.fetch_plays = lambda year, week: (calls.append((year, week)) or
                                          _fake_plays(year, week))
    mod.run_plays(2021, 2021, test=False)

    assert calls == [(2021, 2), (2021, 3)]
    final = pd.read_parquet(mod.PLAYS_PATH)
    assert sorted(final["week"].unique()) == [1, 2, 3]
    assert len(final) == 9  # 3 plays per week, no duplicates
    assert not parts_dir(mod.PLAYS_PATH).exists()

    # Idempotent re-run: nothing fetched, nothing changed
    calls.clear()
    mod.run_plays(2021, 2021, test=False)
    assert calls == []
    assert len(pd.read_parquet(mod.PLAYS_PATH)) == 9


def test_01d_play_stats_resume_from_parts(tmp_path):
    mod = _setup_01d(tmp_path)
    mod.MAX_WEEK = 1
    mod.PLAY_STATS_CHECKPOINT_TEAM_WEEKS = 1

    pd.DataFrame({
        "season_year": [2021], "week": [1],
        "home_team": ["A"], "away_team": ["B"],
        "game_id": [10],
    }).to_parquet(mod.SCHEDULE_PATH, index=False)

    # Team A's stats already written as a part file (interrupted run)
    write_part(mod.PLAY_STATS_PATH, pd.DataFrame({
        "play_id": ["p1"], "season_year": [2021], "week": [1],
        "team": ["A"], "stat": [1.0],
    }))

    calls = []

    def fake_stats(year, week, team=None):
        calls.append((year, week, team))
        return pd.DataFrame({
            "play_id": [f"p_{team}"], "season_year": [year],
            "week": [week], "team": [team], "stat": [2.0],
        })

    mod.fetch_play_stats = fake_stats
    mod.run_play_stats(2021, 2021, test=False)

    assert calls == [(2021, 1, "B")]  # team A resumed from its part file
    final = pd.read_parquet(mod.PLAY_STATS_PATH)
    assert sorted(final["team"].tolist()) == ["A", "B"]
    assert not parts_dir(mod.PLAY_STATS_PATH).exists()
