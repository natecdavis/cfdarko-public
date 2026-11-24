"""
Tests for audit items #2 and #4 (2026-07-26). Item #3 was investigation-only and
produced no code change — its finding is recorded in the memory note.

#2  _drop_opponent_side_phantom_rows — CFBD credits 5 player_ids to BOTH teams of
    one game, with a statless singleton row on the opponent side.
#4  adjusted_stat_semantics.json — the adjusted_* columns carry two different
    semantics, and this sidecar records which applies per head.
"""
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

import config as C

ROOT = Path(__file__).parent.parent


def _load_step1():
    """Import step 1 with every scraper symbol stubbed, so no network occurs.
    The import list is parsed from the source rather than hardcoded — otherwise
    adding a fetcher silently breaks these tests with an ImportError."""
    src = (ROOT / "pipeline" / "01_data_ingestion.py").read_text()
    block = re.search(r"from src\.scrapers\.cfbd import \(([^)]*)\)", src).group(1)
    names = [n.strip().rstrip(",") for n in block.split("\n")
             if n.strip() and not n.strip().startswith("#")]
    stub = types.ModuleType("src.scrapers.cfbd")
    for n in names:
        setattr(stub, n, lambda *a, **kw: pd.DataFrame())

    # Install the stub only for the duration of the exec, then put back whatever
    # was there. Leaving it in sys.modules leaks into every later test module —
    # test_play_ingestion imports the REAL scraper and fails with ImportError if
    # it finds this stub instead. (test_ingestion.py avoids the problem by using
    # setdefault, which is why the collision only appears with an outright
    # assignment.)
    sentinel = object()
    previous = sys.modules.get("src.scrapers.cfbd", sentinel)
    sys.modules["src.scrapers.cfbd"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_p1_items", ROOT / "pipeline" / "01_data_ingestion.py")
        mod = types.ModuleType("_p1_items")
        mod.__file__ = str(ROOT / "pipeline" / "01_data_ingestion.py")
        spec.loader.exec_module(mod)
    finally:
        if previous is sentinel:
            del sys.modules["src.scrapers.cfbd"]
        else:
            sys.modules["src.scrapers.cfbd"] = previous
    return mod


@pytest.fixture(scope="module")
def p1():
    return _load_step1()


# ── #2 phantom opponent-side rows ──────────────────────────────────────────────

def _gl(rows):
    return pd.DataFrame(rows, columns=["player_id", "game_id", "school",
                                        "season_year", "rec"])


def test_drops_the_statless_opponent_row_and_keeps_the_real_one(p1):
    """The observed shape: many rows at the true school, one empty row on the
    opponent's side of a single game."""
    rows = [("p1", g, "UTSA", 2014, 3.0) for g in range(1, 11)]
    rows.append(("p1", 5, "Southern Miss", 2014, np.nan))   # phantom
    out = p1._drop_opponent_side_phantom_rows(_gl(rows))
    assert len(out) == 10
    assert set(out.school) == {"UTSA"}


def test_majority_is_computed_per_season_not_per_career(p1):
    """A real mid-career transfer must survive: the player's modal school differs
    by season, so a career-wide majority would delete legitimate rows."""
    rows = ([("p1", g, "Tulsa", 2021, 1.0) for g in range(1, 9)] +
            [("p1", g, "Houston", 2022, 1.0) for g in range(20, 24)])
    rows.append(("p1", 21, "Tulsa", 2022, np.nan))   # phantom in the 2022 game
    out = p1._drop_opponent_side_phantom_rows(_gl(rows))
    assert len(out) == 12
    # Both real school-seasons intact
    assert set(out[out.season_year == 2021].school) == {"Tulsa"}
    assert set(out[out.season_year == 2022].school) == {"Houston"}


def test_uncontaminated_frames_pass_through_untouched(p1):
    rows = [("p1", 1, "UTSA", 2014, 1.0), ("p2", 1, "Southern Miss", 2014, 2.0)]
    gl = _gl(rows)
    out = p1._drop_opponent_side_phantom_rows(gl)
    assert len(out) == len(gl)


def test_a_genuine_tie_is_left_alone_not_guessed(p1):
    """1-vs-1 with no majority: dropping either row would be a coin flip, so both
    survive and the caller logs it for inspection. Silently picking one is how you
    delete a real row."""
    rows = [("p1", 1, "UTSA", 2014, 1.0), ("p1", 1, "Southern Miss", 2014, 2.0)]
    out = p1._drop_opponent_side_phantom_rows(_gl(rows))
    assert len(out) == 2, "a no-majority tie must not be resolved by guessing"


def test_missing_columns_are_tolerated(p1):
    df = pd.DataFrame({"player_id": ["a"], "rec": [1.0]})
    out = p1._drop_opponent_side_phantom_rows(df)
    assert len(out) == 1


def test_runs_before_any_join_in_assembly():
    """Ordering is load-bearing: every downstream join keys off `school`, so a row
    on the wrong side of the game would resolve the wrong opponent, venue and team
    context. The dedupe must precede _add_schedule_features."""
    src = (ROOT / "pipeline" / "01_data_ingestion.py").read_text()
    marker = src.index("# ── Assemble player_gamelogs.parquet")
    tail = src[marker:]
    assert tail.index("_drop_opponent_side_phantom_rows(gamelogs)") < \
           tail.index("_add_schedule_features(gamelogs, schedule)")


# ── #4 adjusted_* semantics sidecar ────────────────────────────────────────────

SEMANTICS_PATH = C.PROCESSED_DIR / "adjusted_stat_semantics.json"


def test_step5_declares_the_sidecar_in_its_docstring():
    """The whole point of item #4 is that the dual semantics were folklore. If the
    docstring stops explaining it, the sidecar alone will not save the next reader."""
    src = (ROOT / "pipeline" / "05_residual_computation.py").read_text()
    assert "adjusted_stat_semantics.json" in src
    for token in ("raw_fallback", "residual", "no_model"):
        assert token in src, f"docstring/code no longer mentions the {token} semantic"


@pytest.mark.skipif(not SEMANTICS_PATH.exists(),
                    reason="sidecar not generated yet (run step 5)")
def test_sidecar_shape_and_vocabulary():
    d = json.loads(SEMANTICS_PATH.read_text())
    assert d, "sidecar is empty"
    allowed = {"residual", "raw_fallback", "no_model", "failed"}
    for pos, stats in d.items():
        assert isinstance(stats, dict) and stats
        for stat, entry in stats.items():
            assert entry["semantic"] in allowed, f"{pos}/{stat}: {entry['semantic']}"
            assert "cv_r2" in entry


@pytest.mark.skipif(not SEMANTICS_PATH.exists(),
                    reason="sidecar not generated yet (run step 5)")
def test_semantic_matches_the_cv_r2_threshold_that_decided_it():
    """Guards the invariant the sidecar exists to expose: a head is a residual iff
    its CV R² cleared 0.01. If these ever disagree, either the threshold moved or
    the sidecar is being written at the wrong point in the loop."""
    d = json.loads(SEMANTICS_PATH.read_text())
    for pos, stats in d.items():
        for stat, entry in stats.items():
            r2, sem = entry["cv_r2"], entry["semantic"]
            if r2 is None:
                assert sem in {"no_model", "raw_fallback"}
                continue
            if sem == "residual":
                assert r2 >= 0.01, f"{pos}/{stat} residual with cv_r2={r2}"
            elif sem == "raw_fallback":
                assert r2 < 0.01, f"{pos}/{stat} raw_fallback with cv_r2={r2}"


@pytest.mark.skipif(not SEMANTICS_PATH.exists(),
                    reason="sidecar not generated yet (run step 5)")
def test_most_heads_pass_the_raw_stat_through():
    """Not a style assertion — a factual guard on something surprising enough to
    be worth pinning: only 16 of 74 heads (22%) actually adjust. If a change makes
    most heads residual, the contextual layer's behaviour shifted fundamentally and
    every downstream Kalman prior should be re-examined."""
    d = json.loads(SEMANTICS_PATH.read_text())
    total = sum(len(m) for m in d.values())
    residual = sum(1 for m in d.values() for e in m.values()
                   if e["semantic"] == "residual")
    assert total > 50, "unexpectedly few heads — did MODELED_STATS shrink?"
    assert residual < total / 2, (
        f"{residual}/{total} heads now adjust; the raw-passthrough majority that "
        "downstream scale assumptions rest on has flipped")


@pytest.mark.skipif(not SEMANTICS_PATH.exists(),
                    reason="sidecar not generated yet (run step 5)")
def test_no_head_failed_residual_computation():
    d = json.loads(SEMANTICS_PATH.read_text())
    failed = [(p, s) for p, m in d.items() for s, e in m.items()
              if e["semantic"] == "failed"]
    assert not failed, f"heads holding NaN after a failed residual fit: {failed}"
