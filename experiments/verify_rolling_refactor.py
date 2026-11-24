"""
One-off verification: assert that the refactored
compute_rolling_opponent_strength (delegating to
src.utils.rolling.compute_rolling_entity_strength) produces byte-identical
output to the legacy inline implementation.

Runs both, joins on (game_id, team), asserts per-column max-abs-diff is
< 1e-9 (NaN-tolerant). Exits non-zero on mismatch.

Delete after the QB-rolling work lands and CI / review picks up the
refactor cleanly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scrapers.cfbd import compute_rolling_opponent_strength  # refactored


def _legacy(adv_game_stats: pd.DataFrame,
            schedule: pd.DataFrame,
            halflife_days: int = 45) -> pd.DataFrame:
    """Verbatim copy of the pre-refactor inline implementation."""
    if adv_game_stats.empty or schedule.empty:
        return pd.DataFrame()
    sched_slim = schedule[["game_id", "game_date"]].drop_duplicates("game_id")
    df = adv_game_stats.merge(sched_slim, on="game_id", how="left")
    df["game_date"] = pd.to_datetime(df["game_date"])
    stat_cols = [
        "def_ppa", "off_ppa",
        "def_success_rate", "def_stuff_rate", "def_explosiveness",
        "off_success_rate", "off_explosiveness",
        "havoc_total", "havoc_front_seven", "havoc_db",
    ]
    for col in stat_cols:
        if col not in df.columns:
            df[col] = float("nan")
    team_history: dict = {}
    for team, grp in df.groupby("team"):
        team_history[team] = grp.sort_values("game_date")[
            ["game_date", "game_id"] + stat_cols
        ].copy()
    log_half = math.log(0.5) / halflife_days
    records: list[dict] = []
    for _, row in df.iterrows():
        game_id = row["game_id"]
        team = row["team"]
        opp = row["opponent"]
        game_date = row["game_date"]
        _nan_record = {
            "game_id": game_id, "team": team,
            "opp_rolling_def_ppa":             float("nan"),
            "opp_rolling_off_ppa":             float("nan"),
            "opp_rolling_def_success_rate":    float("nan"),
            "opp_rolling_def_stuff_rate":      float("nan"),
            "opp_rolling_def_explosiveness":   float("nan"),
            "opp_rolling_off_success_rate":    float("nan"),
            "opp_rolling_off_explosiveness":   float("nan"),
            "opp_rolling_havoc_total":         float("nan"),
            "opp_rolling_havoc_front_seven":   float("nan"),
            "opp_rolling_havoc_db":            float("nan"),
        }
        if pd.isna(game_date) or opp not in team_history:
            records.append(_nan_record)
            continue
        opp_hist = team_history[opp]
        prior = opp_hist[opp_hist["game_date"] < game_date]
        if prior.empty:
            records.append(_nan_record)
            continue
        days_elapsed = (game_date - prior["game_date"]).dt.days.clip(lower=0).values
        try:
            log_weights = log_half * days_elapsed.astype(float)
            if log_weights.max() > 700:
                log_weights -= log_weights.max() - 700
            weights = pd.Series(data=np.exp(log_weights), index=prior.index)
        except Exception:
            weights = pd.Series(1.0, index=prior.index)

        def _wavg(col: str) -> float:
            vals = prior[col].values.astype(float)
            w = weights.values
            valid = ~np.isnan(vals)
            if not valid.any():
                return float("nan")
            return float(np.average(vals[valid], weights=w[valid]))

        records.append({
            "game_id":                          game_id,
            "team":                             team,
            "opp_rolling_def_ppa":              _wavg("def_ppa"),
            "opp_rolling_off_ppa":              _wavg("off_ppa"),
            "opp_rolling_def_success_rate":     _wavg("def_success_rate"),
            "opp_rolling_def_stuff_rate":       _wavg("def_stuff_rate"),
            "opp_rolling_def_explosiveness":    _wavg("def_explosiveness"),
            "opp_rolling_off_success_rate":     _wavg("off_success_rate"),
            "opp_rolling_off_explosiveness":    _wavg("off_explosiveness"),
            "opp_rolling_havoc_total":          _wavg("havoc_total"),
            "opp_rolling_havoc_front_seven":    _wavg("havoc_front_seven"),
            "opp_rolling_havoc_db":             _wavg("havoc_db"),
        })
    return pd.DataFrame(records)


def main() -> int:
    raw = ROOT / "data" / "raw"
    adv = pd.read_parquet(raw / "advanced_game_stats.parquet")
    sched = pd.read_parquet(raw / "game_schedule.parquet")
    tb = pd.read_parquet(raw / "team_box_scores.parquet")

    havoc_cols = [c for c in ["game_id", "team", "havoc_total",
                              "havoc_front_seven", "havoc_db"]
                  if c in tb.columns]
    if len(havoc_cols) > 2:
        adv = adv.merge(tb[havoc_cols], on=["game_id", "team"], how="left")

    print(f"Inputs: adv={len(adv)} sched={len(sched)} tb={len(tb)}")

    legacy = _legacy(adv, sched, halflife_days=45).sort_values(["game_id", "team"]).reset_index(drop=True)
    new = compute_rolling_opponent_strength(adv, sched, halflife_days=45).sort_values(["game_id", "team"]).reset_index(drop=True)

    print(f"Legacy rows: {len(legacy)}   New rows: {len(new)}")
    if len(legacy) != len(new):
        print("FAIL: row counts differ")
        return 1

    if not (legacy["game_id"].values == new["game_id"].values).all():
        print("FAIL: game_id columns differ after sort")
        return 1
    if not (legacy["team"].values == new["team"].values).all():
        print("FAIL: team columns differ after sort")
        return 1

    stat_cols = [c for c in legacy.columns if c.startswith("opp_rolling_")]
    max_diff_by_col: dict[str, float] = {}
    mismatch = False
    for col in stat_cols:
        a = legacy[col].values.astype(float)
        b = new[col].values.astype(float)
        both_nan = np.isnan(a) & np.isnan(b)
        nan_disagreement = np.isnan(a) ^ np.isnan(b)
        if nan_disagreement.any():
            n = int(nan_disagreement.sum())
            print(f"FAIL: {col} has {n} NaN-status mismatches")
            mismatch = True
            continue
        diff = np.where(both_nan, 0.0, np.abs(a - b))
        max_diff_by_col[col] = float(np.nanmax(diff))

    if mismatch:
        return 1

    overall = max(max_diff_by_col.values())
    print(f"Max abs diff overall: {overall:.3e}")
    for col, d in sorted(max_diff_by_col.items()):
        print(f"  {col:38s}  max_abs_diff={d:.3e}")

    if overall > 1e-9:
        print("FAIL: equivalence broken (max_abs_diff > 1e-9)")
        return 1

    print("OK: refactor is byte-equivalent to legacy (max_abs_diff <= 1e-9)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
