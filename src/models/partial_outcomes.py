"""Partial outcomes for recent classes in the value board's board curve
(production home of experiments/00zv_partial_outcomes.py).

Rookie value needs four NFL seasons, so the board curve would otherwise be
fit only on classes drafted >= 4 years ago. Classes with 1-3 played seasons
get a projected four-year value: played seasons + predicted remaining
seasons, from a small LightGBM per k = seasons played, fit on complete
classes: Σ value_y(k+1..4) ~ value_y1..y_k, log board rank, position.
Board players only; 0 for seasons of players never in the NFL.

00zv (test classes 2019-22): MSE −3.5% (P 0.99); projections as good as
the true labels (rho 0.82-0.99 by k, unbiased). The gain shrinks as complete
classes accumulate; its production value is tracking drift (position levels)
three classes sooner.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJ = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=20,
            subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)
POS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
FIRST_BOARD_SEASON = 2013


def board_value_frame(snapshot: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Board players (step 10 snapshot) with per-season rookie value v1..v4
    (label value_yk; 0 when the player has no label row = never NFL). Unplayed
    seasons are NaN in the label and stay NaN here."""
    from src.models.offboard_model import pid
    b = snapshot.loc[snapshot["hasan_rank"].notna() & snapshot["position_group"].isin(POS),
                     ["player_id", "position_group", "hasan_rank", "last_college_season_year"]]
    b = b.drop_duplicates("player_id").rename(columns={"position_group": "pos_group"})
    b = b[b["last_college_season_year"] >= FIRST_BOARD_SEASON].copy()
    b["draft_year"] = b["last_college_season_year"].astype(int) + 1
    L = labels.dropna(subset=["player_id"]).assign(pid=lambda d: pid(d["player_id"]))
    L = L.dropna(subset=["pid"]).drop_duplicates("pid").set_index("pid")
    p = pd.Series(pid(b["player_id"]), index=b.index)
    has = p.isin(L.index)
    for k in range(1, 5):
        v = p.map(L[f"value_y{k}"])
        b[f"v{k}"] = np.where(has, v, 0.0)
    return b.reset_index(drop=True)


def project_partial(b: pd.DataFrame, end_year: int) -> tuple[pd.DataFrame, dict]:
    """Projected four-year value for board players of classes with 1-3
    played seasons (last played NFL season = end_year − 1). Returns rows
    (hasan_rank, pos_group, last_college_season_year, rv) and diagnostics."""
    played = end_year - b["draft_year"]               # seasons played
    done = b[(played >= 4)].dropna(subset=["v1", "v2", "v3", "v4"])
    out, diag = [], {}
    for k in (1, 2, 3):
        R = b[played == k].copy()
        if R.empty or done.empty:
            continue
        feats = [f"v{j}" for j in range(1, k + 1)] + ["lr", "pos"]

        def X(d):
            return d.assign(lr=np.log(d["hasan_rank"]),
                            pos=pd.Categorical(d["pos_group"], categories=POS))[feats]
        R = R.dropna(subset=[f"v{j}" for j in range(1, k + 1)])
        y = done[[f"v{j}" for j in range(k + 1, 5)]].sum(axis=1)
        mdl = lgb.LGBMRegressor(**PROJ).fit(X(done), y)
        R["rv"] = R[[f"v{j}" for j in range(1, k + 1)]].sum(axis=1) + np.clip(mdl.predict(X(R)), 0, None)
        out.append(R)
        diag[int(end_year - k)] = {"seasons_played": k, "n": int(len(R)), "mean_rv": round(float(R["rv"].mean()), 4)}
    cols = ["hasan_rank", "pos_group", "last_college_season_year", "rv"]
    return (pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)), diag
