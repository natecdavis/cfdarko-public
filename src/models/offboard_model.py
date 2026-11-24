"""Dedicated off-board model: the merged board's engine for players with no
board rank.

    score = P(reach NFL) × E[usage_y34 | NFL]  = expected y3-4 snap share

Port of the arm that passed experiments/00zi_offboard_pff_confirm.py (the
00zd two-stage model + 00zh's PFF side features, seed-bagged):
  stage 1  LightGBM classifier, every player, y = reached the NFL (is_bridge)
  stage 2  LightGBM regressor, NFL players only, y = usage_y34 (NaN → 0)
Both stages are pooled across the nine non-K/P position groups on the
feature list every step-8 impact head shares, plus pos_code, board_era and
the PFF side features (offense or defense grade / snap triplets).
hasan_rank is a monotone feature (NaN off the board). Untuned. Each stage's
prediction is averaged over SEEDS before the two are multiplied.

Training rows come from step 8's _resolve_fit_features (the snapshot with
labels merged, non-NFL players padded to 0), so OOF folds, the final fit and
scoring all see the same universe.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import POSITION_GROUPS, LGBM_PARAMS

logger = logging.getLogger(__name__)

GROUPS = [g for g in POSITION_GROUPS if g not in ("K", "P")]
STAGE1_PARAMS = {**LGBM_PARAMS, "learning_rate": 0.05, "n_estimators": 2000,
                 "num_leaves": 15, "min_child_samples": 50}
STAGE2_PARAMS = {**LGBM_PARAMS, "learning_rate": 0.03, "n_estimators": 3000,
                 "num_leaves": 15, "min_child_samples": 20}
SEEDS = (42, 43, 44, 45, 46)
BOARD_ERA_START = 2013
ID_COLS = ["player_id", "pos_group", "last_college_season_year"]
MONOTONE_DECREASING = ("hasan_rank",)

OFFENSE = {"QB", "RB", "WR", "TE", "OL"}
SIDE_STATS = ["grades", "snap_counts"]
SIDE_AGGS = ["mean", "peak", "last"]
PFF_FEATURES = ([f"pff_side_{s}_{a}" for s in SIDE_STATS for a in SIDE_AGGS]
                + ["pff_side_snap_counts_total"])


def pid(s: pd.Series) -> pd.Series:
    """Canonical string player id ("123", not "123.0")."""
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


def add_side_pff(snap: pd.DataFrame) -> pd.DataFrame:
    """Position-agnostic PFF career features: offense grade / snaps for
    offensive groups, defense for defensive ones."""
    snap = snap.copy()
    off = snap["position_group"].isin(OFFENSE)
    for s in SIDE_STATS:
        for a in SIDE_AGGS + (["total"] if s == "snap_counts" else []):
            o, d = f"pff_{s}_offense_{a}", f"pff_{s}_defense_{a}"
            if o in snap.columns and d in snap.columns:
                snap[f"pff_side_{s}_{a}"] = np.where(off, snap[o], snap[d])
    missing = [c for c in PFF_FEATURES if c not in snap.columns]
    if missing:
        raise KeyError(f"PFF side columns missing from the snapshot: {missing}")
    return snap


def build_rows(step8, snapshot: pd.DataFrame, labels: pd.DataFrame,
               bridge: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pooled rows (one per player × group) and the feature list."""
    snapshot = add_side_pff(snapshot)
    frames, common = [], None
    for g in GROUPS:
        cols, df = step8._resolve_fit_features(snapshot, labels, g, "impact_target", "impact")
        common = cols if common is None else [c for c in common if c in cols]
        frames.append(df.assign(pos_group=g))
    common = common + [c for c in PFF_FEATURES if c not in common]
    keep = ID_COLS + [c for c in common if c not in ID_COLS]
    if "hasan_rank" not in keep:
        keep.append("hasan_rank")
    rows = pd.concat([f[[c for c in keep if c in f.columns]] for f in frames],
                     ignore_index=True)
    rows["player_id"] = pid(rows["player_id"])

    lab = labels[labels["player_id"].notna()].assign(player_id=lambda d: pid(d["player_id"]))
    lab = lab.drop_duplicates("player_id")[["player_id", "is_bridge", "usage_y34"]]
    bp = bridge[bridge["player_id"].notna()].assign(player_id=lambda d: pid(d["player_id"]))
    bp = bp.drop_duplicates("player_id")[["player_id", "round"]]
    rows = rows.merge(lab, on="player_id", how="left").merge(bp, on="player_id", how="left")

    rows["nfl"] = rows["is_bridge"].fillna(False).astype(bool)
    rows["usage"] = np.where(rows["nfl"], rows["usage_y34"].fillna(0.0), 0.0)
    rows["drafted"] = rows["round"].notna().astype(float)
    rows["pos_code"] = rows["pos_group"].map({g: i for i, g in enumerate(GROUPS)})
    rows["board_era"] = (rows["last_college_season_year"] >= BOARD_ERA_START).astype(float)
    # Column order matters with LightGBM column subsampling: base features,
    # pos_code, board_era, then PFF — the order 00zi validated.
    base = [c for c in common if c not in ID_COLS and c not in PFF_FEATURES]
    feats = base + ["pos_code", "board_era"] + PFF_FEATURES
    logger.info("off-board model rows %d, NFL %d, drafted %d; features %d",
                len(rows), int(rows["nfl"].sum()), int(rows["drafted"].sum()), len(feats))
    return rows, feats


def _X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float32, na_value=np.nan)


def _fit(step8, est_cls, params, tr, cols, y):
    mono = [-1 if c in MONOTONE_DECREASING else 0 for c in cols]
    mk = lambda: est_cls(**{**params, "monotone_constraints": mono})
    return step8._fit_fold_honest(mk, _X(tr, cols), y,
                                  tr["last_college_season_year"].fillna(0).values)


class OffboardBag:
    """Seed-bagged two-stage model."""

    def __init__(self, seeds=SEEDS):
        self.seeds = tuple(seeds)
        self.stage1, self.stage2, self.trees = [], [], []

    def fit(self, step8, tr1: pd.DataFrame, tr2: pd.DataFrame, cols: list[str]) -> "OffboardBag":
        """tr1: stage-1 rows (all players); tr2: stage-2 rows (NFL players)."""
        self.cols = list(cols)
        for seed in self.seeds:
            clf, b1 = _fit(step8, lgb.LGBMClassifier, {**STAGE1_PARAMS, "random_state": seed},
                           tr1, self.cols, tr1["nfl"].astype(int).values)
            reg, b2 = _fit(step8, lgb.LGBMRegressor, {**STAGE2_PARAMS, "random_state": seed},
                           tr2, self.cols, tr2["usage"].values)
            self.stage1.append(clf)
            self.stage2.append(reg)
            self.trees.append((b1, b2))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _X(df, self.cols)
        p1 = np.mean([m.predict_proba(X)[:, 1] for m in self.stage1], axis=0)
        m2 = np.mean([np.clip(m.predict(X), 0, 1) for m in self.stage2], axis=0)
        return pd.DataFrame({"p_nfl": p1, "usage_if_nfl": m2, "score": p1 * m2},
                            index=df.index)


def oof_predictions(step8, rows: pd.DataFrame, feats: list[str], folds,
                    seeds=SEEDS) -> pd.DataFrame:
    """Expanding-window OOF on step 8's tuning folds (holdout never touched)."""
    yr = rows["last_college_season_year"].fillna(0)
    parts = []
    for train_end, val_start, val_end in folds:
        tr = rows[yr <= train_end]
        va = rows[yr.between(val_start, val_end)]
        bag = OffboardBag(seeds).fit(step8, tr, tr[tr["nfl"]], feats)
        pred = bag.predict(va)
        parts.append(va[ID_COLS + ["hasan_rank", "nfl", "usage"]].join(pred)
                     .assign(val_start=val_start, val_end=val_end))
        logger.info("OOF fold %d-%d: %d train / %d val rows; trees %s",
                    val_start, val_end, len(tr), len(va), bag.trees)
    return pd.concat(parts, ignore_index=True)
