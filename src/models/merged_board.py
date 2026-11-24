"""Merged board: one ranked list from the board curve + off-board model.

Currency (2026-11-01): fit with target="usage" (y3-4 snap share, the
original build) or target="rv" (rookie-deal value: Σ y1-y4 veteran-market
price of the player's role, % of cap; src/models/rookie_value.py). Step 10
fits both and ranks by rookie value (experiments/00zs: ties snap share on
value captured; adopted because value is the product's currency). The
description below uses "usage"; the value build is identical with the
target swapped.

Production version of the capped, usage-built merge tested in
experiments/00zf_merged_board.py (arm `merged_dep_cap`). Two calibrated
sources share one currency, usage = y3-4 snap share (0 if never in the NFL):

  board players     E[usage | log hasan_rank, position]: decreasing isotonic
                    per position, shrunk toward the pooled curve with weight
                    n/(n + K_BOARD).
  off-board players the deployed impact model's percentile within
                    (class × position) among off-board players → usage, via
                    an increasing isotonic per position shrunk toward pooled
                    with weight n_nfl/(n_nfl + K_OFF). Then capped at the
                    shrunk mean usage of the top 1% (pct ≥ CAP_Q) of
                    off-board players, so a single-observation isotonic
                    plateau can't lift an off-board player above the board.

Both calibrators are fit on the deployed model's OUT-OF-FOLD predictions
(global_oof_predictions.parquet), never on in-sample scores, and only on
classes whose y3-4 seasons have been played (usage_y34 counts unplayed
seasons as 0, which would read as busts).

Current class. The newest class in the output holds every player whose last
college season is the most recent one — underclassmen included — so it is
~3x a normal class. A within-class percentile there would push the top
off-board players down the curve. For that class only, off-board players are
ranked against a reference class size (median of the previous
N_REF_CLASSES classes per position): percentile = 1 − (rank − 1) / N_ref,
floored at 0. For every other class this formula is the ordinary
within-class percentile.

K, P and OTHER get no merged score (the experiment excluded them).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

BOARD_ERA_START = 2013   # first last-college-season with a board (class 2014)
K_BOARD = 100            # board players per position for half weight on its own curve
K_OFF = 50               # off-board NFL players per position for half weight
CAP_Q = 0.99             # off-board cap: mean usage of the top 1% by model percentile
N_REF_CLASSES = 3        # prior classes whose median size sizes the current class
EXCLUDED_GROUPS = ("K", "P", "OTHER")


class ShrunkIso:
    """Per-position isotonic curve shrunk toward a pooled one.

    level=True (experiments/01a, adopted for the rookie-value board): shrink
    toward the pooled curve RESCALED to the position's level,
        w·iso_g(x) + (1 − w)·lvl_g·pooled(x),  lvl_g = mean(y_g) / mean(pooled(x_g)),
    so the shrinkage borrows the pooled SHAPE but keeps the position's
    level (pooled shrinkage under-priced board QBs by ~19%)."""

    def __init__(self, increasing: bool, k: float, level: bool = False):
        self.increasing, self.k, self.level = increasing, k, level

    def _iso(self):
        return IsotonicRegression(increasing=self.increasing, out_of_bounds="clip",
                                  y_min=0.0, y_max=1.0)

    def fit(self, x, y, pos, weight_n: dict | None = None) -> "ShrunkIso":
        x, y, pos = np.asarray(x, float), np.asarray(y, float), np.asarray(pos)
        self.pooled = self._iso().fit(x, y)
        self.per, self.w = {}, {}
        for g in np.unique(pos):
            m = pos == g
            n = m.sum() if weight_n is None else weight_n.get(g, 0)
            self.per[g] = self._iso().fit(x[m], y[m])
            self.w[g] = n / (n + self.k)
        pp = self.pooled.predict(x)
        self.lvl = {g: (y[pos == g].mean() / pp[pos == g].mean())
                    if self.level and pp[pos == g].mean() > 0 else 1.0 for g in np.unique(pos)}
        return self

    def predict(self, x, pos) -> np.ndarray:
        x, pos = np.asarray(x, float), np.asarray(pos)
        out = self.pooled.predict(x)
        for g in np.unique(pos):
            if g in self.per:
                m = pos == g
                out[m] = (self.w[g] * self.per[g].predict(x[m])
                          + (1 - self.w[g]) * self.lvl[g] * out[m])
        return out


def offboard_percentile(score: pd.Series, cls: pd.Series, pos: pd.Series,
                        n_ref: dict | None = None,
                        ref_cls: int | None = None) -> pd.Series:
    """Percentile of `score` within (class × position), best = 1.

    pct = 1 − (rank − 1) / N, rank descending with average ties. N is the
    group's own size, except for class `ref_cls` where N = n_ref[pos]
    (when that is smaller than the group). Floored at 0.
    """
    df = pd.DataFrame({"s": score, "c": cls, "p": pos})
    rank = df.groupby(["c", "p"])["s"].rank(ascending=False, method="average")
    n = df.groupby(["c", "p"])["s"].transform("count").astype(float)
    if n_ref and ref_cls is not None:
        cur = df["c"] == ref_cls
        ref = df.loc[cur, "p"].map(n_ref).astype(float)
        n.loc[cur] = np.fmin(n.loc[cur], ref.fillna(n.loc[cur]))
    return (1.0 - (rank - 1.0) / n).clip(lower=0.0)


@dataclass
class MergedBoard:
    board_cal: ShrunkIso
    off_cal: ShrunkIso
    cap: dict
    cap_pooled: float
    diagnostics: dict = field(default_factory=dict)
    out_col: str = "merged_usage"

    @classmethod
    def fit(cls, oof: pd.DataFrame, fit_last_season: int, target: str = "usage",
            out_col: str = "merged_usage",
            extra_board: pd.DataFrame | None = None) -> "MergedBoard":
        """oof columns: player_id, pos_group, last_college_season_year,
        hasan_rank, nfl, pred (engine OOF prediction), and the `target`
        column (0 if never NFL). extra_board: additional BOARD rows for the
        board curve only (hasan_rank, pos_group, `target`) — projected
        outcomes of recent classes (src/models/partial_outcomes.py)."""
        E = oof[(oof["last_college_season_year"] >= BOARD_ERA_START)
                & ~oof["pos_group"].isin(EXCLUDED_GROUPS)].reset_index(drop=True)
        board = E["hasan_rank"].notna()
        # Percentiles over the whole OOF era (like the experiment), then
        # restrict fitting to classes with observed y3-4 usage.
        E["pct"] = np.nan
        E.loc[~board, "pct"] = offboard_percentile(
            E.loc[~board, "pred"], E.loc[~board, "last_college_season_year"],
            E.loc[~board, "pos_group"])
        fit = E[E["last_college_season_year"] <= fit_last_season]
        bt, ob = fit[fit["hasan_rank"].notna()], fit[fit["hasan_rank"].isna()]
        n_extra = 0
        if extra_board is not None and len(extra_board):
            xb = extra_board[~extra_board["pos_group"].isin(EXCLUDED_GROUPS)]
            n_extra = len(xb)
            bt = pd.concat([bt[["hasan_rank", "pos_group", target]],
                            xb[["hasan_rank", "pos_group", target]]], ignore_index=True)

        level = target == "rv"   # 01a: level-preserving shrinkage for the value board
        board_cal = ShrunkIso(increasing=False, k=K_BOARD, level=level).fit(
            np.log(bt["hasan_rank"]), bt[target], bt["pos_group"])
        n_nfl = ob.groupby("pos_group")["nfl"].sum().to_dict()
        off_cal = ShrunkIso(increasing=True, k=K_OFF, level=level).fit(
            ob["pct"], ob[target], ob["pos_group"], weight_n=n_nfl)
        top = ob[ob["pct"] >= CAP_Q]
        pooled = float(top[target].mean())
        cap = {g: (len(t) * t[target].mean() + K_OFF * pooled) / (len(t) + K_OFF)
               for g, t in top.groupby("pos_group")}
        diag = {
            "target": target,
            "fit_seasons": [int(fit["last_college_season_year"].min()), int(fit_last_season)],
            "n_board": int(len(bt)), "n_board_projected": int(n_extra), "n_offboard": int(len(ob)),
            "n_offboard_nfl": int(ob["nfl"].sum()),
            "cap": {g: round(float(v), 4) for g, v in cap.items()},
            "cap_pooled": round(pooled, 4),
            "level_preserving": level,
            "board_levels": {g: round(float(v), 3) for g, v in board_cal.lvl.items()},
        }
        return cls(board_cal, off_cal, cap, pooled, diag, out_col)

    def predict(self, snap: pd.DataFrame, score_col: str,
                score_mask: pd.Series | None = None,
                ref_sizing: bool = True) -> pd.DataFrame:
        """snap columns: pos_group, last_college_season_year, hasan_rank,
        `score_col`. Returns `out_col` (merged_usage by default) and
        merged_source, indexed like snap.

        score_mask: rows to score (others get NaN), e.g. the draft pool; the
        off-board percentiles are computed among scored rows only.
        ref_sizing: rank the newest class's off-board players against the
        median size of the previous classes. Turn off when score_mask already
        restricts that class to its real draft pool.

        Works on a positional copy, so a non-unique index is safe."""
        orig_index = snap.index
        snap = snap.reset_index(drop=True)
        out = pd.DataFrame({self.out_col: np.nan, "merged_source": None},
                           index=snap.index)
        ok = (~snap["pos_group"].isin(EXCLUDED_GROUPS)
              & snap["last_college_season_year"].notna())
        if score_mask is not None:
            ok &= pd.Series(np.asarray(score_mask, dtype=bool), index=snap.index)
        board = ok & snap["hasan_rank"].notna()
        off = ok & snap["hasan_rank"].isna() & snap[score_col].notna()

        out.loc[board, self.out_col] = self.board_cal.predict(
            np.log(snap.loc[board, "hasan_rank"].astype(float)), snap.loc[board, "pos_group"])
        out.loc[board, "merged_source"] = "board"

        cls_col = snap["last_college_season_year"]
        cur = int(cls_col[ok].max())
        prev = sorted(c for c in cls_col[off].unique() if c < cur)[-N_REF_CLASSES:]
        sizes = (snap[off & cls_col.isin(prev)]
                 .groupby(["last_college_season_year", "pos_group"]).size())
        n_ref = sizes.groupby(level="pos_group").median().to_dict() if ref_sizing else {}
        pct = offboard_percentile(snap.loc[off, score_col], cls_col[off],
                                  snap.loc[off, "pos_group"], n_ref=n_ref,
                                  ref_cls=cur if ref_sizing else None)
        raw = self.off_cal.predict(pct, snap.loc[off, "pos_group"])
        capv = snap.loc[off, "pos_group"].map(self.cap).fillna(self.cap_pooled).to_numpy()
        out.loc[off, self.out_col] = np.minimum(raw, capv)
        out.loc[off, "merged_source"] = np.where(raw > capv, "model_capped", "model")
        self.diagnostics["current_class"] = cur
        self.diagnostics["n_ref"] = {g: float(v) for g, v in n_ref.items()}
        return out.set_axis(orig_index)
