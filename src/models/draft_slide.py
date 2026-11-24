"""Pick-conditioned rookie value for board players (experiments/01f).

Where a player is actually drafted, relative to his big-board rank,
predicts his rookie value beyond the board (01f: two-sided γ −0.19, MSE
−4.1%, P 0.98). For DECISIONS only the slide half is usable: a player still
on the board at our pick has been passed by every earlier team, which we
observe; a player we would take before his board rank has no such signal
(the historical rise premium reflects what the reaching team knew, and our
own reach is not evidence). So value at pick d = board-curve value ×
min(1, (d / board rank)^γ): sliders are marked down, risers keep the board
value. Slide-only LOCO on 01f's cohort: MSE gain on sliders +2.8e-4 (P 0.99)
vs +2.9e-4 two-sided; all players +1.3e-4, P 1.00.

Off-board players (no board rank) are left unchanged. pick / board rank is
capped at RATIO_MAX (99th percentile in 01f; multiplier >= x0.74) so the
curve is never extrapolated, e.g. to the #1 player shown at pick 40 before
anyone has been taken.
SLIDE_GAMMA = fit_gamma() on drafts 2014-22 with the v4 label (one-sided,
capped form); refit when the label or the board curve changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

SLIDE_GAMMA = -0.254       # fit_gamma(), drafts 2014-22, v4 label (2026-11-02)
RATIO_MAX = 3.3            # 99th percentile of pick / board rank in 01f's cohort


def value_at_pick(value, board_rank, pick, gamma: float = SLIDE_GAMMA):
    """Board-curve value → value if drafted at `pick` (scalar or array).
    Slides only: pick <= board rank → unchanged. NaN board rank (off-board
    players) → unchanged."""
    v = np.asarray(value, dtype=float)
    b = np.asarray(board_rank, dtype=float)
    ratio = np.clip(np.asarray(pick, dtype=float) / np.where(np.isnan(b), 1.0, b), 1.0, RATIO_MAX)
    mult = np.where(np.isnan(b), 1.0, ratio ** gamma)
    return v * mult


def slide_cohort(projections: pd.DataFrame, labels: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    """Drafted board players with four-year rookie value: columns pos_group,
    draft_year, hasan_rank, overall_pick, y14 (01f's cohort, all drafts)."""
    from src.models.merged_board import EXCLUDED_GROUPS
    from src.models.offboard_model import pid
    from src.models.partial_outcomes import board_value_frame
    b = board_value_frame(projections, labels)
    b = b[~b["pos_group"].isin(EXCLUDED_GROUPS)].copy()
    br = bridge.dropna(subset=["player_id"])
    br = br[br["overall_pick"].notna() & (br["is_udfa"] != True)]
    br = br.assign(pid=pid(br["player_id"])).dropna(subset=["pid"]).drop_duplicates("pid")
    b["pid"] = pid(b["player_id"])
    b = b.merge(br[["pid", "overall_pick", "draft_year"]].rename(columns={"draft_year": "bdy"}), on="pid")
    b = b[b["bdy"] == b["draft_year"]]
    b["y14"] = b[["v1", "v2", "v3", "v4"]].sum(axis=1, min_count=4)
    return b.dropna(subset=["y14"]).reset_index(drop=True)


def fit_gamma(cohort: pd.DataFrame) -> float:
    """γ by least squares on board-curve predictions (01f's B2_mult form,
    one-sided and capped as in value_at_pick)."""
    from src.models.merged_board import ShrunkIso, K_BOARD
    lb = np.log(cohort["hasan_rank"].to_numpy(float))
    iso = ShrunkIso(increasing=False, k=K_BOARD, level=True).fit(lb, cohort["y14"], cohort["pos_group"])
    base = iso.predict(lb, cohort["pos_group"])
    slide = np.clip(np.log(cohort["overall_pick"].to_numpy(float)) - lb, 0.0, np.log(RATIO_MAX))
    y = cohort["y14"].to_numpy(float)
    return float(minimize_scalar(lambda g: np.mean((base * np.exp(g * slide) - y) ** 2),
                                 bounds=(-3, 0), method="bounded").x)   # a slide never raises value
