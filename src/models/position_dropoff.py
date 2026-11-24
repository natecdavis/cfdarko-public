"""Position drop-off between our current pick and our next one.

At pick p with our next pick at n, for each position:
    best_now        the highest value-at-p player available at the position
    exp_next        expected value of the best player at the position still
                    on the board at n, valued at n
    drop            best_now − exp_next: what waiting on the position costs
    p_best_lasts    P(today's best at the position is still there at n)

and for each player c:
    pos_wait_cost   value_c(p) − exp_next(position of c): take him now vs take
                    the best at his position at n (the "we need this
                    position" view)
    two_pick        value_c(p) + E[best player overall at n | c is gone]: the
                    pure-value view; the best two-pick plan takes the player
                    with the highest two_pick. It prefers high-value players
                    who will not last, whatever the position.

E[best at n] over a set of players sorted by value at n:
    Σ_j v_j · P_j · Π_{i ranked above j} (1 − P_i)
assuming players' availabilities are independent (they are not quite: one
team's pick removes one player), with 0 if nobody is left.

Values at n use value_at_pick (src/models/draft_slide.py): a player who is
still there at n has slid further and is worth less then. Availability comes
from src/models/availability.py (position-aware for board players, 00zx
arm C_pos; off-board players from 00zy).

Position groups for the view: the board's groups, with OL split into OT /
IOL when the listed position says so (values still come from the OL curve).
DB is not split: about half of board DBs are listed only as "DB".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUBPOS = {"OT": "OT", "Offensive Tackle": "OT", "OG": "IOL", "C": "IOL", "Offensive Guard": "IOL",
          "G": "IOL", "OC": "IOL"}


def view_position(group: pd.Series, position: pd.Series) -> pd.Series:
    """Board group, with OL → OT / IOL where the listed position allows;
    otherwise the group itself."""
    sub = position.map(SUBPOS)
    ok = (group == "OL") & sub.isin(["OT", "IOL"])
    return group.where(~ok, sub)


def expected_best(values: np.ndarray, p_avail: np.ndarray) -> float:
    """E[max value among the players still available], independence assumed."""
    if len(values) == 0:
        return 0.0
    o = np.argsort(-values)
    v, p = values[o], np.clip(p_avail[o], 0.0, 1.0)
    none_above = np.concatenate([[1.0], np.cumprod(1 - p)[:-1]])
    return float(np.sum(v * p * none_above))


def _expected_best_excluding_each(values: np.ndarray, p_avail: np.ndarray) -> np.ndarray:
    """E[max at n] with each player removed in turn (O(N) per player via the
    sorted order: removing player k only changes the terms below him)."""
    n = len(values)
    o = np.argsort(-values)
    v, p = values[o], np.clip(p_avail[o], 0.0, 1.0)
    q = 1 - p
    prefix = np.concatenate([[1.0], np.cumprod(q)])          # Π_{i<j} q_i for j = 0..n
    term = v * p * prefix[:-1]
    cum = np.concatenate([[0.0], np.cumsum(term)])
    out_sorted = np.empty(n)
    for k in range(n):
        below = term[k + 1:] / q[k] if q[k] > 1e-12 else _recompute_below(v, p, k)
        out_sorted[k] = cum[k] + below.sum()
    out = np.empty(n)
    out[o] = out_sorted
    return out


def _recompute_below(v, p, k):
    """Terms below k when q_k = 0 (player k certain to be there)."""
    vv, pp = v[k + 1:], p[k + 1:]
    pre = np.concatenate([[1.0], np.cumprod(1 - pp)[:-1]]) * np.prod(1 - p[:k])
    return vv * pp * pre


def dropoff(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pool columns: player_name, view_pos, v_now (value at our pick), v_next
    (value at our next pick), p_next (P still there at the next pick).
    Returns (pool with pos_wait_cost and two_pick, per-position table)."""
    pool = pool.copy()
    vn, pn = pool["v_next"].to_numpy(float), pool["p_next"].to_numpy(float)
    pool["two_pick"] = pool["v_now"].to_numpy(float) + _expected_best_excluding_each(vn, pn)
    rows = []
    for g, G in pool.groupby("view_pos"):
        best = G.loc[G["v_now"].idxmax()]
        e = expected_best(G["v_next"].to_numpy(float), G["p_next"].to_numpy(float))
        rows.append({"pos": g, "best_now": best["player_name"], "best_now_value": best["v_now"],
                     "p_best_lasts": best["p_next"], "exp_next": e, "drop": best["v_now"] - e,
                     "n_available": len(G)})
        pool.loc[G.index, "pos_wait_cost"] = G["v_now"] - e
    T = pd.DataFrame(rows).sort_values("drop", ascending=False).reset_index(drop=True)
    return pool, T
