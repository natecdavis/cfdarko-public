"""
P(a board player is still available at pick n) — production home of 00zx.

Censored log-normal on log big-board rank (arm B_ln, chosen in 00zx):
    log Y ~ N(mu, sigma),  mu = a + b L + c L^2,  log sigma = d + e L,
    L = log hasan_rank, Y = overall pick (undrafted ⇔ Y > 262).
Discrete-interval likelihood (pick n ⇔ log Y in [log(n-0.5), log(n+0.5)]).
S(n) = P(Y > n) = 1 − Φ((log(n+0.5) − mu) / sigma).

At pick c, if we pass on a player still on the board, P(he is there at
our pick m) = S(m-1) / S(c). Off-board players (no hasan_rank) are not modelled.
Validation (00zx, leave-one-class-out 2014-26): calibrated at every
horizon; position offsets (by_position=True, arm C_pos) tie on CRPS but
predict "still there at your next pick" better (conditional Brier, P 0.99).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

N_PICKS = 262


POS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB", "ST"]
RIDGE = 1.0


def _pos_dummies(pos):
    p = np.asarray(pos, dtype=object)
    p = np.where(np.isin(p, POS[:-1]), p, "ST")
    return (p[:, None] == np.array(POS)[None, :]).astype(float)


@dataclass
class AvailabilityModel:
    """by_position=False: arm B_ln (00zx's pick by its CRPS rule).
    by_position=True: arm C_pos — position offsets in mu and log sigma (ridge
    1.0); CRPS tie with B_ln but better on 00zx's decision-relevant
    conditional Brier (P(there at your next pick | there now), P 0.99): QBs go
    earlier than their rank implies and with more spread, RBs later. Used by
    the position drop-off view in scripts/board_at_pick.py."""
    theta: np.ndarray = None
    n_train: int = 0
    by_position: bool = False

    def _design(self, rank, pos):
        L = np.log(np.asarray(rank, float))
        Xm = np.column_stack([np.ones_like(L), L, L ** 2])
        Xs = np.column_stack([np.ones_like(L), L])
        if self.by_position:
            if pos is None:
                raise ValueError("by_position model needs pos")
            D = _pos_dummies(pos)
            Xm, Xs = np.column_stack([Xm, D]), np.column_stack([Xs, D])
        return Xm, Xs

    def fit(self, rank, pick, pos=None):
        """rank: big-board rank; pick: overall pick, NaN if undrafted;
        pos: position group (needed when by_position)."""
        Xm, Xs = self._design(rank, pos)
        y = np.asarray(pick, float)
        drafted = np.isfinite(y) & (y <= N_PICKS)
        lo = np.log(np.where(drafted, y - 0.5, N_PICKS + 0.5))
        hi = np.log(np.where(drafted, y + 0.5, np.inf))
        pm, ps = Xm.shape[1], Xs.shape[1]
        npos = len(POS) if self.by_position else 0

        def nll(th):
            mu, sg = Xm @ th[:pm], np.exp(Xs @ th[pm:])
            zl = (lo - mu) / sg
            p = np.where(drafted, norm.cdf((hi - mu) / sg) - norm.cdf(zl), norm.sf(zl))
            pen = RIDGE * (np.sum(th[pm - npos:pm] ** 2) + np.sum(th[pm + ps - npos:] ** 2)) if npos else 0.0
            return -np.sum(np.log(np.clip(p, 1e-12, None))) + pen

        th0 = np.zeros(pm + ps)
        th0[:2] = [0.3, 0.9]
        th0[pm] = np.log(0.5)
        self.theta = minimize(nll, th0, method="L-BFGS-B").x
        self.pm = pm
        self.n_train = len(y)
        return self

    def mu_sigma(self, rank, pos=None):
        Xm, Xs = self._design(rank, pos)
        pm = getattr(self, "pm", 3)
        return Xm @ self.theta[:pm], np.exp(Xs @ self.theta[pm:])

    def survival(self, rank, n, pos=None):
        """P(pick > n) for each rank; n scalar or array broadcastable to rank."""
        mu, sg = self.mu_sigma(rank, pos)
        return norm.sf((np.log(np.asarray(n, float) + 0.5) - mu) / sg)

    def p_available(self, rank, current, nxt, pos=None):
        """P(still on the board at pick `nxt` | on the board at pick `current`
        and we pass): not taken at picks current+1 .. nxt-1, = S(nxt-1) / S(current)."""
        if nxt <= current + 1:
            return np.ones(np.shape(rank))
        return (self.survival(rank, nxt - 1, pos)
                / np.clip(self.survival(rank, current, pos), 1e-12, None))


def training_frame(projections: pd.DataFrame, bridge: pd.DataFrame, max_class=None):
    """Board players with known outcomes: classes whose draft has happened."""
    P = projections.drop_duplicates("player_id")
    b = P[P["hasan_rank"].notna() & P["hasan_draft_year"].notna()].copy()
    drafted = bridge[bridge["overall_pick"].notna()].drop_duplicates("player_id").set_index("player_id")
    done = int(drafted["draft_year"].max()) if max_class is None else max_class
    b = b[b["hasan_draft_year"] <= done]
    b["pick"] = b["player_id"].map(drafted["overall_pick"])
    return b


def fit_from_frames(projections, bridge, max_class=None, by_position=False):
    b = training_frame(projections, bridge, max_class)
    return AvailabilityModel(by_position=by_position).fit(b["hasan_rank"], b["pick"], b["position_group"])


# ---------------------------------------------------------------------------
# Off-board players (00zy): S(n) = 1 − p_d · F(n | combine invite)
# p_d: logistic on log rank within the class's off-board pool by 8d p_nfl and
# by 8d score, combine invite, position. Ranks, not levels: the 8d p_nfl
# level drifts (OOF 2018 season mean 0.41; current pool 0.007).
# ---------------------------------------------------------------------------
OFF_POS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB", "ST"]
OFF_CLASSES = (2015, 2022)  # classes with out-of-fold 8d predictions


def _off_features(d, cls_col, p_col, s_col, pos_col, inv_col):
    d = d.copy()
    g = d.groupby(cls_col)
    worst = g[p_col].transform("size")
    lr_p = np.log(g[p_col].rank(ascending=False).fillna(worst))
    lr_s = np.log(g[s_col].rank(ascending=False).fillna(worst))
    pos = np.where(d[pos_col].isin(OFF_POS[:-1]), d[pos_col], "ST")
    D = (pos[:, None] == np.array(OFF_POS)[None, :]).astype(float)
    inv = d[inv_col].fillna(0).astype(float).to_numpy()
    return np.column_stack([lr_p, lr_s, inv, D]), inv


@dataclass
class OffboardAvailabilityModel:
    clf: object = None
    F: dict = None
    n_train: int = 0

    def fit(self, X, inv, pick):
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(pick, float)
        drafted = np.isfinite(y) & (y <= N_PICKS)
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, drafted.astype(int))
        grid = np.arange(1, N_PICKS + 1)
        allp = np.sort(y[drafted])
        self.F = {}
        for g in (0.0, 1.0):
            pk = np.sort(y[drafted & (inv == g)])
            pk = pk if len(pk) >= 10 else allp
            self.F[g] = np.searchsorted(pk, grid, side="right") / len(pk)
        self.n_train = len(y)
        return self

    def survival(self, X, inv, n):
        p = self.clf.predict_proba(X)[:, 1]
        F = np.vstack([self.F[float(g)] for g in inv])
        n = int(np.clip(n, 0, N_PICKS))
        return 1 - p * (F[:, n - 1] if n >= 1 else 0.0), p

    def p_available(self, X, inv, current, nxt):
        if nxt <= current + 1:
            return np.ones(len(inv))
        s_next, _ = self.survival(X, inv, nxt - 1)
        s_now, _ = self.survival(X, inv, current)
        return s_next / np.clip(s_now, 1e-12, None)


def fit_offboard(projections: pd.DataFrame, bridge: pd.DataFrame, oof: pd.DataFrame,
                 classes: tuple | None = None):
    """Train on 8d out-of-fold predictions for off-board players, classes
    (draft years) 2015-22, or `classes` (e.g. as of a past draft)."""
    O = oof[oof["hasan_rank"].isna()].drop_duplicates("player_id", keep="last")
    inv = projections.drop_duplicates("player_id").set_index("player_id")["combine_invited"]
    O = O.assign(cls=O["last_college_season_year"].astype(int) + 1, inv=O["player_id"].map(inv))
    O = O[O["cls"].between(*(classes or OFF_CLASSES))]
    drafted = bridge[bridge["overall_pick"].notna()].drop_duplicates("player_id").set_index("player_id")
    X, iv = _off_features(O, "cls", "p_nfl", "score", "pos_group", "inv")
    return OffboardAvailabilityModel().fit(X, iv, O["player_id"].map(drafted["overall_pick"]))


def offboard_features(pool: pd.DataFrame):
    """Features for one class's off-board draft pool (step 10 projections rows).
    Pass the WHOLE off-board pool: ranks are computed within it."""
    return _off_features(pool.assign(_c=0), "_c", "offboard_p_nfl", "offboard_score",
                         "position_group", "combine_invited")
