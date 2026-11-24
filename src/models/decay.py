"""
Exponential Decay Weighting and Parameter Optimization (Game-Level)

For each stat, the decay parameter β controls how quickly past observations
lose influence:

    weight_g = β ^ (days between game_g and reference_date)

β near 1.0 (e.g. 0.9999) → minimal recency bias, uses full career history
β near 0.995 → halves influence every ~139 days (roughly one season)

Ported from cdarko/src/models/decay.py, replacing year-indexed decay with
day-indexed decay. Key difference: weights are β^days, not β^seasons.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DEFAULT_BETA, KALMAN_REF_SNAPS,
    RATE_STAT_ATTEMPT_COL, KALMAN_REF_ATTEMPTS,
)

logger = logging.getLogger(__name__)

_NS_PER_DAY = 86_400 * 10 ** 9


# ── Decay Weight Computation ───────────────────────────────────────────────────

def compute_decay_weights(dates: pd.Series,
                           beta: float,
                           reference_date: Optional[pd.Timestamp] = None,
                           ) -> np.ndarray:
    """
    Compute exponential decay weights for game dates.

        weight_i = beta ^ (days between reference_date and dates[i])

    Returns numpy array of weights in [0, 1].
    """
    dates = pd.to_datetime(dates)
    if reference_date is None:
        reference_date = dates.max()
    reference_date = pd.Timestamp(reference_date)
    days_from_ref = (reference_date - dates).dt.days.clip(lower=0).values
    return beta ** days_from_ref


# ── Vectorized EWMA (analogous to cdarko's ewma_time_series_fast) ─────────────

def ewma_time_series_fast(
        player_games: pd.DataFrame,
        stats: list[str],
        betas: dict[str, float],
        prior_means: Optional[dict[str, float]] = None,
        prior_weight: Union[float, dict[str, float]] = 5.0,
) -> pd.DataFrame:
    """
    Vectorized O(n) EWMA for all stats in one pass over a player's games.

    For each game i:
        ewma_i = Σ_{j≤i} (val_j · decay_ij · game_weight_j) / Σ_{j≤i} (decay_ij · game_weight_j)

    where:
        decay_ij = β ^ (days between game_j and game_i)
        game_weight_j = snaps_j / REF_SNAPS  (counting stats)
                       = att_j / ref_att     (rate stats)

    A Bayesian prior (prior_means, prior_weight) is initialised at day 0
    to prevent extreme first-game values from dominating for low-sample players.
    Identical approach to cdarko's ewma_time_series_fast.

    Args:
        player_games:  Single-player DataFrame sorted by game_date, with
                       adjusted_{stat} and snaps / attempt columns.
        stats:         Stats to process.
        betas:         Per-stat decay parameter β (per-day).
        prior_means:   Per-stat prior mean initialisation (default 0).
        prior_weight:  Equivalent number of reference-weight observations the
                       prior represents. R/P₀ from Kalman, clamped [1, 20].

    Returns player_games with added ewma_{stat} and ewma_var_{stat} columns.
    """
    games = player_games.sort_values("game_date").copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    dates_ns = games["game_date"].values.astype("int64")
    days = (dates_ns - dates_ns[0]) // _NS_PER_DAY   # days since first game, shape (n,)

    for stat in stats:
        adj_col = f"adjusted_{stat}"
        if adj_col not in games.columns:
            continue

        beta = betas.get(stat, DEFAULT_BETA.get(stat, 0.999))
        vals = games[adj_col].values.astype(float)
        valid = ~np.isnan(vals)

        # Stable log-weight computation (shift so max ≤ 700, float64 safe)
        log_decay = -days.astype(float) * np.log(max(beta, 1e-10))
        if log_decay.size > 0 and log_decay.max() > 700.0:
            log_decay -= log_decay.max() - 700.0
        inv_decay = np.exp(log_decay)   # β^(-d_j) — monotone increasing with j

        # Game weights: snap-based for counting stats, attempt-based for rate stats
        if stat in RATE_STAT_ATTEMPT_COL:
            att_col = RATE_STAT_ATTEMPT_COL[stat]
            ref_att = KALMAN_REF_ATTEMPTS.get(stat, 5.0)
            if att_col in games.columns:
                att_vals = games[att_col].fillna(0.0).clip(lower=0.5).values.astype(float)
                game_weights = att_vals / ref_att
            else:
                game_weights = None
        elif "snaps" in games.columns:
            snap_vals = games["snaps"].fillna(KALMAN_REF_SNAPS).clip(lower=1.0).values.astype(float)
            game_weights = snap_vals / KALMAN_REF_SNAPS
        else:
            game_weights = None

        if game_weights is not None:
            inv_decay_w = inv_decay * game_weights
        else:
            inv_decay_w = inv_decay

        # Bayesian prior initialisation
        _pw = (prior_weight.get(stat, 5.0)
               if isinstance(prior_weight, dict) else prior_weight)
        if prior_means is not None and stat in prior_means:
            _pv = float(prior_means[stat])
            _prior_num = _pv * _pw
            _prior_sq  = _pv ** 2 * _pw
            _prior_den = _pw
        else:
            _prior_num = 0.0
            _prior_sq  = 0.0
            _prior_den = 0.0

        cum_num = _prior_num + np.cumsum(np.where(valid, vals * inv_decay_w, 0.0))
        cum_den = _prior_den + np.cumsum(np.where(valid, inv_decay_w, 0.0))

        with np.errstate(invalid="ignore", divide="ignore"):
            estimates = cum_num / cum_den
        estimates[cum_den == 0.0] = np.nan
        games[f"ewma_{stat}"] = estimates

        # EWMA variance
        cum_num_sq = _prior_sq + np.cumsum(np.where(valid, vals ** 2 * inv_decay_w, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            ewma_sq = cum_num_sq / cum_den
        ewma_sq[cum_den == 0.0] = np.nan
        ewma_var = np.maximum(ewma_sq - estimates ** 2, 0.0)
        games[f"ewma_var_{stat}"] = ewma_var

    return games


# ── Vectorized EWMA Across All Players (eliminates per-player Python loop) ─────

def ewma_all_players_fast(
        all_games: pd.DataFrame,
        stats: list[str],
        betas: dict[str, float],
        prior_means: Optional[dict[str, float]] = None,
        prior_weight: Union[float, dict[str, float]] = 5.0,
) -> pd.DataFrame:
    """
    Vectorized EWMA for all players in one pass.

    Equivalent to calling ewma_time_series_fast per player but ~50-100x faster
    because it uses pandas groupby cumsum instead of a Python loop over 80K players.

    all_games must be sorted by (player_id, game_date) with a RangeIndex.
    """
    df = all_games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    dates_ns = df["game_date"].values.astype(np.int64)

    # Days since each player's first game (for inv_decay computation)
    player_first_ns = df.groupby("player_id", sort=False)["game_date"].transform("min")
    days_since_first = ((df["game_date"] - player_first_ns).dt.days.clip(lower=0)).values.astype(float)

    new_cols: dict[str, np.ndarray] = {}

    for stat in stats:
        adj_col = f"adjusted_{stat}"
        if adj_col not in df.columns:
            continue

        beta = betas.get(stat, DEFAULT_BETA.get(stat, 0.999))

        # inv_decay = β^(-days_since_first): more recent games get higher weight
        # Negative sign on days so that recent games (large day) have larger inv_decay
        # Overflow correction is per-player: subtract each player's own max so
        # the relative weights within a player are preserved (avoids global shift
        # that would zero out players with smaller day spans).
        log_decay_raw = -days_since_first * np.log(max(beta, 1e-10))
        player_max_log = (pd.Series(log_decay_raw, index=df.index)
                          .groupby(df["player_id"].values, sort=False)
                          .transform("max").values)
        log_decay_raw = log_decay_raw - np.maximum(player_max_log - 700.0, 0.0)
        inv_decay = np.exp(log_decay_raw)

        # Game weights (snap-based or attempt-based)
        if stat in RATE_STAT_ATTEMPT_COL:
            att_col = RATE_STAT_ATTEMPT_COL[stat]
            ref_att = KALMAN_REF_ATTEMPTS.get(stat, 5.0)
            if att_col in df.columns:
                att_vals = df[att_col].fillna(0.0).clip(lower=0.5).values.astype(float)
                inv_decay_w = inv_decay * (att_vals / ref_att)
            else:
                inv_decay_w = inv_decay
        elif "snaps" in df.columns:
            snap_vals = df["snaps"].fillna(KALMAN_REF_SNAPS).clip(lower=1.0).values.astype(float)
            inv_decay_w = inv_decay * (snap_vals / KALMAN_REF_SNAPS)
        else:
            inv_decay_w = inv_decay

        # Prior parameters per stat
        _pw = prior_weight.get(stat, 5.0) if isinstance(prior_weight, dict) else float(prior_weight)
        if prior_means is not None and stat in prior_means:
            _pv = float(prior_means[stat])
            prior_num = _pv * _pw
            prior_sq = _pv ** 2 * _pw
        else:
            prior_num = 0.0
            prior_sq = 0.0

        vals = df[adj_col].values.astype(float)
        valid = ~np.isnan(vals)

        weighted_vals = np.where(valid, vals * inv_decay_w, 0.0)
        weighted_sq   = np.where(valid, vals ** 2 * inv_decay_w, 0.0)
        valid_w       = np.where(valid, inv_decay_w, 0.0)

        # Cumulative sums per player using groupby
        tmp = pd.DataFrame({
            "player_id": df["player_id"].values,
            "_num": weighted_vals,
            "_sq": weighted_sq,
            "_den": valid_w,
        })
        cum_num = tmp.groupby("player_id", sort=False)["_num"].cumsum().values + prior_num
        cum_sq  = tmp.groupby("player_id", sort=False)["_sq"].cumsum().values + prior_sq
        cum_den = tmp.groupby("player_id", sort=False)["_den"].cumsum().values + _pw

        with np.errstate(invalid="ignore", divide="ignore"):
            estimates = cum_num / cum_den
        estimates[cum_den == 0.0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            ewma_sq = cum_sq / cum_den
        ewma_sq[cum_den == 0.0] = np.nan
        ewma_var = np.maximum(ewma_sq - estimates ** 2, 0.0)

        new_cols[f"ewma_{stat}"] = estimates
        new_cols[f"ewma_var_{stat}"] = ewma_var

    for col, arr in new_cols.items():
        df[col] = arr

    return df


# ── Beta Optimization (next-game prediction error, analogous to cdarko) ────────

def _prediction_error_for_beta(beta: float,
                                player_games: pd.DataFrame,
                                stat: str,
                                min_history: int = 5) -> float:
    """
    Compute mean absolute prediction error for one player using leave-one-out
    (next-game prediction).
    """
    adj_col = f"adjusted_{stat}"
    games = player_games.sort_values("game_date").copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    vals = games[adj_col].values
    dates = games["game_date"].values

    use_snaps = "snaps" in games.columns
    use_att = stat in RATE_STAT_ATTEMPT_COL and RATE_STAT_ATTEMPT_COL[stat] in games.columns
    ref_snaps = KALMAN_REF_SNAPS
    ref_att = KALMAN_REF_ATTEMPTS.get(stat, 5.0)

    errors: list[float] = []
    for i in range(min_history, len(games)):
        history_vals = vals[:i]
        history_dates = pd.Series(dates[:i])
        target_date = pd.Timestamp(dates[i])
        target_val = vals[i]

        if np.isnan(target_val):
            continue

        weights = compute_decay_weights(history_dates, beta, target_date)

        if use_att:
            att_col = RATE_STAT_ATTEMPT_COL[stat]
            att_vals = games[att_col].values[:i].astype(float)
            game_wts = np.clip(att_vals, 0.5, None) / ref_att
            weights = weights * game_wts
        elif use_snaps:
            snap_vals = games["snaps"].values[:i].astype(float)
            game_wts = np.clip(snap_vals, 1.0, None) / ref_snaps
            weights = weights * game_wts

        valid_mask = ~np.isnan(history_vals)
        if not valid_mask.any():
            continue

        predicted = np.average(history_vals[valid_mask], weights=weights[valid_mask])
        errors.append(abs(target_val - predicted))

    return float(np.mean(errors)) if errors else float("inf")


def optimize_beta_for_stat(adjusted_games: pd.DataFrame,
                            stat: str,
                            n_players: int = 500,
                            min_history: int = 5,
                            bounds: tuple[float, float] = (0.990, 1.000),
                            ) -> float:
    """
    Optimize the decay parameter β for a stat by minimizing out-of-sample
    next-game MAE. Uses a random subsample of n_players for efficiency.
    """
    adj_col = f"adjusted_{stat}"
    if adj_col not in adjusted_games.columns:
        logger.warning("Column %s not found; using default beta", adj_col)
        return DEFAULT_BETA.get(stat, 0.999)

    player_counts = (adjusted_games.dropna(subset=[adj_col])
                     .groupby("player_id").size())
    eligible = player_counts[player_counts >= min_history + 5].index.tolist()

    if not eligible:
        logger.warning("No eligible players for beta opt of %s", stat)
        return DEFAULT_BETA.get(stat, 0.999)

    rng = np.random.default_rng(42)
    sample = rng.choice(eligible, size=min(n_players, len(eligible)), replace=False)
    subset = adjusted_games[adjusted_games["player_id"].isin(sample)].copy()

    def objective(beta: float) -> float:
        errors: list[float] = []
        for _, player_df in subset.groupby("player_id"):
            e = _prediction_error_for_beta(beta, player_df, stat, min_history)
            if not np.isinf(e):
                errors.append(e)
        return float(np.mean(errors)) if errors else float("inf")

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 50},
    )

    opt_beta = float(result.x)
    logger.info("Stat %s: optimal beta=%.5f (MAE=%.4f)", stat, opt_beta, result.fun)
    return opt_beta


def optimize_all_betas(adjusted_games: pd.DataFrame,
                        stats: list[str],
                        n_players: int = 500,
                        bounds: tuple[float, float] = (0.990, 1.000),
                        ) -> dict[str, float]:
    """Optimize decay parameters for all stats. Returns stat → optimal β.

    Stats are independent (each runs its own scalar minimization over a
    sampled-player subset), so we run them in parallel. n_jobs=-1 uses all
    cores; first-run cost on a fresh retune drops from ~5–10 min serial to
    ~1–2 min on a typical M-series Mac.
    """
    from joblib import Parallel, delayed

    def _safe(stat: str) -> tuple[str, float]:
        logger.info("Optimizing beta for stat: %s", stat)
        try:
            return stat, optimize_beta_for_stat(
                adjusted_games, stat, n_players=n_players, bounds=bounds,
            )
        except Exception as exc:
            logger.error("Beta optimization failed for %s: %s", stat, exc)
            return stat, DEFAULT_BETA.get(stat, 0.999)

    results = Parallel(n_jobs=-1)(delayed(_safe)(stat) for stat in stats)
    return dict(results)


# ── Role-Outcome Beta Calibration ─────────────────────────────────────────────

def _compute_terminal_ewma_reference(
        bridge_adj: pd.DataFrame,
        stat: str,
        beta: float,
        date_col: str = "game_date",
) -> dict:
    """
    Reference (per-player loop) implementation of the terminal EWMA.

    Kept for equivalence testing of the vectorized `_compute_terminal_ewma`.
    For each bridge player compute EWMA over their full game history and
    return the terminal (last-game) EWMA value.

    Snap-weighted for counting stats; attempt-weighted for rate stats.
    """
    adj_col = f"adjusted_{stat}"
    if adj_col not in bridge_adj.columns:
        return {}

    terminal: dict = {}
    for pid, group in bridge_adj.groupby("player_id"):
        group = group.sort_values(date_col)
        dates_ns = pd.to_datetime(group[date_col]).values.astype("int64")
        days = (dates_ns - dates_ns[0]) // _NS_PER_DAY

        vals = group[adj_col].values.astype(float)
        valid = ~np.isnan(vals)
        if not valid.any():
            continue

        # Game weights
        if stat in RATE_STAT_ATTEMPT_COL:
            att_col = RATE_STAT_ATTEMPT_COL[stat]
            ref_att = KALMAN_REF_ATTEMPTS.get(stat, 5.0)
            if att_col in group.columns:
                att = group[att_col].fillna(0.0).clip(lower=0.5).values.astype(float)
                weights = att / ref_att
            else:
                weights = np.ones(len(vals))
        elif "snaps" in group.columns:
            snaps = group["snaps"].fillna(KALMAN_REF_SNAPS).clip(lower=1.0).values.astype(float)
            weights = snaps / KALMAN_REF_SNAPS
        else:
            weights = np.ones(len(vals))

        log_w = -days.astype(float) * np.log(max(beta, 1e-10))
        if log_w.size > 0 and log_w.max() > 700.0:
            log_w -= log_w.max() - 700.0
        inv_decay = np.exp(log_w)
        w = weights * inv_decay

        cum_num = np.cumsum(np.where(valid, vals * w, 0.0))
        cum_den = np.cumsum(np.where(valid, w, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ewma = cum_num / cum_den
        ewma[cum_den == 0.0] = np.nan

        final_val = ewma[-1]
        if not np.isnan(final_val):
            terminal[pid] = float(final_val)

    return terminal


class _TerminalEwmaArrays:
    """
    Beta-independent flattened arrays for the vectorized terminal EWMA.

    Built once per (bridge_adj, stat); each optimizer eval then only has to
    compute `beta ** days_from_terminal` and two segmented sums.

    Attributes:
        pids:          (n_players,) player id per segment.
        starts:        (n_players,) start index of each player's segment
                       (np.add.reduceat boundaries).
        vals:          (n_obs,) adjusted stat values (float, may contain NaN).
        game_weights:  (n_obs,) snap/attempt game weights (beta-independent).
        days_from_terminal: (n_obs,) integer days between each observation and
                       that player's terminal (last) game date, using the same
                       floored day arithmetic as the reference implementation.
        valid:         (n_obs,) mask of non-NaN vals.
    """

    __slots__ = ("pids", "starts", "vals", "game_weights",
                 "days_from_terminal", "valid")

    def __init__(self, pids, starts, vals, game_weights,
                 days_from_terminal, valid):
        self.pids = pids
        self.starts = starts
        self.vals = vals
        self.game_weights = game_weights
        self.days_from_terminal = days_from_terminal
        self.valid = valid


def _prepare_terminal_ewma_arrays(
        bridge_adj: pd.DataFrame,
        stat: str,
        date_col: str = "game_date",
) -> Optional[_TerminalEwmaArrays]:
    """
    Precompute the beta-independent arrays for `_compute_terminal_ewma`.

    Returns None when the adjusted column is missing or there are no rows
    (callers treat None as "terminal dict is empty").
    """
    adj_col = f"adjusted_{stat}"
    if adj_col not in bridge_adj.columns or len(bridge_adj) == 0:
        return None

    df = bridge_adj.sort_values(["player_id", date_col], kind="mergesort")

    pid_arr = df["player_id"].values
    starts = np.flatnonzero(np.r_[True, pid_arr[1:] != pid_arr[:-1]])
    ends = np.r_[starts[1:], len(pid_arr)]
    counts = ends - starts
    pids = pid_arr[starts]

    # Same floored day arithmetic as the reference: days since the player's
    # first game; days_from_terminal = terminal_days - days_i (exact integers,
    # so beta ** days_from_terminal matches the reference weight ratios).
    dates_ns = pd.to_datetime(df[date_col]).values.astype("int64")
    first_ns = np.repeat(dates_ns[starts], counts)
    days = (dates_ns - first_ns) // _NS_PER_DAY
    terminal_days = np.repeat(days[ends - 1], counts)
    days_from_terminal = (terminal_days - days).astype(float)

    vals = df[adj_col].values.astype(float)
    valid = ~np.isnan(vals)

    # Game weights (beta-independent) — identical branching to the reference
    if stat in RATE_STAT_ATTEMPT_COL:
        att_col = RATE_STAT_ATTEMPT_COL[stat]
        ref_att = KALMAN_REF_ATTEMPTS.get(stat, 5.0)
        if att_col in df.columns:
            att = df[att_col].fillna(0.0).clip(lower=0.5).values.astype(float)
            game_weights = att / ref_att
        else:
            game_weights = np.ones(len(vals))
    elif "snaps" in df.columns:
        snaps = df["snaps"].fillna(KALMAN_REF_SNAPS).clip(lower=1.0).values.astype(float)
        game_weights = snaps / KALMAN_REF_SNAPS
    else:
        game_weights = np.ones(len(vals))

    return _TerminalEwmaArrays(pids=pids, starts=starts, vals=vals,
                               game_weights=game_weights,
                               days_from_terminal=days_from_terminal,
                               valid=valid)


def _terminal_ewma_from_arrays(pre: _TerminalEwmaArrays, beta: float) -> dict:
    """
    Vectorized terminal EWMA for all players in one shot.

    The terminal EWMA is a ratio of weighted sums, so it is invariant to a
    common per-player scale factor on the weights. The reference uses
    beta^(-days_since_first) (with an overflow shift); here we use
    beta^(days_from_terminal), which is the same weights rescaled by
    beta^(terminal_days) — identical ratios, no overflow for beta ≤ 1.
    """
    base = max(beta, 1e-10)
    w = (base ** pre.days_from_terminal) * pre.game_weights

    num = np.add.reduceat(np.where(pre.valid, pre.vals * w, 0.0), pre.starts)
    den = np.add.reduceat(np.where(pre.valid, w, 0.0), pre.starts)

    with np.errstate(divide="ignore", invalid="ignore"):
        ewma = num / den
    ewma[den == 0.0] = np.nan

    keep = ~np.isnan(ewma)
    return {pid: float(v) for pid, v in zip(pre.pids[keep], ewma[keep])}


def _compute_terminal_ewma(
        bridge_adj: pd.DataFrame,
        stat: str,
        beta: float,
        date_col: str = "game_date",
        precomputed: Optional[_TerminalEwmaArrays] = None,
) -> dict:
    """
    Vectorized terminal EWMA (semantics identical to
    `_compute_terminal_ewma_reference`).

    Pass `precomputed` from `_prepare_terminal_ewma_arrays` to skip the
    beta-independent grouping work inside optimizer loops.
    """
    if precomputed is None:
        precomputed = _prepare_terminal_ewma_arrays(bridge_adj, stat, date_col)
    if precomputed is None:
        return {}
    return _terminal_ewma_from_arrays(precomputed, beta)


def _neg_role_pearson(
        beta: float,
        bridge_adj: pd.DataFrame,
        role_labels: pd.Series,
        stat: str,
        eval_pids: Optional[set] = None,
        impact_labels: Optional[pd.Series] = None,
        role_weight: float = 0.5,
        impact_weight: float = 0.5,
        precomputed: Optional[_TerminalEwmaArrays] = None,
) -> float:
    """
    Return negative weighted combination of r(terminal EWMA, role_target) and
    r(terminal EWMA, impact_target). Minimizing this maximizes correlation.
    Returns 0.0 if fewer than 30 matched players.
    """
    from scipy.stats import pearsonr

    terminal = _compute_terminal_ewma(bridge_adj, stat, beta,
                                      precomputed=precomputed)

    ewma_vals: list[float] = []
    role_vals: list[float] = []
    impact_vals: list[float] = []
    for pid, ev in terminal.items():
        if eval_pids is not None and pid not in eval_pids:
            continue
        if pid in role_labels.index:
            rv = role_labels.loc[pid]
            # Guard against duplicate index entries returning a Series
            if isinstance(rv, pd.Series):
                rv = rv.iloc[0]
            if pd.notna(rv):
                ewma_vals.append(ev)
                role_vals.append(float(rv))
                iv = float("nan")
                if impact_labels is not None and pid in impact_labels.index:
                    iv_raw = impact_labels.loc[pid]
                    if isinstance(iv_raw, pd.Series):
                        iv_raw = iv_raw.iloc[0]
                    iv = float(iv_raw) if pd.notna(iv_raw) else float("nan")
                impact_vals.append(iv)

    if len(ewma_vals) < 30:
        return 0.0

    r_role = 0.0
    if role_weight > 0 and len(role_vals) >= 30:
        r, _ = pearsonr(ewma_vals, role_vals)
        r_role = r if np.isfinite(r) else 0.0

    r_impact = 0.0
    if impact_weight > 0 and impact_labels is not None:
        valid = [(e, iv) for e, iv in zip(ewma_vals, impact_vals) if np.isfinite(iv)]
        if len(valid) >= 30:
            e_imp, i_imp = zip(*valid)
            r, _ = pearsonr(list(e_imp), list(i_imp))
            r_impact = r if np.isfinite(r) else 0.0

    return -(role_weight * r_role + impact_weight * r_impact)


def optimize_betas_for_role(
        adjusted_games: pd.DataFrame,
        role_labels: pd.DataFrame,
        stats: list[str],
        cal_season_cutoff: int = 2020,
        bounds: tuple[float, float] = (0.990, 1.0),
        role_weight: float = 0.5,
        impact_weight: float = 0.5,
) -> tuple[dict[str, float], "pd.DataFrame"]:
    """
    Optimize EWMA decay betas using bridge player NFL role prediction as the objective.

    For each stat finds β ∈ bounds that maximises:
        role_weight × r(terminal EWMA, role_target)
      + impact_weight × r(terminal EWMA, impact_target)

    Args:
        adjusted_games:      Full adjusted_gamelogs.parquet.
        role_labels:         nfl_draft_labels with player_id, role_target,
                             impact_target, last_college_season_year.
        stats:               Stats to calibrate.
        cal_season_cutoff:   Players with last_college_season_year ≤ cutoff are cal set.
        bounds:              (lower, upper) bounds for β search.
        role_weight:         Weight on r(EWMA, role_target).
        impact_weight:       Weight on r(EWMA, impact_target).

    Returns:
        betas:   dict stat → optimal β
        summary: DataFrame with columns stat, beta, r_cal, r_val, n_cal, n_val
    """
    # Deduplicate on player_id (keep first) before indexing
    labels_dedup = (role_labels.dropna(subset=["player_id"])
                    .sort_values("player_id")
                    .drop_duplicates("player_id")
                    .set_index("player_id"))

    bridge_pids = set(labels_dedup.index)
    bridge_adj = adjusted_games[adjusted_games["player_id"].isin(bridge_pids)].copy()
    bridge_adj["game_date"] = pd.to_datetime(bridge_adj["game_date"])

    cal_mask = labels_dedup["last_college_season_year"] <= cal_season_cutoff
    cal_pids = set(labels_dedup.index[cal_mask])
    val_pids  = set(labels_dedup.index[~cal_mask])

    cal_adj = bridge_adj[bridge_adj["player_id"].isin(cal_pids)]
    val_adj = bridge_adj[bridge_adj["player_id"].isin(val_pids)]
    cal_role = labels_dedup.loc[labels_dedup.index.isin(cal_pids), "role_target"]
    val_role = labels_dedup.loc[labels_dedup.index.isin(val_pids), "role_target"]

    cal_impact: Optional[pd.Series] = None
    val_impact: Optional[pd.Series] = None
    if impact_weight > 0 and "impact_target" in labels_dedup.columns:
        cal_impact = labels_dedup.loc[labels_dedup.index.isin(cal_pids), "impact_target"]
        val_impact = labels_dedup.loc[labels_dedup.index.isin(val_pids), "impact_target"]

    logger.info(
        "Beta role-calibration: %d cal players (≤%d), %d val players (>%d)",
        len(cal_pids), cal_season_cutoff, len(val_pids), cal_season_cutoff,
    )

    betas: dict[str, float] = {}
    rows: list[dict] = []

    for stat in stats:
        logger.info("Role-calibrating beta for stat: %s", stat)
        # Precompute the beta-independent per-player arrays once per stat —
        # each of the ~50 optimizer evals then only does a vectorized
        # beta ** days + two np.add.reduceat segmented sums.
        pre_cal = _prepare_terminal_ewma_arrays(cal_adj, stat)
        pre_val = _prepare_terminal_ewma_arrays(val_adj, stat)
        try:
            result = minimize_scalar(
                _neg_role_pearson,
                bounds=bounds,
                method="bounded",
                args=(cal_adj, cal_role, stat, None, cal_impact,
                      role_weight, impact_weight, pre_cal),
                options={"xatol": 1e-4, "maxiter": 50},
            )
            opt_beta = float(result.x)
        except Exception as exc:
            logger.error("Role beta calibration failed for %s: %s", stat, exc)
            opt_beta = DEFAULT_BETA.get(stat, 0.999)

        r_cal = -_neg_role_pearson(opt_beta, cal_adj, cal_role, stat,
                                   impact_labels=cal_impact,
                                   role_weight=role_weight, impact_weight=impact_weight,
                                   precomputed=pre_cal)
        r_val = -_neg_role_pearson(opt_beta, val_adj, val_role, stat,
                                   impact_labels=val_impact,
                                   role_weight=role_weight, impact_weight=impact_weight,
                                   precomputed=pre_val)
        n_cal = int(cal_role.notna().sum())
        n_val = int(val_role.notna().sum())

        betas[stat] = opt_beta
        rows.append({"stat": stat, "beta": opt_beta,
                     "r_cal": round(r_cal, 4), "r_val": round(r_val, 4),
                     "n_cal": n_cal, "n_val": n_val})
        logger.info("  %s: β=%.5f  r_cal=%.4f  r_val=%.4f  n_cal=%d  n_val=%d",
                    stat, opt_beta, r_cal, r_val, n_cal, n_val)

    return betas, pd.DataFrame(rows)
