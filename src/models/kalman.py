"""
Game-Level Kalman Filter for Player Skill Estimation

Estimates latent true talent θ from noisy per-game adjusted residuals.
Ported from cdarko/src/models/kalman.py with football-specific adjustments:
  - Process noise per day (handles off-season gaps automatically)
  - Snap count / attempt count scaling of observation noise
  - Same age-based Q multiplier and transfer shock as cdarko

Performance: _kalman_jit_core is JIT-compiled with Numba when available
(~20–50x speedup vs pure Python). Falls back to pure Python transparently.
"""

import gc
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    KALMAN_PRIOR_MEAN, KALMAN_PRIOR_VAR, KALMAN_PRIOR_VAR_BY_STAT,
    KALMAN_PROCESS_NOISE_PER_DAY, KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT,
    KALMAN_OBS_NOISE, KALMAN_OBS_NOISE_BY_STAT,
    KALMAN_REF_SNAPS, RATE_STAT_ATTEMPT_COL, KALMAN_REF_ATTEMPTS,
    KALMAN_PRE2016_NOISE_MULTIPLIER,
    KALMAN_OFFSEASON_GAP_DAYS, KALMAN_PROCESS_NOISE_OFFSEASON_MULTIPLIER,
    TRANSFER_P_MULTIPLIER,
    AGE_Q_REFERENCE_AGE, AGE_Q_HALFLIFE_YEARS, AGE_Q_MAX_MULTIPLIER,
    KALMAN_INNOVATION_CLIP_SIGMA,
)

logger = logging.getLogger(__name__)

_NS_PER_DAY = 86_400 * 10 ** 9

# ── Optional Numba JIT ─────────────────────────────────────────────────────────

try:
    import numba as nb
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False
    logger.debug("numba not installed — Kalman filter will use pure Python loop")


# ── Kalman State ───────────────────────────────────────────────────────────────

@dataclass
class KalmanState:
    """Scalar Kalman filter state for one player / one stat."""
    mean: float = KALMAN_PRIOR_MEAN
    var: float = KALMAN_PRIOR_VAR
    last_date: Optional[pd.Timestamp] = None
    school: Optional[str] = None
    n_obs: int = 0


# ── Core Kalman Operations ─────────────────────────────────────────────────────

def kalman_predict(state: KalmanState, days_elapsed: int,
                   process_noise_per_day: float = KALMAN_PROCESS_NOISE_PER_DAY
                   ) -> KalmanState:
    """Prediction step: advance state by days_elapsed days."""
    new_var = state.var + process_noise_per_day * max(days_elapsed, 0)
    return KalmanState(
        mean=state.mean,
        var=new_var,
        last_date=state.last_date,
        school=state.school,
        n_obs=state.n_obs,
    )


def kalman_update(state: KalmanState, observation: float,
                  obs_noise: float, weight: float = 1.0,
                  innovation_clip_sigma: Optional[float] = KALMAN_INNOVATION_CLIP_SIGMA,
                  ) -> KalmanState:
    """
    Update step: incorporate a new observation.

    Effective observation noise is scaled inversely with weight:
        R_eff = obs_noise / weight
    High-weight (recent) observations update the estimate more aggressively.
    """
    if np.isnan(observation) or weight <= 0:
        return state

    R_eff = obs_noise / max(weight, 1e-10)
    K = state.var / (state.var + R_eff)
    innovation = observation - state.mean

    if innovation_clip_sigma is not None:
        clip = innovation_clip_sigma * np.sqrt(R_eff)
        innovation = float(np.clip(innovation, -clip, clip))

    new_mean = state.mean + K * innovation
    new_var = (1.0 - K) * state.var

    return KalmanState(
        mean=new_mean,
        var=new_var,
        last_date=state.last_date,
        school=state.school,
        n_obs=state.n_obs + 1,
    )


def apply_transfer_shock(state: KalmanState,
                          multiplier: float = TRANSFER_P_MULTIPLIER) -> KalmanState:
    """Increase uncertainty when player changes school."""
    return KalmanState(
        mean=state.mean,
        var=state.var * multiplier,
        last_date=state.last_date,
        school=state.school,
        n_obs=state.n_obs,
    )


def _age_q_multiplier(age: float,
                       halflife: float = AGE_Q_HALFLIFE_YEARS,
                       max_mult: float = AGE_Q_MAX_MULTIPLIER) -> float:
    """Younger players get higher process noise (more development uncertainty)."""
    multiplier = 2.0 ** ((AGE_Q_REFERENCE_AGE - age) / halflife)
    return max(1.0, min(max_mult, multiplier))


# ── JIT-Compatible Inner Loop ──────────────────────────────────────────────────
# _kalman_jit_core_impl contains the pure Kalman iteration logic.
# It takes only typed numpy arrays (no None, no strings) so Numba can compile it.
#
# transfer_mask:  bool array, True at game i when player changed schools (pre-computed)
# snap_arr/att_arr: float arrays (np.nan sentinel when not available)
# use_snap/use_att: flags replacing None-checks

def _kalman_jit_core_impl(
        obs_arr,       # float64[:]
        dates_ns,      # int64[:]
        weight_arr,    # float64[:]
        transfer_mask, # bool[:]
        snap_arr,      # float64[:] — np.nan sentinel when not available
        att_arr,       # float64[:] — np.nan sentinel when not available
        age_arr,       # float64[:] — age at each game (np.nan when unknown)
        initial_mean,
        initial_var,
        process_noise_per_day,
        obs_noise,
        ref_att,
        ref_snaps,
        is_rate_stat,  # bool
        use_snap,      # bool
        use_att,       # bool
        transfer_p_mult,
        innovation_clip,   # float — ignored when clip_enabled=False
        clip_enabled,      # bool
        offseason_gap,     # float — days threshold for off-season gap
        offseason_mult,    # float — process noise multiplier for off-season
        age_ref,           # float — AGE_Q_REFERENCE_AGE
        age_halflife,      # float — AGE_Q_HALFLIFE_YEARS
        age_max_mult,      # float — AGE_Q_MAX_MULTIPLIER
):
    n = len(obs_arr)
    means = np.empty(n, dtype=np.float64)
    variances = np.empty(n, dtype=np.float64)
    mean = initial_mean
    var = initial_var
    last_date_ns = np.int64(0)

    for i in range(n):
        if i > 0 and transfer_mask[i]:
            var *= transfer_p_mult

        if i > 0:
            days = (dates_ns[i] - last_date_ns) / np.float64(86400000000000)
            if days < 0.0:
                days = 0.0
        else:
            days = 0.0

        # Two-rate process noise: off-season gaps grow uncertainty more slowly
        q_mult = offseason_mult if days > offseason_gap else np.float64(1.0)
        # Age-based process noise: younger players have higher development uncertainty
        age = age_arr[i]
        if not np.isnan(age) and age < age_ref:
            raw = np.float64(2.0) ** ((age_ref - age) / age_halflife)
            age_q = raw if raw < age_max_mult else age_max_mult
            if age_q < np.float64(1.0):
                age_q = np.float64(1.0)
            q_mult *= age_q

        var += process_noise_per_day * days * q_mult
        last_date_ns = dates_ns[i]

        obs = obs_arr[i]
        if not np.isnan(obs):
            w = weight_arr[i]
            if is_rate_stat and use_att:
                av = att_arr[i]
                if not np.isnan(av) and av > 0.5:
                    eff_r = obs_noise * (ref_att / av)
                else:
                    eff_r = obs_noise * (ref_att / 0.5)
            elif use_snap:
                sv = snap_arr[i]
                if not np.isnan(sv) and sv >= 1.0:
                    eff_r = obs_noise * (ref_snaps / sv)
                else:
                    eff_r = obs_noise * ref_snaps
            else:
                eff_r = obs_noise

            if w < 1e-10:
                w = 1e-10
            R_eff = eff_r / w
            K = var / (var + R_eff)
            innov = obs - mean
            if clip_enabled:
                clip_val = innovation_clip * (R_eff ** 0.5)
                if innov > clip_val:
                    innov = clip_val
                elif innov < -clip_val:
                    innov = -clip_val
            mean += K * innov
            var = (1.0 - K) * var

        means[i] = mean
        variances[i] = var

    return means, variances


if _USE_NUMBA:
    _kalman_jit_core = nb.jit(nopython=True, cache=True)(_kalman_jit_core_impl)
else:
    _kalman_jit_core = _kalman_jit_core_impl


# ── Full Player History Processing ─────────────────────────────────────────────

def _compute_transfer_mask(school_arr, n: int) -> np.ndarray:
    """
    Pre-compute a boolean mask marking games where the player changed schools.
    Kept in pure Python so string comparison never enters the JIT function.
    """
    mask = np.zeros(n, dtype=np.bool_)
    if school_arr is None:
        return mask
    prev = school_arr[0]
    for i in range(1, n):
        curr = school_arr[i]
        if prev is not None and curr is not None and prev != curr:
            mask[i] = True
        prev = curr
    return mask


def _kalman_player_fast(
        obs_arr: np.ndarray,
        dates_ns: np.ndarray,
        weight_arr: np.ndarray,
        school_arr,          # str array or None — used only to pre-compute transfer_mask
        snap_arr,            # float array or None
        att_arr,             # float array or None
        initial_mean: float,
        initial_var: float,
        process_noise_per_day: float,
        obs_noise: float,
        ref_att: float,
        is_rate_stat: bool,
        transfer_p_mult: float,
        innovation_clip_sigma: Optional[float],
        transfer_mask: Optional[np.ndarray] = None,  # pre-computed; computed here if None
        age_arr: Optional[np.ndarray] = None,        # float array of age-at-game or None
        age_q_halflife: float = AGE_Q_HALFLIFE_YEARS,
        age_q_max_mult: float = AGE_Q_MAX_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Kalman filter for one player / one stat.

    If transfer_mask is provided (pre-computed by the caller once per player),
    the school-change detection pass is skipped. This avoids redundant string
    comparisons when iterating over many stats for the same player.
    """
    n = len(obs_arr)

    if transfer_mask is None:
        transfer_mask = _compute_transfer_mask(school_arr, n)

    _snap = snap_arr if snap_arr is not None else np.full(n, np.nan)
    _att  = att_arr  if att_arr  is not None else np.full(n, np.nan)
    _age  = age_arr  if age_arr  is not None else np.full(n, np.nan)
    _clip = innovation_clip_sigma if innovation_clip_sigma is not None else 0.0

    return _kalman_jit_core(
        obs_arr, dates_ns, weight_arr, transfer_mask, _snap, _att, _age,
        initial_mean, initial_var,
        process_noise_per_day, obs_noise,
        ref_att, float(KALMAN_REF_SNAPS),
        bool(is_rate_stat),
        snap_arr is not None,
        att_arr is not None,
        transfer_p_mult,
        _clip,
        innovation_clip_sigma is not None,
        float(KALMAN_OFFSEASON_GAP_DAYS),
        float(KALMAN_PROCESS_NOISE_OFFSEASON_MULTIPLIER),
        float(AGE_Q_REFERENCE_AGE),
        float(age_q_halflife),
        float(age_q_max_mult),
    )


def run_kalman_for_player_with_decay(
        player_games: pd.DataFrame,
        stat: str,
        decay_beta: float,
        obs_noise: float,
        reference_date: Optional[pd.Timestamp] = None,
        process_noise_per_day: float = KALMAN_PROCESS_NOISE_PER_DAY,
        initial_var: Optional[float] = None,
        initial_mean: Optional[float] = None,
        innovation_clip_sigma: Optional[float] = KALMAN_INNOVATION_CLIP_SIGMA,
        age_col: Optional[str] = None,
        age_q_halflife: float = AGE_Q_HALFLIFE_YEARS,
        age_q_max_mult: float = AGE_Q_MAX_MULTIPLIER,
        transfer_p_mult: float = TRANSFER_P_MULTIPLIER,
        school_col: str = "school",
) -> pd.DataFrame:
    """
    Run Kalman filter with exponential decay weights for one player / one stat.
    """
    adj_col = f"adjusted_{stat}"
    if adj_col not in player_games.columns:
        raise ValueError(f"Column {adj_col} not in DataFrame")

    games = player_games.sort_values("game_date").copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    if reference_date is None:
        reference_date = games["game_date"].max()
    reference_date = pd.Timestamp(reference_date)

    dates_ns = games["game_date"].values.astype(np.int64)
    ref_ns = np.int64(reference_date.value)
    days_from_ref = np.maximum((ref_ns - dates_ns) / _NS_PER_DAY, 0.0)
    weight_arr = decay_beta ** days_from_ref

    is_rate_stat = stat in RATE_STAT_ATTEMPT_COL
    attempt_col = RATE_STAT_ATTEMPT_COL.get(stat, "")
    ref_att = KALMAN_REF_ATTEMPTS.get(stat, 5.0)

    obs_arr = games[adj_col].values.astype(float)
    school_arr = games[school_col].values if school_col in games.columns else None
    att_arr = games[attempt_col].values.astype(float) if (is_rate_stat and attempt_col and attempt_col in games.columns) else None
    snap_arr = games["snaps"].values.astype(float) if ("snaps" in games.columns and att_arr is None) else None
    age_arr = None
    if age_col and age_col in games.columns and "game_date" in games.columns:
        birth_years = games[age_col].values.astype(float)
        game_years = pd.to_datetime(games["game_date"]).dt.year.values.astype(float)
        age_arr = game_years - birth_years

    _init_var = initial_var if initial_var is not None else KALMAN_PRIOR_VAR_BY_STAT.get(stat, KALMAN_PRIOR_VAR)
    _init_mean = initial_mean if initial_mean is not None else KALMAN_PRIOR_MEAN

    means, variances = _kalman_player_fast(
        obs_arr, dates_ns, weight_arr, school_arr, snap_arr, att_arr,
        _init_mean, _init_var, process_noise_per_day, obs_noise,
        ref_att, is_rate_stat, transfer_p_mult, innovation_clip_sigma,
        age_arr=age_arr,
        age_q_halflife=age_q_halflife,
        age_q_max_mult=age_q_max_mult,
    )

    games[f"kalman_mean_{stat}"] = means
    games[f"kalman_var_{stat}"] = variances
    return games


# ── Batch Processing ───────────────────────────────────────────────────────────

def run_kalman_all_players(
        adjusted_games: pd.DataFrame,
        decay_betas: dict[str, float],
        obs_noises: Optional[dict[str, float]] = None,
        reference_date: Optional[pd.Timestamp] = None,
        stats: Optional[list[str]] = None,
        age_col: Optional[str] = None,
        age_q_halflife: float = AGE_Q_HALFLIFE_YEARS,
        age_q_max_mult: float = AGE_Q_MAX_MULTIPLIER,
        transfer_p_mult: float = TRANSFER_P_MULTIPLIER,
        school_col: str = "school",
        q_noise_by_stat: Optional[dict[str, float]] = None,
        prior_means: Optional[dict[str, float]] = None,
        prior_vars: Optional[dict[str, float]] = None,
        innovation_clip_sigma: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run Kalman filter for all players and all stats.

    adjusted_games must have: player_id, game_date, school (optional),
    adjusted_{stat} for each stat.

    Returns adjusted_games with kalman_mean_{stat} and kalman_var_{stat}
    columns added for each stat.
    """
    if obs_noises is None:
        obs_noises = KALMAN_OBS_NOISE_BY_STAT

    if stats is None:
        stats = [c.removeprefix("adjusted_") for c in adjusted_games.columns
                 if c.startswith("adjusted_")]

    _global_ref = pd.Timestamp(reference_date) if reference_date is not None else None

    adjusted_games.sort_values(["player_id", "game_date"], inplace=True)
    adjusted_games.reset_index(drop=True, inplace=True)

    n_rows = len(adjusted_games)

    # Pre-compute prior fill values
    _prior_means_arr: dict[str, float] = {}
    _prior_vars_arr: dict[str, float] = {}
    for stat in stats:
        _prior_means_arr[stat] = (prior_means.get(stat) if prior_means else None) or KALMAN_PRIOR_MEAN
        _prior_vars_arr[stat] = (prior_vars.get(stat) if prior_vars else None) or KALMAN_PRIOR_VAR_BY_STAT.get(stat, KALMAN_PRIOR_VAR)

    # Pre-fill output arrays with prior (players with no observations stay at prior)
    kalman_means: dict[str, np.ndarray] = {s: np.full(n_rows, _prior_means_arr[s]) for s in stats}
    kalman_vars: dict[str, np.ndarray] = {s: np.full(n_rows, _prior_vars_arr[s]) for s in stats}

    # Which players have at least one observation per stat (skip others)
    players_with_obs: dict[str, set] = {}
    for stat in stats:
        adj_col = f"adjusted_{stat}"
        if adj_col in adjusted_games.columns:
            players_with_obs[stat] = set(
                adjusted_games.loc[adjusted_games[adj_col].notna(), "player_id"].unique()
            )
        else:
            players_with_obs[stat] = set()

    # Pre-compute per-stat constants
    _stat_params: dict[str, tuple] = {}
    for stat in stats:
        beta = decay_betas.get(stat, 0.999)
        R = obs_noises.get(stat, KALMAN_OBS_NOISE)
        q = (q_noise_by_stat.get(stat, KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT.get(stat, KALMAN_PROCESS_NOISE_PER_DAY))
             if q_noise_by_stat is not None
             else KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT.get(stat, KALMAN_PROCESS_NOISE_PER_DAY))
        clip = innovation_clip_sigma if innovation_clip_sigma is not None else KALMAN_INNOVATION_CLIP_SIGMA
        is_rate = stat in RATE_STAT_ATTEMPT_COL
        att_col = RATE_STAT_ATTEMPT_COL.get(stat, "")
        ref_att = KALMAN_REF_ATTEMPTS.get(stat, 5.0)
        _stat_params[stat] = (beta, R, q, clip, is_rate, att_col, ref_att)

    # Pre-extract all arrays once (avoids repeated DataFrame column access per player per stat)
    _all_dates_ns = adjusted_games["game_date"].values.astype(np.int64)
    _all_school = adjusted_games[school_col].values if school_col in adjusted_games.columns else None
    _all_snaps = adjusted_games["snaps"].values.astype(float) if "snaps" in adjusted_games.columns else None

    # Pre-2016 snap handling: replace proxy snap values with a fixed effective snap count
    # so observation noise is conservative (KALMAN_PRE2016_NOISE_MULTIPLIER × baseline).
    # eff_r = obs_noise * (ref_snaps / sv) → sv = ref_snaps / multiplier gives eff_r = obs_noise * multiplier
    if _all_snaps is not None and "snaps_are_real" in adjusted_games.columns:
        proxy_snap_val = float(KALMAN_REF_SNAPS) / KALMAN_PRE2016_NOISE_MULTIPLIER
        not_real = ~adjusted_games["snaps_are_real"].values.astype(bool)
        if not_real.any():
            _all_snaps = _all_snaps.copy()
            _all_snaps[not_real] = proxy_snap_val
            logger.info("Pre-2016 proxy snaps: %.0f%% of games get fixed noise (%.1fx)",
                        not_real.mean() * 100, KALMAN_PRE2016_NOISE_MULTIPLIER)

    # Age array: compute age-at-game from birth_year column if available
    _all_ages: Optional[np.ndarray] = None
    if age_col and age_col in adjusted_games.columns and "game_date" in adjusted_games.columns:
        birth_yr = adjusted_games[age_col].values.astype(float)
        game_yr = pd.to_datetime(adjusted_games["game_date"]).dt.year.values.astype(float)
        _all_ages = game_yr - birth_yr
    _all_obs: dict[str, np.ndarray] = {}
    _all_att: dict[str, np.ndarray] = {}
    for stat in stats:
        adj_col = f"adjusted_{stat}"
        _all_obs[stat] = adjusted_games[adj_col].values.astype(float) if adj_col in adjusted_games.columns else None
        _, _, _, _, is_rate, att_col, _ = _stat_params[stat]
        _all_att[stat] = (adjusted_games[att_col].values.astype(float)
                          if is_rate and att_col and att_col in adjusted_games.columns else None)

    # Find player boundaries using numpy (avoids pandas groupby overhead)
    player_id_arr = adjusted_games["player_id"].values
    unique_ids, start_indices = np.unique(player_id_arr, return_index=True)
    end_indices = np.concatenate([start_indices[1:], [n_rows]])

    n_players = len(unique_ids)
    _log_interval = max(n_players // 10, 1000)

    for i_player, (player_id, start, end) in enumerate(
            zip(unique_ids, start_indices, end_indices)):
        if i_player % _log_interval == 0:
            logger.info("  Kalman: player %d / %d", i_player, n_players)

        n_player = int(end - start)

        player_ref = (_global_ref if _global_ref is not None
                      else pd.Timestamp(adjusted_games["game_date"].iloc[end - 1]))
        ref_ns = np.int64(player_ref.value)

        # ── Per-player shared arrays (hoisted outside stat loop) ───────────────
        dates_ns = _all_dates_ns[start:end]
        school_arr = _all_school[start:end] if _all_school is not None else None

        # Pre-compute days_from_ref once per player (beta varies per stat, not dates)
        days_from_ref = np.maximum((ref_ns - dates_ns) / _NS_PER_DAY, 0.0)

        # Pre-compute transfer_mask once per player (same across all stats)
        transfer_mask = _compute_transfer_mask(school_arr, n_player)

        snap_arr = _all_snaps[start:end] if _all_snaps is not None else None
        age_arr = _all_ages[start:end] if _all_ages is not None else None

        for stat in stats:
            if player_id not in players_with_obs.get(stat, set()):
                continue  # Already filled with prior

            obs_arr = _all_obs.get(stat)
            if obs_arr is None:
                continue
            obs_arr = obs_arr[start:end]

            beta, R, q, clip, is_rate, att_col, ref_att = _stat_params[stat]

            weight_arr = beta ** days_from_ref

            att_full = _all_att.get(stat)
            att_arr = att_full[start:end] if att_full is not None else None
            _snap = snap_arr if att_arr is None else None  # snap only when att not available

            try:
                means, variances = _kalman_player_fast(
                    obs_arr, dates_ns, weight_arr, None, _snap, att_arr,
                    _prior_means_arr[stat], _prior_vars_arr[stat],
                    q, R, ref_att, is_rate, transfer_p_mult, clip,
                    transfer_mask=transfer_mask,  # pre-computed; skips string comparison
                    age_arr=age_arr,
                    age_q_halflife=age_q_halflife,
                    age_q_max_mult=age_q_max_mult,
                )
                kalman_means[stat][start:end] = means
                kalman_vars[stat][start:end] = variances
            except Exception as exc:
                logger.warning("Kalman failed for player %s stat %s: %s",
                               player_id, stat, exc)

    for stat in stats:
        adjusted_games[f"kalman_mean_{stat}"] = kalman_means[stat]
        adjusted_games[f"kalman_var_{stat}"] = kalman_vars[stat]

    del kalman_means, kalman_vars
    gc.collect()

    return adjusted_games


# ── Data-Driven Prior Estimation (method-of-moments, same as cdarko) ──────────

def estimate_prior_vars(adjusted_games: pd.DataFrame,
                         stats: list[str],
                         obs_noises: Optional[dict[str, float]] = None,
                         min_games: int = 5,
                         ) -> dict[str, float]:
    """
    Estimate per-stat prior variance P₀ via method-of-moments:

        P₀ = Var(player_career_means) - R / mean_n

    where player_career_means is each player's mean adjusted stat (averaged
    across all their games), R is observation noise, and mean_n is the average
    number of games per qualifying player.

    Uses player career means (not last observation) so that the variance
    estimate reflects true-talent spread, not single-game noise. Only players
    with ≥ min_games observations are included to exclude trick-play outliers
    (e.g. WRs who threw one pass) that would inflate the variance.

    Returns dict stat → P₀ (floored at 0.01).
    """
    if obs_noises is None:
        obs_noises = KALMAN_OBS_NOISE_BY_STAT

    prior_vars: dict[str, float] = {}
    for stat in stats:
        adj_col = f"adjusted_{stat}"
        if adj_col not in adjusted_games.columns:
            prior_vars[stat] = KALMAN_PRIOR_VAR_BY_STAT.get(stat, KALMAN_PRIOR_VAR)
            continue

        obs = adjusted_games.dropna(subset=[adj_col])
        player_counts = obs.groupby("player_id")[adj_col].count()
        qualifying = player_counts[player_counts >= min_games].index
        obs_qual = obs[obs["player_id"].isin(qualifying)]

        if len(qualifying) < 10:
            prior_vars[stat] = KALMAN_PRIOR_VAR_BY_STAT.get(stat, KALMAN_PRIOR_VAR)
            continue

        player_means = obs_qual.groupby("player_id")[adj_col].mean()
        mean_n = float(player_counts[qualifying].mean())
        R = obs_noises.get(stat, KALMAN_OBS_NOISE)
        # Winsorize at 2.5/97.5 percentile to prevent extreme per-100-snap
        # outliers (e.g. 1-snap players) from inflating the variance estimate.
        lo, hi = float(player_means.quantile(0.025)), float(player_means.quantile(0.975))
        clipped = player_means.clip(lower=lo, upper=hi)
        var_player_means = float(clipped.var())
        p0 = max(var_player_means - R / max(mean_n, 1.0), 0.01)
        prior_vars[stat] = p0
        logger.debug("Prior P₀ for %s: %.4f (var=%.4f, R=%.4f, mean_n=%.1f, n_players=%d)",
                     stat, p0, var_player_means, R, mean_n, len(qualifying))

    return prior_vars


def estimate_prior_means(adjusted_games: pd.DataFrame,
                          stats: list[str]) -> dict[str, float]:
    """
    Estimate per-stat prior mean from population average of adjusted residuals.
    For well-calibrated residuals this should be near 0.
    """
    prior_means: dict[str, float] = {}
    for stat in stats:
        adj_col = f"adjusted_{stat}"
        if adj_col not in adjusted_games.columns:
            prior_means[stat] = KALMAN_PRIOR_MEAN
            continue
        vals = adjusted_games[adj_col].dropna()
        prior_means[stat] = float(vals.mean()) if len(vals) > 0 else KALMAN_PRIOR_MEAN
    return prior_means


# ── Draft-Calibrated Kalman Hyperparameters ────────────────────────────────────

def estimate_q_from_yoy_variance(
        adjusted_games: pd.DataFrame,
        stats: list[str],
        interseason_days: float = 240.0,
        min_seasons: int = 2,
        min_games_per_season: int = 5,
) -> dict[str, float]:
    """
    Estimate per-stat Kalman process noise Q from year-over-year variance in
    season averages.

    For a latent random walk with per-day noise Q:
        Var(avg_{t+1} - avg_t) ≈ Q × T
    where T is the typical inter-season gap in days.  Rearranging: Q ≈ Var(YoY) / T.

    Args:
        adjusted_games:       Game-level adjusted stats (adjusted_{stat} columns).
        stats:                Stats to estimate Q for.
        interseason_days:     Assumed days between seasons (football ≈ 240, Jan→Sep).
        min_seasons:          Min consecutive-season pairs required per player.
        min_games_per_season: Min games per season to include in season average.

    Returns:
        dict stat → estimated Q (per day). Falls back to config defaults when
        data is thin.
    """
    q_estimates: dict[str, float] = {}

    for stat in stats:
        adj_col = f"adjusted_{stat}"
        if adj_col not in adjusted_games.columns or "season_year" not in adjusted_games.columns:
            q_estimates[stat] = KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT.get(
                stat, KALMAN_PROCESS_NOISE_PER_DAY
            )
            continue

        sub = adjusted_games.dropna(subset=[adj_col, "season_year"])
        season_stats = (
            sub.groupby(["player_id", "season_year"])[adj_col]
            .agg(["count", "mean"])
            .rename(columns={"count": "n_games", "mean": "season_avg"})
            .reset_index()
        )
        season_stats = season_stats[season_stats["n_games"] >= min_games_per_season]
        season_stats = season_stats.sort_values(["player_id", "season_year"])
        season_stats["prev_year"] = season_stats.groupby("player_id")["season_year"].shift(1)
        season_stats["prev_avg"] = season_stats.groupby("player_id")["season_avg"].shift(1)
        consec = season_stats[
            season_stats["season_year"] - season_stats["prev_year"] == 1
        ].dropna(subset=["prev_avg"])

        if len(consec) < min_seasons * 10:
            q_estimates[stat] = KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT.get(
                stat, KALMAN_PROCESS_NOISE_PER_DAY
            )
            continue

        yoy_var = float((consec["season_avg"] - consec["prev_avg"]).var())
        q_estimates[stat] = max(yoy_var / interseason_days, 1e-6)

    logger.info(
        "Data-derived Q (per day, YoY variance / %.0f days): %s",
        interseason_days,
        {s: f"{v:.5f}" for s, v in q_estimates.items()},
    )
    return q_estimates


def calibrate_kalman_for_draft(
        adjusted_games: pd.DataFrame,
        role_labels: pd.DataFrame,
        betas: dict[str, float],
        obs_noises: dict[str, float],
        process_noises: dict[str, float],
        stats: list[str],
        cal_season_cutoff: int = 2020,
        role_weight: float = 0.5,
        impact_weight: float = 0.5,
        n_global_iter: int = 200,
        param_bounds: tuple = ((0.5, 10.0), (1.0, 8.0), (1.0, 10.0)),
        prior_means: Optional[dict[str, float]] = None,
        prior_vars: Optional[dict[str, float]] = None,
        age_col: Optional[str] = None,
        school_col: str = "school",
        initial_global_params: Optional[dict] = None,
) -> tuple[dict, dict]:
    """
    Calibrate global Kalman hyperparameters by maximising Ridge R² predicting
    NFL draft role and impact outcomes from terminal Kalman features.

    Uses data-derived Q (from estimate_q_from_yoy_variance) as process noise —
    per-stat Q multiplier stage is skipped.

    Stage 1 — Nelder-Mead over (age_q_halflife_years, age_q_max_multiplier,
               transfer_p_multiplier). Objective: 0.5×role_R² + 0.5×impact_R².

    Args:
        adjusted_games:      Full adjusted_gamelogs.parquet.
        role_labels:         nfl_draft_labels with player_id, role_target,
                             impact_target, last_college_season_year.
        betas:               Decay betas (already calibrated via 04b).
        obs_noises:          Per-stat observation noise (within-player variance).
        process_noises:      Per-stat process noise (from estimate_q_from_yoy_variance).
        stats:               Stats to include in terminal feature extraction.
        cal_season_cutoff:   last_college_season_year ≤ cutoff → cal cohort.
        role_weight:         Weight on role R² in combined objective.
        impact_weight:       Weight on impact R² in combined objective.
        n_global_iter:       Max Nelder-Mead iterations.
        param_bounds:        ((halflife_lo, hi), (max_mult_lo, hi), (transfer_lo, hi)).
        prior_means:         Per-stat prior means for Kalman (from estimate_prior_means).
        prior_vars:          Per-stat prior variances (from estimate_prior_vars).
        initial_global_params: If provided, use as Nelder-Mead starting point.

    Returns:
        params_dict: age_q_halflife_years, age_q_max_multiplier, transfer_p_multiplier,
                     q_multipliers (back-computed from process_noises for reference)
        eval_dict:   role_r2_cal, role_r2_val, impact_r2_cal, impact_r2_val,
                     combined_score_cal, combined_score_val, n_cal, n_val
    """
    from scipy.optimize import minimize
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Deduplicate on player_id before indexing (same guard as in optimize_betas_for_role)
    labels_dedup = (role_labels.dropna(subset=["player_id"])
                    .sort_values("player_id")
                    .drop_duplicates("player_id")
                    .set_index("player_id"))

    bridge_pids = set(labels_dedup.index)
    cal_mask = labels_dedup["last_college_season_year"] <= cal_season_cutoff
    cal_pids = set(labels_dedup.index[cal_mask])
    val_pids = set(labels_dedup.index[~cal_mask])

    cal_labels = labels_dedup[labels_dedup.index.isin(cal_pids)]
    val_labels = labels_dedup[labels_dedup.index.isin(val_pids)]

    bridge_adj = adjusted_games[adjusted_games["player_id"].isin(bridge_pids)].copy()

    has_impact = "impact_target" in labels_dedup.columns

    logger.info(
        "Draft Kalman calibration: %d cal players (≤%d), %d val players (>%d), %d stats",
        len(cal_pids), cal_season_cutoff, len(val_pids), cal_season_cutoff, len(stats),
    )

    def _run_kalman_and_score(halflife: float, max_mult: float, transfer_mult: float,
                               cohort_adj: pd.DataFrame, cohort_labels: pd.DataFrame,
                               ) -> tuple[float, float]:
        """Run Kalman on cohort, extract terminal features, Ridge R² for role + impact."""
        try:
            result = run_kalman_all_players(
                cohort_adj.copy(),
                decay_betas=betas,
                obs_noises=obs_noises,
                stats=stats,
                age_col=age_col,
                age_q_halflife=halflife,
                age_q_max_mult=max_mult,
                transfer_p_mult=transfer_mult,
                school_col=school_col,
                q_noise_by_stat=process_noises,
                prior_means=prior_means,
                prior_vars=prior_vars,
            )
        except Exception as exc:
            logger.debug("Kalman run failed: %s", exc)
            return 0.0, 0.0

        # Terminal features: last-game kalman_mean and kalman_var per player
        terminal = (result.sort_values("game_date")
                    .groupby("player_id", as_index=False)
                    .last())

        feat_cols = [c for c in terminal.columns
                     if c.startswith("kalman_mean_") or c.startswith("kalman_var_")]
        feat_cols = [c for c in feat_cols if terminal[c].notna().sum() > 10]
        if not feat_cols:
            return 0.0, 0.0

        merged = terminal.set_index("player_id").join(
            cohort_labels[["role_target"] + (["impact_target"] if has_impact else [])],
            how="inner",
        ).dropna(subset=["role_target"])

        if len(merged) < 20:
            return 0.0, 0.0

        X = merged[feat_cols].fillna(0.0).values
        y_role = merged["role_target"].values

        # CV R², not in-sample: with ~60 terminal features on a few hundred
        # cal players, in-sample R² rewards hyperparameters that make the
        # Kalman features overfit-friendly rather than predictive. 3-fold CV
        # (fixed seed → deterministic objective for Nelder-Mead) scores each
        # candidate on rows its Ridge never saw. Fold scores are clipped to
        # tame small-fold blowups.
        from sklearn.model_selection import KFold, cross_val_score
        _cv = KFold(n_splits=3, shuffle=True, random_state=42)

        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        role_r2 = float(np.mean(np.clip(
            cross_val_score(ridge, X, y_role, cv=_cv, scoring="r2"), -1.0, 1.0)))

        impact_r2 = 0.0
        if has_impact and impact_weight > 0 and "impact_target" in merged.columns:
            imp_mask = merged["impact_target"].notna()
            if imp_mask.sum() >= 20:
                X_imp = merged.loc[imp_mask, feat_cols].fillna(0.0).values
                y_imp  = merged.loc[imp_mask, "impact_target"].values
                ridge_imp = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                impact_r2 = float(np.mean(np.clip(
                    cross_val_score(ridge_imp, X_imp, y_imp, cv=_cv,
                                    scoring="r2"), -1.0, 1.0)))

        return max(role_r2, 0.0), max(impact_r2, 0.0)

    _eval_count = [0]

    def _objective(params: np.ndarray) -> float:
        halflife = float(np.clip(params[0], param_bounds[0][0], param_bounds[0][1]))
        max_mult = float(np.clip(params[1], param_bounds[1][0], param_bounds[1][1]))
        transfer = float(np.clip(params[2], param_bounds[2][0], param_bounds[2][1]))
        _eval_count[0] += 1

        cal_adj = bridge_adj[bridge_adj["player_id"].isin(cal_pids)].copy()
        role_r2, impact_r2 = _run_kalman_and_score(halflife, max_mult, transfer,
                                                    cal_adj, cal_labels)
        combined = role_weight * role_r2 + impact_weight * impact_r2
        if _eval_count[0] % 20 == 0:
            logger.info(
                "  Iter %d: halflife=%.2f max_mult=%.2f transfer=%.2f → "
                "role_R²=%.4f impact_R²=%.4f combined=%.4f",
                _eval_count[0], halflife, max_mult, transfer, role_r2, impact_r2, combined,
            )
        return -combined

    # Starting point
    if initial_global_params is not None:
        x0 = np.array([
            initial_global_params.get("age_q_halflife_years", AGE_Q_HALFLIFE_YEARS),
            initial_global_params.get("age_q_max_multiplier", AGE_Q_MAX_MULTIPLIER),
            initial_global_params.get("transfer_p_multiplier", TRANSFER_P_MULTIPLIER),
        ])
    else:
        x0 = np.array([AGE_Q_HALFLIFE_YEARS, AGE_Q_MAX_MULTIPLIER, TRANSFER_P_MULTIPLIER])

    logger.info("Stage 1: Nelder-Mead over (halflife, max_mult, transfer_mult), max %d iters",
                n_global_iter)
    opt_result = minimize(
        _objective, x0,
        method="Nelder-Mead",
        options={"maxiter": n_global_iter, "xatol": 0.05, "fatol": 1e-4, "disp": False},
    )

    best_halflife = float(np.clip(opt_result.x[0], param_bounds[0][0], param_bounds[0][1]))
    best_max_mult = float(np.clip(opt_result.x[1], param_bounds[1][0], param_bounds[1][1]))
    best_transfer = float(np.clip(opt_result.x[2], param_bounds[2][0], param_bounds[2][1]))
    logger.info(
        "Nelder-Mead complete (%d evals): halflife=%.3f max_mult=%.3f transfer=%.3f",
        _eval_count[0], best_halflife, best_max_mult, best_transfer,
    )

    # Final evaluation on cal + val
    cal_adj_final = bridge_adj[bridge_adj["player_id"].isin(cal_pids)].copy()
    val_adj_final = bridge_adj[bridge_adj["player_id"].isin(val_pids)].copy()

    cal_role_r2, cal_impact_r2 = _run_kalman_and_score(
        best_halflife, best_max_mult, best_transfer, cal_adj_final, cal_labels)
    val_role_r2, val_impact_r2 = _run_kalman_and_score(
        best_halflife, best_max_mult, best_transfer, val_adj_final, val_labels)

    cal_combined = role_weight * cal_role_r2 + impact_weight * cal_impact_r2
    val_combined = role_weight * val_role_r2 + impact_weight * val_impact_r2

    # Back-compute q_multipliers for reference / downstream logging
    base_q = {s: KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT.get(s, KALMAN_PROCESS_NOISE_PER_DAY)
              for s in stats}
    q_multipliers = {s: process_noises.get(s, base_q[s]) / max(base_q[s], 1e-10)
                     for s in stats}

    params_dict = {
        "age_q_halflife_years":  best_halflife,
        "age_q_max_multiplier":  best_max_mult,
        "transfer_p_multiplier": best_transfer,
        "q_multipliers":         q_multipliers,
    }
    eval_dict = {
        "role_r2_cal":          cal_role_r2,
        "role_r2_val":          val_role_r2,
        "impact_r2_cal":        cal_impact_r2,
        "impact_r2_val":        val_impact_r2,
        "combined_score_cal":   cal_combined,
        "combined_score_val":   val_combined,
        "n_cal":                len(cal_pids),
        "n_val":                len(val_pids),
        "cal_season_cutoff":    cal_season_cutoff,
        "stats_used":           stats,
    }

    logger.info(
        "Draft calibration results:\n"
        "  role_R²_cal=%.4f  role_R²_val=%.4f\n"
        "  impact_R²_cal=%.4f  impact_R²_val=%.4f\n"
        "  combined_cal=%.4f  combined_val=%.4f",
        cal_role_r2, val_role_r2, cal_impact_r2, val_impact_r2, cal_combined, val_combined,
    )
    return params_dict, eval_dict
