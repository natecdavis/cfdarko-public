"""
Generic strictly-prior, exponential-decay-weighted rolling aggregation.

Used by `src/scrapers/cfbd.py::compute_rolling_opponent_strength` (opponent
strength) and `src/scrapers/cfbd.py::compute_qb_rolling_stats` (teammate QB
quality). The shared primitive guarantees both pipelines apply identical
date-windowing, weight handling, and missing-value semantics.

Implementation note (2026-07-21 vectorization): for a target at date d,
    avg = Σ_j exp(λ(d−t_j))·w_j·v_j / Σ_j exp(λ(d−t_j))·w_j
factors as exp(λ(d−t_last))·Σ_j exp(λ(t_last−t_j))·w_j·(v_j|1) in both
numerator and denominator — the target-side decay CANCELS in the ratio. So
each entity's history is reduced once to prefix sums via the standard EWMA
recurrence S_i = S_{i−1}·exp(λΔt_i) + w_i·(v_i|1) (multiplicative decay
between consecutive observations: bounded in (0,1], no overflow), and every
target is answered with one searchsorted + one ratio. Replaces a per-target
Python loop that re-sliced the entity's full history per row (~600k pandas
filters per step-1 assembly).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_rolling_entity_strength(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    panel_entity_col: str,
    target_entity_col: str,
    date_col: str,
    stat_cols: list[str],
    halflife_days: float,
    weight_col: Optional[str] = None,
    out_prefix: str = "rolling",
    keep_target_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """For each row in ``targets``, compute the strictly-prior, exponential-
    decay-weighted average of ``stat_cols`` from ``panel`` where
    ``panel[panel_entity_col] == target[target_entity_col]`` and
    ``panel[date_col] < target[date_col]``.

    Parameters
    ----------
    panel
        Per-entity-per-date observations. Must contain ``panel_entity_col``,
        ``date_col``, and every column in ``stat_cols``. ``weight_col`` is
        optional.
    targets
        Rows where the rolling value should be evaluated. Must contain
        ``target_entity_col`` and ``date_col``. Output is one row per
        ``targets`` row, in the same order.
    panel_entity_col, target_entity_col
        Column names identifying the entity. Different names allow lookups
        like "for each (team, opponent) row, fetch the OPPONENT's prior
        team-stat history" — the panel keys on "team", the target keys on
        "opponent".
    date_col
        Date column name (must exist on both panel and targets).
    stat_cols
        Columns from ``panel`` to aggregate. Each emits one output column
        named ``f"{out_prefix}_{stat}"``.
    halflife_days
        Exponential decay halflife (per day).
    weight_col
        Optional column in ``panel`` whose values multiply the per-row decay
        weight. Caller pre-applies any transform (e.g. ``np.sqrt``) and
        clipping. NaN obs weights contribute 0 (excluded from the average).
    out_prefix
        Prefix for the emitted aggregate columns. Default ``"rolling"``.
    keep_target_cols
        Columns from ``targets`` to carry into the output (e.g. ``["game_id",
        "team"]``). Useful for downstream merges.

    Returns
    -------
    DataFrame
        ``len(targets)`` rows × ``len(keep_target_cols) + len(stat_cols)``
        columns. Aggregate columns are NaN where the target entity is absent
        from the panel, no prior observations exist, or all stat values in
        the prior window are NaN.
    """
    if panel.empty or targets.empty:
        return pd.DataFrame()

    if keep_target_cols is None:
        keep_target_cols = []

    # Defensive: ensure panel has every stat_col + (optional) weight_col
    panel = panel.copy()
    for col in stat_cols:
        if col not in panel.columns:
            panel[col] = np.nan
    if weight_col is not None and weight_col not in panel.columns:
        panel[weight_col] = 1.0

    # Normalize dtypes once, up front
    panel[date_col] = pd.to_datetime(panel[date_col])
    targets = targets.copy()
    targets[date_col] = pd.to_datetime(targets[date_col])

    # Undated panel rows can't be placed on the decay timeline. They must be
    # dropped rather than carried: pandas sorts NaT last, and NaT casts to a
    # huge negative day number, so the `days[i] - prev_day` gap on the row
    # after them goes hugely negative and `math.exp` overflows (log_half is
    # negative, so a negative gap is a positive exponent). The reference
    # implementation excludes them implicitly — `hist[date_col] < d` is False
    # against NaT — so dropping here also keeps the two paths equivalent.
    undated = panel[date_col].isna()
    if undated.any():
        logger.warning(
            "Dropping %d/%d panel rows with no %s before rolling aggregation "
            "(entities: %s)", int(undated.sum()), len(panel), date_col,
            panel.loc[undated, panel_entity_col].nunique(),
        )
        panel = panel[~undated]
        if panel.empty:
            return pd.DataFrame()

    log_half = math.log(0.5) / halflife_days  # negative
    n_stats = len(stat_cols)
    out_cols = [f"{out_prefix}_{c}" for c in stat_cols]

    # ── Reduce each entity's history to prefix sums (one pass over panel) ──
    # state[ent] = (dates_ns_sorted, num_pref [n×k], den_pref [n×k])
    state: dict[object, tuple] = {}
    for ent, grp in panel.groupby(panel_entity_col):
        grp = grp.sort_values(date_col)
        dates = grp[date_col].values                     # datetime64[ns]
        # Day units matching the reference (d − t).dt.days semantics: whole
        # days between date-normalized timestamps.
        days = (dates.astype("datetime64[D]")
                     .astype("int64").astype(float))
        vals = grp[stat_cols].to_numpy(dtype=float, na_value=np.nan)
        if weight_col is not None:
            obs_w = grp[weight_col].to_numpy(dtype=float, na_value=np.nan)
            obs_w = np.where(np.isnan(obs_w), 0.0, np.clip(obs_w, 0.0, None))
        else:
            obs_w = np.ones(len(grp), dtype=float)

        valid = ~np.isnan(vals) & (obs_w[:, None] > 0)   # per-stat validity
        add_den = np.where(valid, obs_w[:, None], 0.0)
        add_num = np.where(valid, obs_w[:, None] * np.nan_to_num(vals), 0.0)

        n = len(grp)
        num_pref = np.empty((n, n_stats), dtype=float)
        den_pref = np.empty((n, n_stats), dtype=float)
        s_num = np.zeros(n_stats, dtype=float)
        s_den = np.zeros(n_stats, dtype=float)
        prev_day = days[0] if n else 0.0
        for i in range(n):
            decay = math.exp(log_half * (days[i] - prev_day))
            s_num = s_num * decay + add_num[i]
            s_den = s_den * decay + add_den[i]
            num_pref[i] = s_num
            den_pref[i] = s_den
            prev_day = days[i]
        state[ent] = (dates, num_pref, den_pref)

    # ── Answer all targets per entity via searchsorted ─────────────────────
    result = np.full((len(targets), n_stats), np.nan, dtype=float)
    tgt_dates_all = targets[date_col].values
    positions = np.arange(len(targets))
    tgt_ents = targets[target_entity_col]

    for ent, idxs in pd.Series(positions).groupby(tgt_ents.values).groups.items():
        st = state.get(ent)
        if st is None:
            continue
        dates, num_pref, den_pref = st
        idxs = np.asarray(idxs)
        d = tgt_dates_all[idxs]
        d_ok = ~pd.isna(d)
        if not d_ok.any():
            continue
        idxs = idxs[d_ok]
        d = d[d_ok]
        # Strictly prior: count of obs with date < d
        ins = np.searchsorted(dates, d, side="left")
        has_prior = ins > 0
        if not has_prior.any():
            continue
        idxs = idxs[has_prior]
        prev = ins[has_prior] - 1
        num = num_pref[prev]
        den = den_pref[prev]
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where(den > 0.0, num / den, np.nan)
        result[idxs] = vals

    out = targets[list(keep_target_cols)].reset_index(drop=True).copy() \
        if keep_target_cols else pd.DataFrame(index=range(len(targets)))
    for j, col in enumerate(out_cols):
        out[col] = result[:, j]
    return out


def _compute_rolling_entity_strength_reference(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    panel_entity_col: str,
    target_entity_col: str,
    date_col: str,
    stat_cols: list[str],
    halflife_days: float,
    weight_col: Optional[str] = None,
    out_prefix: str = "rolling",
    keep_target_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Original per-row implementation, kept ONLY as the equivalence oracle
    for tests (see tests/test_rolling_equivalence.py). Do not call from
    pipeline code — it re-slices the entity history per target row and is
    orders of magnitude slower."""
    if panel.empty or targets.empty:
        return pd.DataFrame()

    if keep_target_cols is None:
        keep_target_cols = []

    panel = panel.copy()
    for col in stat_cols:
        if col not in panel.columns:
            panel[col] = np.nan
    if weight_col is not None and weight_col not in panel.columns:
        panel[weight_col] = 1.0

    panel[date_col] = pd.to_datetime(panel[date_col])
    targets = targets.copy()
    targets[date_col] = pd.to_datetime(targets[date_col])

    needed_cols = [date_col] + stat_cols
    if weight_col is not None:
        needed_cols.append(weight_col)

    history: dict[object, pd.DataFrame] = {}
    for ent, grp in panel.groupby(panel_entity_col):
        history[ent] = grp.sort_values(date_col)[needed_cols].copy()

    log_half = math.log(0.5) / halflife_days  # negative

    out_cols = [f"{out_prefix}_{c}" for c in stat_cols]
    nan_record_base = {col: float("nan") for col in out_cols}

    records: list[dict] = []
    for _, row in targets.iterrows():
        rec = {k: row.get(k) for k in keep_target_cols}
        ent = row.get(target_entity_col)
        d = row.get(date_col)

        if pd.isna(d) or ent not in history:
            rec.update(nan_record_base)
            records.append(rec)
            continue

        hist = history[ent]
        prior = hist[hist[date_col] < d]
        if prior.empty:
            rec.update(nan_record_base)
            records.append(rec)
            continue

        days_elapsed = (d - prior[date_col]).dt.days.clip(lower=0).values
        try:
            log_w = log_half * days_elapsed.astype(float)
            if log_w.max() > 700:
                log_w -= log_w.max() - 700
            base_w = np.exp(log_w)
        except Exception:
            base_w = np.ones(len(prior), dtype=float)

        if weight_col is not None:
            obs_w = np.asarray(prior[weight_col].values, dtype=float)
            obs_w = np.where(np.isnan(obs_w), 0.0, np.clip(obs_w, a_min=0.0, a_max=None))
            w = base_w * obs_w
        else:
            w = base_w

        for stat, out_col in zip(stat_cols, out_cols):
            vals = np.asarray(prior[stat].values, dtype=float)
            valid = ~np.isnan(vals) & (w > 0)
            if not valid.any():
                rec[out_col] = float("nan")
            else:
                rec[out_col] = float(np.average(vals[valid], weights=w[valid]))
        records.append(rec)

    return pd.DataFrame(records)
