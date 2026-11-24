"""
Per-player career rollup of teammate-QB rolling features.

Reads ``data/raw/player_gamelogs.parquet`` (after step 1 has joined
``qb_rolling_*``) and emits ``{col}_mean / _peak / _last`` per player. The
rollup is gated to WR/TE/RB games — the only positions consuming these
features in step 8 — and exposure-weighted by the player's per-game
involvement (targets → rec → rush_att → snaps, first non-null/>0 wins).

The function is invoked from ``pipeline/08_draft_models.py::main`` and
``pipeline/08b_global_draft_model.py::main`` after the snapshot has been
assembled. It mirrors the API and shape of
``src.scrapers.pff.load_pff_career_features`` — one row per player_id,
joinable on the snapshot's ``player_id`` column.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure config import resolves when this module is loaded from arbitrary cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    QB_ROLLING_FEATURES,
    QB_CAREER_BASE_FEATURES,
    QB_PEAK_DIRECTION,
    QB_CAREER_POSITION_GROUPS,
)

logger = logging.getLogger(__name__)


# Cascade for per-row exposure weight. First non-null / >0 column wins; the
# last fallback is `snaps`, which is always populated post-step-1.
_EXPOSURE_PRIORITY: list[str] = ["targets", "rec", "rush_att", "snaps"]


def _per_row_exposure(df: pd.DataFrame) -> pd.Series:
    """Per-row exposure weight via the standard cascade. NaN where every
    candidate column is missing/zero on a row — caller drops those rows."""
    w = pd.Series(np.nan, index=df.index, dtype=float)
    for col in _EXPOSURE_PRIORITY:
        if col not in df.columns:
            continue
        candidate = pd.to_numeric(df[col], errors="coerce")
        mask = w.isna() & candidate.notna() & (candidate > 0)
        w.loc[mask] = candidate.loc[mask]
    return w


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """NaN-safe weighted mean. Returns NaN if no value has positive weight."""
    valid = ~np.isnan(values) & (weights > 0) & ~np.isnan(weights)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def load_qb_career_features(
    snapshot: pd.DataFrame,
    processed_dir: Path,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute per-player teammate-QB career triplets.

    Parameters
    ----------
    snapshot
        Draft-model snapshot. Used only to scope the player_id universe
        (we filter the gamelog to players appearing in the snapshot before
        the per-player loop).
    processed_dir
        ``data/processed`` directory — accepted for API parallelism with
        ``load_pff_career_features``; not currently consumed.
    raw_dir
        ``data/raw`` directory containing ``player_gamelogs.parquet``.
        Defaults to ``processed_dir.parent / "raw"`` when omitted.

    Returns
    -------
    pd.DataFrame
        ``["player_id"] + [f"{c}_{s}" for c in QB_ROLLING_FEATURES for s in
        ("_mean", "_peak", "_last")]``. NaN where a player has no qualifying
        WR/TE/RB gamelogs (defenders, kickers, OL, walk-ons absent from
        gamelogs entirely).
    """
    if snapshot is None or snapshot.empty:
        return pd.DataFrame()

    if raw_dir is None:
        raw_dir = Path(processed_dir).parent / "raw"
    gl_path = Path(raw_dir) / "player_gamelogs.parquet"
    if not gl_path.exists():
        logger.warning("load_qb_career_features: %s missing; skipping", gl_path)
        return pd.DataFrame()

    # Probe columns first so we only load what we need (gamelogs are wide).
    probe = pd.read_parquet(gl_path, columns=None).columns.tolist()
    base_cols = ["player_id", "season_year", "position_group"]
    if not all(c in probe for c in base_cols):
        logger.warning("load_qb_career_features: gamelogs missing %s; skipping",
                       set(base_cols) - set(probe))
        return pd.DataFrame()
    # Producer is gated to QB_CAREER_BASE_FEATURES (subset of QB_ROLLING_FEATURES
    # — currently passer_rating only after the 2026-05-16 Boruta audit). The
    # consumer in step 8 / 8b reads only these triplets.
    qb_cols_present = [c for c in QB_CAREER_BASE_FEATURES if c in probe]
    if not qb_cols_present:
        logger.info("load_qb_career_features: no qb_rolling_* columns in "
                    "gamelogs — step 1 hasn't been rerun since the feature "
                    "was added. Returning empty.")
        return pd.DataFrame()
    exposure_cols = [c for c in _EXPOSURE_PRIORITY if c in probe]
    needed = base_cols + qb_cols_present + exposure_cols

    gl = pd.read_parquet(gl_path, columns=needed)

    # Scope to position groups that consume the feature (career triplets are
    # only emitted for WR/TE/RB in step 8).
    gl = gl[gl["position_group"].isin(QB_CAREER_POSITION_GROUPS)]
    if gl.empty:
        return pd.DataFrame()

    # Scope to players the snapshot cares about (avoids carrying the entire
    # historical gamelog through the per-player loop).
    snap_pids = set(snapshot["player_id"].dropna().unique())
    gl = gl[gl["player_id"].isin(snap_pids)].copy()
    if gl.empty:
        return pd.DataFrame()

    gl["__w"] = _per_row_exposure(gl)
    gl = gl[gl["__w"] > 0]
    # Drop rows with no qb_rolling_* signal at all
    has_any = gl[qb_cols_present].notna().any(axis=1)
    gl = gl[has_any]
    if gl.empty:
        return pd.DataFrame()

    # Pre-coerce the QB cols to float64 for numpy paths
    for c in qb_cols_present:
        gl[c] = pd.to_numeric(gl[c], errors="coerce")

    # Compute per-(player, season) weighted means in one pass — vectorized
    # via groupby+apply. We'll then take overall / peak / last over the
    # season-level frame.
    def _season_block(grp: pd.DataFrame) -> pd.Series:
        w = grp["__w"].to_numpy(dtype=float)
        out = {}
        for c in qb_cols_present:
            out[c] = _weighted_mean(grp[c].to_numpy(dtype=float), w)
        out["__exposure"] = float(np.nansum(w))
        return pd.Series(out)

    season_panel = (gl.groupby(["player_id", "season_year"], dropna=False,
                                group_keys=True)
                       .apply(_season_block, include_groups=False)
                       .reset_index())
    if season_panel.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    for pid, grp in season_panel.groupby("player_id", sort=False):
        row = {"player_id": pid}
        w = grp["__exposure"].to_numpy(dtype=float)
        years = grp["season_year"].to_numpy()
        latest_year = years[~pd.isna(years)].max() if (~pd.isna(years)).any() else None

        for c in qb_cols_present:
            vals = grp[c].to_numpy(dtype=float)

            # _mean — exposure-weighted across all of the player's seasons.
            row[f"{c}_mean"] = _weighted_mean(vals, w)

            # _peak — directional max/min across season-mean values (NaN-safe).
            direction = QB_PEAK_DIRECTION.get(c, "max")
            non_nan = vals[~np.isnan(vals)]
            if non_nan.size == 0:
                row[f"{c}_peak"] = float("nan")
            elif direction == "min":
                row[f"{c}_peak"] = float(np.min(non_nan))
            else:
                row[f"{c}_peak"] = float(np.max(non_nan))

            # _last — most recent season's weighted mean.
            if latest_year is None:
                row[f"{c}_last"] = float("nan")
            else:
                last_mask = (years == latest_year)
                last_vals = vals[last_mask]
                if last_vals.size == 0 or np.isnan(last_vals).all():
                    row[f"{c}_last"] = float("nan")
                else:
                    # Single-season weighted mean degenerates to the per-season
                    # value already in season_panel; surface that directly.
                    row[f"{c}_last"] = float(last_vals[~np.isnan(last_vals)][0])
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    logger.info("load_qb_career_features: %d players × %d triplet cols",
                len(out_df), 3 * len(qb_cols_present))
    return out_df
