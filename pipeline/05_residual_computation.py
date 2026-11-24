"""
Step 5: Residual Computation (Game-Level)

Applies fitted contextual models to ALL player-games (not just bridge players).
For counting stats: residual = {stat}_p100 - expected_{stat}_p100
For rate stats:     residual = {stat} - expected_{stat}

Both stored as adjusted_{stat}.

`adjusted_{stat}` CARRIES TWO DIFFERENT SEMANTICS — read this before using it
==========================================================================
Per (position_group, stat), the column holds EITHER a context residual OR the
raw stat, depending on how the step-4 model for that head performed:

  "residual"     the head's expanding-window CV R² ≥ 0.01, so the fitted model
                 was applied and the column is (observed − expected). Centered
                 near zero; sign is meaningful; scale is "vs. context-expected".
  "raw_fallback" CV R² < 0.01, so adjustment is SKIPPED and the column is a
                 verbatim copy of the raw stat (p100 for counting stats, the
                 rate itself for rate stats). NOT centered, NOT comparable in
                 scale to a residual.
  "no_model"     no fitted model exists for that head at all; also a raw copy.

The fallback is deliberate, not a bug: marginal-positive heads over-correct,
because CV picks coefficients that fit training noise and inference then
subtracts that noise from the stat, giving residuals a spurious ±0.6 Spearman
against controls the raw stat was uncorrelated with. Copying the raw stat is the
conservative choice. But it means a consumer that assumes "adjusted_* is always a
residual" will silently mix centered and uncentered columns across heads.

Downstream this is mostly harmless because step 6 builds per-(player, stat)
Kalman/EWMA streams, and those are fit per stat — a stat that is raw everywhere
stays internally consistent. The hazard is any NEW code that pools adjusted_*
columns ACROSS heads, or that reads the sign of adjusted_* as "above/below
expectation" without checking which semantic applies.

To make that checkable rather than folklore, this step writes a sidecar:

    data/processed/adjusted_stat_semantics.json
        {"<position_group>": {"<stat>": {"semantic": ..., "cv_r2": ...}}}

Changing the semantics themselves is deliberately NOT attempted — a unified
scheme would either reintroduce the over-correction or force a scale change on
every downstream Kalman prior. The sidecar documents reality instead.

Output: data/processed/adjusted_gamelogs.parquet
        data/processed/adjusted_stat_semantics.json
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DIR, PROCESSED_DIR, MODELED_STATS, POSITION_TO_GROUP,
)
from src.models.contextual import ContextualAdjustmentModel, build_context_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_models() -> dict[tuple[str, str], ContextualAdjustmentModel]:
    """Load all fitted contextual models. Keys: (position_group, stat)."""
    models_dir = PROCESSED_DIR / "contextual_models"
    models: dict[tuple[str, str], ContextualAdjustmentModel] = {}

    for pkl_path in models_dir.glob("*_model.pkl"):
        try:
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            key = (model.position_group, model.stat)
            models[key] = model
        except Exception as exc:
            logger.warning("Could not load %s: %s", pkl_path, exc)

    logger.info("Loaded %d contextual models", len(models))
    return models


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    gamelogs_path = RAW_DIR / "player_gamelogs.parquet"
    seasons_path = RAW_DIR / "player_seasons.parquet"
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    bridge_path = PROCESSED_DIR / "bridge_players.parquet"

    use_gamelogs = gamelogs_path.exists()
    if use_gamelogs:
        logger.info("Loading game-level data: %s", gamelogs_path)
        player_data = pd.read_parquet(gamelogs_path)
        out_path = PROCESSED_DIR / "adjusted_gamelogs.parquet"
    else:
        logger.warning("player_gamelogs.parquet not found — falling back to season-level data")
        if not seasons_path.exists():
            logger.error("player_seasons.parquet not found — run Step 1 first")
            sys.exit(1)
        player_data = pd.read_parquet(seasons_path)
        out_path = PROCESSED_DIR / "adjusted_seasons.parquet"

    logger.info("Loaded %d player-%s rows", len(player_data),
                "games" if use_gamelogs else "seasons")

    # ── Physical attributes (always merge independently of position_group) ────
    if attrs_path.exists() and "player_id" in player_data.columns:
        attrs = pd.read_parquet(attrs_path)
        phys_cols = [c for c in ["height_in", "weight_lbs", "class_year_int"] if c in attrs.columns]
        if phys_cols:
            player_data = player_data.merge(
                attrs[["player_id"] + phys_cols].drop_duplicates("player_id"),
                on="player_id", how="left",
            )
        attrs_pg = (attrs[["player_id", "position_group"]]
                    .dropna(subset=["position_group"])
                    .drop_duplicates("player_id")
                    if "position_group" in attrs.columns else pd.DataFrame())
    else:
        attrs_pg = pd.DataFrame()

    # ── Three-tier position_group assignment ──────────────────────────────────
    # Tier 1: preserve any existing gamelogs column as base
    _PG_BASE = "__pg_base__"
    if "position_group" in player_data.columns:
        player_data = player_data.rename(columns={"position_group": _PG_BASE})
    else:
        player_data[_PG_BASE] = pd.NA

    # Tier 2: player_attributes (most authoritative)
    if not attrs_pg.empty and "player_id" in player_data.columns:
        player_data = player_data.merge(attrs_pg, on="player_id", how="left")
    else:
        player_data["position_group"] = pd.NA

    player_data["position_group"] = player_data["position_group"].fillna(player_data[_PG_BASE])
    player_data = player_data.drop(columns=[_PG_BASE], errors="ignore")

    # Tier 3: bridge_players (always current from Step 2)
    if bridge_path.exists() and "player_id" in player_data.columns:
        bp = pd.read_parquet(bridge_path, columns=["player_id", "position_group"])
        bridge_pg = bp.dropna(subset=["position_group"]).drop_duplicates("player_id")
        if not bridge_pg.empty:
            player_data = player_data.merge(
                bridge_pg.rename(columns={"position_group": "position_group_bridge"}),
                on="player_id", how="left",
            )
            mask = player_data["position_group"].isna()
            player_data.loc[mask, "position_group"] = player_data.loc[mask, "position_group_bridge"]
            player_data = player_data.drop(columns=["position_group_bridge"], errors="ignore")

    # Tier 4: derive from position string (last resort)
    if "position_group" not in player_data.columns or player_data["position_group"].isna().any():
        if "position" in player_data.columns:
            mask = player_data.get("position_group", pd.Series(pd.NA, index=player_data.index)).isna()
            if "position_group" not in player_data.columns:
                player_data["position_group"] = pd.NA
            player_data.loc[mask, "position_group"] = (
                player_data.loc[mask, "position"].str.upper().map(POSITION_TO_GROUP)
            )
        player_data["position_group"] = player_data["position_group"].fillna("OTHER")

    pg_counts = player_data["position_group"].value_counts()
    logger.info("Position group coverage: %s", pg_counts.to_dict())

    models = load_models()
    if not models:
        logger.error("No contextual models found — run Step 4 first")
        sys.exit(1)

    n_active = sum(1 for m in models.values() if m.cv_r2 is None or m.cv_r2 >= 0)
    n_skipped = len(models) - n_active
    logger.info("Contextual models: %d active (CV R²≥0), %d skipped (CV R²<0)",
                n_active, n_skipped)

    df = build_context_features(player_data)
    all_groups = []
    # (position_group, stat) -> which semantic adjusted_{stat} ends up carrying.
    # Written to adjusted_stat_semantics.json so consumers can check instead of
    # assuming every adjusted_* column is a residual. See module docstring.
    semantics: dict[str, dict[str, dict]] = {}

    def _note(pos: str, stat_name: str, semantic: str, cv_r2=None) -> None:
        semantics.setdefault(pos, {})[stat_name] = {
            "semantic": semantic,
            "cv_r2": (None if cv_r2 is None else float(cv_r2)),
        }

    for pos_group, stats in MODELED_STATS.items():
        if not stats:
            group_df = df[df["position_group"] == pos_group].copy() \
                if "position_group" in df.columns else df.copy()
            all_groups.append(group_df)
            continue

        group_mask = (df["position_group"] == pos_group) \
            if "position_group" in df.columns else pd.Series(True, index=df.index)
        group_df = df[group_mask].copy()

        if group_df.empty:
            continue

        for stat in stats:
            key = (pos_group, stat)
            if key not in models:
                logger.debug("No model for %s/%s — setting adjusted to raw", pos_group, stat)
                from src.models.contextual import get_stat_column
                src_col = get_stat_column(stat, group_df)
                if src_col in group_df.columns:
                    group_df[f"adjusted_{stat}"] = group_df[src_col]
                _note(pos_group, stat, "no_model")
                continue

            model = models[key]

            # Skip adjustment when expanding-window CV R² is below 0.01 —
            # marginal-positive heads over-correct: residuals acquire spearman
            # ρ up to ±0.6 vs controls that the raw stat had near-zero
            # correlation with, because CV picks coefficients that fit training
            # noise and inference subtracts that noise from the stat.
            if model.cv_r2 is not None and model.cv_r2 < 0.01:
                logger.debug("Skipping %s/%s (CV R²=%.3f < 0.01) — using raw stat",
                             pos_group, stat, model.cv_r2)
                from src.models.contextual import get_stat_column
                src_col = get_stat_column(stat, group_df)
                if src_col in group_df.columns:
                    group_df[f"adjusted_{stat}"] = group_df[src_col]
                _note(pos_group, stat, "raw_fallback", model.cv_r2)
                continue

            try:
                group_df[f"adjusted_{stat}"] = model.compute_residuals(group_df)
                _note(pos_group, stat, "residual", model.cv_r2)
            except Exception as exc:
                logger.error("Failed applying %s/%s: %s", pos_group, stat, exc)
                group_df[f"adjusted_{stat}"] = np.nan
                _note(pos_group, stat, "failed", model.cv_r2)

        all_groups.append(group_df)

    # Players with unknown/other position group
    if "position_group" in df.columns:
        other_mask = ~df["position_group"].isin(MODELED_STATS.keys())
        if other_mask.any():
            all_groups.append(df[other_mask].copy())

    adjusted = pd.concat(all_groups, ignore_index=True)

    # Drop expected_ columns — not consumed downstream, just bloat
    expected_cols = [c for c in adjusted.columns if c.startswith("expected_")]
    if expected_cols:
        adjusted = adjusted.drop(columns=expected_cols)
        logger.info("Dropped %d expected_ columns from output", len(expected_cols))

    adjusted.to_parquet(out_path, index=False)
    logger.info("Saved %d adjusted %s → %s", len(adjusted),
                "game rows" if use_gamelogs else "seasons", out_path)

    adj_cols = [c for c in adjusted.columns if c.startswith("adjusted_")]
    logger.info("Adjusted stat columns: %d", len(adj_cols))
    for col in adj_cols[:5]:
        n = adjusted[col].notna().sum()
        logger.info("  %s: %d non-null values", col, n)

    # ── Semantics sidecar ─────────────────────────────────────────────────────
    # adjusted_{stat} is a residual for some heads and a raw copy for others (see
    # module docstring). Emit the per-head verdict so consumers can CHECK rather
    # than assume, and so a shift in the residual/raw mix between runs is visible
    # instead of silent — that mix moves whenever step 4's CV R² crosses 0.01,
    # which a feature-surface or alpha change can do on its own.
    semantics_path = PROCESSED_DIR / "adjusted_stat_semantics.json"
    with open(semantics_path, "w") as f:
        json.dump(semantics, f, indent=2, sort_keys=True)
    tally: dict[str, int] = {}
    for pos_map in semantics.values():
        for entry in pos_map.values():
            tally[entry["semantic"]] = tally.get(entry["semantic"], 0) + 1
    logger.info("adjusted_* semantics across %d heads: %s → %s",
                sum(tally.values()),
                ", ".join(f"{k}={v}" for k, v in sorted(tally.items())),
                semantics_path)
    if tally.get("failed"):
        logger.error("%d head(s) FAILED residual computation and hold NaN — "
                     "see errors above", tally["failed"])

    logger.info("Step 5 complete.")


if __name__ == "__main__":
    main()
