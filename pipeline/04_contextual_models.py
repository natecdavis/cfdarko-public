"""
Step 4: Contextual Model Fitting (Game-Level)

Fits per-stat, per-position-group Ridge regression models trained ONLY on
bridge player game logs. Models adjust snap-normalized stats for context
(opponent strength, location, weather, time-of-day).

Context features are opponent-only rolling advanced stats (de-meaned within
player-season) — no team-side features. Analogous to cdarko.

Output: data/processed/contextual_models/{pos_group}_{stat}_model.pkl
        data/processed/contextual_models/model_summary.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DIR, PROCESSED_DIR, MODELED_STATS, POSITION_GROUPS,
    BRIDGE_SEASON_GAMMA, END_YEAR, CONTEXT_FEATURES,
    CONTEXT_FEATURE_DROP_BY_POSITION,
)
from src.models.contextual import fit_contextual_models_for_group
from config import RIDGE_ALPHA_GRID, ELASTICNET_L1_RATIOS
from src.utils.cache_manifest import (
    write_manifest, check_manifest, log_stale_warning, tag_summary,
    schema_fingerprint, file_size_mtime_fingerprint, manifest_path_for,
)

STEP_NAME = "04_contextual_models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MIN_CV_R2_WARN = -0.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers for per-stat fits (-1 = all cores, 1 = serial)")
    parser.add_argument("--use-cached-alpha", action="store_true",
                        help="Skip Ridge/ElasticNet CV grid; reuse best_alpha + best_l1_ratio "
                             "from a prior model_summary.json. Stats missing from the cache "
                             "fall back to the full grid.")
    return parser.parse_args()


def _load_cached_alphas(summary_path: Path) -> dict[tuple[str, str], dict]:
    """Read prior model_summary.json into a (pos_group, stat) -> params map.
    Returns {} if the file is missing or unreadable."""
    if not summary_path.exists():
        return {}
    try:
        rows = json.loads(summary_path.read_text())
    except Exception:
        return {}
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        pg, stat = row.get("position_group"), row.get("stat")
        if pg is None or stat is None:
            continue
        if row.get("best_alpha") is None:
            continue
        out[(pg, stat)] = {
            "best_alpha":         row["best_alpha"],
            "best_l1_ratio":      row.get("best_l1_ratio"),
            "is_adaptive_lasso":  bool(row.get("is_adaptive_lasso", False)),
        }
    return out


def main():
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    gamelogs_path = RAW_DIR / "player_gamelogs.parquet"
    seasons_path = RAW_DIR / "player_seasons.parquet"
    bridge_path = PROCESSED_DIR / "bridge_players.parquet"
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"

    if not bridge_path.exists():
        logger.error("bridge_players.parquet not found — run Step 2 first")
        sys.exit(1)

    use_gamelogs = gamelogs_path.exists()
    if use_gamelogs:
        logger.info("Loading game-level data from %s", gamelogs_path)
        player_data = pd.read_parquet(gamelogs_path)
        date_col = "game_date"
    else:
        logger.warning("player_gamelogs.parquet not found — falling back to season-level data")
        if not seasons_path.exists():
            logger.error("player_seasons.parquet not found — run Step 1 first")
            sys.exit(1)
        player_data = pd.read_parquet(seasons_path)
        date_col = "season_year"

    bridge_players = pd.read_parquet(bridge_path)

    # ── Three-tier position_group assignment ─────────────────────────────────
    # Tier 1: rename gamelogs column to a temp name to avoid merge conflicts
    _PG_BASE = "__pg_base__"
    if "position_group" in player_data.columns:
        player_data = player_data.rename(columns={"position_group": _PG_BASE})
    else:
        player_data[_PG_BASE] = pd.NA

    # Tier 2: player_attributes — most authoritative (cleaned CFBD roster mapping)
    if attrs_path.exists() and "player_id" in player_data.columns:
        attrs = pd.read_parquet(attrs_path, columns=["player_id", "position_group"])
        player_data = player_data.merge(attrs, on="player_id", how="left")
    else:
        player_data["position_group"] = pd.NA

    # Where attrs had no value, fall back to gamelogs (Tier 1)
    player_data["position_group"] = player_data["position_group"].fillna(player_data[_PG_BASE])
    player_data = player_data.drop(columns=[_PG_BASE], errors="ignore")

    # Tier 3: bridge_players fill remaining NaN (always current from Step 2)
    if "player_id" in bridge_players.columns and "position_group" in bridge_players.columns:
        bridge_pg = (bridge_players[["player_id", "position_group"]]
                     .dropna(subset=["position_group"])
                     .drop_duplicates("player_id"))
        player_data = player_data.merge(
            bridge_pg.rename(columns={"position_group": "position_group_bridge"}),
            on="player_id", how="left",
        )
        mask = player_data["position_group"].isna()
        player_data.loc[mask, "position_group"] = player_data.loc[mask, "position_group_bridge"]
        player_data = player_data.drop(columns=["position_group_bridge"], errors="ignore")

    # ── Filter to bridge players ──────────────────────────────────────────────
    if "player_id" in bridge_players.columns and bridge_players["player_id"].notna().any():
        bridge_ids = set(bridge_players["player_id"].dropna().astype(str))
        if "player_id" in player_data.columns:
            bridge_data = player_data[player_data["player_id"].astype(str).isin(bridge_ids)].copy()
        else:
            logger.warning("No player_id column — using all data as bridge data")
            bridge_data = player_data.copy()
    elif "player_name" in bridge_players.columns:
        bridge_names = set(bridge_players["player_name"].dropna().str.lower())
        if "player_name" in player_data.columns:
            bridge_data = player_data[player_data["player_name"].str.lower().isin(bridge_names)].copy()
        else:
            bridge_data = player_data.copy()
    else:
        bridge_data = player_data.copy()

    logger.info("Bridge player %s: %d rows", "games" if use_gamelogs else "seasons", len(bridge_data))

    # ── Ensure conference_tier is present for contextual model dummies ────────
    # Prefer conference column already in gamelogs (per-season from CFBD).
    # Fall back to player_attributes which has it as a player-level attribute.
    if "conference_tier" not in bridge_data.columns and "conference" not in bridge_data.columns:
        if attrs_path.exists() and "player_id" in bridge_data.columns:
            _attrs_tier = pd.read_parquet(attrs_path, columns=["player_id", "conference_tier"])
            bridge_data = bridge_data.merge(_attrs_tier, on="player_id", how="left")
            logger.info("Joined conference_tier from player_attributes for contextual models")

    # ── Bridge weight: down-weight older seasons ──────────────────────────────
    bridge_data["bridge_weight"] = BRIDGE_SEASON_GAMMA ** (
        END_YEAR - bridge_data["season_year"].fillna(END_YEAR)
    )

    if args.test:
        bridge_data = bridge_data.sample(min(1000, len(bridge_data)), random_state=42)

    models_dir = PROCESSED_DIR / "contextual_models"
    models_dir.mkdir(exist_ok=True)

    team_ratings = pd.DataFrame()

    # Build the fingerprint dict that captures everything that, if changed,
    # would invalidate the cached alphas: candidate context features, the
    # training-data shape (rows + season range), the grid we'd search, and
    # the bridge_players artifact's size+mtime. Computed *before* the per-pos
    # slicing loop so it reflects the whole bridge cohort.
    fingerprint = {
        "context_features":     sorted(CONTEXT_FEATURES),
        "bridge_data_shape":    schema_fingerprint(bridge_data, season_col="season_year"),
        "ridge_alpha_grid":     list(RIDGE_ALPHA_GRID),
        "elasticnet_l1_ratios": list(ELASTICNET_L1_RATIOS),
        "bridge_players_file":  file_size_mtime_fingerprint(bridge_path),
    }

    # Load cached alphas for the fast path. On stale, surface the actual diffs
    # so the user can judge whether to retune now.
    summary_path = models_dir / "model_summary.json"
    cached_alphas: dict[tuple[str, str], dict] = {}
    mode_used: str = "fresh"
    if args.use_cached_alpha:
        cached_alphas = _load_cached_alphas(summary_path)
        if not cached_alphas:
            logger.warning(
                "--use-cached-alpha: no usable cache at %s — falling back to full grid",
                summary_path,
            )
        else:
            result = check_manifest(summary_path, fingerprint=fingerprint)
            if result.is_stale:
                log_stale_warning(
                    summary_path,
                    "python run_pipeline.py --retune-only 4",
                    diffs=result.diffs,
                )
                mode_used = "cached-stale"
            elif result.is_missing:
                logger.warning(
                    "--use-cached-alpha: no manifest sidecar at %s — cannot "
                    "validate that cached alphas are still consistent with "
                    "current inputs (CONTEXT_FEATURES, RIDGE_ALPHA_GRID, "
                    "bridge_players, etc.). Proceeding anyway. To validate, "
                    "re-run without --use-cached-alpha (or "
                    "`python run_pipeline.py --retune-only 4`); to silence, "
                    "delete %s and rerun.",
                    summary_path, summary_path,
                )
                mode_used = "cached"
            else:
                mode_used = "cached"
            logger.info("--use-cached-alpha: loaded cached alphas for %d (pos, stat) entries",
                        len(cached_alphas))

    summary_rows: list[dict] = []
    total_models = 0
    for pos_group, stats in MODELED_STATS.items():
        if not stats:
            logger.info("Skipping %s (no stats to model)", pos_group)
            continue

        if "position_group" in bridge_data.columns:
            group_data = bridge_data[bridge_data["position_group"] == pos_group]
        else:
            group_data = bridge_data

        logger.info("  %s bridge rows: %d", pos_group, len(group_data))

        min_rows = 100 if use_gamelogs else 50
        if len(group_data) < min_rows:
            logger.warning("%s: only %d %s — skipping contextual models",
                           pos_group, len(group_data),
                           "game rows" if use_gamelogs else "seasons")
            continue

        logger.info("Fitting contextual models for %s (%d %s, %d stats)",
                    pos_group, len(group_data),
                    "games" if use_gamelogs else "seasons",
                    len(stats))

        # Per-stat cached params for this pos_group (None for stats not in cache)
        group_cached: dict[str, dict] = {}
        if cached_alphas:
            for s in stats:
                if (pos_group, s) in cached_alphas:
                    group_cached[s] = cached_alphas[(pos_group, s)]
            n_cached = len(group_cached)
            n_missing = len(stats) - n_cached
            if n_missing > 0:
                logger.info("  %s: %d cached, %d falling back to full grid",
                            pos_group, n_cached, n_missing)

        drop = CONTEXT_FEATURE_DROP_BY_POSITION.get(pos_group, [])
        feature_pool = [f for f in CONTEXT_FEATURES if f not in drop] if drop else None
        if drop:
            logger.info("  %s: dropping %d features per CONTEXT_FEATURE_DROP_BY_POSITION (%d remain)",
                        pos_group, len(drop), len(feature_pool))

        models = fit_contextual_models_for_group(
            bridge_seasons=group_data,
            team_ratings=team_ratings,
            stats=stats,
            position_group=pos_group,
            save_dir=models_dir,
            weight_col="bridge_weight",
            n_jobs=args.n_jobs,
            cached_params=group_cached or None,
            feature_pool=feature_pool,
        )
        total_models += len(models)
        logger.info("  Fit %d models for %s", len(models), pos_group)

        for stat, model in models.items():
            cv_r2 = model.cv_r2
            is_al = getattr(model, "is_adaptive_lasso", False)
            best_l1 = getattr(model, "best_l1_ratio", None)
            if is_al:
                model_type = "AdaptiveLasso"
            elif best_l1 is not None:
                model_type = "ElasticNet"
            else:
                model_type = "Ridge"
            summary_rows.append({
                "position_group": pos_group,
                "stat": stat,
                "stat_col": model._stat_col,
                "cv_r2": round(cv_r2, 4) if cv_r2 is not None else None,
                "model_type": model_type,
                "best_alpha": getattr(model, "best_alpha", None),
                "best_l1_ratio": best_l1,
                "is_adaptive_lasso": is_al,
                "n_features": len(model.feature_cols),
            })
            if cv_r2 is not None and cv_r2 < MIN_CV_R2_WARN:
                logger.warning(
                    "Poor model quality: %s/%s CV R²=%.3f (below threshold %.1f) — "
                    "model saved but context correction will be unreliable",
                    pos_group, stat, cv_r2, MIN_CV_R2_WARN,
                )

    # ── Write model summary JSON ──────────────────────────────────────────────
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info("Model summary → %s (%d models)", summary_path, len(summary_rows))

    # Sidecar manifest hashes everything that, if changed, would invalidate
    # the cached alphas. A future --use-cached-alpha run compares against this.
    manifest_path = write_manifest(
        summary_path,
        fingerprint=fingerprint,
        mode_used=mode_used,
        step_name=STEP_NAME,
        extra={"n_models": len(summary_rows)},
    )
    logger.info("Manifest → %s", manifest_path)

    # Provenance sidecar — model_summary.json itself is a JSON list (consumed
    # as such by _load_cached_alphas), so the _tuning_mode tag goes here
    # rather than wrapping the list. Readers comparing cv_r2 across runs
    # should consult this file to know whether numbers came from a fresh
    # tune vs a cached one.
    provenance_path = models_dir / "model_summary_provenance.json"
    provenance_payload = tag_summary(
        {"n_models": len(summary_rows), "use_cached_alpha": bool(args.use_cached_alpha)},
        mode_used=mode_used,
        step_name=STEP_NAME,
    )
    provenance_path.write_text(json.dumps(provenance_payload, indent=2))
    logger.info("Provenance → %s (_tuning_mode=%s)", provenance_path, mode_used)

    # ── Print CV R² summary table ─────────────────────────────────────────────
    if summary_rows:
        r2_grid: dict[str, dict[str, str]] = defaultdict(dict)
        for row in summary_rows:
            v = row["cv_r2"]
            r2_grid[row["position_group"]][row["stat"]] = f"{v:.3f}" if v is not None else "N/A"

        fitted_stats = sorted({row["stat"] for row in summary_rows})
        fitted_groups = [g for g in MODELED_STATS if g in r2_grid]

        col_w = max(len(s) for s in fitted_stats) + 2
        header = f"{'Group':<8}  " + "  ".join(f"{s:<{col_w}}" for s in fitted_stats)
        logger.info("── CV R² Summary ──")
        logger.info(header)
        for grp in fitted_groups:
            vals = "  ".join(f"{r2_grid[grp].get(s, 'N/A'):<{col_w}}" for s in fitted_stats)
            logger.info(f"{grp:<8}  {vals}")

    logger.info("Total models fit: %d", total_models)
    logger.info("Step 4 complete.")


if __name__ == "__main__":
    main()
