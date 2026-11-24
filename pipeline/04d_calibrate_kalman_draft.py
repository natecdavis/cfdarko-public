"""
Step 4d: Calibrate Kalman Q Parameters Against NFL Draft Outcomes

Two-stage calibration on bridge players (cal cohort: last_college_season_year ≤ 2020):

  Stage 1 — Global hyperparameters via Nelder-Mead:
      Tunes (age_q_halflife_years, age_q_max_multiplier, transfer_p_multiplier)
      using a Ridge R² proxy objective (impact_R²; role_R² until 2026-10-31) computed
      from terminal Kalman features (kalman_mean, kalman_var) per stat.

  Q and R are data-derived (not optimised):
      Q: year-over-year variance in season averages / interseason_days
      R: within-player within-season variance

This is a one-shot calibration script — NOT part of the main pipeline sequence.
Run it after Step 07 (and ideally after Step 4b), then re-run Steps 06–10.

Usage:
    python pipeline/04d_calibrate_kalman_draft.py

Inputs:
    data/processed/adjusted_gamelogs.parquet     (Step 05)
    data/processed/nfl_draft_labels.parquet      (Step 07)
    data/processed/decay_betas_role.json         (from 04b; falls back to decay_betas.json)

Outputs:
    data/processed/kalman_hyperparams_draft.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PROCESSED_DIR, MODELED_STATS,
    KALMAN_OBS_NOISE, KALMAN_OBS_NOISE_BY_STAT,
    DEFAULT_BETA,
    AGE_Q_HALFLIFE_YEARS, AGE_Q_MAX_MULTIPLIER, TRANSFER_P_MULTIPLIER,
)
from src.models.kalman import (
    estimate_q_from_yoy_variance,
    calibrate_kalman_for_draft,
    estimate_prior_means,
    estimate_prior_vars,
)
from src.utils.cache_manifest import (
    write_manifest, tag_summary,
    json_content_fingerprint, parquet_metadata_fingerprint,
)

STEP_NAME = "04d_calibrate_kalman_draft"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CAL_SEASON_CUTOFF = 2020
INTERSEASON_DAYS  = 240.0


def main():
    labels_path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    if not labels_path.exists():
        logger.warning(
            "nfl_draft_labels.parquet not found — skipping draft Kalman calibration.\n"
            "Run the full pipeline through Step 07 first, then re-run this script."
        )
        sys.exit(0)

    adj_path = PROCESSED_DIR / "adjusted_gamelogs.parquet"
    if not adj_path.exists():
        logger.error("adjusted_gamelogs.parquet not found — run Step 05 first")
        sys.exit(1)

    logger.info("Loading adjusted gamelogs...")
    adjusted = pd.read_parquet(adj_path)
    adjusted["game_date"] = pd.to_datetime(adjusted["game_date"])
    logger.info("Loaded %d rows, %d cols", *adjusted.shape)

    # Join birth_year so the age-Q process-noise multiplier is live during
    # calibration — without it the (halflife, max_mult) dimensions of the
    # Nelder-Mead objective are flat and the "calibrated" values are just
    # the starting point echoed back. Mirrors step 06's join.
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    if attrs_path.exists():
        import pyarrow.parquet as _pq
        if "birth_year" in _pq.read_schema(attrs_path).names:
            _attrs_age = (pd.read_parquet(attrs_path,
                                          columns=["player_id", "birth_year"])
                          .dropna(subset=["birth_year"]))
            adjusted = adjusted.merge(_attrs_age, on="player_id", how="left")
            logger.info("Birth year joined for %.0f%% of player-game rows",
                        adjusted["birth_year"].notna().mean() * 100)
        else:
            logger.warning("birth_year not in player_attributes — age-Q dims "
                           "will be flat during calibration (re-run Step 3)")

    logger.info("Loading NFL draft labels...")
    label_cols = ["player_id", "role_target", "last_college_season_year", "is_bridge"]
    for opt in ("impact_target",):
        label_cols.append(opt)
    role_labels = pd.read_parquet(labels_path, columns=label_cols)
    role_labels = role_labels[role_labels["is_bridge"]].dropna(
        subset=["role_target", "impact_target", "player_id"])
    logger.info("Bridge players with role labels: %d", len(role_labels))
    logger.info(
        "Cal cohort (≤%d): %d | Val cohort (>%d): %d",
        CAL_SEASON_CUTOFF,
        (role_labels["last_college_season_year"] <= CAL_SEASON_CUTOFF).sum(),
        CAL_SEASON_CUTOFF,
        (role_labels["last_college_season_year"] > CAL_SEASON_CUTOFF).sum(),
    )

    # ── Stats to calibrate ─────────────────────────────────────────────────
    all_stats = sorted({s for stats in MODELED_STATS.values() for s in stats})
    stats_to_run = [s for s in all_stats if f"adjusted_{s}" in adjusted.columns]
    logger.info("Stats: %d", len(stats_to_run))

    # ── Load betas (prefer role-calibrated) ───────────────────────────────
    betas_role_path = PROCESSED_DIR / "decay_betas_role.json"
    betas_path      = PROCESSED_DIR / "decay_betas.json"
    if betas_role_path.exists():
        betas = json.loads(betas_role_path.read_text())
        logger.info("Using role-calibrated betas from %s", betas_role_path)
    elif betas_path.exists():
        betas = json.loads(betas_path.read_text())
        logger.info("Using next-game-MAE betas from %s", betas_path)
    else:
        betas = {s: DEFAULT_BETA.get(s, 0.999) for s in stats_to_run}
        logger.warning("No cached betas found — using defaults")

    # ── Data-derived observation noise (within-player variance) ───────────
    logger.info("Estimating observation noise from within-player variance...")
    obs_noises: dict[str, float] = {}
    for stat in stats_to_run:
        adj_col = f"adjusted_{stat}"
        within_var = (
            adjusted.dropna(subset=[adj_col])
            .groupby("player_id")[adj_col]
            .var()
            .dropna()
        )
        obs_noises[stat] = (
            float(within_var.median()) if len(within_var) > 0
            else KALMAN_OBS_NOISE_BY_STAT.get(stat, KALMAN_OBS_NOISE)
        )
    logger.info("Estimated R (obs noise) for %d stats", len(obs_noises))

    # ── Data-derived process noise Q (year-over-year variance) ────────────
    logger.info("Estimating Q from year-over-year variance...")
    process_noises = estimate_q_from_yoy_variance(
        adjusted, stats_to_run, interseason_days=INTERSEASON_DAYS,
    )

    # ── Data-driven priors ─────────────────────────────────────────────────
    prior_means = estimate_prior_means(adjusted, stats_to_run)
    prior_vars  = estimate_prior_vars(adjusted, stats_to_run, obs_noises=obs_noises)

    # ── Calibrate Kalman global hyperparameters ───────────────────────────
    logger.info(
        "Stage 1: Nelder-Mead over (age_q_halflife, age_q_max_mult, transfer_p_mult)\n"
        "  Objective: impact_R² only (switched from role 2026-10-31)\n"
        "  Q: data-derived (YoY variance)  R: data-derived (within-player variance)"
    )
    params, evals = calibrate_kalman_for_draft(
        adjusted_games=adjusted,
        role_labels=role_labels,
        betas=betas,
        obs_noises=obs_noises,
        process_noises=process_noises,
        stats=stats_to_run,
        cal_season_cutoff=CAL_SEASON_CUTOFF,
        role_weight=0.0,
        impact_weight=1.0,
        n_global_iter=200,
        param_bounds=((0.5, 10.0), (1.0, 8.0), (1.0, 10.0)),
        prior_means=prior_means,
        prior_vars=prior_vars,
        age_col="birth_year" if "birth_year" in adjusted.columns else None,
    )

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "kalman_hyperparams_draft.json"
    payload = tag_summary(
        {**params, **evals},
        mode_used="fresh", step_name=STEP_NAME,
    )
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    fingerprint = {
        "stats":                  sorted(stats_to_run),
        "cal_season_cutoff":      CAL_SEASON_CUTOFF,
        "interseason_days":       INTERSEASON_DAYS,
        "decay_betas_role_hash":  json_content_fingerprint(betas_role_path),
        "nfl_labels_file":        parquet_metadata_fingerprint(labels_path),
        "param_bounds":           [[0.5, 10.0], [1.0, 8.0], [1.0, 10.0]],
        "objective":              "impact_target",
    }
    write_manifest(
        out_path, fingerprint=fingerprint,
        mode_used="fresh", step_name=STEP_NAME,
    )
    logger.info("Saved draft-calibrated Kalman hyperparams → %s", out_path)

    logger.info(
        "\n  age_q_halflife_years  = %.3f"
        "\n  age_q_max_multiplier  = %.3f"
        "\n  transfer_p_multiplier = %.3f",
        params["age_q_halflife_years"],
        params["age_q_max_multiplier"],
        params["transfer_p_multiplier"],
    )
    logger.info(
        "\n  role_R²_cal=%.4f  role_R²_val=%.4f"
        "\n  impact_R²_cal=%.4f  impact_R²_val=%.4f"
        "\n  combined_cal=%.4f  combined_val=%.4f",
        evals["role_r2_cal"], evals["role_r2_val"],
        evals["impact_r2_cal"], evals["impact_r2_val"],
        evals["combined_score_cal"], evals["combined_score_val"],
    )

    if evals["combined_score_cal"] > 0:
        ratio = evals["combined_score_val"] / evals["combined_score_cal"]
        if ratio < 0.7:
            logger.warning("Val/cal ratio %.2f — possible overfit", ratio)
        else:
            logger.info("Val/cal ratio: %.2f (healthy)", ratio)

    logger.info(
        "\n=== Step 4d complete ===\n"
        "Re-run skill estimation to apply draft-calibrated Kalman params:\n"
        "    python pipeline/06_skill_estimation.py\n"
        "    python pipeline/08_draft_models.py\n"
        "    python pipeline/09_comps.py\n"
        "    python pipeline/10_output.py"
    )


if __name__ == "__main__":
    main()
