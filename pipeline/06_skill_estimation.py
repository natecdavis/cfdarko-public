"""
Step 6: Skill Estimation (Game-Level Kalman + EWMA)

Loads adjusted game logs and runs game-by-game Kalman filter and EWMA.

For each player-game, computes:
  - kalman_mean_{stat}:  Bayesian true-talent estimate after this game
  - kalman_var_{stat}:   Uncertainty
  - ewma_{stat}:         Exponentially-weighted game average
  - ewma_var_{stat}:     EWMA variance
  - kalman_yoy_{stat}:   Year-over-year change (last game of season N vs N-1)

Also saves a per-player snapshot (latest game's Kalman/EWMA values)
for use as draft model features.

Data-driven priors:
  - P₀ estimated via method-of-moments: Var(player_means) - R/mean_n
  - Prior means ≈ 0 (calibrated residuals)
  - EWMA prior weight = R/P₀, clamped [1, 20]

Outputs:
  data/processed/skill_estimates.parquet       — all player-games
  data/processed/player_skill_snapshot.parquet — latest game per player
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, RAW_DIR, MODELED_STATS, DEFAULT_BETA,
    KALMAN_OBS_NOISE_BY_STAT, KALMAN_OBS_NOISE,
    AGE_Q_HALFLIFE_YEARS, AGE_Q_MAX_MULTIPLIER, TRANSFER_P_MULTIPLIER,
    KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT, KALMAN_PROCESS_NOISE_PER_DAY,
    POSITION_GROUPS,
)
from src.models.kalman import (
    run_kalman_all_players,
    estimate_prior_vars,
    estimate_prior_means,
)
from src.models.decay import ewma_time_series_fast, ewma_all_players_fast, optimize_all_betas
from src.utils.cache_manifest import (
    write_manifest, check_manifest, log_stale_warning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip-beta-opt", action="store_true",
                        help="Skip beta optimization (use default)")
    parser.add_argument("--force-beta-opt", action="store_true",
                        help="Re-run inline beta optimization even if a cached "
                             "decay_betas*.json exists. Ignored when --skip-beta-opt is set.")
    return parser.parse_args()


def _add_yoy_features(df: pd.DataFrame, stats: list[str],
                       date_col: str = "game_date") -> pd.DataFrame:
    """
    Add year-over-year change features.
    YoY = last game's estimate in season Y - last game's estimate in season Y-1.
    """
    if "season_year" not in df.columns:
        return df

    df = df.sort_values(["player_id", date_col])
    idx = df.set_index(["player_id", "season_year"]).index
    new_cols: dict[str, pd.Series] = {}

    for stat in stats:
        for prefix in ("kalman_mean", "ewma"):
            col = f"{prefix}_{stat}"
            if col not in df.columns:
                continue
            last_by_season = (df.groupby(["player_id", "season_year"])[col]
                              .last().reset_index()
                              .rename(columns={col: f"_last_{col}"}))
            last_by_season["season_year_prev"] = last_by_season["season_year"] - 1
            merged = last_by_season.merge(
                last_by_season[["player_id", "season_year", f"_last_{col}"]].rename(
                    columns={"season_year": "season_year_prev",
                             f"_last_{col}": f"_prev_{col}"}
                ),
                on=["player_id", "season_year_prev"], how="left",
            )
            merged_idx = merged.set_index(["player_id", "season_year"])
            yoy_map = (merged_idx[f"_last_{col}"] - merged_idx[f"_prev_{col}"]).to_dict()
            prefix_key = prefix.split("_")[0]
            yoy_col = f"{prefix_key}_yoy_{stat}"
            new_cols[yoy_col] = pd.Series(idx.map(yoy_map.get), index=df.index, dtype=float)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


def main():
    args = parse_args()

    # Prefer game-level adjusted data; fall back to season-level
    adj_game_path = PROCESSED_DIR / "adjusted_gamelogs.parquet"
    adj_season_path = PROCESSED_DIR / "adjusted_seasons.parquet"

    if adj_game_path.exists():
        adjusted = pd.read_parquet(adj_game_path)
        date_col = "game_date"
        logger.info("Loaded %d adjusted game rows", len(adjusted))

        # ── Per-PLAY residual streams (step 5p) ───────────────────────────────
        # Left-joined on (player_id, game_id). These add adjusted_play_mean_resid_*
        # and adjusted_play_total_resid_* columns that the stats-detection loop
        # below picks up via the f"adjusted_{stat}" pattern.
        play_resid_path = PROCESSED_DIR / "adjusted_player_plays_per_game.parquet"
        if play_resid_path.exists():
            play = pd.read_parquet(play_resid_path)
            keep_cols = ["player_id", "game_id"] + [
                c for c in play.columns
                if c.startswith("adjusted_play_mean_resid_")
                   or c.startswith("adjusted_play_total_resid_")
                   or c.startswith("adjusted_play_rate_")
                   or c.startswith("adjusted_play_raw_mean_")
            ]
            keep_cols = [c for c in keep_cols if c in play.columns]
            if "player_id" in adjusted.columns and "game_id" in adjusted.columns:
                adjusted["player_id"] = adjusted["player_id"].astype(str)
                play["player_id"]     = play["player_id"].astype(str)
                adjusted = adjusted.merge(
                    play[keep_cols].drop_duplicates(["player_id", "game_id"]),
                    on=["player_id", "game_id"], how="left",
                )
                n_play_cols = sum(1 for c in keep_cols if c.startswith("adjusted_"))
                logger.info("Joined %d play-grain residual columns from step 5p",
                            n_play_cols)
        else:
            logger.info("adjusted_player_plays_per_game.parquet not found — "
                        "play-grain Kalman streams will be skipped")
    elif adj_season_path.exists():
        adjusted = pd.read_parquet(adj_season_path)
        date_col = "season_year"
        logger.warning("Using season-level adjusted data (game-level not found)")
        logger.info("Loaded %d adjusted seasons", len(adjusted))
    else:
        logger.error("No adjusted data found — run Step 5 first")
        sys.exit(1)

    if args.test:
        max_year = adjusted["season_year"].max() if "season_year" in adjusted.columns else 2026
        if "season_year" in adjusted.columns:
            adjusted = adjusted[adjusted["season_year"] >= max_year - 3]
        sample_ids = adjusted["player_id"].drop_duplicates().sample(
            min(200, adjusted["player_id"].nunique()), random_state=42
        )
        adjusted = adjusted[adjusted["player_id"].isin(sample_ids)]
        logger.info("Test mode: %d rows, %d players", len(adjusted), sample_ids.nunique())

    # Ensure date column is datetime for game-level data
    if date_col == "game_date":
        adjusted[date_col] = pd.to_datetime(adjusted[date_col])

    # Stats to run: those with adjusted_{stat} columns present
    all_stats = sorted({s for stats in MODELED_STATS.values() for s in stats})
    stats_to_run = sorted(s for s in all_stats if f"adjusted_{s}" in adjusted.columns)
    # Auto-detect play-grain stats from step 5p. Any column with the pattern
    # adjusted_play_(mean_resid|total_resid|rate)_<X> becomes an extra stat.
    play_stats_to_run = sorted({
        c[len("adjusted_"):] for c in adjusted.columns
        if c.startswith("adjusted_play_")
    })
    stats_to_run = stats_to_run + play_stats_to_run
    logger.info("Running Kalman + EWMA for %d stats (%d game-grain + %d play-grain)",
                len(stats_to_run), len(stats_to_run) - len(play_stats_to_run),
                len(play_stats_to_run))

    # ── Observation noise table ────────────────────────────────────────────────
    # Game-grain stats use the hand-tuned config table. Play-grain streams have
    # no entries there and were silently falling back to KALMAN_OBS_NOISE=2.0 —
    # orders of magnitude too large for indicator-rate residuals (within-player
    # var ~0.02-0.05), which drove Kalman gain → 0 and anchored every such
    # stream at its prior. Estimate R per play stream from within-player
    # variance of the observations (the same estimator 04d uses).
    obs_noise_table = {s: KALMAN_OBS_NOISE_BY_STAT.get(s, KALMAN_OBS_NOISE)
                       for s in stats_to_run}
    for stat in play_stats_to_run:
        adj_col = f"adjusted_{stat}"
        if adj_col not in adjusted.columns:
            continue
        within_var = (adjusted.dropna(subset=[adj_col])
                      .groupby("player_id")[adj_col].var().dropna())
        if len(within_var) >= 50:
            obs_noise_table[stat] = max(float(within_var.median()), 1e-4)
    logger.info("Play-grain R estimated from within-player variance for %d/%d streams "
                "(sample: %s)",
                sum(1 for s in play_stats_to_run
                    if obs_noise_table.get(s, KALMAN_OBS_NOISE) != KALMAN_OBS_NOISE),
                len(play_stats_to_run),
                {s: f"{obs_noise_table[s]:.4f}" for s in play_stats_to_run[:3]})

    # ── Data-driven priors (method-of-moments, analogous to cdarko) ────────────
    logger.info("Estimating data-driven priors...")
    prior_vars = estimate_prior_vars(
        adjusted, stats_to_run,
        obs_noises=obs_noise_table,
    )
    prior_means = estimate_prior_means(adjusted, stats_to_run)

    # EWMA prior weight = R/P₀, clamped [1, 20] (same as cdarko)
    ewma_prior_weight: dict[str, float] = {}
    for stat in stats_to_run:
        R = obs_noise_table[stat]
        P0 = prior_vars.get(stat, 1.0)
        ewma_prior_weight[stat] = float(np.clip(R / max(P0, 1e-6), 1.0, 20.0))

    logger.info("Prior P₀ estimates: %s",
                {s: f"{v:.3f}" for s, v in list(prior_vars.items())[:5]})

    # ── Optimize beta per stat ──────────────────────────────────────────────────
    # Priority: --force-beta-opt → role-calibrated (04b) → next-game-MAE cached → compute live
    betas_role_path = PROCESSED_DIR / "decay_betas_role.json"
    betas_path      = PROCESSED_DIR / "decay_betas.json"
    # Consumer-side fingerprint: the only thing step 6 cares about is whether
    # the calibrated betas cover the current `stats` set. Producer (04b/04d)
    # writes a richer fingerprint; the asymmetric diff ignores producer-only keys.
    stats_fingerprint = {"stats": sorted(stats_to_run)}

    if args.test:
        betas = {s: DEFAULT_BETA.get(s, 0.999) for s in stats_to_run}
        logger.info("Using default per-day betas (test mode)")
    elif args.force_beta_opt:
        logger.info("--force-beta-opt: re-optimizing decay betas (cache override)...")
        betas = optimize_all_betas(adjusted, stats_to_run)
        with open(betas_path, "w") as f:
            json.dump(betas, f, indent=2)
        write_manifest(
            betas_path, fingerprint=stats_fingerprint,
            mode_used="fresh", step_name="06_skill_estimation",
        )
        logger.info("Saved decay betas → %s", betas_path)
    elif betas_role_path.exists():
        with open(betas_role_path) as f:
            betas = json.load(f)
        logger.info("Loaded role-calibrated decay betas from %s", betas_role_path)
        result = check_manifest(betas_role_path, fingerprint=stats_fingerprint)
        if result.is_stale:
            log_stale_warning(betas_role_path,
                              "python pipeline/04b_calibrate_betas_role.py",
                              diffs=result.diffs)
        for s in [s for s in stats_to_run if s not in betas]:
            betas[s] = DEFAULT_BETA.get(s, 0.999)
    elif betas_path.exists():
        with open(betas_path) as f:
            betas = json.load(f)
        logger.info("Loaded cached decay betas (next-game MAE) from %s", betas_path)
        result = check_manifest(betas_path, fingerprint=stats_fingerprint)
        if result.is_stale:
            log_stale_warning(betas_path,
                              "python run_pipeline.py --step 6 -- --force-beta-opt",
                              diffs=result.diffs)
        for s in [s for s in stats_to_run if s not in betas]:
            betas[s] = DEFAULT_BETA.get(s, 0.999)
    elif args.skip_beta_opt:
        # --skip-beta-opt means "never optimize live" (the no-tune contract:
        # reuse cached params) — it must NOT bypass the calibrated caches
        # above. Reaching here means no cache exists at all; fall back to
        # defaults rather than a slow live optimization.
        betas = {s: DEFAULT_BETA.get(s, 0.999) for s in stats_to_run}
        logger.warning("--skip-beta-opt with no beta cache on disk — using "
                       "default per-day betas (run 04b or --force-beta-opt)")
    else:
        logger.info("Optimizing decay betas (this may take a while)...")
        betas = optimize_all_betas(adjusted, stats_to_run)
        with open(betas_path, "w") as f:
            json.dump(betas, f, indent=2)
        write_manifest(
            betas_path, fingerprint=stats_fingerprint,
            mode_used="fresh", step_name="06_skill_estimation",
        )
        logger.info("Saved decay betas → %s", betas_path)

    # ── Join birth_year for age-based process noise multiplier ─────────────────
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    if attrs_path.exists() and "player_id" in adjusted.columns:
        import pyarrow.parquet as _pq
        _avail_cols = _pq.read_schema(attrs_path).names
        if "birth_year" not in _avail_cols:
            logger.warning("birth_year not in player_attributes — skipping age-Q multiplier (re-run Step 3)")
            _attrs_age = pd.DataFrame(columns=["player_id", "birth_year"])
        else:
            _attrs_age = pd.read_parquet(attrs_path, columns=["player_id", "birth_year"])
        _attrs_age = _attrs_age.dropna(subset=["birth_year"])
        if not _attrs_age.empty:
            adjusted = adjusted.merge(_attrs_age, on="player_id", how="left")
            n_age = adjusted["birth_year"].notna().sum()
            logger.info("Birth year joined for %.0f%% of player-game rows",
                        n_age / len(adjusted) * 100)

    # ── Kalman hyperparameter priority-tier loading ─────────────────────────────
    # Priority: draft-calibrated (04d) → cached → config defaults
    kalman_draft_path = PROCESSED_DIR / "kalman_hyperparams_draft.json"
    kalman_params_path = PROCESSED_DIR / "kalman_hyperparams.json"

    _age_q_halflife  = AGE_Q_HALFLIFE_YEARS
    _age_q_max_mult  = AGE_Q_MAX_MULTIPLIER
    _transfer_p_mult = TRANSFER_P_MULTIPLIER
    _q_by_stat: Optional[dict] = None
    _obs_noises_base = dict(obs_noise_table)

    if kalman_draft_path.exists():
        with open(kalman_draft_path) as f:
            _kp = json.load(f)
        logger.info("Loaded draft-calibrated Kalman Q params from %s", kalman_draft_path)
        result = check_manifest(kalman_draft_path, fingerprint=stats_fingerprint)
        if result.is_stale:
            log_stale_warning(kalman_draft_path,
                              "python pipeline/04d_calibrate_kalman_draft.py",
                              diffs=result.diffs)
        _age_q_halflife  = float(_kp.get("age_q_halflife_years",  _age_q_halflife))
        _age_q_max_mult  = float(_kp.get("age_q_max_multiplier",  _age_q_max_mult))
        _transfer_p_mult = float(_kp.get("transfer_p_multiplier", _transfer_p_mult))
        if "q_multipliers" in _kp:
            _q_by_stat = {
                stat: KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT.get(stat, KALMAN_PROCESS_NOISE_PER_DAY)
                      * float(mult)
                for stat, mult in _kp["q_multipliers"].items()
                if stat in stats_to_run
            }
            logger.info("Applying per-stat Q multipliers from draft calibration (%d stats)",
                        len(_q_by_stat))
    elif kalman_params_path.exists():
        with open(kalman_params_path) as f:
            _kp = json.load(f)
        logger.info("Loaded cached Kalman Q params from %s", kalman_params_path)
        result = check_manifest(kalman_params_path, fingerprint=stats_fingerprint)
        if result.is_stale:
            log_stale_warning(kalman_params_path,
                              "python pipeline/04d_calibrate_kalman_draft.py",
                              diffs=result.diffs)
        _age_q_halflife  = float(_kp.get("age_q_halflife_years",  _age_q_halflife))
        _age_q_max_mult  = float(_kp.get("age_q_max_multiplier",  _age_q_max_mult))
        _transfer_p_mult = float(_kp.get("transfer_p_multiplier", _transfer_p_mult))
    else:
        logger.info("Using config-default Kalman Q hyperparameters")

    logger.info(
        "Kalman params: halflife=%.2f max_mult=%.2f transfer_mult=%.2f",
        _age_q_halflife, _age_q_max_mult, _transfer_p_mult,
    )

    # ── Run Kalman filter ───────────────────────────────────────────────────────
    logger.info("Running Kalman filter (game-by-game)...")
    _age_col = "birth_year" if "birth_year" in adjusted.columns else None
    skill = run_kalman_all_players(
        adjusted,
        decay_betas=betas,
        obs_noises=_obs_noises_base,
        stats=stats_to_run,
        school_col="school",
        prior_means=prior_means,
        prior_vars=prior_vars,
        age_col=_age_col,
        age_q_halflife=_age_q_halflife,
        age_q_max_mult=_age_q_max_mult,
        transfer_p_mult=_transfer_p_mult,
        q_noise_by_stat=_q_by_stat,
    )

    # ── Run EWMA (vectorized across all players) ────────────────────────────────
    logger.info("Running EWMA (vectorized, all players)...")
    skill = ewma_all_players_fast(
        skill, stats_to_run, betas,
        prior_means=prior_means,
        prior_weight=ewma_prior_weight,
    )

    # ── Year-over-year features ─────────────────────────────────────────────────
    logger.info("Computing year-over-year deltas...")
    skill = _add_yoy_features(skill, stats_to_run, date_col)

    # Save full skill estimates
    out_path = PROCESSED_DIR / "skill_estimates.parquet"
    skill.to_parquet(out_path, index=False)
    logger.info("Saved %d skill estimate rows → %s", len(skill), out_path)

    # ── Snapshot: latest game per player ───────────────────────────────────────
    sort_col = date_col
    snapshot = (skill.sort_values(sort_col)
                .groupby("player_id", as_index=False)
                .last()
                .copy())

    # n_games stays gamelog-only (a true game count). n_seasons is computed
    # below from the union of (rosters, player_seasons, player_gamelogs) so
    # that it captures roster-only OL, redshirt years, and transfer history
    # — i.e. true college tenure rather than just seasons-with-stats. Used
    # by Step 8 as a measured age proxy that's non-collinear with the
    # imputed `age_at_draft` (which collapses to a function of class_year_int
    # whenever birth_year is unknown).
    if date_col == "game_date":
        n_games = skill.groupby("player_id").size().rename("n_games")
        snapshot = snapshot.merge(n_games.reset_index(), on="player_id", how="left")

    # ── Tenure-measured n_seasons (union across all source files) ─────────────
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    # CFBD player_id reuse exclusions (manual ∪ auto-detected). Computed once
    # here and reused for the roster-only padding below. Without this filter
    # the union re-inflates n_seasons for namesake collisions (the gamelogs
    # file is already filtered by step 01, but rosters / player_seasons are not).
    from config import KNOWN_SEASON_EXCLUSIONS
    from src.utils.cfbd_id_reuse import merged_exclusions as _merged_exclusions, exclusion_mask
    _exclusions = _merged_exclusions(KNOWN_SEASON_EXCLUSIONS, raw_dir, PROCESSED_DIR, logger)
    n_seasons_frames = []
    for path, cols in [
        (raw_dir / "rosters.parquet",         ["player_id", "season_year"]),
        (raw_dir / "player_seasons.parquet",  ["player_id", "season_year"]),
        (raw_dir / "player_gamelogs.parquet", ["player_id", "season_year"]),
    ]:
        if path.exists():
            n_seasons_frames.append(
                pd.read_parquet(path, columns=cols).dropna(subset=cols)
            )
    # FCS identities (step 01c) are roster rows CFBD doesn't have.
    from src.utils.fcs_identities import load_fcs_rosters
    fcs_rosters = load_fcs_rosters(["player_id", "player_name", "school",
                                    "season_year", "position"])
    n_seasons_frames.append(fcs_rosters[["player_id", "season_year"]])
    if n_seasons_frames:
        n_seasons_union_df = pd.concat(n_seasons_frames, ignore_index=True).drop_duplicates()
        if _exclusions:
            bad_mask = exclusion_mask(n_seasons_union_df, _exclusions)
            if bad_mask.any():
                logger.info("Dropped %d (player_id, season_year) pairs from "
                            "n_seasons union matching CFBD id-reuse exclusions",
                            int(bad_mask.sum()))
                n_seasons_union_df = n_seasons_union_df[~bad_mask]
        n_seasons_union = (n_seasons_union_df
                           .groupby("player_id")["season_year"]
                           .nunique()
                           .rename("n_seasons"))
    else:
        n_seasons_union = pd.Series(dtype=int, name="n_seasons")
    # Drop any pre-existing n_seasons (e.g. from `skill` if a future change
    # ever computes it upstream) before merging the union value.
    if "n_seasons" in snapshot.columns:
        snapshot = snapshot.drop(columns=["n_seasons"])
    snapshot = snapshot.merge(n_seasons_union.reset_index(),
                               on="player_id", how="left")

    # Career-average opponent composite strength from advanced game stats
    ags_path = Path(__file__).parent.parent / "data" / "raw" / "advanced_game_stats.parquet"
    gl_path  = Path(__file__).parent.parent / "data" / "raw" / "player_gamelogs.parquet"
    if ags_path.exists() and gl_path.exists():
        ags = pd.read_parquet(ags_path, columns=["game_id", "team", "def_ppa", "off_ppa"])
        gl  = pd.read_parquet(gl_path,  columns=["player_id", "game_id", "opponent"])
        gl  = gl.dropna(subset=["game_id", "opponent"])
        ags = ags.dropna(subset=["game_id", "team", "def_ppa", "off_ppa"])
        opp_stats = gl.merge(
            ags.rename(columns={"team": "opponent",
                                 "def_ppa": "opp_def_ppa",
                                 "off_ppa": "opp_off_ppa"}),
            on=["game_id", "opponent"], how="inner",
        )
        career_opp = (opp_stats.groupby("player_id")[["opp_def_ppa", "opp_off_ppa"]]
                      .mean().reset_index()
                      .rename(columns={"opp_def_ppa": "career_opp_def_ppa",
                                       "opp_off_ppa": "career_opp_off_ppa"}))
        snapshot = snapshot.merge(career_opp, on="player_id", how="left")
        n_matched = career_opp["career_opp_def_ppa"].notna().sum()
        logger.info("Added career opponent means for %d players", n_matched)

    # ── Pad snapshot with roster-only players ──────────────────────────────────
    # Players who appear in CFBD rosters but have zero gamelog rows (most OL,
    # because CFBD's stat endpoints don't track OL counting stats) are absent
    # from the gamelog-driven snapshot. Add skeleton rows for them so they
    # flow through to scoring with NaN skill features (LightGBM handles those
    # via missing-direction splits). Attributes (height/weight/age) and
    # hasan_rank are joined later in steps 8/10 and don't depend on the
    # snapshot carrying them here.
    rosters_path = Path(__file__).parent.parent / "data" / "raw" / "rosters.parquet"
    if rosters_path.exists():
        rosters = pd.read_parquet(
            rosters_path,
            columns=["player_id", "player_name", "school", "season_year",
                     "position", "class_year_int"],
        ).dropna(subset=["player_id", "season_year"])
        rosters = pd.concat([rosters, fcs_rosters.reindex(columns=rosters.columns)],
                            ignore_index=True)
        # Mirror the exclusion applied above to the n_seasons union: without
        # this, the CFBD-reused (player_id, season_year) row would re-enter
        # the snapshot as a roster-only skeleton, flipping the player's
        # identity to the namesake's school.
        if _exclusions:
            bad_mask = exclusion_mask(rosters, _exclusions)
            if bad_mask.any():
                rosters = rosters[~bad_mask].reset_index(drop=True)
        existing_pids = set(snapshot["player_id"].dropna().astype(str))
        rosters_only = rosters[~rosters["player_id"].astype(str).isin(existing_pids)]
        if len(rosters_only):
            # Latest season per player_id (matches the snapshot semantic of
            # "most recent observation per player").
            rosters_only = (rosters_only.sort_values("season_year")
                            .groupby("player_id", as_index=False)
                            .last())
            rosters_only["position_group"] = rosters_only["position"].map(
                lambda p: next((g for g, members in POSITION_GROUPS.items()
                                if p in members), "OTHER")
            )
            # NaN, not 0: these players have no gamelog coverage, which is
            # "unknown game count" not "played zero games" (most are OL, whom
            # CFBD doesn't track counting stats for, and bridge-only players).
            # Encoding 0 conflated the two and told LightGBM they literally sat
            # out; NaN routes them via the missing-direction split instead.
            rosters_only["n_games"] = np.nan
            # Pull n_seasons from the union map computed above. For pure
            # roster-only players (no player_seasons / no gamelogs), this is
            # equivalent to nunique(season_year) over their rosters rows.
            rosters_only = rosters_only.merge(
                n_seasons_union.reset_index(),
                on="player_id", how="left",
            )
            rosters_only["n_seasons"] = rosters_only["n_seasons"].fillna(1).astype(int)
            # Drop the raw `position` column — snapshot doesn't carry it
            # (added by attrs join in step 10). Keep position_group only.
            skel = rosters_only.drop(columns=["position"])
            # Reindex to snapshot's columns so concat aligns; missing columns
            # become NaN, which is exactly the desired behavior for skill cols.
            skel = skel.reindex(columns=snapshot.columns, fill_value=np.nan)
            snapshot = pd.concat([snapshot, skel], ignore_index=True)
            logger.info("Padded snapshot with %d roster-only players "
                        "(no gamelogs in CFBD; OL/etc.)", len(skel))

    # ── Career workload totals (tire-wear proxies, RB-relevant) ───────────────
    # Raw career sums from player_gamelogs. NaN→0 fill is honest: players
    # absent from the gamelog table (roster-only OL) genuinely have 0 carries,
    # and gamelog rows with NaN in a stat-cell are zero-by-position-convention.
    gl_workload_path = Path(__file__).parent.parent / "data" / "raw" / "player_gamelogs.parquet"
    if gl_workload_path.exists():
        gl_w = pd.read_parquet(gl_workload_path,
                                columns=["player_id", "rush_att", "rec"])
        gl_w = gl_w.dropna(subset=["player_id"])
        gl_w["rush_att"] = gl_w["rush_att"].fillna(0)
        gl_w["rec"]      = gl_w["rec"].fillna(0)
        gl_w["touches"]  = gl_w["rush_att"] + gl_w["rec"]
        career_workload = (gl_w.groupby("player_id", as_index=False)
                           .agg(career_rush_att=("rush_att", "sum"),
                                career_touches=("touches", "sum")))
        snapshot = snapshot.merge(career_workload, on="player_id", how="left")
        for c in ("career_rush_att", "career_touches"):
            snapshot[c] = snapshot[c].fillna(0)
        logger.info("Career workload totals: %d players with >0 carries, "
                    "%d with >0 touches", int((snapshot["career_rush_att"] > 0).sum()),
                    int((snapshot["career_touches"] > 0).sum()))

    # ── PFF career contact-volume totals (RB tire-wear, 2014+ coverage) ───────
    # Complements the gamelog totals above with PFF's cleaner classification
    # (no sacks-as-rushes pollution) and uniquely carries `yards_after_contact`
    # — a literal contact-volume signal that gamelogs don't expose. NaN is
    # preserved (not filled to 0) because absence here means "no PFF coverage"
    # rather than "zero carries"; LightGBM uses missing-direction splits.
    pff_seasons_path = PROCESSED_DIR / "pff_season_grades.parquet"
    if pff_seasons_path.exists():
        pff_w = pd.read_parquet(
            pff_seasons_path,
            columns=["player_id", "position_group", "pff_yards_after_contact",
                     "pff_avoided_tackles", "pff_yco_attempt", "pff_attempts", "season_year"],
        ).dropna(subset=["player_id"])
        if _exclusions:
            pff_w = pff_w[~exclusion_mask(pff_w, _exclusions)]
        # Rush attempts come off the PFF rushing report as `pff_attempts`, but
        # that column is report-scoped — it reads as pass attempts on QB rows
        # and punt attempts on P rows — so only the RB reading is a rush count.
        # Where it's missing (pre-2014 seasons have no rushing report), recover
        # it from yards_after_contact / yco_attempt: PFF defines yco_attempt as
        # the per-attempt rate, so the division is exact.
        yco = pff_w["pff_yco_attempt"].replace(0, np.nan)
        inferred = pff_w["pff_yards_after_contact"] / yco
        rush_att = pff_w["pff_attempts"].where(pff_w["position_group"] == "RB")
        pff_w["pff_rush_att_inferred"] = rush_att.combine_first(inferred)
        career_pff = (pff_w.groupby("player_id", as_index=False)
                      .agg(pff_career_yards_after_contact=("pff_yards_after_contact", "sum"),
                           pff_career_avoided_tackles=("pff_avoided_tackles", "sum"),
                           pff_career_rush_att=("pff_rush_att_inferred", "sum")))
        snapshot = snapshot.merge(career_pff, on="player_id", how="left")
        n_yac = int(snapshot["pff_career_yards_after_contact"].gt(0).sum())
        n_at  = int(snapshot["pff_career_avoided_tackles"].gt(0).sum())
        n_ra  = int(snapshot["pff_career_rush_att"].gt(0).sum())
        logger.info("PFF career contact totals: %d players with >0 YAC, "
                    "%d with >0 avoided tackles, %d with inferred carries",
                    n_yac, n_at, n_ra)

    # ── Career kick/punt return production ───────────────────────────────────
    # Return work is a real NFL draft-value channel for fringe WR/RB/DB — it's
    # how a fourth receiver or a backup corner earns a roster spot — and it was
    # entirely unused. Sourced from player_seasons rather than gamelogs because
    # only the season file carries return COUNTS; the gamelogs have yards/TDs
    # but no attempts, so per-return averages aren't computable there.
    seasons_ret_path = RAW_DIR / "player_seasons.parquet"
    if seasons_ret_path.exists():
        ret_cols = ["kickreturns_no", "kickreturns_yds", "kickreturns_td",
                    "puntreturns_no", "puntreturns_yds", "puntreturns_td"]
        ps_ret = pd.read_parquet(seasons_ret_path,
                                 columns=["player_id"] + ret_cols)
        ps_ret = ps_ret.dropna(subset=["player_id"])
        for c in ret_cols:
            ps_ret[c] = pd.to_numeric(ps_ret[c], errors="coerce").fillna(0)
        career_ret = (ps_ret.groupby("player_id", as_index=False)[ret_cols].sum()
                      .rename(columns={
                          "kickreturns_no":  "career_kr_att",
                          "kickreturns_yds": "career_kr_yards",
                          "kickreturns_td":  "career_kr_tds",
                          "puntreturns_no":  "career_pr_att",
                          "puntreturns_yds": "career_pr_yards",
                          "puntreturns_td":  "career_pr_tds",
                      }))
        career_ret["career_return_att"]   = (career_ret["career_kr_att"]
                                             + career_ret["career_pr_att"])
        career_ret["career_return_yards"] = (career_ret["career_kr_yards"]
                                             + career_ret["career_pr_yards"])
        career_ret["career_return_tds"]   = (career_ret["career_kr_tds"]
                                             + career_ret["career_pr_tds"])
        # Per-return averages stay NaN with zero attempts — "never returned a
        # kick" is not the same as "averaged 0 yards per return", and LightGBM
        # splits on missing separately. The volume/TD counts DO zero-fill:
        # there, zero is a real observation (mirrors career_rush_att above).
        for avg_col, yds_col, att_col in [
            ("career_kr_avg", "career_kr_yards", "career_kr_att"),
            ("career_pr_avg", "career_pr_yards", "career_pr_att"),
        ]:
            career_ret[avg_col] = (career_ret[yds_col]
                                   / career_ret[att_col].replace(0, np.nan))
        keep = ["player_id", "career_return_att", "career_return_yards",
                "career_return_tds", "career_kr_avg", "career_pr_avg"]
        snapshot = snapshot.merge(career_ret[keep], on="player_id", how="left")
        for c in ("career_return_att", "career_return_yards", "career_return_tds"):
            snapshot[c] = snapshot[c].fillna(0)
        logger.info("Career return production: %d players with >0 returns "
                    "(%d with a return TD)",
                    int((snapshot["career_return_att"] > 0).sum()),
                    int((snapshot["career_return_tds"] > 0).sum()))

    snap_path = PROCESSED_DIR / "player_skill_snapshot.parquet"
    snapshot.to_parquet(snap_path, index=False)
    logger.info("Saved %d player snapshots → %s", len(snapshot), snap_path)

    # Sanity check
    kalman_cols = [c for c in snapshot.columns if c.startswith("kalman_mean_")]
    logger.info("Kalman mean columns: %d", len(kalman_cols))
    if kalman_cols:
        col = kalman_cols[0]
        finite = snapshot[col].replace([np.inf, -np.inf], np.nan).dropna()
        logger.info("  %s: mean=%.3f, std=%.3f", col, finite.mean(), finite.std())

    logger.info("Step 6 complete.")


if __name__ == "__main__":
    main()
