"""
Step 5p: Per-Play Residual Aggregation

Two-branch residual computation:
    1. CREDITED branch: per-play residual = observed - predicted on credited
       rows; aggregate to per-(player, game) mean and total.
    2. CATCH-ALL branch: for each (active-roster-player, game) NOT in the
       credited set, compute team-level expected stat over eligible plays
       (position-gated) and emit -mean / -sum as the catch-all residual.

Inputs:
    data/processed/player_plays.parquet              — from step 1f
    data/processed/play_contextual_models/*.lgb      — from step 4p
    data/processed/play_contextual_models/model_summary_play.json
    data/raw/plays.parquet                           — for catch-all eligibility
    data/raw/rosters.parquet                         — active roster lookup
    data/processed/player_attributes.parquet         — position_group lookup
    data/processed/adjusted_gamelogs.parquet         — game-level context for plays

Output:
    data/processed/adjusted_player_plays_per_game.parquet
        grain: (player_id, game_id)
        cols: identifiers + is_catchall + play_mean_residual_{stat} +
              play_total_residual_{stat} + derived rate residuals

Usage:
    python pipeline/05p_play_residual_aggregation.py
    python pipeline/05p_play_residual_aggregation.py --test
    python pipeline/05p_play_residual_aggregation.py --no-catchall   # debug
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PLAY_CATEGORICAL_FEATURES, PLAY_CONTEXT_FEATURES, PLAY_RATE_STAT_NUMERATOR_DENOM,
    PLAY_RAW_RATE_FALLBACK_STATS, PLAY_STAT_COL, PLAY_START_YEAR,
    PLAY_TYPE_STAT_ELIGIBILITY, POSITION_ELIGIBLE_PLAY_TYPES, POSITION_SIDE,
    PROCESSED_DIR, RAW_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLAYER_PLAYS_PATH = PROCESSED_DIR / "player_plays.parquet"
MODELS_DIR        = PROCESSED_DIR / "play_contextual_models"
SUMMARY_PATH      = MODELS_DIR / "model_summary_play.json"
PLAYS_PATH        = RAW_DIR / "plays.parquet"
ROSTERS_PATH      = RAW_DIR / "rosters.parquet"
PA_PATH           = PROCESSED_DIR / "player_attributes.parquet"
ADJ_GAMELOGS_PATH = PROCESSED_DIR / "adjusted_gamelogs.parquet"

OUT_PATH          = PROCESSED_DIR / "adjusted_player_plays_per_game.parquet"

# CV-R² threshold below which the model is treated as no-op (residual = 0)
R2_GATE_THRESHOLD = 0.01


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",         action="store_true")
    parser.add_argument("--no-catchall",  action="store_true",
                        help="Skip catch-all generation (credited rows only)")
    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────


def _load_models() -> dict:
    """Load all per-stat LightGBM boosters that passed the R² gate."""
    import lightgbm as lgb

    if not SUMMARY_PATH.exists():
        logger.error("model_summary_play.json not found — run step 4p first")
        sys.exit(1)
    summary = json.loads(SUMMARY_PATH.read_text())
    models: dict = {}
    for row in summary["stats"]:
        # Use the gate flag step 4p stamped — direct cv_r2 < threshold lets
        # NaN cv_r2 slip through (NaN < x is False), incorrectly loading 9
        # models that had insufficient signal to compute a CV R².
        if not row.get("passes_r2_gate", False):
            cv = row.get("cv_r2")
            logger.info("[%s] skipping (cv_r2=%s, gate=FAIL)",
                        row["stat_type"],
                        f"{cv:.4f}" if cv is not None and cv == cv else "NaN")
            continue
        model_path = MODELS_DIR / f"{row['stat_col']}.lgb"
        if not model_path.exists():
            logger.warning("Model file %s missing — skipping %s",
                           model_path, row["stat_type"])
            continue
        models[row["stat_type"]] = {
            "booster":  lgb.Booster(model_file=str(model_path)),
            "stat_col": row["stat_col"],
            "cv_r2":    row["cv_r2"],
        }
    logger.info("Loaded %d models that passed R²-gate", len(models))
    return models


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in PLAY_CONTEXT_FEATURES if c in df.columns]
    X = df[feats].copy()
    for cat in PLAY_CATEGORICAL_FEATURES:
        if cat in X.columns:
            X[cat] = X[cat].astype("category")
    return X


def _per_play_residuals(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Compute residual_{stat} = observed - predicted on credited+eligible rows."""
    X = _prepare_features(df)
    for stat_type, mdl in models.items():
        col = mdl["stat_col"]
        eligible_play_types = {pt for pt, stats in PLAY_TYPE_STAT_ELIGIBILITY.items()
                                if stat_type in stats}
        elig_mask = df["play_type"].isin(eligible_play_types)
        credited_mask = df[col].notna()
        active = elig_mask & credited_mask
        if not active.any():
            df[f"residual_{col}"] = np.nan
            continue
        preds = mdl["booster"].predict(X.loc[active])
        observed = df.loc[active, col].astype(float).values
        residuals = observed - preds
        df.loc[active, f"residual_{col}"] = residuals
        # Outside the active rows, leave residual NaN (no signal)
    return df


def _per_player_game_aggregation(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Group credited rows by (athlete_id, game_id); emit mean + total residual."""
    if df.empty:
        return pd.DataFrame()

    # Identifier columns to carry through
    id_cols = ["athlete_id", "game_id", "season_year", "week",
               "team", "opponent", "position_group"]
    id_cols = [c for c in id_cols if c in df.columns]

    agg_funcs: dict = {}
    for stat_type, mdl in models.items():
        col = f"residual_{mdl['stat_col']}"
        if col in df.columns:
            agg_funcs[col] = ["mean", "sum", "count"]

    if not agg_funcs:
        logger.warning("No residual columns to aggregate")
        return pd.DataFrame()

    grouped = (df.groupby(["athlete_id", "game_id"], dropna=False)
                 .agg(agg_funcs))
    # Flatten multi-index columns
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped = grouped.reset_index()

    # Rename to adjusted_play_mean_resid_{c}, adjusted_play_total_resid_{c},
    # play_n_credited_{c}. The `adjusted_` prefix lets step 6 auto-detect
    # these as Kalman/EWMA inputs via its existing convention.
    rename = {}
    for stat_type, mdl in models.items():
        c = mdl["stat_col"]
        rename[f"residual_{c}_mean"]  = f"adjusted_play_mean_resid_{c}"
        rename[f"residual_{c}_sum"]   = f"adjusted_play_total_resid_{c}"
        rename[f"residual_{c}_count"] = f"play_n_credited_{c}"
    grouped = grouped.rename(columns=rename)

    # pandas .sum() over all-NaN returns 0.0, not NaN — so a WR's
    # `adjusted_play_total_resid_field_goal_missed` would land as 0.0 for
    # every game (no FG plays credited), then step 6's EWMA propagates the
    # spurious zero into the snapshot and step 8 splits on it as noise.
    # Mask sum and mean back to NaN where count is 0.
    for stat_type, mdl in models.items():
        c = mdl["stat_col"]
        cnt_col = f"play_n_credited_{c}"
        if cnt_col not in grouped.columns:
            continue
        empty = grouped[cnt_col] == 0
        for col in (f"adjusted_play_total_resid_{c}", f"adjusted_play_mean_resid_{c}"):
            if col in grouped.columns:
                grouped.loc[empty, col] = float("nan")

    # Bring back static (per-game, per-player) identifier columns
    meta = (df[id_cols].dropna(subset=["athlete_id", "game_id"])
                       .drop_duplicates(subset=["athlete_id", "game_id"]))
    grouped = grouped.merge(meta, on=["athlete_id", "game_id"], how="left")

    grouped["is_catchall"] = False
    return grouped


def _catchall_rows(plays: pd.DataFrame, models: dict,
                    rosters: pd.DataFrame, attrs: pd.DataFrame,
                    credited_keys: set[tuple],
                    player_plays_df: pd.DataFrame) -> pd.DataFrame:
    """Emit -E[stat over eligible plays] catch-all rows for non-participating
    active-roster players.

    Implementation:
      1. For each (team, season_year) get active roster from rosters.parquet
      2. For each (team, game_id) team played:
         a. Identify the player's side (offense/defense) for this game
         b. For each position_group in roster, filter plays to that team's
            eligible plays for that position_group
         c. Predict expected stat on each eligible play, aggregate to
            team-level mean and total
         d. For each non-credited active-roster player at that
            (team, game, position_group), emit catch-all row with
            mean_residual = -mean_expected, total_residual = -total_expected
    """
    if plays.empty or attrs.empty:
        return pd.DataFrame()

    # Active-roster lookup keyed (school, season_year): list of (player_id, position)
    rosters = rosters[["player_id", "school", "season_year", "position"]].copy()
    rosters["player_id"] = rosters["player_id"].astype(str)

    # Position → position_group via player_attributes. attrs is one row per
    # player (season_year = last roster season), so join on player_id only —
    # joining on season_year too would drop every non-final-season roster row
    # from the catch-all universe.
    if "position_group" in attrs.columns:
        attrs_pg = attrs[["player_id", "position_group"]].copy()
        attrs_pg["player_id"] = attrs_pg["player_id"].astype(str)
        attrs_pg = attrs_pg.drop_duplicates(subset=["player_id"])
        rosters = rosters.merge(attrs_pg, on="player_id", how="left")
    else:
        rosters["position_group"] = pd.NA

    # Restrict to positions we model at play grain
    modeled_positions = set(POSITION_ELIGIBLE_PLAY_TYPES.keys())
    rosters = rosters[rosters["position_group"].isin(modeled_positions)]
    if rosters.empty:
        logger.info("No roster players match modeled positions — skipping catch-all")
        return pd.DataFrame()

    # Per (game_id, team) determine offense_team / defense_team / kicking_team.
    # Both offense and defense slots — a team is on offense for some plays
    # and defense for others within the same game.
    plays = plays[plays["season_year"] >= PLAY_START_YEAR].copy()

    # For each (game_id, team) where the team appears on EITHER side, build
    # a catch-all roster lookup. Note: kickoff/punt/FG plays the team could
    # be on either side; we use POSITION_SIDE × per-play offense column.
    offense_plays = plays.rename(columns={"offense": "team"})
    defense_plays = plays.rename(columns={"defense": "team"})
    offense_plays["team_side"] = "offense"
    defense_plays["team_side"] = "defense"
    team_plays = pd.concat([offense_plays, defense_plays], ignore_index=True)

    # Raw plays.parquet lacks 26 of the 34 PLAY_CONTEXT_FEATURES the booster
    # was trained on (weather, opp_rolling_*, qb_rolling_*, conf_tier,
    # score_diff, is_garbage_time). These get merged into player_plays in
    # step 1f keyed per (play_id, athlete) — values are constant per
    # (play_id, athlete_team), so dedupe and left-join on (play_id, team).
    ctx_cols = [c for c in PLAY_CONTEXT_FEATURES
                if c in player_plays_df.columns
                and c not in team_plays.columns
                and c != "position_group"]
    if ctx_cols:
        ppd = (player_plays_df[["play_id", "team"] + ctx_cols]
               .dropna(subset=["play_id", "team"])
               .drop_duplicates(subset=["play_id", "team"]))
        team_plays = team_plays.merge(ppd, on=["play_id", "team"], how="left")

    # Use only the columns we need for prediction + grouping
    keep = ["play_id", "game_id", "season_year", "team", "team_side", "play_type"] \
        + [c for c in PLAY_CONTEXT_FEATURES if c in team_plays.columns]
    keep = list(dict.fromkeys(keep))
    keep = [c for c in keep if c in team_plays.columns]
    team_plays = team_plays[keep]

    # Build (team, game_id) → unique pairs
    team_games = team_plays[["team", "game_id", "season_year"]].drop_duplicates()
    logger.info("Catch-all: %d (team, game) tuples to evaluate", len(team_games))

    # Pre-aggregate plays by (team, game_id, position_group) once per position_group
    catchall_records: list[dict] = []
    seen = 0
    for pos, elig_pts in POSITION_ELIGIBLE_PLAY_TYPES.items():
        side = POSITION_SIDE.get(pos, "offense")
        if side == "offense":
            sub = team_plays[(team_plays["team_side"] == "offense") &
                              (team_plays["play_type"].isin(elig_pts))]
        else:  # defense
            sub = team_plays[(team_plays["team_side"] == "defense") &
                              (team_plays["play_type"].isin(elig_pts))]

        if sub.empty:
            continue

        # Add position_group as a categorical column (needed by LightGBM)
        sub = sub.assign(position_group=pos)
        X = _prepare_features(sub)

        # Predict expected stat per (eligible play, position_group)
        for stat_type, mdl in models.items():
            stat_elig = {pt for pt, stats in PLAY_TYPE_STAT_ELIGIBILITY.items()
                          if stat_type in stats}
            pred_mask = sub["play_type"].isin(stat_elig)
            if not pred_mask.any():
                continue
            preds = pd.Series(np.nan, index=sub.index, dtype="float64")
            preds.loc[pred_mask] = mdl["booster"].predict(X.loc[pred_mask])

            # Aggregate per (team, game_id, season_year). pandas sums an
            # all-NaN group to 0.0 — mask those totals back to NaN (count of
            # non-NaN preds == 0), else a spurious "exactly met expectation"
            # observation fans out to every catch-all player for that game.
            # Same bug class as the credited-branch fix above.
            agg = (preds.groupby([sub["team"], sub["game_id"], sub["season_year"]])
                        .agg(["mean", "sum", "count"]))
            agg.loc[agg["count"] == 0, "sum"] = np.nan
            agg = agg.drop(columns="count")
            agg.columns = [f"catchall_mean_{mdl['stat_col']}",
                           f"catchall_total_{mdl['stat_col']}"]
            agg = agg.reset_index()
            agg["position_group"] = pos
            catchall_records.append(agg)

        seen += 1
        logger.info("Catch-all aggregated for position %s (%d/%d)",
                    pos, seen, len(POSITION_ELIGIBLE_PLAY_TYPES))

    if not catchall_records:
        return pd.DataFrame()

    # Stack per-position chunks. Each chunk's position_group is unique, so rows
    # don't overlap; concat preserves per-position stat columns (others NaN).
    catchall = pd.concat(catchall_records, ignore_index=True, sort=False)

    # Now expand to (player, game) by joining with rosters
    school_to_team_map = {}  # rosters use 'school', plays use 'team'.
    # They're the same string for CFBD — verify
    rosters_renamed = rosters.rename(columns={"school": "team"})
    rosters_renamed["season_year"] = rosters_renamed["season_year"].astype(int)
    catchall["season_year"] = catchall["season_year"].astype(int)
    catchall_expanded = catchall.merge(
        rosters_renamed[["team", "season_year", "player_id", "position_group"]],
        on=["team", "season_year", "position_group"],
        how="inner",
    )
    if catchall_expanded.empty:
        logger.warning("Catch-all expansion produced 0 rows — check team/school name mismatch")
        return pd.DataFrame()

    # Drop players who DID appear in credited rows for this (game, player)
    catchall_expanded["_key"] = list(zip(
        catchall_expanded["player_id"].astype(str),
        catchall_expanded["game_id"].astype("Int64"),
    ))
    before = len(catchall_expanded)
    catchall_expanded = catchall_expanded[~catchall_expanded["_key"].isin(credited_keys)]
    catchall_expanded = catchall_expanded.drop(columns=["_key"])
    logger.info("Catch-all expansion: %d → %d rows after removing credited (player, game) pairs",
                before, len(catchall_expanded))

    # Convert mean/total to residual (negative of expectation)
    for stat_type, mdl in models.items():
        c = mdl["stat_col"]
        mc = f"catchall_mean_{c}"
        tc = f"catchall_total_{c}"
        if mc in catchall_expanded.columns:
            catchall_expanded[f"adjusted_play_mean_resid_{c}"]  = -catchall_expanded[mc]
            catchall_expanded[f"adjusted_play_total_resid_{c}"] = -catchall_expanded[tc]
            catchall_expanded[f"play_n_credited_{c}"]            = 0
            catchall_expanded = catchall_expanded.drop(columns=[mc, tc])

    catchall_expanded["is_catchall"] = True
    catchall_expanded = catchall_expanded.rename(columns={"player_id": "athlete_id"})
    return catchall_expanded


def _derive_rate_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived rate-stat residuals from numerator/denominator totals.

    Mirrors RATE_STAT_ATTEMPT_COL in spirit: rate = sum(num) / sum(denom).
    Note: denom names like 'completion_attempt' / 'rush_attempt' are
    SYNTHETIC — they refer to the count of eligible plays of that type. We
    use play_n_credited_<num> for the denominator count, NOT a separate stat.
    """
    for rate_name, (num, _denom) in PLAY_RATE_STAT_NUMERATOR_DENOM.items():
        num_total = f"adjusted_play_total_resid_{num}"
        num_count = f"play_n_credited_{num}"
        if num_total in df.columns and num_count in df.columns:
            # adjusted_play_rate_{rate} pattern for step 6 auto-detection
            df[f"adjusted_play_rate_{rate_name}"] = (
                df[num_total] / df[num_count].replace(0, np.nan)
            )
    return df


def main():
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PLAYER_PLAYS_PATH)
    logger.info("Loaded %d credited (play, athlete) rows", len(df))
    if args.test:
        latest = int(df["season_year"].max())
        df = df[df["season_year"] == latest]
        logger.info("Test mode: %d rows in season %d", len(df), latest)

    models = _load_models()
    if not models:
        logger.error("No models passed R² gate — cannot residualize")
        sys.exit(1)

    # ── Branch 1: credited per-play residuals + per-(player, game) aggregation ──
    logger.info("Computing per-play residuals on credited rows ...")
    df = _per_play_residuals(df, models)
    credited_agg = _per_player_game_aggregation(df, models)
    logger.info("Credited (player, game) aggregations: %d rows", len(credited_agg))

    credited_keys = set(zip(
        credited_agg["athlete_id"].astype(str),
        credited_agg["game_id"].astype("Int64"),
    ))

    # ── Branch 2: catch-all rows for non-participating active-roster players ──
    if args.no_catchall:
        logger.info("Skipping catch-all (--no-catchall)")
        out = credited_agg
    else:
        plays = pd.read_parquet(PLAYS_PATH)
        if args.test:
            plays = plays[plays["season_year"] == int(df["season_year"].max())]
        rosters = pd.read_parquet(ROSTERS_PATH) if ROSTERS_PATH.exists() else pd.DataFrame()
        attrs   = pd.read_parquet(PA_PATH) if PA_PATH.exists() else pd.DataFrame()

        catchall = _catchall_rows(plays, models, rosters, attrs, credited_keys, df)
        logger.info("Catch-all rows: %d", len(catchall))

        if catchall.empty:
            out = credited_agg
        else:
            # Align columns (catchall may be missing some)
            for col in credited_agg.columns:
                if col not in catchall.columns:
                    catchall[col] = pd.NA
            out = pd.concat([credited_agg, catchall[credited_agg.columns]],
                              ignore_index=True)

    # ── Derived rate residuals ──
    out = _derive_rate_residuals(out)

    # ── Raw per-(player, game) rates: fallback for FAIL-gated outcome stats ──
    # Policy (per project convention):
    #   - If contextual model PASSES R² gate: residual already emitted above.
    #     Skip raw rate (would duplicate signal).
    #   - If contextual model FAILS R² gate AND stat is in fallback list:
    #     emit raw mean + count. The raw rate carries skill signal not in
    #     the per-game pipeline (e.g., fumble rate split by carry vs reception).
    #   - If FAIL AND not in fallback list: drop entirely (redundant with
    #     game-grain or otherwise unsupported).
    # Prefix `adjusted_play_raw_mean_` so step 6's `adjusted_play_*` auto-
    # detection picks them up for Kalman/EWMA (the "adjusted_" prefix is a
    # codebase convention meaning "ready for downstream skill estimation,"
    # not literally "context-adjusted").
    summary = json.loads(SUMMARY_PATH.read_text())
    gate_pass = {row["stat_type"] for row in summary["stats"]
                 if row.get("passes_r2_gate")}
    for stat_type in PLAY_RAW_RATE_FALLBACK_STATS:
        if stat_type in gate_pass:
            continue  # residual already emitted by the PASS-gated path
        col = PLAY_STAT_COL.get(stat_type)
        if not col or col not in df.columns:
            continue
        sub = df[df[col].notna()]
        if sub.empty:
            continue
        agg = (sub.groupby(["athlete_id", "game_id"], dropna=False)[col]
                   .agg(["mean", "count"])
                   .reset_index()
                   .rename(columns={"mean": f"adjusted_play_raw_mean_{col}",
                                    "count": f"play_raw_n_{col}"}))
        out = out.merge(agg, on=["athlete_id", "game_id"], how="left")
        logger.info("  raw-rate fallback emitted for %s (FAIL R²): %d (player, game) tuples",
                    stat_type, len(agg))

    # ── Save ──
    # Rename athlete_id → player_id for downstream consistency
    out = out.rename(columns={"athlete_id": "player_id"})
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %d (player, game) residual rows × %d cols → %s",
                len(out), out.shape[1], OUT_PATH)
    logger.info("  credited: %d  catchall: %d",
                int((~out["is_catchall"].fillna(False)).sum()),
                int(out["is_catchall"].fillna(False).sum()))


if __name__ == "__main__":
    main()
