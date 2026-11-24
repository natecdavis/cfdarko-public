"""
Step 7: NFL Outcome Labels

Computes NFL success labels for bridge players using nfl_data_py (nflverse).

Two-stage architecture (analogous to CBB→NBA):
  Stage 1 trajectory models (fitted here):
    - role_trajectory:   partial-snapshot per-year NFL features → P(second contract)
    - impact_trajectory: same + draft position → second contract APY as % of cap

  Single LightGBM model across all positions (position_group as ordinal feature).
  Each settled bridge player generates up to TRAJECTORY_MAX_YEARS training rows
  (partial snapshots: y1, y1-y2, y1-y3, y1-y4) so the model can predict at any
  career stage.

  Labels produced for the draft model:
    role_target:   trajectory model's role probability evaluated at a fixed
                   end-of-year-ROLE_LABEL_HORIZON_YEARS NFL snapshot for every
                   bridge player. Continuous in [0, 1]. Denoised, forward-
                   looking, and never post-resolution (y3 horizon is pre any
                   second contract, including early extensions). Players with
                   no NFL data fall back to the snap/AV-threshold label.
    impact_target: whichever outcome config.IMPACT_TARGET_SOURCE selects —
                   impact_apy_target ("apy") or value_target ("value"). Both
                   are always written, so the switch is a config flip + rerun.
    impact_apy_target: second contract APY as fraction of salary cap at
                   signing (0.0 if none). Recent un-settled players get the
                   standard trajectory override.
    value_target:  mean within-(position group × draft class) percentile of
                   best APY by y5, best guarantees by y5 and y3-y4 usage, for
                   settled classes; a trajectory-model estimate for un-settled
                   ones; NaN for the current class. Components are written too
                   (best_apy_by_y5, guar_by_y5, usage_y34, usage_source,
                   value_group), plus value_apy_map.parquet for step 10.

Bridge players with NFL data get the trajectory-projected role label;
non-bridge players get role_target=0, impact_target=0.

Output: data/processed/nfl_draft_labels.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR,
    REPLACEMENT_MARGIN,
    RAW_DIR,
    NFL_LABEL_WINDOW_YEARS,
    ROLE_SNAP_PCT_THRESHOLD,
    ROLE_SNAP_GAMES_MIN,
    ROLE_SNAP_PCT_THRESHOLD_BY_GROUP,
    TRAJECTORY_ELIGIBILITY_CUTOFF_YEAR,
    TRAJECTORY_MAX_YEARS,
    ROLE_LABEL_HORIZON_YEARS,
    END_YEAR,
    IMPACT_TARGET_SOURCE,
    VALUE_CONTRACT_LAST_YEAR_OFFSET,
    VALUE_USAGE_SEASON_OFFSETS,
    VALUE_SNAP_FIRST_DRAFT_YEAR,
    VALUE_MAP_MIN_DRAFT_YEAR,
)
from src.scrapers.nfl_data import (
    fetch_seasonal_snap_pcts,
    fetch_league_contracts,
    compute_replacement_levels,
    derive_vet_min_apy_pct,
    fetch_contract_data,
    compute_second_contract_label,
    compute_nfl_labels,
    fetch_nfl_seasonal_features,
    build_trajectory_dataset,
    fit_trajectory_models,
    score_trajectory_models,
    score_trajectory_at_horizon,
    load_trajectory_models,
    fetch_depth_charts,
    compute_starter_share,
    fetch_seasonal_rosters,
    build_bridge_gsis_crosswalk,
    compute_value_components,
    compute_value_target,
    build_value_apy_map,
    load_value_trajectory_model,
    fit_value_trajectory_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SNAP_DATA_START = 2016   # nfl_data_py snap counts available from ~2016



def _model_position_groups(bridge_players: pd.DataFrame) -> pd.Series:
    """pfr_id → the draft model's position group (EDGE split out of DL/LB),
    from player_attributes; bridge_players' own group where that is missing.
    The value percentiles must be computed within the groups the heads train
    on, or EDGE players would be ranked against interior linemen."""
    bp = bridge_players.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")
    group = bp["position_group"]
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    if attrs_path.exists() and "player_id" in bp.columns:
        attrs = (pd.read_parquet(attrs_path, columns=["player_id", "position_group"])
                 .drop_duplicates("player_id").set_index("player_id")["position_group"])
        model_group = bp["player_id"].map(attrs)
        group = model_group.where(model_group.notna() & (model_group != "OTHER")).fillna(group)
    return group


def _build_value_target(bridge_players, contracts_df, contract_label, snap_data,
                        seasonal_feats, retrain: bool, variant: str):
    """Value target for every bridge player with a pfr_id.

    Settled classes (draft_year <= END_YEAR - VALUE_CONTRACT_LAST_YEAR_OFFSET:
    y5 offseason signed, y4 season played) get the composite. Later classes get
    the value trajectory model's estimate from the seasons played so far — 0.0
    if they have played none — and the current class (no NFL season yet) NaN.

    Returns (value: Series pfr_id → value_target, components: DataFrame,
    apy_map: DataFrame for step 10).
    """
    bp = bridge_players.dropna(subset=["pfr_id", "draft_year"]).drop_duplicates("pfr_id")
    group = _model_position_groups(bp)
    dy = bp.set_index("pfr_id")["draft_year"].astype(int)
    settled = dy <= END_YEAR - VALUE_CONTRACT_LAST_YEAR_OFFSET

    pre_snap_years = dy[dy < VALUE_SNAP_FIRST_DRAFT_YEAR].unique()
    usage_seasons = sorted({int(y) + o for y in pre_snap_years for o in VALUE_USAGE_SEASON_OFFSETS})
    starter = (compute_starter_share(fetch_depth_charts(usage_seasons))
               if usage_seasons else pd.Series(dtype=float))
    roster_seasons = range(int(dy.min()), VALUE_SNAP_FIRST_DRAFT_YEAR + max(VALUE_USAGE_SEASON_OFFSETS) + 1)
    xwalk = build_bridge_gsis_crosswalk(bp, fetch_seasonal_rosters(roster_seasons))

    comps = compute_value_components(
        bp, contracts_df, snap_data, starter, xwalk,
        last_year_offset=VALUE_CONTRACT_LAST_YEAR_OFFSET,
        usage_offsets=VALUE_USAGE_SEASON_OFFSETS,
        snap_first_draft_year=VALUE_SNAP_FIRST_DRAFT_YEAR,
    )
    comps["value_group"] = group.reindex(comps.index)
    composite = compute_value_target(comps[settled.reindex(comps.index)], group, dy)

    map_rows = settled & (dy >= VALUE_MAP_MIN_DRAFT_YEAR)
    apy_map = build_value_apy_map(comps.loc[map_rows.reindex(comps.index), "best_apy_by_y5"], group)

    value = composite.reindex(comps.index)
    unsettled = comps.index[~settled.reindex(comps.index)]
    if len(unsettled):
        model, cols = (None, None) if retrain else load_value_trajectory_model(variant)
        if model is None and not seasonal_feats.empty:
            train_df = build_trajectory_dataset(
                bridge_players, seasonal_feats, contract_label,
                max_years=TRAJECTORY_MAX_YEARS, value_label=composite,
                settled_cutoff=END_YEAR - VALUE_CONTRACT_LAST_YEAR_OFFSET)
            model, cols = fit_value_trajectory_model(train_df, variant=variant)
        traj = pd.Series(np.nan, index=unsettled)
        if model is not None:
            scored = score_trajectory_models(
                bp[bp["pfr_id"].isin(unsettled)], seasonal_feats,
                None, [], model, cols, max_years=TRAJECTORY_MAX_YEARS)
            traj = scored["impact_trajectory_score"].clip(0.0, 1.0).reindex(unsettled)
        played = dy.reindex(unsettled) < END_YEAR
        value.loc[unsettled] = np.where(played, traj.fillna(0.0), np.nan)

    src = comps["usage_source"].value_counts(dropna=False).to_dict()
    logger.info("Value target: %d settled (composite), %d un-settled (trajectory); "
                "usage source %s; map built on %d players (classes %d-%d)",
                int(settled.sum()), len(unsettled), src, int(map_rows.sum()),
                VALUE_MAP_MIN_DRAFT_YEAR, END_YEAR - VALUE_CONTRACT_LAST_YEAR_OFFSET)
    return value, comps, apy_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip-trajectory", action="store_true",
                        help="Skip trajectory model fitting (use fallback labels only)")
    parser.add_argument("--retrain-trajectory", action="store_true",
                        help="Force refit of trajectory models even if cached .pkl files exist")
    parser.add_argument("--refresh-snap", action="store_true",
                        help="Re-fetch snap counts from nflverse even if cache exists")
    parser.add_argument("--refresh-epa", action="store_true",
                        help="Re-fetch EPA data from nflverse even if cache exists")
    parser.add_argument("--refresh-contracts", action="store_true",
                        help="Re-fetch league contracts from nflverse even if cache exists")
    parser.add_argument("--labels-variant",
                        choices=["legacy", "defect_fix", "replacement"],
                        default="legacy",
                        help="legacy: any non-rookie, non-5th-year-option deal "
                             "counts as a second contract (current production). "
                             "defect_fix: exclude franchise tags and "
                             "out-of-window deals, but keep the raw APY target "
                             "(isolates the defect fixes from the rebase). "
                             "replacement: the deal must also clear the "
                             "replacement line, not look like a franchise tag, "
                             "and fall inside the second-contract window; the "
                             "impact target becomes the above-replacement "
                             "remainder. Writes to a separate labels file and "
                             "uses variant-scoped trajectory models so the "
                             "canonical artifacts are never clobbered.")
    parser.add_argument("--replacement-margin", type=float, default=None,
                        help="Override REPLACEMENT_MARGIN (sensitivity arm)")
    args, _ = parser.parse_known_args()

    # Variant plumbing. The A/B arm must not clobber production artifacts, so
    # both the labels file and the trajectory model pickles are namespaced.
    is_replacement = args.labels_variant == "replacement"
    is_defect_fix = args.labels_variant == "defect_fix"
    use_over_repl = is_replacement
    traj_variant = "" if args.labels_variant == "legacy" else args.labels_variant
    labels_filename = ("nfl_draft_labels.parquet" if args.labels_variant == "legacy"
                       else f"nfl_draft_labels_{args.labels_variant}.parquet")
    if args.labels_variant != "legacy":
        logger.info("LABELS VARIANT: %s (margin=%.2f) → %s",
                    args.labels_variant,
                    args.replacement_margin if args.replacement_margin is not None
                    else REPLACEMENT_MARGIN, labels_filename)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    bridge_path = PROCESSED_DIR / "bridge_players.parquet"
    snapshot_path = PROCESSED_DIR / "player_skill_snapshot.parquet"

    if not bridge_path.exists():
        logger.error("bridge_players.parquet not found — run Step 2 first")
        sys.exit(1)

    bridge_players = pd.read_parquet(bridge_path)
    logger.info("Loaded %d bridge players", len(bridge_players))

    # ── Contract data: second contract APY % of cap as impact label ─────────────
    # The full league-wide contract population is fetched and cached once, then
    # used twice: to derive replacement levels (league-wide, needs every
    # position) and to build the per-bridge-player second-contract label.
    contract_label = None
    contracts_df = None
    contract_apy_raw: dict = {}
    try:
        league_contracts = fetch_league_contracts(refresh=args.refresh_contracts)

        replacement_levels = compute_replacement_levels(league_contracts)
        repl_path = PROCESSED_DIR / "replacement_levels.parquet"
        replacement_levels.to_parquet(repl_path, index=False)
        logger.info("Replacement levels → %s (%d rows)",
                    repl_path, len(replacement_levels))

        contracts_df = fetch_contract_data(bridge_players, contracts=league_contracts)
        vet_min = derive_vet_min_apy_pct(league_contracts)
        margin = (args.replacement_margin if args.replacement_margin is not None
                  else REPLACEMENT_MARGIN)

        contract_label = compute_second_contract_label(
            contracts_df,
            replacement_gate=is_replacement,
            defect_fixes=is_replacement or is_defect_fix,
            vet_min=vet_min,
            margin=margin,
        )
        logger.info("Second contract labels loaded: %d players", len(contract_label))

        # Shared A/B ground truth. Always the DEFECT-FIXED selection (tags and
        # out-of-window deals excluded) but never rebased, so both arms can be
        # scored against an identical, subtraction-free quantity — the target
        # rescaling in the replacement arm otherwise makes MAE incomparable.
        gated = compute_second_contract_label(
            contracts_df, replacement_gate=True, defect_fixes=True,
            vet_min=vet_min, margin=margin)
        contract_apy_raw = gated["label_apy_raw"].to_dict()
    except Exception as exc:
        logger.warning("Contract data unavailable — impact_target will be 0 for all: %s", exc)

    # ── Snap data: load from cache or fetch ─────────────────────────────────────
    # Latest completed NFL season (END_YEAR is the upcoming draft class year).
    snap_end = END_YEAR - 1
    snap_start = SNAP_DATA_START
    if args.test:
        snap_start = 2022

    snap_cache = RAW_DIR / "snap_counts.parquet"
    if snap_cache.exists() and not args.refresh_snap:
        logger.info("Loading snap counts from cache → %s", snap_cache)
        snap_data = pd.read_parquet(snap_cache)
        if args.test:
            snap_data = snap_data[snap_data["season"] >= snap_start]
    else:
        logger.info("Fetching snap counts %d–%d from nflverse...", snap_start, snap_end)
        snap_data = fetch_seasonal_snap_pcts(snap_start, snap_end)
        if not snap_data.empty:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            snap_data.to_parquet(snap_cache, index=False)
            logger.info("Cached %d snap-count rows → %s", len(snap_data), snap_cache)

    if snap_data.empty:
        logger.warning("No snap count data available — role_target will use w_av proxy")

    # ── Stage 1: Trajectory models (fit or load from cache) ─────────────────────
    role_model = None
    role_cols: list = []
    impact_model = None
    impact_cols: list = []
    trajectory_scores: pd.DataFrame = pd.DataFrame()
    horizon_scores: pd.DataFrame = pd.DataFrame()
    seasonal_feats: pd.DataFrame = pd.DataFrame()

    can_run_trajectory = (
        not args.skip_trajectory
        and not snap_data.empty
        and "position_group" in bridge_players.columns
    )

    if can_run_trajectory:
        if not args.retrain_trajectory:
            role_model, role_cols, impact_model, impact_cols = load_trajectory_models(
                variant=traj_variant)
            if role_model is not None:
                logger.info("Using cached trajectory models. Pass --retrain-trajectory to refit.")

        if role_model is None:
            logger.info("Fetching per-year NFL seasonal features for trajectory models...")
            seasonal_feats = fetch_nfl_seasonal_features(
                bridge_players, snap_data, max_years=TRAJECTORY_MAX_YEARS,
                refresh_epa=args.refresh_epa,
            )
            logger.info("Building partial-snapshot training dataset...")
            train_df = build_trajectory_dataset(
                bridge_players, seasonal_feats, contract_label,
                max_years=TRAJECTORY_MAX_YEARS,
                use_over_replacement=use_over_repl,
            )
            logger.info("Fitting trajectory models on %d training rows...", len(train_df))
            role_model, impact_model, role_cols, impact_cols = fit_trajectory_models(
                train_df, max_years=TRAJECTORY_MAX_YEARS, variant=traj_variant,
            )

        if role_model is not None:
            if seasonal_feats.empty:
                logger.info("Fetching per-year NFL seasonal features for scoring...")
                seasonal_feats = fetch_nfl_seasonal_features(
                    bridge_players, snap_data, max_years=TRAJECTORY_MAX_YEARS,
                    refresh_epa=args.refresh_epa,
                )
            logger.info("Scoring trajectory models for all bridge players...")
            trajectory_scores = score_trajectory_models(
                bridge_players, seasonal_feats,
                role_model, role_cols, impact_model, impact_cols,
                max_years=TRAJECTORY_MAX_YEARS,
            )
            n_role = trajectory_scores["role_trajectory_score"].notna().sum()
            n_impact = trajectory_scores["impact_trajectory_score"].notna().sum()
            logger.info("Trajectory coverage: role=%d/%d, impact=%d/%d",
                        n_role, len(bridge_players), n_impact, len(bridge_players))

            # Fixed-horizon scoring → canonical role_target for the draft model.
            # y3 ensures the label is always pre-second-contract-resolution.
            logger.info("Scoring trajectory models at fixed y%d horizon for canonical role_target...",
                        ROLE_LABEL_HORIZON_YEARS)
            horizon_scores = score_trajectory_at_horizon(
                bridge_players, seasonal_feats,
                role_model, role_cols, impact_model, impact_cols,
                horizon=ROLE_LABEL_HORIZON_YEARS,
                max_years=TRAJECTORY_MAX_YEARS,
            )
            role_horizon_col = f"role_traj_y{ROLE_LABEL_HORIZON_YEARS}_score"
            n_horizon = horizon_scores[role_horizon_col].notna().sum() if role_horizon_col in horizon_scores.columns else 0
            logger.info("Horizon-scored: %d/%d bridge players have a y%d-snapshot role label",
                        n_horizon, len(bridge_players), ROLE_LABEL_HORIZON_YEARS)
    else:
        if args.skip_trajectory:
            logger.info("Trajectory models skipped (--skip-trajectory flag)")
        elif snap_data.empty:
            logger.info("Trajectory models skipped (no snap data)")
        else:
            logger.info("Trajectory models skipped (position_group not in bridge_players)")

    # ── Stage 1 fallback: compute labels from snap thresholds + career AV ───────
    labeled = compute_nfl_labels(
        bridge_players,
        snap_data,
        snap_years=NFL_LABEL_WINDOW_YEARS,
        snap_threshold=ROLE_SNAP_PCT_THRESHOLD,
        threshold_by_group=ROLE_SNAP_PCT_THRESHOLD_BY_GROUP,
        contract_label=contract_label,
        use_over_replacement=use_over_repl,
    )

    # ── Value target ────────────────────────────────────────────────────────────
    # Always built (when contracts are available) so both targets sit in the
    # labels file; config.IMPACT_TARGET_SOURCE decides which fills impact_target.
    value_by_pfr: dict = {}
    value_comps = pd.DataFrame()
    if contracts_df is not None:
        value_series, value_comps, value_apy_map = _build_value_target(
            bridge_players, contracts_df, contract_label, snap_data, seasonal_feats,
            retrain=args.retrain_trajectory, variant=traj_variant)
        value_by_pfr = value_series.to_dict()
        map_path = PROCESSED_DIR / ("value_apy_map.parquet" if not traj_variant
                                    else f"value_apy_map_{traj_variant}.parquet")
        value_apy_map.to_parquet(map_path, index=False)
        logger.info("Value percentile → APY map → %s", map_path)
    elif IMPACT_TARGET_SOURCE == "value":
        logger.error("IMPACT_TARGET_SOURCE='value' but contract data is unavailable")
        sys.exit(1)
    comp_by_pfr = (value_comps.to_dict(orient="index") if not value_comps.empty else {})

    # Estimate first NFL season from snap data per pfr_id
    first_snap_season: dict = {}
    if not snap_data.empty and "pfr_player_id" in snap_data.columns:
        first_snap_season = (
            snap_data.groupby("pfr_player_id")["season"].min().to_dict()
        )

    # Build per-player label rows, merging trajectory scores where available
    if not trajectory_scores.empty:
        traj_dedup = trajectory_scores[~trajectory_scores.index.duplicated(keep="first")]
        traj_by_pfr = traj_dedup.to_dict(orient="index")
    else:
        traj_by_pfr = {}

    role_horizon_col = f"role_traj_y{ROLE_LABEL_HORIZON_YEARS}_score"
    impact_horizon_col = f"impact_traj_y{ROLE_LABEL_HORIZON_YEARS}_score"
    if not horizon_scores.empty:
        horiz_dedup = horizon_scores[~horizon_scores.index.duplicated(keep="first")]
        horizon_by_pfr = horiz_dedup.to_dict(orient="index")
    else:
        horizon_by_pfr = {}

    n_horizon_used = 0
    label_records = []
    for _, row in labeled.iterrows():
        pfr_id = row.get("pfr_id")
        draft_year = row.get("draft_year")
        first_nfl = first_snap_season.get(pfr_id)
        if first_nfl is None and draft_year:
            first_nfl = int(draft_year)

        # Use trajectory scores where available; fall back to snap/AV labels
        traj = traj_by_pfr.get(pfr_id, {})
        role_traj = traj.get("role_trajectory_score")
        impact_traj = traj.get("impact_trajectory_score")
        horiz = horizon_by_pfr.get(pfr_id, {})
        role_traj_horizon = horiz.get(role_horizon_col)
        impact_traj_horizon = horiz.get(impact_horizon_col)

        role_target = float(row.get("role_target", 0.0) or 0.0)
        impact_target = float(row.get("impact_target", 0.0) or 0.0)

        # Canonical role_target: trajectory model evaluated at fixed y-horizon
        # snapshot, applied to ALL bridge players with sufficient NFL data.
        # Denoises hard 0/1 outcomes (a y3-snapshot bust → ~0.05, an early-
        # extension star → ~0.85) and is always pre-second-contract-resolution.
        # Fall back to snap/AV-threshold label when horizon scoring isn't available.
        if role_traj_horizon is not None and not np.isnan(role_traj_horizon):
            role_target = float(np.clip(role_traj_horizon, 0.0, 1.0))
            n_horizon_used += 1

        # For recent un-settled players, fall back to the standard trajectory
        # override on the impact label (existing behavior).
        is_recent = (draft_year is not None and not pd.isna(draft_year)
                     and int(draft_year) >= TRAJECTORY_ELIGIBILITY_CUTOFF_YEAR)
        if is_recent and impact_target == 0.0 \
                and impact_traj is not None and not np.isnan(impact_traj):
            impact_target = float(max(0.0, impact_traj))

        # Multi-target Impact: pair the existing APY%cap target with a
        # complementary career-w_AV-per-game target. Two orthogonal NFL
        # outcome signals; LightGBM trains a head per target and Step 10
        # blends them.
        career_w_av = max(0.0, float(row.get("career_w_av", 0.0) or 0.0))
        nfl_games = int(row.get("nfl_games", 0) or 0)
        # w_AV/game: only meaningful with ≥4 games; else NaN (drops out
        # of training rather than getting a misleading 0 value).
        impact_w_av_per_game = (career_w_av / max(nfl_games, 1)
                                if nfl_games >= 4 else float("nan"))

        # Value target. Players with no pfr_id never reached an NFL snap
        # sheet: 0.0 once their class has played a season, NaN before.
        impact_apy_target = impact_target
        value_target = value_by_pfr.get(pfr_id) if pfr_id else None
        if value_target is None:
            played = draft_year is not None and not pd.isna(draft_year) \
                and int(draft_year) < END_YEAR
            value_target = 0.0 if played else float("nan")
        comp = comp_by_pfr.get(pfr_id, {}) if pfr_id else {}
        if IMPACT_TARGET_SOURCE == "value":
            impact_target = value_target

        label_records.append({
            "player_id":                row.get("player_id"),
            "player_name":              row.get("player_name"),
            "pfr_id":                   pfr_id,
            "draft_year":               draft_year,
            "last_college_season_year": row.get("last_college_season_year"),
            "role_target":              role_target,
            "impact_target":            impact_target,         # per config.IMPACT_TARGET_SOURCE
            "impact_apy_target":        impact_apy_target,     # APY%cap (legacy)
            "value_target":             value_target,
            "best_apy_by_y5":           comp.get("best_apy_by_y5"),
            "guar_by_y5":               comp.get("guar_by_y5"),
            "usage_y34":                comp.get("usage_y34"),
            "usage_source":             comp.get("usage_source"),
            "value_group":              comp.get("value_group"),
            # Ungated, un-rebased APY share of the selected deal. Identical in
            # meaning across label variants, so an A/B can score both arms
            # against it without the target rescaling that makes MAE
            # incomparable. Not a training target.
            "label_apy_raw":            float(contract_apy_raw.get(pfr_id, 0.0))
                                        if pfr_id else 0.0,
            "impact_w_av_per_game":     impact_w_av_per_game,  # complementary signal
            "role_trajectory_score":    float(role_traj) if role_traj is not None and not np.isnan(role_traj) else None,
            "impact_trajectory_score":  float(impact_traj) if impact_traj is not None and not np.isnan(impact_traj) else None,
            role_horizon_col:           float(role_traj_horizon) if role_traj_horizon is not None and not np.isnan(role_traj_horizon) else None,
            impact_horizon_col:         float(impact_traj_horizon) if impact_traj_horizon is not None and not np.isnan(impact_traj_horizon) else None,
            "career_w_av":              career_w_av,
            "first_nfl_season":         first_nfl,
            "nfl_games":                nfl_games,
            "is_bridge":                True,
            "has_nfl_data":             bool(pfr_id),
            "is_udfa":                  bool(row.get("is_udfa", False)),
        })

    labels_df = pd.DataFrame(label_records)

    # Append non-bridge players with role=0, impact=0. Universe source:
    # player_attributes (fresh after step 3) — this is the ONLY thing step 7
    # ever needed from downstream, and sourcing it here lets the retune
    # splice run 7 → 04b/04d → 6 without paying a throwaway full step-6
    # pass first. Snapshot fallback keeps partial runs working.
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"
    universe_path = attrs_path if attrs_path.exists() else snapshot_path
    if universe_path.exists():
        snapshot = pd.read_parquet(universe_path, columns=["player_id"])
        bridge_pids = set(labels_df["player_id"].dropna())
        non_bridge_ids = snapshot[~snapshot["player_id"].isin(bridge_pids)]["player_id"].unique()
        non_bridge_df = pd.DataFrame({
            "player_id":            non_bridge_ids,
            "role_target":          0.0,
            "impact_target":        0.0,
            "impact_apy_target":    0.0,
            "value_target":         0.0,
            "impact_w_av_per_game": np.nan,
            "is_bridge":            False,
            "has_nfl_data":         False,
        })
        labels_df = pd.concat([labels_df, non_bridge_df], ignore_index=True)

    out_path = PROCESSED_DIR / labels_filename
    labels_df.to_parquet(out_path, index=False)
    logger.info("Saved %d label rows → %s", len(labels_df), out_path)

    # Summary
    bridge = labels_df[labels_df["is_bridge"] == True]
    with_pfr = bridge[bridge["has_nfl_data"] == True]
    udfa_bridge = bridge[bridge.get("is_udfa", pd.Series(False, index=bridge.index))]
    logger.info("Bridge players: %d total (%d drafted, %d UDFA), %d matched to NFL data",
                len(bridge), len(bridge) - len(udfa_bridge), len(udfa_bridge), len(with_pfr))
    if not with_pfr.empty:
        traj_role_n = with_pfr["role_trajectory_score"].notna().sum() if "role_trajectory_score" in with_pfr.columns else 0
        traj_impact_n = with_pfr["impact_trajectory_score"].notna().sum() if "impact_trajectory_score" in with_pfr.columns else 0
        logger.info("  role_target: mean=%.3f, nonzero=%d/%d (trajectory score: %d/%d, "
                    "y%d-horizon labels used for %d players)",
                    with_pfr["role_target"].mean(),
                    (with_pfr["role_target"] > 0).sum(),
                    len(with_pfr),
                    traj_role_n, len(with_pfr),
                    ROLE_LABEL_HORIZON_YEARS, n_horizon_used)
        logger.info("  impact_apy_target (2nd contract apy_cap_pct): mean=%.3f%%, nonzero=%d/%d "
                    "(trajectory score: %d/%d)",
                    with_pfr["impact_apy_target"].mean() * 100,
                    (with_pfr["impact_apy_target"] > 0).sum(),
                    len(with_pfr),
                    traj_impact_n, len(with_pfr))
        vt = with_pfr["value_target"]
        logger.info("  value_target: mean=%.3f, zero=%d, NaN=%d of %d — impact_target = %s",
                    vt.mean(), int((vt == 0).sum()), int(vt.isna().sum()), len(with_pfr),
                    "value_target" if IMPACT_TARGET_SOURCE == "value" else "impact_apy_target")
        logger.info("  career_w_av > 0: %d players, mean w_av=%.1f (retained for reference)",
                    (with_pfr["career_w_av"] > 0).sum(),
                    with_pfr[with_pfr["career_w_av"] > 0]["career_w_av"].mean())

    logger.info("Step 7 complete.")


if __name__ == "__main__":
    main()
