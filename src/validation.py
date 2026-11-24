"""
Pipeline output validation.

Each check_stepN() function loads the step's output file(s) and returns a list of
CheckResult objects. run_checks() logs results and optionally exits on ERROR failures.

Usage from any script:
    from src.validation import check_step6, run_checks
    run_checks(check_step6())
"""

import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ATTRIBUTE_MERGE_COLS,
    END_YEAR,
    MAX_BRIDGE_DERIVED_AGE,
    MIN_BRIDGE_DERIVED_AGE,
    N_COMPS,
    OUTPUT_DIR,
    PFF_DATA_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    START_YEAR,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    passed: bool
    name: str
    message: str
    severity: str = "ERROR"  # "ERROR" | "WARNING"


def _chk(condition, name: str, message: str, severity: str = "ERROR") -> CheckResult:
    return CheckResult(passed=bool(condition), name=name, message=message, severity=severity)


def _load(path: Path) -> tuple:
    """Return (df, None) on success; (None, error_CheckResult) if missing."""
    if not path.exists():
        return None, CheckResult(
            False, "file_exists", f"{path.name} not found — run preceding step"
        )
    return pd.read_parquet(path), None


def run_checks(results: list, raise_on_error: bool = True) -> bool:
    """
    Log all CheckResults. Return True if all ERROR-severity checks passed.
    If raise_on_error=True and any ERROR failed, raises SystemExit(1).
    """
    any_error = False
    for r in results:
        if r.passed:
            logger.debug("  OK  %s", r.name)
        else:
            if r.severity == "ERROR":
                logger.error("  FAIL %s: %s", r.name, r.message)
                any_error = True
            else:
                logger.warning("  WARN %s: %s", r.name, r.message)
    if any_error and raise_on_error:
        raise SystemExit(1)
    return not any_error


# ── Step 0 (00* pre-stage outputs) ─────────────────────────────────────────────

def check_step0() -> list:
    """
    Validate the 00* pre-stage outputs that feed steps 1-10:
      - PFF raw CSVs (per-year facets, FBS + FCS supplement)
      - PFR combine supplement parquet
      - combine_metrics.parquet (z-score columns)
      - nfl_league_averages.parquet (with is_forward_filled flag)
      - hasan_ranks.parquet
      - wikidata_birth_dates.parquet
      - player_id_audit.parquet
    """
    results = []

    # PFF raw CSVs (per-year). Latest fully-published season is END_YEAR - 1
    # (e.g. for END_YEAR=2026 the most-recent expected complete season is 2025).
    target_year = END_YEAR - 1
    pff_year_dir = PFF_DATA_DIR / str(target_year)
    results.append(_chk(pff_year_dir.exists(), "pff_year_dir_exists",
                        f"{pff_year_dir} does not exist — run 00_pff_download.py"))
    if pff_year_dir.exists():
        csv_count = len(list(pff_year_dir.glob("*.csv")))
        results.append(_chk(csv_count >= 18, "pff_facet_count",
                            f"PFF {target_year}/ has {csv_count} CSV facets "
                            f"(expected ≥18 incl. FBS + FCS supplement)",
                            severity="WARNING"))
        fcs_count = len(list(pff_year_dir.glob("*_fcs.csv")))
        results.append(_chk(fcs_count >= 8, "pff_fcs_supplement",
                            f"PFF {target_year}/ FCS supplement has "
                            f"{fcs_count} CSVs (expected ≥8 — see 00b)",
                            severity="WARNING"))

    # PFR combine supplement for the active draft year (END_YEAR).
    pfr_combine_path = RAW_DIR / "pfr_combine" / f"pfr_combine_{END_YEAR}.parquet"
    results.append(_chk(pfr_combine_path.exists(),
                        "pfr_combine_supplement_exists",
                        f"{pfr_combine_path.name} not found — "
                        f"run 00d0_pfr_combine_supplement.py if nflverse "
                        f"hasn't published {END_YEAR} yet",
                        severity="WARNING"))

    # combine_metrics.parquet — z-score columns + recent-year coverage.
    combine_path = PROCESSED_DIR / "combine_metrics.parquet"
    df, err = _load(combine_path)
    if err:
        results.append(err)
    else:
        results.append(_chk(len(df) >= 1_000, "combine_min_rows",
                            f"combine_metrics has {len(df)} rows "
                            f"(expected ≥1,000)"))
        z_cols = [c for c in df.columns if c.startswith("combine_") and c.endswith("_z")]
        results.append(_chk(len(z_cols) >= 5, "combine_z_cols",
                            f"Found {len(z_cols)} combine_*_z columns "
                            f"(expected ≥5)"))
        if "season" in df.columns:
            recent = df[df["season"] == END_YEAR]
            results.append(_chk(len(recent) > 0, "combine_target_year",
                                f"combine_metrics has 0 rows for "
                                f"season={END_YEAR}",
                                severity="WARNING"))

    # nfl_league_averages.parquet — must carry is_forward_filled flag and
    # have coverage out to END_YEAR.
    nfl_avg_path = PROCESSED_DIR / "nfl_league_averages.parquet"
    df, err = _load(nfl_avg_path)
    if err:
        results.append(err)
    else:
        results.append(_chk("is_forward_filled" in df.columns,
                            "nfl_avg_forward_fill_flag",
                            "nfl_league_averages.parquet missing "
                            "is_forward_filled column — re-run 00c"))
        if "season" in df.columns:
            max_season = int(df["season"].max())
            results.append(_chk(max_season >= END_YEAR - 1,
                                "nfl_avg_recent_coverage",
                                f"nfl_league_averages max season={max_season} "
                                f"(expected ≥{END_YEAR - 1})"))

    # hasan_ranks.parquet — recruiting/big-board ranks.
    hasan_path = PROCESSED_DIR / "hasan_ranks.parquet"
    df, err = _load(hasan_path)
    if err:
        results.append(err)
    else:
        results.append(_chk(len(df) >= 50, "hasan_min_rows",
                            f"hasan_ranks has {len(df)} rows (expected ≥50)",
                            severity="WARNING"))

    # wikidata_birth_dates.parquet — used by step 03 birth-year resolver.
    wd_path = PROCESSED_DIR / "wikidata_birth_dates.parquet"
    df, err = _load(wd_path)
    if err:
        results.append(err)
    else:
        results.append(_chk(len(df) >= 100, "wikidata_min_rows",
                            f"wikidata_birth_dates has {len(df)} rows "
                            f"(expected ≥100)",
                            severity="WARNING"))

    # player_id_audit.parquet — flagged recycled-id anomalies; informational.
    pid_audit_path = PROCESSED_DIR / "player_id_audit.parquet"
    results.append(_chk(pid_audit_path.exists(),
                        "player_id_audit_exists",
                        "player_id_audit.parquet not found — run "
                        "experiments/00_audit_player_ids.py to refresh recycled-id list",
                        severity="WARNING"))

    return results


# ── Step 1 ─────────────────────────────────────────────────────────────────────

def check_step1() -> list:
    """Validate player_gamelogs.parquet (processed preferred, raw fallback)."""
    path = PROCESSED_DIR / "player_gamelogs.parquet"
    if not path.exists():
        path = RAW_DIR / "player_gamelogs.parquet"

    results = []
    df, err = _load(path)
    if err:
        return [err]

    results.append(_chk(len(df) >= 50_000, "min_rows",
                        f"Expected ≥50,000 rows, got {len(df)}"))
    required = ["player_id", "season_year", "school"]
    missing = [c for c in required if c not in df.columns]
    results.append(_chk(not missing, "required_cols",
                        f"Missing columns: {missing}"))
    if "season_year" in df.columns:
        bad = ~df["season_year"].between(START_YEAR, END_YEAR + 1)
        results.append(_chk(not bad.any(), "season_year_range",
                            f"{bad.sum()} rows have season_year outside [{START_YEAR}, {END_YEAR+1}]"))
    if "player_id" in df.columns:
        null_rate = df["player_id"].isna().mean()
        results.append(_chk(null_rate < 0.01, "player_id_nulls",
                            f"player_id null rate={null_rate:.1%} (threshold <1%)"))
    stat_cols = [c for c in ["pass_att", "rush_att", "rec", "tackles"] if c in df.columns]
    any_nonempty = any(df[c].notna().any() for c in stat_cols)
    results.append(_chk(any_nonempty, "stat_cols_present",
                        "All key stat columns (pass_att, rush_att, rec, tackles) are all-null",
                        severity="WARNING"))
    return results


# ── Step 2 ─────────────────────────────────────────────────────────────────────

def check_step2() -> list:
    """Validate bridge_players.parquet."""
    path = PROCESSED_DIR / "bridge_players.parquet"
    df, err = _load(path)
    if err:
        return [err]

    results = []
    results.append(_chk(len(df) >= 200, "min_rows",
                        f"Expected ≥200 bridge players, got {len(df)}"))
    missing = [c for c in ["player_id", "draft_year"] if c not in df.columns]
    results.append(_chk(not missing, "required_cols",
                        f"Missing columns: {missing}"))
    if "draft_year" in df.columns:
        nn_rate = df["draft_year"].notna().mean()
        results.append(_chk(nn_rate > 0.80, "draft_year_nonnull",
                            f"draft_year non-null rate={nn_rate:.1%} (threshold >80%)"))
        valid = df["draft_year"].dropna()
        in_range = valid.between(1990, END_YEAR).all()
        results.append(_chk(in_range, "draft_year_range",
                            f"Some draft_year values outside [1990, {END_YEAR}]",
                            severity="WARNING"))
    if "pfr_id" in df.columns:
        match_rate = df["pfr_id"].notna().mean()
        results.append(_chk(match_rate > 0.40, "pfr_id_match",
                            f"pfr_id match rate={match_rate:.1%} (threshold >40%)",
                            severity="WARNING"))
    if "position_group" in df.columns:
        pg_rate = df["position_group"].notna().mean()
        results.append(_chk(pg_rate > 0.70, "position_group_coverage",
                            f"position_group non-null rate={pg_rate:.1%} (threshold >70%)",
                            severity="WARNING"))
    return results


# ── Step 3 ─────────────────────────────────────────────────────────────────────

def check_step3() -> list:
    """Validate player_attributes.parquet."""
    path = PROCESSED_DIR / "player_attributes.parquet"
    df, err = _load(path)
    if err:
        return [err]

    results = []
    results.append(_chk(len(df) >= 5_000, "min_rows",
                        f"Expected ≥5,000 rows, got {len(df)}"))
    missing = [c for c in ["player_id", "position_group"] if c not in df.columns]
    results.append(_chk(not missing, "required_cols",
                        f"Missing columns: {missing}"))
    if "height_in" in df.columns:
        h = df["height_in"].dropna()
        results.append(_chk(h.between(60, 82).all(), "height_range",
                            f"{(~h.between(60, 82)).sum()} height_in values outside [60, 82]",
                            severity="WARNING"))
    if "weight_lbs" in df.columns:
        w = df["weight_lbs"].dropna()
        results.append(_chk(w.between(150, 380).all(), "weight_range",
                            f"{(~w.between(150, 380)).sum()} weight_lbs values outside [150, 380]",
                            severity="WARNING"))
    if "position_group" in df.columns:
        pg_rate = df["position_group"].notna().mean()
        results.append(_chk(pg_rate > 0.70, "position_group_coverage",
                            f"position_group non-null rate={pg_rate:.1%} (threshold >70%)",
                            severity="WARNING"))

    if {"birth_year", "last_college_season_year"}.issubset(df.columns):
        age = df["last_college_season_year"] - df["birth_year"]
        n_too_old = int((age > MAX_BRIDGE_DERIVED_AGE).sum())
        n_too_young = int((age < MIN_BRIDGE_DERIVED_AGE).sum())
        results.append(_chk(n_too_old < 10, "implausible_age_too_old",
                            f"{n_too_old} players have age > {MAX_BRIDGE_DERIVED_AGE} at last college season "
                            f"(suspect cross-generational namesake join)"))
        results.append(_chk(n_too_young < 10, "implausible_age_too_young",
                            f"{n_too_young} players have age < {MIN_BRIDGE_DERIVED_AGE} at last college season",
                            severity="WARNING"))

    # Roster-only fix invariant: every column declared in ATTRIBUTE_MERGE_COLS
    # must be present in player_attributes.parquet, otherwise it never reaches
    # downstream training/scoring (LightGBM swallows missing columns silently).
    missing_attrs = [c for c in ATTRIBUTE_MERGE_COLS if c not in df.columns]
    results.append(_chk(not missing_attrs, "attribute_merge_cols_present",
                        f"ATTRIBUTE_MERGE_COLS missing from player_attributes: {missing_attrs}"))

    # Career team-context features (added 2026-05-03 with the PFF rate-feature
    # work) must have meaningful coverage. <50% means the advanced_game_stats
    # join in pipeline/03_player_attributes.py:498-565 regressed.
    team_ctx_cols = [c for c in ATTRIBUTE_MERGE_COLS
                     if c.startswith("career_team_") and c in df.columns]
    for col in team_ctx_cols:
        cov = df[col].notna().mean()
        results.append(_chk(cov > 0.50, f"career_team_coverage_{col}",
                            f"{col} non-null rate={cov:.1%} (threshold >50%)",
                            severity="WARNING"))
    return results


# ── Step 3 supplement: PFF feature presence ────────────────────────────────────

# Suffixes / substrings that identify *engineered* PFF feature names —
# computed inside src/scrapers/pff.py:_add_season_level_rates at step 8 time
# (called from load_pff_career_features), NOT persisted to the season-level
# pff_season_grades.parquet. These are checked at the trained-model layer in
# experiments/pipeline_audit/audit_feature_flow.py instead.
_PFF_ENGINEERED_SUFFIXES = (
    "_per_pr_snap",
    "_per_def_snap",
    "_per_off_snap",
    "_per_cov_snap",
    "_per_attempt",
    "_delta",
)
_PFF_ENGINEERED_SUBSTRINGS = ("pff_true_pass_set_",)   # _rate variants


def _is_pff_engineered_name(col: str) -> bool:
    if any(col.endswith(suf) for suf in _PFF_ENGINEERED_SUFFIXES):
        return True
    # true_pass_set rate variants (e.g. pff_true_pass_set_pressures_allowed_rate)
    if any(sub in col for sub in _PFF_ENGINEERED_SUBSTRINGS) and col.endswith("_rate"):
        return True
    return False


def check_step3_pff_features() -> list:
    """
    Validate that the *raw* PFF columns declared in PFF_GRADE_FEATURES_BY_GROUP
    are present in pff_season_grades.parquet. Engineered rate / delta / per-snap
    names are skipped here — they're computed transiently inside step 8's
    load_pff_career_features and surface in trained-model feature_cols, not
    in the season-level parquet. The trained-model lineage check lives in
    experiments/pipeline_audit/audit_feature_flow.py.

    A missing raw name = silent feature drop at season-grade ingestion time.
    """
    from config import PFF_GRADE_FEATURES_BY_GROUP

    path = PROCESSED_DIR / "pff_season_grades.parquet"
    df, err = _load(path)
    if err:
        return [err]

    results = []
    declared_raw = sorted({
        feat
        for feats in PFF_GRADE_FEATURES_BY_GROUP.values()
        for feat in feats
        if not _is_pff_engineered_name(feat)
    })
    missing = [c for c in declared_raw if c not in df.columns]
    results.append(_chk(not missing, "pff_raw_features_present",
                        f"{len(missing)} raw PFF columns declared in "
                        f"PFF_GRADE_FEATURES_BY_GROUP missing from "
                        f"pff_season_grades.parquet: "
                        f"{missing[:8]}{'…' if len(missing) > 8 else ''}"))

    # Spot-check coverage on a few high-value raw columns. <5% non-null
    # would suggest the FBS+FCS supplement union regressed for that report.
    spot_check_cols = [
        "pff_grades_offense",
        "pff_grades_defense",
        "pff_snap_counts_offense",
        "pff_snap_counts_defense",
        "pff_snap_counts_pass_rush",
        "pff_snap_counts_coverage",
    ]
    for col in spot_check_cols:
        if col not in df.columns:
            continue
        cov = df[col].notna().mean()
        results.append(_chk(cov > 0.05, f"pff_raw_coverage_{col}",
                            f"{col} non-null rate={cov:.1%} (threshold >5%)",
                            severity="WARNING"))
    return results


# ── Step 4 ─────────────────────────────────────────────────────────────────────

def check_step4() -> list:
    """Validate contextual_models/ pkl files and model_summary.json."""
    models_dir = PROCESSED_DIR / "contextual_models"
    results = []
    results.append(_chk(models_dir.exists(), "models_dir_exists",
                        f"{models_dir} does not exist"))
    if not models_dir.exists():
        return results

    pkl_files = list(models_dir.glob("*.pkl"))
    results.append(_chk(len(pkl_files) >= 3, "min_model_files",
                        f"Expected ≥3 model .pkl files, found {len(pkl_files)}"))

    summary_path = models_dir / "model_summary.json"
    results.append(_chk(summary_path.exists(), "model_summary_exists",
                        "model_summary.json not found",
                        severity="WARNING"))
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            # model_summary.json is a list of per-(pos, stat) records (each
            # carries position_group, stat, cv_r2, ...). Older code returned
            # a {key: record} dict; handle both shapes for robustness.
            if isinstance(summary, list):
                bad = [
                    f"{rec.get('position_group','?')}_{rec.get('stat','?')}"
                    for rec in summary
                    if isinstance(rec, dict) and rec.get("cv_r2", 0) < -2.0
                ]
            elif isinstance(summary, dict):
                bad = [k for k, v in summary.items()
                       if isinstance(v, dict) and v.get("cv_r2", 0) < -2.0]
            else:
                bad = []
            results.append(_chk(not bad, "cv_r2_sane",
                                f"Models with CV R² < -2.0: {bad}",
                                severity="WARNING"))
        except Exception as exc:
            results.append(_chk(False, "cv_r2_sane",
                                f"Could not parse model_summary.json: {exc}",
                                severity="WARNING"))
    return results


# ── Step 5 ─────────────────────────────────────────────────────────────────────

def check_step5() -> list:
    """Validate adjusted_gamelogs.parquet."""
    path = PROCESSED_DIR / "adjusted_gamelogs.parquet"
    df, err = _load(path)
    if err:
        return [err]

    results = []
    results.append(_chk(len(df) >= 50_000, "min_rows",
                        f"Expected ≥50,000 rows, got {len(df)}"))
    adj_cols = [c for c in df.columns if c.startswith("adjusted_")]
    results.append(_chk(len(adj_cols) >= 3, "adjusted_cols_present",
                        f"Expected ≥3 adjusted_ columns, found {len(adj_cols)}"))
    if adj_cols:
        has_inf = np.isinf(df[adj_cols].select_dtypes("number").values).any()
        results.append(_chk(not has_inf, "no_inf",
                            "Inf values found in adjusted_ columns"))
        nan_rate = df[adj_cols].isna().mean().mean()
        results.append(_chk(nan_rate < 0.80, "nan_rate",
                            f"Mean NaN rate across adjusted_ cols={nan_rate:.1%} (threshold <80%)",
                            severity="WARNING"))
    return results


# ── Step 6 ─────────────────────────────────────────────────────────────────────

def check_step6() -> list:
    """Validate player_skill_snapshot.parquet."""
    path = PROCESSED_DIR / "player_skill_snapshot.parquet"
    df, err = _load(path)
    if err:
        return [err]

    results = []
    results.append(_chk(len(df) >= 500, "min_players",
                        f"Expected ≥500 players in snapshot, got {len(df)}"))
    k_cols = [c for c in df.columns if c.startswith("kalman_mean_")]
    results.append(_chk(len(k_cols) >= 3, "kalman_cols_present",
                        f"Expected ≥3 kalman_mean_* columns, found {len(k_cols)}"))
    if k_cols:
        num_vals = df[k_cols].select_dtypes("number")
        has_inf = np.isinf(num_vals.values).any()
        results.append(_chk(not has_inf, "no_inf",
                            "Inf values found in kalman_mean_ columns"))
        out_of_range = ((num_vals.abs() > 500) & num_vals.notna()).any().any()
        results.append(_chk(not out_of_range, "kalman_range",
                            "Some kalman_mean_ values outside (-500, 500)",
                            severity="WARNING"))
    if "n_games" in df.columns:
        med = df["n_games"].median()
        results.append(_chk(med > 1, "n_games_present",
                            f"n_games median={med} (expected >1)",
                            severity="WARNING"))
    else:
        results.append(_chk(False, "n_games_present",
                            "n_games column absent from snapshot",
                            severity="WARNING"))

    # n_seasons (redefined 2026-05-03 over union of rosters ∪ player_seasons ∪
    # gamelogs) should be ≥ 1 for every snapshot row, including roster-only
    # skeleton rows. A zero would mean step 6's union recomputation regressed.
    if "n_seasons" in df.columns:
        n_zero = int((df["n_seasons"] == 0).sum())
        results.append(_chk(n_zero == 0, "n_seasons_nonzero",
                            f"{n_zero} snapshot rows have n_seasons=0 "
                            f"(union over rosters/player_seasons/gamelogs "
                            f"should yield ≥1 for every player)"))
    else:
        results.append(_chk(False, "n_seasons_present",
                            "n_seasons column absent from snapshot",
                            severity="WARNING"))
    return results


# ── Step 7 ─────────────────────────────────────────────────────────────────────

def check_step7() -> list:
    """Validate nfl_draft_labels.parquet."""
    path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    df, err = _load(path)
    if err:
        return [err]

    results = []
    bridge = df[df["is_bridge"] == True] if "is_bridge" in df.columns else df
    results.append(_chk(len(bridge) >= 50, "min_bridge",
                        f"Expected ≥50 bridge players, found {len(bridge)}"))
    if "role_target" in bridge.columns:
        rt = bridge["role_target"].dropna()
        results.append(_chk(rt.between(0, 1).all(), "role_target_range",
                            f"{(~rt.between(0, 1)).sum()} bridge role_target values outside [0, 1]"))
    if "impact_target" in bridge.columns:
        it_ = bridge["impact_target"].dropna()
        results.append(_chk((it_ >= 0).all(), "impact_target_nonneg",
                            f"{(it_ < 0).sum()} bridge impact_target values < 0"))
    if "value_target" in bridge.columns:
        vt = bridge["value_target"].dropna()
        results.append(_chk(vt.between(0, 1).all(), "value_target_range",
                            f"{(~vt.between(0, 1)).sum()} bridge value_target values outside [0, 1]"))
    results.append(_chk("career_w_av" in df.columns, "career_w_av_present",
                        "career_w_av column missing from labels",
                        severity="WARNING"))
    if "career_w_av" in df.columns:
        av = df["career_w_av"].dropna()
        results.append(_chk((av >= 0).all(), "career_w_av_nonneg",
                            f"{(av < 0).sum()} career_w_av values < 0",
                            severity="WARNING"))
    return results


# ── Step 8 ─────────────────────────────────────────────────────────────────────

def check_step8() -> list:
    """Validate draft_models/ pkl files and training_metrics.json."""
    models_dir = PROCESSED_DIR / "draft_models"
    results = []
    results.append(_chk(models_dir.exists(), "models_dir_exists",
                        f"{models_dir} does not exist"))
    if not models_dir.exists():
        return results

    pkl_files = list(models_dir.glob("*.pkl"))
    results.append(_chk(len(pkl_files) >= 4, "min_model_files",
                        f"Expected ≥4 model .pkl files, found {len(pkl_files)}"))

    bad_bundles = []
    for p in pkl_files:
        try:
            with open(p, "rb") as f:
                bundle = pickle.load(f)
            if "model" not in bundle or "feature_cols" not in bundle:
                bad_bundles.append(p.name)
        except Exception as exc:
            bad_bundles.append(f"{p.name} ({exc})")
    results.append(_chk(not bad_bundles, "bundle_keys",
                        f"Malformed bundles: {bad_bundles}"))

    metrics_path = models_dir / "training_metrics.json"
    results.append(_chk(metrics_path.exists(), "training_metrics_exists",
                        "training_metrics.json not found",
                        severity="WARNING"))
    if metrics_path.exists():
        try:
            import json as _json
            with open(metrics_path) as fp:
                metrics = _json.load(fp)
            missing_holdout = [k for k, v in metrics.items()
                               if "holdout_mae" not in v]
            results.append(_chk(
                not missing_holdout, "holdout_metrics_present",
                f"{len(missing_holdout)} models missing holdout_mae: "
                f"{missing_holdout[:5]}{'…' if len(missing_holdout) > 5 else ''}",
                severity="WARNING",
            ))
        except Exception as exc:
            results.append(_chk(False, "training_metrics_parseable",
                                f"failed to parse training_metrics.json: {exc}",
                                severity="WARNING"))
    return results


# ── Step 9 ─────────────────────────────────────────────────────────────────────

def check_step9() -> list:
    """Validate player_comps.parquet."""
    path = PROCESSED_DIR / "player_comps.parquet"
    df, err = _load(path)
    if err:
        return [err]

    results = []
    results.append(_chk(len(df) >= 50, "min_rows",
                        f"Expected ≥50 comp rows, got {len(df)}"))
    if "comp_rank" in df.columns:
        in_range = df["comp_rank"].between(1, N_COMPS).all()
        results.append(_chk(in_range, "comp_rank_range",
                            f"comp_rank values outside [1, {N_COMPS}]"))
    missing = [c for c in ["comp_career_w_av", "comp_distance"] if c not in df.columns]
    results.append(_chk(not missing, "required_cols",
                        f"Missing columns: {missing}"))
    if "comp_distance" in df.columns:
        neg = (df["comp_distance"] < 0).any()
        results.append(_chk(not neg, "comp_distance_nonneg",
                            "Negative comp_distance values found",
                            severity="WARNING"))
    return results


# ── Step 10 ────────────────────────────────────────────────────────────────────

def check_step10() -> list:
    """Validate projections/latest.parquet."""
    path = OUTPUT_DIR / "projections" / "latest.parquet"
    df, err = _load(path)
    if err:
        return [err]

    results = []
    results.append(_chk(len(df) >= 100, "min_rows",
                        f"Expected ≥100 rows, got {len(df)}"))
    missing = [c for c in ["impact_score"] if c not in df.columns]
    results.append(_chk(not missing, "score_cols_present",
                        f"Missing columns: {missing}"))
    if "impact_score" in df.columns:
        coverage = df["impact_score"].notna().mean()
        results.append(_chk(coverage > 0.05, "impact_score_coverage",
                            f"impact_score non-null rate={coverage:.1%} (threshold >5%)",
                            severity="WARNING"))
    if "player_id" in df.columns:
        n_dup = df["player_id"].duplicated().sum()
        results.append(_chk(n_dup == 0, "no_dup_player_id",
                            f"{n_dup} duplicate player_id values",
                            severity="WARNING"))
    return results
