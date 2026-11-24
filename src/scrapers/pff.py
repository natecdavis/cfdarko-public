"""
PFF college football season-grade data loading and player matching utilities.

Loads season-level CSVs exported from premium.pff.com/ncaa, normalizes player
and school names, and aggregates per-player career features (mean/peak/last).
"""

from __future__ import annotations    # PEP 604 syntax compat for Python <3.10

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# ── PFF position code → cfDARKO position_group ────────────────────────────────

PFF_POSITION_TO_GROUP: dict[str, str] = {
    "QB": "QB",
    "HB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "T":  "OL", "G":  "OL", "C":  "OL",
    "ED": "DL", "DI": "DL",
    "LB": "LB",
    "CB": "DB", "S":  "DB",
    "K":  "K",
    "P":  "P",
}

# ── Known PFF team name → CFBD school name overrides ─────────────────────────

PFF_SCHOOL_OVERRIDES: dict[str, str] = {
    "FAU":         "Florida Atlantic",
    "USF":         "South Florida",
    "UMass":       "Massachusetts",
    "UT San Antonio": "UTSA",
    "Louisiana":   "Louisiana",
    "App State":   "Appalachian State",
    "Coastal Car": "Coastal Carolina",
    "Old Dom":     "Old Dominion",
    "Sam Houst":   "Sam Houston",
    "Jacksonville St": "Jacksonville State",
    "FIU":         "Florida International",
    "NIU":         "Northern Illinois",
    "WKU":         "Western Kentucky",
    "UTEP":        "UTEP",
    "UConn":       "Connecticut",
    "UCF":         "UCF",
    "BYU":         "BYU",
    "SMU":         "SMU",
    "TCU":         "TCU",
    "LSU":         "LSU",
    "UAB":         "UAB",
}

# ── Which reports to load per position group ──────────────────────────────────

REPORTS_BY_GROUP: dict[str, list[str]] = {
    "QB":  ["passing_summary", "passing_allowed_pressure"],
    "RB":  ["rushing_summary", "receiving_summary", "receiving_scheme", "offense_blocking"],
    "WR":  ["receiving_summary", "receiving_scheme", "offense_blocking"],
    "TE":  ["receiving_summary", "receiving_scheme", "offense_blocking"],
    "OL":  ["offense_blocking", "offense_pass_blocking", "offense_run_blockng"],
    "DL":  ["defense_summary", "pass_rush_summary", "run_defense_summary"],
    "LB":  ["defense_summary", "pass_rush_summary", "defense_coverage_summary",
            "defense_coverage_scheme", "run_defense_summary"],
    "DB":  ["defense_summary", "defense_coverage_summary", "defense_coverage_scheme",
            "run_defense_summary"],
    "K":   ["field_goal_summary"],
    "P":   ["punting_summary"],
}

# ── Grade/stat columns to extract per report ──────────────────────────────────
# Keys are report filenames (without .csv); values are columns to keep.
# "player", "player_id", "position", "team_name" are always kept.

_REPORT_GRADE_COLS: dict[str, list[str]] = {
    "passing_summary": [
        "grades_pass", "grades_offense",
        "btt_rate", "twp_rate", "accuracy_percent",
        "passing_snaps", "sack_percent", "completion_percent", "avg_time_to_throw",
        "drop_rate", "pressure_to_sack_rate", "qb_rating", "scrambles",
        "epa", "positive_epa_percent", "dropbacks", "attempts",
        "first_downs", "hit_as_threw",
    ],
    "passing_allowed_pressure": [
        "pressure_grades_pass", "no_pressure_grades_pass",
        "pressure_completion_percent", "no_pressure_completion_percent",
        "pressure_sack_percent", "no_pressure_sack_percent",
        "blitz_grades_pass", "no_blitz_grades_pass",
        "blitz_btt_rate", "no_blitz_btt_rate",
    ],
    "rushing_summary": [
        "grades_run", "grades_offense",
        "elusive_rating", "breakaway_percent", "yco_attempt",
        "grades_pass_route", "yprr",
        # `attempts` is the rush-attempt denominator for every rushing rate in
        # PFF_DENOM_COL_BY_STAT; without it the whole RB rushing-grade family
        # drops out of Bayesian shrinkage aggregation as NaN-denominator rows.
        "attempts",
        "routes", "targets",
        "yards_after_contact", "avoided_tackles",
    ],
    "receiving_summary": [
        "grades_pass_route", "grades_offense",
        "yprr", "avg_depth_of_target", "route_rate", "inline_rate",
        "grades_pass_block",
        "slot_rate", "wide_rate", "contested_catch_rate", "yards_per_reception",
        "inline_snaps", "slot_snaps",
        "grades_hands_drop", "drop_rate", "targeted_qb_rating",
        "yards_after_catch_per_reception",
        # Route/target denominators for the WR/TE receiving rates. RB picks
        # these up from rushing_summary; WR/TE never load that report.
        "routes", "targets",
    ],
    "receiving_scheme": [
        "man_grades_pass_route", "zone_grades_pass_route",
        "man_yprr", "zone_yprr",
        "man_targeted_qb_rating", "zone_targeted_qb_rating",
    ],
    "offense_blocking": [
        "grades_offense", "grades_pass_block", "grades_run_block",
        "pass_block_percent", "pressures_allowed",
        "snap_counts_offense",
        "sacks_allowed", "hurries_allowed", "pbe",
        "hits_allowed",
    ],
    "offense_pass_blocking": [
        "grades_pass_block", "pass_block_percent", "pressures_allowed",
        # True pass set: pre-snap pass-blocking only — strips out PA, screens,
        # scrambles. Cleanest OL pass-block evaluation PFF emits.
        "true_pass_set_grades_pass_block", "true_pass_set_pressures_allowed",
        "true_pass_set_hurries_allowed", "true_pass_set_sacks_allowed",
        "true_pass_set_hits_allowed", "true_pass_set_pbe",
        "true_pass_set_snap_counts_pass_block",
    ],
    "offense_run_blockng": [
        "grades_run_block",
        "gap_grades_run_block", "zone_grades_run_block",
        # `run_block_percent` dropped 2026-07-22: degenerate. It only ever
        # populated on OL rows, and 99.36% of those 32,099 rows were exactly
        # 100.0 (min 50, p1 = p50 = max = 100) — linemen run-block on
        # essentially every run snap, so the column has no variance to offer.
        # Worse, it slipped past the >99% dominant-value filter downstream:
        # Bayesian shrinkage spreads a constant-100 column across the 4th
        # decimal (peak triplet lived in [99.994, 99.9996]), so the filter saw
        # thousands of "distinct" values and kept a pure-noise feature.
    ],
    "defense_summary": [
        "grades_defense", "grades_pass_rush_defense", "grades_run_defense",
        "grades_coverage_defense", "qb_rating_against",
        "snap_counts_defense", "snap_counts_corner", "snap_counts_slot",
        "snap_counts_fs", "snap_counts_pass_rush", "snap_counts_coverage",
        "total_pressures", "missed_tackle_rate", "catch_rate",
        # Coverage-target denominator for `catch_rate` shrinkage on DB/LB rows;
        # the offensive `targets` column only reaches RB/WR/TE.
        "targets",
        "grades_tackle", "pass_break_ups", "sacks",
        # Granular pass-rush + splash production missing from prior config.
        "hurries", "hits", "tackles_for_loss",
        "forced_fumbles", "interceptions",
    ],
    "pass_rush_summary": [
        "grades_pass_rush_defense", "pass_rush_win_rate", "prp",
        "pass_rush_opp", "pass_rush_percent",
    ],
    "defense_coverage_summary": [
        "grades_coverage_defense", "qb_rating_against", "forced_incompletion_rate",
        # Coverage tightness as proper rates (LightGBM doesn't have to derive
        # them from snap denominators it may not see consistently).
        "yards_per_coverage_snap", "coverage_snaps_per_target",
        "coverage_snaps_per_reception", "forced_incompletes",
    ],
    "defense_coverage_scheme": [
        "man_grades_coverage_defense", "zone_grades_coverage_defense",
        "man_qb_rating_against", "zone_qb_rating_against",
        "man_forced_incompletion_rate", "zone_forced_incompletion_rate",
        "man_snap_counts_coverage", "zone_snap_counts_coverage",
    ],
    "run_defense_summary": [
        "grades_defense", "grades_run_defense",
        "stop_percent", "snap_counts_run", "missed_tackle_rate",
        # Depth-of-tackle is a key DL/LB run-defense signal — closer to LOS
        # = penetration, deeper = tackling after gain.
        "avg_depth_of_tackle", "forced_fumbles", "run_stop_opp",
    ],
    "field_goal_summary": [
        "grades_fgep_kicker", "total_percent", "fifty_percent",
        "thirty_percent", "forty_percent",
        # Attempt counts give the model sample-size context for the
        # accuracy rates above; without them a 1-of-1 50+ kicker looks
        # identical to a 12-of-12 50+ kicker.
        "total_attempts", "fifty_attempts", "forty_attempts",
        "thirty_attempts", "twenty_attempts", "twenty_percent",
        "pat_attempts", "pat_percent",
    ],
    "punting_summary": [
        "grades_punter", "average_net_yards", "average_yards_per_attempt", "inside_twenties",
        "average_hangtime", "touchbacks",
        "attempts", "long", "blocks", "total_hangtime", "attempts_with_hangtime",
    ],
}


# ── Normalization helpers ─────────────────────────────────────────────────────

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = _SUFFIX_RE.sub("", name.lower())
    return re.sub(r"[^a-z]", "", name)


def _normalize_school_base(school: str) -> str:
    if not isinstance(school, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", school.lower())


def normalize_pff_school(name: str) -> str:
    """Normalize a PFF team_name to the same key space as CFBD school names."""
    if not isinstance(name, str):
        return ""
    resolved = PFF_SCHOOL_OVERRIDES.get(name, name)
    return _normalize_school_base(resolved)


# ── Report loading ────────────────────────────────────────────────────────────

def load_pff_report(season: int, report: str, pff_dir: Path) -> pd.DataFrame | None:
    """Load one PFF CSV report for a season. Returns None if file is missing.

    Also concatenates an `{report}_fcs.csv` supplement if present (written by
    `pipeline/00b_pff_fcs_supplement.py`). PFF's bulk export is FBS-only; the
    supplement adds FCS players via the JSON facet API.
    """
    base = pff_dir / str(season) / f"{report}.csv"
    fcs  = pff_dir / str(season) / f"{report}_fcs.csv"
    parts: list[pd.DataFrame] = []
    for path, source in [(base, "fbs"), (fcs, "fcs")]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            df["season_year"] = season
            df["_pff_source"] = source
            parts.append(df)
        except Exception as exc:
            logger.warning("Failed loading %s: %s", path, exc)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True, sort=False)


# ── Season-level grade extraction ─────────────────────────────────────────────

def extract_season_grades(season: int, pff_dir: Path) -> pd.DataFrame:
    """
    Load all relevant PFF reports for one season, extract grade columns, and
    return a unified DataFrame with one row per (pff_player_id, season_year).

    Columns: pff_player_id, player_name_norm, school_norm, season_year,
             position_group, pff_{col} for all grade cols present.
    """
    all_rows: list[pd.DataFrame] = []

    # Track which player_ids we've seen per group to avoid double-counting
    # when the same player appears in multiple reports for the same group.
    seen: dict[str, set] = {}  # pos_group → set of pff_player_id

    for pos_group, reports in REPORTS_BY_GROUP.items():
        group_frames: list[pd.DataFrame] = []
        seen.setdefault(pos_group, set())

        for report in reports:
            df = load_pff_report(season, report, pff_dir)
            if df is None:
                continue

            # Filter to relevant positions for this group
            relevant_positions = {
                code for code, grp in PFF_POSITION_TO_GROUP.items() if grp == pos_group
            }
            if "position" in df.columns:
                df = df[df["position"].isin(relevant_positions)]

            if df.empty:
                continue

            # Keep only grade/stat columns present in this report
            wanted_cols = _REPORT_GRADE_COLS.get(report, [])
            keep = ["player", "player_id", "position", "team_name", "season_year"]
            grade_cols_present = [c for c in wanted_cols if c in df.columns]
            keep += grade_cols_present

            sub = df[keep].copy()
            sub["_report"] = report
            group_frames.append(sub)

        if not group_frames:
            continue

        # Merge all reports for this group on player_id
        combined = group_frames[0]
        for frame in group_frames[1:]:
            grade_cols = [c for c in frame.columns
                          if c not in ("player", "player_id", "position", "team_name",
                                       "season_year", "_report")]
            merge_cols = ["player_id", "season_year"]
            meta_present = [c for c in ("player", "position", "team_name") if c in frame.columns]
            right_frame = frame[merge_cols + grade_cols + meta_present].drop_duplicates("player_id")
            combined = combined.merge(right_frame, on=merge_cols, how="outer", suffixes=("", "_r"))
            # Coalesce duplicate columns
            for col in grade_cols:
                if f"{col}_r" in combined.columns:
                    combined[col] = combined[col].combine_first(combined[f"{col}_r"])
                    combined.drop(columns=[f"{col}_r"], inplace=True)
            # Backfill player/team info from right frame
            for meta in ("player", "position", "team_name"):
                if f"{meta}_r" in combined.columns:
                    combined[meta] = combined[meta].combine_first(combined[f"{meta}_r"])
                    combined.drop(columns=[f"{meta}_r"], inplace=True)

        combined["position_group"] = pos_group
        combined["player_name_norm"] = combined["player"].apply(_normalize_name)
        combined["school_norm"] = combined["team_name"].apply(normalize_pff_school)
        # Preserve raw PFF position code (T/G/C for OL, ED/DI for DL, etc.)
        if "position" in combined.columns:
            combined["pff_position"] = combined["position"]

        # Rename grade columns with pff_ prefix
        rename = {}
        for col in combined.columns:
            if col not in ("player", "player_id", "position", "team_name", "season_year",
                           "position_group", "player_name_norm", "school_norm", "_report",
                           "pff_position"):
                rename[col] = f"pff_{col}"
        combined.rename(columns=rename, inplace=True)
        combined.rename(columns={"player_id": "pff_player_id"}, inplace=True)
        combined.drop(columns=["_report", "player", "position", "team_name"],
                      errors="ignore", inplace=True)

        all_rows.append(combined)

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    # Convert grade columns to numeric
    for col in result.columns:
        if col.startswith("pff_") and col not in ("pff_player_id", "pff_position"):
            result[col] = pd.to_numeric(result[col], errors="coerce")

    result = _clamp_percent_columns(result)
    return result


# Raw PFF `*_percent` / `*_rate` export columns are all percentage-scaled
# (verified 2026-07-22: every one has p99 <= 100), so values above 100 are
# impossible. The feed carries 219 of them across 8 columns — pass_block_percent
# up to 200, forced_incompletion_rate up to 400 — essentially all on tiny
# denominators (14-30 snaps), plus a cluster of benign 100.1-100.6 roundings
# where PFF charges a lineman marginally more block snaps than offensive snaps.
#
# Clamped rather than nulled: for the 100.x rows the true value plainly IS 100
# (a center who blocked on all 951 snaps), and for the small-sample outliers
# the row's denominator is still real, so it should keep its (tiny) shrinkage
# weight rather than drop out of the pooled prior entirely.
#
# Applied to RAW ingested columns only, before `_add_season_level_rates` — the
# rates synthesized there are fractions in [0, 1], not percentages.
_PERCENT_COL_SUFFIXES: tuple[str, ...] = ("_percent", "_rate")


def _clamp_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp raw PFF percentage columns into their semantic [0, 100] range."""
    n_clamped = 0
    affected: list[str] = []
    for col in df.columns:
        if not col.startswith("pff_") or not col.endswith(_PERCENT_COL_SUFFIXES):
            continue
        s = df[col]
        out_of_range = ((s < 0) | (s > 100)).sum()
        if out_of_range:
            n_clamped += int(out_of_range)
            affected.append(col)
            df[col] = s.clip(lower=0.0, upper=100.0)
    if n_clamped:
        logger.info("Clamped %d out-of-range PFF percentage value(s) to [0,100] "
                    "across %d column(s): %s",
                    n_clamped, len(affected), ", ".join(sorted(affected)))
    return df


# ── Career feature aggregation ────────────────────────────────────────────────

def _add_season_level_rates(grades: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize per-season rate / delta columns from raw PFF counts before
    career aggregation. Rates are computed within each (player, season)
    so the downstream mean/peak/last triple reflects true rate variation
    rather than mean(num)/mean(denom) which over- or under-states the rate
    when seasons differ in volume.

    Counts contaminated by team scheme (e.g. true-pass-set snaps for OL on
    run-heavy teams) become per-snap rates that normalize scheme out — this
    is the systemic remedy for the Spencer Fano case where Utah's run-heavy
    2025 scheme dropped his true_pass_set count to 101 despite a full
    starting season.

    Denominator clamped to NaN for 0 / NaN; numerators with no denominator
    yield NaN, which LightGBM handles via missing-direction splits.
    """
    def rate(num_col: str, denom_col: str) -> pd.Series:
        if num_col not in grades.columns or denom_col not in grades.columns:
            return pd.Series(np.nan, index=grades.index)
        denom = grades[denom_col].replace(0, np.nan)
        return grades[num_col] / denom

    def delta(a_col: str, b_col: str) -> pd.Series:
        if a_col not in grades.columns or b_col not in grades.columns:
            return pd.Series(np.nan, index=grades.index)
        return grades[a_col] - grades[b_col]

    # ── Defense pass-rush rates (per pass-rush snap) ──────────────────────
    for evt in ("sacks", "hits", "hurries", "total_pressures"):
        grades[f"pff_{evt}_per_pr_snap"] = rate(f"pff_{evt}", "pff_snap_counts_pass_rush")

    # ── Defense event rates (per defensive snap) ──────────────────────────
    for evt in ("tackles_for_loss", "forced_fumbles", "interceptions"):
        grades[f"pff_{evt}_per_def_snap"] = rate(f"pff_{evt}", "pff_snap_counts_defense")

    # ── Coverage event rates (per coverage snap) ──────────────────────────
    grades["pff_pass_break_ups_per_cov_snap"] = rate(
        "pff_pass_break_ups", "pff_snap_counts_coverage"
    )
    grades["pff_forced_incompletes_per_cov_snap"] = rate(
        "pff_forced_incompletes", "pff_snap_counts_coverage"
    )

    # ── OL pressures-allowed rates (per offensive snap) ───────────────────
    # `pff_snap_counts_pass_block` doesn't exist as a separate column in PFF
    # exports; offense-snaps is the closest universally-available denominator.
    for evt in ("pressures_allowed", "sacks_allowed", "hurries_allowed", "hits_allowed"):
        grades[f"pff_{evt}_per_off_snap"] = rate(f"pff_{evt}", "pff_snap_counts_offense")

    # ── OL true-pass-set rates (cleanest pass-pro context) ────────────────
    # True pass set excludes screens / play-action / RPOs — purer measure
    # of pass-blocking quality. Denominator is the true-pass-set snap count.
    for evt in ("pressures_allowed", "sacks_allowed", "hurries_allowed", "hits_allowed"):
        grades[f"pff_true_pass_set_{evt}_rate"] = rate(
            f"pff_true_pass_set_{evt}",
            "pff_true_pass_set_snap_counts_pass_block",
        )

    # ── Receiving rates (per offensive snap) ──────────────────────────────
    for evt in ("routes", "targets"):
        grades[f"pff_{evt}_per_off_snap"] = rate(f"pff_{evt}", "pff_snap_counts_offense")

    # ── Tier 2: man-vs-zone deltas (separation / coverage proxies) ────────
    grades["pff_man_zone_route_grade_delta"] = delta(
        "pff_man_grades_pass_route", "pff_zone_grades_pass_route"
    )
    grades["pff_man_zone_yprr_delta"] = delta("pff_man_yprr", "pff_zone_yprr")
    grades["pff_man_zone_qb_rating_against_delta"] = delta(
        "pff_man_qb_rating_against", "pff_zone_qb_rating_against"
    )
    grades["pff_man_zone_coverage_grade_delta"] = delta(
        "pff_man_grades_coverage_defense", "pff_zone_grades_coverage_defense"
    )

    # ── Tier 2: RB rushing elusiveness rate ───────────────────────────────
    # Rush attempts come straight off the rushing report as `pff_attempts`.
    # PFF also defines `pff_yco_attempt` as yards-after-contact per attempt,
    # so attempts are recoverable as yards_after_contact / yco_attempt where
    # the raw column is missing (pre-2014 seasons lack the rushing report
    # entirely, and FCS supplements can arrive without it). Prefer the real
    # count, fall back to the inferred one.
    if "pff_yards_after_contact" in grades.columns and "pff_yco_attempt" in grades.columns:
        yac      = grades["pff_yards_after_contact"]
        yco_att  = grades["pff_yco_attempt"].replace(0, np.nan)
        inferred = yac / yco_att
    else:
        inferred = pd.Series(np.nan, index=grades.index)
    # `pff_attempts` is report-scoped: rush attempts on RB rows, pass attempts
    # on QB rows, punt attempts on P rows. Only the RB reading is a rush count.
    if "pff_attempts" in grades.columns and "position_group" in grades.columns:
        rush_att = grades["pff_attempts"].where(grades["position_group"] == "RB")
        grades["pff_rush_att_inferred"] = rush_att.combine_first(inferred)
    else:
        grades["pff_rush_att_inferred"] = inferred
    grades["pff_avoided_tackles_per_attempt"] = rate(
        "pff_avoided_tackles", "pff_rush_att_inferred"
    )

    return grades


def _is_grade_col(col: str) -> bool:
    """A PFF column is a 'grade' (PFF analyst-smoothed) iff the name contains
    'grades'. Used to pick the lower shrinkage k-factor for grade columns —
    PFF grades already have implicit smoothing so they need less Bayesian
    pull toward the prior."""
    return "grades" in col


def _shrinkage_k_for(col: str) -> float:
    """Return the shrinkage prior strength k for one stat column.

    k = factor * PFF_REF_DENOM_BY_STAT[col], where factor depends on family:
      - grades → PFF_SHRINKAGE_K_FACTOR_GRADES (less shrinkage)
      - everything else → PFF_SHRINKAGE_K_FACTOR_RATE (more shrinkage)

    The base column for a denom-mapped stat may not itself be in
    `PFF_REF_DENOM_BY_STAT` (most rate columns aren't — e.g.
    `pff_sacks_per_pr_snap` isn't there but its denominator
    `pff_snap_counts_pass_rush` is). We look up the denominator first to
    inherit its reference scale, falling back to `PFF_REF_DENOM_DEFAULT`.
    """
    from config import (
        PFF_DENOM_COL_BY_STAT, PFF_REF_DENOM_BY_STAT, PFF_REF_DENOM_DEFAULT,
        PFF_SHRINKAGE_K_FACTOR_GRADES, PFF_SHRINKAGE_K_FACTOR_RATE,
    )
    denom_col = PFF_DENOM_COL_BY_STAT.get(col)
    ref = PFF_REF_DENOM_BY_STAT.get(denom_col, PFF_REF_DENOM_DEFAULT) \
        if denom_col else PFF_REF_DENOM_DEFAULT
    factor = PFF_SHRINKAGE_K_FACTOR_GRADES if _is_grade_col(col) \
        else PFF_SHRINKAGE_K_FACTOR_RATE
    return factor * float(ref)


def _is_shrunk_stat(col: str) -> bool:
    """A stat is eligible for Bayesian shrinkage iff it has a non-None
    denominator mapping in `PFF_DENOM_COL_BY_STAT`. Raw counts and the
    denominator-stats themselves (mapped to None) are pure volume signals
    and are NOT shrunk."""
    from config import PFF_DENOM_COL_BY_STAT
    if col not in PFF_DENOM_COL_BY_STAT:
        return False
    return PFF_DENOM_COL_BY_STAT[col] is not None


def _compute_position_priors(
    grades: pd.DataFrame,
    stat_cols: list[str],
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """Compute denominator-weighted pooled per-(pos_group, stat) priors.

        prior[pos_group, stat] = Σ(n_i × val_i) / Σ(n_i)

    summed across all (player, season) rows in the position group with
    non-NaN val_i AND n_i > 0. The fallback dict gives an un-stratified
    pooled mean per stat, used when a (pos_group, stat) combo has no rows
    (rare — e.g. a player-mode mismatch).

    Only stats eligible for shrinkage (per `_is_shrunk_stat`) get priors.
    """
    from config import PFF_DENOM_COL_BY_STAT

    pos_priors: dict[tuple[str, str], float] = {}
    fallback: dict[str, float] = {}

    if "position_group" not in grades.columns:
        # Without position info there's no pos-stratified prior to compute.
        # Fall through to global pooled means only.
        for col in stat_cols:
            if not _is_shrunk_stat(col):
                continue
            denom_col = PFF_DENOM_COL_BY_STAT[col]
            if denom_col not in grades.columns:
                continue
            obs = pd.to_numeric(grades[col], errors="coerce")
            n = pd.to_numeric(grades[denom_col], errors="coerce")
            mask = obs.notna() & n.notna() & (n > 0)
            if not mask.any():
                continue
            fallback[col] = float((obs[mask] * n[mask]).sum() / n[mask].sum())
        return pos_priors, fallback

    for col in stat_cols:
        if not _is_shrunk_stat(col):
            continue
        denom_col = PFF_DENOM_COL_BY_STAT[col]
        if denom_col not in grades.columns:
            continue

        obs = pd.to_numeric(grades[col], errors="coerce")
        n = pd.to_numeric(grades[denom_col], errors="coerce")
        valid = obs.notna() & n.notna() & (n > 0)
        if not valid.any():
            continue

        # Un-stratified fallback first (always populated when any rows are valid).
        fallback[col] = float((obs[valid] * n[valid]).sum() / n[valid].sum())

        # Per-(pos_group, stat) stratified prior.
        sub = pd.DataFrame({
            "pg":  grades["position_group"],
            "obs": obs,
            "n":   n,
        })[valid & grades["position_group"].notna()]
        if sub.empty:
            continue
        sub["wn"] = sub["obs"] * sub["n"]
        agg = sub.groupby("pg").agg(wn=("wn", "sum"), n=("n", "sum"))
        agg = agg[agg["n"] > 0]
        for pg, r in agg.iterrows():
            pos_priors[(str(pg), col)] = float(r["wn"] / r["n"])

    return pos_priors, fallback


def _resolve_prior(
    col: str,
    pos_group: str | float | None,
    pos_priors: dict[tuple[str, str], float],
    fallback: dict[str, float],
) -> float:
    """Pick the shrinkage prior for one (pos_group, stat) cell, falling
    back to the un-stratified pooled mean, then to 0.0 as a final safety."""
    if isinstance(pos_group, str) and pos_group:
        v = pos_priors.get((pos_group, col))
        if v is not None and np.isfinite(v):
            return v
    fv = fallback.get(col, 0.0)
    return fv if np.isfinite(fv) else 0.0


def load_pff_career_features(
    snapshot: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Load pff_season_grades.parquet and aggregate per player into career features.

    For each grade column: adds {col}_mean, {col}_peak, {col}_trough, {col}_last.
    Also adds pff_seasons (number of seasons with data).

    When config.USE_PFF_CONTEXTUAL is True and adjusted_pff_seasons.parquet
    exists, the modeled stats are overwritten with their adjusted_<stat>
    counterparts before aggregation, so the triplets describe the player's
    *residualized* performance (context-adjusted). Additionally, for each
    column in PFF_CONTEXT_TRIPLET_COLS, emits ctx_<col>_peak / _trough /
    _last / _wavg (snap-weighted) capturing the team/schedule context the
    player operated in across their CFB seasons.

    Season-level rate / delta features (e.g. per-snap event rates,
    man-vs-zone grade deltas) are synthesized before the aggregation so
    their mean/peak/last reflect true season-by-season rate variation.

    **Bayesian shrinkage** (2026-05-12, k=0.05/0.125): for stats with a
    denominator mapping in `config.PFF_DENOM_COL_BY_STAT`, each per-season
    observation is shrunk toward a position-pooled prior (denom-weighted
    pooled mean) before aggregation:

        shrunk = (n * obs + k * prior) / (n + k)

    where n is the relevant denominator (snaps/routes/attempts) and k =
    factor * PFF_REF_DENOM_BY_STAT[denom], with factor smaller for grade
    columns (0.05) and larger for rate / percent / delta stats (0.125). The
    `_mean` triplet becomes snap-weighted across seasons; `_peak` takes the
    max of the shrunk per-season values (post-shrinkage, so tiny-sample
    outliers can't dominate); `_last` uses the shrunk most-recent-season
    value.

    Pure volume stats (raw counts, denominator-stats-themselves) bypass
    shrinkage and keep the legacy unweighted triplets.

    Returns a DataFrame with player_id index, ready to merge onto snapshot.
    Missing players get NaN (no PFF data).
    """
    grades_path = processed_dir / "pff_season_grades.parquet"
    if not grades_path.exists():
        logger.warning("pff_season_grades.parquet not found — skipping PFF features")
        return pd.DataFrame(index=snapshot.get("player_id", pd.Index([])))

    grades = pd.read_parquet(grades_path)
    if "player_id" not in grades.columns:
        logger.warning("pff_season_grades.parquet missing player_id column")
        return pd.DataFrame()

    grades = grades[grades["player_id"].notna()]
    # Same (player_id, season) exclusions as the CFBD tables: a PFF season
    # attached to a CFBD id can be a namesake's (an FCS "transfer" after the
    # player was already in the NFL: Alex Ward, UCF -> Presbyterian 2025).
    from config import KNOWN_SEASON_EXCLUSIONS, RAW_DIR
    from src.utils.cfbd_id_reuse import exclusion_mask, merged_exclusions
    exclusions = merged_exclusions(KNOWN_SEASON_EXCLUSIONS, RAW_DIR, processed_dir, logger)
    if exclusions:
        bad = exclusion_mask(grades, exclusions)
        if bad.any():
            logger.info("Dropped %d PFF player-seasons matching CFBD id exclusions", int(bad.sum()))
            grades = grades[~bad]

    # Synthesize season-level rate / delta columns before aggregation so
    # the existing mean/peak/last loop picks them up automatically.
    # Synthesis must happen on raw counts/denominators — must NOT see
    # residualized stats — so it runs before the contextual swap below.
    grades = _add_season_level_rates(grades)

    # ── Context-residualized swap + team-context merge ────────────────────
    # When USE_PFF_CONTEXTUAL is on and step 05s has written
    # adjusted_pff_seasons.parquet, (a) overwrite each modeled stat with its
    # adjusted_<stat> counterpart (step 05s fills adjusted_<stat> with raw
    # values for R²-gated heads, so this is safe unconditionally) and
    # (b) merge PFF_CONTEXT_TRIPLET_COLS so the per-player triplet loop
    # below can emit ctx_<col>_peak / _trough / _last / _wavg.
    ctx_cols_present: list[str] = []
    try:
        from config import (
            USE_PFF_CONTEXTUAL, PFF_STATS_TO_MODEL, PFF_CONTEXT_TRIPLET_COLS,
        )
    except Exception:
        USE_PFF_CONTEXTUAL = False
        PFF_STATS_TO_MODEL = {}
        PFF_CONTEXT_TRIPLET_COLS = []

    adj_path = processed_dir / "adjusted_pff_seasons.parquet"
    if USE_PFF_CONTEXTUAL and adj_path.exists():
        modeled = {s for sts in PFF_STATS_TO_MODEL.values() for s in sts}
        adj_cols_wanted = [f"adjusted_{s}" for s in sorted(modeled)]
        ctx_cols_wanted = [c for c in PFF_CONTEXT_TRIPLET_COLS]
        join_keys = ["player_id", "season_year"]
        # Probe the parquet schema before requesting columns so we don't
        # error on missing names (cheaper than reading everything).
        try:
            import pyarrow.parquet as pq
            available = set(pq.ParquetFile(adj_path).schema_arrow.names)
        except Exception:
            available = set()
        select_cols = (join_keys
                       + [c for c in adj_cols_wanted if c in available]
                       + [c for c in ctx_cols_wanted if c in available])
        if all(k in grades.columns for k in join_keys) and len(select_cols) > len(join_keys):
            adj = pd.read_parquet(adj_path, columns=select_cols)
            grades = grades.merge(adj, on=join_keys, how="left")
            n_swapped = 0
            for stat in modeled:
                adj_col = f"adjusted_{stat}"
                if adj_col in grades.columns and stat in grades.columns:
                    grades[stat] = grades[adj_col].where(
                        grades[adj_col].notna(), grades[stat]
                    )
                    n_swapped += 1
            grades = grades.drop(
                columns=[c for c in grades.columns if c.startswith("adjusted_")]
            )
            ctx_cols_present = [c for c in ctx_cols_wanted if c in grades.columns]
            logger.info(
                "PFF career features: context-residualized "
                "(%d modeled stats swapped, %d context cols merged for triplets)",
                n_swapped, len(ctx_cols_present),
            )

    grade_cols = [c for c in grades.columns
                  if c.startswith("pff_") and c != "pff_player_id"
                  and grades[c].dtype != object]

    if not grade_cols:
        return pd.DataFrame()

    # Per-(pos_group, stat) priors used by the shrinkage step below.
    # Computed once over the full panel before the player loop.
    pos_priors, fallback = _compute_position_priors(grades, grade_cols)

    # Per-stat shrinkage metadata: precompute denom-column + k once per stat
    # so the inner loop is just lookups.
    from config import PFF_DENOM_COL_BY_STAT
    stat_meta: dict[str, tuple[str | None, float]] = {}
    for col in grade_cols:
        if _is_shrunk_stat(col):
            denom_col = PFF_DENOM_COL_BY_STAT.get(col)
            if denom_col in grades.columns:
                stat_meta[col] = (denom_col, _shrinkage_k_for(col))
            else:
                stat_meta[col] = (None, 0.0)  # denom missing → no shrinkage
        else:
            stat_meta[col] = (None, 0.0)

    has_pos = "position_group" in grades.columns

    # Coerce all grade columns to numeric arrays once (the original loop also
    # did this implicitly via pd.to_numeric per-stat per-player). We index by
    # sorted-row position below, so it's safe to operate on numpy arrays.
    grades_sorted = grades.sort_values(["player_id", "season_year"])
    col_values: dict[str, np.ndarray] = {
        col: pd.to_numeric(grades_sorted[col], errors="coerce").to_numpy(dtype=float)
        for col in grade_cols
    }
    denom_values: dict[str, np.ndarray] = {}
    for col in grade_cols:
        denom_col = stat_meta[col][0]
        if denom_col is not None and denom_col not in denom_values:
            # Keep NaN denoms as NaN — we drop those seasons from shrinkage
            # aggregation rather than imputing 0 (which would collapse the
            # season to the prior and bias the player's triplet).
            denom_values[denom_col] = pd.to_numeric(
                grades_sorted[denom_col], errors="coerce"
            ).to_numpy(dtype=float)

    # Also load any PFF_DENOM_STATS not already in denom_values, so we can
    # emit career-total reliability features per player below. Some denoms
    # (e.g. pff_total_attempts for K) aren't referenced by any shrunk stat
    # but are still useful as volume signals.
    from config import PFF_DENOM_STATS
    for denom_col in PFF_DENOM_STATS:
        if denom_col in denom_values:
            continue
        if denom_col in grades_sorted.columns:
            denom_values[denom_col] = pd.to_numeric(
                grades_sorted[denom_col], errors="coerce"
            ).to_numpy(dtype=float)

    pg_array = (grades_sorted["position_group"].to_numpy(dtype=object)
                if has_pos else np.array([None] * len(grades_sorted), dtype=object))
    player_ids = grades_sorted["player_id"].to_numpy()

    # Context-feature arrays (only populated when the merge above brought in
    # PFF_CONTEXT_TRIPLET_COLS). Snap weight for the wavg is offense + defense
    # snaps per row; NaN-as-0 so a row with all-NaN snaps falls back to
    # equal-weight in the loop.
    ctx_values: dict[str, np.ndarray] = {
        col: pd.to_numeric(grades_sorted[col], errors="coerce").to_numpy(dtype=float)
        for col in ctx_cols_present if col in grades_sorted.columns
    }
    if ctx_values:
        off_snaps = (pd.to_numeric(grades_sorted.get("pff_snap_counts_offense", 0.0),
                                    errors="coerce").to_numpy(dtype=float)
                     if "pff_snap_counts_offense" in grades_sorted.columns
                     else np.zeros(len(grades_sorted), dtype=float))
        def_snaps = (pd.to_numeric(grades_sorted.get("pff_snap_counts_defense", 0.0),
                                    errors="coerce").to_numpy(dtype=float)
                     if "pff_snap_counts_defense" in grades_sorted.columns
                     else np.zeros(len(grades_sorted), dtype=float))
        ctx_weight = np.nan_to_num(off_snaps, nan=0.0) + np.nan_to_num(def_snaps, nan=0.0)
    else:
        ctx_weight = np.zeros(len(grades_sorted), dtype=float)

    # Build contiguous (start, end) slices per player_id.
    boundaries = np.flatnonzero(player_ids[1:] != player_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(grades_sorted)]))

    agg_frames: list[dict] = []

    for s, e in zip(starts, ends):
        player_id = player_ids[s]
        n_seasons = e - s
        row: dict = {"player_id": player_id, "pff_seasons": int(n_seasons)}
        pg_slice = pg_array[s:e]

        for col in grade_cols:
            obs = col_values[col][s:e]
            mask = ~np.isnan(obs)

            if not mask.any():
                row[f"{col}_mean"] = np.nan
                row[f"{col}_peak"] = np.nan
                row[f"{col}_trough"] = np.nan
                row[f"{col}_last"] = np.nan
                continue

            denom_col, k = stat_meta[col]
            o_m = obs[mask]

            if denom_col is None:
                # Volume stat — legacy unweighted triplet.
                row[f"{col}_mean"] = float(o_m.mean())
                row[f"{col}_peak"] = float(o_m.max())
                row[f"{col}_trough"] = float(o_m.min())
                row[f"{col}_last"] = float(o_m[-1])
                continue

            # Shrinkage path. A season counts only when the grade is observed
            # AND the relevant denominator is observed and positive — we can't
            # weight a grade we don't know the context of.
            n_slice = denom_values[denom_col][s:e]
            shrink_mask = mask & ~np.isnan(n_slice) & (n_slice > 0)

            if not shrink_mask.any():
                row[f"{col}_mean"] = np.nan
                row[f"{col}_peak"] = np.nan
                row[f"{col}_trough"] = np.nan
                row[f"{col}_last"] = np.nan
                continue

            o_m = obs[shrink_mask]
            n_m = n_slice[shrink_mask]
            pg_m = pg_slice[shrink_mask]

            # Resolve prior per unique pos_group across this player's seasons.
            seen_priors: dict = {}
            prior_arr = np.empty(len(o_m), dtype=float)
            for i, pg in enumerate(pg_m):
                key = pg if isinstance(pg, str) else "__none__"
                if key not in seen_priors:
                    seen_priors[key] = _resolve_prior(
                        col, pg, pos_priors, fallback,
                    )
                prior_arr[i] = seen_priors[key]

            shrunk = (n_m * o_m + k * prior_arr) / (n_m + k)

            row[f"{col}_mean"] = float((shrunk * n_m).sum() / n_m.sum())
            row[f"{col}_peak"] = float(shrunk.max())
            row[f"{col}_trough"] = float(shrunk.min())
            row[f"{col}_last"] = float(shrunk[-1])

        # Career-total reliability features. For each denominator stat, sum
        # across this player's seasons. NaN if the column never has data for
        # this player; 0 if the column is present but all zero. LightGBM
        # routes NaN via missing-direction splits, so a kicker who never
        # ran a route gets pff_routes_total = NaN, not 0.
        for denom_col in PFF_DENOM_STATS:
            if denom_col not in denom_values:
                row[f"{denom_col}_total"] = np.nan
                continue
            n_slice = denom_values[denom_col][s:e]
            if np.all(np.isnan(n_slice)):
                row[f"{denom_col}_total"] = np.nan
            else:
                row[f"{denom_col}_total"] = float(np.nansum(n_slice))

        # Team-context triplets. For each PFF_CONTEXT_TRIPLET_COLS column,
        # emit ctx_<col>_peak / _trough / _last / _wavg across the player's
        # seasons. Residualization scrubs context out of the per-season
        # grade; these triplets re-introduce the context as separate
        # features so the draft model can weight the *environment* the
        # player operated in alongside the player-specific residual.
        # _wavg is snap-weighted (offense + defense PFF snaps) so high-
        # snap seasons dominate; falls back to unweighted mean when all
        # snap counts are zero / NaN for the player.
        if ctx_values:
            w_slice = ctx_weight[s:e]
            w_sum = float(w_slice.sum())
            for col, arr in ctx_values.items():
                obs = arr[s:e]
                mask = ~np.isnan(obs)
                if not mask.any():
                    row[f"ctx_{col}_peak"]   = np.nan
                    row[f"ctx_{col}_trough"] = np.nan
                    row[f"ctx_{col}_last"]   = np.nan
                    row[f"ctx_{col}_wavg"]   = np.nan
                    continue
                o_m = obs[mask]
                row[f"ctx_{col}_peak"]   = float(o_m.max())
                row[f"ctx_{col}_trough"] = float(o_m.min())
                row[f"ctx_{col}_last"]   = float(o_m[-1])
                if w_sum > 0:
                    w_m = w_slice[mask]
                    if w_m.sum() > 0:
                        row[f"ctx_{col}_wavg"] = float((o_m * w_m).sum() / w_m.sum())
                    else:
                        row[f"ctx_{col}_wavg"] = float(o_m.mean())
                else:
                    row[f"ctx_{col}_wavg"] = float(o_m.mean())

        agg_frames.append(row)

    if not agg_frames:
        return pd.DataFrame()

    career_df = pd.DataFrame(agg_frames)

    # Compute most common PFF position per player (mode)
    if "pff_position" in grades.columns:
        pos_mode = (grades[grades["pff_position"].notna()]
                    .groupby("player_id")["pff_position"]
                    .agg(lambda s: s.mode().iloc[0] if len(s) > 0 else np.nan)
                    .reset_index()
                    .rename(columns={"pff_position": "pff_position_mode"}))
        career_df = career_df.merge(pos_mode, on="player_id", how="left")

    logger.info("PFF career features: %d players, %d grade columns "
                "(%d shrunk, %d volume)",
                len(career_df), len(grade_cols),
                sum(1 for c in grade_cols if stat_meta[c][0] is not None),
                sum(1 for c in grade_cols if stat_meta[c][0] is None))
    return career_df
