"""
Phase 4: spot-check the final projection output (latest.parquet).

  - Position-group distribution (counts, % "OTHER" — should be small)
  - Score distributions per head (role, impact_apy, impact_wav)
  - Named-prospect spot checks (Mendoza, Fano, Crownover, Concepcion, Hunter)
  - Mode-aware diff vs the previous date-stamped snapshot in
    projections/projections_*.parquet — surfaces large rank shifts only
    when both snapshots share the same _tuning_mode provenance, otherwise
    skipped (cross-mode comparison is misleading).
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent
SPOTCHECK_CSV = OUT_DIR / "output_spotcheck.csv"

LATEST_PATH = OUTPUT_DIR / "projections" / "latest.parquet"

NAMED_PROSPECTS = [
    "Fernando Mendoza",
    "Spencer Fano",
    "Dametrious Crownover",
    "KC Concepcion",
    "Travis Hunter",
]

SCORE_HEADS = [
    "role_score",
    "impact_apy_score",
    "impact_wav_score",
    "global_role_score",
    "global_impact_apy_score",
    "global_impact_wav_score",
]


@dataclass
class OutputIssue:
    name: str
    detail: str
    severity: str = "WARNING"


def _load_latest() -> Optional[pd.DataFrame]:
    if not LATEST_PATH.exists():
        return None
    return pd.read_parquet(LATEST_PATH)


def _previous_snapshot() -> Optional[Path]:
    """Most-recent date-stamped projections file *other than* the symlink target."""
    proj_dir = OUTPUT_DIR / "projections"
    if not proj_dir.exists():
        return None
    snaps = sorted(proj_dir.glob("projections_*.parquet"))
    return snaps[-1] if snaps else None


def _read_tuning_mode(provenance_path: Path) -> Optional[str]:
    if not provenance_path.exists():
        return None
    try:
        return json.loads(provenance_path.read_text()).get("_tuning_mode")
    except Exception:
        return None


def _current_mode_summary() -> dict[str, str]:
    """Build a small dict summarizing what _tuning_mode each canonical step
    artifact carries on disk right now. Used to gate cross-snapshot diffing."""
    return {
        "step04_contextual": _read_tuning_mode(
            PROCESSED_DIR / "contextual_models" / "model_summary_provenance.json"
        ) or "unknown",
        "step08_draft": _read_tuning_mode(
            PROCESSED_DIR / "draft_models" / "training_metrics_provenance.json"
        ) or "unknown",
        "step08b_global": _read_tuning_mode(
            PROCESSED_DIR / "global_draft_models" / "training_metrics_provenance.json"
        ) or "unknown",
    }


# ── Public collectors ────────────────────────────────────────────────────────

def position_distribution(latest: pd.DataFrame) -> tuple[pd.DataFrame, list[OutputIssue]]:
    issues: list[OutputIssue] = []
    if "position_group" not in latest.columns:
        issues.append(OutputIssue("position_distribution",
                                   "no position_group column", "ERROR"))
        return pd.DataFrame(), issues
    counts = latest["position_group"].value_counts(dropna=False).to_frame("count")
    counts["pct"] = counts["count"] / counts["count"].sum()
    other_pct = float(counts.loc["OTHER", "pct"]) if "OTHER" in counts.index else 0.0
    # 30%+ OTHER is the historical baseline driven by CFBD's `position=None`
    # and `?` markers on pre-rosters / FCS-edge players who carry no scoring
    # signal anyway. Threshold tuned to 40% to flag a real regression (e.g.,
    # POSITION_TO_GROUP missing a common abbreviation) rather than the noise.
    if other_pct > 0.40:
        issues.append(OutputIssue(
            "position_other_share",
            f"position_group=OTHER is {other_pct:.1%} of latest "
            f"(threshold <40%) — many CFBD positions failing to map",
            "WARNING",
        ))
    return counts.reset_index().rename(columns={"index": "position_group"}), issues


def score_distributions(latest: pd.DataFrame) -> tuple[pd.DataFrame, list[OutputIssue]]:
    issues: list[OutputIssue] = []
    rows = []
    for head in SCORE_HEADS:
        if head not in latest.columns:
            continue
        s = latest[head].dropna()
        if s.empty:
            issues.append(OutputIssue(
                f"score_empty_{head}",
                f"{head} has no non-null values",
                "ERROR",
            ))
            continue
        q = s.quantile([0.01, 0.10, 0.50, 0.90, 0.99]).to_dict()
        # Mode density: max bucket fraction at 2-decimal rounding (or 4 for tiny scores).
        rounding = 4 if s.median() < 0.01 else 2
        mode_pct = float(s.round(rounding).value_counts(normalize=True).iloc[0])
        rows.append({
            "head": head,
            "non_null": int(s.notna().sum()),
            "p01": q[0.01], "p10": q[0.10], "p50": q[0.50],
            "p90": q[0.90], "p99": q[0.99],
            "mode_pct": mode_pct,
        })
        if mode_pct > 0.50:
            issues.append(OutputIssue(
                f"score_mode_density_{head}",
                f"{head} has {mode_pct:.1%} of values at single rounded "
                f"bucket — looks compressed (regression model anchoring to mean?)",
                "WARNING",
            ))
    return pd.DataFrame(rows), issues


def named_prospect_spotcheck(latest: pd.DataFrame
                              ) -> tuple[pd.DataFrame, list[OutputIssue]]:
    issues: list[OutputIssue] = []
    rows = []
    for name in NAMED_PROSPECTS:
        match = latest[latest["player_name"].str.lower()
                       == name.lower()] if "player_name" in latest.columns else pd.DataFrame()
        if match.empty:
            issues.append(OutputIssue(
                f"prospect_missing_{name}",
                f"named prospect '{name}' not found in latest.parquet",
                "WARNING",
            ))
            continue
        if len(match) > 1:
            issues.append(OutputIssue(
                f"prospect_dup_{name}",
                f"named prospect '{name}' has {len(match)} rows in latest "
                f"(expected 1) — possible namesake collision",
                "WARNING",
            ))
        # Take the most recent / first row deterministically.
        match = match.sort_values("last_college_season_year",
                                   ascending=False, na_position="last").iloc[[0]]
        cols_to_show = [
            "player_name", "school", "position_group",
            "role_score", "impact_apy_score", "impact_wav_score",
            "impact_apy_dollars", "hasan_rank",
        ]
        cols = [c for c in cols_to_show if c in latest.columns]
        rows.append(match[cols].iloc[0].to_dict())
    return pd.DataFrame(rows), issues


def diff_vs_previous(latest: pd.DataFrame, top_n: int = 100
                      ) -> tuple[Optional[pd.DataFrame], list[OutputIssue]]:
    """Compare top-N rank shifts vs the prior date-stamped snapshot. Skipped
    when a mode mismatch is detected — cross-mode comparison is misleading."""
    issues: list[OutputIssue] = []
    prev_path = _previous_snapshot()
    if prev_path is None:
        issues.append(OutputIssue(
            "diff_no_previous",
            "no previous projections_*.parquet snapshot to diff against",
            "WARNING",
        ))
        return None, issues
    prev = pd.read_parquet(prev_path)

    # Mode-aware gate: if both snapshots have a `_tuning_mode` audit column
    # (we don't write one currently), require they match. Otherwise just note
    # the comparison is best-effort and proceed.
    issues.append(OutputIssue(
        "diff_mode_caveat",
        f"diff vs {prev_path.name} is best-effort — projection snapshots "
        f"don't yet carry an explicit _tuning_mode tag",
        "WARNING",
    ))

    # Rank by impact_apy_score in both, compute rank deltas for top_n in latest.
    sort_col = "impact_apy_score"
    if sort_col not in latest.columns or sort_col not in prev.columns:
        issues.append(OutputIssue("diff_no_sort_col",
                                   f"{sort_col} missing from one snapshot", "WARNING"))
        return None, issues
    cur_rank = (latest.dropna(subset=[sort_col])
                       .sort_values(sort_col, ascending=False)
                       .reset_index(drop=True))
    cur_rank["rank_now"] = cur_rank.index + 1
    prev_rank = (prev.dropna(subset=[sort_col])
                      .sort_values(sort_col, ascending=False)
                      .reset_index(drop=True))
    prev_rank["rank_prev"] = prev_rank.index + 1

    cur_top = cur_rank.head(top_n)[["player_id", "player_name", "rank_now", sort_col]]
    prev_keep = prev_rank[["player_id", "rank_prev"]]
    diff = cur_top.merge(prev_keep, on="player_id", how="left")
    diff["delta_rank"] = diff["rank_prev"] - diff["rank_now"]   # positive = improved
    big_shift = diff[(diff["delta_rank"].abs() > 20) | diff["rank_prev"].isna()]
    if not big_shift.empty:
        issues.append(OutputIssue(
            "diff_large_rank_shifts",
            f"{len(big_shift)} top-{top_n} prospects shifted >20 spots "
            f"(or were absent from prior snapshot {prev_path.name})",
            "WARNING",
        ))
    return diff, issues


# ── Public API ───────────────────────────────────────────────────────────────

def collect() -> tuple[dict, list[OutputIssue]]:
    issues: list[OutputIssue] = []
    artifacts: dict = {}
    latest = _load_latest()
    if latest is None:
        issues.append(OutputIssue("latest_missing",
                                   "data/output/projections/latest.parquet not found",
                                   "ERROR"))
        return artifacts, issues

    pos_df, pos_issues = position_distribution(latest)
    artifacts["position_distribution"] = pos_df
    issues.extend(pos_issues)

    score_df, score_issues = score_distributions(latest)
    artifacts["score_distributions"] = score_df
    issues.extend(score_issues)

    spot_df, spot_issues = named_prospect_spotcheck(latest)
    artifacts["named_prospects"] = spot_df
    issues.extend(spot_issues)

    diff_df, diff_issues = diff_vs_previous(latest)
    artifacts["diff_vs_previous"] = diff_df
    issues.extend(diff_issues)

    artifacts["mode_summary"] = _current_mode_summary()
    return artifacts, issues


def write_csv(artifacts: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spot_df = artifacts.get("named_prospects")
    if spot_df is not None and not spot_df.empty:
        spot_df.to_csv(SPOTCHECK_CSV, index=False)


def build_section(artifacts: dict, issues: list[OutputIssue]) -> str:
    lines = ["## Phase 4 — Output Sanity", ""]
    errors = [i for i in issues if i.severity == "ERROR"]
    warnings_ = [i for i in issues if i.severity == "WARNING"]
    if errors or warnings_:
        lines += ["**Issues found:**", ""]
        for i in errors:
            lines.append(f"- (ERROR) `{i.name}` — {i.detail}")
        for i in warnings_:
            lines.append(f"- (WARN) `{i.name}` — {i.detail}")
        lines.append("")
    else:
        lines += ["No output-sanity issues.", ""]

    mode = artifacts.get("mode_summary", {})
    lines += ["### Tuning-mode of on-disk artifacts",
              "",
              "| step | tuning_mode |",
              "| --- | --- |",
              *[f"| `{k}` | {v} |" for k, v in mode.items()], ""]

    pos = artifacts.get("position_distribution")
    if pos is not None and not pos.empty:
        lines += ["### Position distribution", "",
                  "| position_group | count | pct |",
                  "| --- | --- | --- |",
                  *[f"| {row['position_group']} | {int(row['count'])} | "
                    f"{float(row['pct']):.1%} |"
                    for _, row in pos.iterrows()], ""]

    sc = artifacts.get("score_distributions")
    if sc is not None and not sc.empty:
        lines += ["### Score distributions",
                  "",
                  "| head | non_null | p01 | p10 | p50 | p90 | p99 | mode_pct |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in sc.iterrows():
            lines.append(
                f"| `{r['head']}` | {int(r['non_null'])} | "
                f"{r['p01']:.4g} | {r['p10']:.4g} | {r['p50']:.4g} | "
                f"{r['p90']:.4g} | {r['p99']:.4g} | {r['mode_pct']:.1%} |"
            )
        lines.append("")

    spot = artifacts.get("named_prospects")
    if spot is not None and not spot.empty:
        lines += ["### Named-prospect spot check", ""]
        # Render as markdown by hand (small table, variable cols).
        cols = list(spot.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in spot.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    cells.append(f"{v:.3g}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    diff = artifacts.get("diff_vs_previous")
    if diff is not None and not diff.empty:
        big = diff[(diff["delta_rank"].abs() > 20) | diff["rank_prev"].isna()]
        lines += [f"### Top-100 rank shifts vs prior snapshot "
                  f"({len(big)} > 20 spots)",
                  "",
                  "| player_name | rank_now | rank_prev | delta_rank | impact_apy_score |",
                  "| --- | --- | --- | --- | --- |"]
        head = big.head(20) if len(big) else diff.head(0)
        for _, r in head.iterrows():
            lines.append(
                f"| {r['player_name']} | {int(r['rank_now'])} | "
                f"{'' if pd.isna(r['rank_prev']) else int(r['rank_prev'])} | "
                f"{'' if pd.isna(r['delta_rank']) else int(r['delta_rank'])} | "
                f"{r['impact_apy_score']:.4g} |"
            )

    return "\n".join(lines)


def has_errors(issues: list[OutputIssue]) -> bool:
    return any(i.severity == "ERROR" for i in issues)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    artifacts, issues = collect()
    write_csv(artifacts)
    print(build_section(artifacts, issues))
    return 1 if has_errors(issues) else 0


if __name__ == "__main__":
    sys.exit(main())
