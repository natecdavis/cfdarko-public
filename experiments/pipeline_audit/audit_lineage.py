"""
Phase 2: row-count + cohort lineage assertions across adjacent step pairs.

Checks every step's output is consistent with its inputs:
  - bridge_players ⊆ union(player_seasons, rosters, gamelogs)
  - player_attributes covers every snapshot pid (or noted gap)
  - skill_estimates ↔ adjusted_gamelogs (one row per gamelog row, +/- skeleton)
  - snapshot rows == one per player
  - bridge_players ↔ nfl_draft_labels[is_bridge==True] size match
  - latest.parquet rows == projection cohort

Plus the two roster-only-fallback regression guards:
  - roster-only fallback: count of pids in rosters but not in player_seasons
    that DO appear in player_attributes — fail on 0 (means the steps 02/03/06
    fallback regressed to a no-op).
  - n_seasons recomputation: for a 100-row sample, recompute n_seasons over
    union(rosters, player_seasons, gamelogs) and compare against the persisted
    value in player_skill_snapshot.parquet.
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent
LINEAGE_CSV = OUT_DIR / "lineage.csv"


@dataclass
class LineageRow:
    check: str
    passed: bool
    detail: str
    severity: str = "ERROR"   # "ERROR" | "WARNING"


def _pids(path: Path, col: str = "player_id") -> set:
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=[col])[col].dropna().unique())


def _safe_load(path: Path, columns: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=columns)


# ── Checks ───────────────────────────────────────────────────────────────────

def check_bridge_pid_resolution() -> LineageRow:
    """Bridge players with non-null player_id should be findable in
    union(player_seasons, rosters)."""
    bridge = _safe_load(PROCESSED_DIR / "bridge_players.parquet",
                         columns=["player_id"])
    if bridge is None:
        return LineageRow("bridge_pid_resolution", False,
                           "bridge_players.parquet missing")
    bridge_pids = set(bridge["player_id"].dropna().unique())
    pseasons_pids = _pids(RAW_DIR / "player_seasons.parquet")
    rosters_pids = _pids(RAW_DIR / "rosters.parquet")
    universe = pseasons_pids | rosters_pids
    unresolved = bridge_pids - universe
    return LineageRow(
        "bridge_pid_resolution",
        len(unresolved) == 0,
        f"{len(bridge_pids)} bridge pids, "
        f"{len(unresolved)} not in union(player_seasons, rosters)",
        severity="ERROR" if len(unresolved) > 50 else "WARNING",
    )


def check_attributes_covers_snapshot() -> LineageRow:
    """Every snapshot pid should have an attributes row (post roster-only fix)."""
    snap = _safe_load(PROCESSED_DIR / "player_skill_snapshot.parquet",
                       columns=["player_id"])
    attrs = _safe_load(PROCESSED_DIR / "player_attributes.parquet",
                        columns=["player_id"])
    if snap is None or attrs is None:
        return LineageRow("attributes_covers_snapshot", False,
                           "snapshot or attributes missing")
    snap_pids = set(snap["player_id"].dropna().unique())
    attr_pids = set(attrs["player_id"].dropna().unique())
    gap = snap_pids - attr_pids
    return LineageRow(
        "attributes_covers_snapshot",
        len(gap) == 0,
        f"snapshot has {len(snap_pids)} pids, attributes has {len(attr_pids)}; "
        f"{len(gap)} snapshot pids absent from attributes",
        severity="ERROR" if len(gap) > 100 else "WARNING",
    )


def check_skill_estimates_match_gamelogs() -> LineageRow:
    """skill_estimates row count should match adjusted_gamelogs (one estimate
    per game). Allow small slack for skeleton rows added in step 6."""
    est_path = PROCESSED_DIR / "skill_estimates.parquet"
    adj_path = PROCESSED_DIR / "adjusted_gamelogs.parquet"
    if not est_path.exists() or not adj_path.exists():
        return LineageRow("skill_estimates_match_gamelogs", False,
                           "skill_estimates or adjusted_gamelogs missing")
    n_est = pd.read_parquet(est_path, columns=["player_id"]).shape[0]
    n_adj = pd.read_parquet(adj_path, columns=["player_id"]).shape[0]
    diff = n_est - n_adj
    return LineageRow(
        "skill_estimates_match_gamelogs",
        abs(diff) < max(500, n_adj * 0.001),
        f"adjusted_gamelogs rows={n_adj:,}, skill_estimates rows={n_est:,}, "
        f"diff={diff:+,}",
    )


def check_snapshot_unique_pids() -> LineageRow:
    """player_skill_snapshot.parquet must be one row per player_id."""
    snap = _safe_load(PROCESSED_DIR / "player_skill_snapshot.parquet",
                       columns=["player_id"])
    if snap is None:
        return LineageRow("snapshot_unique_pids", False,
                           "snapshot missing")
    n_dup = snap["player_id"].duplicated().sum()
    return LineageRow(
        "snapshot_unique_pids",
        n_dup == 0,
        f"snapshot has {len(snap):,} rows, {n_dup} duplicate player_ids",
    )


def check_bridge_in_labels() -> LineageRow:
    """Every bridge pid should appear in nfl_draft_labels[is_bridge]."""
    bridge = _safe_load(PROCESSED_DIR / "bridge_players.parquet",
                         columns=["player_id"])
    labels_path = PROCESSED_DIR / "nfl_draft_labels.parquet"
    if bridge is None or not labels_path.exists():
        return LineageRow("bridge_in_labels", False,
                           "bridge_players or nfl_draft_labels missing")
    labels = pd.read_parquet(labels_path, columns=["player_id", "is_bridge"])
    bridge_pids = set(bridge["player_id"].dropna().unique())
    labeled_bridge_pids = set(
        labels.loc[labels["is_bridge"] == True, "player_id"].dropna().unique()
    )
    gap = bridge_pids - labeled_bridge_pids
    return LineageRow(
        "bridge_in_labels",
        len(gap) < 50,
        f"bridge has {len(bridge_pids):,} pids, "
        f"labels[is_bridge] has {len(labeled_bridge_pids):,}; "
        f"{len(gap):,} bridge pids missing label rows",
        severity="ERROR" if len(gap) > 200 else "WARNING",
    )


def check_latest_cohort() -> LineageRow:
    """latest.parquet should be a non-trivial subset of snapshot."""
    latest_path = OUTPUT_DIR / "projections" / "latest.parquet"
    snap_path = PROCESSED_DIR / "player_skill_snapshot.parquet"
    if not latest_path.exists() or not snap_path.exists():
        return LineageRow("latest_cohort", False,
                           "latest.parquet or snapshot missing")
    latest = pd.read_parquet(latest_path, columns=["player_id"])
    snap_pids = _pids(snap_path)
    latest_pids = set(latest["player_id"].dropna().unique())
    leak = latest_pids - snap_pids
    return LineageRow(
        "latest_cohort",
        len(leak) == 0 and len(latest_pids) > 0,
        f"latest has {len(latest_pids):,} pids, "
        f"{len(leak):,} not in snapshot",
    )


def check_roster_only_fallback() -> LineageRow:
    """Pids in rosters but NOT player_seasons must appear in attributes.
    Failure here = the steps 02/03/06 roster-only fallback regressed."""
    rosters_pids = _pids(RAW_DIR / "rosters.parquet")
    pseasons_pids = _pids(RAW_DIR / "player_seasons.parquet")
    attr_pids = _pids(PROCESSED_DIR / "player_attributes.parquet")
    roster_only = rosters_pids - pseasons_pids
    landed = roster_only & attr_pids
    return LineageRow(
        "roster_only_fallback",
        len(landed) > 1_000,
        f"{len(roster_only):,} pids in rosters but not player_seasons; "
        f"{len(landed):,} of those reached player_attributes "
        f"(must be >1,000 for fallback to be working)",
    )


def check_n_seasons_recompute(sample_size: int = 100) -> LineageRow:
    """Recompute n_seasons over union(rosters, player_seasons, gamelogs) for
    a random sample of snapshot pids and compare against persisted value."""
    snap_path = PROCESSED_DIR / "player_skill_snapshot.parquet"
    if not snap_path.exists():
        return LineageRow("n_seasons_recompute", False, "snapshot missing")
    snap = pd.read_parquet(snap_path, columns=["player_id", "n_seasons"])
    if "n_seasons" not in snap.columns:
        return LineageRow("n_seasons_recompute", False,
                           "snapshot has no n_seasons column")
    sample = snap.sample(min(sample_size, len(snap)), random_state=42)
    sample_pids = set(sample["player_id"].dropna().unique())

    parts: list[pd.DataFrame] = []
    for f in ("rosters.parquet", "player_seasons.parquet", "player_gamelogs.parquet"):
        p = RAW_DIR / f
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["player_id", "season_year"])
        df = df[df["player_id"].isin(sample_pids)]
        parts.append(df)
    if not parts:
        return LineageRow("n_seasons_recompute", False,
                           "no upstream parquet had usable data")
    combined = pd.concat(parts, ignore_index=True)
    recomputed = (combined.dropna(subset=["player_id", "season_year"])
                          .groupby("player_id")["season_year"].nunique())

    cmp = sample.merge(
        recomputed.rename("n_seasons_recomputed").reset_index(),
        on="player_id", how="left",
    )
    cmp["n_seasons_recomputed"] = cmp["n_seasons_recomputed"].fillna(0).astype(int)
    mismatches = cmp[cmp["n_seasons"] != cmp["n_seasons_recomputed"]]
    return LineageRow(
        "n_seasons_recompute",
        len(mismatches) == 0,
        f"sample={len(cmp)}, "
        f"mismatches={len(mismatches)} "
        f"(expected 0 — n_seasons should be union over rosters/player_seasons/gamelogs)",
        severity="ERROR" if len(mismatches) > 5 else "WARNING",
    )


# ── Public API ───────────────────────────────────────────────────────────────

CHECKS = [
    check_bridge_pid_resolution,
    check_attributes_covers_snapshot,
    check_skill_estimates_match_gamelogs,
    check_snapshot_unique_pids,
    check_bridge_in_labels,
    check_latest_cohort,
    check_roster_only_fallback,
    check_n_seasons_recompute,
]


def collect() -> list[LineageRow]:
    return [c() for c in CHECKS]


def write_csv(rows: list[LineageRow]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LINEAGE_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "passed", "severity", "detail"])
        for r in rows:
            w.writerow([r.check, r.passed, r.severity, r.detail])


def build_section(rows: list[LineageRow]) -> str:
    lines = ["## Phase 2 — Lineage", ""]
    failures = [r for r in rows if not r.passed]
    if failures:
        lines.append("**Issues found:**\n")
        for r in failures:
            lines.append(f"- `{r.check}` ({r.severity}) — {r.detail}")
        lines.append("")
    else:
        lines.append("All lineage checks passed.\n")
    lines += ["| check | passed | severity | detail |",
              "| --- | --- | --- | --- |"]
    for r in rows:
        flag = "✓" if r.passed else "✗"
        lines.append(f"| `{r.check}` | {flag} | {r.severity} | {r.detail} |")
    return "\n".join(lines)


def has_errors(rows: list[LineageRow]) -> bool:
    return any((not r.passed) and r.severity == "ERROR" for r in rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    rows = collect()
    write_csv(rows)
    print(build_section(rows))
    return 1 if has_errors(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
