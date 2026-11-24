"""
Phase 5 (gated by --deep): model behavior audit.

  - For each draft_models/*.pkl: load bundle, list feature_cols whose
    referenced column is absent from player_skill_snapshot.parquet ∪
    player_attributes.parquet (catches "model expects column X but step 6/3
    no longer emits it" regressions).
  - For each contextual_models/*.pkl: same, against gamelogs schema.
  - Lightweight CV refit comparison is OUT OF SCOPE for v1 (would require
    re-deriving the training-data assembly logic from step 8 — too coupled
    to the live pipeline). Surfaces as a known follow-up.
"""

from __future__ import annotations

import csv
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import DERIVED_AT_INFERENCE, PROCESSED_DIR  # noqa: E402

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent
MODELS_CSV = OUT_DIR / "model_features.csv"


@dataclass
class ModelRow:
    bundle: str
    n_features: int
    n_missing: int
    missing_examples: str
    severity: str = "OK"


def _snapshot_columns() -> set[str]:
    p = PROCESSED_DIR / "player_skill_snapshot.parquet"
    if not p.exists():
        return set()
    return set(pd.read_parquet(p, columns=None).columns)


def _attribute_columns() -> set[str]:
    p = PROCESSED_DIR / "player_attributes.parquet"
    if not p.exists():
        return set()
    return set(pd.read_parquet(p, columns=None).columns)


def _gamelog_columns() -> set[str]:
    """Columns in adjusted_gamelogs.parquet — the input shape contextual
    models score against."""
    p = PROCESSED_DIR / "adjusted_gamelogs.parquet"
    if not p.exists():
        return set()
    return set(pd.read_parquet(p, columns=None).columns)


def _check_bundle(path: Path, available: set[str]) -> ModelRow:
    """Compare bundle.feature_cols against `available` ∪ DERIVED_AT_INFERENCE
    ∪ {pff_*_mean,_peak,_last} → the engineered families load_pff_career_features
    adds at scoring time. Anything not in any of those is a real gap."""
    try:
        with path.open("rb") as f:
            bundle = pickle.load(f)
    except Exception as exc:
        return ModelRow(path.name, 0, 0, f"unloadable: {exc}", severity="ERROR")
    # Bundles come in two shapes: dict (draft models) and class instance
    # (ContextualAdjustmentModel). Try dict access, fall back to attr.
    if isinstance(bundle, dict):
        feats = list(bundle.get("feature_cols", []))
    else:
        feats = list(getattr(bundle, "feature_cols", []) or [])
    if not feats:
        return ModelRow(path.name, 0, 0, "no feature_cols in bundle",
                         severity="WARNING")

    # PFF aggregated columns are added at runtime by load_pff_career_features.
    # Recognize them by suffix so we don't false-positive every pff_*_mean.
    def _is_pff_aggregated(c: str) -> bool:
        return c.startswith("pff_") and c.endswith(("_mean", "_peak", "_last"))

    # Runtime-added columns: the aggregated PFF triplets + pff_seasons (from
    # load_pff_career_features), plus position_group_ordinal (mapped from
    # position_group via config.POSITION_GROUP_ORDINAL inside step 8b).
    runtime_pff = {"pff_seasons", "position_group_ordinal"}
    missing = [
        c for c in feats
        if c not in available
        and c not in DERIVED_AT_INFERENCE
        and c not in runtime_pff
        and not _is_pff_aggregated(c)
        and not c.startswith("combine_")    # combine_* merged at step 10 runtime
        and c not in {"hasan_rank", "hasan_draft_year"}   # merged at step 10 runtime
        and not c.startswith("nfl_lg_")     # merged at step 10 runtime
    ]
    severity = "OK"
    if missing:
        severity = "WARNING" if len(missing) <= 3 else "ERROR"
    return ModelRow(
        bundle=path.name,
        n_features=len(feats),
        n_missing=len(missing),
        missing_examples=", ".join(missing[:5]) + ("…" if len(missing) > 5 else ""),
        severity=severity,
    )


def collect() -> list[ModelRow]:
    snap_cols = _snapshot_columns()
    attr_cols = _attribute_columns()
    gamelog_cols = _gamelog_columns()
    available_for_draft = snap_cols | attr_cols
    available_for_ctx = gamelog_cols | attr_cols

    rows: list[ModelRow] = []
    for p in sorted((PROCESSED_DIR / "draft_models").glob("*.pkl")):
        rows.append(_check_bundle(p, available_for_draft))
    for p in sorted((PROCESSED_DIR / "global_draft_models").glob("*.pkl")):
        rows.append(_check_bundle(p, available_for_draft))
    for p in sorted((PROCESSED_DIR / "contextual_models").glob("*.pkl")):
        rows.append(_check_bundle(p, available_for_ctx))
    return rows


def write_csv(rows: list[ModelRow]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MODELS_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bundle", "n_features", "n_missing",
                    "missing_examples", "severity"])
        for r in rows:
            w.writerow([r.bundle, r.n_features, r.n_missing,
                        r.missing_examples, r.severity])


def build_section(rows: list[ModelRow]) -> str:
    lines = ["## Phase 5 — Model Behavior", ""]
    bad = [r for r in rows if r.severity != "OK"]
    if bad:
        lines.append("**Issues found:**\n")
        for r in bad:
            lines.append(f"- ({r.severity}) `{r.bundle}` — "
                         f"{r.n_missing}/{r.n_features} feature_cols absent "
                         f"from input data: {r.missing_examples}")
        lines.append("")
    else:
        lines += ["No feature-availability issues across model bundles.", ""]
    lines += ["| bundle | n_features | n_missing | examples | severity |",
              "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| `{r.bundle}` | {r.n_features} | {r.n_missing} | "
                     f"`{r.missing_examples}` | {r.severity} |")
    lines += ["",
              "_Note: CV refit comparison against `training_metrics.json` is "
              "deferred — needs the step-8 training-data assembly logic to be "
              "factored out of `pipeline/08_draft_models.py` first._"]
    return "\n".join(lines)


def has_errors(rows: list[ModelRow]) -> bool:
    return any(r.severity == "ERROR" for r in rows)


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
