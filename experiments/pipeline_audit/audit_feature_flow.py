"""
Phase 3: trace engineered PFF rate features and attribute merge columns
through every layer:
  raw PFF columns       → pff_season_grades.parquet
  engineered rate base  → step 8's load_pff_career_features (transient)
  rate {base}_{mean|peak|last} → trained-model feature_cols (.pkl)
  career_team_* + ATTRIBUTE_MERGE_COLS → player_attributes.parquet → model feature_cols

Surfaces the exact gap diagnosed in the plan: a feature added to
config.PFF_GRADE_FEATURES_BY_GROUP that fails to appear with _mean/_peak/_last
suffixes in any per-position trained model — that's the silent-drop signature.
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
from config import (   # noqa: E402
    ATTRIBUTE_MERGE_COLS,
    PFF_GRADE_FEATURES_BY_GROUP,
    PROCESSED_DIR,
)

# Reuse the engineered-vs-raw classifier from src/validation.py
from src.validation import _is_pff_engineered_name   # noqa: E402

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent
FEATURE_FLOW_CSV = OUT_DIR / "feature_flow.csv"

DRAFT_MODELS_DIR = PROCESSED_DIR / "draft_models"
CTX_MODELS_DIR = PROCESSED_DIR / "contextual_models"

# Per-base aggregation suffixes that load_pff_career_features emits.
PFF_AGG_SUFFIXES = ("_mean", "_peak", "_last")


@dataclass
class FlowRow:
    pos_group: str
    feature: str            # base name from config
    kind: str               # "raw" | "engineered" | "career_team" | "attribute"
    in_pff_parquet: Optional[bool]
    in_attributes: Optional[bool]
    in_role_model: Optional[bool]   # True if base or any aggregated form is in feature_cols
    in_impact_model: Optional[bool]
    detail: str             # short note
    severity: str = "OK"    # "OK" | "WARNING" | "ERROR"


def _load_model_feature_cols(path: Path) -> Optional[list[str]]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            bundle = pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path.name, exc)
        return None
    return list(bundle.get("feature_cols", []))


def _has_feature_form(feature_cols: list[str], base: str) -> bool:
    """True if `base` appears verbatim or with one of the agg suffixes in
    feature_cols."""
    if base in feature_cols:
        return True
    return any(f"{base}{sfx}" in feature_cols for sfx in PFF_AGG_SUFFIXES)


# ── Public collectors ────────────────────────────────────────────────────────

def collect() -> list[FlowRow]:
    rows: list[FlowRow] = []

    # PFF parquet column set
    pff_path = PROCESSED_DIR / "pff_season_grades.parquet"
    pff_cols = (set(pd.read_parquet(pff_path, columns=None).columns)
                if pff_path.exists() else set())

    # Attributes column set
    attr_path = PROCESSED_DIR / "player_attributes.parquet"
    attr_cols = (set(pd.read_parquet(attr_path, columns=None).columns)
                 if attr_path.exists() else set())

    for pos in PFF_GRADE_FEATURES_BY_GROUP:
        role_cols = _load_model_feature_cols(
            DRAFT_MODELS_DIR / f"{pos}_role_model.pkl"
        ) or []
        impact_cols = _load_model_feature_cols(
            DRAFT_MODELS_DIR / f"{pos}_impact_model.pkl"
        ) or []

        for feat in PFF_GRADE_FEATURES_BY_GROUP[pos]:
            engineered = _is_pff_engineered_name(feat)
            kind = "engineered" if engineered else "raw"

            in_pff = (
                None if not pff_cols else
                feat in pff_cols if not engineered else None
            )
            in_role = _has_feature_form(role_cols, feat) if role_cols else None
            in_impact = _has_feature_form(impact_cols, feat) if impact_cols else None

            severity = "OK"
            detail = ""
            if engineered:
                # Engineered features are computed at step-8 time; they should
                # appear with _mean/_peak/_last suffixes in trained model feature_cols.
                if in_role is False and in_impact is False:
                    severity = "ERROR"
                    detail = ("engineered feature absent from BOTH role and "
                              "impact model feature_cols — silent drop")
                elif in_role is False or in_impact is False:
                    severity = "WARNING"
                    detail = ("engineered feature absent from one of "
                              "role/impact model feature_cols")
            else:
                # Raw features must be present in pff_season_grades.parquet
                # AND in at least one trained model's feature_cols.
                if in_pff is False:
                    severity = "ERROR"
                    detail = "raw feature missing from pff_season_grades.parquet"
                elif (in_role is False) and (in_impact is False):
                    severity = "WARNING"
                    detail = ("raw feature in PFF parquet but absent from "
                              "both trained models")

            rows.append(FlowRow(
                pos_group=pos,
                feature=feat,
                kind=kind,
                in_pff_parquet=in_pff,
                in_attributes=None,
                in_role_model=in_role,
                in_impact_model=in_impact,
                detail=detail,
                severity=severity,
            ))

    # ATTRIBUTE_MERGE_COLS — every entry should land in player_attributes.parquet,
    # and every career_team_* should also reach trained models for at least one
    # position group (signal that the attribute merge in step 8 worked).
    sample_role_cols = _load_model_feature_cols(DRAFT_MODELS_DIR / "OL_role_model.pkl") or []
    sample_impact_cols = _load_model_feature_cols(DRAFT_MODELS_DIR / "OL_impact_model.pkl") or []

    for col in ATTRIBUTE_MERGE_COLS:
        kind = "career_team" if col.startswith("career_team_") else "attribute"
        in_attr = col in attr_cols if attr_cols else None
        # For career_team_* check OL model (which uses all 7 of them).
        in_role = (col in sample_role_cols) if (kind == "career_team" and sample_role_cols) else None
        in_impact = (col in sample_impact_cols) if (kind == "career_team" and sample_impact_cols) else None

        severity = "OK"
        detail = ""
        if in_attr is False:
            severity = "ERROR"
            detail = "ATTRIBUTE_MERGE_COL missing from player_attributes.parquet"
        elif kind == "career_team" and in_role is False and in_impact is False:
            severity = "WARNING"
            detail = ("career_team_* missing from OL model feature_cols "
                      "(check ATTRIBUTE_FEATURES wiring in step 8)")

        rows.append(FlowRow(
            pos_group="—",
            feature=col,
            kind=kind,
            in_pff_parquet=None,
            in_attributes=in_attr,
            in_role_model=in_role,
            in_impact_model=in_impact,
            detail=detail,
            severity=severity,
        ))

    return rows


def write_csv(rows: list[FlowRow]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with FEATURE_FLOW_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos_group", "feature", "kind", "in_pff_parquet",
                    "in_attributes", "in_role_model", "in_impact_model",
                    "severity", "detail"])
        for r in rows:
            w.writerow([
                r.pos_group, r.feature, r.kind,
                "" if r.in_pff_parquet is None else r.in_pff_parquet,
                "" if r.in_attributes is None else r.in_attributes,
                "" if r.in_role_model is None else r.in_role_model,
                "" if r.in_impact_model is None else r.in_impact_model,
                r.severity, r.detail,
            ])


def build_section(rows: list[FlowRow]) -> str:
    lines = ["## Phase 3 — Feature Flow", ""]
    errors = [r for r in rows if r.severity == "ERROR"]
    warnings_ = [r for r in rows if r.severity == "WARNING"]

    if errors or warnings_:
        lines += ["**Issues found:**", ""]
        for r in errors:
            lines.append(
                f"- (ERROR) `{r.pos_group}/{r.feature}` ({r.kind}) — {r.detail}"
            )
        for r in warnings_:
            lines.append(
                f"- (WARN) `{r.pos_group}/{r.feature}` ({r.kind}) — {r.detail}"
            )
        lines.append("")
    else:
        lines += ["No feature-flow issues found.", ""]

    # Summary by kind
    by_kind: dict[str, dict[str, int]] = {}
    for r in rows:
        d = by_kind.setdefault(r.kind, {"OK": 0, "WARNING": 0, "ERROR": 0})
        d[r.severity] = d.get(r.severity, 0) + 1
    lines += ["### Summary", "",
              "| kind | OK | WARNING | ERROR |",
              "| --- | --- | --- | --- |"]
    for kind, counts in sorted(by_kind.items()):
        lines.append(
            f"| {kind} | {counts.get('OK',0)} | "
            f"{counts.get('WARNING',0)} | {counts.get('ERROR',0)} |"
        )
    return "\n".join(lines)


def has_errors(rows: list[FlowRow]) -> bool:
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
