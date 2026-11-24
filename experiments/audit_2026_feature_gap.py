"""Audit feature missingness + prediction-mode density for the 2026 draft class.

For each position group, reports:
  - % of 2026-class players (last_college_season_year=2025) missing each of
    the top-10 most important features for that head per training_metrics.json
  - Mode density of the head's prediction (largest equal-value cluster)

Reads-only. Run: python experiments/audit_2026_feature_gap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PROJ = REPO / "data" / "output" / "projections" / "latest.parquet"
METRICS = REPO / "data" / "processed" / "draft_models" / "training_metrics.json"

# 2026 NFL draft = players whose last college season was the 2025 fall season
DRAFT_CLASS_LAST_SEASON = 2025

HEADS = [
    ("role", "role_score"),
    ("impact", "impact_apy_score"),
    ("impact_wav", "impact_wav_score"),
]


def main() -> None:
    df = pd.read_parquet(PROJ)
    with open(METRICS) as f:
        metrics = json.load(f)

    cohort = df[df["last_college_season_year"].fillna(0) == DRAFT_CLASS_LAST_SEASON].copy()

    pos_groups = sorted(p for p in cohort["position_group"].dropna().unique() if p != "OTHER")

    rows = []
    for pos in pos_groups:
        pos_df = cohort[cohort["position_group"] == pos]
        n = len(pos_df)
        if n == 0:
            continue
        for head, score_col in HEADS:
            key = f"{pos}_{head}"
            if key not in metrics:
                continue
            top = metrics[key].get("feature_importance_top20", [])[:10]
            if not top:
                continue
            present = [f["feature"] for f in top if f["feature"] in pos_df.columns]
            miss_rates = {f: pos_df[f].isna().mean() for f in present}
            avg_miss = float(np.mean(list(miss_rates.values()))) if miss_rates else float("nan")
            high_miss = sum(1 for v in miss_rates.values() if v >= 0.5)

            if score_col in pos_df.columns:
                preds = pos_df[score_col].dropna()
                if len(preds):
                    mode_val = preds.mode().iloc[0]
                    mode_density = float((preds == mode_val).mean())
                    n_distinct = int(preds.nunique())
                else:
                    mode_val = float("nan"); mode_density = float("nan"); n_distinct = 0
            else:
                mode_val = float("nan"); mode_density = float("nan"); n_distinct = 0

            rows.append({
                "pos": pos,
                "head": head,
                "n_2026": n,
                "n_top_feats": len(present),
                "avg_miss": avg_miss,
                "n_feats_>=50%_miss": high_miss,
                "n_distinct_preds": n_distinct,
                "mode_density": mode_density,
                "mode_value": mode_val,
            })

    out = pd.DataFrame(rows).sort_values(["head", "pos"]).reset_index(drop=True)

    print(f"\n2026 draft class feature/prediction audit (last_college_season_year={DRAFT_CLASS_LAST_SEASON})")
    print(f"Source: {PROJ.relative_to(REPO)}\n")
    fmt = out.copy()
    fmt["avg_miss"] = (fmt["avg_miss"] * 100).round(1).astype(str) + "%"
    fmt["mode_density"] = (fmt["mode_density"] * 100).round(1).astype(str) + "%"
    fmt["mode_value"] = fmt["mode_value"].apply(lambda v: f"{v:.5g}" if pd.notna(v) else "—")
    print(fmt.to_string(index=False))

    print("\nDetailed missingness for IMPACT (apy) head — top-10 importance per pos:")
    for pos in pos_groups:
        key = f"{pos}_impact"
        if key not in metrics:
            continue
        top = metrics[key].get("feature_importance_top20", [])[:10]
        pos_df = cohort[cohort["position_group"] == pos]
        if len(pos_df) == 0 or not top:
            continue
        print(f"\n  {pos} (n={len(pos_df)}):")
        for f in top:
            name = f["feature"]
            if name not in pos_df.columns:
                print(f"    {name:35s}  (column not in output)")
                continue
            mr = pos_df[name].isna().mean()
            print(f"    {name:35s}  imp={f['importance']:>4d}  missing={mr:6.1%}")

    print("\nFlag: rows where avg_miss >= 50% AND mode_density >= 50% are 'feature-bound'.")


if __name__ == "__main__":
    main()
