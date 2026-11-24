"""
Compare cv_mae + holdout_mae for the QB-quality + bridge-fix retune
(2026-05-16) against the pre-QB-quality baseline backup. Plus a top-100
projection delta report.

Reads:
  - data/processed/draft_models.bak_pre_qb_quality_20260516/training_metrics.json
  - data/processed/draft_models/training_metrics.json
  - data/processed/global_draft_models.bak_pre_qb_quality_20260516/training_metrics.json
  - data/processed/global_draft_models/training_metrics.json
  - data/processed/player_skill_snapshot.bak_pre_qb_quality_20260516.parquet (optional)
  - the latest data/output/projections/projections_*.parquet vs the baseline backup

Emits a punch-list of per-head deltas (sorted by cv_mae improvement), the
mean delta, and the largest top-100 rank shifts on each model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "output"

BASELINE_TAG = "bak_pre_qb_quality_20260516"


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _per_head_delta(baseline: dict, current: dict, key: str) -> pd.DataFrame:
    rows = []
    heads = set(baseline.keys()) | set(current.keys())
    for head in sorted(heads):
        b = baseline.get(head, {}).get(key)
        c = current.get(head, {}).get(key)
        if b is None and c is None:
            continue
        rows.append({
            "head":      head,
            "baseline":  b,
            "current":   c,
            "delta":     (c - b) if (b is not None and c is not None) else np.nan,
            "pct_delta": ((c - b) / abs(b) * 100.0)
                          if (b not in (None, 0) and c is not None) else np.nan,
        })
    return pd.DataFrame(rows)


def _print_head_table(df: pd.DataFrame, label: str, lower_is_better: bool = True) -> None:
    if df.empty:
        print(f"  [{label}] no overlap")
        return
    df = df.copy()
    df = df.sort_values("pct_delta", ascending=lower_is_better, na_position="last")
    print(f"\n[{label}] (sorted: best improvement first, {'lower' if lower_is_better else 'higher'} is better)")
    print(f"  {'head':<24s} {'baseline':>10s} {'current':>10s} {'delta':>10s} {'pct_delta':>10s}")
    for _, r in df.iterrows():
        b = "    nan" if pd.isna(r["baseline"]) else f"{r['baseline']:10.5f}"
        c = "    nan" if pd.isna(r["current"])  else f"{r['current']:10.5f}"
        d = "    nan" if pd.isna(r["delta"])    else f"{r['delta']:+10.5f}"
        p = "    nan" if pd.isna(r["pct_delta"]) else f"{r['pct_delta']:+9.2f}%"
        print(f"  {r['head']:<24s} {b} {c} {d} {p}")
    n = df["delta"].notna().sum()
    wins = (df["delta"] < 0).sum() if lower_is_better else (df["delta"] > 0).sum()
    mean_pct = df["pct_delta"].mean()
    print(f"  ----  mean pct delta = {mean_pct:+.2f}%, "
          f"wins={wins}/{n}  losses={n - wins}/{n}")


def cv_mae_report() -> None:
    print("=" * 80)
    print("Per-head cv_mae deltas — QB-quality + bridge-fix retune vs baseline")
    print("=" * 80)

    pairs = [
        ("Step 8 (per-position)",
         PROCESSED / f"draft_models.{BASELINE_TAG}" / "training_metrics.json",
         PROCESSED / "draft_models" / "training_metrics.json"),
        ("Step 8.5 (global)",
         PROCESSED / f"global_draft_models.{BASELINE_TAG}" / "training_metrics.json",
         PROCESSED / "global_draft_models" / "training_metrics.json"),
    ]
    for label, b_path, c_path in pairs:
        print(f"\n## {label}")
        b = _load_metrics(b_path)
        c = _load_metrics(c_path)
        if not b:
            print(f"  baseline missing: {b_path}")
            continue
        if not c:
            print(f"  current missing: {c_path}")
            continue
        df_cv = _per_head_delta(b, c, "cv_mae")
        _print_head_table(df_cv, "cv_mae", lower_is_better=True)
        df_ho = _per_head_delta(b, c, "holdout_mae")
        if not df_ho.empty:
            _print_head_table(df_ho, "holdout_mae", lower_is_better=True)


def latest_projections() -> Optional[Path]:
    proj_dir = OUTPUT / "projections"
    if not proj_dir.exists():
        return None
    files = sorted(proj_dir.glob("projections_*.parquet"))
    if not files:
        return None
    return files[-1]


def projection_delta_report() -> None:
    print("\n" + "=" * 80)
    print("Top-100 projection deltas (per position group)")
    print("=" * 80)
    baseline = OUTPUT / "projections" / f"projections.{BASELINE_TAG.replace('20260516', '20260515')}.parquet"
    if not baseline.exists():
        # try other backup naming
        for cand in OUTPUT.glob("projections/projections.bak_pre_qb_quality_*.parquet"):
            baseline = cand
            break
    current = latest_projections()
    if baseline is None or not baseline.exists():
        print(f"  baseline missing under {OUTPUT / 'projections'}/")
        return
    if current is None or not current.exists():
        print(f"  current projection missing under {OUTPUT / 'projections'}/")
        return
    print(f"  baseline: {baseline.name}")
    print(f"  current:  {current.name}")

    b = pd.read_parquet(baseline)
    c = pd.read_parquet(current)

    join_key = "player_id"
    if join_key not in b.columns or join_key not in c.columns:
        print(f"  join key '{join_key}' missing from projections — skipping")
        return

    rank_col_candidates = ("impact_apy_score", "impact_score", "impact_wav_score", "impact")
    rank_col = next((col for col in rank_col_candidates
                      if col in b.columns and col in c.columns), None)
    if rank_col is None:
        print(f"  no shared score column among {rank_col_candidates}")
        return

    pos_col = "position_group" if "position_group" in b.columns else None
    name_col = "player_name" if "player_name" in b.columns else "player_id"

    b_sub = b[[join_key, name_col, pos_col, rank_col]].rename(
        columns={rank_col: f"{rank_col}_b"})
    c_sub = c[[join_key, name_col, pos_col, rank_col]].rename(
        columns={rank_col: f"{rank_col}_c"})

    j = b_sub.merge(c_sub[[join_key, f"{rank_col}_c"]], on=join_key, how="outer")
    j["delta_score"] = j[f"{rank_col}_c"] - j[f"{rank_col}_b"]

    if pos_col:
        for pg, grp in j.groupby(pos_col):
            grp = grp.dropna(subset=[f"{rank_col}_b", f"{rank_col}_c"])
            if grp.empty:
                continue
            grp = grp.sort_values(f"{rank_col}_c", ascending=False).head(100).copy()
            grp["rank_b"] = grp[f"{rank_col}_b"].rank(ascending=False, method="min")
            grp["rank_c"] = grp[f"{rank_col}_c"].rank(ascending=False, method="min")
            grp["rank_shift"] = grp["rank_b"] - grp["rank_c"]
            biggest = grp.reindex(grp["rank_shift"].abs().sort_values(ascending=False).index).head(10)
            if biggest.empty:
                continue
            print(f"\n  --- {pg}: largest rank shifts in top-100 (current ranking) ---")
            print(f"    {'player':<30s} {'baseline':>10s} {'current':>10s} {'Δ score':>10s} "
                  f"{'b rank':>8s} {'c rank':>8s} {'shift':>8s}")
            for _, r in biggest.iterrows():
                print(f"    {str(r[name_col])[:30]:<30s} "
                      f"{r[f'{rank_col}_b']:>10.4f} "
                      f"{r[f'{rank_col}_c']:>10.4f} "
                      f"{r['delta_score']:>+10.4f} "
                      f"{r['rank_b']:>8.0f} "
                      f"{r['rank_c']:>8.0f} "
                      f"{r['rank_shift']:>+8.0f}")


def boruta_audit_report() -> None:
    print("\n" + "=" * 80)
    print("Boruta audit — qb_career_* + qb_rolling_* feature acceptance")
    print("=" * 80)
    paths = [
        ("Step 8 (per-position)", PROCESSED / "draft_models" / "boruta_summary.parquet"),
        ("Step 8.5 (global)",     PROCESSED / "global_draft_models" / "boruta_features.json"),
    ]
    for label, path in paths:
        if not path.exists():
            print(f"  [{label}] missing: {path}")
            continue
        print(f"\n## {label}: {path.name}")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            qb_rows = df[df["feature"].str.startswith(("qb_rolling_", "qb_career_"),
                                                       na=False)] if "feature" in df.columns else pd.DataFrame()
            if qb_rows.empty:
                print("  (no qb_* features in Boruta summary — check column name)")
            else:
                print(qb_rows.to_string(index=False))
        else:
            data = json.load(open(path))
            for head, block in data.items():
                accepted = block.get("accepted", [])
                rejected = block.get("rejected", [])
                qb_accepted = [f for f in accepted if f.startswith(("qb_rolling_", "qb_career_"))]
                qb_rejected = [f for f in rejected if f.startswith(("qb_rolling_", "qb_career_"))]
                if qb_accepted or qb_rejected:
                    print(f"  {head}: accepted={qb_accepted}  rejected={qb_rejected}")


if __name__ == "__main__":
    cv_mae_report()
    boruta_audit_report()
    projection_delta_report()
