"""
Step 00i: Combine Z-Score vs Raw — Empirical Audit

For each (combine_metric × position_group) cell, asks two questions:
  1. Year-drift: how much variance in the metric is explained by cohort_year?
     High R²(year) ⇒ z-scoring is doing real work removing secular drift.
  2. Predictive correlation: does the raw value or the cohort-z'd value
     correlate more strongly with the bridge-cohort labels (role / impact /
     impact_wav)?

Decision rule (CLI-tunable):
  • drift_r2 ≥ --drift-keep (0.05)            → keep z (year-drift is real)
  • drift_r2 < --drift-floor (0.02) AND
    mean(|corr_raw|) − mean(|corr_z|) ≥
                       --corr-min (0.01)      → switch to raw
  • otherwise                                  → keep z (status quo)

Inputs (read-only):
  data/processed/combine_metrics.parquet
  data/processed/nfl_draft_labels.parquet
  data/processed/player_attributes.parquet

Outputs:
  experiments/combine_zscore_audit.parquet — full per-cell analysis
  experiments/combine_zscore_audit.md      — human-readable summary
  stdout footer: paste-ready COMBINE_RAW_BY_GROUP dict for config.py

Usage:
  python experiments/00i_combine_zscore_audit.py
  python experiments/00i_combine_zscore_audit.py --drift-keep 0.10 --corr-min 0.02
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, COMBINE_FEATURES_BY_GROUP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

METRICS = ["forty", "vertical", "broad_jump", "bench", "cone", "shuttle",
           "ht", "wt", "arm", "hand", "wingspan"]
POSITIONS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K", "P"]
TARGETS = ["role_target", "impact_target", "impact_w_av_per_game"]

# Minimum sample sizes
MIN_DRIFT_N = 50
MIN_CORR_N = 30

# Output paths
OUT_DIR = Path(__file__).parent
OUT_PARQUET = OUT_DIR / "combine_zscore_audit.parquet"
OUT_MD = OUT_DIR / "combine_zscore_audit.md"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--drift-keep", type=float, default=0.05,
                   help="R²(year) at or above which we always keep z (default 0.05)")
    p.add_argument("--drift-floor", type=float, default=0.02,
                   help="R²(year) below which raw is allowed to win (default 0.02)")
    p.add_argument("--corr-min", type=float, default=0.01,
                   help="Min Δ in mean |corr| (raw − z) to recommend a swap (default 0.01)")
    return p.parse_args()


def _drift_stats(values: pd.Series, years: pd.Series) -> dict:
    """OLS metric ~ year. Returns slope/decade, R², n."""
    pair = pd.concat([values, years], axis=1).dropna()
    pair.columns = ["v", "y"]
    if len(pair) < MIN_DRIFT_N:
        return {"n_drift": len(pair), "slope_per_decade": np.nan, "drift_r2": np.nan,
                "metric_mean": float(pair["v"].mean()) if len(pair) else np.nan}
    x = pair["y"].astype(float).values
    y = pair["v"].astype(float).values
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "n_drift": int(len(pair)),
        "slope_per_decade": float(slope * 10),
        "drift_r2": float(r2),
        "metric_mean": float(y.mean()),
    }


def _corr(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < MIN_CORR_N:
        return (np.nan, int(len(pair)))
    if pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
        return (np.nan, int(len(pair)))
    return (float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), int(len(pair)))


def _decide(drift_r2: float, mean_abs_raw: float, mean_abs_z: float,
            args) -> str:
    if pd.isna(drift_r2) or pd.isna(mean_abs_raw) or pd.isna(mean_abs_z):
        return "insufficient_data"
    if drift_r2 >= args.drift_keep:
        return "keep_z"
    delta = mean_abs_raw - mean_abs_z
    if drift_r2 < args.drift_floor and delta >= args.corr_min:
        return "switch_to_raw"
    return "keep_z"


def main():
    args = parse_args()

    combine = pd.read_parquet(PROCESSED_DIR / "combine_metrics.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    attrs = pd.read_parquet(PROCESSED_DIR / "player_attributes.parquet")

    # Add position_group from player_attributes
    attrs_pos = attrs[["player_id", "position_group"]].drop_duplicates(subset="player_id")
    combine = combine.merge(attrs_pos, on="player_id", how="left")
    n_no_pos = combine["position_group"].isna().sum()
    if n_no_pos:
        logger.info("Combine rows with no position_group (dropped from analysis): %d", n_no_pos)
    combine = combine[combine["position_group"].notna()].copy()
    logger.info("Combine rows after pos join: %d", len(combine))

    # For drift analysis: use the full combine population (more data than just bridge)
    drift_pool = combine.copy()

    # For correlation: bridge cohort only (these are the rows with non-trivial labels)
    bridge = labels[labels["is_bridge"]][["player_id"] + TARGETS].copy()
    corr_pool = combine.merge(bridge, on="player_id", how="inner")
    logger.info("Bridge × combine rows for correlation: %d", len(corr_pool))

    rows: list[dict] = []
    for metric in METRICS:
        raw_col = f"combine_{metric}"
        z_col = f"combine_{metric}_z"
        if raw_col not in combine.columns or z_col not in combine.columns:
            logger.warning("Missing column for metric %s — skipping", metric)
            continue
        for pos in POSITIONS:
            sub_drift = drift_pool[drift_pool["position_group"] == pos]
            ds = _drift_stats(sub_drift[raw_col], sub_drift["combine_draft_year"])

            sub_corr = corr_pool[corr_pool["position_group"] == pos]
            corr_row: dict = {}
            for tgt in TARGETS:
                cz, n_z = _corr(sub_corr[z_col], sub_corr[tgt])
                cr, n_r = _corr(sub_corr[raw_col], sub_corr[tgt])
                corr_row[f"corr_z_{tgt}"] = cz
                corr_row[f"corr_raw_{tgt}"] = cr
                corr_row[f"n_corr_{tgt}"] = n_z if n_z == n_r else min(n_z, n_r)

            valid_z = [abs(corr_row[f"corr_z_{t}"]) for t in TARGETS
                       if not pd.isna(corr_row[f"corr_z_{t}"])]
            valid_r = [abs(corr_row[f"corr_raw_{t}"]) for t in TARGETS
                       if not pd.isna(corr_row[f"corr_raw_{t}"])]
            mean_abs_z = float(np.mean(valid_z)) if valid_z else np.nan
            mean_abs_r = float(np.mean(valid_r)) if valid_r else np.nan

            decision = _decide(ds["drift_r2"], mean_abs_r, mean_abs_z, args)

            rows.append({
                "metric": metric,
                "pos": pos,
                **ds,
                **corr_row,
                "mean_abs_corr_z": mean_abs_z,
                "mean_abs_corr_raw": mean_abs_r,
                "delta_corr": (mean_abs_r - mean_abs_z) if not (pd.isna(mean_abs_r) or pd.isna(mean_abs_z)) else np.nan,
                "recommendation": decision,
            })

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    logger.info("Wrote %d analysis rows → %s", len(out), OUT_PARQUET)

    _write_markdown(out, args)
    _print_paste_block(out)


def _write_markdown(df: pd.DataFrame, args) -> None:
    lines: list[str] = []
    lines.append(f"# Combine z-score vs raw — audit\n")
    lines.append(f"_Thresholds: drift-keep={args.drift_keep}, drift-floor={args.drift_floor}, corr-min={args.corr_min}_\n")
    lines.append("Decision rule: keep_z if drift_r2 ≥ drift-keep; switch_to_raw if drift_r2 < drift-floor AND Δ|corr| ≥ corr-min.\n\n")

    summary = df.groupby("recommendation").size()
    lines.append("## Decisions summary\n")
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v} cells\n")
    lines.append("\n")

    for metric in df["metric"].unique():
        sub = df[df["metric"] == metric].copy()
        lines.append(f"## {metric}\n\n")
        lines.append("| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |\n")
        lines.append("|---|--:|--:|--:|--:|--:|--:|--:|---|\n")
        for _, r in sub.iterrows():
            n_corr = max(int(r.get("n_corr_role_target", 0) or 0),
                         int(r.get("n_corr_impact_target", 0) or 0),
                         int(r.get("n_corr_impact_w_av_per_game", 0) or 0))
            slope = r["slope_per_decade"]
            r2 = r["drift_r2"]
            mz = r["mean_abs_corr_z"]
            mr = r["mean_abs_corr_raw"]
            d = r["delta_corr"]
            rec = r["recommendation"]
            mark = "**switch_to_raw**" if rec == "switch_to_raw" else rec
            lines.append(
                f"| {r['pos']} | {r['n_drift']} | "
                f"{slope:+.3f} | {r2:.3f} | {n_corr} | "
                f"{mz:.3f} | {mr:.3f} | {d:+.3f} | {mark} |\n"
            )
        lines.append("\n")

    OUT_MD.write_text("".join(lines))
    logger.info("Wrote markdown → %s", OUT_MD)


def _print_paste_block(df: pd.DataFrame) -> None:
    """Emit a paste-ready Python dict listing (pos × metric) where the
    audit recommends switching to raw."""
    swaps = df[df["recommendation"] == "switch_to_raw"].copy()
    print("\n" + "=" * 64)
    print("Audit summary — recommendations vs current COMBINE_FEATURES_BY_GROUP")
    print("=" * 64)
    if swaps.empty:
        print("No swap recommendations. Audit suggests keeping all z-scored features.")
        return
    print(f"\n{len(swaps)} (metric × pos) cells recommend switch_to_raw.\n")

    by_pos: dict[str, list[str]] = {}
    for _, r in swaps.iterrows():
        by_pos.setdefault(r["pos"], []).append(r["metric"])

    print("# Paste this into config.py (proposed):")
    print("COMBINE_RAW_BY_GROUP: dict[str, list[str]] = {")
    for pos in POSITIONS:
        if pos not in by_pos:
            continue
        feats = sorted(f'"combine_{m}"' for m in by_pos[pos])
        print(f'    "{pos}": [{", ".join(feats)}],')
    print("}\n")

    print("# Per-position whitelist BEFORE / AFTER (z-cols would be removed in favor of raw):")
    for pos, metrics in by_pos.items():
        cur = COMBINE_FEATURES_BY_GROUP.get(pos, [])
        z_to_remove = [f"combine_{m}_z" for m in metrics if f"combine_{m}_z" in cur]
        if z_to_remove:
            print(f"  {pos}: drop {z_to_remove}, add raw {[f'combine_{m}' for m in metrics]}")
    print()


if __name__ == "__main__":
    main()
