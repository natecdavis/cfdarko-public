"""
Cross-head Boruta summary report.

After a `--retune-only 8,8.5` (or full `--retune`) run, this script reads
both `boruta_features.json` files (per-position + global), assembles a long
(feature × head) verdict frame, and surfaces the features Boruta confirmed
in zero heads — the global-drop candidates the user inspects to decide
what upstream cost can be removed.

Outputs:
    data/processed/draft_models/boruta_summary.parquet  (long format)
    + a logged report distinguishing:
      - state-space PFF cols (kalman_*_pff_*, ewma_*_pff_*) → step 06s candidates
      - PFF career cols (pff_*_{mean,peak,last})           → step 1c candidates
      - game-grain Kalman/EWMA (kalman_*, ewma_*)          → step 6 candidates
      - other (attributes, combine, derived)               → step 3 / inline

Usage:
    python experiments/boruta_summary.py
    python experiments/boruta_summary.py --include-tentative
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR

PER_POS_BORUTA_PATH = PROCESSED_DIR / "draft_models" / "boruta_features.json"
GLOBAL_BORUTA_PATH  = PROCESSED_DIR / "global_draft_models" / "boruta_features.json"
SUMMARY_OUT_PATH    = PROCESSED_DIR / "draft_models" / "boruta_summary.parquet"
TRAINING_METRICS_PATH        = PROCESSED_DIR / "draft_models" / "training_metrics.json"
GLOBAL_TRAINING_METRICS_PATH = PROCESSED_DIR / "global_draft_models" / "training_metrics.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _classify_feature(name: str) -> str:
    """Map a feature name to the upstream pipeline step that produces it.
    The classifier is the leverage point: if EVERY rejected feature in a
    group comes from one step, that step is a clean drop candidate."""
    # Order matters — kalman_*_pff_* is more specific than kalman_*.
    if name.startswith(("kalman_mean_pff_", "kalman_var_pff_",
                         "ewma_pff_", "ewma_var_pff_",
                         "kalman_yoy_pff_", "ewma_yoy_pff_")):
        return "06s_pff_state_space"
    if name.startswith(("kalman_mean_", "kalman_var_",
                         "kalman_yoy_", "ewma_yoy_",
                         "ewma_var_", "ewma_")):
        return "06_kalman_ewma"
    if name.startswith("pff_") and name.endswith(("_mean", "_peak", "_last")):
        return "1c_pff_career"
    if name.startswith("pff_"):
        return "1c_pff_other"
    if name.startswith("combine_"):
        return "00d_combine"
    if name in {"hasan_rank", "hasan_draft_year"}:
        return "00e_hasan"
    if name in {"is_ol_tackle", "is_ol_guard", "is_ol_center",
                "pass_rush_composite", "team_sack_share",
                "pff_pass_rush_snap_pct", "pff_coverage_snap_pct",
                "pff_corner_snap_pct", "pff_slot_snap_pct", "pff_fs_snap_pct"}:
        return "08_derived"
    if name == "position_group_ordinal":
        return "08b_global"
    return "03_attributes_or_other"


def _load_verdicts(path: Path, prefix: str) -> dict[str, dict]:
    """Load a boruta_features.json and strip the `_meta` block."""
    if not path.exists():
        logger.warning("Missing %s — run --retune-only %s first", path, prefix)
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items()
            if not k.startswith("_") and isinstance(v, dict)}


def _build_long_frame(per_pos: dict, global_: dict, *,
                      include_tentative: bool) -> pd.DataFrame:
    """Long format: one row per (head, feature, verdict)."""
    rows: list[dict] = []
    for head_dict in (per_pos, global_):
        for head, block in head_dict.items():
            confirmed = block.get("confirmed") or []
            tentative = block.get("tentative") or []
            n_in = block.get("n_features_in", 0)
            n_after = block.get("n_features_after", len(confirmed))
            # Confirmed
            for c in confirmed:
                rows.append({"head": head, "feature": c, "verdict": "confirmed"})
            # Tentative
            for c in tentative:
                rows.append({"head": head, "feature": c,
                              "verdict": "tentative" if not include_tentative else "confirmed"})
            # Track n_in and n_after as a head-level annotation row at feature=""
            rows.append({"head": head, "feature": "__head_meta__",
                          "verdict": "meta",
                          "n_features_in":  int(n_in),
                          "n_features_after": int(n_after)})
    df = pd.DataFrame(rows)
    return df


def _drop_candidate_report(per_pos: dict, global_: dict,
                            *, include_tentative: bool) -> pd.DataFrame:
    """Return a DataFrame of features confirmed in zero heads, with their
    upstream-step classification and the count of heads that even saw them."""
    all_heads = list(per_pos.keys()) + list(global_.keys())
    if not all_heads:
        return pd.DataFrame(columns=["feature", "step", "n_heads_confirming",
                                      "n_heads_tentative", "n_heads_rejected"])

    # Aggregate over heads.
    confirm_count: dict[str, int] = {}
    tentative_count: dict[str, int] = {}
    seen_count: dict[str, int] = {}
    for block in list(per_pos.values()) + list(global_.values()):
        confirmed = set(block.get("confirmed") or [])
        tentative = set(block.get("tentative") or [])
        for f in confirmed:
            confirm_count[f] = confirm_count.get(f, 0) + 1
            seen_count[f] = seen_count.get(f, 0) + 1
        for f in tentative:
            tentative_count[f] = tentative_count.get(f, 0) + 1
            seen_count[f] = seen_count.get(f, 0) + 1
        # Rejected features aren't enumerated in the JSON (we save count only),
        # so seen_count is approximate — covers only confirmed/tentative heads.

    all_features = set(confirm_count) | set(tentative_count)
    rows = []
    for f in sorted(all_features):
        nc = confirm_count.get(f, 0)
        nt = tentative_count.get(f, 0)
        if nc == 0 and (not include_tentative or nt == 0):
            rows.append({
                "feature": f,
                "step": _classify_feature(f),
                "n_heads_confirming": nc,
                "n_heads_tentative":  nt,
            })
    return pd.DataFrame(rows).sort_values(["step", "feature"]).reset_index(drop=True)


def _per_head_report(per_pos: dict, global_: dict) -> pd.DataFrame:
    rows = []
    for head_dict in (per_pos, global_):
        for head, block in head_dict.items():
            rows.append({
                "head":              head,
                "n_features_in":     int(block.get("n_features_in", 0)),
                "n_features_after":  int(block.get("n_features_after", 0)),
                "n_confirmed":       len(block.get("confirmed") or []),
                "n_tentative":       len(block.get("tentative") or []),
                "rejected_count":    int(block.get("rejected_count", 0)),
                "n_train_rows":      int(block.get("n_train_rows", 0)),
                "boruta_train_end":  int(block.get("boruta_train_end", 0)),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["frac_kept"] = df["n_features_after"] / df["n_features_in"].clip(lower=1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-tentative", action="store_true",
                        help="Treat tentative features as kept when surfacing "
                             "drop candidates. Default: only confirmed counts.")
    args = parser.parse_args()

    per_pos = _load_verdicts(PER_POS_BORUTA_PATH, "8")
    global_ = _load_verdicts(GLOBAL_BORUTA_PATH, "8.5")
    if not per_pos and not global_:
        logger.error("No Boruta caches found — run `python run_pipeline.py "
                     "--retune-only 8,8.5` first")
        sys.exit(1)

    # ── Per-head shrinkage ──────────────────────────────────────────────
    head_df = _per_head_report(per_pos, global_)
    logger.info("Per-head feature shrinkage:")
    for _, row in head_df.sort_values("frac_kept").iterrows():
        logger.info("  %-30s  %4d → %4d  (%.0f%% kept; n_train=%d, train_end=%d)",
                    row["head"], int(row["n_features_in"]),
                    int(row["n_features_after"]), 100 * row["frac_kept"],
                    int(row["n_train_rows"]), int(row["boruta_train_end"]))

    # ── Drop candidates ────────────────────────────────────────────────
    drops = _drop_candidate_report(per_pos, global_,
                                    include_tentative=args.include_tentative)
    logger.info("")
    logger.info("Drop candidates — confirmed in 0 heads%s:",
                " (and 0 tentative)" if not args.include_tentative else "")
    if drops.empty:
        logger.info("  (none — every feature is confirmed in ≥1 head)")
    else:
        for step, sub in drops.groupby("step"):
            logger.info("  [%s]  %d feature(s)", step, len(sub))
            for f in sub["feature"].head(40):
                logger.info("    %s", f)
            if len(sub) > 40:
                logger.info("    ... +%d more", len(sub) - 40)

    # ── State-space drop check ─────────────────────────────────────────
    ss_drops = drops[drops["step"] == "06s_pff_state_space"]
    ss_total = sum(
        sum(1 for c in (block.get("confirmed") or []) + (block.get("tentative") or [])
            if _classify_feature(c) == "06s_pff_state_space")
        for block in list(per_pos.values()) + list(global_.values())
    )
    logger.info("")
    if ss_total == 0 and not ss_drops.empty:
        logger.info("STATE-SPACE FULL DROP CANDIDATE: 0 confirmed state-space "
                    "features anywhere — consider USE_PFF_STATE_SPACE=False to "
                    "skip pipeline steps 04s/05s/06s (~30-45min savings/retune).")
    else:
        logger.info("State-space cols confirmed in ≥1 head — keep "
                    "USE_PFF_STATE_SPACE=True. (%d state-space confirmations across heads)",
                    ss_total)

    # ── Long frame for the user to slice further ──────────────────────
    long_df = _build_long_frame(per_pos, global_,
                                 include_tentative=args.include_tentative)
    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(SUMMARY_OUT_PATH, index=False)
    logger.info("Wrote long-format summary → %s (%d rows)",
                SUMMARY_OUT_PATH, len(long_df))


if __name__ == "__main__":
    main()
