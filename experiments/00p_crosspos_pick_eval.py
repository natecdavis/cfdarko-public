"""
Cross-position ranking evaluation against draft position.

Motivation
----------
cfDARKO sorts its final board across positions (`10_output.py` sorts by
`impact_score`; the frontend shows a cross-position `impact_rank`), but until
now NOTHING in the repo ever measured whether that cross-position ordering is
sensible. Every tracked Spearman is within-position, and `overall_pick` was
never used in any evaluation. This harness closes that gap.

It answers two questions:

  (A) VALIDATION — on historical fold-val cohorts, how well does each score
      rank players ACROSS positions against where the NFL actually drafted
      them? Compares raw `impact_apy` against its replacement-adjusted form.

  (B) DIAGNOSTIC — on the current class, how much does the replacement
      adjustment actually move the cross-position board? Reports rank
      correlation between the two orderings and top-N composition shift.

Caveat, stated up front: `overall_pick` is the draft MARKET's opinion, not
ground truth about NFL value, and it is partly circular (draft position feeds
the trajectory models that generate the labels these heads train on). Treat it
as a directional sanity check on cross-position calibration — never as a
fitting target and never as a primary adoption gate.

Read-only. Writes one JSON to experiments/.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, OUTPUT_DIR, ROOT_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = ROOT_DIR / "experiments"


def _load_replacement_levels() -> dict[str, float]:
    path = PROCESSED_DIR / "replacement_levels.parquet"
    if not path.exists():
        logger.warning("replacement_levels.parquet missing — run step 7")
        return {}
    lv = pd.read_parquet(path)
    pos = lv[lv["scope"] == "position"]
    return dict(zip(pos["key"].astype(str),
                    pos["replacement_apy_pct"].astype(float)))


def _rho_vs_pick(scores: pd.Series, picks: pd.Series) -> tuple[float, int]:
    """Spearman between score and draft capital.

    `overall_pick` is inverted so that HIGHER = better player; a positive rho
    then means the score agrees with the market.
    """
    ok = scores.notna() & picks.notna()
    if ok.sum() < 20:
        return float("nan"), int(ok.sum())
    r = spearmanr(scores[ok], -picks[ok]).statistic
    return (float(r) if np.isfinite(r) else float("nan")), int(ok.sum())


def evaluate_historical(levels: dict[str, float]) -> dict:
    """(A) Cross-position Spearman vs overall_pick on global OOF folds."""
    oof_path = PROCESSED_DIR / "global_draft_models" / "global_oof_predictions.parquet"
    bridge_path = PROCESSED_DIR / "bridge_players.parquet"
    if not oof_path.exists() or not bridge_path.exists():
        logger.warning("Missing OOF or bridge artifact — skipping historical eval")
        return {}

    oof = pd.read_parquet(oof_path)
    bridge = pd.read_parquet(bridge_path)
    picks = (bridge[["player_id", "overall_pick"]]
             .dropna(subset=["player_id"])
             .drop_duplicates("player_id"))
    picks["overall_pick"] = pd.to_numeric(picks["overall_pick"], errors="coerce")

    df = oof[oof["model_name"] == "impact_apy"].merge(picks, on="player_id",
                                                       how="left")
    # Only drafted players carry a pick; undrafted have no market signal on
    # this scale and would otherwise be scored as "pick = missing".
    df = df[df["overall_pick"].notna() & df["is_bridge"].astype(bool)]
    if df.empty:
        logger.warning("No drafted bridge rows in OOF — skipping")
        return {}

    df["repl"] = df["position_group"].astype(str).map(levels)
    # Unfloored, matching 10_output._add_replacement_columns — see the note
    # there on why clipping at zero was measured and rejected.
    df["pred_ior"] = df["pred"] - df["repl"]

    results = {"folds": [], "pooled": {}}
    for (vs, ve), grp in df.groupby(["val_start", "val_end"]):
        r_raw, n = _rho_vs_pick(grp["pred"], grp["overall_pick"])
        r_ior, _ = _rho_vs_pick(grp["pred_ior"], grp["overall_pick"])
        results["folds"].append({
            "val_start": int(vs), "val_end": int(ve), "n": n,
            "rho_raw": r_raw, "rho_over_replacement": r_ior,
            "delta": (r_ior - r_raw) if np.isfinite(r_raw) and np.isfinite(r_ior)
                     else float("nan"),
        })
        logger.info("  fold %d-%d (n=%4d): raw=%+.4f  over_repl=%+.4f  (%+.4f)",
                    vs, ve, n, r_raw, r_ior, r_ior - r_raw)

    r_raw, n = _rho_vs_pick(df["pred"], df["overall_pick"])
    r_ior, _ = _rho_vs_pick(df["pred_ior"], df["overall_pick"])
    results["pooled"] = {"n": n, "rho_raw": r_raw,
                         "rho_over_replacement": r_ior,
                         "delta": r_ior - r_raw}
    logger.info("  POOLED   (n=%4d): raw=%+.4f  over_repl=%+.4f  (%+.4f)",
                n, r_raw, r_ior, r_ior - r_raw)

    # Per-position within-group rho, as a control: the adjustment is a
    # per-position constant, so within-position rho MUST be unchanged. Any
    # movement here is a bug.
    within = {}
    for pos, grp in df.groupby("position_group"):
        a, npos = _rho_vs_pick(grp["pred"], grp["overall_pick"])
        b, _ = _rho_vs_pick(grp["pred_ior"], grp["overall_pick"])
        within[str(pos)] = {"n": npos, "rho_raw": a, "rho_over_replacement": b}
    results["within_position_control"] = within
    moved = [p for p, v in within.items()
             if np.isfinite(v["rho_raw"]) and np.isfinite(v["rho_over_replacement"])
             and abs(v["rho_raw"] - v["rho_over_replacement"]) > 1e-9]
    if moved:
        logger.warning("  within-position rho moved for %s — expected identical "
                       "(clip at 0 can reorder if scores fall below the line)",
                       ", ".join(moved))
    results["within_position_moved"] = moved
    return results


def evaluate_current_class(levels: dict[str, float], top_n: int = 100) -> dict:
    """(B) How much does the adjustment move the current-class board?"""
    latest = OUTPUT_DIR / "projections" / "latest.parquet"
    if not latest.exists():
        logger.warning("latest.parquet missing — skipping current-class diagnostic")
        return {}
    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(latest).schema.names)
    cols = ["player_id", "player_name", "position_group", "impact_score",
            "impact_over_replacement", "last_college_season_year"]
    missing = [c for c in cols if c not in available]
    if missing:
        logger.warning("latest.parquet lacks %s — rerun step 10; "
                       "skipping current-class diagnostic", ", ".join(missing))
        return {}
    df = pd.read_parquet(latest, columns=cols)
    cur = df[df["last_college_season_year"] >= df["last_college_season_year"].max() - 1]
    cur = cur[cur["impact_score"].notna()].copy()
    if cur.empty or "impact_over_replacement" not in cur.columns:
        logger.warning("No current-class rows with both scores — skipping")
        return {}

    rho = spearmanr(cur["impact_score"], cur["impact_over_replacement"]).statistic
    top_raw = set(cur.nlargest(top_n, "impact_score")["player_id"])
    top_ior = set(cur.nlargest(top_n, "impact_over_replacement")["player_id"])
    overlap = len(top_raw & top_ior)

    def _pos_mix(ids):
        return (cur[cur["player_id"].isin(ids)]["position_group"]
                .value_counts().to_dict())

    out = {
        "n_current": int(len(cur)),
        "rho_raw_vs_over_replacement": float(rho),
        f"top{top_n}_overlap": overlap,
        f"top{top_n}_position_mix_raw": _pos_mix(top_raw),
        f"top{top_n}_position_mix_over_replacement": _pos_mix(top_ior),
    }
    logger.info("  current class n=%d, rank corr between orderings=%.5f",
                len(cur), rho)
    logger.info("  top-%d overlap: %d/%d", top_n, overlap, top_n)
    logger.info("  top-%d mix raw : %s", top_n, out[f"top{top_n}_position_mix_raw"])
    logger.info("  top-%d mix IOR : %s", top_n,
                out[f"top{top_n}_position_mix_over_replacement"])
    return out


def main():
    levels = _load_replacement_levels()
    if not levels:
        logger.error("No replacement levels available — nothing to compare")
        return
    logger.info("Replacement levels: %s",
                {k: round(v, 5) for k, v in sorted(levels.items())})

    logger.info("(A) Historical cross-position ranking vs overall_pick:")
    hist = evaluate_historical(levels)

    logger.info("(B) Current-class board movement:")
    cur = evaluate_current_class(levels)

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "replacement_levels": levels,
        "historical": hist,
        "current_class": cur,
    }
    out = EXPERIMENTS_DIR / f"crosspos_pick_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("→ %s", out)


if __name__ == "__main__":
    main()
