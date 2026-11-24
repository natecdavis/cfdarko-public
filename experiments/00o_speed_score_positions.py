"""
Should `combine_speed_score` be enabled for more position groups?

`SPEED_SCORE_POSITIONS` is currently {DL, EDGE, LB, TE, RB}, selected by the
2026-07-22 audit on two criteria: univariate bridge-only Spearman vs impact
beating the best raw component (combine_wt / combine_forty), plus a fixed-param
A/B on the impact_wav heads.

Re-running that univariate screen over ALL groups (bridge players, vs
impact_target) reproduces the audit's WR exclusion, which is the check that the
method is faithful, and turns up three candidates plus one incumbent worth
re-testing:

  pos    speed   best component   margin
  OL    +0.213      +0.204        +0.009   <- strongest signal of any group
  DB    +0.100      +0.081        +0.019
  QB    +0.105      +0.097        +0.008   (n=180, thinnest cohort)
  RB    +0.115      +0.124        -0.009   <- INCUMBENT that FAILS the screen
  WR    +0.087      +0.097        -0.010   (audit excluded it; reproduced)

Those margins are small enough that the univariate screen cannot decide this on
its own: at n≈540 the standard error of a single Spearman is ~0.043, and while
the SE of the *difference* between two correlated measures on the same players is
much tighter than that, +0.009 is not a result. So this script runs the real
thing — each candidate group's three heads fit through step 8's own
`train_model_for_group`, with and without the feature.

Position groups are independent here: a head only ever sees its own group's
feature list, so enabling OL cannot perturb DB. That makes per-group arms valid
and cheap (~9 fits per group).

The `placebo` arm is load-bearing, not decoration. It re-fits the SAME data with
different LightGBM seeds, so its delta is noise by construction. The hasan
off-board experiment ([[project_hasan_offboard_null]]) found the placebo moving
Δρ more than any real arm, which is exactly the trap a small-margin feature
decision falls into. An arm that does not beat the placebo is not a win.

Decision metric is `cv_rho_bridge` — mean per-fold Spearman over the tuning
folds, restricted to players who reached the NFL. cv_mae is reported but is
dominated by the ~90% zero-padded non-NFL mass.

Usage:
    python3 experiments/00o_speed_score_positions.py
    python3 experiments/00o_speed_score_positions.py --positions OL --seeds 5
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import POSITION_GROUPS, PROCESSED_DIR, SPEED_SCORE_POSITIONS

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

HEADS = [("role_target", "role"),
         ("impact_target", "impact"),
         ("impact_w_av_per_game", "impact_wav")]

# Candidates from the univariate screen + the incumbent that fails it.
DEFAULT_CANDIDATES = ["OL", "DB", "QB", "RB"]
PLACEBO_SEED_OFFSET = 100
FEATURE = "combine_speed_score"


def _load_step8():
    path = Path(__file__).resolve().parent.parent / "pipeline" / "08_draft_models.py"
    spec = importlib.util.spec_from_file_location("_p8_speed", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fold_rho_bridge(oof: pd.DataFrame, bridge_ids: set) -> float:
    """Mean per-fold Spearman restricted to players who reached the NFL."""
    if oof.empty:
        return np.nan
    is_bridge = oof["player_id"].isin(bridge_ids)
    out = []
    for _, fold in oof.groupby(["val_start", "val_end"]):
        sub = fold[is_bridge.loc[fold.index]]
        if len(sub) >= 10 and sub["y"].nunique() > 1:
            r, _ = spearmanr(sub["pred"], sub["y"])
            if r is not None and not np.isnan(r):
                out.append(float(r))
    return float(np.mean(out)) if out else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default=",".join(DEFAULT_CANDIDATES),
                    help=f"Comma-separated groups to test (default {DEFAULT_CANDIDATES}).")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    positions = [p.strip() for p in args.positions.split(",") if p.strip() in POSITION_GROUPS]
    p8 = _load_step8()
    asm = sys.modules[p8.assemble_snapshot.__module__]

    # Build ONE snapshot with the feature computed for every group, then let the
    # step-8 gate decide per arm which heads may use it. Assembling per arm would
    # be 60s of wasted work per cell and risks the two gates drifting apart.
    print("Assembling snapshot (speed score computed for all groups)...", flush=True)
    asm_original = asm.SPEED_SCORE_POSITIONS
    asm.SPEED_SCORE_POSITIONS = frozenset(POSITION_GROUPS.keys())
    try:
        snapshot = p8.assemble_snapshot()
    finally:
        asm.SPEED_SCORE_POSITIONS = asm_original

    if FEATURE not in snapshot.columns:
        raise SystemExit(f"{FEATURE} absent after assembly — gate patch failed")
    cov = snapshot.groupby("position_group")[FEATURE].apply(lambda s: s.notna().mean())
    print("combine_speed_score coverage by group: "
          + ", ".join(f"{k} {v:.1%}" for k, v in cov.items()), flush=True)

    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    bridge_ids = set(labels.loc[labels["is_bridge"].fillna(False).astype(bool),
                                "player_id"].astype(str))

    base_params = dict(p8.LGBM_PARAMS)
    base_gate = frozenset(SPEED_SCORE_POSITIONS)
    rows = []

    for pos in positions:
        incumbent = pos in base_gate
        # `on` = feature available to this group's heads; `off` = withheld.
        # For an incumbent the interesting direction is removal, but the arms are
        # symmetric so one pass answers both.
        arms = {
            "off":     base_gate - {pos},
            "on":      base_gate | {pos},
            "placebo": base_gate | {pos} if incumbent else base_gate - {pos},
        }
        for target_col, model_name in HEADS:
            res = {}
            for arm, gate in arms.items():
                offset = PLACEBO_SEED_OFFSET if arm == "placebo" else 0
                p8.SPEED_SCORE_POSITIONS = gate
                vals, used = [], None
                for seed in range(args.seeds):
                    p8.LGBM_PARAMS = {**base_params, "random_state": 42 + offset + seed}
                    try:
                        _m, fcols, metrics, _q, oof = p8.train_model_for_group(
                            snapshot, labels, pos, target_col, model_name)
                    except Exception as exc:
                        logger.warning("%s/%s %s skipped: %s", pos, model_name, arm, exc)
                        continue
                    used = FEATURE in fcols
                    vals.append((_fold_rho_bridge(oof, bridge_ids), metrics.get("cv_mae")))
                p8.SPEED_SCORE_POSITIONS = base_gate
                p8.LGBM_PARAMS = dict(base_params)
                if vals:
                    res[arm] = {
                        "rho": float(np.nanmean([v[0] for v in vals])),
                        "mae": float(np.nanmean([v[1] for v in vals])),
                        "used": used,
                    }

            if "off" not in res or "on" not in res:
                continue
            row = {
                "pos": pos, "head": model_name, "incumbent": incumbent,
                "rho_off": res["off"]["rho"], "rho_on": res["on"]["rho"],
                "d_rho": res["on"]["rho"] - res["off"]["rho"],
                "mae_off": res["off"]["mae"], "mae_on": res["on"]["mae"],
                "feature_survived_filters": res["on"]["used"],
            }
            if "placebo" in res:
                anchor = "on" if incumbent else "off"
                row["d_rho_placebo"] = res["placebo"]["rho"] - res[anchor]["rho"]
            row["mae_pct"] = (row["mae_on"] - row["mae_off"]) / row["mae_off"] * 100
            rows.append(row)
            print("  %-5s/%-10s %s Δρ=%+.4f  (placebo %+.4f)  cv_mae %+.2f%%%s"
                  % (pos, model_name, "[incumbent]" if incumbent else "           ",
                     row["d_rho"], row.get("d_rho_placebo", float("nan")),
                     row["mae_pct"],
                     "" if row["feature_survived_filters"] else "  [FEATURE FILTERED OUT]"),
                  flush=True)

    if not rows:
        print("No heads evaluated.")
        return

    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / f"speed_score_positions_{args.seeds}seed.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print(f"combine_speed_score per-group A/B — {args.seeds} seeds, cached hyperparameters")
    print("=" * 78)
    for pos in positions:
        sub = df[df.pos == pos]
        if sub.empty:
            continue
        pl = sub["d_rho_placebo"].dropna()
        verdict = "—"
        if len(pl):
            beats = abs(sub.d_rho.mean()) > abs(pl.mean()) and sub.d_rho.mean() > 0
            verdict = "ADD (beats placebo)" if beats else "no (within noise)"
            if pos in base_gate:
                verdict = ("KEEP" if sub.d_rho.mean() > abs(pl.mean())
                           else "consider REMOVING (within noise)")
        print(f"\n{pos}{'  [currently enabled]' if pos in base_gate else ''}")
        print(f"  Δρ_bridge mean {sub.d_rho.mean():+.4f}  "
              f"(wins {(sub.d_rho > 0).sum()}/{len(sub)})   "
              f"placebo {pl.mean() if len(pl) else float('nan'):+.4f}   "
              f"cv_mae {sub.mae_pct.mean():+.2f}%")
        print(f"  verdict: {verdict}")
        print(sub[["head", "d_rho", "d_rho_placebo", "mae_pct"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print(f"\nSaved → {out}")
    print("An arm that does not beat its placebo is not a win — see "
          "the hasan off-board experiment for why this arm exists.")


if __name__ == "__main__":
    main()
