"""
hasan_rank off-board handling A/B, at CACHED hyperparameters.

RESULT (2026-07-24, 3 seeds, 33 heads): ALL THREE ARMS NULL. NOTHING SHIPPED.
Production still leaves hasan_rank NaN off-board. Δρ_bridge over the 31 heads
with a usable bridge cohort: binary −0.0037, sentinel_gated +0.0007,
sentinel_ungated −0.0029 — while the A/A `placebo` arm, which changed nothing but
the LightGBM seed, moved −0.0057 (−2.1 SE). Per-head seed noise is ±0.005 median
and ±0.047 at worst, so the largest single effect in the experiment
(DL/impact_wav under sentinel_gated, +0.045) is the size of pure noise. The
monotone-missing-direction hypothesis is answered in the negative, and measured
well enough not to reopen without a materially different design.

Two findings survive the null:
  - If board membership is ever re-encoded, GATE IT. sentinel_gated beat
    sentinel_ungated 15/31 vs 4/31 (5/8 vs 1/8 on monotone heads).
  - The dominant-value filter drops hasan features on all 6 K/P heads under any
    fill (0.6% board coverage → 99.4% single-value share). Their Δ=0.0000 rows
    are the filter, not indifference.

THE PROBLEM
-----------
`hasan_rank` (Hasan Consensus Big Board, 1–300) is the single feature carrying a
monotone −1 constraint on 8 heads (config.MONOTONE_ENABLED). It is NaN for every
player who isn't on the board, and LightGBM's missing-value handling under a
monotone constraint decides for itself which side of each split NaN falls on —
there is no guarantee it learns "unranked = worse", which is what the constraint
is meant to encode.

WHAT "NaN" ACTUALLY MEANS HERE (measured, not assumed)
------------------------------------------------------
The board only covers draft years 2014–2026, i.e. last_college_season_year
2013–2025. 19.5% of all training rows predate it entirely. Within the board era,
coverage of the FULL training cohort is only 4–10% per position (K/P: 0.6%) —
the earlier "~50% NaN" figure was measured among bridge players only, where
coverage is 62–82% (again K/P excepted, at 13–19%).

So a NaN is one of three very different things:
  (a) board era, made the NFL, unranked   → real signal: the board passed on him
  (b) board era, never made the NFL       → ~95% of rows; label is already 0
  (c) pre-2014                            → the board did not exist

Any sentinel fill that ignores (c) tells the model every pre-2014 player is worse
than the worst-ranked 2020 prospect. Under a monotone −1 constraint that is not a
small mistake — it forces two decades of players below the board's tail. The
`sentinel_ungated` arm below exists to measure exactly that failure, so the
gating decision rests on evidence rather than on the argument above.

ARMS (all four run through step 8's own `train_model_for_group`, so folds,
feature resolution, monotone constraints and the holdout split are production's)
  base              production today — raw rank, NaN off-board
  binary            + `hasan_on_board` ∈ {1, 0, NaN}: ranked / board-era-unranked
                    / pre-board-era. Additive and unconstrained — it cannot
                    disturb the monotone feature, and Boruta can attribute it.
  sentinel_gated    hasan_rank NaN → 400 (worse than worst rank 300) for
                    board-era rows ONLY; pre-2014 stays NaN.
  sentinel_ungated  hasan_rank NaN → 400 everywhere. The naive version.
  placebo           identical data to `base`, different LightGBM seeds — an A/A
                    arm whose delta is noise by construction. Any real arm has
                    to beat it to mean anything.

The exact sentinel value is immaterial: trees split on order, and under monotone
−1 only the ordering vs. rank 300 matters. Anything > 300 gives the same fit.

READING THE RESULT
------------------
Decide on RANKING, not MAE. cv_mae on a target that is ~90% zero-padded is
dominated by the model's ability to say "this player never played", which is not
what the board rank is for.

The load-bearing column is `cv_rho_bridge`: mean per-fold Spearman over the
tuning folds, restricted to players who actually reached the NFL. It is averaged
over 4 folds rather than read off the single holdout fold, which is why it is
usable where holdout Spearman is too noisy to trust (see the evaluation-
preferences note: holdout rho swings hard at this sample size). `cv_rho_all` and
`holdout_spearman` are reported alongside as secondary.

Every arm is run under N LightGBM seeds and averaged. Adding a column perturbs
colsample_bytree/subsample draws, so a single-seed delta partly measures which
features happened to get sampled — an unavoidable confound in any feature-
addition A/B, and the reason for --seeds 3.

The comparison is biased AGAINST the new arms: cached hyperparameters were tuned
on the `base` feature surface. A win here is meaningful; a small loss is not
disqualifying and would justify paying for `--retune-only 8,8.5`.

Watch `has_rank` in the output: the dominant-value filter drops any feature whose
single most common value covers >99% of non-null cohort rows. Under a sentinel
fill K and P sit at 99.4% sentinel, so hasan_rank is expected to be dropped
entirely for those heads — a mechanical side effect, not a modelling result.

Usage:
    python3 experiments/00n_hasan_offboard_ab.py
    python3 experiments/00n_hasan_offboard_ab.py --positions WR,OL,RB --seeds 1
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
from config import POSITION_GROUPS, PROCESSED_DIR

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

HEADS = [("role_target", "role"),
         ("impact_target", "impact"),
         ("impact_w_av_per_game", "impact_wav")]

SENTINEL = 400.0          # > max observed rank (300); value is immaterial (see docstring)
ARMS = ["base", "binary", "sentinel_gated", "sentinel_ungated"]

# A/A calibration arm: byte-identical data to `base`, different LightGBM seeds.
# Its Δρ_bridge is pure noise by construction, so it sets the floor a real arm's
# delta has to clear. Without it a ±0.005 mean delta is uninterpretable — it
# could be signal or it could be which features colsample_bytree happened to draw.
PLACEBO_ARM = "placebo"
PLACEBO_SEED_OFFSET = 100


def _load_step8():
    """Import pipeline/08_draft_models.py under a module name (leading digits
    make it non-importable normally)."""
    path = Path(__file__).resolve().parent.parent / "pipeline" / "08_draft_models.py"
    spec = importlib.util.spec_from_file_location("_p8_hasan_ab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _board_era(snapshot: pd.DataFrame) -> pd.Series:
    """Boolean mask: rows whose expected draft year falls inside the board's
    coverage. Expected draft year is last_college_season_year + 1, the same
    convention _snapshot_assembly uses for its year-sanity nulling."""
    hasan = pd.read_parquet(PROCESSED_DIR / "hasan_ranks.parquet")
    lo, hi = int(hasan["draft_year"].min()), int(hasan["draft_year"].max())
    expected = snapshot["last_college_season_year"] + 1
    logger.warning("Board coverage: draft years %d–%d", lo, hi)
    return expected.between(lo, hi)


def _apply_arm(snapshot: pd.DataFrame, arm: str, era: pd.Series) -> pd.DataFrame:
    """Return a snapshot copy with the arm's hasan encoding applied.

    Only the hasan columns are touched — every other column is shared by
    reference via the shallow copy, so this stays cheap on a 1,383-column frame.
    """
    snap = snapshot.copy(deep=False)
    rank = snapshot["hasan_rank"]

    if arm in ("base", PLACEBO_ARM):
        return snap
    if arm == "binary":
        on_board = pd.Series(np.nan, index=snapshot.index, dtype=float)
        on_board[era] = rank[era].notna().astype(float)
        snap["hasan_on_board"] = on_board
        return snap
    if arm == "sentinel_gated":
        snap["hasan_rank"] = rank.mask(era & rank.isna(), SENTINEL)
        return snap
    if arm == "sentinel_ungated":
        snap["hasan_rank"] = rank.fillna(SENTINEL)
        return snap
    raise ValueError(f"unknown arm: {arm}")


def _fold_rank_metrics(oof: pd.DataFrame, bridge_ids: set) -> dict:
    """Mean per-fold Spearman over the tuning folds, all rows and bridge-only.

    Per-fold then averaged (not pooled) because the folds have different year
    ranges and different label scales; pooling would let the between-fold level
    shift masquerade as within-fold ranking skill.
    """
    if oof.empty:
        return {"cv_rho_all": np.nan, "cv_rho_bridge": np.nan, "n_bridge_val": 0}

    is_bridge = oof["player_id"].isin(bridge_ids)
    rho_all, rho_bridge, n_bridge = [], [], 0

    for _, fold in oof.groupby(["val_start", "val_end"]):
        if fold["y"].nunique() > 1:
            r, _ = spearmanr(fold["pred"], fold["y"])
            if r is not None and not np.isnan(r):
                rho_all.append(float(r))
        sub = fold[is_bridge.loc[fold.index]]
        n_bridge += len(sub)
        # 10 is the same floor _fit_eval_fold uses to accept a validation fold.
        if len(sub) >= 10 and sub["y"].nunique() > 1:
            r, _ = spearmanr(sub["pred"], sub["y"])
            if r is not None and not np.isnan(r):
                rho_bridge.append(float(r))

    return {
        "cv_rho_all":    float(np.mean(rho_all)) if rho_all else np.nan,
        "cv_rho_bridge": float(np.mean(rho_bridge)) if rho_bridge else np.nan,
        "n_bridge_val":  int(n_bridge),
    }


def _run_head(p8, snap, labels, pos, target_col, model_name,
              bridge_ids: set) -> dict | None:
    try:
        model, feature_cols, metrics, _q, oof = p8.train_model_for_group(
            snap, labels, pos, target_col, model_name)
    except Exception as exc:                      # small heads legitimately fail
        logger.warning("%s/%s skipped: %s", pos, model_name, exc)
        return None
    out = {
        "cv_mae":           metrics.get("cv_mae"),
        "holdout_mae":      metrics.get("holdout_mae"),
        "holdout_spearman": metrics.get("holdout_spearman"),
        "has_rank":         "hasan_rank" in feature_cols,
        "has_on_board":     "hasan_on_board" in feature_cols,
        "n_features":       len(feature_cols),
    }
    out.update(_fold_rank_metrics(oof, bridge_ids))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default=None,
                    help="Comma-separated subset (default: all).")
    ap.add_argument("--heads", default=None,
                    help="Comma-separated subset of role,impact,impact_wav.")
    ap.add_argument("--seeds", type=int, default=3,
                    help="LightGBM seeds per arm; results are averaged (default 3).")
    ap.add_argument("--arms", default=",".join(ARMS + [PLACEBO_ARM]),
                    help=f"Comma-separated subset of {ARMS + [PLACEBO_ARM]}. "
                         f"'{PLACEBO_ARM}' is the A/A noise floor.")
    args = ap.parse_args()

    positions = ([p.strip() for p in args.positions.split(",")]
                 if args.positions else list(POSITION_GROUPS.keys()))
    heads = [h for h in HEADS
             if args.heads is None or h[1] in {s.strip() for s in args.heads.split(",")}]
    arms = [a.strip() for a in args.arms.split(",")]

    p8 = _load_step8()
    print("Assembling snapshot...", flush=True)
    snapshot = p8.assemble_snapshot()
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    bridge_ids = set(labels.loc[labels["is_bridge"].fillna(False).astype(bool),
                                "player_id"].astype(str))

    era = _board_era(snapshot)
    n_rank = snapshot["hasan_rank"].notna().sum()
    print(f"Snapshot {len(snapshot):,} rows | board-era {era.sum():,} "
          f"({era.mean():.1%}) | ranked {n_rank:,} ({n_rank / len(snapshot):.2%})",
          flush=True)

    arm_snaps = {a: _apply_arm(snapshot, a, era) for a in arms}
    baseline_params = dict(p8.LGBM_PARAMS)
    baseline_hasan = list(p8.HASAN_FEATURES)

    rows = []
    for pos in positions:
        for target_col, model_name in heads:
            monotone = (pos, model_name) in p8.MONOTONE_ENABLED
            per_arm: dict[str, list[dict]] = {a: [] for a in arms}

            for arm in arms:
                # The binary is additive; every other arm re-encodes the
                # existing column and leaves the feature list alone.
                p8.HASAN_FEATURES = (baseline_hasan + ["hasan_on_board"]
                                     if arm == "binary" else list(baseline_hasan))
                offset = PLACEBO_SEED_OFFSET if arm == PLACEBO_ARM else 0
                for seed in range(args.seeds):
                    p8.LGBM_PARAMS = {**baseline_params,
                                      "random_state": 42 + offset + seed}
                    r = _run_head(p8, arm_snaps[arm], labels, pos,
                                  target_col, model_name, bridge_ids)
                    if r:
                        per_arm[arm].append(r)
                p8.LGBM_PARAMS = dict(baseline_params)
                p8.HASAN_FEATURES = list(baseline_hasan)

            if not per_arm.get("base"):
                continue

            row = {"pos": pos, "head": model_name, "monotone": monotone}
            for arm in arms:
                runs = per_arm[arm]
                if not runs:
                    continue
                for m in ("cv_mae", "cv_rho_all", "cv_rho_bridge",
                          "holdout_spearman", "holdout_mae"):
                    row[f"{m}__{arm}"] = float(np.mean([r[m] for r in runs]))
                row[f"has_rank__{arm}"] = bool(runs[0]["has_rank"])
                if arm == "binary":
                    row["has_on_board"] = bool(runs[0]["has_on_board"])
            row["n_bridge_val"] = per_arm["base"][0]["n_bridge_val"]
            rows.append(row)

            msg = f"  {pos:5s}/{model_name:10s}{' [mono]' if monotone else '       '}"
            for arm in arms:
                if f"cv_rho_bridge__{arm}" in row:
                    msg += f"  {arm[:4]}ρ={row[f'cv_rho_bridge__{arm}']:+.3f}"
            print(msg, flush=True)

    if not rows:
        print("No heads evaluated.")
        return

    df = pd.DataFrame(rows)
    for arm in arms:
        if arm == "base":
            continue
        for m in ("cv_rho_bridge", "cv_rho_all", "holdout_spearman"):
            if f"{m}__{arm}" in df.columns:
                df[f"d_{m}__{arm}"] = df[f"{m}__{arm}"] - df[f"{m}__base"]
        if f"cv_mae__{arm}" in df.columns:
            df[f"d_cv_mae_pct__{arm}"] = (
                (df[f"cv_mae__{arm}"] - df["cv_mae__base"]) / df["cv_mae__base"] * 100)

    # Filename carries the arm set: a partial re-run (`--arms base,placebo`)
    # must not clobber the full run's results.
    tag = "_".join(a[:4] for a in arms if a != "base")
    out = PROCESSED_DIR / f"hasan_offboard_ab__{tag}_{args.seeds}seed.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print(f"hasan_rank off-board A/B — {args.seeds} seed(s), cached hyperparameters")
    print("=" * 78)

    def _block(sub: pd.DataFrame, title: str) -> None:
        if sub.empty:
            return
        print(f"\n{title}  (n={len(sub)} heads)")
        for arm in arms:
            if arm == "base" or f"d_cv_rho_bridge__{arm}" not in sub.columns:
                continue
            rb = sub[f"d_cv_rho_bridge__{arm}"].dropna()
            ra = sub[f"d_cv_rho_all__{arm}"].dropna()
            mae = sub[f"d_cv_mae_pct__{arm}"].dropna()
            print(f"  {arm:17s} "
                  f"Δρ_bridge {rb.mean():+.4f} (wins {(rb > 0).sum()}/{len(rb)})   "
                  f"Δρ_all {ra.mean():+.4f}   "
                  f"cv_mae {mae.mean():+.2f}%")

    _block(df, "ALL HEADS")
    _block(df[df["monotone"]], "MONOTONE heads (the hypothesis)")
    _block(df[~df["monotone"]], "UNCONSTRAINED heads")

    for arm in arms:
        col = f"has_rank__{arm}"
        if col in df.columns:
            lost = df.loc[~df[col], ["pos", "head"]]
            if len(lost):
                print(f"\n  NOTE [{arm}]: hasan_rank dropped by the dominant-value "
                      f"filter on {len(lost)} head(s): "
                      + ", ".join(f"{r.pos}/{r.head}" for r in lost.itertuples()))
    if "has_on_board" in df.columns:
        lost = df.loc[~df["has_on_board"], ["pos", "head"]]
        if len(lost):
            print(f"  NOTE [binary]: hasan_on_board dropped on {len(lost)} head(s): "
                  + ", ".join(f"{r.pos}/{r.head}" for r in lost.itertuples()))

    print("\nPer-head Δρ_bridge (load-bearing metric):")
    show = ["pos", "head", "monotone", "n_bridge_val", "cv_rho_bridge__base"]
    show += [f"d_cv_rho_bridge__{a}" for a in arms if a != "base"
             and f"d_cv_rho_bridge__{a}" in df.columns]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print(f"\nSaved → {out}")
    print("Ranking (Δρ_bridge) decides this; cv_mae is dominated by the "
          "zero-padded non-NFL mass.")


if __name__ == "__main__":
    main()
