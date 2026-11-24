"""
Does the headline impact_score gain from using BOTH outcome heads?

Today `impact_score = impact_apy_score` (pipeline/10_output.py:367, :600) — the
contract head alone. The AV head (`impact_wav`) is trained, calibrated, and
then contributes nothing to the headline score; it appears only in
`impact_blend_pct`, a secondary within-position percentile view that the board
does not sort by.

That matters because the contract label is unreliable in places (OverTheCap's
coverage of small deals improved enormously over time — a third of 2004-06
bridge players with career w_AV >= 30 are labelled "no second contract"),
while career AV comes from PFR and does not share that failure mode. Two
imperfect, independently-sourced views of the same latent quantity should beat
one.

Method. Pure post-processing of step 8's persisted tuning-fold OOF predictions
— nothing is refit, so this is cheap and cannot leak: those predictions were
already produced by the honest-early-stopping fold procedure.

  * The two heads live on different scales (APY%cap ~0.01 vs w_AV/game ~0.2),
    so raw averaging is meaningless. Blend z-scores, per fold and position,
    exactly as `08b._run_blend_ab` does for its scale-incompatible case.
  * Score by Spearman against the realised second-contract APY (`impact_target`)
    on BRIDGE rows. That is the quantity impact_score claims to project and the
    one the dollar columns are built from, so it is the honest target — and it
    deliberately favours the incumbent (w=1.0 is the pure APY head).
  * Weight chosen LEAVE-ONE-FOLD-OUT so the evaluated fold never picks its own
    weight, matching the repo's existing blend-selection discipline.

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
from config import PROCESSED_DIR, ROOT_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 2)   # weight on the APY head


def _z(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 1e-12 else 1.0)


def _load() -> pd.DataFrame:
    oof = pd.read_parquet(PROCESSED_DIR / "draft_models" / "oof_predictions.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    bridge = (labels[["player_id", "is_bridge"]]
              .drop_duplicates("player_id"))

    apy = (oof[oof["model_name"] == "impact"]
           .rename(columns={"pred": "pred_apy", "y": "y_apy"})
           [["player_id", "pos_group", "val_start", "val_end",
             "pred_apy", "y_apy"]])
    wav = (oof[oof["model_name"] == "impact_wav"]
           .rename(columns={"pred": "pred_wav"})
           [["player_id", "pos_group", "val_start", "val_end", "pred_wav"]])

    df = apy.merge(wav, on=["player_id", "pos_group", "val_start", "val_end"],
                   how="inner").merge(bridge, on="player_id", how="left")
    df["is_bridge"] = df["is_bridge"].fillna(False)
    return df


def _fold_rho(rows: pd.DataFrame, w: float) -> float:
    """Spearman of the blended score against realised APY, bridge rows only."""
    b = rows[rows["is_bridge"].astype(bool)]
    if len(b) < 10 or np.std(b["y_apy"]) < 1e-12:
        return float("nan")
    blend = w * _z(b["pred_apy"].values) + (1 - w) * _z(b["pred_wav"].values)
    r = spearmanr(blend, b["y_apy"].values).statistic
    return float(r) if np.isfinite(r) else float("nan")


def main():
    df = _load()
    logger.info("OOF rows with both heads: %d (%d bridge) across %d folds",
                len(df), int(df["is_bridge"].astype(bool).sum()),
                df.groupby(["val_start", "val_end"]).ngroups)

    results = []
    for pos, g in df.groupby("pos_group"):
        folds = [f for _, f in g.groupby(["val_start", "val_end"])]
        folds = [f for f in folds
                 if np.isfinite(_fold_rho(f, 1.0))
                 and np.isfinite(_fold_rho(f, 0.0))]
        if len(folds) < 3:
            logger.info("  %-5s skipped (%d usable folds)", pos, len(folds))
            continue

        per_fold = []
        for i, f in enumerate(folds):
            others = [x for j, x in enumerate(folds) if j != i]
            # LOFO: pick w on the OTHER folds, evaluate on this one.
            scores = {w: np.nanmean([_fold_rho(o, w) for o in others])
                      for w in WEIGHTS}
            w_star = max(scores, key=lambda w: scores[w])
            per_fold.append((_fold_rho(f, 1.0),      # incumbent: APY only
                             _fold_rho(f, 0.0),      # AV head only
                             _fold_rho(f, w_star),   # LOFO blend
                             w_star))
        per_fold = [p for p in per_fold if np.isfinite(p[0]) and np.isfinite(p[2])]
        if not per_fold:
            continue

        rho_apy = float(np.mean([p[0] for p in per_fold]))
        rho_wav = float(np.mean([p[1] for p in per_fold]))
        rho_blend = float(np.mean([p[2] for p in per_fold]))
        ws = [float(p[3]) for p in per_fold]
        rec = {"pos": pos, "n_folds": len(per_fold),
               "rho_apy_only": rho_apy, "rho_wav_only": rho_wav,
               "rho_blend_lofo": rho_blend,
               "delta_vs_incumbent": rho_blend - rho_apy,
               "lofo_weights_on_apy": ws,
               "median_w_apy": float(np.median(ws))}
        results.append(rec)
        logger.info("  %-5s apy_only=%.4f  wav_only=%.4f  blend=%.4f "
                    "(%+.4f)  w_apy=%s",
                    pos, rho_apy, rho_wav, rho_blend,
                    rec["delta_vs_incumbent"], ws)

    out = EXPERIMENTS_DIR / f"impact_head_blend_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    wins = sum(1 for r in results if r["delta_vs_incumbent"] > 0)
    mean_d = float(np.mean([r["delta_vs_incumbent"] for r in results])) if results else float("nan")
    logger.info("Blend beats APY-only on %d/%d positions (mean Δρ %+.4f)",
                wins, len(results), mean_d)
    pure = sum(1 for r in results if r["median_w_apy"] >= 0.999)
    logger.info("Positions whose LOFO weight stays at pure-APY: %d/%d",
                pure, len(results))
    logger.info("→ %s", out)


# ── Holdout referendum ────────────────────────────────────────────────────────
#
# Tuning-fold evidence alone has never been enough for a blend decision in this
# repo (see the 2026-07-21 global-blend adoption, which gated on tuning folds
# then held a pre-registered holdout referendum). Weights are FROZEN from the
# tuning folds above, then evaluated once on the untouched latest fold with
# both heads refit — the holdout never participates in weight selection.

def run_holdout(weights_by_pos: dict) -> None:
    import importlib.util
    import types
    import lightgbm as lgb

    mod_path = ROOT_DIR / "pipeline" / "08_draft_models.py"
    spec = importlib.util.spec_from_file_location("p8_blend", mod_path)
    p8 = types.ModuleType("p8_blend")
    p8.__file__ = str(mod_path)
    sys.modules["p8_blend"] = p8
    spec.loader.exec_module(p8)

    _, holdout = p8._split_holdout(p8._expanding_window_folds())
    if holdout is None:
        logger.error("No holdout fold available")
        return
    train_end, val_start, val_end = holdout
    logger.info("HOLDOUT fold: train<=%d, val %d-%d", train_end, val_start, val_end)

    snapshot = p8.assemble_snapshot(include_derived=True, logger=logger)
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")

    results = []
    for pos, w_apy in sorted(weights_by_pos.items()):
        preds = {}
        for target_col, mname in [("impact_target", "impact"),
                                  ("impact_w_av_per_game", "impact_wav")]:
            cols, df = p8._resolve_fit_features(snapshot, labels, pos,
                                                target_col, mname)
            if not cols or df is None:
                preds = {}
                break
            yr = df["last_college_season_year"].fillna(0)
            tr = df[yr <= train_end]
            va = df[(yr >= val_start) & (yr <= val_end)]
            if len(tr) < 30 or len(va) < 10:
                preds = {}
                break
            params = {**p8.LGBM_PARAMS,
                      **(p8._load_tuned_params(pos, mname) or {})}
            mk = lambda: lgb.LGBMRegressor(**{
                **params, **p8._build_monotone_kwargs(cols, pos, mname)})
            m, _ = p8._fit_fold_honest(
                mk, tr[cols].to_numpy(dtype=float, na_value=np.nan),
                tr[target_col].fillna(0.0).values,
                tr["last_college_season_year"].fillna(0).values)
            preds[mname] = (m.predict(va[cols].to_numpy(dtype=float,
                                                        na_value=np.nan)),
                            va)
        if not preds or len(preds) < 2:
            logger.info("  %-5s skipped (insufficient holdout data)", pos)
            continue

        # Align the two heads on their common validation players.
        va_a = preds["impact"][1][["player_id", "is_bridge", "impact_target"]].copy()
        va_a["p_apy"] = preds["impact"][0]
        va_w = preds["impact_wav"][1][["player_id"]].copy()
        va_w["p_wav"] = preds["impact_wav"][0]
        j = va_a.merge(va_w, on="player_id", how="inner")
        b = j[j["is_bridge"].astype(bool)]
        if len(b) < 10 or np.std(b["impact_target"]) < 1e-12:
            logger.info("  %-5s skipped (too few bridge holdout rows)", pos)
            continue

        y = b["impact_target"].values
        r_apy = spearmanr(_z(b["p_apy"].values), y).statistic
        blend = w_apy * _z(b["p_apy"].values) + (1 - w_apy) * _z(b["p_wav"].values)
        r_blend = spearmanr(blend, y).statistic
        rec = {"pos": pos, "w_apy": w_apy, "n_bridge": int(len(b)),
               "holdout_rho_apy_only": float(r_apy),
               "holdout_rho_blend": float(r_blend),
               "holdout_delta": float(r_blend - r_apy)}
        results.append(rec)
        logger.info("  %-5s w_apy=%.2f  n=%3d  apy_only=%.4f  blend=%.4f (%+.4f)",
                    pos, w_apy, len(b), r_apy, r_blend, r_blend - r_apy)

    out = EXPERIMENTS_DIR / f"impact_head_blend_holdout_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    wins = sum(1 for r in results if r["holdout_delta"] > 0)
    logger.info("HOLDOUT: blend beats APY-only on %d/%d positions (mean %+.4f)",
                wins, len(results),
                float(np.mean([r["holdout_delta"] for r in results])) if results else float("nan"))
    logger.info("→ %s", out)


if __name__ == "__main__":
    if "--holdout" in sys.argv:
        # Weights frozen from the most recent tuning-fold run — the holdout
        # never participates in choosing them.
        path = sorted(EXPERIMENTS_DIR.glob("impact_head_blend_2*.json"))[-1]
        tuning = json.load(open(path))
        logger.info("Freezing LOFO-median weights from %s", path.name)
        run_holdout({r["pos"]: r["median_w_apy"] for r in tuning})
    else:
        main()
