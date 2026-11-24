"""
Hurdle decomposition, retried with its two documented caveats fixed.

Why retry. The 2026-07-21 test rejected the two-stage hurdle (mixture won 6/8;
verdict commit aa32a3b: "the zero-padded non-bridge rows are load-bearing even
for within-bridge ranking; the conditional heads overfit their few hundred
rows"). But `_run_hurdle_ab`'s own docstring records two caveats that
LOWER-BOUND the hurdle:

    1. the conditional head borrows the MIXTURE head's tuned params
    2. the gate uses raw clipped role predictions, with no isotonic

Both matter more than they look. (1) is the sharper one: per-head Optuna optima
in this repo are bimodal — small-N heads want very different params from
signal-rich ones — so handing a few-hundred-row conditional head parameters
tuned on ~10k rows is close to a worst case, and "the conditional heads overfit"
may be a statement about the borrowed params rather than about the
decomposition. (2) matters because the whole hurdle rests on the gate being a
calibrated probability; an uncalibrated gate multiplies the magnitude by a
mis-scaled number.

This also matters more now than it did in July. Everything rejected on
2026-07-26 — the label gate, the early-cohort filter, contract imputation, and
UDFA fill — failed the same way: the zero-padded mixture teaches "weak college
profile -> 0", and any population that contradicts it raises MAE against a
zero-dominated target. The hurdle is the one structure that lets "did they make
it" and "how good were they" be learned separately instead of fighting inside
one target.

Arms (per position, per expanding-window tuning fold, holdout untouched):

    mixture          single-stage regression on the zero-padded target (current
                     production, the incumbent)
    hurdle_orig      raw clipped gate + borrowed mixture params — REPRODUCES the
                     2026-07-21 test, as a control on this harness
    hurdle_fixed     isotonic-calibrated gate + small-N conditional params
    AA_placebo       mixture refit under a different seed — the noise floor

Comparing hurdle_orig against hurdle_fixed isolates whether the caveats were
load-bearing. All arms are scored against the IDENTICAL zero-padded target on
IDENTICAL validation rows, so MAE is directly comparable.

Gate calibration is honest: the role head is fit on a core slice of the training
window, the isotonic map is fit on the most recent training seasons peeled off
as a calibration slice (never the scoring fold), then the role head is refit on
the full training window and the frozen map applied. Same shape as
`_fit_fold_honest`'s early-stopping split.

Read-only. Writes one JSON to experiments/.
"""

import argparse
import importlib.util
import json
import logging
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, ROOT_DIR, LGBM_PARAMS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = ROOT_DIR / "experiments"

# Conditional-head parameters for a few-hundred-row bridge-only sample. The
# borrowed mixture params are tuned on ~10k rows and are far too permissive
# here; these trade capacity for regularisation (shallow, few leaves, large
# leaf minimum, strong L1/L2) in the style this repo's small-N heads (K, P)
# converge to under Optuna.
SMALL_N_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 7,
    "min_child_samples": 15,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "verbose": -1,
}


def _load_p8():
    mod_path = ROOT_DIR / "pipeline" / "08_draft_models.py"
    spec = importlib.util.spec_from_file_location("p8_hurdle", mod_path)
    p8 = types.ModuleType("p8_hurdle")
    p8.__file__ = str(mod_path)
    sys.modules["p8_hurdle"] = p8
    spec.loader.exec_module(p8)
    return p8


def _calibrated_gate(p8, lgb, role_df, role_cols, params, pos, train_end,
                     X_val):
    """Fit the role head and an honest isotonic map, return calibrated gate
    predictions on the validation rows.

    The isotonic map is fit on the most recent TRAINING seasons, peeled off so
    the scoring fold never participates in calibration.
    """
    yr = role_df["last_college_season_year"].fillna(0)
    tr = role_df[yr <= train_end]
    if len(tr) < 60:
        return None
    tr_yrs = tr["last_college_season_year"].fillna(0).values
    uniq = np.unique(tr_yrs)[::-1]
    cal_mask = np.zeros(len(tr), dtype=bool)
    for y in uniq[:-1]:
        cal_mask |= tr_yrs == y
        if cal_mask.sum() >= 30:
            break
    if cal_mask.sum() < 30 or (~cal_mask).sum() < 30:
        return None

    X_tr = tr[role_cols].to_numpy(dtype=float, na_value=np.nan)
    y_tr = tr["role_target"].fillna(0.0).values

    mk = lambda: lgb.LGBMRegressor(**{
        **params, **p8._build_monotone_kwargs(role_cols, pos, "role")})
    core, _ = p8._fit_fold_honest(mk, X_tr[~cal_mask], y_tr[~cal_mask],
                                  tr_yrs[~cal_mask])
    cal_pred = np.clip(core.predict(X_tr[cal_mask]), 0.0, 1.0)
    if len(np.unique(cal_pred)) < 5:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(cal_pred, y_tr[cal_mask])

    full, _ = p8._fit_fold_honest(mk, X_tr, y_tr, tr_yrs)
    return np.clip(iso.predict(np.clip(full.predict(X_val), 0.0, 1.0)), 0.0, 1.0)


def run(target_col: str, model_name: str) -> list:
    import lightgbm as lgb
    p8 = _load_p8()

    snapshot = p8.assemble_snapshot(include_derived=True, logger=logger)
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    tuning_folds, _ = p8._split_holdout(p8._expanding_window_folds())

    logger.info("── hurdle retry: %s head ──", model_name)
    results = []
    for pos in p8.POSITION_GROUPS.keys():
        role_cols, role_df = p8._resolve_fit_features(
            snapshot, labels, pos, "role_target", "role")
        cols, df = p8._resolve_fit_features(
            snapshot, labels, pos, target_col, model_name)
        if not role_cols or not cols or df is None or len(df) < 50:
            continue

        role_params = {**LGBM_PARAMS, **(p8._load_tuned_params(pos, "role") or {})}
        mix_params = {**LGBM_PARAMS, **(p8._load_tuned_params(pos, model_name) or {})}
        cond_df = df[df["is_bridge"].astype(bool)]
        if len(cond_df) < 60:
            continue

        yr = df["last_college_season_year"].fillna(0)
        acc = {k: {"mae": [], "rho": []} for k in
               ["mixture", "hurdle_orig", "hurdle_fixed", "AA_placebo"]}

        for train_end, val_start, val_end in tuning_folds:
            va = df[(yr >= val_start) & (yr <= val_end)]
            tr = df[yr <= train_end]
            if len(va) < 10 or len(tr) < 30:
                continue
            y_val = va[target_col].fillna(0.0).values
            X_val = va[cols].to_numpy(dtype=float, na_value=np.nan)
            X_val_role = va[role_cols].to_numpy(dtype=float, na_value=np.nan)
            bm = va["is_bridge"].astype(bool).values

            def _fit(frame, fcols, params, mname, seed=None):
                sub = frame[frame["last_college_season_year"].fillna(0) <= train_end]
                if len(sub) < 30:
                    return None
                pp = dict(params)
                if seed is not None:
                    pp["random_state"] = seed
                mk = lambda: lgb.LGBMRegressor(**{
                    **pp, **p8._build_monotone_kwargs(fcols, pos, mname)})
                m, _ = p8._fit_fold_honest(
                    mk, sub[fcols].to_numpy(dtype=float, na_value=np.nan),
                    sub[target_col].fillna(0.0).values,
                    sub["last_college_season_year"].fillna(0).values)
                return m

            m_mix = _fit(df, cols, mix_params, model_name)
            m_plc = _fit(df, cols, mix_params, model_name, seed=1337)
            m_cond_borrow = _fit(cond_df, cols, mix_params, model_name)
            m_cond_small = _fit(cond_df, cols, SMALL_N_PARAMS, model_name)
            if any(m is None for m in [m_mix, m_plc, m_cond_borrow, m_cond_small]):
                continue

            # Raw gate (the original test's construction).
            m_role = _fit(role_df.assign(**{target_col: role_df["role_target"]}),
                          role_cols, role_params, "role")
            if m_role is None:
                continue
            g_raw = np.clip(m_role.predict(X_val_role), 0.0, 1.0)
            g_cal = _calibrated_gate(p8, lgb, role_df, role_cols, role_params,
                                     pos, train_end, X_val_role)
            if g_cal is None:
                continue

            preds = {
                "mixture":      m_mix.predict(X_val),
                "AA_placebo":   m_plc.predict(X_val),
                "hurdle_orig":  g_raw * np.clip(m_cond_borrow.predict(X_val), 0, None),
                "hurdle_fixed": g_cal * np.clip(m_cond_small.predict(X_val), 0, None),
            }
            for k, p in preds.items():
                acc[k]["mae"].append(float(np.mean(np.abs(p - y_val))))
                if bm.sum() >= 10 and np.std(y_val[bm]) > 1e-12:
                    r = spearmanr(p[bm], y_val[bm]).statistic
                    if np.isfinite(r):
                        acc[k]["rho"].append(float(r))

        if not acc["mixture"]["mae"]:
            continue
        agg = {k: {"mae": float(np.mean(v["mae"])),
                   "rho": float(np.mean(v["rho"])) if v["rho"] else float("nan")}
               for k, v in acc.items()}
        base = agg["mixture"]
        noise = abs(agg["AA_placebo"]["mae"] - base["mae"])
        rec = {"head": model_name, "pos": pos,
               "n_folds": len(acc["mixture"]["mae"]),
               "mae_mixture": base["mae"], "rho_mixture": base["rho"],
               "placebo_mae_noise": noise}
        parts = []
        for arm in ["hurdle_orig", "hurdle_fixed"]:
            d = (agg[arm]["mae"] - base["mae"]) / base["mae"] * 100
            rec[f"mae_{arm}"] = agg[arm]["mae"]
            rec[f"mae_delta_pct_{arm}"] = d
            rec[f"rho_{arm}"] = agg[arm]["rho"]
            rec[f"beats_noise_{arm}"] = bool(
                abs(agg[arm]["mae"] - base["mae"]) > noise)
            parts.append("%s %+.1f%%%s rho=%.3f" % (
                arm.replace("hurdle_", ""), d,
                "*" if rec[f"beats_noise_{arm}"] else " ", agg[arm]["rho"]))
        results.append(rec)
        logger.info("  %-5s mix mae=%.5f rho=%.3f | %s",
                    pos, base["mae"], base["rho"], " | ".join(parts))

    for arm in ["hurdle_orig", "hurdle_fixed"]:
        key = f"mae_delta_pct_{arm}"
        rows = [r for r in results if key in r]
        if not rows:
            continue
        wins = sum(1 for r in rows if r[key] < 0)
        sig_w = sum(1 for r in rows if r[key] < 0 and r[f"beats_noise_{arm}"])
        sig_l = sum(1 for r in rows if r[key] > 0 and r[f"beats_noise_{arm}"])
        rho_w = sum(1 for r in rows
                    if np.isfinite(r[f"rho_{arm}"])
                    and r[f"rho_{arm}"] > r["rho_mixture"])
        logger.info("[%s / %s] MAE beats mixture %d/%d (%d sig wins, %d sig "
                    "losses), mean %+.2f%% | bridge rho beats %d/%d",
                    arm, model_name, wins, len(rows), sig_w, sig_l,
                    float(np.mean([r[key] for r in rows])), rho_w, len(rows))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="impact_wav,impact",
                    help="comma-separated: impact_wav and/or impact")
    args = ap.parse_args()

    target_of = {"impact_wav": "impact_w_av_per_game", "impact": "impact_target"}
    allr = []
    for h in [x.strip() for x in args.heads.split(",") if x.strip()]:
        allr += run(target_of[h], h)

    out = EXPERIMENTS_DIR / f"hurdle_retry_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(allr, indent=2, default=float))
    logger.info("→ %s", out)


if __name__ == "__main__":
    main()
