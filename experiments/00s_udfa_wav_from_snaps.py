"""
Bring UDFAs into the impact_wav head by imputing career AV from snap counts.

The hole. `load_udfa_players` hardcodes `career_w_av = 0.0, nfl_games = 0`
(src/scrapers/nfl_data.py:483) because `import_players()` carries no AV. Step 7
then computes `impact_w_av_per_game = career_w_av / games` only when
`nfl_games >= 4`, so every one of the 1,700 UDFA bridge rows lands on NaN and
is dropped from the impact_wav head. Those players ARE present in the role and
APY heads (96% carry a real contract label), so the gap is specific to the AV
head — and it is exactly the population that distinguishes "undrafted but
productive" from "undrafted and washed out". 1,292 of them have >= 4 NFL games,
so fixing this adds ~26% to that head's bridge training rows.

Why impute rather than scrape. AV lives on Pro-Football-Reference player pages;
nflverse exposes it only through `import_draft_picks()`, which is drafted-only
by construction. Scraping ~1,700 pages under PFR's rate limits is slow and
brittle. But snap counts are already on disk, league-wide, 2016+ — and they
predict AV well:

    drafted 2016+, >=4 games, 5-fold CV, snap features -> target
        career w_AV (level)   rho 0.948   R2 0.835
        AV PER GAME           rho 0.872   R2 0.714      <- the quantity needed
        (games alone vs AV/game: rho 0.460 — so this is not just the games term)

Compare the 0.658 AV->APY relationship that failed in 00r: this is a much
tighter mapping, and it fills a MISSING value rather than overwriting an
observed one.

Evaluation design. UDFA classes are 2016+, so UDFAs land in both the training
AND validation windows of the later folds. Adding them therefore changes the
validation population, which would make a naive arm-vs-arm comparison
incoherent. So every arm is scored on the COMMON population — the rows both
arms possess (drafted bridge + non-bridge) — while only the TRAINING rows
differ. That isolates the question actually being asked: does adding undrafted
players to the training set improve prediction on everyone else?

Read-only. Writes a variant labels file and a JSON.
"""

import argparse
import json
import logging
import sys
import types
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, RAW_DIR, ROOT_DIR, LGBM_PARAMS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
SNAP_FEATURES = ["games", "seasons", "off_max", "def_max", "off_mean",
                 "def_mean", "st_mean", "off_sum", "def_sum", "st_sum"]
MIN_GAMES = 4          # matches step 7's rule for a meaningful per-game rate


def _snap_features() -> pd.DataFrame:
    s = pd.read_parquet(RAW_DIR / "snap_counts.parquet")
    return s.groupby("pfr_player_id").agg(
        games=("n_games", "sum"), seasons=("season", "nunique"),
        off_max=("offense_pct", "max"), def_max=("defense_pct", "max"),
        off_mean=("offense_pct", "mean"), def_mean=("defense_pct", "mean"),
        st_mean=("st_pct", "mean"), off_sum=("offense_pct", "sum"),
        def_sum=("defense_pct", "sum"), st_sum=("st_pct", "sum"),
    ).reset_index()


N_STRATA = 20          # prediction strata for the quantile-matched variant


def _quantile_match(p_tr: np.ndarray, y_tr: np.ndarray,
                    p_new: np.ndarray) -> np.ndarray:
    """Conditional quantile matching — spread-preserving imputation.

    A conditional-mean prediction shrinks variance by construction (with
    R2 ~ 0.71 the imputed spread came out at sd 0.079 against an observed
    0.164), so imputed UDFAs looked far more alike than they really are.

    Instead: stratify the DRAFTED players by their out-of-fold prediction,
    then for each stratum map the new rows' within-stratum ranks onto the
    ORDER STATISTICS of the observed values in that stratum. Level stays
    correct (a UDFA is matched against drafted players the model scores
    similarly), while spread is inherited from real observations rather than
    smoothed away. Deterministic — no sampling, so no seed dependence.
    """
    edges = np.quantile(p_tr, np.linspace(0, 1, N_STRATA + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    s_tr = np.clip(np.digitize(p_tr, edges[1:-1]), 0, len(edges) - 2)
    s_new = np.clip(np.digitize(p_new, edges[1:-1]), 0, len(edges) - 2)

    out = np.empty(len(p_new), dtype=float)
    for st in np.unique(s_new):
        obs = np.sort(y_tr[s_tr == st])
        idx_new = np.where(s_new == st)[0]
        if len(obs) < 5:
            # Too thin to define a within-stratum distribution: fall back to
            # the global observed distribution at the same relative rank.
            obs = np.sort(y_tr)
        order = np.argsort(np.argsort(p_new[idx_new]))       # within-stratum rank
        q = (order + 0.5) / len(idx_new)
        out[idx_new] = np.quantile(obs, q)
    return out


def build_udfa_wav(method: str = "mean") -> tuple[pd.DataFrame, dict]:
    """Fit AV/game on DRAFTED players inside the snap era, apply to UDFAs."""
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    feat = _snap_features()

    b = bridge[["pfr_id", "is_udfa", "career_w_av", "nfl_games",
                "draft_year"]].drop_duplicates("pfr_id")
    m = b.merge(feat, left_on="pfr_id", right_on="pfr_player_id", how="left")

    # Training set: drafted players whose whole career sits inside the snap era
    # (2016+), so the features describe the same span the AV does.
    tr = m[(m["is_udfa"] != True) & (m["draft_year"] >= 2016)  # noqa: E712
           & m["career_w_av"].notna() & (m["nfl_games"] >= MIN_GAMES)
           & m["games"].notna()].copy()
    tr["av_per_game"] = tr["career_w_av"] / tr["nfl_games"].clip(lower=1)

    X_tr = tr[SNAP_FEATURES].fillna(0).values
    y_tr = tr["av_per_game"].values
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X_tr, y_tr)

    # Apply to UDFAs with enough NFL games to make a per-game rate meaningful.
    ud = m[(m["is_udfa"] == True) & m["games"].notna()  # noqa: E712
           & (m["games"] >= MIN_GAMES)].copy()
    p_new = model.predict(ud[SNAP_FEATURES].fillna(0).values)

    if method == "quantile":
        # Out-of-fold predictions for the donors, so the strata reflect
        # honest model behaviour rather than in-sample fit.
        p_tr = cross_val_predict(GradientBoostingRegressor(random_state=0),
                                 X_tr, y_tr, cv=5)
        ud["imputed_wav"] = np.clip(_quantile_match(p_tr, y_tr, p_new), 0.0, None)
    else:
        ud["imputed_wav"] = np.clip(p_new, 0.0, None)

    out = labels.copy()
    imp = dict(zip(ud["pfr_id"], ud["imputed_wav"]))
    fill = out["pfr_id"].map(imp)
    # Only fill where the column is currently NaN — never overwrite an observed
    # value.
    mask = out["impact_w_av_per_game"].isna() & fill.notna()
    out.loc[mask, "impact_w_av_per_game"] = fill[mask]

    stats = {
        "method": method,
        "n_train_drafted": int(len(tr)),
        "n_udfa_imputed": int(mask.sum()),
        "imputed_mean": float(out.loc[mask, "impact_w_av_per_game"].mean()),
        "drafted_observed_mean": float(tr["av_per_game"].mean()),
        "imputed_sd": float(out.loc[mask, "impact_w_av_per_game"].std()),
        "drafted_observed_sd": float(tr["av_per_game"].std()),
    }
    return out, stats


def run_ab(variant_path: Path, seeds=(42, 1337)) -> None:
    """Train the wav head on baseline vs UDFA-augmented labels, scored on the
    COMMON validation population (rows both arms possess)."""
    import lightgbm as lgb

    mod_path = ROOT_DIR / "pipeline" / "08_draft_models.py"
    spec = importlib.util.spec_from_file_location("p8_udfa", mod_path)
    p8 = types.ModuleType("p8_udfa")
    p8.__file__ = str(mod_path)
    sys.modules["p8_udfa"] = p8
    spec.loader.exec_module(p8)

    snapshot = p8.assemble_snapshot(include_derived=True, logger=logger)
    base = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    aug = pd.read_parquet(variant_path)
    tuning_folds, _ = p8._split_holdout(p8._expanding_window_folds())

    TARGET, MODEL = "impact_w_av_per_game", "impact_wav"
    results = []
    for pos in p8.POSITION_GROUPS.keys():
        cols_b, df_b = p8._resolve_fit_features(snapshot, base, pos, TARGET, MODEL)
        cols_a, df_a = p8._resolve_fit_features(snapshot, aug, pos, TARGET, MODEL)
        if not cols_b or df_b is None or df_a is None or len(df_b) < 50:
            continue
        # Common validation population: players the BASELINE arm has, i.e.
        # excluding the newly-added UDFA rows. Both arms are scored on exactly
        # these rows; only their training sets differ.
        common_ids = set(df_b["player_id"])

        arms = {}
        for arm, frame, cols, seed in [("baseline", df_b, cols_b, seeds[0]),
                                       ("udfa_wav", df_a, cols_a, seeds[0]),
                                       ("AA_placebo", df_b, cols_b, seeds[1])]:
            params = {**LGBM_PARAMS,
                      **(p8._load_tuned_params(pos, MODEL) or {}),
                      "random_state": seed}
            yr = frame["last_college_season_year"].fillna(0)
            maes, rhos = [], []
            for train_end, val_start, val_end in tuning_folds:
                tr = frame[yr <= train_end]
                va = frame[(yr >= val_start) & (yr <= val_end)
                           & frame["player_id"].isin(common_ids)]
                if len(tr) < 30 or len(va) < 10:
                    continue
                mk = lambda: lgb.LGBMRegressor(**{
                    **params, **p8._build_monotone_kwargs(cols, pos, MODEL)})
                mdl, _ = p8._fit_fold_honest(
                    mk, tr[cols].to_numpy(dtype=float, na_value=np.nan),
                    tr[TARGET].fillna(0.0).values,
                    tr["last_college_season_year"].fillna(0).values)
                pred = mdl.predict(va[cols].to_numpy(dtype=float, na_value=np.nan))
                y = va[TARGET].fillna(0.0).values
                maes.append(float(np.mean(np.abs(pred - y))))
                bm = va["is_bridge"].astype(bool).values
                if bm.sum() >= 10 and np.std(y[bm]) > 1e-12:
                    r = spearmanr(pred[bm], y[bm]).statistic
                    if np.isfinite(r):
                        rhos.append(float(r))
            if not maes:
                arms = {}
                break
            arms[arm] = {"mae": float(np.mean(maes)),
                         "rho": float(np.mean(rhos)) if rhos else float("nan"),
                         "n_folds": len(maes)}
        if not arms or "baseline" not in arms or "udfa_wav" not in arms:
            continue

        bse, new = arms["baseline"], arms["udfa_wav"]
        noise = abs(arms["AA_placebo"]["mae"] - bse["mae"]) if "AA_placebo" in arms else float("nan")
        d_mae = (new["mae"] - bse["mae"]) / bse["mae"] * 100
        rec = {"pos": pos, "n_folds": bse["n_folds"],
               "mae_baseline": bse["mae"], "mae_udfa": new["mae"],
               "mae_delta_pct": d_mae, "placebo_mae_noise": noise,
               "beats_noise": bool(np.isfinite(noise)
                                   and abs(new["mae"] - bse["mae"]) > noise),
               "rho_baseline": bse["rho"], "rho_udfa": new["rho"],
               "rho_delta": new["rho"] - bse["rho"]}
        results.append(rec)
        logger.info("  %-5s mae %.5f -> %.5f (%+.2f%%)%s   rho %.4f -> %.4f (%+.4f)",
                    pos, bse["mae"], new["mae"], d_mae,
                    " *" if rec["beats_noise"] else "  ",
                    bse["rho"], new["rho"], rec["rho_delta"])

    out = EXPERIMENTS_DIR / f"udfa_wav_ab_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    wins = sum(1 for r in results if r["mae_delta_pct"] < 0)
    sig_w = sum(1 for r in results if r["mae_delta_pct"] < 0 and r["beats_noise"])
    sig_l = sum(1 for r in results if r["mae_delta_pct"] > 0 and r["beats_noise"])
    rho_w = sum(1 for r in results if r["rho_delta"] > 0)
    logger.info("UDFA wav A/B: MAE improves %d/%d (%d above placebo noise, "
                "%d significant losses), mean %+.2f%%",
                wins, len(results), sig_w, sig_l,
                float(np.mean([r["mae_delta_pct"] for r in results])) if results else float("nan"))
    logger.info("             bridge rho improves %d/%d, mean %+.4f",
                rho_w, len(results),
                float(np.nanmean([r["rho_delta"] for r in results])) if results else float("nan"))
    logger.info("→ %s", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="store_true", help="run the A/B after building")
    ap.add_argument("--method", choices=["mean", "quantile"], default="mean",
                    help="mean: conditional-mean prediction (shrinks spread). "
                         "quantile: conditional quantile matching "
                         "(spread-preserving).")
    args = ap.parse_args()

    df, stats = build_udfa_wav(args.method)
    path = PROCESSED_DIR / f"nfl_draft_labels_udfa_wav_{args.method}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Imputed AV/game for %d UDFAs [%s] (fit on %d drafted 2016+)",
                stats["n_udfa_imputed"], args.method, stats["n_train_drafted"])
    logger.info("  imputed mean=%.3f sd=%.3f   |   drafted observed mean=%.3f sd=%.3f",
                stats["imputed_mean"], stats["imputed_sd"],
                stats["drafted_observed_mean"], stats["drafted_observed_sd"])
    logger.info("→ %s", path)
    (EXPERIMENTS_DIR / f"udfa_wav_stats_{datetime.now():%Y%m%d_%H%M%S}.json"
     ).write_text(json.dumps(stats, indent=2, default=float))

    if args.ab:
        run_ab(path)


if __name__ == "__main__":
    main()
