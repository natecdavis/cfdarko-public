"""
Step 00j: Contextual Residual Functional-Form Audit

After step 04 fits per-(stat × position_group) linear contextual models (Lasso /
Adaptive Lasso / ElasticNet / Ridge), this script checks whether each control's
functional form was well-specified. If the linear model captured the true
relationship, residuals should be unrelated to the control (Cremieux-style
diagnostic). Curvature or monotone drift in residuals-vs-control means signal
was left on the table — that control is a candidate for spline expansion.

Two outputs:
  1. Full scan table — one row per (pos × stat × control) with Spearman ρ,
     spline-F (or Welch-t for binary controls) p-values.
  2. Top-N targeted plots — only the worst offenders, ranked by form-test
     p-value, with LOWESS overlay for visual confirmation.

Inputs (read-only):
  data/processed/adjusted_gamelogs.parquet
  data/processed/contextual_models/model_summary.json

Outputs:
  experiments/00j_context_residual_audit.parquet
  experiments/plots_context_residuals/{rank:02d}_{pos}_{stat}_{control}.png

Usage:
  python experiments/00j_context_residual_audit.py
  python experiments/00j_context_residual_audit.py --positions QB --top-n 5
  python experiments/00j_context_residual_audit.py --no-plots
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats as sps
from sklearn.preprocessing import SplineTransformer
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, MODELED_STATS, CONTEXT_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Controls treated as binary — t-test instead of spline F-test
BINARY_FEATURES = {"is_home", "is_neutral", "game_indoors",
                   "conf_tier_power4", "conf_tier_g5"}

# Features stored post-demeaning in adjusted_gamelogs.parquet — sanity check
DEMEANED_FEATURES = ["opp_rolling_def_ppa", "opp_rolling_off_ppa"]

MIN_N = 200          # below this, skip both tests for that cell
SPLINE_N_KNOTS = 5
SPLINE_DEGREE = 3
LOWESS_FRAC = 0.3
PLOT_SAMPLE_CAP = 8000   # max points scattered per plot

OUT_DIR = Path(__file__).parent
OUT_PARQUET = OUT_DIR / "00j_context_residual_audit.parquet"
PLOTS_DIR = OUT_DIR / "plots_context_residuals"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--top-n", type=int, default=30,
                   help="Max plots to produce (default 30). Set 0 to skip plotting.")
    p.add_argument("--positions", type=str, default="",
                   help="Comma-separated subset of position groups (default: all modeled).")
    p.add_argument("--min-r2", type=float, default=0.0,
                   help="Skip cells whose contextual CV R² is below this (default 0.0). "
                        "Step 05 falls back to raw stats when cv_r2 < 0 (pipeline/"
                        "05_residual_computation.py:178), so those cells have NO "
                        "adjustment applied — diagnosing 'functional form leakage' is "
                        "meaningless there. Pass --min-r2 -1 to override and inspect "
                        "raw-stat correlations anyway.")
    p.add_argument("--input", type=str, default=str(PROCESSED_DIR / "adjusted_gamelogs.parquet"),
                   help="Override the adjusted_gamelogs.parquet path.")
    p.add_argument("--no-plots", action="store_true",
                   help="Audit table only; skip plot generation.")
    p.add_argument("--no-dedupe", action="store_true",
                   help="By default the top-N plots/print dedupe to one row per "
                        "(pos × stat) cell to avoid 10 versions of the same cell. "
                        "Pass this to see all top-N regardless of cell duplication.")
    return p.parse_args()


def _verify_demeaning(df: pd.DataFrame) -> None:
    """The two opponent-PPA features should be de-meaned within player-season.
    If a feature with a non-zero player-season mean shows up, the parquet has
    changed shape under us — bail out rather than silently misreport leakage."""
    for col in DEMEANED_FEATURES:
        if col not in df.columns:
            logger.warning("De-meaned feature %s missing from input — skipping check", col)
            continue
        ps_means = df.groupby(["player_id", "season_year"])[col].mean()
        worst = float(ps_means.abs().max())
        if worst > 1e-6:
            raise RuntimeError(
                f"Feature {col!r} has player-season mean up to {worst:.3e}, "
                f"expected near zero (de-meaned within player-season). "
                f"The persisted parquet's feature pipeline has changed — "
                f"either update DEMEANED_FEATURES or investigate step 05."
            )
    logger.info("De-meaning sanity check OK (max |player-season mean| < 1e-6 for %s)",
                ", ".join(DEMEANED_FEATURES))


def _spearman(resid: np.ndarray, ctrl: np.ndarray) -> tuple[float, float]:
    if resid.size < MIN_N:
        return (np.nan, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = sps.spearmanr(resid, ctrl)
    return (float(r), float(p))


def _spline_f_test(resid: np.ndarray, ctrl: np.ndarray) -> tuple[float, float, float]:
    """OLS of resid on B-spline(ctrl) vs intercept-only. Returns (F, p, partial_R²).
    Catches curvature the linear Lasso couldn't absorb. partial_R² is the
    n-independent effect size used for ranking — share of residual variance
    that the flexible spline of the control absorbs on top of the linear fit."""
    n = resid.size
    if n < MIN_N:
        return (np.nan, np.nan, np.nan)
    if np.nanstd(ctrl) == 0:
        return (np.nan, np.nan, np.nan)

    try:
        st = SplineTransformer(
            n_knots=SPLINE_N_KNOTS,
            degree=SPLINE_DEGREE,
            include_bias=False,
            extrapolation="constant",
        )
        B = st.fit_transform(ctrl.reshape(-1, 1))
    except Exception as exc:
        logger.debug("Spline transform failed: %s", exc)
        return (np.nan, np.nan, np.nan)

    # Drop any zero-variance columns (can happen on heavily clipped controls)
    keep = B.std(axis=0) > 1e-12
    if not keep.any():
        return (np.nan, np.nan, np.nan)
    B = B[:, keep]
    k = B.shape[1]

    # Full model: resid = α + Bβ + ε
    X_full = np.column_stack([np.ones(n), B])
    try:
        beta, *_ = np.linalg.lstsq(X_full, resid, rcond=None)
    except np.linalg.LinAlgError:
        return (np.nan, np.nan, np.nan)
    yhat = X_full @ beta
    ss_res_full = float(np.sum((resid - yhat) ** 2))

    # Reduced model: resid = α + ε  →  SS_res = SS_total
    ss_res_red = float(np.sum((resid - resid.mean()) ** 2))

    df_num = k
    df_den = n - k - 1
    if df_den <= 0 or ss_res_full <= 0 or ss_res_red <= 0:
        return (np.nan, np.nan, np.nan)

    F = ((ss_res_red - ss_res_full) / df_num) / (ss_res_full / df_den)
    if not np.isfinite(F) or F < 0:
        return (np.nan, np.nan, np.nan)
    p = float(sps.f.sf(F, df_num, df_den))
    partial_r2 = max(0.0, (ss_res_red - ss_res_full) / ss_res_red)
    return (float(F), p, float(partial_r2))


def _welch_t(resid: np.ndarray, ctrl: np.ndarray) -> tuple[float, float, float]:
    """Difference in residual means across a binary control. Returns
    (t, p, partial_R²). partial_R² = t² / (t² + df) puts the binary effect
    size on the same scale as the spline F's partial R²."""
    g0 = resid[ctrl <= 0.5]
    g1 = resid[ctrl > 0.5]
    if g0.size < 30 or g1.size < 30:
        return (np.nan, np.nan, np.nan)
    t, p = sps.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
    # Welch-Satterthwaite df approximation
    v0, v1 = g0.var(ddof=1), g1.var(ddof=1)
    n0, n1 = g0.size, g1.size
    num = (v0 / n0 + v1 / n1) ** 2
    den = (v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1)
    df = num / den if den > 0 else (n0 + n1 - 2)
    partial_r2 = float(t ** 2 / (t ** 2 + df)) if np.isfinite(t) else np.nan
    return (float(t), float(p), partial_r2)


def _scan_cell(df: pd.DataFrame, pos: str, stat: str,
               controls: list[str], cv_r2: float) -> list[dict]:
    """Return one row per (control) for the given (pos × stat) cell."""
    adj_col = f"adjusted_{stat}"
    if adj_col not in df.columns:
        logger.warning("[%s/%s] %s missing from input — skipping cell", pos, stat, adj_col)
        return []

    cell = df[df["position_group"] == pos]
    rows: list[dict] = []

    for ctrl in controls:
        if ctrl not in cell.columns:
            logger.debug("[%s/%s] control %s missing — skipping", pos, stat, ctrl)
            continue

        pair = cell[[adj_col, ctrl]].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(pair)
        is_binary = ctrl in BINARY_FEATURES or pair[ctrl].nunique(dropna=True) <= 2

        resid = pair[adj_col].astype(float).values
        c = pair[ctrl].astype(float).values

        sp_r, sp_p = _spearman(resid, c) if n >= MIN_N else (np.nan, np.nan)
        if is_binary:
            f_stat, f_p, partial_r2 = _welch_t(resid, c)
        else:
            f_stat, f_p, partial_r2 = _spline_f_test(resid, c)

        rows.append({
            "position_group": pos,
            "stat": stat,
            "control": ctrl,
            "n": int(n),
            "spearman_r": sp_r,
            "spearman_p": sp_p,
            "form_test_stat": f_stat,
            "form_test_p": f_p,
            "partial_r2": partial_r2,
            "model_cv_r2": float(cv_r2) if cv_r2 is not None else np.nan,
            "was_adjusted": bool(cv_r2 is not None and cv_r2 >= 0),
            "is_binary": bool(is_binary),
        })

    return rows


def _plot_continuous(df: pd.DataFrame, pos: str, stat: str, ctrl: str,
                     row: pd.Series, out_path: Path) -> None:
    adj_col = f"adjusted_{stat}"
    cell = df[df["position_group"] == pos][[adj_col, ctrl]].dropna()
    if len(cell) > PLOT_SAMPLE_CAP:
        cell = cell.sample(PLOT_SAMPLE_CAP, random_state=0)

    x = cell[ctrl].astype(float).values
    y = cell[adj_col].astype(float).values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.15, s=6, color="#4477AA", linewidths=0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sm = lowess(y, x, frac=LOWESS_FRAC, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color="#CC3311", lw=2.0, label="LOWESS")
    except Exception as exc:
        logger.debug("LOWESS failed for %s/%s/%s: %s", pos, stat, ctrl, exc)
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel(ctrl)
    ax.set_ylabel(adj_col)
    ax.set_title(
        f"{pos} / {stat}  vs  {ctrl}\n"
        f"n={int(row['n']):,}  spearman r={row['spearman_r']:+.3f} (p={row['spearman_p']:.2g})  "
        f"spline F={row['form_test_stat']:.2f}  partial R²={row['partial_r2']:.4f}  "
        f"model cv_r2={row['model_cv_r2']:.3f}",
        fontsize=9,
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_binary(df: pd.DataFrame, pos: str, stat: str, ctrl: str,
                 row: pd.Series, out_path: Path) -> None:
    adj_col = f"adjusted_{stat}"
    cell = df[df["position_group"] == pos][[adj_col, ctrl]].dropna()
    g0 = cell[cell[ctrl] <= 0.5][adj_col].astype(float).values
    g1 = cell[cell[ctrl] > 0.5][adj_col].astype(float).values

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([g0, g1], tick_labels=[f"{ctrl}=0\n(n={g0.size:,})",
                                      f"{ctrl}=1\n(n={g1.size:,})"],
               showmeans=True, meanline=True)
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_ylabel(adj_col)
    ax.set_title(
        f"{pos} / {stat}  vs  {ctrl}\n"
        f"mean Δ = {g1.mean() - g0.mean():+.4f}   "
        f"Welch t={row['form_test_stat']:+.2f}  partial R²={row['partial_r2']:.4f}   "
        f"model cv_r2={row['model_cv_r2']:.3f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _print_top10(table: pd.DataFrame, min_r2: float, dedupe: bool) -> None:
    print("\n" + "=" * 96)
    print(f"Top 10 (pos × stat × control) cells (cv_r2 ≥ {min_r2}) ranked by partial R²")
    print("— extra variance a flexible spline/binary of the control would absorb"
          + ("  [deduped: worst control per (pos×stat) cell]" if dedupe else ""))
    print("=" * 96)
    filtered = table[table["model_cv_r2"] >= min_r2].sort_values(
        "partial_r2", ascending=False, na_position="last"
    )
    if dedupe:
        filtered = filtered.drop_duplicates(subset=["position_group", "stat"], keep="first")
    top = filtered.head(10)
    if top.empty:
        print("No flagged cells (every cell either failed checks or filtered out).")
        return
    fmt = "{:<3}  {:<22}  {:<32}  {:>8}  {:>8}  {:>9}  {:>10}  {:>8}"
    print(fmt.format("pos", "stat", "control", "n", "spear_r",
                     "partialR²", "form_p", "cv_r2"))
    print("-" * 108)
    for _, r in top.iterrows():
        print(fmt.format(
            r["position_group"], r["stat"], r["control"],
            f"{int(r['n']):,}",
            f"{r['spearman_r']:+.3f}" if pd.notna(r["spearman_r"]) else "  nan ",
            f"{r['partial_r2']:.4f}" if pd.notna(r["partial_r2"]) else "  nan  ",
            f"{r['form_test_p']:.2e}" if pd.notna(r["form_test_p"]) else "  nan ",
            f"{r['model_cv_r2']:+.3f}" if pd.notna(r["model_cv_r2"]) else "  nan ",
        ))
    print()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Adjusted gamelogs not found: {input_path}")
    logger.info("Loading %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows × %d cols", *df.shape)

    _verify_demeaning(df)

    summary_path = PROCESSED_DIR / "contextual_models" / "model_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    cv_r2_map = {(s["position_group"], s["stat"]): s["cv_r2"] for s in summary}
    logger.info("Loaded contextual cv_r2 for %d cells", len(cv_r2_map))

    requested = [p.strip() for p in args.positions.split(",") if p.strip()]
    pos_filter = set(requested) if requested else None

    scan_rows: list[dict] = []
    for pos, stats in MODELED_STATS.items():
        if pos_filter is not None and pos not in pos_filter:
            continue
        if not stats:
            continue
        for stat in stats:
            cv_r2 = cv_r2_map.get((pos, stat))
            if cv_r2 is None:
                logger.debug("[%s/%s] no cv_r2 in summary — likely not fit", pos, stat)
                continue
            scan_rows.extend(_scan_cell(df, pos, stat, CONTEXT_FEATURES, cv_r2))

    table = pd.DataFrame(scan_rows).sort_values("partial_r2",
                                                ascending=False,
                                                na_position="last")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_PARQUET, index=False)
    logger.info("Wrote %d audit rows → %s", len(table), OUT_PARQUET)

    n_total = len(table)
    n_above_r2 = int((table["model_cv_r2"] >= args.min_r2).sum())
    n_adjusted = int(table["was_adjusted"].sum())
    logger.info("Rows where contextual model was actually applied (cv_r2 ≥ 0): %d / %d",
                n_adjusted, n_total)
    logger.info("Rows meeting --min-r2 ≥ %.3f filter: %d / %d",
                args.min_r2, n_above_r2, n_total)

    if args.no_plots or args.top_n <= 0:
        logger.info("Plot generation skipped.")
        _print_top10(table, args.min_r2, dedupe=not args.no_dedupe)
        return

    candidates = table[
        (table["model_cv_r2"] >= args.min_r2)
        & table["partial_r2"].notna()
    ].sort_values("partial_r2", ascending=False)
    if not args.no_dedupe:
        candidates = candidates.drop_duplicates(subset=["position_group", "stat"],
                                                keep="first")
    candidates = candidates.head(args.top_n)

    if candidates.empty:
        logger.info("No plottable candidates after filtering.")
        _print_top10(table, args.min_r2, dedupe=not args.no_dedupe)
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating %d plots → %s", len(candidates), PLOTS_DIR)
    for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
        pos, stat, ctrl = row["position_group"], row["stat"], row["control"]
        out_path = PLOTS_DIR / f"{rank:02d}_{pos}_{stat}_{ctrl}.png"
        try:
            if row["is_binary"]:
                _plot_binary(df, pos, stat, ctrl, row, out_path)
            else:
                _plot_continuous(df, pos, stat, ctrl, row, out_path)
        except Exception as exc:
            logger.warning("Plot failed for %s/%s/%s: %s", pos, stat, ctrl, exc)

    _print_top10(table, args.min_r2, dedupe=not args.no_dedupe)


if __name__ == "__main__":
    main()
