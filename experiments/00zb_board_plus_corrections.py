"""
RESULT (2026-10-30): FAIL. 1,121 board players drafted 2019-22 (961 reached
the NFL); value incl. never-NFL zeros, within (group × class), Δ vs board:
    board      0.455
    corr_disc  0.432   -0.023 [-0.064, +0.020]   PRIMARY — worse than the board
    corr_perm  0.449   -0.006 [-0.026, +0.014]   placebo
    corr_p5    0.466   +0.011 [-0.021, +0.044]   optimistic upper bound, still n.s.
    deployed   0.459   +0.004 [-0.042, +0.051]
AUC for reaching the NFL: board 0.787, corr_disc 0.747, corr_p5 0.796,
deployed 0.821. Referees on NFL board players (00u --within-class): nothing
significant (late_* cover only the 2019 class). With 10 traits + missing
flags, the fit spends a coefficient as large as the board's on
combine_invited and overfits missingness (age_at_draft_missing -0.32).
The persistent board failures in 00za are real as partial correlations but
too small to build a score that beats the board on future classes.
Run on the forty-sign-fixed data (combine_forty_z: higher = faster).

Board-plus-corrections score: the consensus board rank, adjusted only for
the traits it persistently misprices (experiments/00za).

Why (2026-10-30): no model beats hasan_rank on board players (00x-00z); the
board aggregates what teams know and keeps improving. 00za found pre-draft
traits that predict NFL value beyond the board in 2014-18 AND 2019-22
(agility, combine invite, skipping the forty). This tests whether a small,
interpretable correction on top of the board out-ranks the board itself.

Model, per evaluation class T (fit on board players drafted 2014..T-1):
    pct(value) ~ pct(board) + pct(trait_1) + ... + pct(trait_k)
pooled across positions, OLS. Every column is a percentile within
(position group × draft class); a missing trait is 0.5 (neutral) and gets
a missing-indicator column. value = value_target, 0 for board players who
never reached the NFL.

Arms, OOF on board players drafted 2019-2022 (the 00za confirmation era):
    board        -hasan_rank alone (the bar)
    corr_disc    PRIMARY: every trait with 00za DISCOVERY q < 0.10
                 (2014-18 only — nothing about 2019-22 chose these)
    corr_perm    placebo: corr_disc with every trait permuted within
                 (group × class), in training and evaluation
    corr_p5      the five traits that passed 00za's full rule. Selection
                 used 2019-22, so this is an optimistic upper bound
    deployed     the deployed 8b global impact model's OOF (for reference)

Metric: Spearman within (group × class), averaged over groups then classes,
on ALL board players (incl. never-NFL zeros, so busts count). Also AUC for
reaching the NFL. Paired bootstrap over players within (group × class).

PRE-REGISTERED BAR (written before any run): corr_disc beats board on the
all-board-players value metric with its 95% paired interval excluding 0,
AND corr_perm's interval vs board includes 0. Reported, not decided on:
corr_p5, the deployed model, AUC, and 00u --within-class --subset board
on the referees (NFL players only).

Usage:
    00zb_board_plus_corrections.py [--out-dir DIR]
"""
import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
EVAL_CLASSES = range(2019, 2023)
N_BOOT = 1000
FAILURES_CSV = PROCESSED_DIR / "_board_failures_20261030" / "results_post_fix.csv"
P5 = ["combine_shuttle_z", "combine_cone_z", "combine_forty_z",
      "combine_invited", "skipped_forty"]


def _pct(df, col):
    return df.groupby(["position_group", "draft_year"])[col].rank(pct=True)


def _design(df, traits):
    X = {"board": _pct(df, "neg_board")}
    for t in traits:
        p = _pct(df, t)
        X[t] = p.fillna(0.5)
        if p.isna().any():
            X[f"{t}_missing"] = p.isna().astype(float)
    return pd.DataFrame(X, index=df.index)


def _oof(df, traits, rng=None):
    """OOF scores for EVAL_CLASSES; rng given → traits permuted within (group × class)."""
    d = df.copy()
    if rng is not None:
        for t in traits:
            for _, ix in d.groupby(["position_group", "draft_year"]).indices.items():
                d.iloc[ix, d.columns.get_loc(t)] = rng.permutation(d[t].to_numpy()[ix])
    X = _design(d, traits)
    y = _pct(d, "value")
    out = pd.Series(np.nan, index=d.index)
    coefs = {}
    for T in EVAL_CLASSES:
        tr, te = d["draft_year"] < T, d["draft_year"] == T
        A = np.column_stack([np.ones(tr.sum()), X[tr].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, y[tr].to_numpy(), rcond=None)
        out[te] = np.column_stack([np.ones(te.sum()), X[te].to_numpy()]) @ beta
        coefs[T] = dict(zip(["const"] + list(X.columns), beta))
    return out, coefs


def _metric(df, col, idx=None):
    d = df if idx is None else df.iloc[idx]
    by_class = []
    for _, c in d.groupby("draft_year"):
        r = [spearmanr(g[col], g["value"])[0] for _, g in c.groupby("position_group")
             if len(g) >= 10 and g["value"].nunique() > 1]
        if r:
            by_class.append(np.mean(r))
    return float(np.mean(by_class))


def _auc(df, col):
    return float(np.mean([roc_auc_score(g["is_bridge"].astype(bool), g[col])
                          for _, g in df.groupby(["position_group", "draft_year"])
                          if 0 < g["is_bridge"].astype(bool).sum() < len(g)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR / "_board_corrections_20261030")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    fails = pd.read_csv(FAILURES_CSV, index_col="trait")
    disc = fails.index[fails["disc_q"] < 0.10].tolist()
    logging.info("corr_disc traits (00za discovery q < 0.10): %s", disc)

    df = importlib.import_module("00za_board_failures")._cohort()
    ev = df["draft_year"].isin(EVAL_CLASSES)
    scores, coefs = {"board": df["neg_board"].astype(float)}, {}
    scores["corr_disc"], coefs["corr_disc"] = _oof(df, disc)
    scores["corr_perm"], _ = _oof(df, disc, rng)
    scores["corr_p5"], coefs["corr_p5"] = _oof(df, P5)
    g = pd.read_parquet(PROCESSED_DIR / "global_draft_models" / "global_oof_predictions.parquet")
    g = g[g["model_name"] == "impact_apy"].drop_duplicates("player_id", keep="last")
    scores["deployed"] = df["player_id"].map(g.set_index(g["player_id"].astype(str))["pred"])

    e = df[ev].assign(**{k: v[ev] for k, v in scores.items()})
    e = e.dropna(subset=list(scores)).reset_index(drop=True)
    logging.info("evaluation: %d board players, drafted %d-%d (%d reached the NFL)",
                 len(e), min(EVAL_CLASSES), max(EVAL_CLASSES), int(e["is_bridge"].sum()))
    strata = list(e.groupby(["position_group", "draft_year"]).indices.values())
    point = {k: _metric(e, k) for k in scores}
    boots = []
    for _ in range(N_BOOT):
        ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        boots.append({k: _metric(e, k, ix) for k in scores})

    print("\nboard players drafted 2019-22, value incl. never-NFL zeros, within (group × class):")
    passes = {}
    for k in scores:
        d = np.array([b[k] - b["board"] for b in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        passes[k] = lo > 0
        print(f"  {k:10s} rho {point[k]:.3f}  Δ vs board {point[k] - point['board']:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]   AUC reached-NFL {_auc(e, k):.3f}")
    perm_lo, perm_hi = np.percentile([b["corr_perm"] - b["board"] for b in boots], [2.5, 97.5])
    verdict = passes["corr_disc"] and perm_lo <= 0 <= perm_hi
    print(f"\nVERDICT (corr_disc vs board, placebo check): {'PASS' if verdict else 'FAIL'}")
    for name in ["corr_disc", "corr_p5"]:
        last = coefs[name][max(EVAL_CLASSES)]
        print(f"\n{name} coefficients (fit on 2014-{max(EVAL_CLASSES) - 1}, percentile scale):")
        for c, b in sorted(last.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {c:28s} {b:+.3f}")

    for k, s in scores.items():
        pd.DataFrame({"player_id": df.loc[ev, "player_id"], "pos_group": df.loc[ev, "position_group"],
                      "model_name": "impact", "pred": s[ev]}).dropna().to_parquet(
            args.out_dir / f"oof_{k}.parquet", index=False)
    print(f"\n→ OOF files in {args.out_dir} (score with 00u --within-class --subset board)")


if __name__ == "__main__":
    main()
