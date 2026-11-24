"""
Off-board model (step 8d) with stage 2 trained on rookie-deal value
(pre-registered 2026-11-01).

The merged board now ranks in rookie-deal value (00zs; step 10 merged_rank).
Board players come from the board curve; off-board players from 8d's
within-(class × position) percentile, mapped to rookie value by the
calibrator. 8d still learns WHO is good from y3-4 snap share:
    score = P(NFL) x E[usage_y34 | NFL]
Rookie value differs from snap share within a position in ways that should
matter for ranking: it counts all four rookie seasons (y1-y2 too), prices
the role at the player's NFL position (a college LB who rushes the passer
is priced as EDGE), and adjusts QB/RB/WR/TE seasons for EPA quality.

Arms (5-seed bags, production recipe src/models/offboard_model.py):
    base      production 8d, seeds 42-46 (must reproduce step 8d's OOF)
    base_alt  same, seeds 142-146 (placebo)
    rv        stage 2 trained on rookie value value_y14 (v3.1) among the
              fold's training NFL rows, multiplied by c = sd(usage) /
              sd(value_y14) on those rows so the usage-tuned, scale-dependent
              LightGBM params carry over (a constant does not change the
              within-class order). Predictions clipped at 0. Stage 1
              unchanged. NFL rows without a rookie value (linkage) are
              dropped from stage 2 and from scoring.
Scoring: draft classes 2014-22, off-board players, K/P excluded.
PRIMARY: value captured. Per (class × position), the 10 off-board players
with the highest score; mean realized rookie value (0 if never NFL);
averaged over (class × position) cells. rv − base. ADOPT if P(> 0) > 0.5
(paired bootstrap, players resampled within class, 1000 reps): aligning the
training target with the product currency is principled and costs
nothing. |base_alt − base| reported as the seed-noise scale.
Reported: top 25 per (class × position); Spearman vs rookie value and vs
usage among off-board NFL players within (position × fold); contributor
AUC (usage >= 0.25); per-position top-10 value captured.

BUG FIX (2026-11-01, before reading valid results): the first run wrote
each arm's score to a column named after the arm, so arm "rv" overwrote the
outcome column "rv" (every metric was scored against the rv arm's own
predictions; rho_nfl_rv = 1.000 gave it away). Arm scores now live in
S_<arm>; --rescore rebuilds the realized outcome from the labels and
re-scores the saved OOF scores (the fits were unaffected: base reproduced
production 8d exactly).

RESULT (2026-11-01, corrected scoring): KEEP the usage target.
44,636 players scored, 42,169 off-board, 1,164 off-board NFL; base
reproduced production 8d OOF exactly (max |Δ| 0).
    rv − base        top10_rv −0.00061 [−0.00106, +0.00052] P 0.19
                     top25_rv +0.00023 (P 0.87); rho_nfl_rv −0.018 (P 0.30);
                     rho_nfl_usage +0.003 (P 0.59); AUC −0.0036 (P 0.07)
    base_alt − base  top10_rv +0.00021 (P 0.56); rho_nfl_rv −0.017
The rv arm is within seed noise on every metric and slightly worse on the
primary; top-10 is lower at 8 of 9 positions (DL, RB, TE most). Reading:
off-board NFL players are mostly fringe players whose four-year value is
dominated by noise in y1-2 snaps and games; y3-4 usage is the cleaner
signal of who sticks, and the calibrator already maps the 8d percentile to
rookie value. (The first, buggy run's "ADOPT" with rho_nfl_rv = 1.000 was
the column clobber; see BUG FIX.)

Usage: 00zw_offboard_rookie_value.py [--n-boot 1000] [--rescore]
"""
import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from config import PROCESSED_DIR  # noqa: E402
from src.models import offboard_model as om  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("00zw")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zn = _load("zn", "experiments/00zn_offboard_usage_norm.py")
OUT = PROCESSED_DIR / "_offboard_rookie_value_20261101"
SEEDS = [42, 43, 44, 45, 46]
ARMS = {"base": (False, SEEDS), "base_alt": (False, [s + 100 for s in SEEDS]), "rv": (True, SEEDS)}
PAIRS = [("rv", "base"), ("base_alt", "base")]
EVAL_CLASSES = (2014, 2022)
CONTRIB = 0.25


def run_bags(step8, rows, feats, folds):
    yr = rows["last_college_season_year"].fillna(0)
    fold_of = pd.Series(np.nan, index=rows.index)
    score = {a: pd.Series(np.nan, index=rows.index) for a in ARMS}
    for train_end, val_start, val_end in folds:
        tr_m, va_m = yr <= train_end, yr.between(val_start, val_end)
        fold_of[va_m] = val_start
        tr, va = rows[tr_m], rows[va_m]
        nfl = tr[tr["nfl"]]
        for arm, (use_rv, seeds) in ARMS.items():
            if use_rv:
                t2 = nfl[nfl["rv"].notna()]
                c = t2["usage"].std() / t2["rv"].std()
                log.info("fold %d rv stage 2: %d rows (dropped %d), scale c = %.3f",
                         val_start, len(t2), len(nfl) - len(t2), c)
                bag = zn.NormBag(seeds).fit(step8, tr, t2.assign(usage=t2["rv"] * c), feats)
            else:
                bag = om.OffboardBag(seeds).fit(step8, tr, nfl, feats)
            score[arm][va.index] = bag.predict(va)["score"]
            log.info("fold %d %s done; trees %s", val_start, arm, bag.trees)
    return score, fold_of


def metrics(E, idx=None):
    d = E if idx is None else E.iloc[idx]
    ob = d[~d["board"]]
    cells = list(ob.groupby(["cls", "pos_group"]))
    res = {}
    for a in ARMS:
        r, s = {}, f"S_{a}"
        for k in (10, 25):
            r[f"top{k}_rv"] = np.mean([g.nlargest(k, s)["rv"].mean() for _, g in cells])
        nf = ob[ob["nfl"]]
        for y in ("rv", "usage"):
            r[f"rho_nfl_{y}"] = np.nanmean([spearmanr(g[s], g[y])[0]
                                            for _, g in nf.groupby(["pos_group", "val_start"]) if len(g) >= 20])
        r["auc_contrib"] = np.nanmean([roc_auc_score(g["contrib"], g[s]) for _, g in ob.groupby("cls")
                                       if 0 < g["contrib"].sum() < len(g)])
        res[a] = r
    return res


def rescore_frame():
    """Saved OOF from the first (buggy) run: column "rv" holds the rv arm's
    score; rebuild the realized rookie value exactly as main() does."""
    E = pd.read_parquet(OUT / "oof_all_arms.parquet").rename(
        columns={"rv": "S_rv", "base": "S_base", "base_alt": "S_base_alt"})
    L = pd.read_parquet(PROCESSED_DIR / "rookie_value_labels.parquet")
    L = L.assign(pid=om.pid(L["player_id"])).dropna(subset=["pid"]).drop_duplicates("pid").set_index("pid")
    E["rv"] = np.where(E["nfl"], E["player_id"].map(L["value_y14"]), 0.0)
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--rescore", action="store_true", help="re-score saved OOF scores")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    if args.rescore:
        return evaluate(rescore_frame(), args.n_boot)
    step8 = importlib.import_module("08_draft_models")
    folds, _ = step8._split_holdout(step8._expanding_window_folds())
    snap = step8.assemble_snapshot(include_derived=True, logger=step8.logger)
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    rows, feats = om.build_rows(step8, snap, labels, bridge)
    L = pd.read_parquet(PROCESSED_DIR / "rookie_value_labels.parquet")
    L = L.assign(pid=om.pid(L["player_id"])).dropna(subset=["pid"]).drop_duplicates("pid").set_index("pid")
    rows["rv"] = np.where(rows["nfl"], rows["player_id"].map(L["value_y14"]), 0.0)
    log.info("NFL rows %d, with rookie value %d", int(rows["nfl"].sum()),
             int(rows.loc[rows["nfl"], "rv"].notna().sum()))

    score, fold_of = run_bags(step8, rows, feats, folds)
    E = rows[["player_id", "pos_group", "last_college_season_year", "hasan_rank", "nfl", "usage",
              "rv"]].assign(val_start=fold_of)
    for a in ARMS:
        E[f"S_{a}"] = score[a]
    E = E.dropna(subset=["val_start", "S_base"])
    E["val_start"] = E["val_start"].astype(int)
    E.to_parquet(OUT / "oof_all_arms.parquet", index=False)

    prod = pd.read_parquet(PROCESSED_DIR / "offboard_models" / "offboard_oof_predictions.parquet")
    chk = E.merge(prod[["player_id", "pos_group", "val_start", "score"]],
                  on=["player_id", "pos_group", "val_start"])
    log.info("base vs production 8d OOF: %d rows, max |Δ| %.2e", len(chk),
             (chk["S_base"] - chk["score"]).abs().max())

    evaluate(E, args.n_boot)


def evaluate(E, n_boot):
    E = E.drop_duplicates("player_id", keep="last")
    E["cls"] = E["last_college_season_year"].astype(int) + 1
    E = E[E["cls"].between(*EVAL_CLASSES) & ~E["pos_group"].isin(["K", "P"]) & E["rv"].notna()]
    E["board"] = E["hasan_rank"].notna()
    E["contrib"] = (E["usage"] >= CONTRIB).astype(int)
    E = E.reset_index(drop=True)
    ob = E[~E["board"]]
    log.info("scoring: %d players, off-board %d, off-board NFL %d", len(E), len(ob), int(ob["nfl"].sum()))

    rng = np.random.default_rng(0)
    pt = metrics(E)
    groups = [g.index.to_numpy() for _, g in E.groupby("cls")]
    boots = [metrics(E, np.concatenate([rng.choice(g, len(g)) for g in groups])) for _ in range(n_boot)]
    print(pd.DataFrame(pt).T.round(5).to_string())
    rows_d = []
    for a, b in PAIRS:
        for k in pt[a]:
            ds = np.array([bt[a][k] - bt[b][k] for bt in boots])
            rows_d.append({"Δ": f"{a} − {b}", "metric": k, "delta": pt[a][k] - pt[b][k],
                           "lo95": np.nanpercentile(ds, 2.5), "hi95": np.nanpercentile(ds, 97.5),
                           "P(>0)": np.nanmean(ds > 0)})
    D = pd.DataFrame(rows_d)
    print(D.round(5).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)
    p = D[(D["Δ"] == "rv − base") & (D["metric"] == "top10_rv")].iloc[0]
    pl = D[(D["Δ"] == "base_alt − base") & (D["metric"] == "top10_rv")].iloc[0]
    print(f"\nDECISION: rv − base top10_rv {p['delta']:+.5f} [{p['lo95']:+.5f}, {p['hi95']:+.5f}] "
          f"P(>0) {p['P(>0)']:.3f}; seed placebo {pl['delta']:+.5f} → "
          f"{'ADOPT' if p['P(>0)'] > 0.5 else 'KEEP usage target'}")
    print("\nper position, off-board top-10 value captured (mean rookie value):")
    print(pd.DataFrame({a: ob.groupby("pos_group").apply(
        lambda g: np.mean([c.nlargest(10, f"S_{a}")["rv"].mean() for _, c in g.groupby("cls")])) for a in ARMS})
        .round(5).to_string())


if __name__ == "__main__":
    main()
