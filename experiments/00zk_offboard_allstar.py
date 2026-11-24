"""
Off-board model + college all-star game invites (pre-registered 2026-11-01).

Question. Do invitations to the Senior Bowl, Shrine Bowl and NFLPA
Collegiate Bowl — scouts' own pre-draft judgement, announced in December /
January, before the April draft — improve the ranking of off-board players?

Data: pipeline/00g_allstar_ingestion.py → data/processed/allstar_invites.
parquet (4,073 roster rows, 93.2% matched to CFBD ids against the FULL
college universe, never the bridge). Features, by the player's draft class
(last college season + 1):
    allstar_senior   invited to the Senior Bowl (covered 2014-2026)
    allstar_shrine   Shrine Bowl (2014-2026; no 2021 game → 0)
    allstar_nflpa    NFLPA Collegiate Bowl (2014, 2016, 2018-2023; no game
                     after 2023 → 0; 2015 and 2017 rosters lost → NaN)
    allstar_any      any of the three (NaN only where every source is NaN)
Before 2014 every feature is NaN. Leak diagnostic in the log: invite rates
among NFL vs non-NFL rows (the combine leak showed as 0% for non-NFL).

Arms — seed bags of 5, the production recipe (src/models/offboard_model.py):
    base      the production model, seeds 42-46 (reproduces step 8d's OOF)
    base_alt  same, seeds 142-146 (placebo)
    allstar   base + the 4 invite features, seeds 42-46
Scoring: draft classes 2014-22, K/P excluded, off-board players.

Multiple testing: the FOURTH feature test on these classes (00zh, 00zi,
00zj). One primary test; a pass is provisional until the 2023 class's y3-4
usage can confirm it.

PRE-REGISTERED BAR (95% paired bootstrap, 1000 reps, players resampled
within class):
    PASS if allstar − base on rho_ob_nfl has its 95% interval above 0 AND
    exceeds |base_alt − base| on rho_ob_nfl.
Secondary (reported, not tested): auc_ob, auc_ob_pos, auc_udfa, rho_udfa_nfl.

Deployment check, only on a PASS: 00zf's capped merge with allstar vs base
on the off-board side. Recommend adding the features to step 8d if top-M
and top-100 usage are not significantly negative.

RESULT (2026-11-01): FAIL. Classes 2014-22, 42,178 off-board / 1,173 NFL /
176 contributors; base reproduces production 8d exactly. Leak check: 872
invitees who never reached the NFL carry the feature (any-invite rate NFL
0.435, non-NFL 0.021); 1,029 off-board invitees, 369 reached the NFL.
    rho_ob_nfl  allstar − base +0.0087 [−0.0067, +0.0247]; placebo 0.0144
                → FAIL (interval includes 0, below the placebo)
    secondary   auc_ob +0.0027 [+0.0015, +0.0038], auc_udfa +0.0023
                [+0.0009, +0.0039], rho_udfa_nfl +0.006 (ns)
Production unchanged. Same pattern as 00zj: new scouting signal helps find
off-board contributors (AUC) but the ranking effect, if any, sits below the
~0.015 noise floor of rho on these classes.

Usage:
    00zk_offboard_allstar.py --out-dir DIR [--n-boot 1000]
"""
import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from config import PROCESSED_DIR  # noqa: E402
from src.models import offboard_model as om  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("00zk")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zh = _load("zh", "experiments/00zh_offboard_visits_pff.py")
zd = zh.zd

ALLSTAR_FEATURES = ["allstar_senior", "allstar_shrine", "allstar_nflpa", "allstar_any"]
NFLPA_MISSING = {2015, 2017}
SEEDS = [42, 43, 44, 45, 46]
ARM_SPECS = {"base": ([], SEEDS), "base_alt": ([], [s + 100 for s in SEEDS]),
             "allstar": (ALLSTAR_FEATURES, SEEDS)}
PAIRS = [("allstar", "base"), ("base_alt", "base"), ("allstar", "dep_impact"),
         ("base", "dep_impact")]
EVAL_CLASSES = (2014, 2022)
ROLE_FEATURES = ALLSTAR_FEATURES   # run_bags / main reuse the 00zj names


def add_allstar(rows: pd.DataFrame) -> pd.DataFrame:
    inv = pd.read_parquet(PROCESSED_DIR / "allstar_invites.parquet").dropna(subset=["player_id"])
    inv["player_id"] = om.pid(inv["player_id"])
    rows = rows.copy()
    cls = rows["last_college_season_year"] + 1
    for game in ("senior", "shrine", "nflpa"):
        ids = set(inv.loc[inv["game"] == game, "player_id"])
        v = rows["player_id"].isin(ids).astype(float)
        v[cls < 2014] = np.nan
        if game == "nflpa":
            v[cls.isin(NFLPA_MISSING)] = np.nan
        rows[f"allstar_{game}"] = v
    parts = rows[["allstar_senior", "allstar_shrine", "allstar_nflpa"]]
    rows["allstar_any"] = parts.max(axis=1, skipna=True)
    rows.loc[parts.isna().all(axis=1), "allstar_any"] = np.nan
    era = cls.between(2014, 2022)
    for c in ALLSTAR_FEATURES:
        log.info("  %-15s rate NFL %.3f  non-NFL %.4f  (classes 2014-22; n invited non-NFL %d)", c,
                 rows.loc[era & rows["nfl"], c].mean(), rows.loc[era & ~rows["nfl"], c].mean(),
                 int((rows.loc[era & ~rows["nfl"], c] == 1).sum()))
    ob = era & rows["hasan_rank"].isna()
    log.info("  off-board invitees 2014-22: %d (NFL %d)", int((rows.loc[ob, "allstar_any"] == 1).sum()),
             int(((rows.loc[ob, "allstar_any"] == 1) & rows.loc[ob, "nfl"]).sum()))
    return rows


def run_bags(step8, rows, feats, folds):
    yr = rows["last_college_season_year"].fillna(0)
    fold_of = pd.Series(np.nan, index=rows.index)
    score = {a: pd.Series(np.nan, index=rows.index) for a in ARM_SPECS}
    for train_end, val_start, val_end in folds:
        tr_m, va_m = yr <= train_end, yr.between(val_start, val_end)
        fold_of[va_m] = val_start
        tr, va = rows[tr_m], rows[va_m]
        for arm, (extra, seeds) in ARM_SPECS.items():
            bag = om.OffboardBag(seeds).fit(step8, tr, tr[tr["nfl"]], feats + extra)
            score[arm][va.index] = bag.predict(va)["score"]
            log.info("fold %d %s: %d cols", val_start, arm, len(feats + extra))
    return score, fold_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step8 = importlib.import_module("08_draft_models")
    folds, _ = step8._split_holdout(step8._expanding_window_folds())
    snap = step8.assemble_snapshot(include_derived=True, logger=step8.logger)
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    rows, feats = om.build_rows(step8, snap, labels, bridge)
    rows = add_allstar(rows)

    score, fold_of = run_bags(step8, rows, feats, folds)
    base = rows[zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted"]].assign(val_start=fold_of)
    for a in ARM_SPECS:
        base[a] = score[a]
    base = base.dropna(subset=["val_start", "base"])
    base["val_start"] = base["val_start"].astype(int)
    base.to_parquet(args.out_dir / "oof_all_arms.parquet", index=False)
    base.assign(pre=base["allstar"], pre_43=base["base"])[
        zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted", "val_start", "pre", "pre_43"]
    ].to_parquet(args.out_dir / "oof_00zd_allstar.parquet", index=False)

    # The base arm must reproduce production step 8d.
    prod = pd.read_parquet(PROCESSED_DIR / "offboard_models" / "offboard_oof_predictions.parquet")
    chk = base.merge(prod[["player_id", "pos_group", "val_start", "score"]],
                     on=["player_id", "pos_group", "val_start"])
    log.info("base vs production 8d OOF: %d rows, max |Δ| %.2e", len(chk),
             (chk["base"] - chk["score"]).abs().max())

    E = zd._deployed(base).dropna(subset=["dep_impact"]).drop_duplicates("player_id", keep="last")
    E["cls"] = E["last_college_season_year"].astype(int) + 1
    E = E[E["cls"].between(*EVAL_CLASSES)]
    E["board"] = E["hasan_rank"].notna()
    E["contrib"] = (E["usage"] >= zd.CONTRIB).astype(int)
    E["starter"] = (E["usage"] >= zd.STARTER).astype(int)
    ob = E[~E["board"]]
    log.info("scoring cohort %d: off-board %d, off-board NFL %d, contributors %d",
             len(E), len(ob), int(ob["nfl"].sum()), int(ob["contrib"].sum()))

    arms = list(ARM_SPECS) + ["dep_impact"]
    point, deltas = zh.bootstrap(E, arms, PAIRS, args.n_boot, levels=(95.0,))
    pd.set_option("display.width", 200)
    print("\nheadline metrics:")
    print(pd.DataFrame(point).T.round(4).to_string())
    print("\nΔ, paired bootstrap 95%:")
    print(deltas.round(4).to_string(index=False))
    print("\ntop-K vs the bottom of the board:")
    print(zd._topk_table(E, arms).to_string())

    g = lambda a, b: deltas[(deltas["Δ"] == f"{a} − {b}") & (deltas["metric"] == "rho_ob_nfl")].iloc[0]
    d, plac = g("allstar", "base"), abs(g("base_alt", "base")["delta"])
    v = {"pass": bool(d["lo95"] > 0 and d["delta"] > plac), "d_rho": round(d["delta"], 4),
         "ci95": f"[{d['lo95']:+.4f}, {d['hi95']:+.4f}]", "placebo": round(plac, 4)}
    print("\nVERDICT:", v)
    deltas.to_csv(args.out_dir / "deltas.csv", index=False)
    pd.Series(v, dtype=object).to_json(args.out_dir / "verdict.json")


if __name__ == "__main__":
    main()
