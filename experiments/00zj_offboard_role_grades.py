"""
Off-board model + position-specific PFF grades (pre-registered 2026-11-01).

Question. The off-board engine (step 8d) sees PFF only through crude side
features: the offense or defense grade and snap counts. Do the grades for the
skill each position is drafted for (pass rush, coverage, pass blocking, route
running...) improve the ranking of off-board players?

Role-grade features. The model is pooled across positions, so each position's
grades go into two position-neutral slots (mean / peak / last each, 6 cols):
    group  role_a (primary)       role_b (secondary)
    QB     pass                   run
    RB     run                    pass_route
    WR     pass_route             hands_drop
    TE     pass_route             run_block
    OL     pass_block             run_block
    DL     pass_rush_defense      run_defense
    EDGE   pass_rush_defense      run_defense
    LB     run_defense            coverage_defense
    DB     coverage_defense       tackle
(source columns pff_grades_<stem>_<agg> from the assembled snapshot)

Arms — seed bags of 5, the production recipe (src/models/offboard_model.py):
    base      the production model (00zd features + PFF side features),
              seeds 42-46 — reproduces step 8d's OOF
    base_alt  same, seeds 142-146 (placebo: an independent bag)
    role      base + the 6 role-grade columns, seeds 42-46
Scoring: draft classes 2014-22 (every board-era class with observed y3-4
usage; 00zh/00zi used 2015-22 only because visits start in 2015), K/P
excluded, off-board players.

Multiple testing: this is the third feature test on these classes (00zh,
00zi). One primary test here; nothing else is promoted.

PRE-REGISTERED BAR (95% paired bootstrap, 1000 reps, players resampled
within class):
    PASS if role − base on rho_ob_nfl (usage Spearman among off-board players
    who reached the NFL, per group × fold) has its 95% interval above 0 AND
    exceeds |base_alt − base| on rho_ob_nfl.
Secondary (reported, not tested): auc_ob, auc_ob_pos, auc_udfa, rho_udfa_nfl.

Deployment check, only on a PASS: 00zf's capped merge with role vs base on
the off-board side (merged_pre_cap − merged_pre43_cap, pre = role, pre_43 =
base). Recommend adding the features to step 8d if top-M and top-100 usage
are not significantly negative (95% interval wholly below 0).

RESULT (2026-11-01): FAIL. Classes 2014-22: 42,178 off-board, 1,173 off-board
NFL, 176 contributors. Base reproduces production 8d OOF exactly.
    rho_ob_nfl  role − base −0.0043 [−0.0287, +0.0216]; placebo (base_alt −
                base) −0.0144 → FAIL
    secondary   auc_ob +0.0053 [+0.0024, +0.0082] (placebo +0.0001),
                auc_udfa +0.0038 [+0.0010, +0.0072]; rho_udfa_nfl −0.006
No deployment check (primary failed); production unchanged.
Reading. Position grades help find WHO reaches the NFL / contributes off the
board (AUC, ~50x the placebo) but not ORDER the off-board players who make
it. The bag placebo on rho is 0.014 over 2014-22, so ranking effects under
~0.015 can't be seen even with 5-seed bags; 00zi's PFF-side gain was 0.042.
The AUC gain is a secondary result on reused classes: a lead, not a pass.

Usage:
    00zj_offboard_role_grades.py --out-dir DIR [--n-boot 1000]
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
log = logging.getLogger("00zj")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zh = _load("zh", "experiments/00zh_offboard_visits_pff.py")
zd = zh.zd

ROLE_STEMS = {
    "QB": ("pass", "run"), "RB": ("run", "pass_route"), "WR": ("pass_route", "hands_drop"),
    "TE": ("pass_route", "run_block"), "OL": ("pass_block", "run_block"),
    "DL": ("pass_rush_defense", "run_defense"), "EDGE": ("pass_rush_defense", "run_defense"),
    "LB": ("run_defense", "coverage_defense"), "DB": ("coverage_defense", "tackle"),
}
AGGS = ("mean", "peak", "last")
ROLE_FEATURES = [f"pff_role_{slot}_{a}" for slot in ("a", "b") for a in AGGS]
SEEDS = [42, 43, 44, 45, 46]
ARM_SPECS = {"base": ([], SEEDS), "base_alt": ([], [s + 100 for s in SEEDS]),
             "role": (ROLE_FEATURES, SEEDS)}
PAIRS = [("role", "base"), ("base_alt", "base"), ("role", "dep_impact"),
         ("base", "dep_impact")]
EVAL_CLASSES = (2014, 2022)


def add_role_grades(snap: pd.DataFrame) -> pd.DataFrame:
    snap = snap.copy()
    for slot, i in (("a", 0), ("b", 1)):
        for a in AGGS:
            out = pd.Series(np.nan, index=snap.index)
            for g, stems in ROLE_STEMS.items():
                col = f"pff_grades_{stems[i]}_{a}"
                if col in snap.columns:
                    m = snap["position_group"] == g
                    out[m] = snap.loc[m, col]
            snap[f"pff_role_{slot}_{a}"] = out
    for c in ROLE_FEATURES:
        log.info("  %-20s non-null %.1f%%", c, 100 * snap[c].notna().mean())
    return snap


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
    snap = add_role_grades(step8.assemble_snapshot(include_derived=True, logger=step8.logger))
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    rows, feats = om.build_rows(step8, snap, labels, bridge)
    keep = snap[["player_id"] + ROLE_FEATURES].assign(player_id=lambda d: om.pid(d["player_id"]))
    rows = rows.merge(keep.drop_duplicates("player_id"), on="player_id", how="left")

    score, fold_of = run_bags(step8, rows, feats, folds)
    base = rows[zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted"]].assign(val_start=fold_of)
    for a in ARM_SPECS:
        base[a] = score[a]
    base = base.dropna(subset=["val_start", "base"])
    base["val_start"] = base["val_start"].astype(int)
    base.to_parquet(args.out_dir / "oof_all_arms.parquet", index=False)
    base.assign(pre=base["role"], pre_43=base["base"])[
        zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted", "val_start", "pre", "pre_43"]
    ].to_parquet(args.out_dir / "oof_00zd_role.parquet", index=False)

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
    d, plac = g("role", "base"), abs(g("base_alt", "base")["delta"])
    v = {"pass": bool(d["lo95"] > 0 and d["delta"] > plac), "d_rho": round(d["delta"], 4),
         "ci95": f"[{d['lo95']:+.4f}, {d['hi95']:+.4f}]", "placebo": round(plac, 4)}
    print("\nVERDICT:", v)
    deltas.to_csv(args.out_dir / "deltas.csv", index=False)
    pd.Series(v, dtype=object).to_json(args.out_dir / "verdict.json")


if __name__ == "__main__":
    main()
