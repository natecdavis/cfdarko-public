"""
Off-board model + draft visits / PFF performance (pre-registered 2026-11-01).

Question. Earlier feature tests were scored mostly on NFL players, two thirds
of whom are on the board, where hasan_rank swamps everything. A feature that
only helps OFF the board would be diluted. Do draft visits and PFF
performance features improve the ranking of off-board players?

Model. The 00zd pre-draft two-stage model, unchanged:
    score = P(reach NFL) × E[usage_y34 | NFL]   (pooled, untuned, OOF on the
    step-8 tuning folds; the holdout fold is never touched)

Arms (factorial):
    base      00zd features
    base_43   base, seed 43 (placebo)
    pff       base + PFF side features: offense (QB/RB/WR/TE/OL) or defense
              grade and snap mean / peak / last, plus total snaps (00ze's
              SIDE_FEATURES; the pooled feature list otherwise has no PFF
              performance at all)
    vis       base + draft visits
    both      base + PFF side features + draft visits
    dep_impact the deployed 8b OOF on the same rows (reference)

Draft visits (data/raw/draft_visits.parquet, draft years 2015-2026; 2020-21
coverage is thin). Per (draft year, player): teams visited, total, private
visits (PRI), combine meetings (COM), pro-day meetings (PRO), workouts (WOR).
LEAK GUARD: the visits data has no school, and 00w joined it through the
bridge, i.e. only NFL players — used as a feature that would leak the
outcome (as the combine join did, [[project_combine_join_leak]]). Here
records are matched to the FULL college universe on normalized name + draft
class (last college season + 1), keeping only keys unique on BOTH sides,
whether or not the player reached the NFL. In covered draft years an
unmatched player gets 0 visits; before 2015 the features are NaN.
Diagnostic: match rate for drafted vs undrafted visit records.

Scoring cohort: draft classes 2015-2022 (visits coverage ∩ y3-4 observable),
K/P excluded, off-board players. Outcome: contributor = usage_y34 >= 0.25.

PRE-REGISTERED BAR. Two primary contrasts, pff − base and vis − base, on
auc_ob (contributor AUC among off-board players per class, averaged over
classes). Bonferroni for two tests: 97.5% paired-bootstrap intervals (500
reps, players resampled within class). A family PASSES if
    1. Δauc_ob's 97.5% interval excludes 0, and
    2. Δauc_ob > |base_43 − base| on auc_ob, and
    3. Δrho_ob_nfl (usage Spearman among off-board NFL players) is not below
       −|base_43 − base| on rho_ob_nfl with its interval wholly below 0.
both − base is reported, not tested.

Deployment (only if a family passes): the best passing arm replaces the
deployed score on the off-board side of the merged board if, in 00zf's
capped merge, merged_pre_cap − merged_dep_cap on top-M usage is not below 0
with its 95% interval wholly below 0, AND the arm's auc_ob beats dep_impact.
Otherwise production is unchanged.

RESULT (2026-11-01): BOTH FAIL the bar. Classes 2015-22: 38,232 off-board,
1,110 off-board NFL, 161 contributors. Placebo (base_43 − base): auc_ob
−0.0051 [97.5% −0.0082, −0.0023], rho_ob_nfl +0.0045.
    pff − base  auc_ob +0.0044 [+0.0002, +0.0089]  CI excludes 0 but does
                NOT beat |placebo| 0.0051 → FAIL (condition 2)
                rho_ob_nfl +0.053 [+0.008, +0.103], auc_udfa +0.005,
                rho_udfa_nfl +0.051 [+0.001, +0.099]
    vis − base  auc_ob +0.0029 [−0.0010, +0.0065] → FAIL
                rho_ob_nfl +0.031 [−0.004, +0.067], auc_udfa −0.009
    both − base auc_ob +0.0050, rho_ob_nfl +0.047 (not tested)
No deployment check was run (no family passed); production is unchanged.
Reading. The single-seed placebo moves AUC by as much as either family, so
AUC can't resolve effects this size. PFF's gain shows up where the off-board
problem actually is — ranking players once they reach the NFL (rho 0.064 →
0.117, ~12x the placebo) — but that was a secondary metric, so it's a lead to
confirm, not a pass. Visits add little beyond PFF (both ≈ pff). Visit match
rate: 91.3% of drafted vs 81.7% of undrafted records (name mismatches at
small schools), so the 0-visit value is slightly noisier for UDFAs.
Side note: the dedicated base model beats dep_impact on auc_ob (+0.009) and
its off-board top 10/25/50 reach the NFL at 91/90/84% vs 74/73/72%.

Usage:
    00zh_offboard_visits_pff.py --out-dir DIR [--n-boot 500]
"""
import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from config import RAW_DIR  # noqa: E402
from src.utils.college_id_match import _norm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("00zh")

spec = importlib.util.spec_from_file_location("zd", ROOT / "experiments/00zd_offboard_model.py")
zd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zd)

OFFENSE = {"QB", "RB", "WR", "TE", "OL"}
SIDE_STATS = ["grades", "snap_counts"]
SIDE_AGGS = ["mean", "peak", "last"]
PFF_FEATURES = ([f"pff_side_{s}_{a}" for s in SIDE_STATS for a in SIDE_AGGS]
                + ["pff_side_snap_counts_total"])
VIS_FEATURES = ["vis_teams", "vis_total", "vis_pri", "vis_com", "vis_pro", "vis_wor"]
VIS_YEARS = (2015, 2026)
EVAL_CLASSES = (2015, 2022)
ARM_SPECS = {"base": ([], 42), "base_43": ([], 43), "pff": (PFF_FEATURES, 42),
             "vis": (VIS_FEATURES, 42), "both": (PFF_FEATURES + VIS_FEATURES, 42)}
PAIRS = [("pff", "base"), ("vis", "base"), ("both", "base"), ("base_43", "base"),
         ("base", "dep_impact"), ("pff", "dep_impact"), ("vis", "dep_impact"),
         ("both", "dep_impact")]


def add_side_pff(snap: pd.DataFrame) -> pd.DataFrame:
    snap = snap.copy()
    off = snap["position_group"].isin(OFFENSE)
    for s in SIDE_STATS:
        for a in SIDE_AGGS + (["total"] if s == "snap_counts" else []):
            o, d = f"pff_{s}_offense_{a}", f"pff_{s}_defense_{a}"
            if o in snap.columns and d in snap.columns:
                snap[f"pff_side_{s}_{a}"] = np.where(off, snap[o], snap[d])
    missing = [c for c in PFF_FEATURES if c not in snap.columns]
    assert not missing, f"PFF side columns missing: {missing}"
    return snap


def add_visits(snap: pd.DataFrame) -> pd.DataFrame:
    """Visits matched to the FULL college universe (see the leak guard)."""
    v = pd.read_parquet(RAW_DIR / "draft_visits.parquet")
    agg = (v.groupby(["draft_year", "player"])
           .agg(vis_teams=("Total", lambda s: int((s > 0).sum())), vis_total=("Total", "sum"),
                vis_pri=("PRI", "sum"), vis_com=("COM", "sum"), vis_pro=("PRO", "sum"),
                vis_wor=("WOR", "sum"), drafted=("round", lambda r: r.notna().any()))
           .reset_index())
    agg["k"] = agg["player"].map(_norm)
    agg = agg[agg.groupby(["k", "draft_year"])["player"].transform("size") == 1]

    snap = snap.copy()
    snap["_k"] = snap["player_name"].map(_norm)
    snap["_cls"] = snap["last_college_season_year"] + 1
    uniq = snap.groupby(["_k", "_cls"])["player_id"].transform("size") == 1
    keys = snap.loc[uniq, ["_k", "_cls", "player_id"]]
    m = agg.merge(keys, left_on=["k", "draft_year"], right_on=["_k", "_cls"], how="left")
    log.info("visit records %d (unique name-year); matched %.1f%% (drafted %.1f%%, undrafted %.1f%%)",
             len(m), 100 * m["player_id"].notna().mean(),
             100 * m.loc[m["drafted"], "player_id"].notna().mean(),
             100 * m.loc[~m["drafted"], "player_id"].notna().mean())
    m = m.dropna(subset=["player_id"]).drop_duplicates("player_id")
    snap = snap.merge(m[["player_id"] + VIS_FEATURES], on="player_id", how="left")
    covered = snap["_cls"].between(*VIS_YEARS)
    for c in VIS_FEATURES:
        snap[c] = np.where(covered, snap[c].fillna(0.0), np.nan)
    return snap.drop(columns=["_k", "_cls"])


def run_arms(step8, rows, feats, folds):
    yr = rows["last_college_season_year"].fillna(0)
    fold_of = pd.Series(np.nan, index=rows.index)
    score = {a: pd.Series(np.nan, index=rows.index) for a in ARM_SPECS}
    for train_end, val_start, val_end in folds:
        tr_m, va_m = yr <= train_end, yr.between(val_start, val_end)
        fold_of[va_m] = val_start
        tr, va = rows[tr_m], rows[va_m]
        nfl = tr[tr["nfl"]]
        for arm, (extra, seed) in ARM_SPECS.items():
            cols = feats + extra
            clf, b1 = zd._fit(step8, lgb.LGBMClassifier,
                              {**zd.STAGE1_PARAMS, "random_state": seed},
                              tr, cols, tr["nfl"].astype(int).values)
            reg, b2 = zd._fit(step8, lgb.LGBMRegressor,
                              {**zd.STAGE2_PARAMS, "random_state": seed},
                              nfl, cols, nfl["usage"].values)
            p1 = clf.predict_proba(zd._X(va, cols))[:, 1]
            m2 = np.clip(reg.predict(zd._X(va, cols)), 0, 1)
            score[arm][va.index] = p1 * m2
            log.info("fold %d %s: %d cols, stage 1 %d trees, stage 2 %d trees",
                     val_start, arm, len(cols), b1, b2)
    return score, fold_of


def bootstrap(E, arms, pairs, n_boot, levels=(95.0, 97.5), seed=0):
    rng = np.random.default_rng(seed)
    E = E.reset_index(drop=True)
    by_cls = [g.index.to_numpy() for _, g in E.groupby("cls")]
    point = zd._metrics(E, arms)
    boots = []
    for i in range(n_boot):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in by_cls])
        boots.append(zd._metrics(E.iloc[ix], arms))
        if (i + 1) % 100 == 0:
            log.info("bootstrap %d/%d", i + 1, n_boot)
    lines = []
    for a, b in pairs:
        for k in point[a]:
            d = np.array([bt[a][k] - bt[b][k] for bt in boots])
            row = {"Δ": f"{a} − {b}", "metric": k, "delta": point[a][k] - point[b][k]}
            for lv in levels:
                lo, hi = np.nanpercentile(d, [(100 - lv) / 2, 100 - (100 - lv) / 2])
                row[f"lo{lv:g}"], row[f"hi{lv:g}"] = lo, hi
            lines.append(row)
    return point, pd.DataFrame(lines)


def verdict(deltas):
    g = lambda a, b, k: deltas[(deltas["Δ"] == f"{a} − {b}") & (deltas["metric"] == k)].iloc[0]
    plac_auc = abs(g("base_43", "base", "auc_ob")["delta"])
    plac_rho = abs(g("base_43", "base", "rho_ob_nfl")["delta"])
    out = {}
    for arm in ("pff", "vis"):
        a, r = g(arm, "base", "auc_ob"), g(arm, "base", "rho_ob_nfl")
        c1 = a["lo97.5"] > 0
        c2 = a["delta"] > plac_auc
        c3 = not (r["delta"] < -plac_rho and r["hi97.5"] < 0)
        out[arm] = {"pass": bool(c1 and c2 and c3), "ci_excludes_0": bool(c1),
                    "beats_placebo": bool(c2), "rho_ok": bool(c3),
                    "d_auc": round(a["delta"], 4),
                    "ci97.5": f"[{a['lo97.5']:+.4f}, {a['hi97.5']:+.4f}]",
                    "d_rho": round(r["delta"], 4)}
    return {"placebo_auc": round(plac_auc, 4), "placebo_rho": round(plac_rho, 4), **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step8 = importlib.import_module("08_draft_models")
    folds, _ = step8._split_holdout(step8._expanding_window_folds())
    snap = step8.assemble_snapshot(include_derived=True, logger=step8.logger)
    snap = add_visits(add_side_pff(snap))
    rows, feats_all = zd._build_rows(step8, snapshot=snap, extra=PFF_FEATURES + VIS_FEATURES)
    feats = [c for c in feats_all if c not in PFF_FEATURES + VIS_FEATURES]
    for c in PFF_FEATURES + VIS_FEATURES:
        nfl = rows["nfl"]
        log.info("  %-28s coverage all %.1f%%  NFL %.1f%%  non-NFL %.1f%%", c,
                 100 * rows[c].notna().mean(), 100 * rows.loc[nfl, c].notna().mean(),
                 100 * rows.loc[~nfl, c].notna().mean())
    vis_any = rows["vis_teams"] > 0
    log.info("rows with >=1 visit: %d (NFL %d, non-NFL %d)", int(vis_any.sum()),
             int((vis_any & rows["nfl"]).sum()), int((vis_any & ~rows["nfl"]).sum()))

    score, fold_of = run_arms(step8, rows, feats, folds)
    base = rows[zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted"]].assign(val_start=fold_of)
    for a in ARM_SPECS:
        base[a] = score[a]
    base = base.dropna(subset=["val_start", "base"])
    base["val_start"] = base["val_start"].astype(int)
    base.to_parquet(args.out_dir / "oof_all_arms.parquet", index=False)
    # 00zf-format files (column 'pre' = arm, 'pre_43' = placebo) for the merge check
    for a in ("base", "pff", "vis", "both"):
        base.assign(pre=base[a], pre_43=base["base_43"])[
            zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted", "val_start", "pre", "pre_43"]
        ].to_parquet(args.out_dir / f"oof_00zd_{a}.parquet", index=False)

    E = zd._deployed(base).dropna(subset=["dep_impact"])
    E = E.drop_duplicates("player_id", keep="last")
    E["cls"] = E["last_college_season_year"].astype(int) + 1
    E = E[E["cls"].between(*EVAL_CLASSES)]
    E["board"] = E["hasan_rank"].notna()
    E["contrib"] = (E["usage"] >= zd.CONTRIB).astype(int)
    E["starter"] = (E["usage"] >= zd.STARTER).astype(int)
    ob = E[~E["board"]]
    log.info("scoring cohort %d (classes %d-%d): off-board %d, off-board NFL %d, contributors %d",
             len(E), E["cls"].min(), E["cls"].max(), len(ob), int(ob["nfl"].sum()),
             int(ob["contrib"].sum()))

    arms = list(ARM_SPECS) + ["dep_impact"]
    point, deltas = bootstrap(E, arms, PAIRS, args.n_boot)
    pd.set_option("display.width", 200)
    print("\nheadline metrics:")
    print(pd.DataFrame(point).T.round(4).to_string())
    print("\nΔ, paired bootstrap (97.5% = pre-registered Bonferroni interval):")
    print(deltas.round(4).to_string(index=False))
    print("\ntop-K vs the bottom of the board:")
    print(zd._topk_table(E, arms).to_string())
    v = verdict(deltas)
    print("\nVERDICT:", v)
    deltas.to_csv(args.out_dir / "deltas.csv", index=False)
    pd.Series(v, dtype=object).to_json(args.out_dir / "verdict.json")


if __name__ == "__main__":
    main()
