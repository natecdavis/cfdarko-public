"""
Position-normalized usage target for the global (8b) model.

RESULT (2026-11-01): H1 PASS, H2 PASS → recommend norm over raw.
norm − raw, 3-seed mean (max |seed placebo|):
    within  value +0.041 (0.005)  late_snap +0.026 (0.004)  late_apy +0.014 (0.002)
    cross   value +0.022 (0.005)  late_snap +0.012 (0.003)  late_apy +0.009 (0.002)
Every seed agrees. Biggest within-group gains: RB, DL, OL, TE. The rescale
leaves within-position target order unchanged, yet within-position ranking
improves: on the raw scale the pooled model spends its splits on telling
positions apart. Player-bootstrap CIs exclude 0 for within value/late_snap
and cross value only; late_apy passes the placebo bar but its bootstrap CI
includes 0.
Reference, NOT a decision contrast (no seed placebo for it): the deployed
APY-trained head beats norm on late_apy (within 0.327 vs 0.317, cross
0.341 vs 0.330) and is beaten on value (0.306 vs 0.321), which includes
usage. Both usage arms put ~4 QBs in each class's top 100 vs 8 for APY.
Normalization decides raw vs norm for a usage target. It does NOT show that
usage should replace APY as the shipped target.
Outputs: data/processed/_usage_posnorm_20261101/ (report.txt).

Question (2026-11-01): if the draft model is trained on y3-4 snap share
(usage_y34), should that target be raw snap share, or snap share relative
to what a starter at the position plays? Starting QBs and OL play ~100% of
snaps while starting RB / DL / WR rotate (60-80%), so on the raw scale a
pooled model is told a rotational DL starter is "worse" than an average OL
starter. Per-position heads are unaffected by any per-position rescale
(within-position order is identical); the pooled 8b global model, which is
what ships for every position, is where it can matter.

Arms (all refit the 8b global head at the deployed global/impact_apy tuned
hyperparameters, same feature set, same folds):
    raw    target = usage_y34 (0 for non-NFL players)
    norm   target = usage_y34 / starter benchmark, where the benchmark is the
           80th percentile of usage_y34 among bridge players with usage > 0,
           per (value_group × usage_source), settled classes only. A pure
           per-group rescale: no cap, zeros stay zero, within-position order
           of the target is unchanged.
    seeds  42 / 43 / 44 for each arm. raw seed pairs are the placebo.
Both targets are then multiplied by one constant so their overall std
matches impact_target's: reg_alpha and min_split_gain are target-scale
dependent, and this keeps the APY-tuned hyperparameters acting the same.
Labels: bridge rows drafted <= END_YEAR-4 (y3-4 played); non-bridge rows
with last college season <= END_YEAR-5 get 0; everything later is NaN
(usage_y34 counts unplayed seasons as 0, so later classes are not labels).
Reference row (not a decision arm): the deployed APY-trained 8b OOF.

Scoring (bridge players, K/P excluded, settled classes, via 00u's cohort):
    within   00u statistic: Spearman within position group, averaged over
             groups, vs value / late_snap / late_apy.
    cross    Spearman across ALL positions within draft class (averaged over
             classes), vs the same referees. This is where the target scale
             can act. Note each referee carries its own cross-position
             currency: late_snap is raw snaps (leans raw), value is a
             within-group percentile (leans norm), late_apy is market pay
             (neither arm's target; a cross-position value measure).

PRE-REGISTERED BAR (written before any arm was fit):
  H1 (does normalization help the pooled model learn positions better?)
     norm − raw, seed-matched, mean over the 3 seeds, on the WITHIN statistic:
     PASS if Δ > 0 on value AND late_snap, and both exceed the largest
     |raw-vs-raw placebo| Δ on that referee.
     Expectation: ≈ tie (the rescale doesn't change within-position order).
  H2 (cross-position ordering) — decision referee late_apy:
     PASS if norm − raw cross Δ on late_apy > 0 and exceeds the largest
     |placebo| cross Δ. value and late_snap cross Δs are reported but are
     NOT decision metrics (each leans toward one arm by construction).
  Ship recommendation for a usage-trained model: norm if H2 passes and H1
  is not a loss beyond placebo; otherwise raw (simpler).

Usage:
    00zl_usage_position_norm.py --fit --arm raw|norm --seed S --out DIR
    00zl_usage_position_norm.py --report DIR
"""
import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, END_YEAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
BENCH_Q = 0.80
SEEDS = (42, 43, 44)
REFS = ["value", "late_snap", "late_apy"]
N_BOOT = 500


def build_targets(frame: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw / norm usage targets (unscaled) on the 8b training frame, plus the
    benchmark table. The frame supplies last_college_season_year (labels has
    it for bridge rows only); labels supply usage_y34 / usage_source / group."""
    extra = (labels.dropna(subset=["player_id"]).drop_duplicates("player_id")
             [["player_id", "usage_y34", "usage_source", "value_group"]])
    lab = frame[["player_id", "is_bridge", "draft_year", "last_college_season_year"]].merge(
        extra, on="player_id", how="left")
    bridge = lab["is_bridge"].astype(bool)
    settled_b = bridge & (lab["draft_year"] <= END_YEAR - 4)
    settled_nb = ~bridge & (lab["last_college_season_year"] <= END_YEAR - 5)
    raw = pd.Series(np.nan, index=lab.index)
    raw[settled_b] = lab.loc[settled_b, "usage_y34"]
    raw[settled_nb] = 0.0

    pos_rows = (settled_b & (lab["usage_y34"] > 0) & lab["value_group"].notna()
                & lab["usage_source"].notna())
    bench = (lab[pos_rows].drop_duplicates("player_id")
             .groupby(["value_group", "usage_source"])["usage_y34"]
             .quantile(BENCH_Q).rename("bench").reset_index())
    b = lab[["value_group", "usage_source"]].merge(
        bench, how="left", on=["value_group", "usage_source"])["bench"].to_numpy()
    norm = raw.copy()
    has = (settled_b & raw.notna() & (raw > 0)).to_numpy()
    norm[has] = raw[has] / b[has]   # NaN benchmark → NaN target (dropped)
    return pd.DataFrame({"usage_raw": raw.values, "usage_norm": norm.values},
                        index=frame.index), bench


def _fit(args):
    g = importlib.import_module("08b_global_draft_model")
    g.LGBM_PARAMS["random_state"] = args.seed
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    df, feature_cols = g.build_global_training_frame(labels)
    tgt, bench = build_targets(df, labels)
    df = df.join(tgt)
    col = f"usage_{args.arm}"
    # One constant so the arm's std matches impact_target's (APY-tuned params).
    scale = float(df["impact_target"].std() / df[col].std())
    df["usage_target"] = df[col] * scale
    logging.info("arm=%s seed=%d scale=%.5f n_labelled=%d", args.arm, args.seed,
                 scale, int(df["usage_target"].notna().sum()))
    tuned = json.loads(g.GLOBAL_TUNED_PARAMS_PATH.read_text()).get("global/impact_apy")
    model, metrics, fc, oof, _ = g._train_one_global(
        df, feature_cols, "usage_target", "impact_apy", tuned)
    oof = oof.rename(columns={"position_group": "pos_group"})
    oof["model_name"] = "impact"
    oof["pred"] = oof["pred"] / scale
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(out / f"oof_{args.arm}_s{args.seed}.parquet", index=False)
    bench.to_csv(out / "bench.csv", index=False)
    json.dump({"scale": scale, **metrics}, open(out / f"metrics_{args.arm}_s{args.seed}.json", "w"),
              indent=1)


def _within(df, col, ref):
    rhos = []
    for _, sub in df.groupby("value_group"):
        s = sub[[col, ref]].dropna()
        if len(s) >= 10:
            rhos.append(spearmanr(s[col], s[ref])[0])
    return float(np.mean(rhos))


def _cross(df, col, ref):
    rhos = []
    for _, sub in df.groupby("draft_year"):
        s = sub[[col, ref]].dropna()
        if len(s) >= 30:
            rhos.append(spearmanr(s[col], s[ref])[0])
    return float(np.mean(rhos))


STATS = {"within": _within, "cross": _cross}


def _report(args):
    u = importlib.import_module("00u_value_target_referee_ab")
    df = u._cohort().reset_index(drop=True)
    d = Path(args.report)
    arms = {}
    dep = pd.read_parquet(PROCESSED_DIR / "global_draft_models/global_oof_predictions.parquet")
    dep = dep[dep["model_name"] == "impact_apy"].drop_duplicates("player_id", keep="last")
    arms["deployed_apy"] = dep.set_index("player_id")["pred"]
    for a in ("raw", "norm"):
        for s in SEEDS:
            p = d / f"oof_{a}_s{s}.parquet"
            if p.exists():
                o = pd.read_parquet(p).drop_duplicates("player_id", keep="last")
                arms[f"{a}_s{s}"] = o.set_index("player_id")["pred"]
    for name, ser in arms.items():
        df[name] = df["player_id"].map(ser)
    # Common rows: every arm has a prediction.
    df = df[df[list(arms)].notna().all(axis=1)].reset_index(drop=True)
    print(f"cohort: {len(df)} bridge players, classes "
          f"{int(df.draft_year.min())}-{int(df.draft_year.max())}; "
          f"late_* rows {int(df.late_snap.notna().sum())}")

    pt = {(st, a, r): STATS[st](df, a, r) for st in STATS for a in arms for r in REFS}
    print("\nPoint estimates (Spearman):")
    print(pd.DataFrame({f"{st}|{r}": [round(pt[(st, a, r)], 3) for a in arms]
                        for st in STATS for r in REFS}, index=list(arms)).to_string())

    # Paired contrasts: norm−raw per seed; placebo raw−raw seed pairs.
    contrasts = [(f"norm_s{s}", f"raw_s{s}") for s in SEEDS]
    placebos = [("raw_s43", "raw_s42"), ("raw_s44", "raw_s42"), ("raw_s44", "raw_s43"),
                ("norm_s43", "norm_s42"), ("norm_s44", "norm_s42"), ("norm_s44", "norm_s43")]
    contrasts = [c for c in contrasts if all(x in arms for x in c)]
    placebos = [c for c in placebos if all(x in arms for x in c)]

    rng = np.random.default_rng(0)
    groups = {g: sub.index.to_numpy() for g, sub in df.groupby("value_group")}
    boot_idx = [np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in groups.values()])
                for _ in range(N_BOOT)]
    rows = []
    for st in STATS:
        for r in REFS:
            mean_d = np.mean([pt[(st, a, r)] - pt[(st, b, r)] for a, b in contrasts])
            pl = [pt[(st, a, r)] - pt[(st, b, r)] for a, b in placebos]
            # Bootstrap of the seed-mean contrast (players resampled within group).
            bd = []
            for ix in boot_idx[: (N_BOOT if st == "within" else N_BOOT // 2)]:
                sub = df.loc[ix]
                bd.append(np.mean([STATS[st](sub, a, r) - STATS[st](sub, b, r)
                                   for a, b in contrasts]))
            rows.append({"stat": st, "ref": r,
                         "norm−raw (3-seed mean)": round(mean_d, 4),
                         "boot 95%": f"[{np.percentile(bd, 2.5):+.4f}, {np.percentile(bd, 97.5):+.4f}]",
                         "per-seed": " ".join(f"{pt[(st, a, r)] - pt[(st, b, r)]:+.4f}"
                                              for a, b in contrasts),
                         "max|placebo|": round(max(abs(x) for x in pl), 4)})
    res = pd.DataFrame(rows)
    print("\nContrasts:")
    print(res.to_string(index=False))

    # Per-group within contrast (value referee), seed-mean.
    print("\nPer-group within Δ (norm−raw, seed mean):")
    pg = {}
    for gname, sub in df.groupby("value_group"):
        pg[gname] = {r: round(np.mean([spearmanr(*sub[[a, r]].dropna().T.values)[0]
                                       - spearmanr(*sub[[b, r]].dropna().T.values)[0]
                                       for a, b in contrasts]), 3)
                     if sub[r].notna().sum() >= 10 else np.nan for r in REFS}
    print(pd.DataFrame(pg).T.to_string())

    # Cross-position mix: share of each group in the top 32 / top 100 per class.
    print("\nPosition mix of each class's top 100 (mean count per class):")
    mix = {}
    for a in ["deployed_apy"] + [f"{x}_s42" for x in ("raw", "norm") if f"{x}_s42" in arms]:
        top = df.sort_values(a, ascending=False).groupby("draft_year").head(100)
        mix[a] = (top.groupby("value_group").size() / top["draft_year"].nunique()).round(1)
    print(pd.DataFrame(mix).fillna(0).to_string())
    res.to_csv(d / "contrasts.csv", index=False)

    v = {(r["stat"], r["ref"]): r for r in rows}
    h1 = all(v[("within", r)]["norm−raw (3-seed mean)"] > v[("within", r)]["max|placebo|"]
             for r in ("value", "late_snap"))
    h2 = v[("cross", "late_apy")]["norm−raw (3-seed mean)"] > v[("cross", "late_apy")]["max|placebo|"]
    h1_loss = any(v[("within", r)]["norm−raw (3-seed mean)"] < -v[("within", r)]["max|placebo|"]
                  for r in ("value", "late_snap"))
    print(f"\nH1 within: {'PASS' if h1 else 'FAIL'}   (loss beyond placebo: {h1_loss})")
    print(f"H2 cross late_apy: {'PASS' if h2 else 'FAIL'}")
    print("Recommendation:", "norm" if (h2 and not h1_loss) else "raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--arm", choices=["raw", "norm"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PROCESSED_DIR / "_usage_posnorm_20261101"))
    ap.add_argument("--report")
    args = ap.parse_args()
    if args.fit:
        _fit(args)
    elif args.report:
        _report(args)


if __name__ == "__main__":
    main()
