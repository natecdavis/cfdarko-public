"""
Value-target A/B, scored against later-career referees — and the standing
yardstick for any future impact-head A/B (feature, label or objective change).

RESULT (2026-10-29): value target REJECTED as a training target. As a label it
beats the APY label 9/9 on both late referees, but trained on it the impact
head loses (−0.070 / −0.052 vs APY, wins 1/9 and 2/9; placebo −0.021 /
−0.019): being NFL vs not explains 64% of value_target's variance vs 24% of
APY's, so the squared-error head learns "made the NFL" instead of ranking NFL
players. Pushed through the APY curve it ties (−0.008 / −0.005). Kept as a
REFEREE, which is what it is good at.

Why a separate harness. The value target (config.IMPACT_TARGET_SOURCE) changes
what the impact head is trained to predict, so neither arm's MAE nor its
own-target fold Spearman is comparable across arms — and scoring both against
the incumbent label (label_apy_raw) is what wrongly rejected the July label
gate. Both arms are scored against outcomes neither target contains:

    value       value_target on settled classes (draft <= END_YEAR -
                VALUE_CONTRACT_LAST_YEAR_OFFSET): best APY + best guarantees
                by y5 + y3-y4 usage, as a within-(group × class) percentile.
                The widest cohort — every settled class the OOF folds cover.
    late_snap   mean snap share, NFL seasons 5-7 (draft_year+4 .. +6)
    late_apy    best APY % of cap signed draft_year+6 .. +8
                (late_* only for drafted 2016-2019: snap data starts 2016 and
                y5-y7 must be observable)

Bridge players only, K/P excluded, grouped by the model's position group
(value_group). Statistic: Spearman within position group, averaged over
groups; paired bootstrap over players within group. A pair is scored on
the rows where both the prediction and that referee exist.

Noise: the bootstrap resamples players, not model seeds, so its intervals
UNDERSTATE noise. The seed-only placebo moved −0.009 / −0.026 / −0.039
(value / late_snap / late_apy) and its late_apy interval excluded 0. An arm
must beat the placebo's spread, not just show an interval excluding 0. The
value referee is the tightest (±~0.015 vs ±~0.03 for late_*). It shares the
best-APY component with the APY target, so it leans slightly toward
APY-trained heads; for comparisons between APY-trained arms that cancels.

Label fixes check (2026-10-29): July OOF vs the same hyperparameters on the
refreshed contracts + stable sort + 2023 cutoff: −0.005 / −0.005 / −0.020,
all within placebo — neutral, shipped as correctness fixes.

BASELINE WARNING (2026-10-30): the deployed impact_score is NOT the step-8
per-position head — config.GLOBAL_SCORE_OVERRIDES takes the step-8b global
model for every position. Compare against
global_draft_models/global_oof_predictions.parquet (model_name impact_apy;
rename position_group → pos_group, set model_name "impact"), or arms that
only change step 8 measure a model that does not ship.

Reading a future A/B: run every arm through step 8 at the same hyperparameters,
copy draft_models/oof_predictions.parquet aside after each, include an A/A
placebo arm (LGBM_PARAMS["random_state"] 42 → 43), and pass them as --oof.
`value` has the most power; the late_* referees are the independent check.

Modes
    --labels                  label columns vs referees (sanity: should
                              reproduce the 2026-10-29 scratch result, composite
                              ≈ 0.78 / 0.68 vs legacy ≈ 0.66 / 0.54)
    --oof NAME=PATH ...       impact-head OOF predictions per arm
                              (draft_models/oof_predictions.parquet copied
                              aside after each arm's step-8 run). The FIRST arm
                              is the base; include an A/A placebo arm.

Pre-registered ship bar (value arm vs apy arm, --oof mode):
    1. Δ > 0 on BOTH referees with the 95% bootstrap interval excluding 0;
    2. wins in >= 6 of the position groups on at least one referee;
    3. the placebo arm's Δ interval contains 0.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, RAW_DIR, END_YEAR, VALUE_CONTRACT_LAST_YEAR_OFFSET
from src.scrapers.nfl_data import (
    SEASONAL_ROSTERS_CACHE, _build_pfr_to_gsis_crosswalk, build_bridge_gsis_crosswalk,
)

LATE_COHORT = (2016, 2019)
REFS = ["value", "late_snap", "late_apy"]
N_BOOT = 1000


def _cohort() -> pd.DataFrame:
    lab = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    settled_max = END_YEAR - VALUE_CONTRACT_LAST_YEAR_OFFSET
    df = lab[(lab["is_bridge"] == True) & lab["pfr_id"].notna()
             & (lab["draft_year"] <= settled_max)].drop_duplicates("pfr_id").copy()
    df = df[df["value_group"].notna() & ~df["value_group"].isin(["K", "P"])]
    df["draft_year"] = df["draft_year"].astype(int)
    df["value"] = df["value_target"]
    late = df["draft_year"].between(*LATE_COHORT)

    snap = pd.read_parquet(RAW_DIR / "snap_counts.parquet")
    snap["share"] = snap[["offense_pct", "defense_pct"]].max(axis=1)
    snap_by = snap.groupby(["pfr_player_id", "season"])["share"].max().to_dict()
    df["late_snap"] = [np.mean([snap_by.get((p, y), 0.0) for y in range(d + 4, d + 7)])
                       if is_late else np.nan
                       for p, d, is_late in zip(df["pfr_id"], df["draft_year"], late)]

    bp = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    xwalk = build_bridge_gsis_crosswalk(bp, pd.read_parquet(SEASONAL_ROSTERS_CACHE),
                                        base=_build_pfr_to_gsis_crosswalk())
    c = pd.read_parquet(RAW_DIR / "nfl_contracts.parquet").dropna(subset=["gsis_id"])
    by_gsis = {g: grp for g, grp in c.groupby("gsis_id")}

    def late_apy(pfr, d):
        grp = by_gsis.get(xwalk.get(pfr))
        if grp is None:
            return np.nan
        s = grp[grp["year_signed"].between(d + 6, d + 8)]["apy_cap_pct"]
        return float(s.max()) if len(s) else 0.0

    df["late_apy"] = [late_apy(p, d) if is_late else np.nan
                      for p, d, is_late in zip(df["pfr_id"], df["draft_year"], late)]
    print(f"cohort: {len(df)} settled bridge players (draft <= {settled_max}); late "
          f"referees on {int(late.sum())} drafted {LATE_COHORT[0]}-{LATE_COHORT[1]}, "
          f"late_apy linked for {df.loc[late, 'late_apy'].notna().mean():.1%}")
    return df


def _score(df: pd.DataFrame, cols: list[str], rng: np.random.Generator):
    """Point estimates and bootstrap draws of the group-averaged Spearman."""
    groups = {g: sub.index.to_numpy() for g, sub in df.groupby("value_group")}

    def stat(idx_by_group):
        out = {}
        for col in cols:
            for ref in REFS:
                rhos = []
                for ix in idx_by_group.values():
                    sub = df.loc[ix, [col, ref]].dropna()
                    if len(sub) >= 10:
                        rhos.append(spearmanr(sub[col], sub[ref])[0])
                out[(col, ref)] = np.mean(rhos)
        return out

    point = stat(groups)
    per_group = {(col, ref, g): spearmanr(*df.loc[ix, [col, ref]].dropna().T.values)[0]
                 if df.loc[ix, [col, ref]].dropna().shape[0] >= 10 else np.nan
                 for col in cols for ref in REFS for g, ix in groups.items()}
    boots = [stat({g: rng.choice(ix, len(ix), replace=True) for g, ix in groups.items()})
             for _ in range(N_BOOT)]
    return point, per_group, boots, sorted(groups)


def _report(df, cols, base, rng):
    point, per_group, boots, groups = _score(df, cols, rng)
    rows = []
    for col in cols:
        r = {"arm": col}
        for ref in REFS:
            d = np.array([b[(col, ref)] - b[(base, ref)] for b in boots])
            delta = point[(col, ref)] - point[(base, ref)]
            wins = sum(per_group[(col, ref, g)] > per_group[(base, ref, g)] for g in groups)
            r[ref] = round(point[(col, ref)], 3)
            r[f"Δ{ref}"] = (f"{delta:+.3f} [{np.percentile(d, 2.5):+.3f}, "
                            f"{np.percentile(d, 97.5):+.3f}]  wins {wins}/{len(groups)}")
        rows.append(r)
    print(pd.DataFrame(rows).set_index("arm").to_string())
    print("\nper group:")
    print(pd.DataFrame({f"{c}|{ref}": [round(per_group[(c, ref, g)], 3) for g in groups]
                        for c in cols for ref in REFS}, index=groups).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--oof", nargs="*", default=[], metavar="NAME=PATH")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--within-class", action="store_true",
                    help="rank predictions and referees within (group × draft class) "
                         "before scoring. Use for arms whose OOF scale differs "
                         "between CV folds: pooled-across-classes Spearman then "
                         "penalizes fold-to-fold scale, which deployment never sees")
    ap.add_argument("--subset", choices=["all", "board", "offboard"], default="all",
                    help="restrict the cohort to players with / without a consensus "
                         "board rank (hasan_ranks.parquet; boards start in 2014)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    df = _cohort()
    if args.subset != "all":
        board = set(pd.read_parquet(PROCESSED_DIR / "hasan_ranks.parquet")["player_id"])
        on = df["player_id"].isin(board)
        df = df[on if args.subset == "board" else ~on].copy()
        print(f"subset {args.subset}: {len(df)} players")
    within_class = args.within_class

    if args.labels:
        df["label_value"] = df["value_target"]   # the column under test, not the referee
        cols = ["impact_apy_target", "label_value", "best_apy_by_y5", "usage_y34"]
        _report(df, cols, "impact_apy_target", rng)

    if args.oof:
        cols = []
        for spec in args.oof:
            name, path = spec.split("=", 1)
            oof = pd.read_parquet(path)
            oof = oof[oof["model_name"] == "impact"].drop_duplicates("player_id", keep="last")
            df[name] = df["player_id"].map(oof.set_index("player_id")["pred"])
            print(f"{name}: OOF coverage {df[name].notna().mean():.1%} of cohort")
            cols.append(name)
        if within_class:
            for c in cols + REFS:
                df[c] = df.groupby(["value_group", "draft_year"])[c].rank(pct=True)
            print("scoring within (group × draft class)")
        _report(df, cols, cols[0], rng)


if __name__ == "__main__":
    main()
