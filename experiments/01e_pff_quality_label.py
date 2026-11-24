"""
v4 rookie-value label: price every position's seasons on (snap share, PFF
grade quality). Does it beat the v3.1 label (EPA quality, QB/RB/WR/TE
only)? (pre-registered 2026-11-02)

Background: 01b found PFF grades add to snap-only value in predicting the
second contract at every position (Pearson +0.046, P 1.00) and subsume
EPA at the skill positions (pff_epa − epa +0.001). 01b's pre-registered
next step: build a v4 label with PFF-quality pricing at every position and
test it cross-fitted, as 00zq did for EPA.

v4 (src/models/rookie_value.py, build_label(quality="pff")):
    q = within-(pricing group x season) percentile of the shrunk PFF season
        grade, rate = (grade*snaps + 200*mu) / (snaps + 200), mu = the
        group-season snap-weighted mean grade. Tables as 01b (QB passing,
        RB rushing, WR/TE receiving grades_offense; OL offense_blocking
        grades_offense; DL/EDGE/LB/DB defense_summary grades_defense);
        snaps from offense_blocking / defense_summary (QB: max with
        passing_snaps). PFF id -> gsis via nflverse players.csv.
        Seasons without a PFF row for the player (incl. all of 2025, not
        downloaded) get the group's neutral q (percentile of mu), as EPA.
    price f_pos(share, q): monotone LightGBM for ALL nine groups (v3: skill
        groups only; others isotonic on share), same 2016-24 fit and
        2023-24 level recalibration. The depth-chart model (pre-2013
        seasons) uses q as its fourth feature at every group.
    The shrinkage constant (200 snaps) is fixed here, not tuned.

Arms, each built 5 times cross-fitted (--xval-fold 0..4: the pay curves of
fold k exclude every contract of validation players drafted 2011-20 whose
gsis hashes to fold k; each player's value comes from his own fold's run):
    none   snaps-only (skill groups LightGBM on share, others isotonic)
    epa    v3.1 production definition (incumbent)
    pff    v4 candidate
Cohort: drafted players, classes 2011-20, K/P excluded, gsis linked,
complete value_y14 in every arm. Outcome (00zp / 01b): second-contract pay
= max apy_cap_pct signed draft_year+3..+5, excluding the rookie deal and
round-1 option; 0 if none.

PRIMARY: within-group (NFL pricing group) Pearson of value_y14 vs the
outcome, n-weighted over the nine groups, pff − epa. Pearson because the
label is a dollar-valued sum (00zq).
Guard: Spearman among players WITH a second contract (outcome > 0), pff −
epa, must not be clearly worse: P(> 0) >= 0.2. This replaces 00zq's
whole-cohort Spearman guard, because 01b showed that statistic falls by
construction when grades reorder the tied no-contract players (−0.016,
P 0.00 there) while moving no money; the whole-cohort Spearman is still
reported.
Decision (feedback_bayesian_decisions): paired player bootstrap within
group (1000). ADOPT quality="pff" as the production label (step 7.5
default) if P(pff − epa > 0) > 0.5 AND the guard holds. Prior leans toward
the arm (01b). Cost: a manual PFF download each season (2025 neutral until
then); the q-pricing of OL/defense is new.
Reported: pff − none, epa − none; scopes non-skill / skill / paid > 1%;
per-group Pearson; mean value_y14 by group and by pick bin, v3 vs v4
(level recalibration should hold levels); largest markups / markdowns.
No placebo arm: builds are deterministic (fixed seeds); 00zp's seed-only
spread was <= 0.002.

RESULT (2026-11-02): ADOPT v4 (quality="pff"); production default switched.
2,333 drafted players, 2011-20 (80% with a 2nd contract), cross-fitted.
    within-group Pearson  pff − epa  +0.040 [+0.033, +0.047] P 1.00 (primary)
                          pff − none +0.050; epa − none +0.010
    guard Spearman (contract > 0) pff − epa +0.018 [+0.014, +0.023] P 1.00
    whole-cohort Spearman pff − epa +0.013 P 1.00 (did not fall, unlike 01b)
    non-skill +0.048, skill +0.025 (P 1.00); paid > 1% +0.077 Pearson,
    +0.061 Spearman.
Every group improves on Pearson: OL .763 → .841, EDGE .809 → .865, DL .854
→ .891, LB .810 → .853, DB .768 → .803, WR .863 → .906, TE .842 → .872,
QB .809 → .828, RB .856 → .858 (RB Spearman(contract>0) .765 → .761, flat).
Levels: mean value_y14 (drafted 2011-22) falls 9-10% for EDGE/OL, 4% for
DL/WR, ~0 elsewhere — the 2023-24 recalibration matches veterans, and
drafted players' seasons grade below veterans' at equal snaps. By pick,
top 10 +1%, picks 11-262 −3..−7%.
Markups: Mack, Von Miller, Zack Martin, Ramczyk, Mike Evans (+0.09..+0.14
of cap over y1-4); markdowns: Newton, Kaepernick, Winston, Da'Ron Payne,
Trubisky, Ifedi.

Usage: 01e_pff_quality_label.py [--n-boot 1000] [--build-only]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR  # noqa: E402
from src.models.rookie_value import build_label  # noqa: E402

OUT = PROCESSED_DIR / "_v4_label_20261102"
ARMS = ["none", "epa", "pff"]
K = 5
CLASSES = (2011, 2020)
SKILL = ["QB", "RB", "WR", "TE"]

spec = importlib.util.spec_from_file_location("b", ROOT / "experiments/01b_pff_quality_check.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def path(arm, k):
    return OUT / f"labels_{arm}_xval{k}.parquet"


def build_all():
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for k in range(K):
            if not path(arm, k).exists():
                print(f"\n=== build {arm} fold {k} ===", flush=True)
                build_label(path(arm, k), quality=arm, xval_fold=k, xval_k=K)


def load_arm(arm):
    out = []
    for k in range(K):
        f = pd.read_parquet(path(arm, k))
        h = pd.util.hash_pandas_object(f["gsis"].fillna(""), index=False).to_numpy() % K
        out.append(f[h == k].set_index("pfr_id")["value_y14"])
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    build_all()
    if args.build_only:
        return
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(0)
    base = pd.read_parquet(path("epa", 0))
    df = base[base["drafted"] & base["draft_year"].between(*CLASSES) & base["gsis"].notna()
              & ~base["group"].isin(["K", "P"])].drop_duplicates("gsis").copy()
    df["y"] = df["gsis"].map(b.second_contract(df)).fillna(0.0)
    for a in ARMS:
        df[a] = df["pfr_id"].map(load_arm(a))
    df = df.dropna(subset=ARMS).reset_index(drop=True)
    print(f"cohort {len(df)} drafted players, classes {CLASSES}; with a 2nd contract {(df['y'] > 0).mean():.2f}")

    scopes = {"all": df, "non-skill": df[~df["group"].isin(SKILL)], "skill": df[df["group"].isin(SKILL)],
              "contract>0": df[df["y"] > 0], "paid>1%": df[df["y"] > 0.01]}
    pairs = [("pff", "epa"), ("pff", "none"), ("epa", "none")]
    out = []
    for sc, d in scopes.items():
        d = d.reset_index(drop=True)
        for mname, f in (("pearson", pearsonr), ("spearman", spearmanr)):
            out.append(b.boot(d, pairs, args.n_boot, rng, f).assign(scope=sc, metric=mname))
    D = pd.concat(out, ignore_index=True)
    print("\nΔ within-group correlation of value_y14 with 2nd-contract pay (paired bootstrap):")
    print(D.round(4).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)

    print("\nper group Pearson / Spearman(contract>0):")
    per = {}
    for g, G in df.groupby("group"):
        Gp = G[G["y"] > 0]
        per[g] = {**{f"P_{a}": pearsonr(G[a], G["y"])[0] for a in ARMS},
                  **{f"S+_{a}": spearmanr(Gp[a], Gp["y"])[0] for a in ARMS}, "n": len(G)}
    print(pd.DataFrame(per).T.round(3).to_string())

    sel = lambda s, m, dd: D[(D["scope"] == s) & (D["metric"] == m) & (D["Δ"] == dd)].iloc[0]
    p, g = sel("all", "pearson", "pff − epa"), sel("contract>0", "spearman", "pff − epa")
    adopt = bool(p["P(>0)"] > 0.5 and g["P(>0)"] >= 0.2)
    print(f"\nDECISION: pff − epa Pearson {p['delta']:+.4f} [{p['lo95']:+.4f}, {p['hi95']:+.4f}] "
          f"P(>0) {p['P(>0)']:.3f}; guard Spearman(contract>0) {g['delta']:+.4f} P(>0) {g['P(>0)']:.3f} "
          f"→ {'ADOPT v4 (quality=pff)' if adopt else 'KEEP v3.1 (quality=epa)'}")

    full = {a: pd.concat([pd.read_parquet(path(a, k)).assign(fold=k) for k in range(K)])
            for a in ("epa", "pff")}
    lv = base[base["drafted"] & base["draft_year"].between(2011, 2022)][["pfr_id", "group", "overall_pick"]].copy()
    for a in ("epa", "pff"):
        lv[a] = lv["pfr_id"].map(full[a][full[a]["fold"] == 0].set_index("pfr_id")["value_y14"])
    lv = lv.dropna()
    print("\nmean value_y14, drafted classes 2011-22 (fold-0 builds), v3 epa vs v4 pff:")
    print(lv.groupby("group")[["epa", "pff"]].mean().round(4).to_string())
    lv["pick_bin"] = pd.cut(lv["overall_pick"], [0, 10, 32, 64, 100, 160, 262])
    print(lv.groupby("pick_bin", observed=True)[["epa", "pff"]].mean().round(4).to_string())
    df["d"] = df["pff"] - df["epa"]
    show = ["player_name", "draft_year", "group", "overall_pick", "epa", "pff", "d", "y"]
    print("\nlargest v4 markups / markdowns vs v3 (value_y14):")
    print(df.nlargest(10, "d")[show].round(4).to_string(index=False))
    print(df.nsmallest(10, "d")[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
