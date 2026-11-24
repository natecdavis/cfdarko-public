"""
Do PFF NFL grades add to snap-based rookie value in predicting the second
contract, at every position? (pre-registered 2026-11-02)

Background: 00zp showed quality decides HOW MUCH a second contract pays
(snaps decide whether there is one); 00zq folded EPA quality into the
rookie-value label, but only for QB/RB/WR/TE — OL, DL, EDGE, LB, DB snaps
are priced on share alone. PFF's historical NFL grades are now
downloadable (pipeline/00_pff_download.py --league nfl).

Outcome (as 00zp): second-contract pay = max apy_cap_pct of contracts
signed draft_year+3 .. +5, excluding the rookie deal and the round-1
option; 0 if none. Drafted players, classes 2011-20, K/P excluded.
Base value: the production label rebuilt with quality = "none"
(snap-only price, v3.1 NFL-position pricing), value_y1..y4.
PFF grade by NFL position group (label `group`), season-level:
    QB passing_summary grades_offense      RB rushing_summary grades_offense
    WR/TE receiving_summary grades_offense OL offense_blocking grades_offense
    DL/EDGE/LB/DB defense_summary grades_defense
weights = that season's snaps (offense_blocking snap_counts_offense /
defense_summary snap_counts_defense; QB passing_snaps if larger).
grade_y14, grade_y34 = snap-weighted mean of season grades over rookie
seasons 1-4 / 3-4 (NaN without PFF snaps). Players linked PFF id → gsis
via nflverse players.csv.

Arms — pooled LightGBM, group categorical, monotone +1 on value and grade
features, leave-one-class-out, 3 seeds (as 00zp):
    base      value_y1..y4
    base_alt  same, seeds +100 (placebo)
    pff       base + grade_y14, grade_y34
    epa       base + epa_y14, epa_y34 (QB/RB/WR/TE only, as 00zp)
    pff_epa   base + both (QB/RB/WR/TE only)
Score: Pearson of OOF prediction vs outcome within group, n-weighted mean
(00zq's adopted label metric; the label is a dollar-valued mean).
PRIMARY: pff − base, all groups. P(> 0) > 0.5 (paired bootstrap of players
within group, 1000) → build a v4 label with PFF-quality pricing at every
position (its own pre-registered cross-fitted test, as 00zq).
Reported: the same on non-skill groups (OL/DL/EDGE/LB/DB, where EPA does
not exist) and skill groups (pff vs epa, pff_epa vs epa); Spearman;
players paid > 1% of cap; per-group Pearson; partial Spearman of grade_y34
with the outcome given value_y14 by group; PFF coverage of the cohort.

RESULT (2026-11-02): BUILD a v4 label with PFF-quality pricing.
2,333 drafted players, 2011-20, 80% with a 2nd contract; PFF grades for
80-95% of each group (2006-2024 seasons downloaded; 2025 not yet).
Pearson (within group, n-weighted), pff − base:
    all        +0.046 [+0.033, +0.058] P 1.00 (placebo −0.0002)
    non-skill  +0.049 [+0.032, +0.065] P 1.00 (OL/DL/EDGE/LB/DB)
    skill      +0.041 P 1.00; epa − base +0.029; pff − epa +0.012 (P 0.97);
               pff_epa − epa +0.001 (P 0.58): grades subsume EPA
    paid > 1%  +0.095 [+0.068, +0.125]; Spearman +0.079
Every group improves (Pearson): OL .780 → .842, EDGE .812 → .882, DB
.789 → .843, LB .819 → .853, DL .872 → .896, QB .827 → .883, WR .846 →
.901. Partial Spearman of grade_y34 with 2nd-contract pay given value_y14:
0.31-0.44 at every position, QB 0.53 (EPA 0.26).
Whole-cohort Spearman falls (−0.016, P 0.00), as in 00zp: the bottom of
each group (no second contract, many ties) is ordered by snaps, and
grades reshuffle it without changing the money. The label is a dollar
mean, so Pearson is the pre-registered score.

Usage: 01b_pff_quality_check.py [--n-boot 1000]
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.models.rookie_value import QUAL_CACHE, build_label  # noqa: E402

OUT = PROCESSED_DIR / "_pff_quality_20261102"
PFF = RAW_DIR / "pff_nfl"
CLASSES = (2011, 2020)
SEEDS = [42, 43, 44]
SKILL = ["QB", "RB", "WR", "TE"]
VALUE = ["value_y1", "value_y2", "value_y3", "value_y4"]
GRADE = ["grade_y14", "grade_y34"]
EPA = ["epa_y14", "epa_y34"]
ARMS = {"base": (VALUE, SEEDS), "base_alt": (VALUE, [s + 100 for s in SEEDS]),
        "pff": (VALUE + GRADE, SEEDS), "epa": (VALUE + EPA, SEEDS), "pff_epa": (VALUE + GRADE + EPA, SEEDS)}
SKILL_ONLY = {"epa", "pff_epa"}
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)
TABLE = {"QB": ("passing_summary", "grades_offense"), "RB": ("rushing_summary", "grades_offense"),
         "WR": ("receiving_summary", "grades_offense"), "TE": ("receiving_summary", "grades_offense"),
         "OL": ("offense_blocking", "grades_offense")}
DEF = ("defense_summary", "grades_defense")


def _read(year, name):
    f = PFF / str(year) / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else None


def pff_seasons(years):
    """One row per (pff player_id, season, group-table): grade and snaps."""
    rows = []
    for y in years:
        ob, de = _read(y, "offense_blocking"), _read(y, "defense_summary")
        osn = ob.set_index("player_id")["snap_counts_offense"] if ob is not None else pd.Series(dtype=float)
        for g, (name, col) in {**TABLE, **{g: DEF for g in ["DL", "EDGE", "LB", "DB"]}}.items():
            t = _read(y, name)
            if t is None:
                continue
            t = t.drop_duplicates("player_id")
            if name == "defense_summary":
                snaps = t["snap_counts_defense"]
            else:
                snaps = t["player_id"].map(osn)
                if name == "passing_summary":
                    snaps = np.fmax(snaps, t["passing_snaps"])
            rows.append(pd.DataFrame({"pff_id": t["player_id"], "season": y, "group": g,
                                      "grade": t[col], "snaps": snaps}))
    return pd.concat(rows, ignore_index=True)


def grade_features(lab):
    years = range(CLASSES[0], CLASSES[1] + 4)
    s = pff_seasons(years)
    pl = pd.read_csv(RAW_DIR / "rookie_value" / "nflverse_players.csv", low_memory=False,
                     usecols=["gsis_id", "pff_id"]).dropna()
    pl = pl.assign(pff_id=pl["pff_id"].astype(int)).drop_duplicates("pff_id")
    s = s.merge(pl, on="pff_id").merge(lab[["gsis", "draft_year", "group"]],
                                         left_on=["gsis_id", "group"], right_on=["gsis", "group"])
    s["y"] = s["season"] - s["draft_year"] + 1
    s = s[s["snaps"].fillna(0) > 0]

    def wmean(d):
        return np.average(d["grade"], weights=d["snaps"])
    f = {}
    for name, (a, b) in {"grade_y14": (1, 4), "grade_y34": (3, 4)}.items():
        d = s[s["y"].between(a, b)].dropna(subset=["grade"])
        f[name] = d.groupby("gsis")[["grade", "snaps"]].apply(wmean)
    cov = s[s["y"].between(1, 4)].groupby("gsis").size()
    return pd.DataFrame(f), cov


def second_contract(lab):
    c = pd.read_parquet(RAW_DIR / "nfl_contracts.parquet").dropna(subset=["gsis_id"])
    c = c.merge(lab[["gsis", "draft_year", "round"]], left_on="gsis_id", right_on="gsis")
    dy = c["draft_year_y"]
    opt = (c["round"] == 1) & (c["years"] == 1) & (c["year_signed"] == dy + 4)
    ok = c["year_signed"].between(dy + 3, dy + 5) & ~opt
    return c[ok].groupby("gsis")["apy_cap_pct"].max()


def epa_features(lab):
    s = pd.read_parquet(QUAL_CACHE)
    s["epa"] = s[["passing_epa", "rushing_epa", "receiving_epa"]].fillna(0).sum(axis=1)
    s = s.merge(lab[["gsis", "draft_year"]], left_on="player_id", right_on="gsis")
    s["y"] = s["season"] - s["draft_year"] + 1
    return pd.DataFrame({"epa_y14": s[s["y"].between(1, 4)].groupby("gsis")["epa"].sum(),
                         "epa_y34": s[s["y"].between(3, 4)].groupby("gsis")["epa"].sum()})


def build():
    path = OUT / "labels_none.parquet"
    if not path.exists():
        build_label(path, quality="none")
    lab = pd.read_parquet(path)
    lab = lab[lab["drafted"] & lab["draft_year"].between(*CLASSES) & lab["gsis"].notna()
              & ~lab["group"].isin(["K", "P"])].drop_duplicates("gsis").copy()
    lab["y"] = lab["gsis"].map(second_contract(lab)).fillna(0.0)
    g, cov = grade_features(lab)
    lab = lab.join(g, on="gsis").join(epa_features(lab), on="gsis")
    lab["pff_seasons"] = lab["gsis"].map(cov).fillna(0)
    skill = lab["group"].isin(SKILL)
    for col in EPA:
        lab.loc[skill, col] = lab.loc[skill, col].fillna(0.0)
        lab.loc[~skill, col] = np.nan
    lab["value_y14"] = lab[VALUE].sum(axis=1)
    return lab.reset_index(drop=True)


def oof(df, feats, seeds):
    X = df[feats + ["group"]].copy()
    X["group"] = X["group"].astype("category")
    mono = [1] * len(feats) + [0]
    pred = np.full(len(df), np.nan)
    for cls in range(CLASSES[0], CLASSES[1] + 1):
        te = (df["draft_year"] == cls).to_numpy()
        ps = [lgb.LGBMRegressor(**PARAMS, random_state=s, monotone_constraints=mono)
              .fit(X[~te], df.loc[~te, "y"]).predict(X[te]) for s in seeds]
        pred[te] = np.mean(ps, axis=0)
    return pred


def within(df, col, f, idx=None):
    d = df if idx is None else df.iloc[idx]
    num = den = 0.0
    for _, g in d.groupby("group"):
        if len(g) < 10 or g["y"].nunique() < 2:
            continue
        num += len(g) * f(g[col], g["y"])[0]
        den += len(g)
    return num / den


def boot(df, pairs, n_boot, rng, f=pearsonr):
    groups = [np.flatnonzero(df["group"].to_numpy() == g) for g in df["group"].unique()]
    idxs = [np.concatenate([rng.choice(ix, len(ix)) for ix in groups]) for _ in range(n_boot)]
    rows = []
    for a, b in pairs:
        ds = np.array([within(df, a, f, i) - within(df, b, f, i) for i in idxs])
        rows.append({"Δ": f"{a} − {b}", "n": len(df), "delta": within(df, a, f) - within(df, b, f),
                     "lo95": np.percentile(ds, 2.5), "hi95": np.percentile(ds, 97.5), "P(>0)": (ds > 0).mean()})
    return pd.DataFrame(rows)


def partial_rho(g, x, ctrl="value_y14", y="y"):
    g = g.dropna(subset=[x])
    r = g[[x, ctrl, y]].rank()
    res = lambda a: r[a] - np.polyval(np.polyfit(r[ctrl], r[a], 1), r[ctrl])
    return np.corrcoef(res(x), res(y))[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(0)
    df = build()
    print(f"cohort {len(df)} drafted players, classes {CLASSES}; with a 2nd contract {(df['y'] > 0).mean():.2f}")
    print("PFF coverage (share with any graded rookie season / mean graded seasons) by group:")
    print(df.groupby("group").agg(n=("y", "size"), any_pff=("pff_seasons", lambda s: (s > 0).mean()),
                                  seasons=("pff_seasons", "mean"),
                                  played=("value_y14", lambda s: (s > 0).mean())).round(3).to_string())
    for arm, (feats, seeds) in ARMS.items():
        df[arm] = oof(df, feats, seeds) if arm not in SKILL_ONLY else np.nan
    sk = df[df["group"].isin(SKILL)].reset_index(drop=True)
    for arm in SKILL_ONLY:
        sk[arm] = oof(sk, ARMS[arm][0], ARMS[arm][1])
    ns = df[~df["group"].isin(SKILL)].reset_index(drop=True)
    paid = df[df["y"] > 0.01].reset_index(drop=True)
    df.to_parquet(OUT / "oof.parquet", index=False)
    sk.to_parquet(OUT / "oof_skill.parquet", index=False)

    out = []
    for scope, d, pairs in (("all", df, [("pff", "base"), ("base_alt", "base")]),
                            ("non-skill", ns, [("pff", "base"), ("base_alt", "base")]),
                            ("skill", sk, [("pff", "base"), ("epa", "base"), ("pff", "epa"),
                                           ("pff_epa", "epa"), ("base_alt", "base")]),
                            ("paid>1%", paid, [("pff", "base"), ("base_alt", "base")])):
        for mname, f in (("pearson", pearsonr), ("spearman", spearmanr)):
            out.append(boot(d, pairs, args.n_boot, rng, f).assign(scope=scope, metric=mname))
    D = pd.concat(out, ignore_index=True)
    print("\nΔ within-group correlation of OOF prediction with 2nd-contract pay:")
    print(D.round(4).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)
    p = D[(D["scope"] == "all") & (D["metric"] == "pearson") & (D["Δ"] == "pff − base")].iloc[0]
    pl = D[(D["scope"] == "all") & (D["metric"] == "pearson") & (D["Δ"] == "base_alt − base")].iloc[0]
    print(f"\nDECISION: pff − base Pearson {p['delta']:+.4f} [{p['lo95']:+.4f}, {p['hi95']:+.4f}] "
          f"P(>0) {p['P(>0)']:.3f}; placebo {pl['delta']:+.4f} → "
          f"{'BUILD v4 label with PFF quality' if p['P(>0)'] > 0.5 else 'KEEP v3.1 label'}")

    print("\nper group Pearson (OOF vs 2nd-contract pay):")
    per = {a: df.groupby("group").apply(lambda g: pearsonr(g[a], g["y"])[0], include_groups=False)
           for a in ("base", "base_alt", "pff")}
    for a in SKILL_ONLY:
        per[a] = sk.groupby("group").apply(lambda g: pearsonr(g[a], g["y"])[0], include_groups=False)
    print(pd.DataFrame(per).round(3).to_string())
    print("\npartial Spearman with 2nd-contract pay given value_y14:")
    print(pd.DataFrame({g: {"grade_y14": partial_rho(gg, "grade_y14"), "grade_y34": partial_rho(gg, "grade_y34"),
                            **({"epa_y34": partial_rho(gg, "epa_y34")} if g in SKILL else {})}
                        for g, gg in df.groupby("group")}).T.round(3).to_string())


if __name__ == "__main__":
    main()
