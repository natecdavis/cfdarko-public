"""
Does where a player is actually drafted, relative to his big-board rank,
predict his rookie value beyond the board? Should a player who slides past
his slot be marked down? (pre-registered 2026-11-02)

Motivation: the value board prices a player by big-board rank and position
only. During a draft every pick that passes on him is information the public
board lacks (medicals, interviews, character), and draft position also buys
playing time (teams play high picks; the label is built on playing time).
For the pick-time decision both effects count: "what is he worth if WE take
him HERE" is E[value | board rank b, position, drafted at pick d]. That is
estimable directly from past drafts; this test does not try to separate
information from opportunity. Risers (d < b) are included, so the test is
symmetric.

Cohort: board players (hasan_rank) who were drafted, drafts 2014-22 (last
college season 2013-21; complete four rookie seasons through 2025), K/P/
OTHER excluded. Outcome: value_y14 from the production rookie-value label
(v4, PFF quality); 0 for drafted players who never played. Position =
college position group (the board's). d = overall pick (bridge).
Leave-one-class-out (the board curve's own validation).

Arms:
    B0_iso   production board curve: ShrunkIso(decreasing, K_BOARD, level)
             on log b by position (src/models/merged_board.py), fit on the
             drafted board players of the other classes
    B2_mult  B0_iso × (d / b)^γ, one parameter γ fit by least squares on
             the training classes (in-sample B0 predictions there). The form
             that would be wired: a slide (or rise) multiplier on the
             current curve. γ < 0 = sliding lowers value.
    B0_lgb   LightGBM (monotone decreasing in log b; position categorical),
             seed 42
    B0_alt   B0_lgb with seed 142 (seed placebo)
    B1_lgb   B0_lgb + log d (monotone decreasing in log d): the flexible
             version of the question
PRIMARY: MSE of value_y14, B2_mult − B0_iso (what would ship). ADOPT pick-
conditioned values in the pick tools (board_at_pick, draft review: value
of a board player AT the current pick) if P(B2 better) > 0.5 (paired
bootstrap of players within class, 1000) AND value captured is not clearly
worse: Spearman of prediction vs outcome within class, P(B2 better) >= 0.2.
Prior leans toward the arm (draft position beating public boards is well
documented); cost is one parameter and a pick-dependent value in the tools.
Reported: B1_lgb − B0_lgb with the seed placebo (does a flexible form find
more than the multiplier?); γ on all classes with a bootstrap interval;
γ fit separately for slides (d > b) and rises (d < b); γ by position
group; γ for value_y12 and value_y34 (opportunity should load on y1-2);
mean realized value by board-rank band × slide band (d/b <= 0.7, 0.7-1.3,
1.3-2, > 2); worked examples (value at b = 20 if taken at 20, 32, 50).

RESULT (2026-11-02): ADOPT pick-conditioned values (γ = −0.19).
1,759 drafted board players, drafts 2014-22 (25% went at > 1.3x their board
rank, 22% at < 0.7x).
    B2_mult − B0_iso  MSE gain +2.06e-4 [+0.08, +3.71] (−4.1%), P 0.98;
                      within-class Spearman +0.030 [+0.023, +0.037], P 1.00
    B1_lgb − B0_lgb   MSE +4.26e-4 P 1.00; Spearman +0.082 P 1.00
    seed placebo      MSE +0.01e-4, Spearman −0.001
    B1_lgb − B2_mult  MSE +2.11e-4 P 0.97; Spearman +0.069 P 1.00: the
                      flexible model finds about twice as much; one
                      multiplier is a first step, not the whole effect.
γ = −0.192 [−0.259, −0.104]; LOCO γ −0.18 to −0.21 in every class. Slides
−0.224, rises −0.183 (symmetric). y12 −0.162, y34 −0.213 (intervals
overlap): the effect is not smaller in years 3-4, so it is not just the
early playing time that draft capital buys — teams' passes carry information. By position: TE −0.48, RB −0.27, OL −0.23, QB
−0.20, DB −0.19, WR −0.15, EDGE −0.10, LB −0.09, DL −0.08.
Board #20: taken at 12 → 0.159, at 20 → 0.145, at 32 → 0.132, at 50 →
0.121, at 80 → 0.111 (−23%). Raw table: board top-32 players who rose
(< 0.7x) realized 0.232 vs 0.155 near their rank and 0.120 when they slid
past 2x; ranks 101-150: 0.074 rose, 0.050 near, 0.018 slid.
Wired: src/models/draft_slide.py (value_at_pick; pick / board rank clipped
to the cohort's 1st-99th percentile, 0.31-3.3, so the curve is never
extrapolated — LOCO MSE 0.004840 clipped vs 0.004827 unclipped); board_at_pick ranks by
value at the current pick; the draft review fits γ as of each draft.
Within ONE pick every available player shares d, so the multiplier mainly
compresses gaps between board ranks (ordering by B0(b)·b^0.19); its bigger
use is across picks (a player still there at your next pick is worth less
then) — the input the position drop-off feature needs.

AMENDMENT (2026-11-02, after the result; raised by the user): only the
SLIDE half is usable for decisions. A player on the board at our pick has
been passed by every earlier team (observed); a player we take before his
board rank carries no such signal — the historical rise premium is what the
reaching team knew, and our own reach is not evidence (it would also credit
reaches in the draft review). Wiring changed to slides only: value x
min(1, (d/b)^γ), d/b capped at 3.3, γ refit in that form = −0.254. LOCO on
this cohort (B0_iso base): slide-only MSE gain on sliders +2.81e-4 (P 0.99)
vs two-sided +2.92e-4; risers 0 by construction (two-sided +0.95e-4, P
0.72); all players +1.30e-4, P 1.00.

Bug fix before results (2026-11-02): metrics() indexed bootstrap samples by
label, so resampled duplicates broke the within-class Spearman; now
positional. The first run stopped there; it had printed only the LOCO γ by
class (all near −0.19), not the primary.

Usage: 01f_draft_slide.py [--n-boot 1000]
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from src.models.merged_board import ShrunkIso, K_BOARD, EXCLUDED_GROUPS  # noqa: E402
from src.models.offboard_model import pid  # noqa: E402
from src.models.partial_outcomes import board_value_frame  # noqa: E402

OUT = PROCESSED_DIR / "_draft_slide_20261102"
DRAFTS = (2014, 2022)
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
              subsample=0.8, subsample_freq=1, verbose=-1)


def cohort():
    cols = ["player_id", "position_group", "hasan_rank", "last_college_season_year"]
    P = pd.read_parquet(OUTPUT_DIR / "projections" / "latest.parquet", columns=cols)
    L = pd.read_parquet(PROCESSED_DIR / "rookie_value_labels.parquet")
    b = board_value_frame(P, L)
    b = b[~b["pos_group"].isin(EXCLUDED_GROUPS) & b["draft_year"].between(*DRAFTS)].copy()
    br = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet").dropna(subset=["player_id"])
    br = br[br["overall_pick"].notna() & (br["is_udfa"] != True)]
    br = br.assign(pid=pid(br["player_id"])).dropna(subset=["pid"]).drop_duplicates("pid")
    b["pid"] = pid(b["player_id"])
    b = b.merge(br[["pid", "overall_pick", "draft_year"]].rename(columns={"draft_year": "bridge_dy"}),
                on="pid", how="inner")
    b = b[b["bridge_dy"] == b["draft_year"]]
    b["y14"] = b[["v1", "v2", "v3", "v4"]].sum(axis=1)
    b["y12"] = b[["v1", "v2"]].sum(axis=1)
    b["y34"] = b[["v3", "v4"]].sum(axis=1)
    b = b.dropna(subset=["y14"])
    b["lb"], b["ld"] = np.log(b["hasan_rank"]), np.log(b["overall_pick"])
    b["slide"] = b["ld"] - b["lb"]
    return b.reset_index(drop=True)


def fit_gamma(base, slide, y, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    f = lambda g: np.mean((base[m] * np.exp(g * slide[m]) - y[m]) ** 2)
    return minimize_scalar(f, bounds=(-3, 3), method="bounded").x


def lgb_pred(tr, te, feats, y, seed):
    X = lambda d: d[feats].assign(pos=pd.Categorical(d["pos_group"]))
    mono = [-1] * len(feats) + [0]
    m = lgb.LGBMRegressor(**PARAMS, random_state=seed, monotone_constraints=mono)
    Xtr = X(tr)
    Xte = X(te).assign(pos=pd.Categorical(te["pos_group"], categories=Xtr["pos"].cat.categories))
    return np.clip(m.fit(Xtr, tr[y]).predict(Xte), 0, None)


def loco(b, y="y14"):
    out = pd.DataFrame(index=b.index, columns=["B0_iso", "B2_mult", "B0_lgb", "B0_alt", "B1_lgb"], dtype=float)
    gam = {}
    for c in sorted(b["draft_year"].unique()):
        te, tr = b["draft_year"] == c, b["draft_year"] != c
        T, V = b[tr], b[te]
        iso = ShrunkIso(increasing=False, k=K_BOARD, level=True).fit(T["lb"], T[y], T["pos_group"])
        base_tr = iso.predict(T["lb"], T["pos_group"])
        g = fit_gamma(base_tr, T["slide"].to_numpy(), T[y].to_numpy())
        gam[int(c)] = g
        base = iso.predict(V["lb"], V["pos_group"])
        out.loc[te, "B0_iso"] = base
        out.loc[te, "B2_mult"] = base * np.exp(g * V["slide"].to_numpy())
        out.loc[te, "B0_lgb"] = lgb_pred(T, V, ["lb"], y, 42)
        out.loc[te, "B0_alt"] = lgb_pred(T, V, ["lb"], y, 142)
        out.loc[te, "B1_lgb"] = lgb_pred(T, V, ["lb", "ld"], y, 42)
    return out, gam


def metrics(b, out, idx, y="y14"):
    d = b.iloc[idx].reset_index(drop=True)
    o = out.iloc[idx].reset_index(drop=True)
    res = {}
    for a in out.columns:
        res[("mse", a)] = np.mean((o[a].to_numpy() - d[y].to_numpy()) ** 2)
        rh = [spearmanr(o.loc[g.index, a], g[y])[0] for _, g in d.groupby("draft_year") if g[y].nunique() > 1]
        res[("rho", a)] = np.mean(rh)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    rng = np.random.default_rng(0)
    b = cohort()
    print(f"cohort: {len(b)} drafted board players, drafts {b['draft_year'].min()}-{b['draft_year'].max()}; "
          f"slides (d > 1.3b) {(b['slide'] > np.log(1.3)).mean():.2f}, rises (d < 0.7b) "
          f"{(b['slide'] < np.log(0.7)).mean():.2f}")
    out, gam = loco(b)
    b.join(out).to_parquet(OUT / "oof.parquet", index=False)
    print("LOCO γ by test class:", {k: round(v, 3) for k, v in gam.items()})

    base = metrics(b, out, np.arange(len(b)))
    groups = [g.index.to_numpy() for _, g in b.groupby("draft_year")]
    boots = [metrics(b, out, np.concatenate([rng.choice(g, len(g)) for g in groups]))
             for _ in range(args.n_boot)]
    rows = []
    for a, r in [("B2_mult", "B0_iso"), ("B1_lgb", "B0_lgb"), ("B0_alt", "B0_lgb"), ("B1_lgb", "B2_mult")]:
        for met, sign in (("mse", -1), ("rho", 1)):
            pt = sign * (base[(met, a)] - base[(met, r)])
            bs = np.array([sign * (x[(met, a)] - x[(met, r)]) for x in boots])
            rows.append({"Δ": f"{a} − {r}", "metric": met, "gain": pt, "lo95": np.percentile(bs, 2.5),
                         "hi95": np.percentile(bs, 97.5), "P(better)": (bs > 0).mean()})
    D = pd.DataFrame(rows)
    print("\nlevels:", {f"{m}/{a}": round(v, 5) for (m, a), v in base.items()})
    print("\ngain (> 0 = first arm better; mse gain = reduction):")
    print(D.round(6).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)
    p = D[(D["Δ"] == "B2_mult − B0_iso") & (D["metric"] == "mse")].iloc[0]
    g = D[(D["Δ"] == "B2_mult − B0_iso") & (D["metric"] == "rho")].iloc[0]
    adopt = bool(p["P(better)"] > 0.5 and g["P(better)"] >= 0.2)
    print(f"\nDECISION: B2_mult − B0_iso MSE gain {p['gain']:+.6f} [{p['lo95']:+.6f}, {p['hi95']:+.6f}] "
          f"P {p['P(better)']:.3f}; rho P {g['P(better)']:.3f} → "
          f"{'ADOPT pick-conditioned values' if adopt else 'KEEP board-only values'}")

    # γ on all classes: overall, slides vs rises, by position, by outcome window.
    def gamma_all(y, mask=None):
        iso = ShrunkIso(increasing=False, k=K_BOARD, level=True).fit(b["lb"], b[y], b["pos_group"])
        bp = iso.predict(b["lb"], b["pos_group"])
        return fit_gamma(bp, b["slide"].to_numpy(), b[y].to_numpy(), mask)
    idxs = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(200)]

    def gamma_ci(y, mask_fn=lambda d: np.ones(len(d), bool)):
        pt = gamma_all(y, mask_fn(b))
        bs = []
        for ix in idxs:
            d = b.iloc[ix].reset_index(drop=True)
            iso = ShrunkIso(increasing=False, k=K_BOARD, level=True).fit(d["lb"], d[y], d["pos_group"])
            bs.append(fit_gamma(iso.predict(d["lb"], d["pos_group"]), d["slide"].to_numpy(),
                                d[y].to_numpy(), mask_fn(d)))
        return round(pt, 3), round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)
    print("\nγ (all classes, [95% bootstrap]):")
    print("  y14 all     ", gamma_ci("y14"))
    print("  y14 slides  ", gamma_ci("y14", lambda d: (d["slide"] > 0).to_numpy()))
    print("  y14 rises   ", gamma_ci("y14", lambda d: (d["slide"] < 0).to_numpy()))
    print("  y12         ", gamma_ci("y12"))
    print("  y34         ", gamma_ci("y34"))
    print("  by position:", {g: round(gamma_all("y14", (b["pos_group"] == g).to_numpy()), 3)
                             for g in sorted(b["pos_group"].unique())})

    band = pd.cut(b["hasan_rank"], [0, 32, 64, 100, 150, 300], labels=["1-32", "33-64", "65-100", "101-150", "151+"])
    sl = pd.cut(np.exp(b["slide"]), [0, 0.7, 1.3, 2, np.inf], labels=["rose (<0.7)", "near (0.7-1.3)",
                                                                       "slid (1.3-2)", "slid far (>2)"])
    print("\nmean realized value_y14 (n) by board-rank band × where drafted relative to board rank:")
    t = b.assign(band=band, sl=sl).pivot_table(index="band", columns="sl", values="y14",
                                               aggfunc=["mean", "size"], observed=True)
    print(t.round(4).to_string())
    g0 = gamma_all("y14")
    iso = ShrunkIso(increasing=False, k=K_BOARD, level=True).fit(b["lb"], b["y14"], b["pos_group"])
    print("\nworked example, board #20 (pooled-position mean of the curve):")
    v20 = np.mean([iso.predict([np.log(20)], [p])[0] for p in b["pos_group"].unique()])
    for d in (12, 20, 32, 50, 80):
        print(f"  taken at {d:>3}: {v20 * (d / 20) ** g0:.4f}  (board-only {v20:.4f})")


if __name__ == "__main__":
    main()
