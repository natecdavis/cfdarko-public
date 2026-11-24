"""
Merged board in rookie-deal value instead of y3-4 snap share (pre-registered
2026-11-01).

Product (user, 2026-11-01): project each player's value correctly, then at
any pick rank the players still available. At a given pick the rookie cost
is set by the slot, not the player, so ranking available players by
expected surplus = ranking by expected rookie-deal value. The production
merged board ranks in y3-4 snap share, which orders players within a
position but treats a starting RB's snaps like a starting QB's. The rookie
surplus label (00zo v3.1) gives a cross-position currency: value_y14 = Σ
over y1-y4 of the veteran-market price of the player's role at his NFL
position (EPA quality-adjusted for QB/RB/WR/TE) x games, % of cap.

Arms (list columns L_*; production recipe, src/models/merged_board.py, per eval fold; only the
calibration target differs):
    usage   board curve E[usage | log hasan_rank, pos] + off-board 8d
            percentile → usage, capped at the top-1% mean (= production)
    value   the same three calibrators fit to rookie value (value_y14,
            0 if never in the NFL)
    value_alt  value arm with the off-board side scored by 8d's seed-142-146
            bag (00zn base_alt) instead of production seeds 42-46: the
            placebo for model noise (the calibrators are deterministic)
Off-board scores: production 8d OOF (00zn's base reproduces it exactly).
Folds as in 00zf: calibrators fit on board-era rows (last season >= 2013)
with last season <= train_end; eval val 2016-18 and 2019-21 = classes
2017-22. NFL players without a rookie value (14 of 3,268, linkage) are
dropped from fitting and scoring in every arm. K/P excluded.

Within a position the two arms order players almost identically (both are
monotone maps of the same board rank / model percentile; isotonic ties
aside). The test is about the CROSS-POSITION exchange rate.

PRIMARY: per class, the top M of the whole class list (M = board players
in the class); mean realized rookie value of those M, averaged over
classes. value − usage. Decision (feedback_bayesian_decisions): currency is
a symmetric choice with zero cost, and the product has already chosen value
as the currency, so ADOPT value unless it is clearly worse: P(value − usage
> 0) on the primary >= 0.2 (paired bootstrap, players resampled within
class, 1000 reps), and |value_alt − value| reported as the model-noise
scale.
Reported: top 32 / 100 / 256 (a draft's worth) on rookie value; the same
on usage and on realized second-contract pay (best_apy_by_y5); pooled
cross-position Spearman of score vs rookie value within class; position mix
of each list's top 100; calibration (predicted vs realized value) by
position for board players.

RESULT (2026-11-01): ADOPT by the pre-registered rule, but it is a tie.
29,438 players, classes 2017-22, 1,706 board, 868 off-board NFL. The two
lists agree within position (Spearman 0.981).
    rookie value  topM −0.0000 [−0.0005, +0.0004] P 0.51; top32 +0.0033
                  P 0.89; top100 −0.0001; top256 −0.0000
    usage         top32 −0.037 [−0.070, −0.011] (value list picks fewer
                  full-time players at the very top, as intended)
    2nd contract  all ties
    model-noise placebo (value_alt − value): 0.0000 on every top-K
The whole-class Spearman drop (−0.009, P 0) is a tie artifact at the
bottom of the class: with a tiny tiebreak by model percentile the lists
match (0.4425 vs 0.4431). Board players only, cross-position: usage 0.618
vs value 0.610.
Top 100 mix per class: QB 4.7 → 6.3, RB 5.3 → 6.2, DL 7.8 → 9.2, DB 18.7 →
16.8, LB 7.7 → 6.5, OL 22.5 → 21.2.
Why so little change: the calibrator shrinks each position's curve toward
the POOLED curve (K_BOARD = 100). In value units, position levels differ
far more than in snap share, so the shrinkage erases part of the exchange
rate: board QBs are predicted 0.088 vs realized 0.123 (−29%), and
DL/EDGE/LB/RB are over-predicted by 10-15%. Next arm: shrink toward the
pooled SHAPE rescaled to each position's level.

LEVEL-PRESERVING ARM — PRE-REGISTERED 2026-11-01, after the result above,
before any run of it. `--level`: every calibrator (board curve, off-board
map; the cap is unchanged) shrinks each position's isotonic curve toward
the pooled curve RESCALED to that position's level:
    pred_g(x) = w_g·iso_g(x) + (1 − w_g)·lvl_g·pooled(x),
    lvl_g = mean(y over g's training rows) / mean(pooled(x) over them)
Arms L_value_lvl and L_value_lvl_alt (8d seeds 142-146, placebo).
PRIMARY: board players' cross-position Spearman of list vs realized rookie
value within class (board_rho), L_value_lvl − L_value — the ordering the
exchange rate controls, where the off-board model does not enter. Adopt if
P(> 0) > 0.5 (a principled correction to a measured calibration bias; zero
cost). Reported: the top-K metrics, board calibration by position, the
same contrast vs L_usage, top-100 mix.

LEVEL RESULT (2026-11-01): KEEP pooled shrinkage (pre-registered primary
fails). board_rho L_value_lvl − L_value −0.012 [−0.017, −0.006], P 0.00;
placebo 0. It does fix the calibration it targeted (board QB predicted
0.115 vs realized 0.123, was 0.088; RB/LB/DL closer too), and it puts
QB 7.7 / WR 14.7 / RB 4.3 per class in the top 100. On value CAPTURED it
is mixed and inside the noise: top32 +0.0036 (P 0.84; +0.0069 vs usage,
P 0.95), top100 −0.0006 (P 0.18), topM −0.0000.
Reading: realized rookie value is very skewed at QB (a few huge hits,
many small ones), so ranking by the MEAN moves QBs up while a rank
statistic scores them by their typical (low) outcome. In hindsight a rank
metric was a poor primary for a mean-valued currency; noted, not
re-decided. The usage list has the best board_rho of all (0.618).
Overall: the board's rank already prices position, so changing the
exchange rate barely moves value captured.

V4 RECHECK (2026-11-02): --label data/processed/rookie_value_labels.parquet
--out data/processed/_value_currency_board_v4. Same decision (ADOPT value,
a tie): topM +0.0000 [−0.0004, +0.0005], P 0.65; placebo 0.0000.

Usage: 00zs_value_currency_board.py [--n-boot 1000] [--level]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR  # noqa: E402
from src.models.merged_board import ShrunkIso, offboard_percentile, K_BOARD, K_OFF, CAP_Q  # noqa: E402
from src.models.offboard_model import pid  # noqa: E402

OUT = PROCESSED_DIR / "_value_currency_board_20261101"
LABEL = PROCESSED_DIR / "_rookie_surplus_20261101" / "rookie_surplus_labels.parquet"
BOARD_ERA_START = 2013
FOLDS = [(2015, 2016, 2018), (2018, 2019, 2021)]   # (train_end, val_start, val_end)
ARMS = {"L_usage": ("usage", "base"), "L_value": ("rv", "base"), "L_value_alt": ("rv", "base_alt")}
PAIRS = [("L_value", "L_usage"), ("L_value_alt", "L_value")]
OUTCOMES = ["rv", "usage", "apy"]


def load():
    o = pd.read_parquet(PROCESSED_DIR / "_offboard_usage_norm_20261101" / "oof_all_arms.parquet")
    o = o.drop_duplicates("player_id", keep="last")
    o = o[(o["last_college_season_year"] >= BOARD_ERA_START) & ~o["pos_group"].isin(["K", "P", "OTHER"])]
    L = pd.read_parquet(LABEL).assign(pid=lambda d: pid(d["player_id"])).dropna(subset=["pid"])
    rv = L.drop_duplicates("pid").set_index("pid")["value_y14"]
    lab = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    lab = lab.dropna(subset=["player_id"]).assign(pid=lambda d: pid(d["player_id"])).drop_duplicates("pid")
    apy = lab.set_index("pid")["best_apy_by_y5"]
    o["rv"] = np.where(o["nfl"], o["player_id"].map(rv), 0.0)
    o["apy"] = np.where(o["nfl"], o["player_id"].map(apy).fillna(0.0), 0.0)
    n0 = len(o)
    o = o[o["rv"].notna()].reset_index(drop=True)
    print(f"dropped {n0 - len(o)} NFL rows without a rookie value; {len(o)} rows")
    o["cls"] = o["last_college_season_year"].astype(int) + 1
    o["board"] = o["hasan_rank"].notna()
    for s in ("base", "base_alt"):
        o[f"{s}_pct"] = np.nan
        ob = ~o["board"]
        o.loc[ob, f"{s}_pct"] = offboard_percentile(o.loc[ob, s], o.loc[ob, "last_college_season_year"],
                                                     o.loc[ob, "pos_group"])
    return o


class LevelShrunkIso(ShrunkIso):
    """ShrunkIso, but the pooled curve is rescaled to each position's level."""

    def fit(self, x, y, pos, weight_n=None):
        super().fit(x, y, pos, weight_n)
        x, y, pos = np.asarray(x, float), np.asarray(y, float), np.asarray(pos)
        pp = self.pooled.predict(x)
        self.lvl = {g: (y[pos == g].mean() / pp[pos == g].mean()) if pp[pos == g].mean() > 0 else 1.0
                    for g in np.unique(pos)}
        return self

    def predict(self, x, pos):
        x, pos = np.asarray(x, float), np.asarray(pos)
        base = self.pooled.predict(x)
        out = base.copy()
        for g in np.unique(pos):
            if g in self.per:
                m = pos == g
                out[m] = self.w[g] * self.per[g].predict(x[m]) + (1 - self.w[g]) * self.lvl[g] * base[m]
        return out


LEVEL = False


def score_fold(E, train_end, val_start, val_end):
    tr = E[E["last_college_season_year"] <= train_end]
    va = E[E["last_college_season_year"].between(val_start, val_end)].copy()
    bt, ob = tr[tr["board"]], tr[~tr["board"]]
    vb = va["board"].to_numpy()
    n_nfl = ob.groupby("pos_group")["nfl"].sum().to_dict()
    for arm, (target, s) in ARMS.items():
        Cal = LevelShrunkIso if arm.startswith("L_value_lvl") else ShrunkIso
        bc = Cal(increasing=False, k=K_BOARD).fit(np.log(bt["hasan_rank"]), bt[target], bt["pos_group"])
        oc = Cal(increasing=True, k=K_OFF).fit(ob[f"{s}_pct"], ob[target], ob["pos_group"], weight_n=n_nfl)
        if arm == "L_value_lvl":
            print(f"fold {val_start} board levels: {({g: round(v, 2) for g, v in bc.lvl.items()})}")
        top = ob[ob[f"{s}_pct"] >= CAP_Q]
        pooled = float(top[target].mean())
        cap = {g: (len(t) * t[target].mean() + K_OFF * pooled) / (len(t) + K_OFF)
               for g, t in top.groupby("pos_group")}
        bp = np.full(len(va), np.nan)
        bp[vb] = bc.predict(np.log(va.loc[vb, "hasan_rank"]), va.loc[vb, "pos_group"])
        off = oc.predict(va[f"{s}_pct"].fillna(0), va["pos_group"])
        off = np.minimum(off, va["pos_group"].map(cap).fillna(pooled).to_numpy())
        va[arm] = np.where(vb, bp, off)
        if arm != "L_usage":
            print(f"fold {val_start} {arm} cap: {({g: round(v, 4) for g, v in cap.items()})}")
    return va


def metrics(V, y):
    res = {}
    by = list(V.groupby("cls"))
    for a in ARMS:
        r = {"topM": np.mean([g.nlargest(int(g["board"].sum()), a)[y].mean() for _, g in by])}
        for k in (32, 100, 256):
            r[f"top{k}"] = np.mean([g.nlargest(k, a)[y].mean() for _, g in by])
        r["rho_class"] = np.mean([spearmanr(g[a], g[y])[0] for _, g in by])
        r["board_rho"] = np.mean([spearmanr(g.loc[g["board"], a], g.loc[g["board"], y])[0] for _, g in by])
        res[a] = r
    return res


def bootstrap(V, y, n_boot, rng):
    V = V.reset_index(drop=True)
    groups = [g.index.to_numpy() for _, g in V.groupby("cls")]
    point = metrics(V, y)
    boots = [metrics(V.iloc[np.concatenate([rng.choice(g, len(g)) for g in groups])], y)
             for _ in range(n_boot)]
    rows = []
    for a, b in PAIRS:
        for k in point[a]:
            d = np.array([bt[a][k] - bt[b][k] for bt in boots])
            rows.append({"Δ": f"{a} − {b}", "metric": k, "delta": point[a][k] - point[b][k],
                         "lo95": np.percentile(d, 2.5), "hi95": np.percentile(d, 97.5),
                         "P(>0)": (d > 0).mean()})
    return point, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--level", action="store_true")
    ap.add_argument("--label", type=Path, default=None,
                    help="rookie-value label to use (default: the 2026-11-01 v3 build); "
                         "the v4 recheck passes the production label")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    global OUT, LABEL
    if args.label is not None:
        LABEL = args.label
    if args.out is not None:
        OUT = args.out
    if args.level:
        ARMS["L_value_lvl"] = ("rv", "base")
        ARMS["L_value_lvl_alt"] = ("rv", "base_alt")
        PAIRS[0:0] = [("L_value_lvl", "L_value"), ("L_value_lvl_alt", "L_value_lvl"),
                      ("L_value_lvl", "L_usage")]
        OUT = OUT / "level"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 200)
    E = load()
    V = pd.concat([score_fold(E, *f) for f in FOLDS], ignore_index=True)
    print(f"eval {len(V)} players, classes {V['cls'].min()}-{V['cls'].max()}, board {int(V['board'].sum())}, "
          f"off-board NFL {int((~V['board'] & V['nfl']).sum())}")
    V.to_parquet(OUT / "merged_oof.parquet", index=False)

    w = np.mean([spearmanr(g["L_usage"], g["L_value"])[0]
                 for _, g in V.groupby(["cls", "pos_group"]) if len(g) > 10])
    print(f"within (class × position) Spearman between the usage and value lists: {w:.4f}")

    all_d = []
    for y in OUTCOMES:
        point, d = bootstrap(V, y, args.n_boot, rng)
        print(f"\n=== outcome: {y} ===")
        print(pd.DataFrame(point).T.round(4).to_string())
        print(d.round(4).to_string(index=False))
        all_d.append(d.assign(outcome=y))
    D = pd.concat(all_d)
    D.to_csv(OUT / "deltas.csv", index=False)

    if args.level:
        q = D[(D["outcome"] == "rv") & (D["Δ"] == "L_value_lvl − L_value") & (D["metric"] == "board_rho")].iloc[0]
        ql = D[(D["outcome"] == "rv") & (D["Δ"] == "L_value_lvl_alt − L_value_lvl") & (D["metric"] == "board_rho")].iloc[0]
        print(f"\nDECISION (level): L_value_lvl − L_value board_rho {q['delta']:+.4f} "
              f"[{q['lo95']:+.4f}, {q['hi95']:+.4f}], P(>0) {q['P(>0)']:.3f}; placebo {ql['delta']:+.4f} → "
              f"{'ADOPT level-preserving' if q['P(>0)'] > 0.5 else 'KEEP pooled shrinkage'}")
    p = D[(D["outcome"] == "rv") & (D["Δ"] == "L_value − L_usage") & (D["metric"] == "topM")].iloc[0]
    pl = D[(D["outcome"] == "rv") & (D["Δ"] == "L_value_alt − L_value") & (D["metric"] == "topM")].iloc[0]
    print(f"\nDECISION: value − usage topM on rookie value {p['delta']:+.4f} "
          f"[{p['lo95']:+.4f}, {p['hi95']:+.4f}], P(>0) {p['P(>0)']:.3f}; model-noise placebo "
          f"{pl['delta']:+.4f} → {'ADOPT value' if p['P(>0)'] >= 0.2 else 'KEEP usage'}")

    print("\nposition mix of each list's top 100 (players per class):")
    mix = {a: V.groupby("cls", group_keys=False)[V.columns.tolist()].apply(lambda g: g.nlargest(100, a))
           ["pos_group"].value_counts() / V["cls"].nunique() for a in [x for x in ("L_usage", "L_value", "L_value_lvl") if x in ARMS]}
    print(pd.DataFrame(mix).round(1).fillna(0).to_string())
    print("\nboard players, predicted vs realized rookie value by position (value arm):")
    b = V[V["board"]]
    agg = {"n": ("rv", "size"), "actual": ("rv", "mean"), "pred": ("L_value", "mean")}
    if "L_value_lvl" in ARMS:
        agg["pred_lvl"] = ("L_value_lvl", "mean")
    print(b.groupby("pos_group").agg(**agg)
          .round(4).to_string())
    print("\nvalue arm, top 15 of the 2022 class:")
    print(V[V["cls"] == V["cls"].max()].nlargest(15, "L_value")
          [["player_id", "pos_group", "hasan_rank", "L_value", "L_usage", "rv", "usage"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
