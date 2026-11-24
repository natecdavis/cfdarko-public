"""
P(available at pick n): when does a board player get drafted? (pre-registered
2026-11-01)

Product use: at pick c, for each available player, P(still available at our
next pick m) = S(m) / S(c), where S(n) = P(drafted after pick n | pre-draft
info). Combined with rookie value this says whether a player can be waited
on.

Data: big-board players (hasan_rank non-null), classes 2014-2026 (the 2026
draft is in), actual overall pick from bridge_players; undrafted = Y > 262.
Off-board players are out of scope here (reported only: share of picks
taken from off the board, by pick range).

Arms (all fit on training classes only):
    A_knn   empirical conditional survival: tricube-weighted kNN in log
            board rank, k = 300 (pooled positions; nonparametric check)
    B_ln    censored log-normal: log Y ~ N(mu, sigma),
            mu = a + b L + c L^2, log sigma = d + e L, L = log rank;
            discrete-interval likelihood (pick n ⇔ log Y in
            [log(n-0.5), log(n+0.5)]), undrafted ⇔ log Y > log 262.5.
            S(n) = 1 − Φ((log(n+0.5) − mu) / sigma).
    C_pos   B_ln + position offsets in mu and log sigma (ridge 1.0 on the
            offsets); positions K/P/OTHER pooled into one group.
Validation: leave-one-class-out over 2014-2026 (each test class predicted
from the other 12).
Score: discrete CRPS per player = mean over n = 1..262 of
(S_pred(n) − 1[Y > n])^2 (integrated Brier), averaged over players.
PRIMARY: C_pos − B_ln CRPS; adopt C_pos if P(C better) > 0.5 (paired
bootstrap, players resampled within class, 1000 reps). B_ln is the default
(smooth, extrapolates to rank 1 and 300); A_knn replaces the chosen
parametric arm only if it beats it with P > 0.5.
Reported: the decision-relevant conditional Brier: for current picks c in
{1, 8, 16, ..., 224} and the next pick m = c + 32 (a team's next pick when
picking at the same slot each round) and m = c + 16, over players still
available at c: Brier of S(m)/S(c) vs 1[Y > m]; calibration by predicted
decile; CRPS by class (drift); per-position mean predicted vs realized
log pick (does position shift draft timing relative to rank?).

RESULT (2026-11-01): USE B_ln (censored log-normal on log board rank).
3,704 board players, 2014-26, 24.9% undrafted. Mean CRPS: A 0.09469,
B 0.09416, C 0.09418.
    C_pos vs B_ln   CRPS tie (gain −0.00002, P 0.49) → B_ln by rule;
                    C_pos IS better on the decision-relevant conditional
                    Brier (next pick +32: +0.00043, P 0.99; +16: P 0.98)
    A_knn vs B_ln   B better on CRPS (P 1.00) and +16 Brier
Calibration of B is close at every horizon (e.g. n = 100 deciles
0.33/0.29, 0.58/0.61, 0.77/0.84, 0.89/0.88). No drift in CRPS by class.
Position: QB draft timing is far more dispersed than rank implies
(log sigma +0.35, 1.4x the spread); RB/TE tighter (−0.20/−0.13); mean
offsets small (|mu| <= 0.05) except K/P (+0.26, later). Fitted B (all
classes): mu = 0.479 + 0.821 L + 0.021 L^2, log sigma = −0.053 − 0.139 L
(one LOCO fold shown; production refits on all classes).
Board players supply ~98% of picks in rounds 1-2, 95% to pick 150, 78% for
150-200, 60% after 200.

Usage: 00zx_availability.py [--n-boot 1000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

OUT = PROCESSED_DIR / "_availability_20261101"
N_PICKS = 262
GRID = np.arange(1, N_PICKS + 1)
POS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB", "ST"]
ARMS = ["A_knn", "B_ln", "C_pos"]
K_KNN = 300
RIDGE = 1.0


def load():
    P = pd.read_parquet(OUTPUT_DIR / "projections" / "latest.parquet",
                        columns=["player_id", "player_name", "position_group", "hasan_rank",
                                 "hasan_draft_year"]).drop_duplicates("player_id")
    B = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    d = B[B["overall_pick"].notna()].drop_duplicates("player_id").set_index("player_id")
    b = P[P["hasan_rank"].notna()].copy()
    b["cls"] = b["hasan_draft_year"].astype(int)
    b["Y"] = b["player_id"].map(d["overall_pick"]).fillna(N_PICKS + 1).clip(upper=N_PICKS + 1)
    b["pos"] = np.where(b["position_group"].isin(POS[:-1]), b["position_group"], "ST")
    b["L"] = np.log(b["hasan_rank"])
    return b.reset_index(drop=True)


def offboard_share(b):
    D = pd.read_parquet(RAW_DIR / "draft_picks.parquet")
    D = D[D["draft_year"].between(2015, 2026)]
    got = set(zip(b["cls"], b["Y"].astype(int)))
    D["on_board"] = [(a, int(p)) in got for a, p in zip(D["draft_year"], D["overall_pick"])]
    return D.groupby(pd.cut(D["overall_pick"], [0, 32, 64, 100, 150, 200, 262]),
                     observed=True)["on_board"].mean()


# ---------- arms: each returns S, an (n_test x 262) matrix of P(Y > n) ----------

def knn_surv(tr, te, k=K_KNN):
    ind = (tr["Y"].to_numpy()[:, None] > GRID[None, :]).astype(float)
    Ltr = tr["L"].to_numpy()
    S = np.empty((len(te), N_PICKS))
    for i, l in enumerate(te["L"].to_numpy()):
        dist = np.abs(Ltr - l)
        h = np.partition(dist, k - 1)[k - 1] * 1.0001 + 1e-9
        w = np.clip(1 - (dist / h) ** 3, 0, None) ** 3
        S[i] = w @ ind / w.sum()
    return S


def _design(d, pos):
    L = d["L"].to_numpy()
    Xm = np.column_stack([np.ones_like(L), L, L ** 2])
    Xs = np.column_stack([np.ones_like(L), L])
    if pos:
        D = (d["pos"].to_numpy()[:, None] == np.array(POS)[None, :]).astype(float)
        Xm, Xs = np.column_stack([Xm, D]), np.column_stack([Xs, D])
    return Xm, Xs


class LogNormal:
    def __init__(self, pos=False):
        self.pos = pos

    def fit(self, d):
        Xm, Xs = _design(d, self.pos)
        y = d["Y"].to_numpy()
        drafted = y <= N_PICKS
        lo = np.log(np.where(drafted, y - 0.5, N_PICKS + 0.5))
        hi = np.log(y + 0.5)
        pm, ps = Xm.shape[1], Xs.shape[1]
        npos = len(POS) if self.pos else 0

        def nll(th):
            mu, sg = Xm @ th[:pm], np.exp(Xs @ th[pm:])
            zl, zh = (lo - mu) / sg, (hi - mu) / sg
            p = np.where(drafted, norm.cdf(zh) - norm.cdf(zl), norm.sf(zl))
            pen = RIDGE * (np.sum(th[pm - npos:pm] ** 2) + np.sum(th[pm + ps - npos:] ** 2)) if npos else 0.0
            return -np.sum(np.log(np.clip(p, 1e-12, None))) + pen

        th0 = np.zeros(pm + ps)
        th0[:2] = [0.3, 0.9]
        th0[pm] = np.log(0.5)
        self.th = minimize(nll, th0, method="L-BFGS-B").x
        self.pm = pm
        return self

    def mu_sigma(self, d):
        Xm, Xs = _design(d, self.pos)
        return Xm @ self.th[:self.pm], np.exp(Xs @ self.th[self.pm:])

    def surv(self, d):
        mu, sg = self.mu_sigma(d)
        return norm.sf((np.log(GRID[None, :] + 0.5) - mu[:, None]) / sg[:, None])


def predict_arms(tr, te):
    return {"A_knn": knn_surv(tr, te), "B_ln": LogNormal(False).fit(tr).surv(te),
            "C_pos": LogNormal(True).fit(tr).surv(te)}


# ---------- scoring ----------

def crps(S, y):
    return np.mean((S - (y[:, None] > GRID[None, :])) ** 2, axis=1)


def cond_brier(S, y, gap):
    """Per player, mean Brier of S(m)/S(c) over current picks c where the
    player is still available (Y > c), m = c + gap."""
    out = np.full(len(y), np.nan)
    cs = np.array([1] + list(range(8, 225, 8)))
    for i in range(len(y)):
        c = cs[y[i] > cs]
        c = c[c + gap <= N_PICKS]
        if len(c) == 0:
            continue
        p = S[i, c + gap - 1] / np.clip(S[i, c - 1], 1e-9, None)
        out[i] = np.mean((p - (y[i] > c + gap)) ** 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(0)
    b = load()
    print(f"board players {len(b)}, classes {b['cls'].min()}-{b['cls'].max()}, "
          f"undrafted {(b['Y'] > N_PICKS).mean():.1%}")
    print("share of picks taken from the board, by pick range:")
    print(offboard_share(b).round(3).to_string())

    S = {a: np.empty((len(b), N_PICKS)) for a in ARMS}
    mus = pd.DataFrame(index=b.index, columns=["B_mu", "C_mu"], dtype=float)
    for t in sorted(b["cls"].unique()):
        te_m = (b["cls"] == t).to_numpy()
        tr, te = b[~te_m], b[te_m]
        for a, s in predict_arms(tr, te).items():
            S[a][te_m] = s
        mus.loc[te.index, "B_mu"] = LogNormal(False).fit(tr).mu_sigma(te)[0]
        mus.loc[te.index, "C_mu"] = LogNormal(True).fit(tr).mu_sigma(te)[0]
    y = b["Y"].to_numpy()
    sc = pd.DataFrame({"cls": b["cls"], "pos": b["pos"]})
    for a in ARMS:
        sc[f"crps_{a}"] = crps(S[a], y)
        sc[f"cb32_{a}"] = cond_brier(S[a], y, 32)
        sc[f"cb16_{a}"] = cond_brier(S[a], y, 16)
    sc.to_parquet(OUT / "scores.parquet")
    np.savez_compressed(OUT / "surv_oof.npz", **S, Y=y)

    mets = ["crps", "cb32", "cb16"]
    print("\nmean score (lower is better):")
    print(pd.DataFrame({m: {a: sc[f"{m}_{a}"].mean() for a in ARMS} for m in mets}).round(5).to_string())

    groups = [g.index.to_numpy() for _, g in sc.groupby("cls")]
    rows = []
    for a, base in (("C_pos", "B_ln"), ("A_knn", "B_ln"), ("A_knn", "C_pos")):
        for m in mets:
            diff = (sc[f"{m}_{base}"] - sc[f"{m}_{a}"]).to_numpy()  # > 0: a better
            bs = []
            for _ in range(args.n_boot):
                idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
                bs.append(np.nanmean(diff[idx]))
            bs = np.array(bs)
            rows.append({"Δ (base − arm, >0 = arm better)": f"{base} − {a}", "metric": m,
                         "gain": np.nanmean(diff), "lo95": np.percentile(bs, 2.5),
                         "hi95": np.percentile(bs, 97.5), "P(arm better)": (bs > 0).mean()})
    D = pd.DataFrame(rows)
    print(D.round(6).to_string(index=False))
    D.to_csv(OUT / "deltas.csv", index=False)
    p = D.iloc[0]
    pick = "C_pos" if p["P(arm better)"] > 0.5 else "B_ln"
    q = D[(D.iloc[:, 0] == f"{'B_ln' if pick == 'B_ln' else 'C_pos'} − A_knn") & (D["metric"] == "crps")].iloc[0]
    final = "A_knn" if q["P(arm better)"] > 0.5 else pick
    print(f"\nDECISION: C_pos vs B_ln CRPS gain {p['gain']:+.6f} [{p['lo95']:+.6f}, {p['hi95']:+.6f}] "
          f"P {p['P(arm better)']:.3f} → {pick}; A_knn vs {pick} P {q['P(arm better)']:.3f} → USE {final}")

    print("\ncalibration of S(n) at n = 32, 64, 100, 150, 200 (predicted vs realized share available):")
    cal = []
    for n in (32, 64, 100, 150, 200):
        s = S[final][:, n - 1]
        dec = pd.qcut(s, 10, duplicates="drop")
        g = pd.DataFrame({"pred": s, "real": (y > n).astype(float), "dec": dec}).groupby("dec", observed=True).mean()
        cal.append(g.assign(n=n))
        print(f" n={n}: " + "  ".join(f"{r.pred:.2f}/{r.real:.2f}" for r in g.itertuples()))
    print("\nCRPS by class:")
    print(sc.groupby("cls")[[f"crps_{a}" for a in ARMS]].mean().round(5).to_string())
    drafted = y <= N_PICKS
    pp = pd.DataFrame({"pos": b["pos"], "logY": np.log(y), "B_mu": mus["B_mu"], "C_mu": mus["C_mu"]})[drafted]
    print("\nby position (drafted players): realized − predicted mean log pick (>0 = goes later than rank implies)")
    print(pp.groupby("pos").apply(lambda g: pd.Series({"n": len(g), "B_resid": (g.logY - g.B_mu).mean(),
                                                       "C_resid": (g.logY - g.C_mu).mean()}),
                                  include_groups=False).round(3).to_string())
    full = LogNormal(True).fit(b)
    print("\nC_pos fitted on all classes: position offsets (mu, log sigma):")
    print(pd.DataFrame({"mu": full.th[3:3 + len(POS)], "log_sigma": full.th[3 + len(POS) + 2:]},
                       index=POS).round(3).to_string())


if __name__ == "__main__":
    main()
