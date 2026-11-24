"""
QB-only: does an entrenched starter on the drafting team lower a drafted
QB's realized rookie value beyond what the pick implies? (pre-registered
2026-11-02, a narrow follow-up chosen AFTER seeing 01c)

01c: four team-room features did not improve out-of-sample MSE for all
positions, but the all-class fit showed the largest, most plausible effect
for QBs behind an entrenched starter (years 1-2). One QB plays every snap,
so if the room matters anywhere it is here. This test spends one parameter
on it.

Data: 01c's cohort and out-of-fold predictions
(data/processed/_team_roster_state_20261102/oof.parquet): drafted QBs,
classes 2014-22; B0_y14 = LOCO pick + position prediction (pooled
positions); top = the largest prior-season snap share in the team's QB room
on week 1 (see 01c).
Arms (leave-one-class-out within QBs; fit on the other classes' QBs):
    A0  B0 + a           (QB-level correction: mean residual y − B0)
    A1  B0 + a + b·top   (OLS on the residual; b unconstrained)
Outcome: value_y14 (primary); y12, y34 reported with their own B0.
PRIMARY: MSE, A1 vs A0. Adopt (a QB entrenched-starter adjustment for the
pick-time tool) if P(A1 better) > 0.5 (bootstrap of QBs within class,
2000) AND the real gain exceeds the 80th percentile of a permutation null
(top shuffled among QBs within class, 500 permutations; the same LOCO
fit). Reported: b (all-class fit) with bootstrap interval; the indicator
form (top >= 0.9) as a check; y12 and y34; listing of entrenched QBs.

RESULT (2026-11-02): no adjustment (primary fails).
103 QBs, 2014-22, 44 behind a starter with top >= 0.9.
    y14  top: gain −4.78e-4 [−15.1, +4.5], P 0.18, permutation pct 0.35;
         b = −0.038 [−0.139, +0.064]. Indicator form: tie (P 0.50).
    y12  top: gain +1.71e-4, P 0.72, permutation pct 0.98;
         b = −0.053 [−0.101, −0.007] — the one clear effect: behind an
         entrenched starter a drafted QB plays less in years 1-2.
    y34  top: gain −1.91e-4, P 0.14; b = +0.017 [−0.047, +0.080] — none.
Reading: the depth chart delays a QB, it doesn't change his four-year
value; the cases that matter go both ways in years 3-4 (Love, Hurts,
Jackson took over; Hackenberg, Mond, Trask never played), and the pick
already prices who teams expect to play. The y12 effect was not the
primary and is not wired.

Rechecked on the v4 label (01e, 2026-11-02): same conclusion, y14 top gain
−4.33e-4, P 0.20, permutation pct 0.38; y12 b −0.044 [−0.095, +0.003]
(perm pct 0.94). v3.1 outputs kept in _qb_entrenched_20261102_v31.

Usage: 01d_qb_entrenched.py [--n-boot 2000] [--n-perm 500]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR  # noqa: E402

SRC = PROCESSED_DIR / "_team_roster_state_20261102" / "oof.parquet"
OUT = PROCESSED_DIR / "_qb_entrenched_20261102"


def loco(q, y, x):
    """Per class: A0 = B0 + mean residual; A1 = B0 + OLS(residual ~ x),
    both fit on the other classes."""
    r = q[y] - q[f"B0_{y}"]
    a0, a1 = np.empty(len(q)), np.empty(len(q))
    for c in q["draft_year"].unique():
        te = (q["draft_year"] == c).to_numpy()
        a0[te] = q.loc[te, f"B0_{y}"] + r[~te].mean()
        X = np.column_stack([np.ones((~te).sum()), q.loc[~te, x]])
        beta = np.linalg.lstsq(X, r[~te], rcond=None)[0]
        a1[te] = q.loc[te, f"B0_{y}"] + beta[0] + beta[1] * q.loc[te, x]
    return a0, a1


def gain(q, y, a0, a1, idx=None):
    e0, e1 = (a0 - q[y]) ** 2, (a1 - q[y]) ** 2
    d = (e0 - e1).to_numpy()
    return d.mean() if idx is None else d[idx].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    rng = np.random.default_rng(0)
    q = pd.read_parquet(SRC)
    q = q[q["draft_group"] == "QB"].reset_index(drop=True)
    q["ent"] = (q["top"] >= 0.9).astype(float)
    print(f"QBs {len(q)}, classes {q['draft_year'].min()}-{q['draft_year'].max()}, "
          f"entrenched (top >= 0.9) {int(q['ent'].sum())}")
    groups = [g.index.to_numpy() for _, g in q.groupby("draft_year")]
    idxs = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(args.n_boot)]
    rows = []
    for y in ("y14", "y12", "y34"):
        for x in ("top", "ent"):
            a0, a1 = loco(q, y, x)
            g = gain(q, y, pd.Series(a0), pd.Series(a1))
            bs = np.array([gain(q, y, pd.Series(a0), pd.Series(a1), i) for i in idxs])
            null = []
            for _ in range(args.n_perm):
                p = q.copy()
                for ix in groups:
                    p.loc[ix, x] = rng.permutation(p.loc[ix, x].to_numpy())
                b0, b1 = loco(p, y, x)
                null.append(gain(p, y, pd.Series(b0), pd.Series(b1)))
            null = np.array(null)
            r = q[y] - q[f"B0_{y}"]
            X = np.column_stack([np.ones(len(q)), q[x]])
            b_all = np.linalg.lstsq(X, r, rcond=None)[0][1]
            b_bs = [np.linalg.lstsq(X[i], r.to_numpy()[i], rcond=None)[0][1] for i in idxs]
            rows.append({"outcome": y, "feature": x, "mse_A0_x1e4": 1e4 * np.mean((a0 - q[y]) ** 2),
                         "gain_x1e4": 1e4 * g, "lo95": 1e4 * np.percentile(bs, 2.5),
                         "hi95": 1e4 * np.percentile(bs, 97.5), "P(A1 better)": (bs > 0).mean(),
                         "perm_pct": (null < g).mean(), "null_p80_x1e4": 1e4 * np.percentile(null, 80),
                         "b_all": b_all, "b_lo95": np.percentile(b_bs, 2.5), "b_hi95": np.percentile(b_bs, 97.5)})
    D = pd.DataFrame(rows)
    print(D.round(4).to_string(index=False))
    D.to_csv(OUT / "results.csv", index=False)
    p = D[(D["outcome"] == "y14") & (D["feature"] == "top")].iloc[0]
    ok = p["P(A1 better)"] > 0.5 and p["perm_pct"] > 0.8
    print(f"\nDECISION: A1 vs A0 y14 gain {p['gain_x1e4']:+.3f}e-4 [{p['lo95']:+.3f}, {p['hi95']:+.3f}] "
          f"P {p['P(A1 better)']:.3f}; permutation percentile {p['perm_pct']:.2f}; b = {p['b_all']:+.4f} "
          f"[{p['b_lo95']:+.4f}, {p['b_hi95']:+.4f}] → {'ADOPT' if ok else 'no adjustment'}")
    cols = ["player_name", "draft_year", "team", "overall_pick", "top", "y12", "y34", "y14", "B0_y14"]
    print("\nentrenched-starter QBs (top >= 0.9):")
    print(q[q["ent"] == 1].sort_values("overall_pick")[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
