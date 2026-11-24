"""
Step 2 of the rookie-surplus plan: the draft-slot curve and excess value
(2026-11-01).

Product question (user, 2026-11-01): how much excess value can a team expect
from picking a certain player at a certain slot? That needs a baseline: what
a pick at that slot is worth on average (its opportunity cost). Then
    excess(player) = surplus(player) − slot_curve(pick).
Historical realized excess is the training label for a future model of
E[excess | player, slot]; this script builds the curve and the label.

Data: rookie surplus label v3 (00zo; EPA-priced skill positions), drafted
non-specialists (no K/P/LS), classes 2006-22, surplus_y14 (y1-y4 % of cap).
The option-year version (surplus_y15) uses classes 2011-21.

Curve construction. Surplus = value − cost, and the two have different
shapes:
    E[value | pick]: smoothed, then forced non-increasing in pick (a better
                     slot should not expect less playing value).
    E[cost | pick]:  smoothed (cost is nearly deterministic rookie scale).
    slot_curve      = E[value] − E[cost]   (surplus need not be monotone:
                     Massey & Thaler's rise-then-fall comes from here)
Smoother: local-linear regression in log(pick) with a nearest-neighbour
tricube window of k players; k chosen by leave-one-class-out squared error
on value, taking the smoothest k within 0.5% of the best (the criterion is
flat). A first pass with a fixed Gaussian bandwidth in log(pick) left each
top pick smoothed almost alone (pick 1 = its 17 players, mostly QBs:
0.21 → 0.03 → 0.07 at picks 1/3/10). The direct kernel smooth of surplus is
also printed as a check. Bands: class-cluster bootstrap (classes resampled,
500 reps), whole pipeline re-run per rep.
UDFAs get a single slot value: mean surplus of UDFAs who reached the NFL
(non-NFL UDFAs have surplus 0 and are outside this table, so it is an
upper bound for "an undrafted player").

Checks printed: era stability (2006-10 / 2011-15 / 2016-22 curves), the
late-round-1 dip (mean surplus picks 21-32 minus 33-48, bootstrap P), a
sensitivity where unlabelled non-specialist picks (mostly 2006-10 rounds
6-7) count as surplus 0, and realized excess by position × round (a first
look at positional mispricing; step 3 tests it properly).

Outputs (data/processed/_rookie_surplus_20261101/):
    slot_curve.parquet     pick 1..262: value, cost, surplus (+ lo/hi 90%),
                           surplus_y15; plus a "UDFA" row
    rookie_excess.parquet  per player: surplus_y14, slot, excess_y14
                           (+ y15 versions where complete)

RESULT (2026-11-01, label v3.1, 4,146 drafted players, k = 380):
  slot surplus (% of cap, y1-4): pick 1 0.148 [0.104, 0.195] 90%; 3 0.082;
  5 0.054; 10 0.070; 15 0.073; 25 0.058; 32 0.062; 40 0.069; 64 0.063;
  100 0.037; 160 0.021; 250 0.012; UDFA (NFL only) 0.006.
  Value falls smoothly with pick; the cost scale falls faster through
  picks 1-15 and 21-40, which makes the bumps. Late-R1 dip (21-32 vs 33-48)
  −0.014 [−0.027, −0.004], P 0.98. Top-10 vs 11-20 tie (P 0.61).
  Era: the top of round 1 rises by era (pick 1 0.057 / 0.117 / 0.155);
  picks 20+ are stable. Pre-2011 values come from the compressed depth-
  chart model, so the pooled top is conservative.
  Sensitivity (unlabelled picks = 0): moves rounds 6-7 by <= 0.002.
  Realized excess by position: QB R1 +0.139 (but median < 0: a few huge
  hits); WR, EDGE, OL R2-3 positive; RB and LB negative in every round
  (R1 −0.074 / −0.069); DB R1 −0.040.
  The LOCO criterion is flat (40.6-40.8 for k >= 80); k only sets
  smoothness.

MOVED (2026-11-01): the code now lives in src/models/slot_curve.py
(build_slot_curve) and runs in the pipeline as step 7.5. This script is a
thin CLI over the experiment-directory label.

Usage: 00zr_slot_curve.py [--n-boot 500]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR  # noqa: E402
from src.models.slot_curve import build_slot_curve  # noqa: E402

D = PROCESSED_DIR / "_rookie_surplus_20261101"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    build_slot_curve(D / "rookie_surplus_labels.parquet", D, a.n_boot)


if __name__ == "__main__":
    main()
