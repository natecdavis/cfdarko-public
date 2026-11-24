"""
Rookie-contract surplus value label (v1, 2026-11-01).

Goal (user, 2026-11-01): estimate how much a player is worth over his rookie
contract, and how much excess value a team can expect from picking him at a
given slot. Until now the system predicted long-run quality (y3-4 snap
share, second-contract APY), with no cost side and no slot.

Definition, all in % of the salary cap (no discounting):
    value_y  = f_pos(role share) × games played / team REG games
                                                     if role share > 0
             = replacement level (≈ league minimum)  if role share == 0 but
                                                     special-teams share >= 0.25
             = 0                                     otherwise
        role share: max(offense, defense) snap pct averaged over the games
        the player appeared in, i.e. his role when available.
        f_pos: 00zf's veteran pay curve fit on that same per-game-appeared
        share (`--pay-share appeared`): what the market pays per season for
        that role at that position. Scaling by games played charges for
        missed games. A curve fit on season share (missed games = 0) overprices
        partial seasons, because injured veterans keep their pay while their
        season share falls (v1 first pass: QB at 0.5 season share → 0.12 of
        the cap).
    cost_y   = rookie-deal APY % of cap, charged
             - every season y1-y4 for rounds 1-2 (largely guaranteed), and
             - only in seasons with any snap (offense/defense/ST) for rounds
               3-7 and UDFAs (non-guaranteed; a cut player stops costing).
        Rookie APY = OTC APY ($) of the deal signed in the draft year ÷ that
        year's cap (median implied cap of big contracts; OTC rounds
        apy_cap_pct to 0.001). Missing drafted deals → per-year isotonic in
        overall pick; missing UDFA deals → that year's median UDFA APY.
    y1..y4   = draft_year .. draft_year+3.
    y5       = 5th-year option, first-rounders only: if an option contract
               exists (round 1, 1 year, signed draft_year+4), value_y5 −
               option APY; otherwise 0 (declined, or replaced by an extension
               at market price, which carries no surplus).
    surplus_y14 = Σ value − Σ cost over y1-y4      (complete: classes 2016-22)
    surplus_y15 = surplus_y14 + option-year surplus (complete: classes 2016-21)
Players who never reach the NFL have value 0, cost 0 and surplus 0. They are
in the universe the model scores, but this table covers bridge players.
K/P excluded (no snap-share market curve).

v2 (2026-11-01): classes back to 2006.
    Seasons 2013-2015 use real snap counts (nflverse, fetched to the
    experiment cache; 2012 is not published). Aggregated the same way as
    data/raw/snap_counts.parquet (verified identical on 2016).
    Seasons 2006-2012 use a depth-chart model. Features per (player,
    season): share of REG weeks listed first string, second string, and at
    all (offense/defense formations). Target: the snap-based season value
    above. One LightGBM per position, monotone increasing in all three
    features, fit on 2013-2024 seasons of bridge players' y1-y4 windows.
    Validated leave-one-season-out, and on classes 2013-21 by comparing the
    depth-chart y1-y4 sum against the snap-based one. Cost-charging presence
    in those seasons = listed on a depth chart at least once.
    Classes before 2011 (pre-CBA rookie pay) are costed with the post-2011
    rookie scale (pooled isotonic in overall pick, % of cap) and get no
    option year. The goal is today's economics, not the old wage scale.

v3.1 (2026-11-01): players are priced at their NFL position (most common
OTC position_group across their contracts, ties → latest), falling back to
the college value_group. The veteran curves were already grouped by OTC
position, so pricing by college group mispriced position switchers: Von
Miller and Khalil Mack (college LB) were capped at the off-ball LB price
(~0.032/season) instead of EDGE. ~10% of players move, mostly among
DL/EDGE/LB. Output: group = pricing (NFL) position, draft_group = college.

Known approximations (v1): snap presence stands in for roster presence (the
roster cache ends in 2017), so IR or practice-squad seasons for rounds 3+
cost 0. Dead money is ignored. Snaps × price credits playing time, not play
quality. The pay curves come from 2023-25 contracts only. Classes before
2016 lack snap data for y1-y4; the depth-chart extension is v2.

Sanity check (the reason to look before building on it): mean surplus by
pick should rise from the top of round 1 to a peak somewhere in rounds
1-3 and then decline (Massey & Thaler's surplus curve), while value alone
falls with pick.

v3 (2026-11-01, --quality epa; DEFAULT since 00zq adopted it): quality-adjusted price for QB/RB/WR/TE.
    00zp found that snaps decide WHETHER a player earns a second contract
    and quality decides HOW MUCH (rookie-window EPA adds +0.08 to +0.13
    within-position rho among players paid > 1% of cap). So the price of a
    season becomes f_pos(role share, q) instead of f_pos(role share).
    q = within-(position × season) percentile of shrunk EPA per opportunity:
        rate = (EPA + k·mu) / (opportunities + k), opportunities = pass
        attempts + sacks + carries + targets, EPA = passing + rushing +
        receiving (nflverse seasonal REG), mu = the position-season EPA per
        opportunity, k = QB 150 / RB 60 / WR 40 / TE 40. A player with snaps
        but no touches gets rate = mu. Percentiles are over all position-
        season players with at least one opportunity. 2025 has no EPA yet
        (nflverse 404), so 2025 seasons get the position's neutral q.
    f_pos: monotone LightGBM on (share, q), fit on veteran contracts in force
    in 2016-24 (--pay-fit wide), then multiplied per position by
    Σ actual / Σ fitted APY over 2023-24 veteran seasons, so the LEVEL is
    today's market while the shape (incl. the quality slope) uses 9 seasons.
    Without it QB/WR values fell 17% and LB rose 23% vs v2, because QB and
    WR cap shares rose over the window. A per-position constant leaves the
    within-position validation (00zq) unchanged. Other positions keep a 1-D isotonic curve
    on the same seasons. The depth-chart model gains q as a fourth monotone
    feature. Positions are from the label (cohort) or the OTC contract.
    --pay-fit wide without --quality gives a matched snaps-only reference
    (same seasons, same LightGBM for skill groups) so an A/B isolates q.
    --xval-fold k (of --xval-k K) drops ALL contracts of the validation-
    cohort players (drafted 2011-20) whose gsis hashes to fold k from the
    pay fit. Run once per fold and take each player's label from his own
    fold's run: cross-fitted, so no player is priced with a curve fit on his
    own contracts, while the veteran sample stays near full (validation
    runs only; see 00zq). A first attempt dropped every 2011-20 second
    contract at once and gutted the 2016-24 veteran sample (QB 289 seasons).
    OL: nflverse's id table has no pfr→gsis for OL, so the wide pay fit
    falls back to unique-name matching like 00zf (names from nflverse snap
    counts, cached).

MOVED (2026-11-01): the code now lives in src/models/rookie_value.py
(build_label) and runs in the pipeline as step 7.5. This script is a thin
CLI for experiment builds. The v1/v2 "--pay-fit zf" mode (2023-25 isotonic
curves from 00zf) was dropped in the move; it is in git history (4a82749).

Usage:
    00zo_rookie_surplus_label.py [--out-dir DIR] [--out-name NAME]
        [--quality none|epa|pff] [--xval-fold K [--xval-k 5]]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR  # noqa: E402
from src.models.rookie_value import build_label  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR / "_rookie_surplus_20261101")
    ap.add_argument("--out-name", default="rookie_surplus_labels.parquet")
    ap.add_argument("--quality", choices=["none", "epa", "pff"], default="pff")
    ap.add_argument("--xval-fold", type=int, default=None)
    ap.add_argument("--xval-k", type=int, default=5)
    a = ap.parse_args()
    build_label(a.out_dir / a.out_name, a.quality, a.xval_fold, a.xval_k)


if __name__ == "__main__":
    main()
