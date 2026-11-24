"""
Merged board: one pre-draft ranked list from two models, on one scale.

RESULT (2026-10-31): FAIL on the pre-registered bar, but the merge is no
worse anywhere and better at the top. 29,565 players, classes 2017-22, 1,708
board, 884 off-board NFL. Usage outcome, Δ [95%]:
    merged_dep − dep_cal  topM   +0.0044 [-0.0027, +0.0118]   PRIMARY
                          top100 +0.0214 [+0.0034, +0.0384]
                          top200 +0.0122 [-0.0027, +0.0194]
    placebo (pre43 − pre) topM   +0.0015 [-0.0024, +0.0035]
    board_only beats merged_dep at top100 (-0.006, sig) and ties at topM.
Calibration: the board curve tracks actual usage on board players (top
deciles 0.42/0.46, 0.56/0.57, 0.73/0.73). The single model under-predicts
the board's top decile (0.60 vs 0.68). The seam is a wash: about 60 off-board
players per class enter the top M under either arm, at actual usage 0.12,
vs 0.11 for the board players they displace. The off-board side is
over-predicted at its very top: ~2 off-board players per class reach the
top 100, predicted 0.38, actual 0.06 (the single-observation isotonic top
plateau again, cf. 08c). value_target outcome: same picture, nothing sig.

Why (2026-10-31): for players with a hasan_rank nothing cfDARKO builds beats
the board ([[project_board_is_ceiling]]); off-board players need a model
([[project_offboard_profile]]). Ranks from two populations can't be merged,
but expected values can. Both sides predict the same quantity:

    usage = y3-4 NFL snap share (usage_y34), 0 if the player never reached
            the NFL. Alive for UDFAs, unlike second-contract APY.

    board side      E[usage | hasan_rank, position]. Per-position isotonic
                    (decreasing in rank), shrunk toward a pooled isotonic
                    curve with weight n_pos / (n_pos + K_BOARD).
    off-board side  E[usage | model percentile, position]. The model's score
                    is turned into a percentile within (class × position)
                    among OFF-BOARD players, then mapped through a shrunk
                    per-position isotonic curve fit on off-board rows only.
    merged score    board side for board players, off-board side otherwise.

Every calibrator for eval fold (train_end, val_start, val_end) is fit only on
board-era rows (last college season >= 2013) with last season <= train_end,
using the models' OOF scores from EARLIER folds. Eval folds: val 2016-18 and
2019-21 (draft classes 2017-22); fold 2013 has no board seasons before it.
The holdout fold is never touched.

Arms (all in usage units, so they are directly comparable across positions
and populations):
    dep_cal       SINGLE model: deployed 8b impact OOF, percentile within
                  (class × position) over ALL players, mapped to usage the
                  same way. The incumbent: one model, board as a feature.
    pre_cal       same for 00zd's pre-draft two-stage usage model
    merged_dep    board curve + off-board side from the deployed model
    merged_pre    board curve + off-board side from 00zd pre
    merged_pre43  same with 00zd's seed-43 refit (placebo)
    board_only    board curve; off-board players ranked last (status quo)

PRE-REGISTERED (written before any run). Primary metric: per draft class,
take the top M players of the whole class list, where M = number of board
players in that class; mean usage of those M, averaged over classes 2017-22.
Primary contrast: merged_dep − dep_cal (same model, merge vs no merge).
PASS if its 95% paired bootstrap interval (players resampled within class)
is > 0 AND the placebo merged_pre43 − merged_pre interval includes 0.
Reported, not decided on: merged_pre − pre_cal; top-100/200/500 capture;
seam capture (pool = hasan 151-300 ∪ off-board, top 150 per class);
contributor AUC; value_target versions; calibration by population; how many
off-board players each arm puts into the top M.

CAP ARM — PRE-REGISTERED 2026-10-31, after the result above, before any run
of it. Problem: the off-board calibrator's top isotonic block rests on a
handful of players (pred 0.38, actual 0.06 in the top 100). Fix, chosen
a priori with no tuning: cap every off-board prediction at the mean usage of
the top 1% of off-board training rows for that position (off-board
percentile >= 0.99), shrunk toward the pooled top-1% mean with weight
n / (n + K_OFF), n = top-1% rows for the position. Board side unchanged.
    merged_dep_cap / merged_pre_cap / merged_pre43_cap
PRIMARY: merged_dep_cap − dep_cal on topM, same bar as before (95% interval
> 0 AND placebo merged_pre43_cap − merged_pre_cap interval includes 0).
Reported, not decided on: merged_dep_cap − merged_dep and − board_only on
top100 / top200 / topM; off-board players in the top 100 and their
predicted vs actual usage. Run on the same 00zd OOF (oof_00zd.parquet), so
the uncapped arms reproduce exactly.

CAP RESULT (2026-10-31): FAIL. merged_dep_cap − dep_cal topM +0.0044
[-0.0026, +0.0118], placebo +0.0015 [-0.0025, +0.0033]. Caps were 0.14-0.25
by position and fold. The cap does what it was meant to at the top: no
off-board player reaches the top 100 any more, top100 vs dep_cal +0.027
[+0.010, +0.045], vs uncapped +0.006 [+0.001, +0.015], equal to board_only.
It can't move topM, because the cap sits above the seam (~0.11-0.12), where
the off-board model is no better than the bottom of the board: about 60
off-board entrants per class at actual 0.118 (predicted 0.153 capped) vs
0.113 for the displaced board players. value outcome: same.

VALUE-BUILT ARM — PRE-REGISTERED 2026-10-31, after the cap result, before
any run of it. `--cal-target value`: every calibrator (board curve, off-board
map, top-1% cap, single-model calibration) is fit to value_target (0 if never
NFL) instead of usage. Same 00zd OOF, same folds, same arms, same cap rule.
PRIMARY: merged_dep_cap − dep_cal on topM, scored on the VALUE outcome (the
target both lists are now built on), same bar: 95% interval > 0 AND placebo
merged_pre43_cap − merged_pre_cap interval includes 0.
Reported, not decided on: the same contrasts on the usage outcome; the
value-built capped merge vs the usage-built one (`--compare-with` the usage
run's merged_oof.parquet; arm usage_built_cap) on BOTH outcomes, since
neither outcome is neutral between them; off-board entrants into the top M.

VALUE-BUILT RESULT (2026-10-31): FAIL. merged_dep_cap − dep_cal topM on
value +0.0032 [-0.0065, +0.0123], placebo -0.0019 [-0.0062, +0.0022]. Top
100 +0.029 [+0.006, +0.048], as with the usage build. Against the
usage-built capped merge: value outcome topM +0.0043 [-0.0015, +0.0098],
seam150 +0.0085 (ns); usage outcome topM -0.0007 (ns) but top100 -0.016
[-0.029, -0.000]. The value board curve is compressed at the top (top
three deciles predict 0.53 / 0.58 / 0.65 vs actual 0.62 / 0.62 / 0.67), so it
separates the best board players less well. It does NOT admit fewer
off-board players: 58 per class into the top M (usage build 60), at value
0.235 vs 0.256 for the board players they displace. The usage build stays
the preferred version. A reproduction run of the usage build (--n-boot 2)
matched cap/merged_oof.parquet exactly.

RECENCY-WEIGHTED ARM — PRE-REGISTERED 2026-10-31, before any run of it.
`--weight recent --half-life H`: every training row in every calibrator of
the snap-share build (board curve, off-board map, top-1% cap, single-model
calibration) gets weight 0.5 ** (age / H), age = latest training class −
the row's class (0 for the newest class). Shrinkage weights still use raw
counts. Only the calibrators are reweighted; the off-board model scores
(00zd, deployed 8b) are unchanged OOF. Training spans only classes 2014-16
(fold 2016) and 2014-19 (fold 2019), so there is little room to move.
PRIMARY: H = 3. Contrast: weighted merged_dep_cap − the unweighted
snap-share capped merge (`--compare-with cap/merged_oof.parquet`, arm
base_cap) on topM, usage outcome. PASS if its 95% interval > 0 AND the
seed placebo (weighted merged_pre43_cap − merged_pre_cap) includes 0.
Reported, not decided on: H = 1.5; `--weight reverse` (oldest class
weighted most, H = 3) as a directional check; value outcome; the same
contrasts vs dep_cal.

RECENCY RESULT (2026-10-31): FAIL. Weighted − unweighted capped merge, topM,
usage outcome:
    recent H=3   (primary)  +0.0014 [-0.0011, +0.0040]   placebo +0.0004
    recent H=1.5            +0.0022 [-0.0017, +0.0053]
    reverse H=3             -0.0017 [-0.0050, +0.0012]
Same picture on the value outcome. The ordering is the right way round
(reverse < none < H3 < H1.5) but every step is inside the noise. In the
H=1.5 run, the merge vs the (also weighted) deployed calibration reaches
topM +0.0076 [+0.0002, +0.0134]. That is a reported contrast in the fourth
variant tried, barely clear of 0, so not a pass. Caveat: these folds give
the most weight to classes whose y3-4 outcomes would not yet be observed at
draft time (labels settle ~4 years later), so the CV flatters recency; in
real use the newest usable class is about T-4. The unweighted rerun
reproduced cap/merged_oof.parquet exactly.

DOLLARS-BUILT ARM — PRE-REGISTERED 2026-10-31, before any run of it.
`--cal-target dollars`: y3-4 snap share stays the measure, but each training
row's usage is converted to the market price of that snap share at its
position, and every calibrator is fit to that:
    dollars = f_pos(usage) if usage > 0 else 0
    f_pos   isotonic (increasing) fit of APY % of cap on season snap share,
            for veteran player-seasons 2023-2025 pooled: the contract in force
            that season (latest signed with year_signed <= season <
            year_signed + years), excluding rookie deals (signed in the draft
            year) and 5th-year options. Snap share = max(offense, defense)
            pct, mean over games, the same definition as usage_y34. Position
            = the contract's position_group (EDGE separate). Player-seasons
            with no veteran contract in force are dropped, not zeroed.
            pfr -> gsis via nflverse import_ids, then a unique normalized-name
            match against the contract table.
Same 00zd OOF, folds, arms, cap rule, no recency weighting.
PRIMARY: dollars-built merged_dep_cap − the snap-share-built capped merge
(`--compare-with cap/merged_oof.parquet`, arm base_cap) on topM, scored on
REALIZED pay: best_apy_by_y5 (best post-rookie APY % of cap signed by y5,
0 if none or never NFL). PASS if its 95% interval > 0 AND the seed placebo
(merged_pre43_cap − merged_pre_cap) on the same outcome includes 0.
Reported, not decided on: the usage / value / dollars outcomes; the same
contrasts vs dep_cal; the curves; the position mix of each list's top 100.

DOLLARS RESULT (2026-10-31): FAIL, a dead tie. Dollars-built − snap-share-
built capped merge on realized pay (best APY % of cap by y5), topM -0.0001
[-0.0002, +0.0001] on a base of 0.0173 (-0.6% relative); top100 +0.0002
[-0.0006, +0.0013]; placebo 0.0000. Ties on usage, value and dollars too.
Curves (veteran contracts in force, 2023-25; 6,454 of 6,519 player-seasons
linked, 3,590 with a veteran deal) are steep: at 95% snap share QB 0.158,
WR 0.144, DL 0.108, EDGE 0.104, RB 0.074, TE 0.072, OL 0.053, DB 0.050,
LB 0.036 of the cap. The conversion only reorders across positions, and it
moved the top 100 less than expected: QB 4.7 -> 6.8 per class, LB 8.0 -> 6.2,
OL 22.8 -> 20.7, the rest within ~1. The board alone still has the best topM
on realized pay (0.0176), as on value: real pay favors drafted players.
Side note: usage_y34 averages snap pct over the games a player appeared in,
so an injured or part-time season can read as a high share.

RERUN ON FIXED usage_y34 (2026-10-31, after 8f88777: missed games now count
as 0). 00zd refit (its stage 2 trains on usage), usage-built arms, same cap
bar: merged_dep_cap − dep_cal topM +0.0030 [-0.0012, +0.0107], placebo
-0.0022 [-0.0042, +0.0018] → FAIL, as before (+0.0044 on the old label).
Top 100 +0.028 [+0.013, ...] still holds. New-label vs old-label capped
merge on realized pay: topM +0.0001, top100 -0.0001 — the same list for
practical purposes. Seam still a wash: 59 off-board entrants per class at
0.086 vs 0.084 for the board players displaced. The board curve stays well
calibrated (top deciles 0.33/0.35, 0.46/0.46, 0.63/0.60); the single model
still under-predicts the board's top decile (0.50 vs 0.57).

DOLLARS RERUN ON FIXED usage_y34 — PRE-REGISTERED 2026-11-01, before any
run of it. The dollars arm predates the missed-games fix. Rerun on the
current state (role-drop retune, split identities, fixed labels), both
arms fresh on the same 00zd refit:
    usage-built   `--out-dir R/usage` (refits 00zd)
    dollars-built `--out-dir R/dollars --offboard-oof R/usage/oof_00zd.parquet
                   --cal-target dollars --pay-share season
                   --compare-with R/usage/merged_oof.parquet`
`--pay-share season`: the pay curve's snap share now counts missed games
as 0 (n_games / team REG games, capped at 1), the same definition as the
fixed usage_y34. Same primary and bar as the original dollars arm:
dollars-built merged_dep_cap − usage-built merged_dep_cap, topM on realized
pay (best_apy_by_y5); PASS if its 95% interval > 0 AND the seed placebo
(merged_pre43_cap − merged_pre_cap) on that outcome includes 0.

DOLLARS RERUN RESULT (2026-11-01): a dead tie again, so FAIL on the bar.
On realized pay: topM −0.0000 [−0.0002, +0.0001], top100 +0.0003, placebo
0.0000. With season shares the curves are steeper for WR/QB (at 95%: QB
0.180, WR 0.137, EDGE 0.101, DL 0.084, RB 0.074, TE 0.068, OL 0.052, DB
0.045, LB 0.037). Top-100 mix moves more than in the original run: QB 4.7 → 7.5
per class, EDGE 13.8 → 14.8, LB 7.7 → 5.8, OL 22.8 → 21.5. Caveat: realized
second-contract pay measures long-run quality. It is not the rookie-contract
value the product is meant to estimate. Outputs:
data/processed/_merged_board_dollars_rerun_20261101/.

SUB-POSITION ARM — PRE-REGISTERED 2026-10-31, before any run of it.
`--board-sub`: the snap-share capped merge, but the board curve is split by
the BOARD's own position label (hasan.csv pos, joined on draft year +
rank; tied ranks with different positions are left unmapped) inside three
model groups:
    DB   CB (CB, NCB)  vs  S (S, FS, SS)
    OL   OT (OT)       vs  IOL (OG, OC, IL)
    DL   1T/3T (DL1T, DL3T, NT, 3T)  vs  5T (DL5T, 5T)
A sub-curve applies only when the board label belongs to the player's own
model group (e.g. a board "DL5T" whose model group is EDGE keeps the EDGE
curve); unmapped labels (2015's generic "OL", ID/IDL, ...) keep the parent.
Each sub-curve is an isotonic fit (decreasing in log rank) on that
sub-position's training board rows, shrunk toward the PARENT group's curve
with weight n_sub / (n_sub + K_BOARD). Off-board side unchanged. Same
00zd OOF (usage_fix run), folds, cap, no weighting.
PLACEBO: the same arm with sub labels permuted within (model group ×
class) among board players that have one (seed 0).
PRIMARY: merged_dep_cap_sub − merged_dep_cap on topM, usage outcome. PASS
if its 95% interval > 0 AND merged_dep_cap_subperm − merged_dep_cap
includes 0. Reported, not decided on: top100/top200; board_rho_split =
Spearman(score, outcome) among board players within (DB/OL/DL × class),
averaged, the metric most directly exposed to the split; value and realized
pay outcomes; the fitted sub-curves.

SUB-POSITION RESULT (2026-10-31): FAIL. merged_dep_cap_sub − merged_dep_cap
topM +0.0011 [-0.0017, +0.0025]; permuted placebo +0.0000 [-0.0038,
+0.0011]. The metric most exposed to the split, board_rho_split, moves
the WRONG way: real labels -0.006 [-0.023, ...], permuted labels -0.002.
The real split does no better than a random one. Top100 -0.005 (ns).
Value and realized pay: same picture. The fitted sub-curves don't agree
across folds: at rank 10, S > CB in fold 2016 (0.79 vs 0.72) and S = CB in
2019 (0.69 vs 0.69); IOL > OT in 2016 (0.78 vs 0.72) and OT > IOL in 2019
(0.82 vs 0.77); the DL 1T/3T vs 5T curves are near-identical. The board's
rank already prices sub-position: the 9 model groups (EDGE split from
DL) are enough.

Usage:
    00zf_merged_board.py --out-dir DIR [--n-boot 500] [--offboard-oof PATH]
    [--cal-target usage|value|dollars] [--compare-with MERGED_OOF [--compare-label L]]
    [--weight none|recent|reverse] [--half-life H] [--board-sub]
    [--pay-share appeared|season]
    --offboard-oof reuses a 00zd oof_all_arms.parquet instead of refitting.
"""
import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
BOARD_ERA_START = 2013
K_BOARD = 100          # shrinkage: board players per position for half weight
K_OFF = 50             # shrinkage: off-board NFL players per position for half weight
EVAL_FOLDS = (2016, 2019)
CONTRIB = 0.25
CAP_Q = 0.99
ARMS = ["dep_cal", "pre_cal", "merged_dep", "merged_pre", "merged_pre43",
        "merged_dep_cap", "merged_pre_cap", "merged_pre43_cap", "board_only"]
PAIRS = [("merged_dep_cap", "dep_cal"), ("merged_pre43_cap", "merged_pre_cap"),
         ("merged_dep_cap", "merged_dep"), ("merged_dep_cap", "board_only"),
         ("merged_dep", "dep_cal"), ("merged_pre43", "merged_pre"),
         ("merged_pre", "pre_cal"), ("merged_dep", "board_only"),
         ("dep_cal", "board_only")]


class ShrunkIso:
    """Per-position isotonic curve shrunk toward a pooled one."""

    def __init__(self, increasing: bool, k: float):
        self.increasing, self.k = increasing, k

    def _iso(self):
        return IsotonicRegression(increasing=self.increasing, out_of_bounds="clip",
                                  y_min=0.0, y_max=1.0)

    def fit(self, x, y, pos, weight_n=None, sw=None):
        x, y, pos = np.asarray(x, float), np.asarray(y, float), np.asarray(pos)
        sw = np.ones(len(x)) if sw is None else np.asarray(sw, float)
        self.pooled = self._iso().fit(x, y, sample_weight=sw)
        self.per, self.w = {}, {}
        for g in np.unique(pos):
            m = pos == g
            n = m.sum() if weight_n is None else weight_n[g]
            self.per[g] = self._iso().fit(x[m], y[m], sample_weight=sw[m])
            self.w[g] = n / (n + self.k)
        return self

    def predict(self, x, pos):
        x, pos = np.asarray(x, float), np.asarray(pos)
        out = self.pooled.predict(x)
        for g in np.unique(pos):
            if g in self.per:
                m = pos == g
                out[m] = self.w[g] * self.per[g].predict(x[m]) + (1 - self.w[g]) * out[m]
        return out


def _pct(df, score, mask):
    """Percentile of `score` within (class × position) among rows in `mask`."""
    s = df[score].where(mask)
    return s.groupby([df["cls"], df["pos_group"]]).rank(pct=True)


def _load(args, zd, step8):
    folds, _ = step8._split_holdout(step8._expanding_window_folds())
    if args.offboard_oof:
        base = pd.read_parquet(args.offboard_oof)
        logging.info("reusing 00zd OOF %s", args.offboard_oof)
    else:
        rows, feats = zd._build_rows(step8)
        out, fold_of = zd._run_arms(step8, rows, feats, folds)
        base = rows[zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted"]].assign(val_start=fold_of)
        for a in ("pre", "pre_43"):
            base[a] = out[a]["p1"] * out[a]["m2"]
        base = base.dropna(subset=["val_start", "pre"])
        base["val_start"] = base["val_start"].astype(int)
        base.to_parquet(args.out_dir / "oof_00zd.parquet", index=False)
    E = zd._deployed(base)
    E = E.dropna(subset=["dep_impact"]).drop_duplicates(["player_id", "val_start"], keep="last")
    lab = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    lab = lab[lab["player_id"].notna()].assign(player_id=lambda d: zd._pid(d["player_id"]))
    lab = lab.drop_duplicates("player_id")[["player_id", "value_target"]]
    E = E.merge(lab, on="player_id", how="left")
    E["value"] = E["value_target"].fillna(0.0)
    E["cls"] = E["last_college_season_year"].astype(int) + 1
    E["board"] = E["hasan_rank"].notna()
    E = E[E["last_college_season_year"] >= BOARD_ERA_START].reset_index(drop=True)
    # percentiles: off-board-only (for the merge) and all-player (single model)
    for s in ("dep_impact", "pre", "pre_43"):
        E[f"{s}_pct_ob"] = _pct(E, s, ~E["board"])
        E[f"{s}_pct_all"] = _pct(E, s, pd.Series(True, index=E.index))
    return E, folds


SUBPOS = {
    "DB": {"CB": "CB", "NCB": "CB", "S": "S", "FS": "S", "SS": "S"},
    "OL": {"OT": "OT", "OG": "IOL", "OC": "IOL", "IL": "IOL"},
    "DL": {"DL1T": "1T3T", "NT": "1T3T", "DL3T": "1T3T", "3T": "1T3T",
           "DL5T": "5T", "5T": "5T"},
}
SPLIT_GROUPS = tuple(SUBPOS)


def _board_subpositions(E):
    """Board sub-position key ("DB:CB", ...) per row, from hasan.csv's pos via
    (draft year, rank); NaN when unmapped or outside the player's group."""
    from config import ROOT_DIR
    hr = pd.read_parquet(PROCESSED_DIR / "hasan_ranks.parquet")
    hr = hr.assign(player_id=pd.to_numeric(hr["player_id"], errors="coerce")
                   .astype("Int64").astype(str))
    csv = pd.read_csv(ROOT_DIR / "hasan.csv", encoding="utf-8-sig")
    pos = csv.groupby(["year", "rank"])["pos"].agg(lambda p: p.iloc[0] if p.nunique() == 1 else None)
    hr["board_pos"] = [pos.get((int(y), int(r))) for y, r in zip(hr["draft_year"], hr["hasan_rank"])]
    key = hr.drop_duplicates(["player_id", "hasan_rank"]).set_index(["player_id", "hasan_rank"])["board_pos"]
    bpos = pd.Series([key.get((p, r)) if pd.notna(r) else None
                      for p, r in zip(E["player_id"], E["hasan_rank"])], index=E.index)
    sub = [f"{g}:{SUBPOS[g][b]}" if g in SUBPOS and isinstance(b, str) and b in SUBPOS[g] else np.nan
           for g, b in zip(E["pos_group"], bpos)]
    return pd.Series(sub, index=E.index, dtype=object)


def _permute_within(E, col, seed=0):
    rng = np.random.default_rng(seed)
    out = E[col].copy()
    for _, idx in E[E[col].notna()].groupby(["pos_group", "cls"]).groups.items():
        out.loc[idx] = rng.permutation(E.loc[idx, col].to_numpy())
    return out


def _class_weights(cls, weight, half_life):
    if weight == "none":
        return np.ones(len(cls))
    age = cls.max() - cls if weight == "recent" else cls - cls.min()
    return 0.5 ** (age.to_numpy(float) / half_life)


def _wmean(t, col="target"):
    return np.average(t[col], weights=t["w"])


PAY_SEASONS = (2023, 2024, 2025)
PAY_GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]


def _pay_curves(cache_dir: Path, share: str = "appeared") -> dict:
    """Per-position isotonic map: season snap share → APY % of cap of the
    veteran contract in force that season (see DOLLARS-BUILT ARM).
    share="appeared": mean pct over games played (the original run);
    share="season": summed over games / team REG games, capped at 1, which
    matches usage_y34 after the missed-games fix (8f88777)."""
    import nfl_data_py as nfl
    from src.scrapers.nfl_data import _normalize_name, reg_season_games
    snaps_path, ids_path = cache_dir / "snaps_raw_pay.parquet", cache_dir / "nflverse_ids.parquet"
    if not snaps_path.exists():
        raw = nfl.import_snap_counts(list(PAY_SEASONS))
        raw[raw["game_type"] == "REG"].to_parquet(snaps_path, index=False)
    if not ids_path.exists():
        nfl.import_ids()[["pfr_id", "gsis_id"]].dropna().to_parquet(ids_path, index=False)
    raw, ids = pd.read_parquet(snaps_path), pd.read_parquet(ids_path)
    ps = (raw.groupby(["pfr_player_id", "season"])
          .agg(player=("player", "first"), off=("offense_pct", "mean"),
               de=("defense_pct", "mean"), n_games=("offense_pct", "size")).reset_index())
    ps["share"] = ps[["off", "de"]].max(axis=1)
    if share == "season":
        ps["share"] = (ps["share"] * ps["n_games"] / ps["season"].map(reg_season_games)).clip(upper=1.0)
    logging.info("pay curves: snap share = %s", share)

    c = pd.read_parquet(RAW_DIR / "nfl_contracts.parquet")
    for col in ("year_signed", "years", "apy_cap_pct", "draft_year", "draft_round"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.dropna(subset=["gsis_id", "year_signed", "years", "apy_cap_pct"])
    rookie = c["year_signed"] == c["draft_year"]
    fifth = (c["draft_round"] == 1) & (c["years"] == 1) & (c["year_signed"] == c["draft_year"] + 4)
    vet = c[~rookie & ~fifth & c["position_group"].isin(PAY_GROUPS)]

    xw = dict(zip(ids["pfr_id"], ids["gsis_id"]))
    names = c.drop_duplicates("gsis_id").assign(n=lambda d: d["player"].map(_normalize_name))
    uniq = names.groupby("n")["gsis_id"].agg(lambda g: g.iloc[0] if len(g) == 1 else None)
    ps["gsis"] = ps["pfr_player_id"].map(xw)
    by_name = ps["player"].map(_normalize_name).map(uniq)
    n_id, n_name = ps["gsis"].notna().sum(), (ps["gsis"].isna() & by_name.notna()).sum()
    ps["gsis"] = ps["gsis"].fillna(by_name)

    m = ps.dropna(subset=["gsis"]).merge(vet, left_on="gsis", right_on="gsis_id")
    m = m[(m["year_signed"] <= m["season"]) & (m["season"] < m["year_signed"] + m["years"])]
    m = m.sort_values("year_signed").drop_duplicates(["gsis", "season"], keep="last")
    logging.info("pay curves: %d player-seasons; linked %d by id + %d by name of %d; "
                 "%d with a veteran contract in force", len(ps), n_id, n_name, len(ps), len(m))
    curves, rows = {}, []
    for g, d in m.groupby("position_group"):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0)
        curves[g] = iso.fit(d["share"], d["apy_cap_pct"])
        rows.append({"group": g, "n": len(d), **{f"f({q})": round(float(iso.predict([q])[0]), 4)
                                                 for q in (0.05, 0.25, 0.5, 0.75, 0.95)}})
    print("\npay curves, APY % of cap at season snap share:")
    print(pd.DataFrame(rows).set_index("group").to_string())
    return curves


def _score_fold(E, train_end, val_start, val_end, weight="none", half_life=3.0):
    tr = E[E["last_college_season_year"] <= train_end].copy()
    va = E[E["last_college_season_year"].between(val_start, val_end)].copy()
    # weights relative to the board-era training classes (tr starts at 2013 seasons)
    tr["w"] = _class_weights(tr["cls"], weight, half_life)
    if weight != "none":
        logging.info("fold %d class weights: %s", val_start,
                     tr.groupby("cls")["w"].first().round(3).to_dict())
    bt = tr[tr["board"]]
    board_cal = ShrunkIso(increasing=False, k=K_BOARD).fit(
        np.log(bt["hasan_rank"]), bt["target"], bt["pos_group"], sw=bt["w"])
    vb = va["board"]
    board_pred = pd.Series(np.nan, index=va.index)
    board_pred[vb] = board_cal.predict(np.log(va.loc[vb, "hasan_rank"]), va.loc[vb, "pos_group"])
    logging.info("fold %d: board calibration on %d board rows (classes %d-%d), eval %d rows",
                 val_start, len(bt), bt["cls"].min(), bt["cls"].max(), len(va))

    ob = tr[~tr["board"]]
    n_nfl = ob.groupby("pos_group")["nfl"].sum().to_dict()
    for s, tag in [("dep_impact", "dep"), ("pre", "pre"), ("pre_43", "pre43")]:
        cal = ShrunkIso(increasing=True, k=K_OFF).fit(
            ob[f"{s}_pct_ob"], ob["target"], ob["pos_group"], weight_n=n_nfl, sw=ob["w"])
        off = pd.Series(cal.predict(va[f"{s}_pct_ob"].fillna(0), va["pos_group"]), index=va.index)
        va[f"merged_{tag}"] = np.where(vb, board_pred, off)
        top = ob[ob[f"{s}_pct_ob"] >= CAP_Q]
        pooled = _wmean(top)
        cap = {g: (len(t) * _wmean(t) + K_OFF * pooled) / (len(t) + K_OFF)
               for g, t in top.groupby("pos_group")}
        capv = va["pos_group"].map(cap).fillna(pooled)
        va[f"merged_{tag}_cap"] = np.where(vb, board_pred, np.minimum(off, capv))
        logging.info("fold %d %s cap: %s", val_start, tag,
                     {g: round(v, 3) for g, v in cap.items()})
        va[f"{tag}_offcal"] = np.where(vb, np.nan, off)
    for s, tag in [("dep_impact", "dep_cal"), ("pre", "pre_cal")]:
        cal = ShrunkIso(increasing=True, k=K_BOARD).fit(
            tr[f"{s}_pct_all"], tr["target"], tr["pos_group"],
            weight_n=tr.groupby("pos_group")["nfl"].sum().to_dict(), sw=tr["w"])
        va[tag] = cal.predict(va[f"{s}_pct_all"], va["pos_group"])
    va["board_only"] = np.where(vb, board_pred, -1.0)
    va["board_cal"] = board_pred
    if "board_sub" in E.columns:
        parent_tr = pd.Series(board_cal.predict(np.log(bt["hasan_rank"]), bt["pos_group"]),
                              index=bt.index)
        off_cap = va["merged_dep_cap"].where(~vb)
        for col, arm in [("board_sub", "merged_dep_cap_sub"),
                         ("board_sub_perm", "merged_dep_cap_subperm")]:
            sub_pred = board_pred.copy()
            for key, t in bt.groupby(col):
                iso = IsotonicRegression(increasing=False, out_of_bounds="clip",
                                         y_min=0.0, y_max=1.0)
                iso.fit(np.log(t["hasan_rank"]), t["target"], sample_weight=t["w"])
                w = len(t) / (len(t) + K_BOARD)
                m = vb & (va[col] == key)
                if m.any():
                    sub_pred[m] = (w * iso.predict(np.log(va.loc[m, "hasan_rank"]))
                                   + (1 - w) * board_pred[m])
                if col == "board_sub":
                    logging.info("fold %d sub %s: n %d, w %.2f, E[usage] at rank 10/50/150: %s",
                                 val_start, key, len(t), w,
                                 [round(float(w * iso.predict([np.log(r)])[0]
                                              + (1 - w) * board_cal.predict([np.log(r)], [key.split(":")[0]])[0]), 3)
                                  for r in (10, 50, 150)])
            va[arm] = np.where(vb, sub_pred, off_cap)
    return va


def _topk(g, a, k, y="usage"):
    return g.nlargest(min(k, len(g)), a)[y].mean()


def _metrics(V, arms, y="usage"):
    res = {}
    by_cls = list(V.groupby("cls"))
    seam_pool = [(c, g[(~g["board"]) | (g["hasan_rank"] > 150)]) for c, g in by_cls]
    for a in arms:
        r = {}
        r["topM"] = np.mean([_topk(g, a, int(g["board"].sum()), y) for _, g in by_cls])
        for k in (100, 200, 500):
            r[f"top{k}"] = np.mean([_topk(g, a, k, y) for _, g in by_cls])
        r["seam150"] = np.mean([_topk(g, a, 150, y) for _, g in seam_pool])
        r["auc_contrib"] = np.mean([roc_auc_score((g["usage"] >= CONTRIB).astype(int), g[a])
                                    for _, g in by_cls])
        bs = V[V["board"] & V["pos_group"].isin(SPLIT_GROUPS)]
        r["board_rho_split"] = np.nanmean([spearmanr(g[a], g[y])[0]
                                           for _, g in bs.groupby(["pos_group", "cls"])
                                           if len(g) >= 10])
        res[a] = r
    return res


def _bootstrap(V, arms, pairs, n_boot, y="usage", seed=0):
    rng = np.random.default_rng(seed)
    V = V.reset_index(drop=True)
    groups = [g.index.to_numpy() for _, g in V.groupby("cls")]
    point = _metrics(V, arms, y)
    boots = []
    for i in range(n_boot):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        boots.append(_metrics(V.iloc[ix], arms, y))
        if (i + 1) % 100 == 0:
            logging.info("bootstrap %s %d/%d", y, i + 1, n_boot)
    lines = []
    for a, b in pairs:
        for k in point[a]:
            d = np.array([bt[a][k] - bt[b][k] for bt in boots])
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            lines.append({"Δ": f"{a} − {b}", "metric": k,
                          "delta": round(point[a][k] - point[b][k], 4),
                          "ci": f"[{lo:+.4f}, {hi:+.4f}]", "lo": lo, "hi": hi})
    return point, pd.DataFrame(lines)


def _diagnostics(V):
    rows = []
    for a in ARMS[:-1]:
        top = V.groupby("cls", group_keys=False).apply(
            lambda g: g.nlargest(int(g["board"].sum()), a))
        ob = top[~top["board"]]
        rows.append({"arm": a, "offboard_in_topM_per_class": len(ob) / V["cls"].nunique(),
                     "their_target": ob["target"].mean(), "their_pred": ob[a].mean(),
                     "board_dropped_target": V[V["board"] & ~V.index.isin(top.index)]["target"].mean()})
    print("\noff-board players entering the top M, and the board players they displace:")
    print(pd.DataFrame(rows).set_index("arm").round(3).to_string())

    print("\noff-board players in the top 100 / 200 per class (count, actual, predicted):")
    for a in ARMS[:-1]:
        parts = []
        for k in (100, 200):
            t = V.groupby("cls", group_keys=False)[V.columns.tolist()].apply(lambda g: g.nlargest(k, a))
            o = t[~t["board"]]
            parts.append(f"top{k}: {len(o) / V['cls'].nunique():5.1f} {o['target'].mean():.3f} {o[a].mean():.3f}")
        print(f"  {a:17s} " + "   ".join(parts))

    print("\ncalibration (mean predicted vs actual calibration target), by population and predicted decile:")
    for a in ("merged_dep", "merged_dep_cap", "dep_cal"):
        for pop, m in (("board", V["board"]), ("off-board", ~V["board"])):
            d = V[m]
            q = pd.qcut(d[a].rank(method="first"), 10, labels=False)
            t = d.groupby(q).agg(pred=(a, "mean"), actual=("target", "mean")).round(3)
            print(f"  {a:10s} {pop:9s} top 3 deciles pred/actual: "
                  + "  ".join(f"{p:.3f}/{r:.3f}" for p, r in t.tail(3).values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--offboard-oof", type=Path)
    ap.add_argument("--cal-target", choices=["usage", "value", "dollars"], default="usage")
    ap.add_argument("--compare-with", type=Path)
    ap.add_argument("--compare-label", default="usage_built_cap")
    ap.add_argument("--weight", choices=["none", "recent", "reverse"], default="none")
    ap.add_argument("--half-life", type=float, default=3.0)
    ap.add_argument("--board-sub", action="store_true")
    ap.add_argument("--pay-share", choices=["appeared", "season"], default="appeared")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step8 = importlib.import_module("08_draft_models")
    zd = importlib.import_module("00zd_offboard_model")
    E, folds = _load(args, zd, step8)
    lab = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    lab = lab[lab["player_id"].notna()].assign(player_id=lambda d: zd._pid(d["player_id"]))
    apy = lab.drop_duplicates("player_id").set_index("player_id")["best_apy_by_y5"]
    E["apy"] = np.where(E["nfl"], E["player_id"].map(apy).fillna(0.0), 0.0)
    if args.cal_target == "dollars":
        curves = _pay_curves(args.out_dir, args.pay_share)
        f = np.array([curves[g].predict([u])[0] if g in curves else np.nan
                      for g, u in zip(E["pos_group"], E["usage"])])
        E["dollars"] = np.where(E["usage"] > 0, f, 0.0)
    E["target"] = E[args.cal_target]
    logging.info("calibration target: %s", args.cal_target)
    logging.info("class weighting: %s, half-life %.1f", args.weight, args.half_life)
    if args.board_sub:
        E["board_sub"] = _board_subpositions(E)
        E["board_sub_perm"] = _permute_within(E, "board_sub")
        logging.info("board sub-positions: %s",
                     E.loc[E["board"], "board_sub"].value_counts(dropna=False).to_dict())
    V = pd.concat([_score_fold(E, *f, weight=args.weight, half_life=args.half_life)
                   for f in folds if f[1] in EVAL_FOLDS], ignore_index=True)
    V = V.drop_duplicates("player_id", keep="last")
    V.to_parquet(args.out_dir / "merged_oof.parquet", index=False)
    logging.info("eval %d players, classes %d-%d, board %d, off-board NFL %d",
                 len(V), V["cls"].min(), V["cls"].max(), int(V["board"].sum()),
                 int((~V["board"] & V["nfl"]).sum()))

    arms, pairs = list(ARMS), list(PAIRS)
    if args.board_sub:
        arms[-1:-1] = ["merged_dep_cap_sub", "merged_dep_cap_subperm"]
        pairs[0:0] = [("merged_dep_cap_sub", "merged_dep_cap"),
                      ("merged_dep_cap_subperm", "merged_dep_cap")]
    if args.compare_with:
        o = pd.read_parquet(args.compare_with)[["player_id", "merged_dep_cap"]]
        lab = args.compare_label
        V = V.merge(o.rename(columns={"merged_dep_cap": lab}), on="player_id", how="left")
        assert V[lab].notna().all(), "compare-with rows don't match"
        arms.insert(-1, lab)
        pairs.insert(0, ("merged_dep_cap", lab))
    outcomes = ["usage", "value", "apy"] + (["dollars"] if args.cal_target == "dollars" else [])
    decide_on = "apy" if args.cal_target == "dollars" else args.cal_target
    for y in outcomes:
        point, deltas = _bootstrap(V, arms, pairs, args.n_boot, y)
        print(f"\n=== outcome: {y} ===")
        print(pd.DataFrame(point).T.round(4).to_string())
        print(deltas.drop(columns=["lo", "hi"]).to_string(index=False))
        deltas.to_csv(args.out_dir / f"deltas_{y}.csv", index=False)
        if args.board_sub and y == "usage":
            q = deltas[(deltas["Δ"] == "merged_dep_cap_sub − merged_dep_cap") & (deltas["metric"] == "topM")].iloc[0]
            pq = deltas[(deltas["Δ"] == "merged_dep_cap_subperm − merged_dep_cap") & (deltas["metric"] == "topM")].iloc[0]
            oks = q["lo"] > 0 and pq["lo"] <= 0 <= pq["hi"]
            print(f"\nPRE-REGISTERED (board sub-positions): merged_dep_cap_sub − merged_dep_cap topM "
                  f"{q['delta']:+.4f} {q['ci']}; permuted placebo {pq['delta']:+.4f} {pq['ci']} "
                  f"→ {'PASS' if oks else 'FAIL'}")
        if y == decide_on:
            p = deltas[(deltas["Δ"] == "merged_dep_cap − dep_cal") & (deltas["metric"] == "topM")].iloc[0]
            pl = deltas[(deltas["Δ"] == "merged_pre43_cap − merged_pre_cap") & (deltas["metric"] == "topM")].iloc[0]
            ok = p["lo"] > 0 and pl["lo"] <= 0 <= pl["hi"]
            if args.compare_with and (args.weight != "none" or args.cal_target == "dollars"):
                q = deltas[(deltas["Δ"] == f"merged_dep_cap − {args.compare_label}")
                           & (deltas["metric"] == "topM")].iloc[0]
                okw = q["lo"] > 0 and pl["lo"] <= 0 <= pl["hi"]
                print(f"\nPRE-REGISTERED ({args.cal_target}-built, weight {args.weight}, scored on {y}): "
                      f"merged_dep_cap − {args.compare_label} topM {q['delta']:+.4f} {q['ci']}; "
                      f"placebo {pl['delta']:+.4f} {pl['ci']} → {'PASS' if okw else 'FAIL'}")
            print(f"\nPRE-REGISTERED (cap, built on {args.cal_target}, scored on {y}): merged_dep_cap − dep_cal topM {p['delta']:+.4f} {p['ci']}; "
                  f"placebo {pl['delta']:+.4f} {pl['ci']} → {'PASS' if ok else 'FAIL'}")
    _diagnostics(V)
    print("\nposition mix of each list's top 100 (players per class):")
    mix = {}
    for a in [x for x in arms if x != "board_only"]:
        t = V.groupby("cls", group_keys=False)[V.columns.tolist()].apply(lambda g: g.nlargest(100, a))
        mix[a] = (t["pos_group"].value_counts() / V["cls"].nunique()).round(1)
    print(pd.DataFrame(mix).fillna(0).to_string())


if __name__ == "__main__":
    main()
