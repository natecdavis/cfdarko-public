"""
Do pre-draft team visits add NFL-outcome signal beyond the model?

RESULT (2026-10-30): NO USABLE SIGNAL — visits are a stand-in for the
consensus board / draft capital. Not worth a feature.
  - Controlling for the impact head's OOF prediction, visits pass the
    pre-registered bar on every referee (value +0.16, late_snap +0.18,
    late_apy +0.10; signs 7-9/9; no-COVID run +0.18).
  - But controlling for hasan_rank they mostly fail (+0.04, p 0.04-0.10), and
    controlling for the actual draft pick nothing is left (+0.01 to +0.02).
  - By CV fold the visits signal fades as the model sees more board data:
    +0.246 / +0.154 / +0.069 with 0 / 3 / 6 board seasons in training.
  - The bigger finding: hasan_rank itself adds +0.24 to +0.25 beyond the
    model's prediction in EVERY fold, regardless of board seasons in training
    (and +0.27 / +0.38 / +0.29 on value / late_snap / late_apy, 9/9). The
    model leaves most of the board's information unused.
  Caveat: the permutation null is stratified on OOF-prediction deciles, so
  it is not properly centered for the hasan / pick controls; read those rows
  qualitatively.

Data: kentweyrauch.com's NFL draft visits dashboard (scraped 2026-10-29, one
table per team × draft year, 2015-2026). Per player-team: PRI, COM, PRO, WOR,
Other, Total meeting counts and "drafted" (whether THAT team picked him —
post-draft information, never used here).

Question: among NFL (bridge) players, does visit volume correlate with later
NFL value once the impact head's own out-of-fold prediction is controlled for?

    statistic  within-position-group partial Spearman of the visit feature
               vs a referee, both rank-residualized on the OOF prediction;
               averaged over groups
    referees   value (value_target, settled classes), late_snap, late_apy
               — the same yardstick as experiments/00u
    null       visit feature permuted within (draft year × group × OOF
               decile), N_PERM times

Visit features are within-draft-year percentiles (reporting coverage grows
over time): total meetings, number of teams met, PRI meetings. Players with
no recorded visit are 0 — "not reported" and "did not visit" are confounded,
which the decile-stratified permutation partly controls for (coverage favors
high-profile players, who sit high in the OOF ranking).

Pre-registered bar (scope, 2026-10-29), on the value referee: the pooled
statistic lies outside the middle 95% of the permutation null AND has the
same sign in >= 6 of the 9 position groups. late_* and a no-COVID run are
checks: --no-covid drops 2020 AND 2021 (in-person visits restricted; the
scrape has 323 and 103 players those years vs ~400-480 normally — chosen from
coverage before any outcome was looked at). A pass means a proper step-8 A/B,
not a ship.

Usage:
    00w_draft_visits_check.py --build RAW_DIR   # scraped JSON → data/raw/draft_visits.parquet
    00w_draft_visits_check.py [--oof PATH] [--no-covid] [--control hasan]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, RAW_DIR
from src.scrapers.nfl_data import _normalize_name

VISITS_PATH = RAW_DIR / "draft_visits.parquet"
COUNT_COLS = ["PRI", "COM", "PRO", "WOR", "Other", "Total"]
FEATURES = ["v_total", "v_teams", "v_pri"]
N_PERM = 1000


def build(raw_dir: Path) -> None:
    rows = []
    files = sorted(f for f in raw_dir.glob("*_*.json") if not f.name.startswith("_"))
    for f in files:
        d = json.loads(f.read_text())
        hdr = d["headers"]
        for r in d["data"]:
            rec = dict(zip(hdr, r))
            name, _, pos = str(rec["name"]).rpartition(", ")
            rows.append({"draft_year": int(d["year"]), "team": d["team"],
                         "player": name or rec["name"], "pos": pos,
                         "round": pd.to_numeric(rec["round"], errors="coerce"),
                         "pick": pd.to_numeric(rec["pick"], errors="coerce"),
                         "drafted_by_team": rec["drafted"] == "Yes",
                         **{c: pd.to_numeric(rec[c], errors="coerce") for c in COUNT_COLS}})
    v = pd.DataFrame(rows)
    v.to_parquet(VISITS_PATH, index=False)
    print(f"{len(files)} team-year tables → {len(v)} player-team rows, "
          f"{v.groupby(['draft_year', 'player']).ngroups} player-years → {VISITS_PATH}")
    print(v.groupby("draft_year").agg(rows=("player", "size"), players=("player", "nunique"),
                                      total=("Total", "sum")).to_string())


def _player_visits(bridge: pd.DataFrame) -> pd.DataFrame:
    v = pd.read_parquet(VISITS_PATH)
    agg = (v.groupby(["draft_year", "player", "pos"])
           .agg(v_total=("Total", "sum"), v_teams=("Total", lambda s: int((s > 0).sum())),
                v_pri=("PRI", "sum"), round=("round", "first"))
           .reset_index())
    agg["k"] = agg["player"].map(_normalize_name)

    b = bridge.copy()
    b["k"] = b["player_name"].map(_normalize_name)
    b["draft_year"] = b["draft_year"].astype(int)
    # unique (name, year) keys on both sides; position breaks bridge-side ties
    bkeys = b.groupby(["k", "draft_year"])["pfr_id"].transform("nunique")
    vkeys = agg.groupby(["k", "draft_year"])["player"].transform("size")
    m = agg[vkeys == 1].merge(b[bkeys == 1][["k", "draft_year", "pfr_id"]],
                              on=["k", "draft_year"], how="left")
    drafted = m["round"].notna()
    print(f"visit players: {len(agg)}; matched to bridge {m['pfr_id'].notna().mean():.1%} "
          f"(drafted: {m.loc[drafted, 'pfr_id'].notna().mean():.1%}, "
          f"undrafted: {m.loc[~drafted, 'pfr_id'].notna().mean():.1%}); "
          f"{int((vkeys > 1).sum())} ambiguous visit names dropped")
    return m.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")[["pfr_id"] + FEATURES]


def _cohort(oof_path: Path, drop_covid: bool) -> pd.DataFrame:
    import importlib
    ref = importlib.import_module("00u_value_target_referee_ab")
    df = ref._cohort()                                   # settled bridge, referees attached
    df = df[df["draft_year"].between(2015, 2022)]
    if drop_covid:
        df = df[~df["draft_year"].isin([2020, 2021])]
    oof = pd.read_parquet(oof_path)
    oof = oof[oof["model_name"] == "impact"].drop_duplicates("player_id", keep="last")
    df["pred"] = df["player_id"].map(oof.set_index("player_id")["pred"])
    df = df.dropna(subset=["pred"])

    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet").dropna(subset=["pfr_id"])
    vis = _player_visits(bridge.drop_duplicates("pfr_id"))
    df = df.merge(vis, on="pfr_id", how="left")
    for f in FEATURES:
        df[f] = df[f].fillna(0.0)
        df[f] = df.groupby("draft_year")[f].rank(pct=True)
    hasan = pd.read_parquet(PROCESSED_DIR / "hasan_ranks.parquet")
    df["hasan_rank"] = df["player_id"].map(hasan.drop_duplicates("player_id")
                                           .set_index("player_id")["hasan_rank"])
    df["pred_decile"] = df.groupby("value_group")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    print(f"cohort: {len(df)} bridge players drafted 2015-2022{' (no 2020-21)' if drop_covid else ''}; "
          f"with any visit: {(df['v_total'] > df.groupby('draft_year')['v_total'].transform('min')).mean():.1%}")
    return df


def _resid(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    rx, rc = rankdata(x), rankdata(c)
    return rx - np.polyval(np.polyfit(rc, rx, 1), rc)


def _partial_by_group(df, feat, ref, control):
    out = {}
    for g, s in df.groupby("value_group"):
        s = s.dropna(subset=[ref, control])
        if len(s) < 30 or s[feat].nunique() < 2:
            continue
        out[g] = spearmanr(_resid(s[feat].values, s[control].values),
                           _resid(s[ref].values, s[control].values))[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, metavar="RAW_DIR")
    ap.add_argument("--oof", type=Path,
                    default=PROCESSED_DIR / "_value_ab_20261029" / "oof_apy.parquet")
    ap.add_argument("--no-covid", action="store_true")
    ap.add_argument("--control", choices=["pred", "hasan", "pick", "pred+pick"], default="pred",
                    help="pick / pred+pick use the ACTUAL draft pick (UDFA = 300) — post-draft, "
                         "a diagnostic of whether visits are just early draft capital, never a feature")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.build:
        build(args.build)
        return

    df = _cohort(args.oof, args.no_covid)
    control = {"pred": "pred", "hasan": "hasan_rank", "pick": "neg_pick",
               "pred+pick": "pred_pick"}[args.control]
    if control in ("neg_pick", "pred_pick"):
        bp = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet").dropna(subset=["pfr_id"])
        pick = bp.drop_duplicates("pfr_id").set_index("pfr_id")["overall_pick"]
        df["neg_pick"] = -df["pfr_id"].map(pick).fillna(300.0)
        # single control = mean of the two ranks (rank-residualizing on one index)
        df["pred_pick"] = (df.groupby("value_group")["pred"].rank(pct=True)
                           + df.groupby("value_group")["neg_pick"].rank(pct=True))
    if control == "hasan_rank":
        df = df.dropna(subset=["hasan_rank"])
        df["hasan_rank"] = -df["hasan_rank"]             # higher = better, like pred
        print(f"hasan control: {len(df)} on-board players")
    rng = np.random.default_rng(args.seed)
    strata = df.groupby(["draft_year", "value_group", "pred_decile"]).indices

    rows = []
    for ref in ["value", "late_snap", "late_apy"]:
        for feat in FEATURES:
            obs = _partial_by_group(df, feat, ref, control)
            pooled = float(np.mean(list(obs.values())))
            null = []
            for _ in range(N_PERM):
                perm = df[feat].to_numpy().copy()
                for ix in strata.values():
                    perm[ix] = rng.permutation(perm[ix])
                null.append(np.mean(list(_partial_by_group(df.assign(**{feat: perm}),
                                                           feat, ref, control).values())))
            lo, hi = np.percentile(null, [2.5, 97.5])
            same_sign = sum(np.sign(v) == np.sign(pooled) for v in obs.values())
            rows.append({"referee": ref, "feature": feat, "partial_rho": round(pooled, 4),
                         "null_2.5": round(lo, 4), "null_97.5": round(hi, 4),
                         "p_two_sided": round(float(np.mean(np.abs(null) >= abs(pooled))), 3),
                         "same_sign": f"{same_sign}/{len(obs)}",
                         "PASS": (pooled < lo or pooled > hi) and same_sign >= 6})
            print(f"  {ref:9s} {feat:8s} rho={pooled:+.4f}  null[{lo:+.4f},{hi:+.4f}]  "
                  f"sign {same_sign}/{len(obs)}  groups {dict((k, round(v, 3)) for k, v in obs.items())}",
                  flush=True)
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
