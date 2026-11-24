"""
RESULT AFTER THE COMBINE-JOIN FIX (2026-10-30, results_post_fix.csv, same
rule, nothing changed): FIVE traits meet it. Sign note: this run predates
the forty-sign fix (00d compared NEGATE_METRICS {"forty"} with the prefixed
"combine_forty", so forty was never negated); in THIS table
combine_forty_z means higher = SLOWER, like shuttle/cone. Since the fix,
combine_forty_z is higher = faster, so a rerun shows its rho with the
opposite sign.
                       discovery   confirm (p1)   beyond pick (p)
    combine_shuttle_z  -0.203      -0.164 (.003)  -0.080 (.039)   agile > board
    combine_cone_z     -0.169      -0.110 (.031)  -0.075 (.052)   agile > board
    combine_forty_z    -0.106      -0.068 (.050)  -0.016 (.576)   fast > board; teams already fix it
    combine_invited    +0.061      +0.045 (.035)  +0.057 (.010)   invitees > board
    skipped_forty      -0.071      -0.089 (.004)  -0.066 (.003)   invited, no forty < board AND pick
Late referees agree in sign for all five. Agility traits cover only players
who ran the drill (~55%); per-class cohorts are too thin to estimate a trend
for them. Broad jump passed discovery (q .011) but not confirmation (.085).
Strictly "unfixed by teams" (beyond-pick p < .05): shuttle, invited,
skipped_forty; cone is borderline; forty is priced by teams.

FIRST RUN (2026-10-30): INVALIDATED for every combine trait by a data leak
it uncovered. Two traits met the rule (combine_invited +0.33 / conf +0.34 /
beyond-pick +0.27; combine_shuttle_z -0.18 / -0.14 / -0.09), but
pipeline/00d_combine_ingestion.py joins combine records to CFBD player_ids
ONLY through bridge_players (pfr_id, then name + draft year), i.e. only
players who were drafted or reached the NFL. Among board players drafted
2014-22, combine_invited = 1 for 93.9% of NFL players and 0.0% of the 340
who never reached it (no combine measurement either); the same 0.0% holds
across the whole training universe. Combine traits here are presence-
leaked (invited, skipped_forty) or tested only on NFL players (all z's).
Among the valid traits none met the rule: team_sp_overall, conference_tier,
age_at_draft passed discovery but failed confirmation; n_games confirmed
but failed discovery. Re-run after the combine join is fixed.

Persistent board failures: player traits the consensus board misprices,
consistently across draft classes, that teams have not corrected for.

Why (2026-10-30): every attempt to beat the consensus board (hasan_rank) on
board players failed or tied (00x, 00y, 00z, and the within-position rank).
The board aggregates what teams know each year and keeps improving, so the
useful question is narrower: does any pre-draft trait predict NFL value
BEYOND the board rank, in the same direction across eras, and also beyond
the actual draft pick (if teams already fix it on draft day, it is not an
open failure)?

Cohort: every board player (hasan_ranks.parquet), model position groups
except K/P, draft 2014-2022 (value_target settled). Players who never reach
the NFL count as value 0: busts are board errors too.

Statistic, per trait: residualize the trait's rank and the outcome's rank on
the control (board rank; or board rank + actual pick) within (position group
× draft class), Spearman of the residuals within position group, averaged
over groups. Null: trait permuted within (group × class), N_PERM times.

PRE-REGISTERED RULE (written before any result was seen). Discovery classes
2014-2018, confirmation classes 2019-2022, primary outcome `value`, control
= board rank. A trait is a PERSISTENT BOARD FAILURE iff
    1. discovery: Benjamini-Hochberg q < 0.10 across all traits
       (two-sided permutation p);
    2. confirmation: same sign, one-sided permutation p < 0.05;
    3. full sample, control = board rank + actual pick: same sign
       (reported with its p; this is the "teams haven't fixed it" check).
Reported for every trait, not decided on: late_snap / late_apy (2016-19
classes only), and the per-class partials (a downward drift would mean
teams or the board are catching up).

Usage:
    00za_board_failures.py [--out PATH]
"""
import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
N_PERM = 1000
DISCOVERY, CONFIRM = (2014, 2018), (2019, 2022)
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
TRAITS = [
    # age / experience
    "age_at_draft", "class_year_int", "n_seasons", "n_games", "n_transfers_career",
    # athletic testing (z = within position)
    "combine_forty_z", "combine_broad_jump_z", "combine_cone_z", "combine_shuttle_z",
    "combine_bench_z", "combine_ht_z", "combine_wt_z", "combine_arm_z",
    "combine_hand_z", "combine_wingspan_z", "combine_invited",
    "skipped_forty",            # invited but no forty: derived below
    # pedigree / context
    "recruiting_composite", "conference_tier", "team_sp_overall",
    # college production quality
    "pff_grades_offense_mean", "pff_grades_defense_mean",
]


def _cohort() -> pd.DataFrame:
    g = importlib.import_module("08b_global_draft_model")
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    df, _ = g.build_global_training_frame(labels)
    board = pd.read_parquet(PROCESSED_DIR / "hasan_ranks.parquet").drop_duplicates("player_id")
    df = df.drop(columns=["draft_year"]).merge(
        board[["player_id", "draft_year"]], on="player_id", how="inner")
    df = df[df["hasan_rank"].notna() & df["position_group"].isin(GROUPS)
            & df["draft_year"].between(DISCOVERY[0], CONFIRM[1])].copy()
    df["skipped_forty"] = np.where(df["combine_invited"] == 1,
                                   df["combine_forty_z"].isna().astype(float), np.nan)

    lab = labels.drop_duplicates("player_id").set_index("player_id")
    df["value"] = df["player_id"].map(lab["value_target"])
    df.loc[~df["is_bridge"].astype(bool), "value"] = 0.0
    ref = importlib.import_module("00u_value_target_referee_ab")
    late = ref._cohort().drop_duplicates("player_id").set_index("player_id")
    in_late = df["draft_year"].between(*ref.LATE_COHORT)
    for r in ["late_snap", "late_apy"]:
        df[r] = df["player_id"].map(late[r])
        df.loc[in_late & ~df["is_bridge"].astype(bool), r] = 0.0
        df.loc[~in_late, r] = np.nan

    bp = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    pick = (bp.dropna(subset=["overall_pick"]).drop_duplicates("player_id")
            .set_index("player_id")["overall_pick"])
    df["neg_pick"] = -df["player_id"].map(pick).fillna(300.0)   # UDFA / undrafted = 300
    df["neg_board"] = -df["hasan_rank"]
    logging.info("cohort: %d board players, %d drafted, value non-null %d",
                 len(df), int((df["neg_pick"] > -300).sum()), int(df["value"].notna().sum()))
    return df.reset_index(drop=True)


def _resid_within(df: pd.DataFrame, col: str, controls: list[str]) -> np.ndarray:
    """Rank-residualize `col` on the controls within (group × class)."""
    out = np.full(len(df), np.nan)
    for _, ix in df.groupby(["position_group", "draft_year"]).indices.items():
        s = df.iloc[ix]
        ok = s[[col] + controls].notna().all(axis=1).to_numpy()
        if ok.sum() < 8:
            continue
        y = rankdata(s[col].to_numpy()[ok])
        X = np.column_stack([np.ones(ok.sum())]
                            + [rankdata(s[c].to_numpy()[ok]) for c in controls])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[ix[ok]] = y - X @ beta
    return out


def _stat(df, trait_res, out_res):
    rhos = []
    for _, ix in df.groupby("position_group").indices.items():
        a, b = trait_res[ix], out_res[ix]
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() >= 30 and np.std(a[ok]) > 0:
            rhos.append(spearmanr(a[ok], b[ok])[0])
    return float(np.mean(rhos)) if rhos else np.nan, len(rhos)


def _test(df, trait, outcome, controls, rng, n_perm=N_PERM):
    """Observed statistic and permutation p-values (trait shuffled within group × class)."""
    sub = df[df[trait].notna() & df[outcome].notna()].reset_index(drop=True)
    if len(sub) < 60:
        return dict(rho=np.nan, n=len(sub), groups=0, p2=np.nan, p_pos=np.nan, p_neg=np.nan)
    out_res = _resid_within(sub, outcome, controls)
    obs, k = _stat(sub, _resid_within(sub, trait, controls), out_res)
    strata = list(sub.groupby(["position_group", "draft_year"]).indices.values())
    null = np.empty(n_perm)
    vals = sub[trait].to_numpy()
    for i in range(n_perm):
        perm = vals.copy()
        for ix in strata:
            perm[ix] = rng.permutation(perm[ix])
        null[i] = _stat(sub, _resid_within(sub.assign(**{trait: perm}), trait, controls), out_res)[0]
    null = null[~np.isnan(null)]
    return dict(rho=obs, n=len(sub), groups=k,
                p2=float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)),
                p_pos=float((np.sum(null >= obs) + 1) / (len(null) + 1)),
                p_neg=float((np.sum(null <= obs) + 1) / (len(null) + 1)))


def _bh(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    m = len(order)
    q = (order * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    return q.reindex(p.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=PROCESSED_DIR / "_board_failures_20261030" / "results.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = _cohort()
    disc = df[df["draft_year"].between(*DISCOVERY)]
    conf = df[df["draft_year"].between(*CONFIRM)]

    rows = []
    for t in TRAITS:
        d = _test(disc, t, "value", ["neg_board"], rng)
        c = _test(conf, t, "value", ["neg_board"], rng)
        f = _test(df, t, "value", ["neg_board"], rng)
        fp = _test(df, t, "value", ["neg_board", "neg_pick"], rng)
        r = {"trait": t, "n": f["n"],
             "disc_rho": d["rho"], "disc_p": d["p2"],
             "conf_rho": c["rho"],
             "conf_p1": c["p_pos"] if d["rho"] > 0 else c["p_neg"],
             "full_rho": f["rho"], "full_p": f["p2"],
             "beyond_pick_rho": fp["rho"], "beyond_pick_p": fp["p2"]}
        for ref in ["late_snap", "late_apy"]:
            lr = _test(df, t, ref, ["neg_board"], rng, n_perm=200)
            r[f"{ref}_rho"], r[f"{ref}_p"] = lr["rho"], lr["p2"]
        by_year = {}
        for y, s in df.groupby("draft_year"):
            sub = s[s[t].notna() & s["value"].notna()].reset_index(drop=True)
            if len(sub) >= 60:
                by_year[int(y)] = _stat(sub, _resid_within(sub, t, ["neg_board"]),
                                        _resid_within(sub, "value", ["neg_board"]))[0]
        r["by_year"] = {k: round(v, 2) for k, v in by_year.items()}
        yrs = [k for k, v in by_year.items() if np.isfinite(v)]
        r["trend_per_yr"] = (np.polyfit(yrs, [by_year[k] for k in yrs], 1)[0]
                             if len(yrs) >= 5 else np.nan)
        rows.append(r)
        logging.info("%-24s disc %+.3f (p %.3f)  conf %+.3f  beyond-pick %+.3f",
                     t, d["rho"], d["p2"], c["rho"], fp["rho"])

    res = pd.DataFrame(rows).set_index("trait")
    res["disc_q"] = _bh(res["disc_p"])
    res["PERSISTENT"] = ((res["disc_q"] < 0.10)
                         & (np.sign(res["conf_rho"]) == np.sign(res["disc_rho"]))
                         & (res["conf_p1"] < 0.05)
                         & (np.sign(res["beyond_pick_rho"]) == np.sign(res["disc_rho"])))
    res.to_csv(args.out)
    pd.set_option("display.width", 250)
    cols = ["n", "disc_rho", "disc_p", "disc_q", "conf_rho", "conf_p1", "full_rho",
            "beyond_pick_rho", "beyond_pick_p", "late_snap_rho", "late_apy_rho",
            "trend_per_yr", "PERSISTENT"]
    print(res[cols].sort_values("disc_p").round(3).to_string())
    print("\nper-class partials (value | board):")
    print(res["by_year"].to_string())
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
