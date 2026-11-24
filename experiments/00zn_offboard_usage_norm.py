"""
Off-board model (step 8d) with a position-normalized stage-2 target
(pre-registered 2026-11-01).

RESULT (2026-11-01): FAIL; the point estimate goes the WRONG way. Classes
2014-22, 42,178 off-board / 1,173 off-board NFL / 176 contributors; base
reproduces production 8d exactly (max |Δ| 0).
    rho_ob_nfl  norm − base −0.032 [−0.063, +0.001]; placebo 0.014
    secondary   auc_ob −0.001 (ns), auc_ob_pos −0.004 (ns),
                auc_udfa +0.001 (ns), rho_udfa_nfl −0.018 (ns)
Per position, norm gains DL/TE/DB and loses RB/EDGE/LB. 00zl's gain does not
transfer. Differences from 00zl: there the pooled model trained on everyone,
with non-NFL zeros, where position scale is a big share of the variance.
Here stage 2 sees only NFL players, and normalization up-weights the loss
on low-benchmark positions (RB bench ~0.42 → errors ×2.4). That
explanation is a hypothesis, not tested. Production 8d unchanged; the
merged board's currency question is unaffected.

Question. 00zl found that a POOLED model ranks players better within their
position when its usage target is divided by a per-position starter
benchmark: on raw snap share the model spends splits telling positions apart
(OL/QB starters play ~100% of snaps, RB/DL/WR starters rotate). Step 8d's
stage 2 (E[usage_y34 | NFL]) is exactly such a pooled model on raw usage.
Does normalizing its target improve the ranking of off-board players?

The merged board is unaffected in currency: it maps 8d's within-(class ×
position) percentile to RAW usage with its own calibrator, so only the
within-position order of 8d's score can change, and that is what is scored.

Arms — seed bags of 5, the production recipe (src/models/offboard_model.py):
    base      production 8d, seeds 42-46 (must reproduce step 8d's OOF)
    base_alt  same, seeds 142-146 (placebo)
    norm      stage 2 trained on usage / bench, seeds 42-46. bench = 80th
              percentile of usage among the fold's TRAINING NFL rows with
              usage > 0 and last college season <= END_YEAR-5, per
              (pos_group × usage_source) (usage_source: snaps vs depth chart).
              Stage 1 unchanged. Stage-2 predictions clipped at 0 (not 1,
              since normalized usage can exceed 1). score = p_nfl × m2.
Scoring: draft classes 2014-22, K/P excluded, off-board players (00zd
metrics). rho_ob_nfl is Spearman within (position × fold) against RAW usage
among off-board NFL players, so a per-position rescale alone cannot move it.

Multiple testing: the FIFTH test on these classes (00zh, 00zi, 00zj, 00zk).
One primary test; a pass is provisional until the 2023 class's y3-4 usage
can confirm it.

PRE-REGISTERED BAR (95% paired bootstrap, 1000 reps, players resampled
within class):
    PASS if norm − base on rho_ob_nfl has its 95% interval above 0 AND
    exceeds |base_alt − base| on rho_ob_nfl.
Secondary (reported, not tested): auc_ob, auc_ob_pos, auc_udfa, rho_udfa_nfl.
On a PASS: switch 8d's stage-2 target (no board-curve change needed, see
above), after a merge check (00zf capped merge, norm vs base off-board side:
top-M and top-100 usage not significantly negative).

Usage:
    00zn_offboard_usage_norm.py --out-dir DIR [--n-boot 1000]
"""
import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from config import PROCESSED_DIR, END_YEAR  # noqa: E402
from src.models import offboard_model as om  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("00zn")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zh = _load("zh", "experiments/00zh_offboard_visits_pff.py")
zd = zh.zd

BENCH_Q = 0.80
SEEDS = [42, 43, 44, 45, 46]
ARM_SPECS = {"base": (False, SEEDS), "base_alt": (False, [s + 100 for s in SEEDS]),
             "norm": (True, SEEDS)}
PAIRS = [("norm", "base"), ("base_alt", "base"), ("norm", "dep_impact"),
         ("base", "dep_impact")]
EVAL_CLASSES = (2014, 2022)


class NormBag(om.OffboardBag):
    """Production bag; stage 2 fit on the `usage` column the caller supplies
    (normalized), predictions clipped at 0 only."""

    def predict(self, df):
        X = om._X(df, self.cols)
        p1 = np.mean([m.predict_proba(X)[:, 1] for m in self.stage1], axis=0)
        m2 = np.mean([np.clip(m.predict(X), 0, None) for m in self.stage2], axis=0)
        return pd.DataFrame({"p_nfl": p1, "usage_if_nfl": m2, "score": p1 * m2},
                            index=df.index)


def fold_bench(tr: pd.DataFrame) -> pd.Series:
    ok = (tr["nfl"] & (tr["usage"] > 0) & tr["usage_source"].notna()
          & (tr["last_college_season_year"] <= END_YEAR - 5))
    return (tr[ok].drop_duplicates("player_id")
            .groupby(["pos_group", "usage_source"])["usage"].quantile(BENCH_Q))


def normalized(tr2: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    key = pd.MultiIndex.from_frame(tr2[["pos_group", "usage_source"]])
    b = bench.reindex(key).to_numpy()
    u = tr2["usage"].to_numpy()
    out = np.where(u > 0, u / b, 0.0)
    keep = ~((u > 0) & np.isnan(b))
    if (~keep).any():
        log.warning("dropping %d stage-2 rows with no benchmark", int((~keep).sum()))
    return tr2[keep].assign(usage=out[keep])


def run_bags(step8, rows, feats, folds):
    yr = rows["last_college_season_year"].fillna(0)
    fold_of = pd.Series(np.nan, index=rows.index)
    score = {a: pd.Series(np.nan, index=rows.index) for a in ARM_SPECS}
    for train_end, val_start, val_end in folds:
        tr_m, va_m = yr <= train_end, yr.between(val_start, val_end)
        fold_of[va_m] = val_start
        tr, va = rows[tr_m], rows[va_m]
        bench = fold_bench(tr)
        log.info("fold %d bench:\n%s", val_start, bench.unstack().round(3).to_string())
        for arm, (norm, seeds) in ARM_SPECS.items():
            if norm:
                bag = NormBag(seeds).fit(step8, tr, normalized(tr[tr["nfl"]], bench), feats)
            else:
                bag = om.OffboardBag(seeds).fit(step8, tr, tr[tr["nfl"]], feats)
            score[arm][va.index] = bag.predict(va)["score"]
            log.info("fold %d %s done; trees %s", val_start, arm, bag.trees)
    return score, fold_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step8 = importlib.import_module("08_draft_models")
    folds, _ = step8._split_holdout(step8._expanding_window_folds())
    snap = step8.assemble_snapshot(include_derived=True, logger=step8.logger)
    labels = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    rows, feats = om.build_rows(step8, snap, labels, bridge)
    src = (labels.dropna(subset=["player_id"]).assign(player_id=lambda d: om.pid(d["player_id"]))
           .drop_duplicates("player_id").set_index("player_id")["usage_source"])
    rows["usage_source"] = rows["player_id"].map(src)

    score, fold_of = run_bags(step8, rows, feats, folds)
    base = rows[zd.ID_COLS + ["hasan_rank", "nfl", "usage", "drafted"]].assign(val_start=fold_of)
    for a in ARM_SPECS:
        base[a] = score[a]
    base = base.dropna(subset=["val_start", "base"])
    base["val_start"] = base["val_start"].astype(int)
    base.to_parquet(args.out_dir / "oof_all_arms.parquet", index=False)

    prod = pd.read_parquet(PROCESSED_DIR / "offboard_models" / "offboard_oof_predictions.parquet")
    chk = base.merge(prod[["player_id", "pos_group", "val_start", "score"]],
                     on=["player_id", "pos_group", "val_start"])
    log.info("base vs production 8d OOF: %d rows, max |Δ| %.2e", len(chk),
             (chk["base"] - chk["score"]).abs().max())

    E = zd._deployed(base).dropna(subset=["dep_impact"]).drop_duplicates("player_id", keep="last")
    E["cls"] = E["last_college_season_year"].astype(int) + 1
    E = E[E["cls"].between(*EVAL_CLASSES)]
    E["board"] = E["hasan_rank"].notna()
    E["contrib"] = (E["usage"] >= zd.CONTRIB).astype(int)
    E["starter"] = (E["usage"] >= zd.STARTER).astype(int)
    ob = E[~E["board"]]
    log.info("scoring cohort %d: off-board %d, off-board NFL %d, contributors %d",
             len(E), len(ob), int(ob["nfl"].sum()), int(ob["contrib"].sum()))

    arms = list(ARM_SPECS) + ["dep_impact"]
    point, deltas = zh.bootstrap(E, arms, PAIRS, args.n_boot, levels=(95.0,))
    pd.set_option("display.width", 200)
    print("\nheadline metrics:")
    print(pd.DataFrame(point).T.round(4).to_string())
    print("\nΔ, paired bootstrap 95%:")
    print(deltas.round(4).to_string(index=False))
    print("\nper-position rho_ob_nfl (pooled over folds):")
    obn = ob[ob["nfl"]]
    print(pd.DataFrame({a: obn.groupby("pos_group").apply(
        lambda g: g[a].corr(g["usage"], method="spearman")) for a in ARM_SPECS}).round(3).to_string())
    print("\ntop-K vs the bottom of the board:")
    print(zd._topk_table(E, arms).to_string())

    g = lambda a, b: deltas[(deltas["Δ"] == f"{a} − {b}") & (deltas["metric"] == "rho_ob_nfl")].iloc[0]
    d, plac = g("norm", "base"), abs(g("base_alt", "base")["delta"])
    v = {"pass": bool(d["lo95"] > 0 and d["delta"] > plac), "d_rho": round(d["delta"], 4),
         "ci95": f"[{d['lo95']:+.4f}, {d['hi95']:+.4f}]", "placebo": round(plac, 4)}
    print("\nVERDICT:", v)
    deltas.to_csv(args.out_dir / "deltas.csv", index=False)
    pd.Series(v, dtype=object).to_json(args.out_dir / "verdict.json")


if __name__ == "__main__":
    main()
