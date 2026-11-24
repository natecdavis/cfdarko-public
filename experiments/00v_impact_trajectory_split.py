"""
Impact trajectory model: legacy vs by-class validation split, backtested.

The impact trajectory model imputes impact_target for un-settled classes
(draft >= TRAJECTORY_ELIGIBILITY_CUTOFF_YEAR) that have no contract yet, from
their first 1-4 NFL seasons. Its legacy validation split infers draft year from
n_years_observed, so "validation" is every 1-2 season row: the model never
trains on short snapshots, yet the 2024-25 classes are exactly that.

Step-8 OOF folds never contain these labels (they only reach the final fit and
the holdout fold), so 00u cannot see this change. Judge the trajectory model
on its own job instead: fit on draft classes <= TRAIN_MAX, predict the settled
TEST classes from their first k seasons, and score each snapshot length k
against
    impact   the settled second-contract APY label (what it imputes)
    value    value_target (the independent referee)
Spearman, pooled and mean-within-position-group. Both splits run through
production's own fit_trajectory_models (impact only), under several LightGBM
seeds: the seed spread is the noise floor.

Pre-registered bar: by_class beats legacy on k=1 and k=2 under BOTH scorers by
more than twice the larger seed sd, and does not lose on k=3/4 by more than
that.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, RAW_DIR, LGBM_PARAMS
import src.scrapers.nfl_data as nd

TRAIN_MAX, TEST = 2018, (2019, 2020)


def _dataset() -> pd.DataFrame:
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    contracts = nd.fetch_contract_data(bridge, contracts=nd.fetch_league_contracts())
    contract_label = nd.compute_second_contract_label(contracts)
    snap = pd.read_parquet(RAW_DIR / "snap_counts.parquet")
    feats = nd.fetch_nfl_seasonal_features(bridge, snap, max_years=nd.TRAJECTORY_MAX_YEARS)
    lab = pd.read_parquet(PROCESSED_DIR / "nfl_draft_labels.parquet")
    lab = lab[lab["is_bridge"] == True].dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    ds = nd.build_trajectory_dataset(bridge, feats, contract_label,
                                     value_label=lab.set_index("pfr_id")["value_target"])
    ds["position_group_ordinal"] = ds["position_group_ordinal"].astype(int)
    print(f"dataset: {len(ds)} snapshot rows, classes {ds.draft_year.min()}-{ds.draft_year.max()}")
    return ds


def _fit(train: pd.DataFrame, split: str, seed: int, tmp: Path):
    nd.TRAJECTORY_MODELS_DIR = tmp
    nd.LGBM_PARAMS = {**LGBM_PARAMS, "random_state": seed}
    _, model, _, cols = nd.fit_trajectory_models(train, impact_split=split, only="impact",
                                                 variant=f"bt_{split}_{seed}")
    return model, cols


def _score(test: pd.DataFrame, pred: np.ndarray) -> dict:
    out = {}
    t = test.assign(pred=pred)
    for k in sorted(t["n_years_observed"].astype(int).unique()):
        tk = t[t["n_years_observed"] == k]
        for name, col in [("impact", "target_impact"), ("value", "target_value")]:
            sub = tk.dropna(subset=[col])
            within = [spearmanr(g["pred"], g[col])[0]
                      for _, g in sub.groupby("position_group_ordinal") if len(g) >= 20]
            out[(k, name, "pooled")] = spearmanr(sub["pred"], sub[col])[0]
            out[(k, name, "within")] = float(np.nanmean(within))
        out[(k, "n", "")] = len(tk)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-max", type=int, default=TRAIN_MAX)
    ap.add_argument("--test", type=int, nargs=2, default=TEST, metavar=("FIRST", "LAST"))
    args = ap.parse_args()
    ds = _dataset()
    train = ds[ds["draft_year"] <= args.train_max]
    test = ds[ds["draft_year"].between(*args.test)]
    print(f"train {len(train)} rows (classes <= {args.train_max}), test {len(test)} rows {tuple(args.test)}")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for split in ["legacy", "by_class"]:
            for seed in range(42, 42 + args.seeds):
                model, cols = _fit(train, split, seed, Path(tmp))
                pred = np.clip(model.predict(test[cols].to_numpy(dtype=float, na_value=np.nan)), 0, None)
                r = _score(test, pred)
                r.update({("split", "", ""): split, ("seed", "", ""): seed})
                rows.append(r)
    res = pd.DataFrame(rows)
    res.columns = pd.MultiIndex.from_tuples(res.columns)
    metrics = [c for c in res.columns if c[1] in ("impact", "value")]
    mean = res.groupby(("split", "", ""))[metrics].mean().T
    sd = res.groupby(("split", "", ""))[metrics].std().T
    table = pd.DataFrame({
        "legacy": mean["legacy"].round(3),
        "by_class": mean["by_class"].round(3),
        "Δ": (mean["by_class"] - mean["legacy"]).round(3),
        "seed_sd_max": np.maximum(sd["legacy"], sd["by_class"]).round(3),
    })
    table["Δ/sd"] = (table["Δ"] / table["seed_sd_max"].replace(0, np.nan)).round(1)
    table.index = [f"k={k} {name:6s} {how}" for k, name, how in table.index]
    print(f"\nrows per k: { {k: int(res[(k, 'n', '')].iloc[0]) for k in sorted({c[0] for c in metrics})} }")
    print(table.to_string())


if __name__ == "__main__":
    main()
