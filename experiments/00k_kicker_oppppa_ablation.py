"""
Step 00k: Kicker Ablation — drop opp_rolling_* from K contextual models

00j flagged the K (kicker) contextual heads (xp_att, xp_made, kicking_pts,
fg_att, fg_made, fg_pct) as showing residual leakage on `opp_rolling_def_ppa`
(partial R² 1.4–1.7%). The mechanism for opponent defensive PPA affecting a
kicker's success isn't clean — at best it's a proxy for game-state / field-
position confounding, at worst it's noise that the model is over-fitting.

This ablation refits the K heads with `opp_rolling_*` excluded and compares
cv_r2 vs the current full-feature fits. Decision rule:
  • cv_r2 holds (within ~0.005) ⇒ opp_rolling_* was just confounding noise;
    safe to drop in production via CONTEXT_FEATURE_DROP_BY_POSITION.
  • cv_r2 drops materially          ⇒ those features were doing real work;
    leave them, accept the residual leakage.

Uses cached α from data/processed/contextual_models/model_summary.json for
both arms. Caveat: the cached α was tuned WITH opp_rolling_* in the pool,
so the treatment arm runs with possibly-stale α. If the result is borderline,
a follow-up with --retune-only 4 (K only) is the next step.

Output: scratch model pkls under experiments/_ablation_k_* and a printed
side-by-side cv_r2 comparison. Production model_summary.json untouched.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BRIDGE_SEASON_GAMMA, END_YEAR, CONTEXT_FEATURES, PROCESSED_DIR, RAW_DIR,
)
from src.models.contextual import fit_contextual_models_for_group

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

POS_GROUP = "K"
DROPPED_PREFIX = "opp_rolling_"
SCRATCH_CONTROL = Path(__file__).parent / "_ablation_k_control"
SCRATCH_TREATMENT = Path(__file__).parent / "_ablation_k_treatment"


def _load_bridge_data_for_k() -> pd.DataFrame:
    """Mirror pipeline/04_contextual_models.py:88-178 — load bridge gamelogs,
    join position_group from attrs/bridge, filter to K, attach bridge_weight."""
    gamelogs_path = RAW_DIR / "player_gamelogs.parquet"
    seasons_path = RAW_DIR / "player_seasons.parquet"
    bridge_path = PROCESSED_DIR / "bridge_players.parquet"
    attrs_path = PROCESSED_DIR / "player_attributes.parquet"

    if gamelogs_path.exists():
        player_data = pd.read_parquet(gamelogs_path)
        logger.info("Loaded gamelogs: %d rows", len(player_data))
    elif seasons_path.exists():
        player_data = pd.read_parquet(seasons_path)
        logger.info("Loaded seasons (gamelogs missing): %d rows", len(player_data))
    else:
        raise FileNotFoundError("Neither player_gamelogs nor player_seasons found.")

    bridge_players = pd.read_parquet(bridge_path)

    _PG_BASE = "__pg_base__"
    if "position_group" in player_data.columns:
        player_data = player_data.rename(columns={"position_group": _PG_BASE})
    else:
        player_data[_PG_BASE] = pd.NA

    if attrs_path.exists() and "player_id" in player_data.columns:
        attrs = pd.read_parquet(attrs_path, columns=["player_id", "position_group"])
        player_data = player_data.merge(attrs, on="player_id", how="left")
    else:
        player_data["position_group"] = pd.NA

    player_data["position_group"] = player_data["position_group"].fillna(player_data[_PG_BASE])
    player_data = player_data.drop(columns=[_PG_BASE], errors="ignore")

    if "player_id" in bridge_players.columns and "position_group" in bridge_players.columns:
        bridge_pg = (bridge_players[["player_id", "position_group"]]
                     .dropna(subset=["position_group"])
                     .drop_duplicates("player_id"))
        player_data = player_data.merge(
            bridge_pg.rename(columns={"position_group": "position_group_bridge"}),
            on="player_id", how="left",
        )
        mask = player_data["position_group"].isna()
        player_data.loc[mask, "position_group"] = player_data.loc[mask, "position_group_bridge"]
        player_data = player_data.drop(columns=["position_group_bridge"], errors="ignore")

    if "player_id" in bridge_players.columns and bridge_players["player_id"].notna().any():
        bridge_ids = set(bridge_players["player_id"].dropna().astype(str))
        bridge_data = player_data[player_data["player_id"].astype(str).isin(bridge_ids)].copy()
    else:
        bridge_data = player_data.copy()

    if "conference_tier" not in bridge_data.columns and "conference" not in bridge_data.columns:
        if attrs_path.exists() and "player_id" in bridge_data.columns:
            _attrs_tier = pd.read_parquet(attrs_path, columns=["player_id", "conference_tier"])
            bridge_data = bridge_data.merge(_attrs_tier, on="player_id", how="left")

    bridge_data["bridge_weight"] = BRIDGE_SEASON_GAMMA ** (
        END_YEAR - bridge_data["season_year"].fillna(END_YEAR)
    )

    k_data = bridge_data[bridge_data["position_group"] == POS_GROUP].copy()
    logger.info("Bridge K games: %d rows", len(k_data))
    return k_data


def _load_cached_params_for_k() -> dict[str, dict]:
    """Pull current K best_alpha / l1 / AL flag from model_summary.json so
    both arms run with the same fit-time hyperparameters."""
    summary_path = PROCESSED_DIR / "contextual_models" / "model_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    out: dict[str, dict] = {}
    for row in summary:
        if row.get("position_group") != POS_GROUP:
            continue
        out[row["stat"]] = {
            "best_alpha":         float(row["best_alpha"]) if row.get("best_alpha") is not None else None,
            "best_l1_ratio":      row.get("best_l1_ratio"),
            "is_adaptive_lasso":  bool(row.get("is_adaptive_lasso", False)),
        }
    return out


def main():
    SCRATCH_CONTROL.mkdir(parents=True, exist_ok=True)
    SCRATCH_TREATMENT.mkdir(parents=True, exist_ok=True)

    bridge_k = _load_bridge_data_for_k()
    cached = _load_cached_params_for_k()
    logger.info("Cached params for %d K stats: %s", len(cached), sorted(cached))

    from config import MODELED_STATS
    k_stats = MODELED_STATS[POS_GROUP]
    logger.info("K stats to fit: %s", k_stats)

    treatment_pool = [f for f in CONTEXT_FEATURES if not f.startswith(DROPPED_PREFIX)]
    n_dropped = len(CONTEXT_FEATURES) - len(treatment_pool)
    logger.info("Treatment drops %d features (prefix %r): %s",
                n_dropped, DROPPED_PREFIX,
                [f for f in CONTEXT_FEATURES if f.startswith(DROPPED_PREFIX)])

    logger.info("=" * 60)
    logger.info("CONTROL: full CONTEXT_FEATURES (%d features)", len(CONTEXT_FEATURES))
    logger.info("=" * 60)
    control_models = fit_contextual_models_for_group(
        bridge_seasons=bridge_k,
        team_ratings=None,
        stats=k_stats,
        position_group=POS_GROUP,
        save_dir=SCRATCH_CONTROL,
        weight_col="bridge_weight",
        n_jobs=-1,
        cached_params=cached,
        feature_pool=None,  # defaults to CONTEXT_FEATURES
    )

    logger.info("=" * 60)
    logger.info("TREATMENT: opp_rolling_* dropped (%d features)", len(treatment_pool))
    logger.info("=" * 60)
    treatment_models = fit_contextual_models_for_group(
        bridge_seasons=bridge_k,
        team_ratings=None,
        stats=k_stats,
        position_group=POS_GROUP,
        save_dir=SCRATCH_TREATMENT,
        weight_col="bridge_weight",
        n_jobs=-1,
        cached_params=cached,
        feature_pool=treatment_pool,
    )

    print("\n" + "=" * 84)
    print(f"K (kicker) ablation: drop {DROPPED_PREFIX}* features (N={n_dropped})")
    print("=" * 84)
    fmt = "{:<14}  {:>10}  {:>10}  {:>+10.4f}  {:>8}  {:>8}  {:>10}"
    print("{:<14}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}  {:>10}".format(
        "stat", "cv_r2 ctrl", "cv_r2 treat", "Δ cv_r2", "n_feat C", "n_feat T", "verdict"))
    print("-" * 84)

    rows = []
    for stat in k_stats:
        c = control_models.get(stat)
        t = treatment_models.get(stat)
        if c is None or t is None:
            print(f"{stat:<14}  fit failed for one arm — skipped")
            continue
        c_r2 = c.cv_r2 if c.cv_r2 is not None else float("nan")
        t_r2 = t.cv_r2 if t.cv_r2 is not None else float("nan")
        delta = t_r2 - c_r2
        if delta >= -0.005:
            verdict = "DROP_OK"
        elif delta >= -0.02:
            verdict = "borderline"
        else:
            verdict = "KEEP"
        print(fmt.format(stat, f"{c_r2:+.4f}", f"{t_r2:+.4f}", delta,
                         len(c.feature_cols), len(t.feature_cols), verdict))
        rows.append({"stat": stat, "cv_r2_control": c_r2, "cv_r2_treatment": t_r2,
                     "delta": delta, "verdict": verdict,
                     "n_features_control": len(c.feature_cols),
                     "n_features_treatment": len(t.feature_cols)})

    out_table = pd.DataFrame(rows)
    out_path = Path(__file__).parent / "00k_kicker_oppppa_ablation.csv"
    out_table.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print("\nVerdict legend:")
    print("  DROP_OK     — cv_r2 held within 0.005 of control; safe to drop")
    print("  borderline  — cv_r2 dropped 0.005–0.02; consider follow-up retune before deciding")
    print("  KEEP        — cv_r2 dropped >0.02; opp_rolling_* was doing real work")
    print("\nCaveat: both arms ran with cached α tuned with the FULL feature pool.")
    print("If treatment shows borderline/KEEP, follow up with --retune-only 4 (K only)")
    print("to give the smaller pool a fair α search before treating it as definitive.")


if __name__ == "__main__":
    main()
