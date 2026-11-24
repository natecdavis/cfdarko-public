"""
Best available players at a draft pick, in rookie-deal value.

At a given pick the rookie wage is set by the slot, not the player, so the
available players are ranked by expected rookie-deal value (merged_value
from step 10: Σ y1-y4 veteran-market price of the projected role at the
position, % of cap). For each player it also shows:

    value      expected rookie-deal value IF TAKEN AT THIS PICK: board players
               still there past their board rank are marked down, x (pick /
               board rank)^-0.25, capped at 3.3x (experiments/01f: sliders
               turn out worse; src/models/draft_slide.py). Taking a player
               before his board rank leaves his value unchanged (our own reach
               is not evidence). Off-board players unchanged. rank = order by this value;
               list_rank = the pre-draft merged list (merged_rank).
    cost       expected rookie-deal cost at this pick (slot curve, y1-y4)
    surplus    value − cost: expected surplus if taken here
    vs_slot    surplus − the slot's average surplus (= value − the slot's
               average value): how much better than a typical pick here
    pos_wait   his value − the expected best player at his position (OL split
               into OT / IOL) still there at your next pick
    two_pick   his value + the expected best player overall at your next pick
               once he is gone; the highest is the best two-pick plan on value
    p_next     P(he is still there at your next pick) if you pass now
               (src/models/availability.py: board players from big-board
               rank and position, 00zx arm C_pos; off-board players from their rank in the
               off-board pool by the 8d model plus combine invite, 00zy)

Inputs: data/output/projections/latest.parquet (step 10) and
data/processed/slot_curve.parquet (step 7.5).

Usage:
    python scripts/board_at_pick.py --pick 40
    python scripts/board_at_pick.py --pick 40 --taken taken.txt --top 30
    python scripts/board_at_pick.py --pick 12 --pos QB WR --class 2025
    python scripts/board_at_pick.py --pick 20 --next 52
    python scripts/board_at_pick.py --pick 40 --class 2025 --actual --sort two_pick
A table of each position's drop-off to your next pick follows the list.
--actual removes the players taken before this pick in that class's real draft.
--next: your next pick (default: pick + 32, the same slot next round).
--taken: a text file with one player per line (player_id or exact name);
those players are removed. Default class = the newest (the upcoming draft),
restricted to its draft pool when step 10 determined one.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, PROCESSED_DIR, SALARY_CAP  # noqa: E402
from src.models.availability import fit_from_frames, fit_offboard, offboard_features  # noqa: E402
from src.models.draft_slide import value_at_pick  # noqa: E402
from src.models.position_dropoff import dropoff, view_position  # noqa: E402


def load_taken(path):
    if path is None:
        return set()
    return {ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()}


def availability(P, pool, pick, nxt):
    """P(on the board at pick nxt | there at pick, we pass) for the whole class
    pool: board players by big-board rank (00zx), off-board players by their
    rank within the off-board pool (00zy). Computed before taken/position
    filters, since off-board ranks are relative to the full pool."""
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    out = pd.Series(np.nan, index=pool.index)
    on = pool["hasan_rank"].notna()
    am = fit_from_frames(P, bridge, by_position=True)
    out[on] = am.p_available(pool.loc[on, "hasan_rank"].to_numpy(), pick, nxt,
                             pool.loc[on, "position_group"].to_numpy())
    off = pool[~on]
    if len(off) and off["offboard_score"].notna().any():
        oof = pd.read_parquet(PROCESSED_DIR / "offboard_models" / "offboard_oof_predictions.parquet")
        X, inv = offboard_features(off)
        out[off.index] = fit_offboard(P, bridge, oof).p_available(X, inv, pick, nxt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pick", type=int, required=True, help="overall pick number (1-262)")
    ap.add_argument("--taken", type=Path, help="file of player_ids / names already drafted")
    ap.add_argument("--next", dest="nxt", type=int, help="your next pick (default: pick + 32)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--actual", action="store_true",
                    help="remove players taken before this pick in the class's real draft (past drafts)")
    ap.add_argument("--sort", choices=["value", "two_pick"], default="value")
    ap.add_argument("--pos", nargs="+", help="only these position groups")
    ap.add_argument("--class", dest="cls", type=int,
                    help="last college season of the class (default: newest)")
    ap.add_argument("--projections", type=Path, default=OUTPUT_DIR / "projections" / "latest.parquet")
    args = ap.parse_args()

    P = pd.read_parquet(args.projections)
    if "merged_value" not in P.columns or P["merged_value"].isna().all():
        sys.exit("projections have no merged_value — rerun steps 7.5 and 10")
    S = pd.read_parquet(PROCESSED_DIR / "slot_curve.parquet")
    S = S[S["pick"] != "UDFA"].assign(pick=lambda d: d["pick"].astype(int)).set_index("pick")
    pick = int(np.clip(args.pick, S.index.min(), S.index.max()))
    cost, slot_surplus, slot_value = (float(S.loc[pick, c]) for c in ("cost", "surplus", "value"))

    cls = args.cls if args.cls is not None else int(P["last_college_season_year"].max())
    d = P[P["last_college_season_year"] == cls].drop_duplicates("player_id").copy()
    if "draft_pool" in d.columns and d["draft_pool"].notna().any():
        d = d[d["draft_pool"].fillna(False).astype(bool)]
    nxt = args.nxt if args.nxt is not None else args.pick + 32
    d["p_next"] = availability(P, d, args.pick, nxt)
    d = d[d["merged_value"].notna()]
    taken = load_taken(args.taken)
    if args.actual:
        br = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
        br = br[(br["draft_year"] == cls + 1) & (br["overall_pick"] < pick)].dropna(subset=["player_id"])
        taken |= set(br["player_id"].astype(str))
        print(f"--actual: {len(br)} players taken before pick {pick} in the real {cls + 1} draft")
    if taken:
        gone = d["player_id"].astype(str).isin(taken) | d["player_name"].isin(taken)
        print(f"removed {int(gone.sum())} of {len(taken)} taken players")
        d = d[~gone]
    d["value"] = value_at_pick(d["merged_value"], d["hasan_rank"], pick)
    # Position drop-off to the next pick (src/models/position_dropoff.py),
    # over everyone still available, before the position filter.
    d["view_pos"] = view_position(d["position_group"], d["position"]) if "position" in d.columns \
        else d["position_group"]
    d["v_now"], d["v_next"] = d["value"], value_at_pick(d["merged_value"], d["hasan_rank"], nxt)
    d, T = dropoff(d.assign(p_next=d["p_next"].fillna(0.0)))
    if args.pos:
        d = d[d["position_group"].isin(args.pos) | d["view_pos"].isin(args.pos)]
    d["surplus"] = d["value"] - cost
    d["vs_slot"] = d["value"] - slot_value
    sort = ["two_pick", "value"] if args.sort == "two_pick" else ["value", "merged_rank"]
    d = d.sort_values(sort, ascending=[False, args.sort != "two_pick"]).head(args.top)
    m = SALARY_CAP / 1e6
    out = pd.DataFrame({
        "rank": np.arange(1, len(d) + 1), "list_rank": d["merged_rank"].astype(int),
        "player": d["player_name"], "school": d.get("school"), "pos": d["view_pos"],
        "board": d["hasan_rank"].map(lambda r: "" if pd.isna(r) else int(r)),
        "value_$M": (d["value"] * m).round(1), "surplus_$M": (d["surplus"] * m).round(1),
        "vs_slot_$M": (d["vs_slot"] * m).round(1),
        f"p_at_{nxt}": d["p_next"].map(lambda p: "" if pd.isna(p) else f"{p:.2f}"),
        "pos_wait_$M": (d["pos_wait_cost"] * m).round(1), "two_pick_$M": (d["two_pick"] * m).round(1),
        "source": d["merged_source"],
    })
    print(f"\nDraft class {cls + 1}, pick {pick}: expected rookie cost ${cost * m:.1f}M over 4 years; "
          f"an average pick here returns ${slot_value * m:.1f}M of value (${slot_surplus * m:.1f}M surplus).")
    print("value = expected y1-4 value at today's veteran prices if taken at THIS pick (board players "
          "who slid past their board rank marked down, 01f); $ at the current cap "
          f"(${m:.0f}M).\n")
    print(out.to_string(index=False))
    print(f"\nPosition drop-off, pick {pick} → your next pick {nxt} (sorted by drop):")
    tt = pd.DataFrame({"pos": T["pos"], "best now": T["best_now"],
                       "value_$M": (T["best_now_value"] * m).round(1),
                       f"P(lasts to {nxt})": T["p_best_lasts"].round(2),
                       f"exp. best at {nxt}_$M": (T["exp_next"] * m).round(1),
                       "drop_$M": (T["drop"] * m).round(1), "available": T["n_available"]})
    print(tt.to_string(index=False))
    print("\npos_wait = his value now − expected best at his position at your next pick (if you need the "
          "position).\ntwo_pick = his value now + expected best player overall at your next pick once he is "
          "gone;\nthe highest two_pick is the best two-pick plan on value alone (--sort two_pick). "
          "Availabilities are treated as independent.")


if __name__ == "__main__":
    main()
