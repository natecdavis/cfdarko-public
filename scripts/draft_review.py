"""
Draft review: for a past draft and a team, step through the team's picks
and compare who our model had as the most valuable players still available
with the player the team actually took, and how each player turned out.

Model value = expected rookie-deal value (merged_value: Σ y1-y4 veteran-
market price of the projected role at the position, % of cap), the same
currency as scripts/board_at_pick.py. At a given pick the rookie wage is
set by the slot, so ranking the available players by value ranks them by
expected surplus at that pick.

STRICTLY AS OF DRAFT DAY. For draft D the merged board is refit using only
what was known before draft D:
    board curve      classes whose four rookie seasons were complete before
                     D (last college season <= D - 5), plus partial outcomes
                     of the three classes after them, projected from the
                     seasons played before D only (src/models/partial_outcomes)
    off-board curve  the same complete classes
    off-board score  step 8d's expanding-window out-of-fold score (classes
                     with last college season <= 2021); later classes use the
                     production 8d model, fit on classes up to 2021 (no
                     player's own outcome; the training classes' outcomes
                     run past draft day)
Big-board ranks start with the 2014 draft and the first complete class is
2014's, so drafts 2018-2026 are covered; 2018 rests on one complete class
plus three partial ones and is the noisiest.
Value AT THE PICK (experiments/01f, src/models/draft_slide.py): a board
player still available past his board rank is marked down, value x
min(1, (k / board rank)^γ), ratio capped at 3.3. Reaches are NOT marked up
(a team's reach is its own information, not evidence the page should
credit). γ is fit as of each draft on drafts whose four rookie seasons were
complete before it (2014 .. D-4). Available players are ranked by value at
the current pick.
Not as of draft day: the value CURRENCY itself (the rookie-value label's
veteran pay curves, 2016-24 contracts), the yardstick everything is
measured in.

Realized value = the same label's value over the rookie seasons played so
far (through 2025): complete for drafts <= 2022, partial after, none for
2026. 0 for players who never reached the NFL.

Output: data/output/draft_review/draft_review.html (self-contained; open in
a browser) and draft_review.json.

Drafted players link to college records via the bridge (pfr_id), else a
unique normalized-name match within the class (nickname gaps in the bridge).

Usage: python scripts/draft_review.py [--drafts 2018 2026] [--keep 500]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR, SALARY_CAP, END_YEAR  # noqa: E402
from src.models.merged_board import MergedBoard, BOARD_ERA_START, EXCLUDED_GROUPS  # noqa: E402
from src.models.offboard_model import pid  # noqa: E402
from src.models.partial_outcomes import board_value_frame, project_partial  # noqa: E402
from src.models.draft_slide import slide_cohort, fit_gamma, RATIO_MAX  # noqa: E402
from src.models.availability import fit_from_frames, fit_offboard, _off_features, OFF_CLASSES  # noqa: E402
from src.models.position_dropoff import view_position  # noqa: E402

OUT = OUTPUT_DIR / "draft_review"
LAST_SEASON = END_YEAR - 1                   # last NFL season with outcomes
FIRST_DRAFT = BOARD_ERA_START + 5            # first draft with one complete board class
FRANCHISE = {"OAK": "LVR", "SDG": "LAC", "STL": "LAR"}
TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GNB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KAN": "Kansas City Chiefs", "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LVR": "Las Vegas / Oakland Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NOR": "New Orleans Saints", "NWE": "New England Patriots", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SFO": "San Francisco 49ers", "TAM": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}


def load_inputs():
    cols = ["player_id", "player_name", "school", "position_group", "position", "last_college_season_year",
            "hasan_rank", "hasan_draft_year", "offboard_score", "offboard_p_nfl", "combine_invited"]
    P = pd.read_parquet(OUTPUT_DIR / "projections" / "latest.parquet", columns=cols)
    P = P.dropna(subset=["last_college_season_year"])
    P = P.sort_values("last_college_season_year").drop_duplicates("player_id", keep="last")
    P["pid"] = pid(P["player_id"])
    P = P.dropna(subset=["pid"]).reset_index(drop=True)

    oof = (pd.read_parquet(PROCESSED_DIR / "offboard_models" / "offboard_oof_predictions.parquet")
           .drop_duplicates("player_id", keep="last").rename(columns={"score": "pred"}))
    oof["pid"] = pid(oof["player_id"])
    # As-of off-board score: expanding-window OOF where it exists, else production.
    P["score"] = P["pid"].map(oof.dropna(subset=["pid"]).set_index("pid")["pred"])
    P["score_src"] = np.where(P["score"].notna(), "oof", "production")
    P["score"] = P["score"].fillna(P["offboard_score"])
    P["pnfl"] = P["pid"].map(oof.dropna(subset=["pid"]).set_index("pid")["p_nfl"]).fillna(P["offboard_p_nfl"])

    L = pd.read_parquet(PROCESSED_DIR / "rookie_value_labels.parquet")
    L["pid"] = pid(L["player_id"])
    rv = L.dropna(subset=["pid"]).drop_duplicates("pid").set_index("pid")["value_y14"]
    oof["rv"] = np.where(oof["nfl"], oof["pid"].map(rv), 0.0)
    oof = oof[oof["rv"].notna()]

    picks = pd.read_parquet(RAW_DIR / "team_context" / "draft_picks.parquet")
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet")
    b = bridge.dropna(subset=["pfr_id", "player_id"]).drop_duplicates("pfr_id")
    picks["pid"] = picks["pfr_player_id"].map(dict(zip(b["pfr_id"], pid(b["player_id"]))))
    picks["team"] = picks["team"].replace(FRANCHISE)
    picks["link"] = np.where(picks["pid"].notna(), "bridge", None)
    # Fallback for picks the bridge leaves unlinked (nicknames: "Mo Hurst" for
    # Maurice Hurst): unique normalized-name match within the class.
    from src.scrapers.nfl_data import _normalize_name
    P["nn"] = P["player_name"].map(_normalize_name)
    for i, r in picks[picks["pid"].isna()].iterrows():
        c = P[(P["nn"] == _normalize_name(r["pfr_player_name"]))
              & P["last_college_season_year"].between(r["season"] - 2, r["season"] - 1)]
        if len(c) == 1 and not c["pid"].iloc[0] in set(picks.loc[picks["season"] == r["season"], "pid"]):
            picks.loc[i, ["pid", "link"]] = [c["pid"].iloc[0], "name"]
    print("pick links:", picks[picks["season"] >= FIRST_DRAFT]["link"].value_counts(dropna=False).to_dict())
    return P, oof, L, picks, bridge


def availability_as_of(D, P, cls, oof_raw, bridge):
    """Per-player availability inputs as of draft D (fit only on drafts before
    D): board players → (mu, sigma) of the position-aware log-normal (00zx
    C_pos); off-board players → p_d (P drafted) and the invite flag, with the
    pick CDFs F[invite] (00zy). The page computes P(still there at pick n |
    there at pick k) = S(n-1) / S(k)."""
    am = fit_from_frames(P, bridge, max_class=D - 1, by_position=True)
    on = cls["hasan_rank"].notna().to_numpy()
    mu = np.full(len(cls), np.nan)
    sg = np.full(len(cls), np.nan)
    m, s = am.mu_sigma(cls.loc[on, "hasan_rank"].to_numpy(), cls.loc[on, "position_group"].to_numpy())
    mu[on], sg[on] = m, s
    pd_ = np.full(len(cls), np.nan)
    inv = np.zeros(len(cls))
    F = None
    lo = max(OFF_CLASSES[0], 2015)
    if D - 1 >= lo:
        om = fit_offboard(P, bridge, oof_raw, classes=(lo, min(D - 1, OFF_CLASSES[1])))
        off = cls[~on]
        X, iv = _off_features(off.assign(_c=0), "_c", "pnfl", "score", "position_group", "combine_invited")
        pd_[~on] = om.clf.predict_proba(X)[:, 1]
        inv[~on] = iv
        F = {str(int(g)): [round(float(x), 4) for x in f] for g, f in om.F.items()}
    return mu, sg, pd_, inv, F, am.n_train


def realized(L, draft):
    """Realized rookie value by pfr_id over seasons played through LAST_SEASON."""
    n = int(np.clip(LAST_SEASON - draft + 1, 0, 4))
    if n == 0:
        return pd.Series(dtype=float), 0
    cols = [f"value_y{k}" for k in range(1, n + 1)]
    v = L.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")[cols]
    return v.sum(axis=1, min_count=n), n


def build_draft(D, P, oof, L, picks, keep, slide, oof_raw, bridge):
    extra, _ = project_partial(board_value_frame(P, L), D)
    mb = MergedBoard.fit(oof, D - 5, target="rv", out_col="merged_value", extra_board=extra)
    pk = picks[picks["season"] == D].sort_values("pick")
    cls = P[(P["last_college_season_year"] == D - 1) | P["pid"].isin(pk["pid"])].copy()
    cls["last_college_season_year"] = D - 1          # drafted fifth-years etc. join the class
    snap = cls.rename(columns={"position_group": "pos_group"})
    pred = mb.predict(snap, "score", ref_sizing=False)
    cls["value"] = pred["merged_value"].to_numpy(dtype=float)
    cls["source"] = pred["merged_source"].to_numpy()
    cls = cls.merge(pk[["pid", "pick", "round", "team", "pfr_player_id"]].dropna(subset=["pid"])
                    .drop_duplicates("pid"), on="pid", how="left")
    mu, sg, p_d, inv, F, n_av = availability_as_of(D, P, cls.reset_index(drop=True), oof_raw, bridge)
    cls = cls.reset_index(drop=True).assign(mu=mu, sg=sg, p_d=p_d, inv=inv)
    cls["vp"] = view_position(cls["position_group"], cls["position"])
    cls = cls.sort_values(["value", "hasan_rank"], ascending=[False, True], na_position="last")
    cls["model_rank"] = np.arange(1, len(cls) + 1, dtype=float)
    cls.loc[cls["value"].isna(), "model_rank"] = np.nan
    rv, n_seasons = realized(L, D)
    cls["real"] = cls["pfr_player_id"].map(rv) if n_seasons else np.nan
    if n_seasons:       # never reached the NFL → 0 (K/P have no label → stays NaN)
        nolab = cls["real"].isna() & ~cls["position_group"].isin(EXCLUDED_GROUPS)
        cls.loc[nolab, "real"] = 0.0
    keep_rows = (cls["model_rank"] <= keep) | cls["pick"].notna()
    cls = cls[keep_rows].reset_index(drop=True)

    players = []
    for r in cls.itertuples():
        players.append({
            "n": r.player_name, "s": r.school, "p": r.position_group,
            "b": None if pd.isna(r.hasan_rank) else int(r.hasan_rank),
            "v": None if pd.isna(r.value) else round(float(r.value), 4),
            "m": None if pd.isna(r.model_rank) else int(r.model_rank),
            "src": r.source,
            "k": None if pd.isna(r.pick) else int(r.pick),
            "t": None if pd.isna(r.team) else r.team,
            "r": None if pd.isna(r.real) else round(float(r.real), 4),
            "vp": r.vp,
            "a": ([round(float(r.mu), 4), round(float(r.sg), 4)] if not pd.isna(r.mu)
                  else None if pd.isna(r.p_d) else [round(float(r.p_d), 4), int(r.inv)])})
    unl = pk[pk["pid"].isna() | ~pk["pid"].isin(cls.loc[cls["value"].notna(), "pid"])]
    unlinked = [{"k": int(r.pick), "t": r.team, "n": r.pfr_player_name, "p": r.position,
                 "s": r.college} for r in unl.itertuples()]
    if n_seasons:   # realized value for picks we could not score (still show it)
        for u, r in zip(unlinked, unl.itertuples()):
            x = rv.get(r.pfr_player_id, np.nan)
            u["r"] = None if pd.isna(x) else round(float(x), 4)
    rounds = pk.groupby("round")["pick"].min().astype(int).to_dict()
    sc = slide[slide["draft_year"].between(BOARD_ERA_START + 1, D - 4)]
    gamma = fit_gamma(sc)
    meta = {"fit_classes": mb.diagnostics["fit_seasons"], "n_board": mb.diagnostics["n_board"],
            "n_projected": mb.diagnostics["n_board_projected"], "seasons_realized": n_seasons,
            "round_starts": {int(k): v for k, v in rounds.items()},
            "gamma": round(gamma, 4), "gamma_n": int(len(sc)), "clip": [1.0, RATIO_MAX],
            "avail_F": F, "avail_n": int(n_av),
            "picks": [{"k": int(r.pick), "rd": int(r.round), "t": r.team} for r in pk.itertuples()]}
    print(f"draft {D}: fit classes {meta['fit_classes']} ({meta['n_board']} board rows, "
          f"{meta['n_projected']} projected); {len(players)} players kept; "
          f"{len(unlinked)} picks unscored; realized seasons {n_seasons}; "
          f"slide γ {gamma:+.3f} (n {len(sc)}); availability fit on {n_av} board players "
          f"(drafts < {D}), off-board {'yes' if F else 'no'}")
    return {"players": players, "unlinked": unlinked, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", nargs=2, type=int, default=[FIRST_DRAFT, END_YEAR])
    ap.add_argument("--keep", type=int, default=500, help="model-ranked players kept per draft (plus all drafted)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P, oof, L, picks, bridge = load_inputs()
    oof_raw = pd.read_parquet(PROCESSED_DIR / "offboard_models" / "offboard_oof_predictions.parquet")
    slide = slide_cohort(P, L, pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet"))
    data = {"cap": SALARY_CAP, "last_season": LAST_SEASON, "teams": TEAM_NAMES, "drafts": {}}
    for D in range(max(args.drafts[0], FIRST_DRAFT), args.drafts[1] + 1):
        if (picks["season"] == D).any():
            data["drafts"][str(D)] = build_draft(D, P, oof, L, picks, args.keep, slide, oof_raw, bridge)
    js = json.dumps(data, separators=(",", ":"))
    (OUT / "draft_review.json").write_text(js)
    html = (ROOT / "scripts" / "draft_review_template.html").read_text().replace("__DATA__", js)
    (OUT / "draft_review.html").write_text(html)
    print(f"wrote {OUT / 'draft_review.html'} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
