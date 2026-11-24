"""
cfDARKO Frontend

Streamlit dashboard for exploring college football → NFL draft projections.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from config import OUTPUT_DIR, PROCESSED_DIR, MODELED_STATS, POSITION_GROUPS

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="cfDARKO — College Football NFL Draft Projections",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_projections() -> pd.DataFrame:
    latest = OUTPUT_DIR / "projections" / "latest.parquet"
    if not latest.exists():
        return pd.DataFrame()
    return pd.read_parquet(latest)


@st.cache_data(ttl=3600)
def load_comps() -> pd.DataFrame:
    path = PROCESSED_DIR / "player_comps.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_score(v, fmt=".3f") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:{fmt}}"


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.1f}%"


def _stat_display_cols(pos_group: str) -> list[str]:
    """Return the Kalman mean columns to display for a given position group."""
    stats = MODELED_STATS.get(pos_group, [])
    return [f"kalman_mean_{s}" for s in stats]


def _height_display(height_in) -> str:
    if pd.isna(height_in):
        return "—"
    feet = int(height_in) // 12
    inches = int(height_in) % 12
    return f"{feet}'{inches}\""


def _render_score_bar(score: float, label: str, color: str = "#1f77b4"):
    pct = min(max(score * 100, 0), 100)
    st.markdown(f"""
    <div style='margin-bottom:4px'>
      <span style='font-size:13px;color:#888'>{label}</span>
      <div style='background:#eee;border-radius:4px;height:12px;width:100%'>
        <div style='background:{color};border-radius:4px;height:12px;width:{pct:.1f}%'></div>
      </div>
      <span style='font-size:12px;color:#555'>{score:.3f}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("cfDARKO")
st.sidebar.caption("College Football → NFL Draft Projections")

df = load_projections()
comps_df = load_comps()

if df.empty:
    st.error(
        "No projections found. Run the pipeline first:\n\n"
        "```bash\npython run_pipeline.py\n```"
    )
    st.stop()

# Hard-filter to the current draft class (max last_college_season_year).
# Also include players from the prior year who appear on the Hasan board —
# these are players who sat out the most recent season due to injury but are
# still draft-eligible (e.g., medical redshirts, season-ending injuries).
if "last_college_season_year" in df.columns:
    _max_year = int(df["last_college_season_year"].dropna().max())
    is_current = df["last_college_season_year"] == _max_year
    # Also include players on the Hasan board for the current draft year
    # who played their last snap before this season (e.g. injury/medical
    # redshirt). Use hasan_draft_year so the check is exact regardless of
    # how many seasons ago they last played.
    _draft_year = _max_year + 1
    if "hasan_rank" in df.columns and "hasan_draft_year" in df.columns:
        is_hasan_this_class = (
            df["hasan_rank"].notna() & (df["hasan_draft_year"] == _draft_year)
        )
    else:
        is_hasan_this_class = pd.Series(False, index=df.index)
    # When step 10 could determine the real draft pool (left school and
    # three years out of high school, or drafted / signed / on the board —
    # src/models/draft_pool.py), show exactly that pool; otherwise fall back
    # to everyone whose last season was the most recent one.
    if "draft_pool" in df.columns and df["draft_pool"].notna().any():
        df = df[df["draft_pool"].fillna(False).astype(bool)].copy()
    else:
        df = df[is_current | is_hasan_this_class].copy()

    # Global impact rank — computed before sidebar filters so it stays stable
    if "impact_score" in df.columns:
        df["impact_rank"] = (
            df["impact_score"]
            .rank(method="min", ascending=False, na_option="bottom")
            .astype(int)
        )

    # Merged board rank: one list in expected y3-4 snap share (step 10's
    # merged_usage) — board players by board-rank curve, off-board players by
    # the capped off-board model (step 8d). Ranked over the displayed pool;
    # ties break by board rank, then the off-board model's score.
    if "merged_usage" in df.columns:
        _tb = next(df[c] for c in ("offboard_score", "raw_impact_score", "impact_score")
                   if c in df.columns)
        _order = (df.assign(_tb=_tb)
                  .sort_values(["merged_usage", "hasan_rank", "_tb"],
                               ascending=[False, True, False], na_position="last"))
        df["merged_board_rank"] = pd.Series(range(1, len(_order) + 1),
                                            index=_order.index).reindex(df.index)
        df.loc[df["merged_usage"].isna(), "merged_board_rank"] = np.nan

    # Value-above-replacement rank, surfaced alongside (not replacing)
    # impact_rank. Replacement cost is near the league minimum at every
    # position except QB, so this mostly re-prices quarterbacks.
    if "impact_over_replacement" in df.columns:
        df["ior_rank"] = (
            df["impact_over_replacement"]
            .rank(method="min", ascending=False, na_option="bottom")
            .astype(int)
        )

    # Draft age = draft year minus birth year (partial coverage)
    if "birth_year" in df.columns and "last_college_season_year" in df.columns:
        df["draft_age"] = (
            (df["last_college_season_year"] + 1 - df["birth_year"])
            .where(df["birth_year"].notna())
        )

elif "season_year" in df.columns:
    _max_year = int(df["season_year"].dropna().max())
    df = df[df["season_year"] == _max_year].copy()
else:
    _max_year = None

# Position group filter
pos_groups_available = sorted(df["position_group"].dropna().unique()) \
    if "position_group" in df.columns else list(POSITION_GROUPS.keys())
selected_pos = st.sidebar.multiselect(
    "Position Group",
    options=pos_groups_available,
    default=pos_groups_available,
)

# School filter
if "school" in df.columns:
    all_schools = sorted(df["school"].dropna().unique())
    selected_schools = st.sidebar.multiselect("School", options=all_schools, default=[])

# Class year (1–5 only: FR/SO/JR/SR/GR)
CLASS_YEAR_MAP = {1: "Freshman", 2: "Sophomore", 3: "Junior", 4: "Senior", 5: "Graduate"}
if "class_year_int" in df.columns:
    valid_class_years = sorted([
        y for y in df["class_year_int"].dropna().unique()
        if 1 <= int(y) <= 5
    ])
    selected_years = st.sidebar.multiselect(
        "Class Year",
        options=valid_class_years,
        format_func=lambda y: CLASS_YEAR_MAP.get(int(y), str(int(y))),
        default=valid_class_years,
    )
else:
    selected_years = []

# Sort
sort_options = ["merged_board_rank", "impact_score", "upside"]
# "upside" is derived from impact_apy_p90 below, so gate on the source col.
sort_options = [s for s in sort_options
                if s in df.columns
                or (s == "upside" and "impact_apy_p90" in df.columns)]
sort_by = st.sidebar.selectbox("Sort by", options=sort_options,
                                 format_func=lambda s: s.replace("_", " ").title())

# Player search
search_query = st.sidebar.text_input("Search player", "")

# ── Apply filters ─────────────────────────────────────────────────────────────

filtered = df.copy()
if selected_pos:
    filtered = filtered[filtered["position_group"].isin(selected_pos)]
if selected_schools:
    filtered = filtered[filtered["school"].isin(selected_schools)]
# Filter by class year only when the selection is narrowed: players with no
# listed class year (no roster row — many FCS players) would otherwise drop
# out even with every year selected.
if ("class_year_int" in filtered.columns and selected_years
        and set(selected_years) != set(valid_class_years)):
    filtered = filtered[filtered["class_year_int"].isin(selected_years)]
if search_query and "player_name" in filtered.columns:
    filtered = filtered[
        filtered["player_name"].str.contains(search_query, case=False, na=False)
    ]

# Derived columns used for sorting and display.
# Upside = the player's 90th-percentile 2nd-contract APY (upper bound of the
# 80% interval from the quantile heads) — the same quantity shown on the
# player-detail page. Replaces the old impact/role ratio, which was unstable
# for small role scores and, post global-first scoring, mixed model families.
if "impact_apy_p90" in filtered.columns:
    filtered = filtered.copy()
    filtered["upside"] = filtered["impact_apy_p90"]

if sort_by in filtered.columns:
    ascending = "rank" in sort_by
    filtered = filtered.sort_values(sort_by, ascending=ascending, na_position="last")

# ── Main page ─────────────────────────────────────────────────────────────────

draft_year = f"{_max_year + 1} " if _max_year else ""
st.title(f"cfDARKO: {draft_year}NFL Draft Projections")
if "projection_date" in df.columns:
    st.caption(f"Last updated: {df['projection_date'].iloc[0]}")

tab_board, tab_player = st.tabs(["Draft Board", "Player Detail"])

# ── Draft Board tab ────────────────────────────────────────────────────────────

with tab_board:
    st.subheader(f"Draft Board ({len(filtered):,} players)")

    display_cols = ["impact_rank", "merged_board_rank", "player_name", "school", "position",
                     "position_group", "impact_score", "merged_usage", "upside",
                     "height_in", "weight_lbs", "draft_age", "class_year_int",
                     "hasan_rank", "best_comp_name", "best_comp_school", "best_comp_draft_year"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    display_df = filtered[display_cols].copy()

    # Build "Name (School 'YY)" best comp string
    if "best_comp_name" in display_df.columns:
        def _fmt_comp(row) -> str:
            name = row.get("best_comp_name", "")
            if not isinstance(name, str) or not name:
                return "—"
            school = row.get("best_comp_school", "")
            year = row.get("best_comp_draft_year", None)
            suffix = ""
            if isinstance(school, str) and school:
                suffix += school
            if pd.notna(year):
                yy = str(int(year) - 1)[-2:]   # last college year = first NFL season - 1
                suffix += f" '{yy}"
            return f"{name} ({suffix})" if suffix else name
        display_df["best_comp"] = display_df.apply(_fmt_comp, axis=1)
    else:
        display_df["best_comp"] = "—"

    display_df = display_df.drop(
        columns=["best_comp_name", "best_comp_school", "best_comp_draft_year"],
        errors="ignore",
    )

    # Format height and class year
    if "height_in" in display_df.columns:
        display_df["height_in"] = display_df["height_in"].apply(_height_display)
    if "class_year_int" in display_df.columns:
        yr_abbr = {1: "FR", 2: "SO", 3: "JR", 4: "SR", 5: "GR"}
        display_df["class_year_int"] = display_df["class_year_int"].apply(
            lambda y: yr_abbr.get(int(y), "—") if pd.notna(y) and 1 <= int(y) <= 5 else "—"
        )

    col_rename = {
        "impact_rank": "Rank",
        "merged_board_rank": "Merged Rank",
        "merged_usage": "Exp. Snap %",
        "player_name": "Name", "school": "School", "position": "Pos",
        "position_group": "Group",
        "impact_score": "Impact",
        "upside": "Upside (p90 APY)",
        "height_in": "Ht", "weight_lbs": "Wt", "draft_age": "Age",
        "class_year_int": "Yr", "hasan_rank": "Board Rank", "best_comp": "Best Comp",
    }
    display_df = display_df.rename(columns={k: v for k, v in col_rename.items()
                                             if k in display_df.columns})

    col_cfg = {}
    if "Rank" in display_df.columns:
        col_cfg["Rank"] = st.column_config.NumberColumn(format="%d")
    if "Merged Rank" in display_df.columns:
        col_cfg["Merged Rank"] = st.column_config.NumberColumn(
            format="%d", help="Rank on the merged board: board players by "
                              "their board rank's historical snap share, "
                              "off-board players by the model (capped)")
    if "Exp. Snap %" in display_df.columns:
        display_df["Exp. Snap %"] = display_df["Exp. Snap %"] * 100
        col_cfg["Exp. Snap %"] = st.column_config.NumberColumn(
            format="%.1f%%", help="Expected NFL snap share in years 3-4 "
                                  "(0 if never in the NFL)")
    if "Impact" in display_df.columns:
        display_df["Impact"] = display_df["Impact"] * 100
        col_cfg["Impact"] = st.column_config.NumberColumn(
            format="%.2f%%", help="Projected 2nd-contract APY as %% of "
                                  "salary cap (median outcome)")
    if "Upside (p90 APY)" in display_df.columns:
        display_df["Upside (p90 APY)"] = display_df["Upside (p90 APY)"] * 100
        col_cfg["Upside (p90 APY)"] = st.column_config.NumberColumn(
            format="%.2f%%", help="90th-percentile projected 2nd-contract "
                                  "APY as %% of salary cap")
    if "Wt" in display_df.columns:
        col_cfg["Wt"] = st.column_config.NumberColumn(format="%d lbs")
    if "Age" in display_df.columns:
        col_cfg["Age"] = st.column_config.NumberColumn(format="%d")
    if "Board Rank" in display_df.columns:
        col_cfg["Board Rank"] = st.column_config.NumberColumn(format="%d")

    st.dataframe(display_df, use_container_width=True, hide_index=True,
                  height=600, column_config=col_cfg)

    # Download button
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv,
                        file_name="cfdarko_projections.csv", mime="text/csv")


# ── Player Detail tab ──────────────────────────────────────────────────────────

with tab_player:
    if "player_name" not in filtered.columns or filtered.empty:
        st.info("No players to display.")
    else:
        def _player_label(row) -> str:
            name = row.get("player_name", "Unknown") or "Unknown"
            school = row.get("school", "") or ""
            year = row.get("season_year", "")
            year_str = f", {int(year)}" if pd.notna(year) else ""
            return f"{name} ({school}{year_str})" if school else name

        player_options = filtered.reset_index(drop=True)
        player_labels = player_options.apply(_player_label, axis=1).tolist()
        selected_label = st.selectbox("Select player", options=player_labels)
        selected_idx = player_labels.index(selected_label)
        player_row = player_options.iloc[selected_idx]

        pos_group = player_row.get("position_group", "")
        stats = MODELED_STATS.get(pos_group, [])

        # ── Header ─────────────────────────────────────────────────────────

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            name = player_row.get("player_name", "Unknown")
            school = player_row.get("school", "")
            pos = player_row.get("position", "")
            st.subheader(f"{name}")
            st.caption(f"{school} · {pos} ({pos_group})")
            ht = _height_display(player_row.get("height_in"))
            wt = player_row.get("weight_lbs", "")
            wt_str = f"{int(wt)} lbs" if pd.notna(wt) else "—"
            year_map = {1: "Freshman", 2: "Sophomore", 3: "Junior", 4: "Senior", 5: "Graduate"}
            _cyi = player_row.get("class_year_int")
            yr = year_map.get(int(_cyi) if pd.notna(_cyi) else 0, "—")
            st.markdown(f"**{ht}** · **{wt_str}** · {yr}")

        with col2:
            board = player_row.get("hasan_rank", np.nan)
            st.metric("Board Rank", f"{int(board)}" if pd.notna(board) else "Off board")
            mrank = player_row.get("merged_board_rank", np.nan)
            musage = player_row.get("merged_usage", np.nan)
            if pd.notna(mrank):
                st.metric("Merged Rank", f"{int(mrank)}",
                          help=f"Expected y3-4 snap share {musage:.1%}")

        with col3:
            impact = player_row.get("impact_score", np.nan)
            st.metric("Impact Score", _fmt_score(impact))
            if pd.notna(impact):
                # impact_score is second-contract APY as a share of the cap —
                # NOT AV/season, which this was mislabelled as. Scale the bar
                # against a realistic ceiling (0.15 of cap ≈ a top-of-market
                # deal); /10 made every bar invisible.
                _render_score_bar(min(float(impact) / 0.15, 1.0),
                                   "APY % of Cap", "#FF9800")
            ior = player_row.get("impact_over_replacement", np.nan)
            if pd.notna(ior):
                st.caption(f"Above replacement: {float(ior):+.4f} of cap")

        # APY prediction interval (p10/p50/p90 from quantile heads).
        p10 = player_row.get("impact_apy_p10")
        p90 = player_row.get("impact_apy_p90")
        apy = player_row.get("impact_apy_score")
        if pd.notna(p10) and pd.notna(p90) and pd.notna(apy):
            st.caption(
                f"Projected 2nd-contract APY (% of cap): **{apy*100:.2f}%** "
                f"  · 80% interval: {p10*100:.2f}% – {p90*100:.2f}%"
            )

        # SHAP top contributors (why this score?).
        shap_json = player_row.get("shap_top_features_impact")
        if isinstance(shap_json, str) and shap_json.strip():
            import json as _json
            try:
                contribs = _json.loads(shap_json)
                if contribs:
                    with st.expander("Why this Impact score? (SHAP top contributors)"):
                        for c in contribs:
                            f = c.get("feature", "")
                            sv = c.get("shap", 0.0)
                            v = c.get("value")
                            arrow = "▲" if sv > 0 else "▼"
                            v_str = f"value={v}" if v is not None else ""
                            st.markdown(
                                f"- **{f}** {arrow} `{sv:+.4f}` {v_str}"
                            )
            except Exception:
                pass

        st.divider()

        # ── Skill projections ──────────────────────────────────────────────

        if stats:
            st.subheader("Skill Estimates")
            stat_rows = []
            for stat in stats:
                k_mean = player_row.get(f"kalman_mean_{stat}", np.nan)
                k_var = player_row.get(f"kalman_var_{stat}", np.nan)
                ewma = player_row.get(f"ewma_{stat}", np.nan)
                yoy = player_row.get(f"kalman_yoy_{stat}", np.nan)
                pct = player_row.get(f"pct_kalman_mean_{stat}", np.nan)
                stat_rows.append({
                    "Stat": stat.replace("_", " ").title(),
                    "Kalman Mean": _fmt_score(k_mean),
                    "Uncertainty": _fmt_score(k_var),
                    "EWMA": _fmt_score(ewma),
                    "YoY Δ": _fmt_score(yoy),
                    "Percentile": _fmt_pct(pct),
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True,
                          hide_index=True)
        else:
            st.info(f"No per-stat skill estimates available for {pos_group} "
                    f"(OL uses physical attributes only).")

        st.divider()

        # ── Historical comps ───────────────────────────────────────────────

        st.subheader("Historical Comps")
        if comps_df.empty:
            st.info("No comps data available.")
        else:
            player_comps = comps_df[
                comps_df["player_id"] == player_row.get("player_id")
            ].sort_values("comp_rank")

            if player_comps.empty:
                st.info("No comps found for this player.")
            else:
                comp_display = player_comps[[
                    "comp_rank", "comp_player_name", "comp_school",
                    "comp_impact_target", "comp_career_w_av",
                    "comp_first_nfl_season",
                ]].copy()
                comp_display = comp_display.rename(columns={
                    "comp_rank": "#",
                    "comp_player_name": "Name",
                    "comp_school": "School",
                    "comp_impact_target": "Impact (actual)",
                    "comp_career_w_av": "Career AV",
                    "comp_first_nfl_season": "NFL Year",
                })
                for col in ["Impact (actual)"]:
                    if col in comp_display.columns:
                        comp_display[col] = comp_display[col].apply(_fmt_score)
                if "Career AV" in comp_display.columns:
                    comp_display["Career AV"] = comp_display["Career AV"].apply(
                        lambda v: f"{int(v)}" if pd.notna(v) else "—"
                    )
                if "NFL Year" in comp_display.columns:
                    comp_display["NFL Year"] = comp_display["NFL Year"].apply(
                        lambda v: f"{int(v)}" if pd.notna(v) else "—"
                    )
                st.dataframe(comp_display, use_container_width=True, hide_index=True)
