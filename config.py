"""
cfDARKO Configuration
Central configuration for all pipeline parameters.
"""

from __future__ import annotations    # PEP 604 syntax (str | None) compat for Python <3.10
from pathlib import Path

# ── Directories ────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# ── Coverage Years ─────────────────────────────────────────────────────────────

START_YEAR = 2004   # CFBD API coverage start (solid data from ~2004)
END_YEAR = 2026

# Bridge player NFL outcome window: how many NFL seasons to look at
NFL_LABEL_WINDOW_YEARS = 4

# ── Position Groups ────────────────────────────────────────────────────────────

POSITION_GROUPS = {
    "QB":    ["QB"],
    "RB":    ["RB", "FB"],
    "WR":    ["WR"],
    "TE":    ["TE"],
    "OL":    ["OT", "OG", "C", "IOL", "OL", "G"],   # "OL"/"G" used as generic by CFBD
    "DL":    ["DE", "DT", "NT", "DL"],               # "DL" used as generic by CFBD
    "EDGE":  ["EDGE"],                                # edge rushers; membership is
                                                      # reconciled in step 03 from
                                                      # PFF pos-code (+ roster DE
                                                      # fallback), see _reconcile_edge
    "LB":    ["LB", "ILB", "OLB", "MLB"],
    "DB":    ["CB", "S", "FS", "SS", "DB"],
    "K":     ["K", "PK"],
    "P":     ["P"],
}

# Reverse map: position string → group key (abbreviations)
POSITION_TO_GROUP: dict[str, str] = {
    pos: group
    for group, positions in POSITION_GROUPS.items()
    for pos in positions
}

# Additional full-name position aliases (CFBD draft_picks uses full names)
_POSITION_FULL_NAMES: dict[str, str] = {
    "Quarterback":         "QB",
    "Running Back":        "RB",
    "Fullback":            "RB",
    "Wide Receiver":       "WR",
    "Tight End":           "TE",
    "Offensive Tackle":    "OT",
    "Offensive Guard":     "OG",
    "Center":              "C",
    "Defensive End":       "DE",
    "Defensive Edge":      "EDGE",
    "Defensive Tackle":    "DT",
    "Outside Linebacker":  "OLB",
    "Inside Linebacker":   "ILB",
    "Linebacker":          "LB",
    "Cornerback":          "CB",
    "Safety":              "S",
    "Defensive Back":      "DB",
    "Place Kicker":        "K",
    "Punter":              "P",
    "Long Snapper":        "LS",
    # Other abbreviations not in POSITION_GROUPS
    "G":   "OG",
    "SAF": "S",
    # Long snappers are intentionally NOT mapped here: they fall through to
    # OTHER via the fillna in step 03 (the same path as any unmapped token).
    # A prior "LS": "OTHER" entry looked like it did this but was dead code —
    # the update guard below drops it because "OTHER" isn't a POSITION_TO_GROUP
    # key, so it never produced a mapping. Left unmapped for honesty.
}
POSITION_TO_GROUP.update({
    full: POSITION_TO_GROUP.get(abbr, None)
    for full, abbr in _POSITION_FULL_NAMES.items()
    if POSITION_TO_GROUP.get(abbr) is not None
})

# ── Stats to Model Per Position Group ─────────────────────────────────────────
# All per-game (counting stats are per-game totals)
# OL is empty — no per-play stats available without PFF

MODELED_STATS: dict[str, list[str]] = {
    "QB": [
        "pass_att", "pass_yards", "pass_tds", "interceptions",
        "completion_pct", "rush_att", "rush_yards", "rush_tds",
        "sacks_taken", "passer_rating", "ppa_avg_passing",
        "pass_share",   # fraction of team pass attempts this QB ran
    ],
    "RB": [
        "rush_att", "rush_yards", "rush_tds",
        "rec", "rec_yards", "rec_tds",
        "fumbles", "yards_per_carry", "ppa_avg_rushing",
        "rush_share",   # fraction of team rush attempts
        "rec_share",    # fraction of team receptions
    ],
    "WR": [
        "rec", "rec_yards", "rec_tds",
        "catch_pct", "yards_per_rec", "ppa_avg",
        "rec_share",    # fraction of team receptions (role prominence signal)
    ],
    "TE": [
        "rec", "rec_yards", "rec_tds",
        "catch_pct", "yards_per_rec", "ppa_avg",
        "rec_share",    # fraction of team receptions
    ],
    "OL": [],   # Physical attributes + team context only
    "DL": [
        "tackles", "sacks", "tfl", "ff", "passes_defended", "ints",
        "qb_hurries",
    ],
    "EDGE": [   # mirrors DL — same box-score defensive stats
        "tackles", "sacks", "tfl", "ff", "passes_defended", "ints",
        "qb_hurries",
    ],
    "LB": [
        "tackles", "sacks", "tfl", "ff", "passes_defended", "ints",
        "qb_hurries",
    ],
    "DB": [
        "tackles", "sacks", "tfl", "ff", "passes_defended", "ints",
    ],
    "K": [
        "fg_att", "fg_made", "fg_pct",
        "xp_att", "xp_made", "xp_pct",
        "kicking_pts",
    ],
    "P": [
        "punt_yards", "net_punt_yards", "punts",
    ],
}

# All stats across all position groups (flat list, deduplicated)
ALL_STATS: list[str] = sorted({
    s for stats in MODELED_STATS.values() for s in stats
})

# ── Contextual Model Parameters ────────────────────────────────────────────────

RIDGE_ALPHA = 1.0
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]   # Grid-searched per stat in Step 4
ELASTICNET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]   # L1 mix tried alongside Ridge in Step 4

# Adaptive Lasso fallback (Zou 2006). Pilot Ridge coefficients drive per-feature
# shrinkage in a final Lasso fit, pruning noise features that shared-alpha
# couldn't suppress. Attempted whenever Ridge/ElasticNet CV R² <
# ADAPTIVE_LASSO_TRIGGER_R2; AL is only swapped in if it strictly beats the
# pilot's CV R² (see contextual.py:466), so this gate just controls whether
# AL is tried — the loser is always thrown away. We set it to +∞ to attempt
# AL on every head, since the diagnostic showed 58/60 heads have CV R² < 0.05
# and there's no cost to trying when the comparator is honest.
ADAPTIVE_LASSO_GAMMAS       = [0.5, 1.0, 2.0]   # exponent on pilot |β̂_j|
ADAPTIVE_LASSO_TRIGGER_R2   = float("inf")
ADAPTIVE_LASSO_EPS          = 1e-4              # avoids w_j = 0
ADAPTIVE_LASSO_MIN_FEATURES = 3                 # never collapse below this

# Game-level contextual features for Ridge regression (opponent-only rolling stats,
# de-meaned within player-season, plus location/weather/time).
# No team-side features — mirrors cdarko's approach of removing self-referential context.
# Conference tier dummies (conf_tier_power4, conf_tier_g5) capture systematic baseline
# opponent quality; computed in build_context_features() from the conference_tier column.
CONTEXT_FEATURES = [
    # Opponent composite efficiency — de-meaned within player-season
    # (removes schedule-strength temporal confound; same pattern as cdarko)
    "opp_rolling_def_ppa",
    "opp_rolling_off_ppa",
    # Opponent four-factor components — left raw (carry legitimate within-season signal)
    "opp_rolling_def_success_rate",
    "opp_rolling_def_stuff_rate",
    "opp_rolling_off_success_rate",
    "opp_rolling_off_explosiveness",
    "opp_rolling_def_explosiveness",
    "opp_rolling_havoc_total",
    "opp_rolling_havoc_front_seven",
    "opp_rolling_havoc_db",
    # NOTE: conf_tier_power4/g5 dropped 2026-05-18. These were the PLAYER's
    # conference (constant per (player, season)), so they functionally acted
    # as school-tier fixed effects rather than per-game context. AdaptiveLasso
    # kept them on 9/11 and 8/11 active heads with universal retention — the
    # "kept everywhere without per-game causal channel" red flag. Opponent
    # quality is already captured by opp_rolling_def_* directly. The
    # conference_tier column remains on player_attributes.parquet and feeds
    # step 8 via ATTRIBUTE_MERGE_COLS where it's a legitimate career attribute.
    # build_context_features() retains the column derivation so fixtures
    # keep building it; just not in CONTEXT_FEATURES anymore.

    # Location
    "is_home",
    "is_neutral",
    # Time / schedule context
    "days_rest",
    "opp_days_rest",
    "game_hour",         # local kickoff hour at the home stadium (0-23)
    "tz_diff",           # game_tz - player_tz (positive = traveled east)
    "body_clock_hour",   # game_hour - tz_diff = what time it feels for the
                         # visiting player; equals game_hour for home games
    "week",              # week of season (1-15ish); proxy for early-season
                         # noise / late-season opt-outs / bowl context
    # Weather (NaN-safe: missing values are mean-imputed inside the model pipeline)
    "temperature",
    "wind_speed",
    "precipitation",
    "game_indoors",
    "humidity",
    # Venue (joined from venues.parquet via game_schedule.venue_id; see
    # _add_venue_features() in step 01). Encoded as DIFFERENCES from the
    # player's school's home stadium to avoid leaking school / conference
    # identity through fingerprintable absolute values (Air Force=7218ft
    # uniquely tags Falcon Stadium; SEC≈grass / MAC≈turf would let the model
    # memorize program identity via surface). Diff form means HOME games
    # contribute 0 (no travel shock) and AWAY games carry the signal.
    #   altitude_diff_from_home_ft : game_elevation - school_home_elevation,
    #                                rounded to 500ft buckets to avoid
    #                                unique-pair fingerprints (e.g. Stanford
    #                                → Air Force = +7188 → bucket to +7000).
    #   surface_diff_from_home     : game_grass - home_grass, signed in
    #                                {-1, 0, +1}; NaN → 0 default (no signal).
    # Strongest a-priori channel is K (FG distance in thin air, footing on
    # long attempts); other heads' contribution will surface via Boruta +
    # cv_mae A/B after the next retune.
    "altitude_diff_from_home_ft",
    "surface_diff_from_home",
    # NOTE: capacity_diff_from_home — BUILT, TESTED 2026-07-26, REJECTED.
    # It PASSED the cheap step-4 screen and still lost downstream, which is the
    # useful part of the result:
    #   screen      retention 37/49 heads (76%); coefficient sign matches the
    #               causal prior (16 neg / 5 pos on offensive heads, K heads
    #               −0.42..−0.53); median Δcv_r2 +0.0023 over the 16 heads whose
    #               adjustment is actually used, 12/16 improved.
    #   downstream  cv_mae +0.68% (improved 9/33), holdout ρ −0.0178 (improved
    #               6/33), from a paired 4→5→6→8 rebuild with identical seeds,
    #               params and gamelogs. Damage concentrated in RB_impact_wav:
    #               +17.8% cv_mae, Δρ −0.380.
    # LESSON: the retention+sign screen is NECESSARY BUT NOT SUFFICIENT for a
    # context feature. travel_distance_mi (below) failed the screen; this one
    # passed it and still hurt. Local contextual-fit quality does not predict
    # draft-model ranking value — always pay for the downstream rebuild before
    # enabling a CONTEXT_FEATURE.
    # Step 01 still computes the column and src/models/contextual.py still
    # defaults it, so re-enabling is one line here. 97.1% coverage. Original
    # rationale and the sign caveat below remain valid if anyone retests.
    # src/models/contextual.py defaults it, so enabling is just re-adding the
    # name here — but it has never been evaluated, because reaching a draft model
    # requires rebuilding adjusted_gamelogs (step 1 venue join) plus steps 4/5/6,
    # and per the calibration-coupling rule a CONTEXT_FEATURES change also wants
    # 04b/04d recalibration and a `--retune-only 8.5`. Left out deliberately so
    # it cannot ride along untested on the next retune and silently trigger that
    # recalibration.
    # When it IS tested, watch for the SAME failure mode that got
    # travel_distance_mi dropped below: capacity correlates with program
    # strength, so a POSITIVE coefficient on offensive production means it is
    # proxying "played a good team / big game" rather than "hostile crowd hurt
    # performance." The causal prior is NEGATIVE on the away side (crowd noise →
    # false starts, communication breakdowns), strongest for QB/OL (audible and
    # snap-count dependence) and K (pressure kicks). Positive sign on offensive
    # heads → drop it rather than rationalize it.
    # NOTE: travel_distance_mi dropped 2026-05-18. AL kept it on 10/12 active
    # heads, but RB_rush_yards β=+6.78 / RB_rec_yards β=+7.98 (POSITIVE on
    # rushing yards) is backward from the fatigue hypothesis. Likely
    # team-strength proxy ("good teams travel further to bowl games" or
    # "important games where RBs run more"). Same "kept everywhere without
    # causal-sign agreement" failure mode as team_rolling_*. Helper code
    # (_haversine_miles, _compute_school_home_venue lat/lon return) stays
    # alive in pipeline/01_data_ingestion.py for future re-enable as either
    # opponent-side or demean'd variant.
    # Season-level NCAA production environment. Pruned 2026-05-18 from 13 → 3
    # features after retention audit showed many lg_* are redundant year-effect
    # proxies (lg_off_ppa / _success_rate / _explosiveness all describe the
    # same era trends). Kept the three with highest information density and
    # cleanest interpretation: overall offensive / defensive efficiency and
    # league pass-rate. The other 10 (off_ppa, off_explosiveness, off_stuff,
    # def_ppa, def_stuff, pass_att, rush_att, pass_tds, rush_tds, interceptions)
    # are still computed in cfb_season_averages.parquet — re-add to this list
    # to restore.
    "lg_off_success_rate",
    "lg_def_success_rate",
    "lg_pass_rate",
    # Teammate-QB rolling quality (decay-weighted prior-game QB rates at the
    # same school). Activated only on WR/TE/RB heads — see
    # CONTEXT_FEATURE_DROP_BY_POSITION below, which strips these from QB
    # (self-referential) and OL/D/K/P (no defensible signal).
    "qb_rolling_passer_rating",
    "qb_rolling_completion_pct",
    "qb_rolling_ppa_avg_passing",
    "qb_rolling_ypa",
    "qb_rolling_td_rate",
    "qb_rolling_int_rate",
    # NOTE: team_rolling_* (pace, pass_rate, OL run-block four-factor) was
    # built 2026-05-16 and intentionally NOT enabled here. AdaptiveLasso at
    # step 4 retained the features in 29/33 skill-position heads, but step 8
    # cv_mae A/B (no-tune) showed net +1.94% regression even after gating
    # QB off; only RB_role (-1.64%) and global_role (-5.33%) clearly won.
    # The scraper / step 1 wiring / tests stay in place for future
    # reactivation — flip via re-adding to this list + populating
    # CONTEXT_FEATURE_DROP_BY_POSITION.

    # Opponent defense ALLOWED — rolling EWMA of stats their defense has given
    # up, derived from team_box_scores via inverse-perspective (see
    # OPP_DEF_ALLOWED_RAW_STATS below). Adds run-defense detail and
    # standard/passing-downs splits that the existing opp_rolling_* features
    # (advanced_game_stats) can't provide. Gated to offensive skill positions
    # via CONTEXT_FEATURE_DROP_BY_POSITION (run-def → QB/RB; pass-def → all
    # offensive skill).
    "opp_rolling_def_rushing_stuff_rate",
    "opp_rolling_def_rushing_power_success",
    "opp_rolling_def_line_yards_avg",
    "opp_rolling_def_second_level_yards_avg",
    "opp_rolling_def_open_field_yards_avg",
    "opp_rolling_def_success_rate_passing_downs",
    "opp_rolling_def_success_rate_standard_downs",
    "opp_rolling_def_points_per_opportunity",
    # Own-team rolling scoring environment from team_box_scores. Gated to
    # offensive skill positions + K (scoring_opps drives K xp_att / fg_att).
    # NOTE: team_rolling_points_per_opportunity is computed but deliberately
    # NOT included as a context feature — it's a result, not context. The
    # numerator (TDs and FGs per red-zone trip) IS what we're trying to
    # predict for pass_td / rush_td / rec_td / fg_made; including it would
    # partial out the player's own contribution.
    "team_rolling_scoring_opportunities",
    "team_rolling_avg_starting_field_pos",
    # NOTE: Three defender-side feature families were enabled 2026-05-17
    # then partial-reverted same day after retune A/B regressed (+3.51%
    # mean cv_mae, 12W/17L, global_role +52%). Defenders broadly hurt
    # (DB_role +19.8%, LB_role +7.4%); pre-2018 NaN on def_rolling_* added
    # dilution. The compute functions and step-1 wiring stay in place; only
    # the CONTEXT_FEATURES entries are removed:
    #   - opp_rolling_off_pass_att, _rush_att, _plays (opp volume)
    #   - opp_qb_rolling_* (6 cols)
    #   - def_rolling_pressure_rate, def_rolling_coverage_rate
    # See data/processed/draft_models.bak_pre_tier_a_retune_20260517/ for
    # the pre-retune state.
]

CFB_SEASON_AVG_FEATURES: list[str] = [
    "lg_off_ppa",
    "lg_off_success_rate",
    "lg_off_explosiveness",
    "lg_off_stuff_rate",
    "lg_def_ppa",
    "lg_def_success_rate",
    "lg_def_stuff_rate",
    "lg_pass_rate",
    "lg_pass_att",
    "lg_rush_att",
    "lg_pass_tds",
    "lg_rush_tds",
    "lg_interceptions",
]

# qb_rolling_* features are only defensible for WR/TE/RB — drop everywhere
# else. Self-referential for QB; no clean causal channel for OL/D/K/P. The
# list is materialized once here to avoid forward-referencing
# `QB_ROLLING_FEATURES` (defined below) in the dict literal.
_QB_ROLLING_FEATURE_NAMES: list[str] = [
    "qb_rolling_passer_rating",
    "qb_rolling_completion_pct",
    "qb_rolling_ppa_avg_passing",
    "qb_rolling_ypa",
    "qb_rolling_td_rate",
    "qb_rolling_int_rate",
]

# Opponent defense-allowed features (added 2026-05-17). Two subsets gated
# differently:
#   - run-defense detail: only QB/RB face this on the ground; drop everywhere
#     else (WR/TE pass-catchers don't run, K/P/OL no rushing stats, defenders
#     don't face opp defense).
#   - pass-defense + red-zone-D: applies to QB/RB/WR/TE (full skill-pos);
#     drop from K (kicker volume captured by team_rolling scoring env, not
#     opponent pass D) and from defenders / P / OL.
_OPP_RUN_DEF_ALLOWED: list[str] = [
    "opp_rolling_def_rushing_stuff_rate",
    "opp_rolling_def_rushing_power_success",
    "opp_rolling_def_line_yards_avg",
    "opp_rolling_def_second_level_yards_avg",
    "opp_rolling_def_open_field_yards_avg",
]
_OPP_PASS_DEF_ALLOWED: list[str] = [
    "opp_rolling_def_success_rate_passing_downs",
    "opp_rolling_def_success_rate_standard_downs",
    "opp_rolling_def_points_per_opportunity",
]
_OPP_DEF_ALL: list[str] = [*_OPP_RUN_DEF_ALLOWED, *_OPP_PASS_DEF_ALLOWED]

# Own-team scoring env (added 2026-05-17, partial-reverted same day):
# scoring_opportunities + avg_starting_field_pos kept for K only — drove
# K_impact −7.9% in retune. Drop from everyone else.
# (team_rolling_points_per_opportunity is computed but excluded from
# CONTEXT_FEATURES — downstream of the TD/FG production we're predicting.)
_TEAM_SCORING_ENV: list[str] = [
    "team_rolling_scoring_opportunities",
    "team_rolling_avg_starting_field_pos",
]

# Helper lists for partial-reverted defender features (opp volume, opp-QB
# rolling, def teammate rolling). The features are no longer in
# CONTEXT_FEATURES so these are unreferenced; keeping the empty lists
# documents what was tried.
# Originally: _OPP_OFF_VOLUME = opp_rolling_off_pass_att/_rush_att/_plays
#             _OPP_QB_ROLLING = opp_qb_rolling_passer_rating/_completion_pct/...
#             _DEF_ROLLING_ALL = def_rolling_pressure_rate/_coverage_rate
# See data/processed/draft_models.bak_pre_tier_a_retune_20260517/ for state
# before the retune that triggered this revert.

# Per-position-group features to exclude from the contextual model fit. The
# pool defaults to CONTEXT_FEATURES; entries here are removed before the fit.
CONTEXT_FEATURE_DROP_BY_POSITION: dict[str, list[str]] = {
    # K: all opp_rolling_* drop. Most are four-factor breakdowns
    # (success/stuff/explosiveness/havoc) with no clean causal channel for
    # kicker volume. opp_def_ppa was previously kept as a composite
    # "how good is the opponent overall" signal, BUT 2026-05-18 retention
    # audit showed K_kicking_pts retained it with max |β|=21 via the
    # indirect chain "opp def strong → team scores less → fewer xp attempts".
    # Team scoring env (scoring_opportunities + avg_starting_field_pos) is
    # the direct signal and stays in for K (xp_att / fg_att depend on it).
    # All opp_rolling_def_* (run + pass) drop for K (kicker volume is driven
    # by own scoring environment, not opponent defense detail).
    "K": [
        "opp_rolling_def_ppa",        # added 2026-05-18 — indirect channel only
        "opp_rolling_off_ppa",
        "opp_rolling_def_success_rate", "opp_rolling_def_stuff_rate",
        "opp_rolling_off_success_rate", "opp_rolling_off_explosiveness",
        "opp_rolling_def_explosiveness", "opp_rolling_havoc_total",
        "opp_rolling_havoc_front_seven", "opp_rolling_havoc_db",
        *_QB_ROLLING_FEATURE_NAMES,
        *_OPP_DEF_ALL,
    ],
    "QB": [
        *_QB_ROLLING_FEATURE_NAMES,   # self-feature
        # Partial revert (2026-05-17): keep opp_rolling_def_* (run + pass-def)
        # which had wins on QB_impact / QB_impact_wav, but drop both
        # team_rolling_* — scoring_opps is QB self-reference, and
        # avg_start_field_pos didn't carry standalone signal in the A/B
        # (gated to K only now).
        *_TEAM_SCORING_ENV,
    ],
    "OL": [
        *_QB_ROLLING_FEATURE_NAMES,
        *_OPP_DEF_ALL,
        *_TEAM_SCORING_ENV,
        # OL has no per-game stats (MODELED_STATS["OL"] is empty), so venue /
        # body-clock features are inert here — included for completeness /
        # belt-and-suspenders. (travel_distance_mi dropped from CONTEXT_FEATURES
        # entirely 2026-05-18 — no need to list here.)
        "altitude_diff_from_home_ft", "surface_diff_from_home",
        # capacity_diff_from_home listed for when it is enabled (it is built but
        # withheld from CONTEXT_FEATURES — see the note there). Harmless while
        # absent: the drop lists are filtered against the active feature set.
        "capacity_diff_from_home",
        "body_clock_hour",
    ],
    # Defenders KEEP qb_rolling_* drop (teammate-QB is offensive context)
    # and KEEP _OPP_DEF_ALL drop (opponent defense isn't relevant for a
    # defender). Defender-only features (opp volume, opp-QB rolling, def
    # teammate rolling) were partial-reverted on 2026-05-17 — gone from
    # CONTEXT_FEATURES so no drops needed here.
    "DL": [
        *_QB_ROLLING_FEATURE_NAMES,
        *_OPP_DEF_ALL,
        *_TEAM_SCORING_ENV,
    ],
    "EDGE": [   # mirrors DL
        *_QB_ROLLING_FEATURE_NAMES,
        *_OPP_DEF_ALL,
        *_TEAM_SCORING_ENV,
    ],
    "LB": [
        *_QB_ROLLING_FEATURE_NAMES,
        *_OPP_DEF_ALL,
        *_TEAM_SCORING_ENV,
    ],
    "DB": [
        *_QB_ROLLING_FEATURE_NAMES,
        *_OPP_DEF_ALL,
        *_TEAM_SCORING_ENV,
    ],
    "P":  [
        *_QB_ROLLING_FEATURE_NAMES,
        *_OPP_DEF_ALL,
        *_TEAM_SCORING_ENV,
        # opp_rolling_def_ppa dropped 2026-05-18 — punter performance has
        # no direct causal channel through opp defensive efficiency;
        # retention on P heads was indirect via team-scoring chain.
        "opp_rolling_def_ppa",
    ],
    "WR": [
        # WR keeps opp_rolling_def_* (run + pass-def) — WR_role −11% in the
        # retune A/B made this the biggest Tier-A win. Drops team_rolling_*
        # (scoring_opps is self-reference channel; avg_start_field_pos gated
        # to K only after partial revert).
        *_TEAM_SCORING_ENV,
    ],
    "TE": [
        # Same logic as WR — TE_impact +9% loss on team_rolling features
        # in the retune, but the run-def features delivered.
        *_TEAM_SCORING_ENV,
    ],
    "RB": [
        # RB keeps opp_rolling_def_* (run + pass-def) which drove the
        # RB_impact_wav −8.2% / RB_impact −5.8% / RB_role −3.7% wins.
        # Drops team_rolling_* (K-only after partial revert).
        *_TEAM_SCORING_ENV,
    ],
    # K is the ONLY position that keeps team_rolling_scoring_opportunities
    # + avg_starting_field_pos (no _TEAM_SCORING_ENV in its drop list).
    # Justification: scoring_opps directly drives xp_att / fg_att; start
    # field position drives long-FG attempts. K_impact −7.9% in retune.
}

# ── Kalman Filter Parameters ───────────────────────────────────────────────────
# Game-by-game (day-based process noise), analogous to cdarko

KALMAN_PRIOR_MEAN = 0.0         # Prior: league-average-caliber player
KALMAN_PRIOR_VAR = 2.0          # Prior variance (uncertainty for new players)

# Per-stat prior variances (computed via method-of-moments in Step 6; these are
# reasonable defaults before calibration)
KALMAN_PRIOR_VAR_BY_STAT: dict[str, float] = {
    # Counting stats are per-100-snaps, so scale is ~0–50 range → variance ~25
    "pass_yards":      25.0,
    "rush_yards":      10.0,
    "rec_yards":       10.0,
    "pass_att":         8.0,
    "rush_att":         6.0,
    "rec":              4.0,
    "pass_tds":         0.5,
    "rush_tds":         0.3,
    "rec_tds":          0.3,
    "tackles":          3.0,
    "sacks":            0.5,
    "tfl":              0.5,
    "ints":             0.3,
    "passes_defended":  1.0,
    "ff":               0.2,
    # Rate stats (0–1 or small values)
    "completion_pct":   0.02,
    "passer_rating":    5.0,
    "yards_per_carry":  0.5,
    "yards_per_rec":    1.0,
    "catch_pct":        0.05,
    "fg_pct":           0.05,
    "xp_pct":           0.02,
    "punt_yards":       5.0,
    "punts":            1.0,
    "fg_att":           0.5,
    "fg_made":          0.5,
    "xp_att":           1.0,
    "xp_made":          1.0,
    "net_punt_yards":   5.0,
    "sacks_taken":      0.5,
    "interceptions":    0.3,
    "fumbles":          0.3,
    "qb_hurries":       0.5,
    # Share stats — prior var ≈ inter-player spread in role prominence
    "rec_share":        0.015,
    "rush_share":       0.015,
    "pass_share":       0.015,
}

# Process noise per day (random walk — analogous to cdarko's KALMAN_PROCESS_NOISE_PER_DAY=0.001)
# Football skill changes slowly → smaller than basketball
KALMAN_PROCESS_NOISE_PER_DAY = 0.0001

# Two-rate process noise: off-season gaps (>60 days between games) accumulate at lower rate
# since skill changes less without games to reveal it
KALMAN_OFFSEASON_GAP_DAYS: float = 60.0
KALMAN_PROCESS_NOISE_OFFSEASON_MULTIPLIER: float = 0.3

KALMAN_PROCESS_NOISE_PER_DAY_BY_STAT: dict[str, float] = {
    "pass_yards":      0.00020,
    "rush_yards":      0.00012,
    "rec_yards":       0.00012,
    "pass_att":        0.00010,
    "rush_att":        0.00008,
    "rec":             0.00006,
    "pass_tds":        0.000015,
    "rush_tds":        0.000010,
    "rec_tds":         0.000010,
    "tackles":         0.00012,
    "sacks":           0.000020,
    "tfl":             0.000020,
    "ints":            0.000010,
    "passes_defended": 0.000030,
    # Rate stats (very small — percentage points per day)
    "completion_pct":  0.0000005,
    "passer_rating":   0.00003,
    "yards_per_carry": 0.000005,
    "yards_per_rec":   0.000008,
    "fg_pct":          0.000005,
    "xp_pct":          0.000002,
    # Share stats — slow-moving (role changes gradually over a career)
    "rec_share":       0.000003,
    "rush_share":      0.000003,
    "pass_share":      0.000003,
}

# Multiplier applied to observation noise for pre-2016 games (proxy snap counts only)
# eff_r = obs_noise * (REF_SNAPS / proxy_snap_val) where proxy_snap_val = REF_SNAPS / multiplier
KALMAN_PRE2016_NOISE_MULTIPLIER: float = 5.0

# Observation noise (per-game variance at reference snap count REF_SNAPS)
KALMAN_OBS_NOISE = 2.0

KALMAN_OBS_NOISE_BY_STAT: dict[str, float] = {
    "pass_yards":      15.0,
    "rush_yards":       8.0,
    "rec_yards":        8.0,
    "pass_att":         6.0,
    "rush_att":         4.0,
    "rec":              3.0,
    "tackles":          4.0,
    "completion_pct":   0.015,
    "passer_rating":    5.0,
    "yards_per_carry":  0.5,
    "yards_per_rec":    1.0,
    "fg_pct":           0.04,
    "xp_pct":           0.02,
    # Share stats (bounded [0, 1]; obs noise reflects game-to-game variation in role)
    "rec_share":        0.04,
    "rush_share":       0.04,
    "pass_share":       0.05,
}

# Reference snap count: a "full game" worth of snaps (analogous to cdarko's KALMAN_REF_POSS=70)
KALMAN_REF_SNAPS: float = 70.0

# Rate stats: use attempt-count weighting in Kalman/EWMA instead of snap count
RATE_STAT_ATTEMPT_COL: dict[str, str] = {
    "completion_pct":  "pass_att",
    "passer_rating":   "pass_att",
    "yards_per_carry": "rush_att",
    "yards_per_rec":   "rec",
    "catch_pct":       "rec",
    "fg_pct":          "fg_att",
    "xp_pct":          "xp_att",
    "net_punt_yards":  "punts",
    # PPA stats are per-play averages — weight by attempt count, not snap count
    "ppa_avg_passing": "pass_att",
    "ppa_avg_rushing": "rush_att",
    "ppa_avg":         "rec",
    # Share stats — weight by team totals (denominator of the share)
    "rec_share":  "team_rec",
    "rush_share": "team_rush_att",
    "pass_share": "team_pass_att",
}

# Reference attempt counts for noise scaling (analogous to cdarko's KALMAN_REF_ATTEMPTS)
KALMAN_REF_ATTEMPTS: dict[str, float] = {
    "completion_pct":  30.0,
    "passer_rating":   30.0,
    "yards_per_carry": 15.0,
    "yards_per_rec":    5.0,
    "catch_pct":        5.0,
    "fg_pct":           3.0,
    "xp_pct":           3.0,
    "net_punt_yards":   5.0,
    "ppa_avg_passing": 30.0,
    "ppa_avg_rushing": 15.0,
    "ppa_avg":          5.0,
    # Share stat reference denominators (typical team totals per game)
    "rec_share":        25.0,
    "rush_share":       32.0,
    "pass_share":       35.0,
}

# Counting stats that get per-100-snap normalization in gamelogs
SNAP_COUNTING_STATS: frozenset[str] = frozenset({
    "pass_att", "pass_yards", "pass_tds", "interceptions", "rush_att",
    "rush_yards", "rush_tds", "sacks_taken", "rec", "rec_yards", "rec_tds",
    "fumbles", "tackles", "sacks", "tfl", "ff", "passes_defended", "ints",
    "fg_att", "fg_made", "xp_att", "xp_made", "punt_yards", "punts",
    "qb_hurries", "kicking_pts",
})

# Snap proxy columns by position group (used when actual snaps are unavailable)
SNAP_PROXY_BY_POSITION: dict[str, list[str]] = {
    "QB":    ["pass_att", "rush_att"],
    "RB":    ["rush_att", "rec"],
    "WR":    ["rec"],
    "TE":    ["rec"],
    "DL":    ["tackles", "sacks", "tfl", "passes_defended"],
    "EDGE":  ["tackles", "sacks", "tfl", "passes_defended"],
    "LB":    ["tackles", "sacks", "tfl", "passes_defended"],
    "DB":    ["tackles", "passes_defended"],
    # K/P intentionally empty — see CONSTANT_SNAP_POSITIONS below. Their own
    # volume stats cannot serve as the denominator (they are the numerators).
    "K":     [],
    "P":     [],
    "OL":    [],
}

# Position groups that get a constant per-game snap denominator instead of a
# stat-derived proxy. Specialists are on the field for essentially every
# scoring/punting opportunity, so there is no playing-time variation to
# normalize away — the per-game count IS the signal.
#
# Using their own counting stats as the proxy (K→fg_att, P→punts) was
# self-referential and destroyed that signal: punts_p100 collapsed to exactly
# 100.0 for every punter-game (numerator == denominator), which drove
# adjusted_punts to a constant 0.0 and left kalman_mean_punts / ewma_punts
# with no variance at all. On the kicker side fg_made_p100 silently became FG
# accuracy — already modeled properly as `fg_pct` via RATE_STAT_ATTEMPT_COL —
# while xp_made_p100 inflated up to 100x on zero-FGA games, where the
# clip(lower=1) left a single "snap" in the denominator.
#
# The constant is KALMAN_REF_SNAPS so the downstream observation weight
# (snaps / KALMAN_REF_SNAPS in decay.py / kalman.py) lands at exactly 1.0 —
# every specialist game counts equally, which is the intended semantics.
CONSTANT_SNAP_POSITIONS: frozenset[str] = frozenset({"K", "P"})

# Transfer: boost uncertainty
TRANSFER_P_MULTIPLIER = 2.5

# Innovation clipping (sigma): clip extreme single-game surprises
KALMAN_INNOVATION_CLIP_SIGMA = 3.0

# Age-based process noise scaling
AGE_Q_REFERENCE_AGE = 22.0
AGE_Q_HALFLIFE_YEARS = 2.5      # football development is slower than basketball
AGE_Q_MAX_MULTIPLIER = 3.0

# ── EWMA Decay Parameters ──────────────────────────────────────────────────────
# Beta = per-day decay (applied to days elapsed, not seasons)
# beta=0.999 → half-life ≈ 693 days ≈ 2 seasons
# beta=0.995 → half-life ≈ 139 days ≈ one season

DEFAULT_BETA: dict[str, float] = {s: 0.999 for s in [
    "pass_att", "pass_yards", "pass_tds", "interceptions", "completion_pct",
    "passer_rating", "rush_att", "rush_yards", "rush_tds", "yards_per_carry",
    "sacks_taken", "rec", "rec_yards", "rec_tds", "yards_per_rec", "catch_pct",
    "fumbles", "tackles", "sacks", "tfl", "ff", "passes_defended", "ints",
    "qb_hurries", "fg_att", "fg_made", "fg_pct", "xp_att", "xp_made", "xp_pct",
    "kicking_pts", "punt_yards", "net_punt_yards", "punts",
    "ppa_avg_passing", "ppa_avg_rushing", "ppa_avg",
    "rec_share", "rush_share", "pass_share",
]}
BETA_BOUNDS = (0.990, 1.000)   # per-day (same as cdarko)

# ── Rolling Opponent Strength ──────────────────────────────────────────────────

# Halflife in days for exponential decay weighting of opponent's prior games
OPP_ROLLING_HALFLIFE_DAYS: int = 45     # tuned from data (grid search)

# All rolling opponent features (composite stats are de-meaned; four-factor components are raw)
OPP_ROLLING_FEATURES: list[str] = [
    "opp_rolling_def_ppa",        # composite — de-meaned
    "opp_rolling_off_ppa",        # composite — de-meaned
    "opp_rolling_def_success_rate",
    "opp_rolling_def_stuff_rate",
    "opp_rolling_off_success_rate",
    "opp_rolling_off_explosiveness",
    "opp_rolling_def_explosiveness",
    "opp_rolling_havoc_total",
    "opp_rolling_havoc_front_seven",
    "opp_rolling_havoc_db",
]

OPP_ROLLING_DEMEAN_COLS: list[str] = [
    "opp_rolling_def_ppa",
    "opp_rolling_off_ppa",
    # opp_rolling_off_* four-factor components added 2026-05-18. These are
    # kept on offensive heads via indirect game-script channel (opp has
    # high-octane offense → my team falls behind → more pass volume).
    # Demeaning strips team-strength identity baseline while preserving the
    # within-season variance — captures the legitimate "this opponent's
    # offense was unusually strong today" signal without letting AL fit on
    # absolute opp-team identity. Defense-side four-factor components stay
    # raw (no analogous game-script chain on defenders facing offensive).
    "opp_rolling_off_success_rate",
    "opp_rolling_off_explosiveness",
]

# ── Teammate-QB Rolling Quality ───────────────────────────────────────────────
# For each offensive-skill game (WR/TE/RB), the team's QBs' decay-weighted
# rate stats over their strictly-prior games at the SAME school. Residualizes
# receiver/RB stats for the quality of the QB feeding them. Mirrors the
# opponent-rolling pattern (same halflife default, strictly prior, exp decay)
# but adds an exposure weight: each prior QB-game contributes
# sqrt(pass_att) — caps single-blowout influence vs linear weighting.

QB_ROLLING_HALFLIFE_DAYS: int = 45      # match OPP_ROLLING_HALFLIFE_DAYS
QB_ROLLING_MIN_PASS_ATT: int = 5        # drop garbage-time / snow-game QB rows
                                        # from the prior panel — sub-5-att rows
                                        # are noise, not skill signal

QB_ROLLING_FEATURES: list[str] = [
    "qb_rolling_passer_rating",          # NCAA passer rating composite
    "qb_rolling_completion_pct",         # share of pass_att completed
    "qb_rolling_ppa_avg_passing",        # CFBD predicted-points-added per pass
    "qb_rolling_ypa",                    # pass_yards / pass_att
    "qb_rolling_td_rate",                # pass_tds / pass_att
    "qb_rolling_int_rate",               # interceptions / pass_att (negative)
]

# Composite/tempo confounders are de-meaned within (player, season) —
# mirrors OPP_ROLLING_DEMEAN_COLS. Atomic rates (cmp%, ypa, td_rate, int_rate)
# stay raw because they carry within-season signal (a backup who throws
# poorly mid-season should depress the WR's environment value).
QB_ROLLING_DEMEAN_COLS: list[str] = [
    "qb_rolling_passer_rating",
    "qb_rolling_ppa_avg_passing",
]

# Per-stat sign for the career _peak triplet (step 8). For positively-signed
# stats, "peak" = max season-mean; for negatives (int_rate), peak = min.
QB_PEAK_DIRECTION: dict[str, str] = {
    "qb_rolling_passer_rating":    "max",
    "qb_rolling_completion_pct":   "max",
    "qb_rolling_ppa_avg_passing":  "max",
    "qb_rolling_ypa":              "max",
    "qb_rolling_td_rate":          "max",
    "qb_rolling_int_rate":         "min",
}

# Subset of QB_ROLLING_FEATURES that gets the career mean/peak/last triplet
# treatment in step 8. The Boruta audit on 2026-05-16 (initial fit with all 6
# bases → 18 cols × 9 WR/TE/RB heads = 162 acceptance slots) confirmed only 3:
#   RB/impact: qb_rolling_passer_rating_peak
#   RB/role:   qb_rolling_passer_rating_last
#   WR/role:   qb_rolling_passer_rating_mean
# All 3 winners are passer_rating variants; the 5 atomic-rate stats
# (cmp%, ypa, td_rate, int_rate, ppa) were rejected on every head. Slimming
# to passer_rating only reduces the career feature surface from 18 → 3 cols
# (6× reduction), which matters most for small-n heads (TE has ~120 bridge
# rows; 18 dilutional cols per head hurts).
QB_CAREER_BASE_FEATURES: list[str] = ["qb_rolling_passer_rating"]

# Position groups that consume teammate-QB career features in step 8.
# Drop QB (self-feature) and OL/DL/LB/DB/K/P (no defensible signal).
QB_CAREER_POSITION_GROUPS: frozenset = frozenset({"WR", "TE", "RB"})

# Master kill-switch for the career-grain features (game-grain residualization
# is gated by `QB_ROLLING_FEATURES`; setting that list to `[]` and rerunning
# steps 1/4/5 reverts cleanly). Mirrors USE_PFF_CONTEXTUAL pattern.
# Flipped to False 2026-05-16 for option-1 A/B: isolates whether the career
# triplets help on top of the game-grain residualization.
USE_QB_CAREER_FEATURES: bool = False

# Gate on the four REVERTED context-feature families still wired in step 1:
# team_rolling_* (pace/pass_rate/OL run-block), def_rolling_* (pressure /
# coverage), opp_rolling_off_* volume, and opp_qb_rolling_*. Their columns
# were removed from CONTEXT_FEATURES (team-rolling reverted 2026-05-16;
# defender-side trio partial-reverted 2026-05-17 — see the CONTEXT_FEATURES
# note), so computing them every step-1 run is pure waste. Reverted feature
# families kept for reactivation; flip to recompute + see git history for the
# revert dates. When False, step 1 omits the columns entirely — downstream
# consumers default them to NaN (src/models/contextual.py).
COMPUTE_REVERTED_CONTEXT: bool = False

# ── Team Rolling Context (own-team pace, pass rate, OL run-block) ─────────────
# For each (game_id, school), exponential-decay-weighted EWMA over the team's
# strictly-prior games of:
#   - pace (plays/game = team_pass_att + team_rush_att)
#   - pass_rate (team_pass_att / plays — RAW, contaminated by game-script; a
#     PBP-based neutral-script variant is the eventual upgrade)
#   - OL run-block four-factor (line_yards_avg, second_level_yards_avg,
#     open_field_yards_avg, rushing_power_success) from team_box_scores
# Exposure weight = sqrt(plays). Pace/pass_rate gate to skill positions
# (QB/RB/WR/TE); OL run-block gates to RB-only — see
# CONTEXT_FEATURE_DROP_BY_POSITION above.

TEAM_ROLLING_HALFLIFE_DAYS: int = 45     # match OPP/QB rolling halflife
TEAM_ROLLING_MIN_PLAYS: int = 30          # drop weather/forfeit games whose
                                          # play counts aren't representative
                                          # team-style signal

TEAM_PACE_FEATURES: list[str] = [
    "team_rolling_pace",                  # plays per game (volume)
    "team_rolling_pass_rate",             # share of plays that are pass att
]

OL_RUN_BLOCK_FEATURES: list[str] = [
    "team_rolling_line_yards_avg",        # OL yards charged to line
    "team_rolling_second_level_yards_avg",
    "team_rolling_open_field_yards_avg",
    "team_rolling_rushing_power_success", # short-yardage / goal-line
]

TEAM_ROLLING_FEATURES: list[str] = [
    *TEAM_PACE_FEATURES,
    *OL_RUN_BLOCK_FEATURES,
]

# Composites that get within-(player, season) demeaning (schedule-strength
# temporal confound — same pattern as OPP_ROLLING_DEMEAN_COLS). The four-factor
# OL components stay raw; pace/pass_rate are scheme signals we want raw too
# (Air Raid teams persistently high, triple-option persistently low — the
# EWMA already smooths within-season variance).
TEAM_ROLLING_DEMEAN_COLS: list[str] = []

# ── Opponent Defense Allowed (rolling, derived from team_box_scores) ──────────
# For each (game_id, school), rolling EWMA of stats the OPPONENT's defense
# has allowed over their strictly-prior games. Source is team_box_scores
# offensive rows produced AGAINST that defense. Gives run-defense /
# pass-defense detail that the existing opp_rolling_* (advanced_game_stats)
# can't split (def_stuff_rate is overall, not run-only; def_success_rate is
# overall, not standard/passing-downs split).

OPP_DEF_ALLOWED_HALFLIFE_DAYS: int = 45      # match OPP_ROLLING_HALFLIFE_DAYS

# Raw stats pulled from team_box_scores. The compute function prefixes each
# with "opp_rolling_def_" → e.g. rushing_stuff_rate → opp_rolling_def_rushing_stuff_rate.
OPP_DEF_ALLOWED_RAW_STATS: list[str] = [
    # Run defense — four-factor split (existing opp_rolling_def_stuff_rate
    # from advanced_game_stats is rush+pass combined)
    "rushing_stuff_rate",
    "rushing_power_success",
    "line_yards_avg",
    "second_level_yards_avg",
    "open_field_yards_avg",
    # Pass defense — down-state splits (existing opp_rolling_def_success_rate
    # is overall, not standard/passing-downs split)
    "success_rate_passing_downs",
    "success_rate_standard_downs",
    # Red-zone defense
    "points_per_opportunity",
]

OPP_DEF_ALLOWED_FEATURES: list[str] = [
    f"opp_rolling_def_{s}" for s in OPP_DEF_ALLOWED_RAW_STATS
]

# All atomic rate features — none get demeaned (mirrors OPP_ROLLING four-factor
# components which also stay raw; only the composite PPA cols there get
# demeaned).
OPP_DEF_ALLOWED_DEMEAN_COLS: list[str] = []

# ── Team Rolling Scoring Environment (own offense, from team_box_scores) ──────
# For each (game_id, school), rolling EWMA of own-offense scoring efficiency
# stats. Affects K (scoring_opportunities → xp_att / fg_att), RB (rush_td
# opportunities), QB/WR/TE (red-zone targets). Held separate from
# TEAM_ROLLING_FEATURES so it can be reverted without disturbing the previously
# rejected pace/pass_rate/OL pipeline.

TEAM_SCORING_ENV_RAW_STATS: list[str] = [
    "scoring_opportunities",       # # red-zone trips per game (volume)
    "points_per_opportunity",      # red-zone scoring efficiency
    "avg_starting_field_pos",      # where own drives start (yard line)
]

TEAM_SCORING_ENV_FEATURES: list[str] = [
    f"team_rolling_{s}" for s in TEAM_SCORING_ENV_RAW_STATS
]

# Scoring-env stats are own-team scheme signals — keep raw (no demean).
TEAM_SCORING_ENV_DEMEAN_COLS: list[str] = []

# ── Opponent Offense Volume (rolling, from gamelogs) ──────────────────────────
# For each (game_id, school), rolling EWMA over opponent's strictly-prior games
# of opponent's offensive VOLUME (team_pass_att, team_rush_att, total plays).
# Aimed at defenders — gives opportunity-side signal that opponent-quality
# metrics can't supply.

OPP_OFF_VOLUME_HALFLIFE_DAYS: int = 45        # match OPP_ROLLING_HALFLIFE_DAYS
OPP_OFF_VOLUME_MIN_PLAYS: int = 30            # drop forfeit / heavy-weather

OPP_OFF_VOLUME_FEATURES: list[str] = [
    "opp_rolling_off_pass_att",
    "opp_rolling_off_rush_att",
    "opp_rolling_off_plays",
]

# Volume features stay raw — they're absolute counts, not composite rates.
OPP_OFF_VOLUME_DEMEAN_COLS: list[str] = []

# ── Opponent QB Rolling (mirror of teammate qb_rolling, joined on opponent) ───
# For each defender's game, the rolling rate stats of the OPPONENT's QB over
# the QB's strictly-prior games. Reuses compute_qb_rolling_stats output
# (per-school rolling); the ingestion helper merges on opponent and renames
# qb_rolling_* → opp_qb_rolling_*.

OPP_QB_ROLLING_FEATURES: list[str] = [
    "opp_qb_rolling_passer_rating",
    "opp_qb_rolling_completion_pct",
    "opp_qb_rolling_ppa_avg_passing",
    "opp_qb_rolling_ypa",
    "opp_qb_rolling_td_rate",
    "opp_qb_rolling_int_rate",
]

# Same demean policy as QB_ROLLING_DEMEAN_COLS — composite/tempo confounders
# get within-(player, season) demeaning; atomic rates stay raw.
OPP_QB_ROLLING_DEMEAN_COLS: list[str] = [
    "opp_qb_rolling_passer_rating",
    "opp_qb_rolling_ppa_avg_passing",
]

# ── Defensive Teammate Rolling (own-defense pressure / coverage) ──────────────
# For each (game_id, school), rolling EWMA over school's strictly-prior games of:
#   - def_rolling_pressure_rate  = (sacks + qb_hurries) / opp_pass_att
#   - def_rolling_coverage_rate  = (passes_defended + ints) / opp_pass_att
# Coverage caveat: CFBD defensive box stats begin ~2018; pre-2018 emit NaN.
# Position gating handles self-reference: DL drops pressure (own production),
# DB drops coverage (own production), LB gets both (partial self acceptable).

DEF_ROLLING_HALFLIFE_DAYS: int = 45            # match other rolling halflives
DEF_ROLLING_MIN_OPP_PASS_ATT: int = 10         # drop near-zero denominators
DEF_ROLLING_MIN_OPP_PLAYS: int = 30            # drop forfeit / weather games

DEF_ROLLING_FEATURES: list[str] = [
    "def_rolling_pressure_rate",
    "def_rolling_coverage_rate",
]

# Atomic rates — no within-(player, season) demean.
DEF_ROLLING_DEMEAN_COLS: list[str] = []

# ── School Time Zones ──────────────────────────────────────────────────────────
# Standard UTC offset for each FBS school's home location (non-DST).
# Used to compute tz_diff = home_team_tz - away_team_tz per game.
# Default: -6 (Central). Names must match CFBD API team names.

SCHOOL_TIMEZONE_OFFSET: dict[str, int] = {
    # Hawaii
    "Hawaii": -10,
    # Pacific (-8)
    "California": -8, "Stanford": -8, "UCLA": -8, "USC": -8,
    "Oregon": -8, "Oregon State": -8, "Washington": -8, "Washington State": -8,
    "San Diego State": -8, "Fresno State": -8, "San Jose State": -8,
    "Nevada": -8, "UNLV": -8,
    # Mountain (-7)
    "Arizona": -7, "Arizona State": -7,
    "Air Force": -7, "Boise State": -7, "BYU": -7,
    "Colorado": -7, "Colorado State": -7, "Wyoming": -7,
    "Utah": -7, "Utah State": -7,
    "New Mexico": -7, "New Mexico State": -7, "UTEP": -7,
    # Central (-6)
    "Alabama": -6, "Auburn": -6, "South Alabama": -6, "Troy": -6,
    "Arkansas": -6, "Arkansas State": -6,
    "Louisiana": -6, "Louisiana Monroe": -6, "Louisiana Tech": -6,
    "LSU": -6, "Tulane": -6,
    "Mississippi State": -6, "Ole Miss": -6, "Southern Miss": -6,
    "Iowa": -6, "Iowa State": -6,
    "Illinois": -6, "Northern Illinois": -6, "Northwestern": -6,
    "Wisconsin": -6, "Minnesota": -6, "Nebraska": -6,
    "Kansas": -6, "Kansas State": -6, "Missouri": -6,
    "Oklahoma": -6, "Oklahoma State": -6, "Tulsa": -6,
    "Texas": -6, "Texas A&M": -6, "Texas Tech": -6,
    "Baylor": -6, "TCU": -6, "Houston": -6, "Rice": -6, "SMU": -6,
    "UTSA": -6, "North Texas": -6, "Texas State": -6, "Sam Houston": -6,
    "Memphis": -6, "Vanderbilt": -6,
    "Jacksonville State": -6, "Western Kentucky": -6,
    "Middle Tennessee": -6,
    # Eastern (-5)
    "Boston College": -5, "Clemson": -5, "Duke": -5, "Florida State": -5,
    "Georgia Tech": -5, "Louisville": -5, "Miami": -5, "NC State": -5,
    "North Carolina": -5, "Pittsburgh": -5, "Syracuse": -5,
    "Virginia": -5, "Virginia Tech": -5, "Wake Forest": -5,
    "Indiana": -5, "Maryland": -5, "Michigan": -5, "Michigan State": -5,
    "Ohio State": -5, "Penn State": -5, "Purdue": -5, "Rutgers": -5,
    "Notre Dame": -5,
    "Florida": -5, "Florida Atlantic": -5, "FIU": -5, "UCF": -5, "South Florida": -5,
    "Georgia": -5, "Georgia Southern": -5, "Georgia State": -5,
    "Kentucky": -5, "South Carolina": -5, "Tennessee": -5,
    "Army": -5, "Navy": -5, "Connecticut": -5, "UConn": -5, "Temple": -5,
    "Akron": -5, "Ball State": -5, "Bowling Green": -5, "Buffalo": -5,
    "Central Michigan": -5, "Eastern Michigan": -5, "Kent State": -5,
    "Miami (OH)": -5, "Ohio": -5, "Toledo": -5, "Western Michigan": -5,
    "West Virginia": -5, "Marshall": -5,
    "Appalachian State": -5, "Charlotte": -5, "Coastal Carolina": -5,
    "East Carolina": -5, "Old Dominion": -5,
    "James Madison": -5, "Kennesaw State": -5, "Liberty": -5,
}

DEFAULT_TIMEZONE_OFFSET: int = -6  # Central — fallback for unknown schools

# ── Bridge Player Weighting ────────────────────────────────────────────────────

BRIDGE_RECENCY_HALFLIFE_YEARS = 5   # Bridge players from 5+ years ago get lower weight
BRIDGE_SEASON_GAMMA = 0.8           # Down-weight older seasons of bridge players

# ── NFL Label Parameters ───────────────────────────────────────────────────────

# Role: fraction of seasons in first NFL_LABEL_WINDOW_YEARS where player hit threshold
ROLE_SNAP_PCT_THRESHOLD = 0.20   # 20% snap share fallback (used when position group unknown)
ROLE_SNAP_GAMES_MIN = 4          # Minimum games in a season to count

# Position-specific snap thresholds — kept for trajectory model internals;
# no longer used to define role_target (now: binary second-contract label).
ROLE_SNAP_PCT_THRESHOLD_BY_GROUP: dict[str, float] = {
    "QB":    0.40,   # starter threshold; backup QBs rarely contribute
    "RB":    0.20,   # shared backfields common
    "WR":    0.20,   # rotational use is normal
    "TE":    0.20,
    "OL":    0.35,   # starters nearly every snap; backups rarely play
    "DL":    0.20,   # rotational role is standard
    "EDGE":  0.20,
    "LB":    0.25,
    "DB":    0.25,
    "K":     0.05,   # special teamers rarely approach 20%
    "P":     0.05,
}

# (Removed 2026-07-26: AV_ERA_REFERENCE_YEAR / MIN_NFL_SEASONS_FOR_IMPACT were
#  dead config — defined here, imported nowhere, and the "era-adjusted" claim
#  they carried was false: impact_w_av_per_game is raw career w_AV / games with
#  no era normalization anywhere in the path. Removed with the replacement-value
#  work rather than left as a misleading comment.)

# ── Replacement Value ─────────────────────────────────────────────────────────
#
# Two distinct replacement lines, for two distinct purposes:
#
#  1. LABEL layer (step 7) — a GLOBAL, position-independent, year-indexed line
#     anchored to the league minimum salary. Used to gate what counts as a
#     qualifying second contract and to rebase the impact target. Position
#     independence is deliberate: per-(position × year) label subtraction is the
#     regime the 2026-05-14 demean experiment failed in (0W/10L, +53.5% CV MAE),
#     and per-position level differences are re-learned downstream by the
#     per-position isotonic calibration in step 8c anyway.
#
#  2. OUTPUT layer (step 10) — a PER-POSITION line from current-era contracts.
#     This is where cross-position comparability lives: replacement-level QB
#     money is far above replacement-level RB money, and that gap is the whole
#     point of a VORP-style cross-position board.
#
# The minimum-salary line is DERIVED FROM THE CONTRACT DATA, not hardcoded from
# a CBA table. Rationale: the OTC feed contains the CBA minimum tiers as sharp
# modes (e.g. 2025 one-year non-rookie deals spike at exactly $0.840M and
# $1.17M), so extracting them is reproducible and self-consistent, whereas a
# hand-typed table is unverifiable and silently rots each March. The derivation
# lives in src/scrapers/nfl_data.py::derive_vet_min_apy_pct and is persisted to
# data/processed/replacement_levels.parquet with provenance.

# Minimum count for an APY value to count as a "sharp mode" (a CBA salary tier
# rather than a one-off negotiated number).
VET_MIN_MODE_MIN_COUNT: int = 20
# Floor (as a fraction of cap) below which a sharp mode is a PRORATED deal
# — practice-squad elevations and partial-season signings cluster around
# 0.0005–0.001 of cap. Full-season minimums sit at 0.0027–0.0045.
VET_MIN_PRORATION_FLOOR_PCT: float = 0.002
# Fallback for years too sparse to derive (OTC is thin before ~2013 — only 283
# non-rookie rows in 2011 vs ~3,500/yr now). The minimum-salary line is
# remarkably stable as a share of cap (0.0027–0.0037 across 2014–2026), so
# extrapolating backward by cap fraction is better motivated than reciting
# pre-CBA-2011 dollar figures.
VET_MIN_FALLBACK_PCT: float = 0.00304
# Multiplier on the derived minimum line to set the replacement threshold. The
# derived base is the LOWEST full-season tier (0 accrued seasons); a player
# signing a genuine second contract has 3-4 accrued seasons and so a higher
# minimum. This constant carries that adjustment, and is the knob the Stage-1
# A/B sensitivity arm varies (1.0 / 1.25 / 1.5).
REPLACEMENT_MARGIN: float = 1.25

# ── Output-layer (per-position) replacement ───────────────────────────────────
# Quantile of veteran multi-year contracts, per position group, over a recent
# dense window. p25 of real multi-year veteran deals ≈ "the price of a starter
# you can readily replace" at that position.
REPLACEMENT_POS_QUANTILE: float = 0.25
REPLACEMENT_CONTRACT_WINDOW_YEARS: int = 5
# Multi-year filter: 1-year deals are dominated by minimum/prove-it contracts
# and would collapse every position onto the same minimum line.
REPLACEMENT_POS_MIN_YEARS: int = 2
# Minimum contracts per position cohort before its own quantile is trusted;
# below this the global vet-min line is used instead.
REPLACEMENT_POS_MIN_N: int = 30

# OverTheCap position codes → cfDARKO position groups. LS (long snapper) has no
# cfDARKO group and is dropped.
OTC_POSITION_TO_GROUP: dict[str, str] = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "LT": "OL", "LG": "OL", "C": "OL", "RG": "OL", "RT": "OL",
    "IDL": "DL",
    "ED": "EDGE",
    "LB": "LB",
    "CB": "DB", "S": "DB",
    "K": "K",
    "P": "P",
}

# Franchise/transition tag heuristic. OTC carries no tag flag, so tags are
# identified structurally: a one-year deal that is essentially fully guaranteed
# at a high cap share. Without this, a tag pre-empts the real second contract
# under earliest-qualifying selection (Dak Prescott labels as his 0.158 tag
# instead of his 0.219 extension).
TAG_MAX_YEARS: int = 1
TAG_MIN_GUARANTEED_RATIO: float = 0.95
TAG_MIN_APY_CAP_PCT: float = 0.05

# ── Early-cohort contract coverage ────────────────────────────────────────────
#
# OverTheCap's coverage of SMALL contracts improved enormously over time, so the
# second-contract label does not mean the same thing in 2005 as in 2018.
# Measured against career w_AV (an independent PFR outcome), the share of bridge
# players labelled "no second contract" is:
#
#     career w_AV   2004-06   2007-10   2011-19
#     0-5             99.1%     90.4%     11.6%
#     5-15            92.6%     61.8%      9.3%
#     15-30           77.3%     25.3%      2.1%
#     30+             34.0%      2.0%      0.6%
#
# A third of 2004-06 players with w_AV >= 30 — clear multi-year starters — are
# labelled washouts. Those are coverage artifacts, not outcomes, and the impact
# heads were training on them.
#
# TESTED AND REJECTED (2026-07-26). Dropping or down-weighting these rows does
# NOT help — it hurts. Left here, disabled, because the mechanism is correct and
# the corruption above is real; only the remedy failed.
#
# The first reading of the A/B was an ARTIFACT and is worth understanding before
# anyone reopens this. Dropping pre-2011 bridge rows removes 100% of the bridge
# training population for the earliest expanding-window fold (train_end 2009)
# and 70% for the next. A magnitude head with no bridge rows sees only the
# zero-padded non-bridge universe, collapses to predicting ~0 — and because the
# target is mostly zeros, that posts a BETTER MAE while its ranking falls to
# chance. That single degenerate fold produced an apparent -13.98% mean CV MAE.
#
# Re-run with (a) starved folds excluded via _usable_folds and (b) every arm
# scored on the SAME fold intersection, the result inverts — all three arms are
# worse on the impact heads:
#
#     drop_pre2007        1/9 improve, mean +3.48%
#     drop_pre2011        4/9 improve, mean +2.94%
#     downweight_pre2011  4/9 improve, mean +2.66%
#
# Role arms regressed too (mean +3.6% to +5.1%). Conclusion: for these
# tree-ensemble heads the extra training volume is worth more than the label
# noise it carries, even when a third of the 2004-06 rows are demonstrably
# mislabelled. See experiments/cohort_ab_20260726_*.json — the LAST file is the
# honest one; earlier files carry the starved-fold artifact.
#
# Set to a year to re-enable (the plumbing in step 8 / 8b is intact and tested).
IMPACT_LABEL_MIN_DRAFT_YEAR: int | None = None

# Training weight for impact-head rows that carry a consensus board rank
# (hasan_rank non-null). Board rows are ~2.8% of the zero-padded training set,
# so splits on hasan_rank gain little and the head under-uses the board:
# hasan_rank adds +0.24 partial Spearman beyond the head's OOF prediction in
# every fold (experiments/00w). 1.0 = unweighted (production). REJECTED
# 2026-10-30 (experiments/00x): ×3/×10/×30 changed neither the head's use of
# the board nor any referee beyond the seed placebo. Not wired into tuning.
IMPACT_BOARD_ROW_WEIGHT: float = 1.0

# Window (years after draft) in which a second contract is looked for. Beyond
# this the deal is a third contract or a late-career journeyman signing, not
# the outcome the draft model is trying to predict.
SECOND_CONTRACT_MAX_YEARS_AFTER_DRAFT: int = 6

# Two-stage trajectory model parameters
TRAJECTORY_LONG_TERM_WINDOW = (5, 8)  # years post-draft to use as long-term target (legacy)
TRAJECTORY_TRAIN_CUTOFF_YEAR = 2016   # only draftees ≤ this year have stable long-term data (legacy)
TRAJECTORY_FEATURE_WINDOW = 4         # first N NFL seasons used as Stage 1 features (legacy)
TRAJECTORY_MAX_YEARS: int = 4         # max seasons of NFL data per training row
TRAJECTORY_SETTLED_CUTOFF_YEARS: int = 6   # draft_year + N ≤ current_year → player is settled
# Players drafted >= this may lack a 2nd contract yet. 2023 since the
# 2026-10-29 contract refresh: the 2022 class's >1%-cap-by-y5 rate (45.8%) is
# inside the 2015-21 range (38-55%), so its zeros are real.
TRAJECTORY_ELIGIBILITY_CUTOFF_YEAR: int = 2023
# Validation split for the impact trajectory model (imputes impact_target for
# un-settled classes with no contract yet). "legacy" infers draft year from
# n_years_observed, so validation = every 1-2 season row and the model never
# trains on them; "by_class" holds out real draft classes and refits on all
# rows. See experiments/00v_impact_trajectory_split.py.
#
# Backtest (2026-10-29; 5 seeds; predict settled classes from their first k
# seasons, scored vs the APY label and vs value_target): by_class wins at
# k=1-2 on the 2019-20 test (+0.06 to +0.09, 3.5-10 seed sd) but LOSES at
# k=3-4 (value referee −0.03 to −0.04). The legacy model is effectively a
# k>=3 specialist, since those are the only rows it trains on. A k<=2-only
# hybrid failed confirmation on the 2018 class: k=1 still won on the APY label
# (+0.07 to +0.09) but not clearly on the value referee (1.4-1.7 sd), k=2 tied,
# and k>=3 lost by 10 sd. Not shipped. No fresh window exists to test a
# k=1-only rule (snap data starts 2016, so only classes 2016-20 are usable).
TRAJECTORY_IMPACT_VAL_SPLIT: str = "legacy"
# Fixed horizon (NFL seasons) at which the trajectory model is evaluated to
# produce the canonical role_target for the draft model. y3 is pre-resolution
# even for early extensions like Jefferson/Burrow, so labels are never leaky.
ROLE_LABEL_HORIZON_YEARS: int = 3

# ── Value target (2026-10-29) ─────────────────────────────────────────────────
# Which outcome fills `impact_target`, the column every impact consumer
# (04b/04d calibration, step 8/8b heads, step 9 comps) trains on. Step 7
# always writes BOTH explicitly — `impact_apy_target` and `value_target` — so
# the switch is this one flag plus a step-7 rerun.
#   "apy"   — first qualifying second contract's APY % of cap (legacy).
#   "value" — mean of within-(position group × draft class) percentiles of
#             (1) best post-rookie APY % of cap signed by the y5 offseason,
#             (2) best guaranteed money % of cap signed in that window,
#             (3) NFL usage in seasons 3-4 (snap share; depth-chart starter
#                 share for classes whose y3-y4 predate snap data).
#             Step 10 maps predicted percentiles back to APY % of cap.
# Referee test (drafted 2016-19, scored against y5-7 snap share and y7-9
# contracts, never against either arm's own target): the composite beat the
# legacy label at 8/8 positions on both referees. Earliest-deal-wins put the
# legacy label below the player's own best y5 deal for 46% of players.
#
# REJECTED as a training target (2026-10-29 A/B, cached hyperparameters):
# the impact head trained on "value" ranked NFL players WORSE (−0.070 / −0.052
# on the late referees, wins 1/9 and 2/9; seed-only placebo −0.021 / −0.019).
# Being NFL vs not explains 64% of value_target's variance vs 24% of APY's,
# so the squared-error head learns "made the NFL" instead of ranking NFL
# players. Even pushed through the APY curve it lost on the value referee
# (−0.035). value_target stays in the labels as an evaluation REFEREE — see
# experiments/00u_value_target_referee_ab.py. Keep "apy".
IMPACT_TARGET_SOURCE: str = "apy"
# Contracts signed through draft_year + this count (the y5 offseason).
VALUE_CONTRACT_LAST_YEAR_OFFSET: int = 4
# NFL seasons (offset from draft year) whose usage is averaged: seasons 3 and 4.
# Later seasons beat y1-y4 on both referees (+0.024 / +0.018) — rookie-year
# snaps penalize developmental players.
VALUE_USAGE_SEASON_OFFSETS: tuple[int, ...] = (2, 3)
# First draft class whose usage seasons sit inside snap_counts (2016+).
# Earlier classes use depth-chart starter share: rho 0.846 with snap share on
# the overlap, and still beats contract-only at 8/8 positions (RB weakest, 0.78).
VALUE_SNAP_FIRST_DRAFT_YEAR: int = 2014
# Classes used for step 10's percentile → APY map. OTC small-contract coverage
# is unreliable before 2011 (see the early-cohort audit), which would put too
# many zeros in the bottom of the map.
VALUE_MAP_MIN_DRAFT_YEAR: int = 2011

POSITION_GROUP_ORDINAL: dict[str, int] = {
    "QB": 0, "RB": 1, "WR": 2, "TE": 3, "OL": 4,
    "DL": 5, "LB": 6, "DB": 7, "K": 8, "P": 9, "EDGE": 10,
}

# ── Attribute Merge Schema ─────────────────────────────────────────────────────
# Single source of truth for which columns flow from player_attributes.parquet
# into downstream snapshots. Used by Steps 3, 8, 8b, 9, 10. Adding a new
# attribute requires adding it HERE and nowhere else; if it's missing from
# this list, it never reaches training or scoring (LightGBM swallows missing
# columns as NaN, so the regression is silent — see test_pipeline_alignment).
ATTRIBUTE_MERGE_COLS: list[str] = [
    # Identity / display
    "player_id", "player_name", "school", "position", "position_group",
    # Physical
    "height_in", "weight_lbs", "class_year_int",
    # Bio / age
    "birth_date", "birth_year",
    # College context
    "conference_tier",
    # 2026-05-15: tested career_max_conf_tier_power4/g5 (still computed by
    # step 03; kept in player_attributes.parquet for reference). Result was
    # 18W/12L step 8 (median -1.02%, mean +2.51% noise-skewed) and all 3 step
    # 8.5 global heads improved (-2.85% / -1.06% / -0.01%), BUT Boruta
    # rejected both binaries 0/30 + 0/3 — verdict was the wins were Optuna
    # seed variance, not feature signal. Reverted; re-add here if re-testing.
    "team_sp_offense", "team_sp_defense", "team_sp_overall",
    # Career team context (advanced_game_stats career means — captures the
    # offensive / defensive environment a player developed in)
    "career_team_off_ppa_mean", "career_team_off_success_rate_mean",
    "career_team_off_explosiveness_mean", "career_team_off_stuff_rate_mean",
    "career_team_def_ppa_mean", "career_team_def_success_rate_mean",
    "career_team_def_stuff_rate_mean", "career_team_def_explosiveness_mean",
    # Team box-score rushing four-factor + havoc career means. The line-yards
    # family is the only production-adjacent career signal available for OL,
    # who have no per-game stats and no pre-2014 PFF coverage. (The per-game
    # ROLLING variant of these stats was reverted 2026-05-16; this is the
    # career-mean channel, which is a different construction.)
    "career_team_line_yards_avg_mean",
    "career_team_second_level_yards_avg_mean",
    "career_team_open_field_yards_avg_mean",
    "career_team_rushing_power_success_mean",
    "career_team_rushing_stuff_rate_mean",
    "career_team_havoc_total_mean",
    "career_team_havoc_front_seven_mean",
    "career_team_havoc_db_mean",
    # Specialist career detail (step 03). Gated to the K/P heads in step 8's
    # _build_feature_cols — NaN for everyone else by construction.
    "career_fg_long_max", "career_punt_long_max",
    "career_punt_in20_rate", "career_punt_tb_rate",
    # NOTE: the six audit item-5 career features were built 2026-07-25 and
    # REVERTED the same day — all three families came in negative on a 3-seed
    # placebo-anchored A/B (Δ cv_rho_bridge, scored only on the heads each family
    # actually reaches):
    #   long_max  (rec/rush longest)     −0.0128, wins  1/12, −4.2 SE  ← harmful
    #   def_ball  (int-ret avg, solo %)  −0.0019, wins  6/12, −0.7 SE
    #   quarter   (Q1/Q4 play share)     −0.0026, wins 12/27, −1.0 SE
    # long_max is the surprise: it is the BEST-covered family (all years
    # 2004-2025) and the only one clearly outside noise, in the wrong direction.
    # The likely reason is that a career MAX is an extreme-value statistic — one
    # 99-yard catch defines it — so it competes for splits against the
    # well-estimated Kalman/EWMA receiving streams while carrying far more
    # variance. Sparsity was not the problem here; construction was.
    # Step 03 still COMPUTES all six (cheap, ~3s) so re-testing is a one-line
    # change; they are withheld from the model here and in step 8's
    # _build_feature_cols. Same convention as the career_max_conf_tier binaries
    # above.
    # NOTE: career_avg_elevation_ft tested 2026-05-18, dropped same day.
    # Boruta confirmed 0/30 heads (strongest possible no-signal verdict at
    # this layer). Step 03 still computes the column into player_attributes
    # .parquet (harmless extra data) — re-add this line to flow into step 8
    # if a future cycle wants to retest.
    # Recruiting
    "recruiting_stars", "recruiting_composite",
    "recruit_rank_national", "recruit_rank_position",
    # Career trajectory
    "last_college_season_year",
    "n_transfers_career",
]

# Features that the per-position scoring/training code computes inline (on a
# local group_df copy) rather than merging from a parquet. Used by the
# Step 10 runtime guard to suppress false-positive "missing feature" warnings.
DERIVED_AT_INFERENCE: set[str] = {
    "is_ol_tackle", "is_ol_guard", "is_ol_center",
    "pass_rush_composite",
    "pff_pass_rush_snap_pct", "pff_coverage_snap_pct",
    "pff_corner_snap_pct", "pff_slot_snap_pct", "pff_fs_snap_pct",
    # Built in add_derived_features() from raw combine_wt / combine_forty for
    # SPEED_SCORE_POSITIONS, so it never lands in latest.parquet as a column.
    "combine_speed_score",
    # These live in the snapshot today but are computed in main() rather than
    # merged from attrs — listed so a downstream removal still passes the guard.
    "age_at_draft", "team_sack_share",
}

# ── Monotone Constraints ──────────────────────────────────────────────────────
# Sign convention (LightGBM): +1 = output must increase with feature, -1 = must
# decrease, 0 = unconstrained (default for any feature not listed here).
# Confined to domain-unambiguous priors — wrong-sign constraints actively hurt
# accuracy, so the held-out fold's Spearman ρ is the safety check (a regression
# after adding a constraint should trigger removing it).
#
# z-score directionality reminder: in pipeline/00d_combine_ingestion.py, only
# `forty` is negated (line 49: NEGATE_METRICS={"forty"}), so combine_forty_z
# means "faster = higher". cone/shuttle z-scores are NOT negated — lower z
# means lower (faster) time, so the monotone direction is -1 there.
#
# Constraints are skipped automatically for LightGBM quantile heads (the
# objective is incompatible).
MONOTONE_CONSTRAINTS: dict[str, int] = {
    # Consensus draft board rank: lower number = higher prospect grade. The
    # safest constraint we have — Hasan's rank is itself an NFL-outcome
    # forecast, so monotonicity in our impact target is by construction.
    "hasan_rank":              -1,
    # Younger at draft = more developmental runway. Mirrors cdarko's
    # `age_at_season=-1` constraint and is uncontroversial across positions.
    "age_at_draft":            -1,
    # Other candidates considered but DROPPED based on holdout testing:
    #   recruiting_stars/composite (+1), recruit_rank_national/position (-1)
    #     — top recruits often bust, so models need non-monotonic patterns
    #       like "5-star with low PFF grade = bust". Constraint hurt
    #       DB_impact -0.26, WR_impact -0.20 on holdout. Kept unconstrained.
    #   combine_*_z — direction is right in expectation but signal-to-noise
    #       across all positions is too low to justify constraint; testing
    #       with all 6 combine z-scores constrained showed -0.04 mean
    #       Spearman change. Kept unconstrained.
}

# Per-(position_group, model_name) opt-in for the constraints above.
# Selected from a tune-with-constraints run that compared each model's
# constrained vs. unconstrained holdout Spearman; only models with a
# meaningful gain (≥+0.02 ΔSpearman) are listed. Models NOT in this set
# train and score with no monotone constraint at all. All models read
# hyperparameters from the single tuned_params.json (the former
# tuned_params_monotone.json mirror was collapsed 2026-07-21 — it was
# always byte-identical). Switching a model in/out requires retraining
# Step 8 — the constraint and the hyperparameter tune are paired.
MONOTONE_ENABLED: set[tuple[str, str]] = {
    # Re-tuned 2026-05-02 after the namesake-contamination fix cleaned ~2k
    # bridge-player training rows. Selected via paired sweep A (unconstrained)
    # vs sweep B (all-on); kept iff Δ holdout Spearman ≥ +0.02. Effect sizes
    # (sweep A → sweep B):
    ("OL", "impact_wav"),   # 0.379 → 0.691  (+0.312)
    ("RB", "impact"),       # 0.686 → 0.885  (+0.198)
    ("LB", "impact"),       # 0.501 → 0.659  (+0.158)
    ("K",  "impact"),       # 0.710 → 0.845  (+0.135)
    ("WR", "impact"),       # 0.710 → 0.811  (+0.101)
    ("DL", "impact_wav"),   # 0.545 → 0.640  (+0.095)
    ("RB", "impact_wav"),   # 0.405 → 0.478  (+0.073)
    ("WR", "impact_wav"),   # 0.458 → 0.529  (+0.071)
    # Mean Δ across the 8 keepers: +0.143 Spearman.
    # Dropped vs prior set: P/impact_wav (was +0.605 at n=203, now −0.067 —
    # confirms the user's worry that the original was a small-n fluke) and
    # QB/impact_wav (was +0.082, now −0.028).
    # Global cross-position model: mean Δ +0.004 across the 3 heads — does
    # not benefit from the constraint at the all-or-nothing granularity.
}


# ── Draft Model Parameters ─────────────────────────────────────────────────────

DRAFT_TRAIN_CUTOFF_YEAR = 2019   # Eval-split cutoff: train on ≤ this year, val on next 3 years
DRAFT_VAL_START_YEAR = 2020      # Eval-split val window: [DRAFT_VAL_START_YEAR, +2]
# After eval, final model retrains on all data up to END_YEAR - 2 (excludes current class)

# Salary cap used to convert impact_apy_score (apy_cap_pct) to dollar amounts
# at output time. Update annually as the league cap changes.
SALARY_CAP = 300_000_000

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}

# ── Mutual-Information Feature Pre-Filter ──────────────────────────────────────
# State-space PFF block (kalman_*_pff_*, ewma_pff_*, ewma_var_pff_*,
# kalman_yoy_pff_*, ewma_yoy_pff_*) ships ~240 candidate cols carrying ~14% of
# total LightGBM importance — i.e. ~0.06% per col on average vs combine's
# ~3.3% per col. That's classic dilution: many low-info features compete with
# a few high-info features for splits, and on small-n heads the spurious
# splits dominate.
#
# MI ranks each state-space candidate against the target on the training panel
# (one-shot, deterministic — no Optuna seed dependence) and keeps the top-N.
# Other feature categories (combine, attrs, hasan, pff_career, game-grain
# Kalman/EWMA) pass through unchanged.
#
# When `--mi-filter` is passed to step 8 / 8.5, MI is computed and cached at
# `data/processed/draft_models/mi_features.json`. Subsequent fits read the
# cache and filter feature_cols accordingly.

MI_TOP_N = 40              # state-space cols kept per (pos × target). 240 → 40 = 6× reduction.
MI_RANDOM_STATE = 42
MI_N_NEIGHBORS = 3         # sklearn default; affects MI estimation noise.

# ── Boruta Feature Selection ───────────────────────────────────────────────────
# Boruta is used as an AUDIT tool — it does NOT gate fits. The flow:
#   1. `--boruta` flag (gated to retune mode) runs Boruta per (pos × target) head,
#      33 heads total (30 in step 8 + 3 in step 8b).
#   2. `experiments/boruta_summary.py` aggregates verdicts across heads and
#      surfaces features confirmed in 0 heads — those become drop candidates.
#   3. The user manually promotes drop candidates into BORUTA_DROP_FEATURES below.
#   4. `_build_feature_cols` / `_build_global_feature_cols` strip these features
#      at the source — identical effect on every head, no per-head over-pruning.
#
# This matches cdarko's posture (one Boruta pass, drop the obvious noise, keep
# the rest) rather than treating each per-head Boruta verdict as gospel —
# single-run Boruta confirmation is too noisy for the small-n heads.

BORUTA_N_ESTIMATORS = "auto"      # boruta_py picks n_estimators per iteration
BORUTA_MAX_ITER = 100             # default; first-pass economy. Override via --boruta-trials
BORUTA_PERCENTILE = 100           # strict — beat the BEST shadow (vs 95 = 95th-pct shadow)
BORUTA_ALPHA = 0.05               # binomial-test significance for keep / reject
BORUTA_RANDOM_STATE = 42

# Empty for now. Two consecutive Boruta audit passes (2026-05-06, 2026-05-07)
# produced "confirmed in 0 heads" sets that did not overlap (13 features the
# first pass, 11 different features the second), suggesting single-pass
# Boruta verdicts have too much run-to-run variance to be worth acting on.
# Promote a feature in here only when it shows up "confirmed in 0 heads"
# across multiple independent audit passes (intersect the rejected sets).
#
# Reference — features rejected in 0/33 heads on 2026-05-06:
#   combine_cone_z, recruit_rank_national, ewma_rush_share, ewma_var_fg_made,
#   ewma_var_pass_share, ewma_var_pass_tds, ewma_yoy_fg_att,
#   kalman_mean_pass_att, kalman_yoy_pass_yards,
#   ewma_var_pff_grades_pass_rush_defense, ewma_var_pff_pbe,
#   ewma_var_pff_zone_grades_pass_route, pff_zone_grades_run_block_mean
#
# Reference — features rejected in 0/33 heads on 2026-05-07 (with the above
# already pruned from input):
#   nfl_lg_plays_pg, ewma_pass_share, ewma_var_passes_defended,
#   ewma_yoy_rec_share, kalman_var_net_punt_yards, kalman_yoy_rec,
#   ewma_pff_grades_defense, ewma_var_pff_grades_offense,
#   pff_no_pressure_completion_percent_last,
#   pff_no_pressure_completion_percent_mean, pff_pressures_allowed_peak
# Promoted 2026-05-24 from the Boruta audit after the Phase 3a retune.
# These are features confirmed in 0 heads AND 0 tentative across all 33
# Boruta heads (30 per-position + 3 global). See experiments/boruta_summary.py
# for the audit report that produced this list. Refresh whenever the
# feature surface or retune changes substantially.
BORUTA_DROP_FEATURES: set[str] = {
    # Combine (00d_combine)
    "combine_vertical_z",
    "combine_wingspan",
    # Career-totals derived (08_derived / 03_attributes_or_other)
    "career_rush_att",
    "career_touches",
    # Game-grain Kalman/EWMA (step 6)
    "ewma_interceptions",
    "ewma_var_passes_defended",
    "ewma_var_rush_yards",
    "ewma_yoy_pass_share",
    "ewma_yoy_punt_yards",
    "kalman_mean_pass_att",
    "kalman_mean_yards_per_rec",
    # Play-grain Kalman/EWMA variants from Phase 3a (step 6 over step 5p)
    "ewma_var_play_mean_resid_reception",
    "ewma_var_play_mean_resid_rushing_touchdown_outcome",
    "kalman_mean_play_raw_mean_rushing_fumble_outcome",
    "kalman_mean_play_total_resid_passing_touchdown_outcome",
    "kalman_yoy_play_mean_resid_receiving_touchdown_outcome",
    "kalman_yoy_play_total_resid_completion",
    # PFF career static triplets (step 1c)
    "pff_average_hangtime_mean",
    "pff_avg_time_to_throw_mean",
    "pff_grades_offense_peak",
    "pff_grades_tackle_mean",
    "pff_no_pressure_completion_percent_last",
    "pff_sack_percent_last",
    "pff_zone_grades_run_block_last",
}

# ── Conference Tier Mapping ────────────────────────────────────────────────────

CONFERENCE_TIERS: dict[str, int] = {
    # Power Four (tier 1)
    "SEC": 1, "Big Ten": 1, "Big 12": 1, "ACC": 1, "Pac-12": 1, "Pac-10": 1,
    "Big Ten Conference": 1,
    "Big East": 1,          # BCS conference until 2013; most historical data is from that era
    # Group of Five (tier 2) — both short names and full CFBD names
    "AAC": 2, "American Athletic": 2,
    "MAC": 2, "Mid-American": 2,
    "Mountain West": 2, "Conference USA": 2, "Sun Belt": 2,
    "Western Athletic": 2,  # WAC had FBS members through ~2013
    "FBS Independents": 2,
    # FCS (tier 3)
    "Big Sky": 3, "CAA": 3, "MVFC": 3, "Southland": 3, "Southern": 3,
    "SWAC": 3, "MEAC": 3, "OVC": 3, "NEC": 3, "Patriot": 3,
    "Pioneer": 3, "Big South": 3, "Ivy": 3, "Atlantic Sun": 3,
    "FCS Independents": 3, "Big South-OVC": 3, "UAC": 3, "AWC": 3,
}

DEFAULT_CONFERENCE_TIER = 3     # FCS / unknown

# ── Known Player-ID / Season Exclusions ────────────────────────────────────────
# CFBD occasionally recycles a player_id for a different person (name collision).
# Each entry maps player_id → list of season_years to drop from gamelogs and
# player_seasons during Step 01 assembly.  The rest of the player's history
# (e.g. a bridge player's college career) is preserved.
#
# To add a new exclusion: player_id (int) → [season_year, ...]
# Run `python experiments/00_audit_player_ids.py` to detect candidates.
# ── Manual birth year corrections ─────────────────────────────────────────────
# Keyed by CFBD player_id (int) → correct birth year (int).
# Applied in Step 03 as the highest-priority source, overriding both
# nfl_data_py PFR data and Wikidata matches.
BIRTH_YEAR_OVERRIDES: dict[int, int] = {
    4870707: 2004,  # Keldric Faulk (Auburn DB) — born Sep 7, 2004; class-year estimate was 2005
}

# ── Venue Feature Bucketing ───────────────────────────────────────────────────
# altitude_diff_from_home_ft is bucketed to avoid unique (home, away) elevation
# pairs from acting as a school-pair fingerprint (e.g. Stanford → Air Force =
# +7188 ft would be a 1-of-1 signature; bucketed to +7000 it joins ~10 other
# travel-to-altitude pairs and the model can't reverse-identify the schools).
# 500ft is fine enough that the thin-air gradient (which ramps non-linearly
# above ~4000ft) still carries signal, coarse enough to dissolve identity.
ALTITUDE_DIFF_BUCKET_FT: int = 500

# travel_distance_mi is Haversine miles from the player's home stadium to the
# game venue, bucketed for the same identity-protection reason — a specific
# distance like 2143mi uniquely tags Boston College ↔ USC. 250mi gives ~13
# buckets across the 0-3000mi domain (Hawaii being the outlier).
TRAVEL_DISTANCE_BUCKET_MI: int = 250

# capacity_diff_from_home is the game venue's seating capacity minus the player's
# own home capacity — a crowd-hostility shock — bucketed for the same
# identity-protection reason as the two above. An exact diff like +47,318 tags a
# single (home, away) stadium pair; 10,000 seats is coarse enough to dissolve that
# while preserving the ordering that matters (a G5 team walking into a 100k SEC
# stadium vs. a lateral move between two 40k venues). 96.6% of games have a
# capacity for both sides. 70 of 840 venues report capacity 0, which means
# unknown rather than a genuinely empty stadium — those are nulled in step 1 so
# the diff comes out NaN instead of a fabricated -100k shock.
CAPACITY_DIFF_BUCKET: int = 10_000

# ── Stadium Surface Overrides ─────────────────────────────────────────────────
# CFBD `/venues` returns a CURRENT snapshot of `grass` (bool). Stadium surfaces
# change over the decades we model — LSU's Tiger Stadium went grass→turf→grass,
# Carrier Dome got AstroTurf upgrades, several MAC venues converted grass→turf
# in the 2010s. Without overrides, those years carry an anachronistic surface
# label and the model learns spurious patterns at the grass/turf seam.
#
# Format: venue_id → list of (start_year, end_year_inclusive, surface) tuples.
# `surface` is "grass" or "turf". The lookup in
# pipeline/01_data_ingestion.py:_add_venue_features() checks overrides FIRST
# (returning the matched surface), then falls back to venues.parquet's `grass`
# field. Years outside any override range fall through to the snapshot value.
#
# Add an entry only when you can verify the change date (Wikipedia stadium
# pages + news archive searches work). Untreated unknowns are NaN — the model
# gets no signal for that game-row rather than a wrong one.
#
# Populate as needed; the scaffold below documents the known unknowns.
# Candidates flagged but not yet verified:
#   - LSU Tiger Stadium (venue_id TBD): grass → turf (~mid-2000s) → grass (~2009)
#   - Carrier Dome (Syracuse, venue_id TBD): multiple AstroTurf generations
#   - MAC conversions (Ball State, Bowling Green, etc.) in the 2010s
STADIUM_SURFACE_OVERRIDES: dict[int, list[tuple[int, int, str]]] = {}

# ── Manual bridge-name overrides ──────────────────────────────────────────────
# Maps (cfbd_player_name, college_team, draft_year) → pfr_player_name. Used in
# match_draft_picks() as the highest-priority lookup before any name/school
# matching pass. Each entry corresponds to a player whose CFBD college name
# and PFR/nflverse NFL name are too dissimilar for fuzzy matching — most are
# nickname swaps where the first name is unrelated (Sauce/Ahmad Gardner,
# Beanie/Chris Wells, Mo/Maurice Hurst, Nnamdi/Justin Madubuike, Tank/
# Nathaniel Dell, O.J./Oshiomogho Atogwe, Shaquille/Darius Leonard).
#
# Untreated, these players are flagged "unmatched" with career_w_av=0 even
# though they have substantial NFL careers — which would corrupt the bridge
# training labels. Add new entries here when `match_quality='unmatched'`
# surfaces a player whose NFL identity you can verify manually.
BRIDGE_NAME_OVERRIDES: dict[tuple[str, str, int], str] = {
    # (cfbd_name, college_team, draft_year) → pfr_player_name
    ("Ahmad Gardner",       "Cincinnati",           2022): "Sauce Gardner",
    ("Alton McCann",        "West Virginia",        2006): "Dee McCann",
    ("Buck Allen",          "USC",                  2015): "Javorius Allen",
    ("Carlos Basham Jr.",   "Wake Forest",          2021): "Boogie Basham",
    ("Chris Wells",         "Ohio State",           2009): "Beanie Wells",
    ("Christopher Owens",   "San José State",       2009): "Chris Owens",
    ("Darius Leonard",      "South Carolina State", 2018): "Shaquille Leonard",
    ("Daymeion Hughes",     "California",           2007): "Dante Hughes",
    ("DeMarcus Tyler",      "NC State",             2007): "Tank Tyler",
    ("Dwight Bentley",      "Louisiana",            2012): "Bill Bentley",
    ("Edmond Gates",        "Abilene Christian",    2011): "Clyde Gates",
    ("James Grigsby",       "Illinois State",       2005): "Boomer Grigsby",
    ("Janoris Jenkins",     "North Alabama",        2012): "Jackrabbit Jenkins",
    ("Jartavius Martin",    "Illinois",             2023): "Quan Martin",
    ("Justin Madubuike",    "Texas A&M",            2020): "Nnamdi Madubuike",
    ("L.T. Walton",         "Central Michigan",     2015): "Leterrius Walton",
    ("Lazarius Levingston", "LSU",                  2011): "Pep Levingston",
    ("Mo Hurst",            "Michigan",             2018): "Maurice Hurst Jr.",
    ("Nathaniel Dell",      "Houston",              2023): "Tank Dell",
    ("Olisaemeka Udoh",     "Elon",                 2019): "Oli Udoh",
    ("Oshiomogho Atogwe",   "Stanford",             2005): "O.J. Atogwe",
    ("Roderick Issac",      "Middle Tennessee",     2011): "Rod Issac",
}

# Tolerance (years) for matching a draft pick's last_college_season_year to a
# CFBD player_seasons row when resolving bridge_players.player_id. Cross-
# generational namesakes (e.g. Will Smith / Will Smith Jr. at Ohio State)
# share normalized name+school but their season ranges are decades apart, so
# requiring proximity prevents stamping the old draftee's bridge row with the
# current player's CFBD ID.
BRIDGE_YEAR_TOLERANCE = 2

# Extra years a UDFA's college career may end before his NFL entry year. UDFA
# rows estimate last_college_season_year as entry - 1, but late signings
# (kickers, long snappers, players who spent a year out) entered 2-5 years
# after their last college season. Exact name + school matches only.
BRIDGE_UDFA_LATE_ENTRY_YEARS = 5

# nflverse college spellings → the CFBD school name, for bridge resolution
# (step 02). Only names that occur in the UDFA college strings and differ
# from CFBD's; every target was checked against player_seasons ∪ rosters.
BRIDGE_SCHOOL_ALIASES = {
    "Albany":                      "UAlbany",
    "Appalachian State":           "App State",
    "Cal Poly (San Luis Obispo)":  "Cal Poly",
    "Connecticut":                 "UConn",
    "Grambling State":             "Grambling",
    "Hawaii":                      "Hawai'i",
    "Louisiana-Lafayette":         "Louisiana",
    "Louisiana-Monroe":            "UL Monroe",
    "McNeese State":               "McNeese",
    "Miami (Ohio)":                "Miami (OH)",
    "Middle Tennessee State":      "Middle Tennessee",
    "Mississippi":                 "Ole Miss",
    "N.C. State":                  "NC State",
    "Northwestern State (LA)":     "Northwestern State",
    "Penn":                        "Pennsylvania",
    "Sam Houston State":           "Sam Houston",
    "San Jose State":              "San José State",
    "Southeastern Louisiana":      "SE Louisiana",
    "Southern Mississippi":        "Southern Miss",
    "Stephen F. Austin State":     "Stephen F. Austin",
    "Tennessee-Chattanooga":       "Chattanooga",
    "Texas State-San Marcos":      "Texas State",
    "Texas-El Paso":               "UTEP",
    "Texas-San Antonio":           "UTSA",
    "UMass Amherst":               "Massachusetts",
    "University of South Florida": "South Florida",
}

# Plausible age window (in years) at last college season for a bridge- or
# wikidata-derived birth_year. Outside this window, the birth_year is
# rejected as a likely namesake collision and falls through to the class-
# year estimate. 5th-year seniors and COVID-waiver guys top out ~23-24;
# 26 leaves slack while catching the 30-45 contamination cases. The lower
# bound 16 catches the Mitchell-Evans-aged-7 type errors where a current
# player gets matched to an unrelated younger person.
MIN_BRIDGE_DERIVED_AGE = 16
MAX_BRIDGE_DERIVED_AGE = 26

KNOWN_SEASON_EXCLUSIONS: dict[int, list[int]] = {
    4039307: [2025],  # Houston WR (2016-2020, Bills R6 2021) — CFBD recycled this ID
                      # for a different Marquez Stevenson (Texas Tech DB, 2025)
    4685405: [2022, 2025],  # 2011 Cam Johnson (NFL bridge, drafted 2012) — CFBD recycled
                            # this ID for a different Cam Johnson (Virginia Tech 2022,
                            # Stony Brook 2025); VT/Stony Brook are the same person but
                            # not the same person as the bridge player
    4241676: [2025],  # Texas State DB Kordell Rodgers (2017-2022) — CFBD recycled this
                      # ID for a different Kordell Rodgers (Texas Southern WR, 2025)
}

# ── Data Source Parameters ─────────────────────────────────────────────────────

# CFBD API base URL (used by scrapers)
CFBD_API_BASE = "https://api.collegefootballdata.com"

# PFR scraping
PFR_BASE_URL = "https://www.pro-football-reference.com"

# Minimum games in a college season to include in modeling
MIN_GAMES_SEASON = 3

# ── Comp Generation ────────────────────────────────────────────────────────────

N_COMPS = 5                     # Number of historical comps per player
COMP_FEATURE_TOP_K = 40         # Top-K features from role model importance

# ── PFF Season Grades ──────────────────────────────────────────────────────────

PFF_DATA_DIR = RAW_DIR / "pff"

# Deprecated: conference tier in contextual models now handles baseline opponent quality.
# Kept for backward compat but no longer added to draft model features.
OPP_CAREER_FEATURES: list[str] = []

# Arif Hasan Consensus Big Board rank (1–300; lower = higher prospect grade)
HASAN_FEATURES: list[str] = ["hasan_rank"]

COMBINE_FEATURES_BY_GROUP: dict[str, list[str]] = {
    "QB": ["combine_forty_z", "combine_vertical_z", "combine_broad_jump_z", "combine_cone_z",
           "combine_shuttle_z", "combine_ht_z", "combine_wt_z"],
    "RB": ["combine_forty_z", "combine_vertical_z", "combine_broad_jump_z",
           "combine_bench_z", "combine_shuttle_z", "combine_wt_z"],
    "WR": ["combine_forty_z", "combine_vertical_z", "combine_broad_jump_z", "combine_cone_z",
           "combine_shuttle_z", "combine_ht_z", "combine_wt_z",
           "combine_arm_z", "combine_hand_z"],
    "TE": ["combine_forty_z", "combine_vertical_z", "combine_broad_jump_z", "combine_bench_z",
           "combine_cone_z", "combine_shuttle_z", "combine_ht_z", "combine_wt_z",
           "combine_arm_z"],
    "OL": ["combine_broad_jump_z", "combine_forty_z",
           "combine_cone_z", "combine_shuttle_z",
           "combine_arm_z", "combine_wingspan_z"],
    "DL": ["combine_forty_z", "combine_broad_jump_z",
           "combine_shuttle_z", "combine_ht_z", "combine_wt_z",
           "combine_wingspan_z"],
    "EDGE": ["combine_forty_z", "combine_broad_jump_z",
             "combine_shuttle_z", "combine_ht_z", "combine_wt_z",
             "combine_wingspan_z"],
    "LB": ["combine_forty_z", "combine_vertical_z", "combine_broad_jump_z", "combine_cone_z",
           "combine_shuttle_z", "combine_ht_z", "combine_wt_z"],
    "DB": ["combine_forty_z", "combine_vertical_z", "combine_broad_jump_z",
           "combine_ht_z", "combine_wt_z"],
    "K":  ["combine_forty_z", "combine_vertical_z", "combine_ht_z", "combine_wt_z"],
    "P":  ["combine_forty_z", "combine_vertical_z", "combine_ht_z", "combine_wt_z"],
}

# Per-position swaps from cohort-z to ABSOLUTE values, sourced from the
# 2026-05-10 audit (`experiments/00i_combine_zscore_audit.py`). These metrics
# either had near-zero year-drift R² (anatomy is anatomy — arm/hand/wingspan)
# or showed raw-correlates-better with bridge labels than the cohort-z
# version. Step 8's `_build_feature_cols` swaps the `_z` for the listed raw
# col on a per-position basis. Step 8b unions both raw + z into the global
# feature pool (the global model can't do per-pos surgery, so it gets both).
COMBINE_RAW_BY_GROUP: dict[str, list[str]] = {
    "QB": ["combine_hand", "combine_wingspan"],
    "RB": ["combine_cone", "combine_hand", "combine_ht"],
    "WR": ["combine_wingspan"],
    "TE": ["combine_hand"],
    "OL": ["combine_bench", "combine_hand", "combine_ht", "combine_wt"],
    "DL": ["combine_arm", "combine_bench", "combine_cone", "combine_hand"],
    "EDGE": ["combine_arm", "combine_bench", "combine_cone", "combine_hand"],
    "LB": ["combine_arm"],
    "DB": ["combine_arm", "combine_cone", "combine_shuttle", "combine_wingspan"],
    "K":  [],
    "P":  [],
}

# Position groups that get the weight-adjusted "speed score"
# (combine_wt × 200 / combine_forty⁴) as an explicit derived feature.
#
# Restricted to the groups where the 2026-07-22 audit found it beats its own
# components: univariate bridge-only Spearman vs impact was DL +0.231 (best
# component 0.132), LB +0.250 (0.238), TE +0.180 (0.178), and a fixed-param
# A/B on the impact_wav heads moved MAE by DL −14.3%, TE −2.6%, LB −1.3%,
# RB −1.2%. WR is deliberately excluded — neither plain nor height-adjusted
# speed score separated from the raw components there.
#
# This is a construction, not a rescaling: LightGBM is invariant to monotone
# transforms of a single feature, so raw combine_wt / combine_forty stay in
# the surface and the model picks whichever it prefers.
#
# 2026-07-25 extension (experiments/00o_speed_score_positions.py). The univariate
# screen above was re-run over ALL groups and reproduced the WR exclusion, then
# each candidate got a 3-seed per-head A/B against an A/A placebo arm. Decided on
# Δ cv_rho_bridge vs that placebo, because the univariate margins (OL +0.009,
# DB +0.019, QB +0.008 over the best component) are far too small to call on
# their own:
#   OL  ADDED    Δρ +0.0141 vs placebo +0.0027  (~5× the noise floor, 2/3 heads)
#   QB  ADDED    Δρ +0.0328 vs placebo +0.0167  (3/3 heads, but the placebo is
#                itself large — QB is the thinnest cohort at n=180 bridge, so
#                this is the weaker of the two additions)
#   DB  rejected Δρ +0.0036 vs placebo +0.0045  (inside noise, 1/3 heads)
#   RB  KEPT     Δρ +0.0122 vs placebo −0.0034  — worth noting RB now FAILS the
#                univariate screen (+0.115 vs forty's +0.124) yet still clears
#                its placebo, which is why the A/B and not the screen is what
#                decides membership here.
# cv_mae rose slightly for both additions (OL +2.4%, QB +1.4%); that metric is
# dominated by the ~90% zero-padded non-NFL mass, so ranking is what governs.
# Note the feature is only 4-6% non-null per group (combine invitees only).
SPEED_SCORE_POSITIONS: frozenset[str] = frozenset({
    "DL", "EDGE", "LB", "TE", "RB", "OL", "QB",
})

# Position groups that get career kick/punt return features (built in step 06).
# Returners come almost exclusively from these three groups — 9.3% of WR/RB/DB
# player-games carry return yardage, versus ~0 elsewhere — so the features are
# gated rather than global to keep them out of surfaces where they'd be a
# constant zero (which the >99% dominant-value filter would drop anyway, but
# only after they'd diluted the MI ranking).
RETURN_FEATURE_POSITIONS: frozenset[str] = frozenset({"WR", "RB", "DB"})

NFL_LG_FEATURES: list[str] = [
    "nfl_lg_pass_att_pg",
    "nfl_lg_completion_pct",
    "nfl_lg_pass_yards_pg",
    "nfl_lg_pass_td_pg",
    "nfl_lg_rush_att_pg",
    "nfl_lg_rush_yards_pg",
    "nfl_lg_pts_pg",
    "nfl_lg_plays_pg",
    "nfl_lg_sacks_pg",
    "nfl_lg_turnovers_pg",
]

PFF_GRADE_FEATURES_BY_GROUP: dict[str, list[str]] = {
    "QB":  ["pff_grades_pass", "pff_grades_offense", "pff_btt_rate", "pff_twp_rate", "pff_accuracy_percent",
            "pff_passing_snaps", "pff_sack_percent", "pff_completion_percent", "pff_avg_time_to_throw",
            "pff_drop_rate", "pff_pressure_to_sack_rate", "pff_qb_rating", "pff_scrambles",
            "pff_pressure_grades_pass", "pff_no_pressure_grades_pass",
            "pff_pressure_completion_percent", "pff_no_pressure_completion_percent",
            "pff_pressure_sack_percent", "pff_no_pressure_sack_percent",
            "pff_blitz_grades_pass", "pff_no_blitz_grades_pass",
            "pff_blitz_btt_rate", "pff_no_blitz_btt_rate",
            "pff_epa", "pff_positive_epa_percent", "pff_dropbacks", "pff_attempts",
            "pff_first_downs", "pff_hit_as_threw"],
    "RB":  ["pff_grades_run", "pff_grades_offense", "pff_elusive_rating", "pff_breakaway_percent",
            "pff_yco_attempt", "pff_grades_pass_route", "pff_yprr",
            "pff_routes", "pff_targets", "pff_yards_after_contact", "pff_avoided_tackles",
            "pff_grades_hands_drop", "pff_drop_rate", "pff_targeted_qb_rating",
            "pff_yards_after_catch_per_reception",
            "pff_man_grades_pass_route", "pff_zone_grades_pass_route",
            "pff_man_yprr", "pff_zone_yprr",
            "pff_man_targeted_qb_rating", "pff_zone_targeted_qb_rating",
            "pff_grades_pass_block", "pff_pbe", "pff_pressures_allowed", "pff_sacks_allowed",
            # Engineered season-level rates / deltas
            "pff_routes_per_off_snap", "pff_targets_per_off_snap",
            "pff_man_zone_route_grade_delta", "pff_man_zone_yprr_delta",
            "pff_avoided_tackles_per_attempt"],
    "WR":  ["pff_grades_pass_route", "pff_grades_offense", "pff_yprr", "pff_avg_depth_of_target",
            "pff_route_rate", "pff_slot_rate", "pff_wide_rate",
            "pff_contested_catch_rate", "pff_yards_per_reception",
            "pff_grades_hands_drop", "pff_drop_rate", "pff_targeted_qb_rating",
            "pff_yards_after_catch_per_reception",
            "pff_man_grades_pass_route", "pff_zone_grades_pass_route",
            "pff_man_yprr", "pff_zone_yprr",
            "pff_man_targeted_qb_rating", "pff_zone_targeted_qb_rating",
            "pff_grades_run_block",
            # Engineered season-level rates / deltas
            "pff_routes_per_off_snap", "pff_targets_per_off_snap",
            "pff_man_zone_route_grade_delta", "pff_man_zone_yprr_delta"],
    "TE":  ["pff_grades_pass_route", "pff_grades_offense", "pff_yprr", "pff_inline_rate",
            "pff_grades_pass_block", "pff_inline_snaps", "pff_slot_snaps",
            "pff_grades_hands_drop", "pff_drop_rate", "pff_targeted_qb_rating",
            "pff_yards_after_catch_per_reception",
            "pff_man_grades_pass_route", "pff_zone_grades_pass_route",
            "pff_man_yprr", "pff_zone_yprr",
            "pff_man_targeted_qb_rating", "pff_zone_targeted_qb_rating",
            "pff_pressures_allowed", "pff_sacks_allowed", "pff_pbe",
            # Engineered season-level rates / deltas
            "pff_routes_per_off_snap", "pff_targets_per_off_snap",
            "pff_man_zone_route_grade_delta", "pff_man_zone_yprr_delta"],
    "OL":  ["pff_grades_offense", "pff_grades_pass_block", "pff_grades_run_block",
            "pff_pass_block_percent", "pff_pressures_allowed", "pff_snap_counts_offense",
            "pff_sacks_allowed", "pff_hurries_allowed", "pff_pbe", "pff_hits_allowed",
            "pff_gap_grades_run_block", "pff_zone_grades_run_block",
            # True pass set: pre-snap pass-blocking only, the cleanest OL eval
            "pff_true_pass_set_grades_pass_block", "pff_true_pass_set_pressures_allowed",
            "pff_true_pass_set_hurries_allowed", "pff_true_pass_set_sacks_allowed",
            "pff_true_pass_set_hits_allowed", "pff_true_pass_set_pbe",
            "pff_true_pass_set_snap_counts_pass_block",
            # Engineered season-level rates (scheme-normalized — addresses Fano
            # case where Utah's run-heavy 2025 dropped his raw true-pass-set
            # snap count but his per-snap pressure rate was excellent)
            "pff_pressures_allowed_per_off_snap", "pff_sacks_allowed_per_off_snap",
            "pff_hurries_allowed_per_off_snap", "pff_hits_allowed_per_off_snap",
            "pff_true_pass_set_pressures_allowed_rate",
            "pff_true_pass_set_sacks_allowed_rate",
            "pff_true_pass_set_hurries_allowed_rate",
            "pff_true_pass_set_hits_allowed_rate"],
    "DL":  ["pff_grades_defense", "pff_grades_pass_rush_defense", "pff_grades_run_defense",
            "pff_pass_rush_win_rate", "pff_prp",
            "pff_snap_counts_defense", "pff_snap_counts_pass_rush", "pff_total_pressures",
            "pff_grades_tackle", "pff_pass_break_ups", "pff_sacks",
            "pff_stop_percent", "pff_snap_counts_run",
            "pff_hurries", "pff_hits", "pff_tackles_for_loss",
            "pff_forced_fumbles", "pff_interceptions",
            "pff_pass_rush_opp", "pff_pass_rush_percent",
            "pff_avg_depth_of_tackle", "pff_run_stop_opp",
            # Engineered season-level rates (per pass-rush / defensive snap)
            "pff_sacks_per_pr_snap", "pff_hits_per_pr_snap",
            "pff_hurries_per_pr_snap", "pff_total_pressures_per_pr_snap",
            "pff_tackles_for_loss_per_def_snap", "pff_forced_fumbles_per_def_snap"],
    "EDGE": ["pff_grades_defense", "pff_grades_pass_rush_defense", "pff_grades_run_defense",
             "pff_pass_rush_win_rate", "pff_prp",
             "pff_snap_counts_defense", "pff_snap_counts_pass_rush", "pff_total_pressures",
             "pff_grades_tackle", "pff_pass_break_ups", "pff_sacks",
             "pff_stop_percent", "pff_snap_counts_run",
             "pff_hurries", "pff_hits", "pff_tackles_for_loss",
             "pff_forced_fumbles", "pff_interceptions",
             "pff_pass_rush_opp", "pff_pass_rush_percent",
             "pff_avg_depth_of_tackle", "pff_run_stop_opp",
             # Engineered season-level rates (per pass-rush / defensive snap)
             "pff_sacks_per_pr_snap", "pff_hits_per_pr_snap",
             "pff_hurries_per_pr_snap", "pff_total_pressures_per_pr_snap",
             "pff_tackles_for_loss_per_def_snap", "pff_forced_fumbles_per_def_snap"],
    "LB":  ["pff_grades_defense", "pff_grades_pass_rush_defense", "pff_grades_coverage_defense",
            "pff_grades_run_defense", "pff_forced_incompletion_rate",
            "pff_snap_counts_defense", "pff_snap_counts_pass_rush", "pff_snap_counts_coverage",
            "pff_missed_tackle_rate", "pff_grades_tackle", "pff_pass_break_ups",
            "pff_man_grades_coverage_defense", "pff_zone_grades_coverage_defense",
            "pff_man_qb_rating_against", "pff_zone_qb_rating_against",
            "pff_man_forced_incompletion_rate", "pff_zone_forced_incompletion_rate",
            "pff_stop_percent", "pff_snap_counts_run",
            "pff_hurries", "pff_hits", "pff_tackles_for_loss",
            "pff_forced_fumbles", "pff_interceptions",
            "pff_yards_per_coverage_snap", "pff_coverage_snaps_per_target",
            "pff_avg_depth_of_tackle", "pff_run_stop_opp",
            # Engineered season-level rates / deltas
            "pff_sacks_per_pr_snap", "pff_hits_per_pr_snap",
            "pff_hurries_per_pr_snap", "pff_total_pressures_per_pr_snap",
            "pff_tackles_for_loss_per_def_snap", "pff_forced_fumbles_per_def_snap",
            "pff_interceptions_per_def_snap", "pff_pass_break_ups_per_cov_snap",
            "pff_forced_incompletes_per_cov_snap",
            "pff_man_zone_qb_rating_against_delta",
            "pff_man_zone_coverage_grade_delta"],
    "DB":  ["pff_grades_defense", "pff_grades_coverage_defense", "pff_qb_rating_against",
            "pff_forced_incompletion_rate",
            "pff_snap_counts_defense", "pff_snap_counts_corner", "pff_snap_counts_slot",
            "pff_snap_counts_fs", "pff_missed_tackle_rate", "pff_catch_rate",
            "pff_grades_tackle", "pff_pass_break_ups",
            "pff_man_grades_coverage_defense", "pff_zone_grades_coverage_defense",
            "pff_man_qb_rating_against", "pff_zone_qb_rating_against",
            "pff_man_forced_incompletion_rate", "pff_zone_forced_incompletion_rate",
            "pff_man_snap_counts_coverage", "pff_zone_snap_counts_coverage",
            "pff_stop_percent", "pff_snap_counts_run",
            "pff_yards_per_coverage_snap", "pff_coverage_snaps_per_target",
            "pff_coverage_snaps_per_reception", "pff_forced_incompletes",
            "pff_forced_fumbles", "pff_interceptions",
            # Engineered season-level rates / deltas
            "pff_tackles_for_loss_per_def_snap", "pff_forced_fumbles_per_def_snap",
            "pff_interceptions_per_def_snap", "pff_pass_break_ups_per_cov_snap",
            "pff_forced_incompletes_per_cov_snap",
            "pff_man_zone_qb_rating_against_delta",
            "pff_man_zone_coverage_grade_delta"],
    "K":   ["pff_grades_fgep_kicker", "pff_total_percent",
            "pff_thirty_percent", "pff_forty_percent", "pff_fifty_percent",
            "pff_total_attempts", "pff_fifty_attempts", "pff_forty_attempts",
            "pff_thirty_attempts", "pff_twenty_attempts", "pff_twenty_percent",
            "pff_pat_attempts", "pff_pat_percent"],
    "P":   ["pff_grades_punter", "pff_average_net_yards", "pff_average_yards_per_attempt",
            "pff_inside_twenties", "pff_average_hangtime", "pff_touchbacks",
            "pff_attempts", "pff_long", "pff_blocks",
            "pff_total_hangtime", "pff_attempts_with_hangtime"],
}

# ── Season-Grain PFF State-Space Pipeline ──────────────────────────────────────
# Mirrors the game-grain residualize → EWMA/Kalman pipeline for PFF
# season-level stats. Steps 04s/05s/06s consume these constants;
# 04f/04g calibrate the betas and Kalman hyperparams against role/impact
# on bridge players ≤ 2020. (see design notes)
# yea-that-s-a-good-quizzical-biscuit.md

# Denominators we model as state-space stats themselves (analogous to cdarko
# residualizing player_poss). Their context residualization captures team
# scheme — playing time is partly outside the player's control.
PFF_DENOM_STATS: list[str] = [
    "pff_snap_counts_offense",
    "pff_snap_counts_defense",
    "pff_snap_counts_pass_rush",
    "pff_snap_counts_coverage",
    "pff_snap_counts_run",
    "pff_snap_counts_corner",   # DB sub-position
    "pff_snap_counts_slot",     # DB sub-position
    "pff_snap_counts_fs",       # DB sub-position
    "pff_routes",
    "pff_targets",
    "pff_attempts",
    "pff_dropbacks",
    "pff_total_attempts",   # K
    "pff_passing_snaps",    # QB
    "pff_true_pass_set_snap_counts_pass_block",  # OL
]

# Raw counts excluded from state-space modeling — their per-snap rate
# (already in PFF_GRADE_FEATURES_BY_GROUP via _add_season_level_rates)
# carries the meaningful signal; the raw count is volume-only.
_PFF_RAW_COUNT_EXCLUDES: frozenset[str] = frozenset({
    "pff_total_pressures", "pff_sacks", "pff_hits", "pff_hurries",
    "pff_tackles_for_loss", "pff_forced_fumbles", "pff_interceptions",
    "pff_pass_break_ups", "pff_forced_incompletes",
    "pff_pressures_allowed", "pff_sacks_allowed",
    "pff_hurries_allowed", "pff_hits_allowed",
    "pff_true_pass_set_pressures_allowed", "pff_true_pass_set_sacks_allowed",
    "pff_true_pass_set_hurries_allowed", "pff_true_pass_set_hits_allowed",
    "pff_avoided_tackles", "pff_yards_after_contact", "pff_first_downs",
    "pff_scrambles", "pff_inline_snaps", "pff_slot_snaps",
    "pff_man_snap_counts_coverage", "pff_zone_snap_counts_coverage",
    "pff_pat_attempts", "pff_fifty_attempts", "pff_forty_attempts",
    "pff_thirty_attempts", "pff_twenty_attempts",
    "pff_pass_rush_opp", "pff_run_stop_opp",
    "pff_blocks", "pff_long", "pff_touchbacks", "pff_inside_twenties",
    "pff_attempts_with_hangtime", "pff_total_hangtime",
    "pff_hit_as_threw", "pff_epa",  # raw EPA total; positive_epa_percent kept
})

# Which denominator-stats apply to each position group (controls inclusion
# in PFF_STATS_TO_MODEL[pos] when they're not already in PFF_GRADE_FEATURES_BY_GROUP)
_PFF_DENOM_RELEVANCE: dict[str, frozenset[str]] = {
    "pff_snap_counts_offense":  frozenset({"QB", "RB", "WR", "TE", "OL"}),
    "pff_snap_counts_defense":  frozenset({"DL", "EDGE", "LB", "DB"}),
    "pff_snap_counts_pass_rush": frozenset({"DL", "EDGE", "LB"}),
    "pff_snap_counts_coverage": frozenset({"LB", "DB"}),
    "pff_snap_counts_run":      frozenset({"DL", "EDGE", "LB"}),
    "pff_snap_counts_corner": frozenset({"DB"}),
    "pff_snap_counts_slot":   frozenset({"DB"}),
    "pff_snap_counts_fs":     frozenset({"DB"}),
    "pff_routes":   frozenset({"RB", "WR", "TE"}),
    "pff_targets":  frozenset({"RB", "WR", "TE"}),
    "pff_attempts": frozenset({"QB", "RB", "P"}),
    "pff_dropbacks": frozenset({"QB"}),
    "pff_total_attempts": frozenset({"K"}),
    "pff_passing_snaps": frozenset({"QB"}),
    "pff_true_pass_set_snap_counts_pass_block": frozenset({"OL"}),
}


def _build_pff_stats_to_model() -> dict[str, list[str]]:
    """Build the per-pos PFF stat list from PFF_GRADE_FEATURES_BY_GROUP minus
    raw counts, plus the relevant denominators. Computed once at import time."""
    out: dict[str, list[str]] = {}
    for pos, feats in PFF_GRADE_FEATURES_BY_GROUP.items():
        kept = [s for s in feats if s not in _PFF_RAW_COUNT_EXCLUDES]
        for d in PFF_DENOM_STATS:
            if d in kept:
                continue
            if pos in _PFF_DENOM_RELEVANCE.get(d, frozenset()):
                kept.append(d)
        out[pos] = kept
    return out


PFF_STATS_TO_MODEL: dict[str, list[str]] = _build_pff_stats_to_model()

# Per-stat denominator = observation weight column (analog of game-grain
# RATE_STAT_ATTEMPT_COL + snap routing). None means the stat IS its own
# denominator and gets uniform per-season weight (cdarko player_poss pattern).
# Default fallback (any stat not listed): pff_snap_counts_offense for offense
# pos groups, pff_snap_counts_defense for defense pos groups (resolved at runtime).
PFF_DENOM_COL_BY_STAT: dict[str, str | None] = {
    # ── QB ─────────────────────────────────────────────────────────────────
    "pff_grades_pass":                     "pff_dropbacks",
    "pff_grades_offense":                  "pff_snap_counts_offense",
    "pff_btt_rate":                        "pff_dropbacks",
    "pff_twp_rate":                        "pff_dropbacks",
    "pff_accuracy_percent":                "pff_attempts",
    "pff_sack_percent":                    "pff_dropbacks",
    "pff_completion_percent":              "pff_attempts",
    "pff_avg_time_to_throw":               "pff_dropbacks",
    "pff_drop_rate":                       "pff_targets",
    "pff_pressure_to_sack_rate":           "pff_dropbacks",
    "pff_qb_rating":                       "pff_attempts",
    "pff_pressure_grades_pass":            "pff_dropbacks",
    "pff_no_pressure_grades_pass":         "pff_dropbacks",
    "pff_pressure_completion_percent":     "pff_attempts",
    "pff_no_pressure_completion_percent":  "pff_attempts",
    "pff_pressure_sack_percent":           "pff_dropbacks",
    "pff_no_pressure_sack_percent":        "pff_dropbacks",
    "pff_blitz_grades_pass":               "pff_dropbacks",
    "pff_no_blitz_grades_pass":            "pff_dropbacks",
    "pff_blitz_btt_rate":                  "pff_dropbacks",
    "pff_no_blitz_btt_rate":               "pff_dropbacks",
    "pff_positive_epa_percent":            "pff_dropbacks",
    # ── RB / receivers ────────────────────────────────────────────────────
    # Rush-attempt denominators use the coalesced `pff_rush_att_inferred`
    # (raw rushing-report attempts, falling back to yards_after_contact /
    # yco_attempt) rather than `pff_attempts` directly — the raw column is
    # report-scoped and reads as pass attempts on QB rows.
    "pff_grades_run":                      "pff_rush_att_inferred",
    "pff_elusive_rating":                  "pff_rush_att_inferred",
    "pff_breakaway_percent":               "pff_rush_att_inferred",
    "pff_yco_attempt":                     "pff_rush_att_inferred",
    "pff_grades_pass_route":               "pff_routes",
    "pff_yprr":                            "pff_routes",
    "pff_grades_hands_drop":               "pff_targets",
    "pff_targeted_qb_rating":              "pff_targets",
    "pff_yards_after_catch_per_reception": "pff_targets",
    "pff_yards_per_reception":             "pff_targets",
    "pff_avg_depth_of_target":             "pff_targets",
    "pff_man_grades_pass_route":           "pff_routes",
    "pff_zone_grades_pass_route":          "pff_routes",
    "pff_man_yprr":                        "pff_routes",
    "pff_zone_yprr":                       "pff_routes",
    "pff_man_targeted_qb_rating":          "pff_targets",
    "pff_zone_targeted_qb_rating":         "pff_targets",
    "pff_route_rate":                      "pff_snap_counts_offense",
    "pff_slot_rate":                       "pff_snap_counts_offense",
    "pff_wide_rate":                       "pff_snap_counts_offense",
    "pff_inline_rate":                     "pff_snap_counts_offense",
    "pff_contested_catch_rate":            "pff_targets",
    "pff_catch_rate":                      "pff_targets",
    "pff_grades_pass_block":               "pff_snap_counts_offense",
    "pff_pbe":                             "pff_snap_counts_offense",
    "pff_grades_run_block":                "pff_snap_counts_offense",
    "pff_pass_block_percent":              "pff_snap_counts_offense",
    # ── OL ─────────────────────────────────────────────────────────────────
    "pff_gap_grades_run_block":            "pff_snap_counts_offense",
    "pff_zone_grades_run_block":           "pff_snap_counts_offense",
    "pff_true_pass_set_grades_pass_block": "pff_true_pass_set_snap_counts_pass_block",
    "pff_true_pass_set_pbe":               "pff_true_pass_set_snap_counts_pass_block",
    # ── DL / LB ───────────────────────────────────────────────────────────
    "pff_grades_defense":                  "pff_snap_counts_defense",
    "pff_grades_pass_rush_defense":        "pff_snap_counts_pass_rush",
    "pff_grades_run_defense":              "pff_snap_counts_run",
    "pff_pass_rush_win_rate":              "pff_snap_counts_pass_rush",
    "pff_prp":                             "pff_snap_counts_pass_rush",
    "pff_grades_tackle":                   "pff_snap_counts_defense",
    "pff_pass_rush_percent":               "pff_snap_counts_defense",
    "pff_stop_percent":                    "pff_snap_counts_defense",
    "pff_avg_depth_of_tackle":             "pff_snap_counts_defense",
    "pff_missed_tackle_rate":              "pff_snap_counts_defense",
    # ── LB / DB coverage ───────────────────────────────────────────────────
    "pff_grades_coverage_defense":         "pff_snap_counts_coverage",
    "pff_qb_rating_against":               "pff_snap_counts_coverage",
    "pff_forced_incompletion_rate":        "pff_snap_counts_coverage",
    "pff_man_grades_coverage_defense":     "pff_snap_counts_coverage",
    "pff_zone_grades_coverage_defense":    "pff_snap_counts_coverage",
    "pff_man_qb_rating_against":           "pff_snap_counts_coverage",
    "pff_zone_qb_rating_against":          "pff_snap_counts_coverage",
    "pff_man_forced_incompletion_rate":    "pff_snap_counts_coverage",
    "pff_zone_forced_incompletion_rate":   "pff_snap_counts_coverage",
    "pff_yards_per_coverage_snap":         "pff_snap_counts_coverage",
    "pff_coverage_snaps_per_target":       "pff_snap_counts_coverage",
    "pff_coverage_snaps_per_reception":    "pff_snap_counts_coverage",
    # ── Synthesized rates from _add_season_level_rates ─────────────────────
    "pff_sacks_per_pr_snap":               "pff_snap_counts_pass_rush",
    "pff_hits_per_pr_snap":                "pff_snap_counts_pass_rush",
    "pff_hurries_per_pr_snap":             "pff_snap_counts_pass_rush",
    "pff_total_pressures_per_pr_snap":     "pff_snap_counts_pass_rush",
    "pff_tackles_for_loss_per_def_snap":   "pff_snap_counts_defense",
    "pff_forced_fumbles_per_def_snap":     "pff_snap_counts_defense",
    "pff_interceptions_per_def_snap":      "pff_snap_counts_defense",
    "pff_pass_break_ups_per_cov_snap":     "pff_snap_counts_coverage",
    "pff_forced_incompletes_per_cov_snap": "pff_snap_counts_coverage",
    "pff_pressures_allowed_per_off_snap":  "pff_snap_counts_offense",
    "pff_sacks_allowed_per_off_snap":      "pff_snap_counts_offense",
    "pff_hurries_allowed_per_off_snap":    "pff_snap_counts_offense",
    "pff_hits_allowed_per_off_snap":       "pff_snap_counts_offense",
    "pff_true_pass_set_pressures_allowed_rate": "pff_true_pass_set_snap_counts_pass_block",
    "pff_true_pass_set_sacks_allowed_rate":     "pff_true_pass_set_snap_counts_pass_block",
    "pff_true_pass_set_hurries_allowed_rate":   "pff_true_pass_set_snap_counts_pass_block",
    "pff_true_pass_set_hits_allowed_rate":      "pff_true_pass_set_snap_counts_pass_block",
    "pff_routes_per_off_snap":             "pff_snap_counts_offense",
    "pff_targets_per_off_snap":            "pff_snap_counts_offense",
    "pff_avoided_tackles_per_attempt":     "pff_attempts",
    "pff_man_zone_route_grade_delta":      "pff_routes",
    "pff_man_zone_yprr_delta":             "pff_routes",
    "pff_man_zone_qb_rating_against_delta": "pff_snap_counts_coverage",
    "pff_man_zone_coverage_grade_delta":   "pff_snap_counts_coverage",
    # ── K / P ──────────────────────────────────────────────────────────────
    "pff_grades_fgep_kicker":              "pff_total_attempts",
    "pff_total_percent":                   "pff_total_attempts",
    "pff_thirty_percent":                  "pff_total_attempts",
    "pff_forty_percent":                   "pff_total_attempts",
    "pff_fifty_percent":                   "pff_total_attempts",
    "pff_twenty_percent":                  "pff_total_attempts",
    "pff_pat_percent":                     "pff_total_attempts",
    "pff_grades_punter":                   "pff_attempts",
    "pff_average_net_yards":               "pff_attempts",
    "pff_average_yards_per_attempt":       "pff_attempts",
    "pff_average_hangtime":                "pff_attempts",
    # ── Denominators themselves: uniform weight (cdarko player_poss) ───────
    **{d: None for d in PFF_DENOM_STATS},
}

# Reference denominator counts (analog of KALMAN_REF_SNAPS=70 per game).
# Tuned to ≈ "one full season's worth" — observation noise scales as
# obs_noise × ref_denom / actual_denom, so a typical full season gets
# weight ≈ 1 and partial seasons weight < 1.
PFF_REF_DENOM_BY_STAT: dict[str, float] = {
    "pff_snap_counts_offense":                700.0,
    "pff_snap_counts_defense":                700.0,
    "pff_snap_counts_pass_rush":              350.0,
    "pff_snap_counts_coverage":               400.0,
    "pff_snap_counts_run":                    300.0,
    "pff_routes":                             300.0,
    "pff_targets":                             80.0,
    "pff_attempts":                           300.0,
    "pff_dropbacks":                          400.0,
    "pff_passing_snaps":                      450.0,
    "pff_total_attempts":                      25.0,
    "pff_true_pass_set_snap_counts_pass_block": 250.0,
}
PFF_REF_DENOM_DEFAULT: float = 500.0

# ── PFF Career-Triplet Bayesian Shrinkage ──────────────────────────────────────
# Per-(player, season) PFF observations are shrunk toward a position-pooled
# prior before the _mean / _peak / _last triplet aggregation in
# `load_pff_career_features`. The shrinkage weight is governed by the relevant
# denominator (`PFF_DENOM_COL_BY_STAT[stat]`):
#
#   shrunk_val = (n * obs + k * prior) / (n + k)
#
# where k = factor * PFF_REF_DENOM_BY_STAT[stat]. Two-family split: grade
# columns (PFF analyst-smoothed already) get less aggressive shrinkage than
# per-snap rates / percents / deltas (which can swing wildly on small samples).
# Stats with `PFF_DENOM_COL_BY_STAT[stat] is None` or no entry (raw counts,
# denominator stats themselves) are NOT shrunk — they're volume signals.
#
# 2026-05-12: Tuned to k=0.05/0.125 after comparing three regimes (k=0.10/0.25,
# k=0.05/0.125, min-denom-gate-0.25) on expanding-window CV MAE. With small-n
# K/P heads filtered out, k=0.05/0.125 was the only regime with a net mean
# CV-MAE improvement (mean −0.64%, median −3.00%, 15:8 win:loss ratio across
# 27 large-n heads). Holdout-Spearman swung wildly across all three regimes
# (single-fold noise on small per-position panels); CV MAE is the stable
# signal and the one we tune against.
PFF_SHRINKAGE_K_FACTOR_GRADES: float = 0.05
PFF_SHRINKAGE_K_FACTOR_RATE: float = 0.125

# Per-day decay default (matches game-grain DEFAULT_BETA[stat]=0.999).
# With ~365-day inter-season gaps this is ~0.694 effective per-season decay.
DEFAULT_BETA_PFF: dict[str, float] = {
    s: 0.999 for s in {st for sts in PFF_STATS_TO_MODEL.values() for st in sts}
}

# Inter-season gap for Q estimation (full year vs game-grain's 240-day NCAA gap)
PFF_SEASON_INTERSEASON_DAYS: float = 365.0

# Beta search bounds (per-day, same as game-grain BETA_BOUNDS)
PFF_BETA_BOUNDS: tuple[float, float] = (0.990, 1.0 - 1e-7)

# Bridge season down-weight (older seasons of bridge players matter less)
PFF_BRIDGE_SEASON_GAMMA: float = 0.8

# Season-grain context features for ElasticNet residualization. No opponent-
# rolling stats (those are game-grain-only); no weather/location.
# team_sp_overall_season dropped: collinear with off + def. team_pass_rate_season,
# opp_sp_offense_avg, opp_sp_defense_avg added 2026-05-15.
PFF_CONTEXT_FEATURES: list[str] = [
    # Conference tier per season (Power 4 / G5 / FCS reference)
    "conf_tier_power4", "conf_tier_g5",
    # Team-season SP+ (from team_ratings.parquet, joined on (school, season_year))
    "team_sp_offense_season", "team_sp_defense_season",
    # Team-season advanced game stats (from advanced_game_stats.parquet,
    # aggregated to (team, season_year) means)
    "team_off_ppa_season", "team_off_success_rate_season",
    "team_off_explosiveness_season", "team_off_stuff_rate_season",
    "team_def_ppa_season", "team_def_success_rate_season",
    "team_def_stuff_rate_season",
    # Team-season pass rate (team_pass_att / (team_pass_att + team_rush_att))
    "team_pass_rate_season",
    # Opponent strength-of-schedule: mean opponent SP+ on each side
    "opp_sp_offense_avg", "opp_sp_defense_avg",
    # League-season averages from cfb_season_averages.parquet
    "lg_off_ppa", "lg_off_success_rate", "lg_off_explosiveness",
    "lg_off_stuff_rate", "lg_def_ppa", "lg_def_success_rate",
    "lg_def_stuff_rate", "lg_pass_rate",
]

# Subset of PFF_CONTEXT_FEATURES that captures the player's own-team and
# schedule-quality context across their CFB seasons. Aggregated to per-player
# triplets (peak/trough/last/wavg) in load_pff_career_features() and fed into
# step 8 alongside the residualized PFF triplets, so the draft model retains
# awareness of the context the player operated in even after residualization
# removes it from the per-season grade. Excludes lg_* (year-level, no
# player-level signal — same value for every player in that year).
PFF_CONTEXT_TRIPLET_COLS: list[str] = [
    "conf_tier_power4", "conf_tier_g5",
    "team_sp_offense_season", "team_sp_defense_season",
    "team_off_ppa_season", "team_off_success_rate_season",
    "team_off_explosiveness_season", "team_off_stuff_rate_season",
    "team_def_ppa_season", "team_def_success_rate_season",
    "team_def_stuff_rate_season",
    "team_pass_rate_season",
    "opp_sp_offense_avg", "opp_sp_defense_avg",
]

# NOTE: the season-grain PFF state-space path (USE_PFF_STATE_SPACE flag,
# steps 04s/05s/06s + calibration siblings 04f/04g, src/models/
# pff_kalman_runner.py + season_panel.py) was removed 2026-07-21 after three
# failed A/Bs (2026-05-08 initial disable, 2026-05-13 slim retest at
# +3.28% mean CV MAE, 2026-05-15 contextual-no-state-space at +1.67%).
# Recover via git history if ever needed.

# When True, run steps 4.7 (PFF contextual fit) + 5.5 (residualize) and feed
# the residualized seasons into load_pff_career_features() so the static
# triplets (mean / peak / trough / last) are computed on context-adjusted PFF
# values. Defensive R²-gate at the (pos × stat) level: cv_r2 < 0.01 → fall
# through to raw (handled in step 05s, matching step 5's gate).
#
# Also gates emission of context triplets (ctx_<col>_peak / _trough / _last
# / _wavg) over PFF_CONTEXT_TRIPLET_COLS, so the draft model can weight the
# context the player operated in alongside the player-specific residual.
#
# 2026-05-15: tested end-to-end retune with USE_PFF_CONTEXTUAL=True. Result:
# +1.67% mean CV MAE across 30 heads (12W / 18L), and Boruta rejected
# 0/1,500 ctx_* + 0/~5,000 _trough audit decisions across all heads — the
# new features acted as pure dilution. Wins clustered on positions whose
# careers span varied team contexts (TE_impact_wav -19.65%, WR_role
# -10.54%); losses on OL_role (+10.84%) and a handful of small-CV heads.
# Reverted: flag flipped to False; models restored from
# draft_models.bak_20260514_085615 + global_draft_models.bak_totals_3seed.
# Code wiring is preserved so the option can be re-enabled cleanly.
USE_PFF_CONTEXTUAL: bool = False
# NOTE: the 04s/05s season-grain scripts this flag once gated in the runner
# were removed 2026-07-22 (recover via git history before flipping this to
# True); the flag still gates step 8's adjusted-vs-raw PFF season read.


# ── Play-Grain Contextual Layer ───────────────────────────────────────────────
# Per-PLAY residualization (slot 4.5 / 5.3 in the pipeline).
# Steps 1.4 (ingest plays + play_stats), 1.5 (OSM stadium orientation scrape),
# 1.6 (join), 4.5 (per-stat LightGBM with expanding-window CV), 5.3 (per-play
# residuals + per-(player,game) aggregation + catch-all rows for non-
# participating active-roster players).
# Aggregated mean/total residuals feed step 6 as additional Kalman + EWMA
# observation streams.

ENABLE_PLAY_LAYER: bool = True

# CFBD play-by-play has solid FBS coverage from ~2008 onward; 2005-2007 is
# spotty (~50% of plays missing play_stats credits).
PLAY_START_YEAR: int = 2008

# Stat types emitted by CFBD's PlaysApi.get_play_stat_types. Stored verbatim
# from the API (Title Case) for matrix lookups; lowercased + snake_case at
# the join step (see PLAY_STAT_COL).
ALL_PLAY_STAT_TYPES: list[str] = [
    # `Incompletion` is in the universe purely so step 1f pivots its column
    # (needed for the Phase 3a `Pass Completion Outcome` derivation — the
    # QB-attempted mask is the union of completion/incompletion/int-thrown/
    # sack-taken). Its own R² model will FAIL the gate (constant target on
    # credited rows = no variance), which is fine — step 5p skips it.
    "Rush", "Completion", "Incompletion", "Reception", "Target",
    "Tackle", "Sack", "Sack Taken", "QB Hurry", "Pass Breakup",
    "Interception", "Interception Thrown",
    "Fumble", "Fumble Forced", "Fumble Recovered",
    "Touchdown",
    "Field Goal Attempt", "Field Goal Made",
    "Field Goal Block", "FG Attempt Blocked",
    "Punt", "Punt Block",
    "Kickoff", "Kickoff Return", "Onside Kick",
    # Phase 3a (2026-05-23) — hierarchical decomposition: outcome stats
    # (binary 1/0 conditional on a pre-condition being credited) and
    # TD/fumble-by-type magnitude stats. Derived in step 1f from existing
    # stat columns + play_type. Pre-condition met → 0 or 1; otherwise NaN
    # so step 4p's credited-filter skips them naturally.
    # Pass-attempt outcomes (QB-side): pre-condition = QB had any of
    # {completion, incompletion, interception_thrown, sack_taken} credited.
    "Pass Completion Outcome",
    "Pass Interception Outcome",
    "Pass Sack Outcome",
    # Target outcomes (receiver-side) removed 2026-05-23 after empirical
    # test:
    #   `Target Reception Outcome`: modelable (Target ∪ Reception = ~647k
    #     targeted rows, 74% caught), but cv_r2 = -0.027 — pre-snap context
    #     (down/distance/opp_rolling) genuinely can't predict catch outcome.
    #     Signal lives in receiver-defender matchup detail CFBD doesn't expose.
    #   `Target Interception Outcome`: structurally un-modelable — CFBD
    #     credits ZERO receiver-side stats on INT play_types (only the
    #     defender and the QB are credited). Would need PFF coverage data.
    # TD by type: pre-condition = relevant skill stat credited.
    "Passing Touchdown Outcome",
    "Rushing Touchdown Outcome",
    "Receiving Touchdown Outcome",
    # Fumble by type: pre-condition = ball-carrying event credited.
    "Rushing Fumble Outcome",
    "Receiving Fumble Outcome",
]

# Stat types whose `stat` value is a magnitude (yards / kick distance), not
# an indicator. Used in step 1.6 to coerce missing-value-on-credited-row to 0
# vs to mean for these continuous targets.
PLAY_CONTINUOUS_STATS: set[str] = {
    "Rush", "Completion", "Reception", "Sack Taken",
    "Field Goal Attempt", "Field Goal Made",
}

# Play-grain stats whose raw per-(player, game) rate carries skill signal
# NOT already in the per-game pipeline (totals + rates from MODELED_STATS).
# Step 5p's policy: if the per-play contextual model PASSES the R² gate,
# emit the context-adjusted residual (additive over the raw rate). If it
# FAILS, fall back to the raw mean of the 0/1 outcome over precondition-
# credited rows — `adjusted_play_raw_mean_<stat>`. Stats outside this list
# that fail R² are dropped (either redundant with game-grain or unsupported).
#
# Currently:
#   - Phase 3a TD-by-type outcomes (PASS R²~0.31-0.39) → emit residual
#   - Phase 3a fumble-by-type outcomes (FAIL R²~0) → emit raw rate
#   - Pass Completion/Interception/Sack Outcomes (redundant with game-grain
#     completion_pct, interceptions/pass_att, sacks_taken/pass_att) → NOT
#     in this list, so they drop on R² fail (Pass Completion Outcome
#     currently passes barely at 0.011 so its residual is emitted)
PLAY_RAW_RATE_FALLBACK_STATS: set[str] = {
    "Passing Touchdown Outcome",     # game-grain has TD totals, not TD-rate-by-type
    "Rushing Touchdown Outcome",
    "Receiving Touchdown Outcome",
    "Rushing Fumble Outcome",        # game-grain has fumble total, not split by play_type
    "Receiving Fumble Outcome",
}

# Wide-column naming: Title Case → snake_case. "FG Attempt Blocked" → "fg_attempt_blocked".
PLAY_STAT_COL: dict[str, str] = {
    s: s.lower().replace(" ", "_") for s in ALL_PLAY_STAT_TYPES
}

# ── Play-type buckets ──────────────────────────────────────────────────────
# Buckets are coarser than the 49-row CFBD play_type enumeration because
# eligibility only needs to distinguish scrimmage vs. ST-specialist plays.

SCRIMMAGE_PLAY_TYPES: set[str] = {
    "Rush", "Pass", "Pass Reception", "Pass Incompletion", "Sack",
    "Rushing Touchdown", "Passing Touchdown",
    "Pass Interception Return", "Interception", "Interception Return Touchdown",
    "Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown",
    "Two Point Rush", "2pt Conversion", "Defensive 2pt Conversion",
    "Safety",
}
KICKOFF_PLAY_TYPES: set[str] = {
    "Kickoff", "Kickoff Return (Offense)", "Kickoff Return Touchdown",
}
PUNT_PLAY_TYPES: set[str] = {
    "Punt", "Blocked Punt", "Blocked Punt Touchdown", "Punt Return Touchdown",
}
FG_PLAY_TYPES: set[str] = {
    "Field Goal Good", "Field Goal Missed", "Blocked Field Goal",
    "Blocked Field Goal Touchdown",
    "Missed Field Goal Return", "Missed Field Goal Return Touchdown",
}
XP_PLAY_TYPES: set[str] = {
    "Extra Point Good", "Extra Point Missed",
}
NON_ACTION_PLAY_TYPES: set[str] = {
    "Penalty", "End Period", "End of Game", "Timeout",
    "Uncategorized", "placeholder",
}

# Per-position eligibility for the "didn't play" catch-all expansion.
# Each set names play_types where the position COULD be on the field. The
# eligibility join in step 1.6 additionally restricts to plays where the
# player's team was on the relevant side (offense vs defense).
POSITION_ELIGIBLE_PLAY_TYPES: dict[str, set[str]] = {
    "QB": SCRIMMAGE_PLAY_TYPES,
    "RB": SCRIMMAGE_PLAY_TYPES,
    "WR": SCRIMMAGE_PLAY_TYPES,
    "TE": SCRIMMAGE_PLAY_TYPES,
    "DL": SCRIMMAGE_PLAY_TYPES,
    "EDGE": SCRIMMAGE_PLAY_TYPES,
    "LB": SCRIMMAGE_PLAY_TYPES,
    "DB": SCRIMMAGE_PLAY_TYPES,
    "K":  FG_PLAY_TYPES | XP_PLAY_TYPES | KICKOFF_PLAY_TYPES,
    "P":  PUNT_PLAY_TYPES,
    # OL excluded — never appears as `athlete` in play_stats; stays on
    # box-score / PFF surface only.
}

# Position side (offense / defense). Used to gate catch-all expansion to
# plays where the team was on the right side of the ball.
# K and P are treated as offense because CFBD labels the kicking team's side
# as `offense` on kickoffs / punts / FG attempts / XPs (the unit on offense
# brings out the kicker after a score / on 4th down).
POSITION_SIDE: dict[str, str] = {
    "QB": "offense", "RB": "offense", "WR": "offense", "TE": "offense",
    "DL": "defense", "EDGE": "defense", "LB": "defense", "DB": "defense",
    "K":  "offense",
    "P":  "offense",
}

# Which position groups each play-grain SKILL stat is a real production signal
# for. Step 8 gates a head's play-grain features (kalman/ewma over
# play_*resid_<stat>) to the listed positions; every other position's residual
# for that stat is degenerate noise (a DB's `reception`, a K's `rush`) that
# only slipped past the >99% dominant-value filter because shrinkage/EWMA
# spread the constant across low-order digits. Defenders, OL and P produce
# none of these offensive stats and so get zero play-grain skill features.
# Membership is football role, cross-checked against per-position
# dominant-value shares (2026-07-24): QB passing 0.32 vs RB/WR/TE 0.90+ noise;
# WR/TE `rush` 0.25/0.34 (jet sweeps) is real and kept.
PLAY_SKILL_STAT_POSITIONS: dict[str, frozenset[str]] = {
    # Passing production — credited to the passer.
    "completion":                 frozenset({"QB"}),
    "pass_completion_outcome":    frozenset({"QB"}),
    "passing_touchdown_outcome":  frozenset({"QB"}),
    # Receiving production — pass catchers (mobile-QB receptions show signal).
    "reception":                  frozenset({"QB", "RB", "WR", "TE"}),
    "receiving_fumble_outcome":   frozenset({"RB", "WR", "TE"}),
    "receiving_touchdown_outcome": frozenset({"RB", "WR", "TE"}),
    # Rushing production — ball carriers (WR/TE end-arounds included).
    "rush":                       frozenset({"QB", "RB", "WR", "TE"}),
    "rushing_fumble_outcome":     frozenset({"QB", "RB", "WR", "TE"}),
    "rushing_touchdown_outcome":  frozenset({"QB", "RB", "WR", "TE"}),
    # Place kicking.
    "field_goal_attempt":         frozenset({"K"}),
    "field_goal_made":            frozenset({"K"}),
}

# Coarse play_type × stat_type "physically possible" matrix. Used by step 4p
# to mask training rows for impossible combinations and by step 5p to zero
# out predicted residuals on ineligible plays. Coarse on purpose — exact
# combos are sparse, so we lean on the bucket buckets above.
def _build_play_type_stat_eligibility() -> dict[str, set[str]]:
    elig: dict[str, set[str]] = {}
    # Scrimmage offensive plays
    elig["Rush"] = {"Rush", "Touchdown", "Fumble", "Fumble Forced",
                    "Fumble Recovered", "Tackle"}
    elig["Pass"] = {"Completion", "Incompletion", "Reception", "Target",
                    "Pass Breakup", "Sack", "Sack Taken", "QB Hurry",
                    "Interception", "Interception Thrown", "Touchdown",
                    "Fumble", "Fumble Forced", "Fumble Recovered", "Tackle"}
    elig["Pass Reception"] = {"Completion", "Reception", "Target", "Touchdown",
                              "Fumble", "Fumble Forced", "Fumble Recovered",
                              "Tackle"}
    elig["Pass Incompletion"] = {"Incompletion", "Target", "Pass Breakup",
                                 "QB Hurry"}
    elig["Sack"] = {"Sack", "Sack Taken", "QB Hurry", "Fumble",
                   "Fumble Forced", "Fumble Recovered", "Tackle"}
    elig["Rushing Touchdown"] = {"Rush", "Touchdown", "Tackle"}
    elig["Passing Touchdown"] = {"Completion", "Reception", "Target",
                                 "Touchdown", "Tackle"}
    elig["Interception"] = {"Interception", "Interception Thrown", "Tackle"}
    elig["Pass Interception Return"] = {"Interception", "Interception Thrown",
                                        "Tackle", "Fumble", "Fumble Recovered",
                                        "Touchdown"}
    elig["Interception Return Touchdown"] = elig["Pass Interception Return"] | {"Touchdown"}
    elig["Fumble Recovery (Own)"] = {"Fumble", "Fumble Forced", "Fumble Recovered", "Tackle"}
    elig["Fumble Recovery (Opponent)"] = elig["Fumble Recovery (Own)"] | {"Touchdown"}
    elig["Fumble Return Touchdown"] = elig["Fumble Recovery (Opponent)"]
    elig["Safety"] = {"Tackle", "Sack", "Sack Taken", "Touchdown"}
    elig["Two Point Rush"] = {"Rush", "Touchdown", "Tackle"}
    elig["2pt Conversion"] = {"Completion", "Reception", "Target",
                              "Touchdown", "Rush", "Tackle"}
    elig["Defensive 2pt Conversion"] = {"Touchdown", "Tackle"}
    # Kickoffs
    elig["Kickoff"] = {"Kickoff", "Onside Kick", "Tackle", "Kickoff Return",
                       "Fumble", "Fumble Forced", "Fumble Recovered", "Touchdown"}
    elig["Kickoff Return (Offense)"] = elig["Kickoff"]
    elig["Kickoff Return Touchdown"] = elig["Kickoff"] | {"Touchdown"}
    # Punts
    elig["Punt"] = {"Punt", "Punt Block", "Tackle", "Fumble",
                    "Fumble Forced", "Fumble Recovered", "Touchdown"}
    elig["Blocked Punt"] = {"Punt", "Punt Block", "Tackle", "Fumble",
                            "Fumble Recovered", "Touchdown"}
    elig["Blocked Punt Touchdown"] = elig["Blocked Punt"] | {"Touchdown"}
    elig["Punt Return Touchdown"] = elig["Punt"] | {"Touchdown"}
    # FG / XP
    elig["Field Goal Good"] = {"Field Goal Attempt", "Field Goal Made"}
    # NOTE: there is no "Field Goal Missed" stat type — a miss is just a
    # "Field Goal Attempt" credit with no "Field Goal Made".
    elig["Field Goal Missed"] = {"Field Goal Attempt"}
    elig["Blocked Field Goal"] = {"Field Goal Attempt",
                                  "FG Attempt Blocked", "Field Goal Block",
                                  "Tackle"}
    elig["Blocked Field Goal Touchdown"] = elig["Blocked Field Goal"] | {"Touchdown"}
    elig["Missed Field Goal Return"] = elig["Field Goal Missed"] | {"Tackle"}
    elig["Missed Field Goal Return Touchdown"] = elig["Missed Field Goal Return"] | {"Touchdown"}
    elig["Extra Point Good"] = set()
    elig["Extra Point Missed"] = set()

    # Phase 3a derived outcome stats — propagate eligibility from their
    # underlying pre-condition stats.
    # Pass-attempt outcomes piggyback on Completion/Incompletion/Sack
    # Taken/Interception Thrown play_type sets:
    _pass_attempt_outcomes = {
        "Pass Completion Outcome",
        "Pass Interception Outcome",
        "Pass Sack Outcome",
    }
    _pass_attempt_play_types = ["Pass", "Pass Reception", "Pass Incompletion",
                                 "Passing Touchdown", "Sack", "Interception",
                                 "Pass Interception Return",
                                 "Interception Return Touchdown", "Safety",
                                 "2pt Conversion"]
    for pt in _pass_attempt_play_types:
        if pt in elig:
            elig[pt] = elig[pt] | _pass_attempt_outcomes

    # (Target outcomes removed 2026-05-23 — see ALL_PLAY_STAT_TYPES note.)

    # TD/fumble by type — propagate where the relevant skill stat is creditable:
    # Passing/Receiving TD + Receiving Fumble: pass play_types where
    # Completion/Reception possible.
    _pass_skill_play_types = ["Pass", "Pass Reception", "Passing Touchdown",
                              "2pt Conversion"]
    _pass_skill_outcomes = {"Passing Touchdown Outcome",
                            "Receiving Touchdown Outcome",
                            "Receiving Fumble Outcome"}
    for pt in _pass_skill_play_types:
        if pt in elig:
            elig[pt] = elig[pt] | _pass_skill_outcomes
    # Rushing TD + Rushing Fumble: rush play_types where Rush is creditable.
    _rush_play_types = ["Rush", "Rushing Touchdown", "Two Point Rush",
                        "2pt Conversion"]
    _rush_outcomes = {"Rushing Touchdown Outcome", "Rushing Fumble Outcome"}
    for pt in _rush_play_types:
        if pt in elig:
            elig[pt] = elig[pt] | _rush_outcomes

    # Non-action: no stats
    for pt in ["Penalty", "End Period", "End of Game", "Timeout",
               "Uncategorized", "placeholder"]:
        elig[pt] = set()
    return elig

PLAY_TYPE_STAT_ELIGIBILITY: dict[str, set[str]] = _build_play_type_stat_eligibility()

# ── Play-grain feature surface ─────────────────────────────────────────────
# Context features for step 4p LightGBM. Mix of play-level (varies within
# game) and game-level (static; carried from adjusted_gamelogs.parquet).
# `play_type` and `position_group` are categorical; LightGBM handles via
# pd.Categorical / categorical_feature kwarg.

PLAY_CONTEXT_FEATURES: list[str] = [
    # Play-grain situation (numerics — no bucketing per locked decision).
    # NOTE: `play_type` deliberately excluded from features 2026-05-22 — we
    # want residuals to capture "given the game state (down/distance/score/
    # field position/etc.), how much did the player exceed expectation",
    # WITHOUT conditioning away the coach's play-call selection. If an RB is
    # good and the coach therefore runs more often in favorable spots, that
    # selection effect SHOULD flow into the residual as player-quality signal.
    # `play_type` remains in player_plays.parquet for the eligibility mask in
    # step 4p and step 5p (filters to physically-possible play types per stat).
    "down", "distance", "yards_to_goal",
    "period", "clock_seconds", "score_diff",
    "offense_timeouts", "defense_timeouts",
    # `is_garbage_time` removed 2026-05-23 — derivable from score_diff +
    # period; redundant. Still computed in step 1f for potential v2 weighting.
    "position_group",      # categorical
    # Game-grain venue / weather (already on adjusted_gamelogs)
    "grass",               # RAW value
    "game_indoors",
    # `altitude_ft` removed 2026-05-23 — use altitude_diff_from_home_ft only
    # (the diff is the meaningful travel-shock signal; absolute altitude
    # mostly identifies which stadium).
    "temperature", "dew_point", "humidity",
    "wind_speed",
    # `wind_parallel` / `wind_perpendicular` were listed here but have NEVER
    # existed in player_plays.parquet: the decomposition in 01f needs
    # data/raw/venue_orientation.parquet, and the 01e OSM scrape that builds it
    # has never completed (the 2026-05-20 run was killed at 150/747 venues).
    # Because 04p resolves features with `[c for c in PLAY_CONTEXT_FEATURES if c
    # in df.columns]`, they were silently dropped every run — the config asserted
    # a feature surface the model never had. Removed 2026-07-25 so the list tells
    # the truth (and so the play_context_features fingerprint reflects reality).
    #
    # Re-adding them means fixing 01e first, and the public Overpass API is not a
    # viable path: it returns 406/429 on a single trivial query, bulk bbox queries
    # time out, and the old per-venue × per-tag loop cost ~54s/venue with
    # rate-limit failures miscounted as "missing" (found rate decayed 76% → 32%).
    # The clean route is an offline Geofabrik extract queried with pyrosm/osmium.
    # Deliberately not pursued: FG attempts are conditional on wind, so kicking
    # weather features already carry selection bias (see the FG-Missed note near
    # PLAY_RAW_RATE_FALLBACK_STATS — wind_speed ↑ → miss-residual ↓, backwards
    # from physics), which is a poor payoff for a multi-GB dependency plus a
    # 2.5M-row player_plays rebuild.
    "precipitation", "snowfall", "pressure",
    # Game-grain rolling opponent context (carried from adjusted_gamelogs)
    "opp_rolling_def_ppa", "opp_rolling_off_ppa",
    "opp_rolling_def_success_rate", "opp_rolling_def_stuff_rate",
    "opp_rolling_off_success_rate", "opp_rolling_off_explosiveness",
    "opp_rolling_def_explosiveness",
    "opp_rolling_havoc_total", "opp_rolling_havoc_front_seven",
    "opp_rolling_havoc_db",
    "opp_rolling_def_rushing_stuff_rate", "opp_rolling_def_rushing_power_success",
    "opp_rolling_def_line_yards_avg", "opp_rolling_def_second_level_yards_avg",
    "opp_rolling_def_open_field_yards_avg",
    "opp_rolling_def_success_rate_passing_downs",
    "opp_rolling_def_success_rate_standard_downs",
    "opp_rolling_def_points_per_opportunity",
    # Game-grain teammate QB context (offensive heads only — LightGBM
    # handles missing via NaN-aware splits)
    "qb_rolling_passer_rating", "qb_rolling_completion_pct",
    "qb_rolling_ppa_avg_passing", "qb_rolling_ypa",
    "qb_rolling_td_rate", "qb_rolling_int_rate",
    # Own-team rolling scoring context (team_rolling_*) removed 2026-05-23
    # — these are own-team outcomes that conflate with player skill (a good
    # offense's RB will see better scoring environment; the residual would
    # then partial out a chunk of the skill we're trying to measure).
    # Location / travel / schedule context (carried from adjusted_gamelogs)
    "is_home", "is_neutral",
    "days_rest", "opp_days_rest",
    "game_hour", "tz_diff", "body_clock_hour", "week",
    # Venue diff: altitude only (surface_diff removed 2026-05-23 — `grass`
    # raw indicator already carries the surface signal).
    "altitude_diff_from_home_ft",
    # Season league environment
    "lg_off_success_rate", "lg_def_success_rate", "lg_pass_rate",
    # Conference / classification
    "conf_tier",
]
PLAY_CATEGORICAL_FEATURES: list[str] = ["position_group"]

# Restrict FG-related stat-model training to attempts within reasonable range
# (yards_to_goal ≤ this value). Outside the threshold, coaches rarely attempt
# FGs — those attempts are conditional on wind/timing/etc., introducing
# selection bias on weather features (high wind → coaches kick from closer,
# making the residual wind/miss relationship sign-flipped vs reality). Audit
# 2026-05-22 saw `wind_speed ↑ → miss-residual ↓` in the FG-Missed model.
FG_TRAINING_MAX_YARDS_TO_GOAL: int = 40  # ≈ 47-yard FG attempt

# Rate stats derived AT PER-GAME AGGREGATION (step 5.3), parallel to the
# game-grain RATE_STAT_ATTEMPT_COL pattern at config.py:582. Each entry maps
# the derived rate name → (numerator_stat, denominator_stat). The numerator
# is the sum of (residualized) stat over credited+eligible plays; the
# denominator is the count of eligible plays.
PLAY_RATE_STAT_NUMERATOR_DENOM: dict[str, tuple[str, str]] = {
    "play_completion_pct":  ("completion", "completion_attempt"),  # synth denom: completion+incompletion
    "play_catch_pct":       ("reception",  "target"),
    "play_fg_pct":          ("field_goal_made", "field_goal_attempt"),
    "play_yards_per_carry": ("rush",       "rush_attempt"),        # synth denom: rush events
    "play_yards_per_target":("reception",  "target"),              # uses rec-yards numerator
    "play_yards_per_punt":  ("punt",       "punt_attempt"),
}

# LightGBM hyperparameters for step 4p. Fixed defaults in v1; tuning will be
# added later via `--retune-only 4.5` honoring the same pattern as step 8.
LGBM_PLAY_PARAMS: dict = {
    "objective":            "regression",
    "metric":               "rmse",
    "num_leaves":           63,
    "min_data_in_leaf":     200,
    "learning_rate":        0.05,
    "feature_fraction":     0.8,
    "bagging_fraction":     0.8,
    "bagging_freq":         5,
    "lambda_l2":            1.0,
    "num_boost_round":      500,
    "early_stopping_round": 30,
    "verbose":              -1,
}

# NOTE (2026-07-20): PLAY_OBS_NOISE_BY_STAT / PLAY_DEFAULT_BETA removed —
# they were consumed nowhere and their keys ("play_mean_rush") never matched
# the actual stream names step 6 discovers ("play_mean_resid_rush"). Play-
# grain Kalman R is now estimated from within-player observation variance
# directly in step 06 (same estimator as 04d); play-stream decay betas fall
# through to DEFAULT_BETA's 0.999 fallback.

# Stadium orientation overrides (degrees, 0-180). Used by step 1.5 when the
# OSM polygon scrape can't resolve a venue (typically rebuilt stadia or
# venues with ambiguous boundary tags). Hand-curated. Empty by default;
# add (venue_id, orientation_deg) entries as gaps surface.
VENUE_ORIENTATION_OVERRIDES: dict[int, float] = {}

# ── Global-model score overrides (2026-07-21 blend A/B) ──────────────────────
# experiments/blend_ab_20260721_163302.json + blend_ab_fold_preds.parquet:
# the pooled global model (8b) beats per-position heads on nearly every
# (position × target) — role logloss 9/10 positions, impact_apy bridge-only
# Spearman 10/10 — with leave-one-fold-out blend weights collapsing to pure
# global. Production scope so far: the small-N positions the pooling was
# built for (K/P/TE), where variance is documented and the LOFO weights
# chose pure global for role + impact_apy (TE impact_wav stays per-position,
# the one head that beat its global counterpart). Broader adoption across
# the remaining positions is deliberately a separate decision.
# (position_group, score) → weight on the PER-POSITION score; 0.0 = take
# the global score wholesale.
# Phase 2 (2026-07-21): full adoption per calibrated tuning-fold gates
# (scratchpad phase2_verdicts; harnesses: 08b --blend-ab / --blend-holdout).
# Holdout referendum on the adopted set: 17 wins / 9 losses — favorable in
# aggregate; per-head holdout losses (DB/LB/WR role, DL/K/OL/QB/WR apy,
# RB wav) are monitoring flags, NOT grounds for re-selection (choosing
# per-head on the holdout would burn its honesty).
# Exceptions kept per-position: QB role (ordering favors pos, logloss
# marginal — consistent with all prior evidence), TE impact_wav (beat its
# global counterpart in every A/B), K/P impact_wav (insufficient data).
GLOBAL_SCORE_OVERRIDES: dict[tuple[str, str], float] = {
    # role — removed 2026-10-31 with the role heads (all 11 positions were
    # global wholesale). EDGE note kept for the impact entries below: EDGE was
    # split out of DL/LB on 2026-07-24 and added here 2026-07-26 after the
    # blend A/B (experiments/blend_ab_20260726_153238.json) returned the
    # largest margins of any position (impact_apy -63.4%, impact_wav -204.9%,
    # LOFO weights 0.0 on every fold); its per-position heads are starved.
    # impact_apy — global wholesale (all 11)
    ("DB", "impact_apy"): 0.0, ("DL", "impact_apy"): 0.0,
    ("K", "impact_apy"): 0.0,  ("LB", "impact_apy"): 0.0,
    ("OL", "impact_apy"): 0.0, ("P", "impact_apy"): 0.0,
    ("QB", "impact_apy"): 0.0, ("RB", "impact_apy"): 0.0,
    ("TE", "impact_apy"): 0.0, ("WR", "impact_apy"): 0.0,
    ("EDGE", "impact_apy"): 0.0,
    # impact_wav — partial blends, LOFO-median weights (calibrated to the
    # mixture scale by 8c, so raw blending is legitimate)
    ("DB", "impact_wav"): 0.0, ("DL", "impact_wav"): 0.1,
    ("LB", "impact_wav"): 0.2, ("OL", "impact_wav"): 0.0,
    ("QB", "impact_wav"): 0.1, ("RB", "impact_wav"): 0.2,
    ("WR", "impact_wav"): 0.1,
    ("EDGE", "impact_wav"): 0.0,   # LOFO weights [0.2, 0.0, 0.0] → median 0.0
}

# Garbage-time rules. Computed as a column in step 1.6 but NOT yet used as
# a weighting factor. v2 will apply `weight = 0.1` to garbage-time training
# rows after A/B comparison.
GARBAGE_TIME_RULES: dict = {
    "score_diff_abs_gte": 21,   # blowout threshold
    "period_gte":         4,    # only Q4 onward
    "weight_if_garbage":  1.0,  # placeholder — set <1.0 in v2 to down-weight
}
