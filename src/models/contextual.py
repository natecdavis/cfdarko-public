"""
Contextual Adjustment Models for College Football (Game-Level)

Fits and applies per-stat, per-position-group Ridge regression models
trained on bridge player game logs.

adjusted_stat = snap-normalized stat - expected snap-normalized stat in context

Residual meaning:
  0  = NFL-caliber average for that position in that context
  +  = above expectation
  -  = below expectation

Context features are opponent-only rolling advanced stats (de-meaned within
player-season) plus location, weather, and time-of-day. No team-side features.
Analogous to cdarko's contextual model but for football game logs.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    RIDGE_ALPHA, RIDGE_ALPHA_GRID, ELASTICNET_L1_RATIOS,
    ADAPTIVE_LASSO_GAMMAS, ADAPTIVE_LASSO_TRIGGER_R2,
    ADAPTIVE_LASSO_EPS, ADAPTIVE_LASSO_MIN_FEATURES,
    PROCESSED_DIR, CONTEXT_FEATURES,
    SNAP_COUNTING_STATS, RATE_STAT_ATTEMPT_COL,
    CONFERENCE_TIERS, DEFAULT_CONFERENCE_TIER,
)

logger = logging.getLogger(__name__)

MODELS_DIR = PROCESSED_DIR / "contextual_models"


def _make_imputer() -> SimpleImputer:
    """Mean imputer used as the first stage of every contextual pipeline.

    Fit and predict MUST share one imputation policy: training previously
    dropped feature-incomplete rows while predict fillna(0.0)'d raw values
    BEFORE the scaler, turning a missing temperature (train mean ~60) into a
    several-σ pseudo-observation that biased every adjusted_ residual on
    rows with missing context (systematically: first games of a season,
    games without weather). Imputing train means inside the pipeline makes
    the transform identical at fit and predict time, and lets feature-
    incomplete rows contribute to training instead of being discarded.
    """
    return SimpleImputer(strategy="mean", keep_empty_features=True)


class _ExpandingSeasonCV:
    """
    Expanding-window CV on season_year groups: test = one season,
    train = all prior seasons. Prevents lg_* constant leakage across seasons.
    min_test_size guards against tiny folds that produce extreme R² values.
    """
    def __init__(self, min_train_seasons: int = 5, min_test_size: int = 30):
        self.min_train_seasons = min_train_seasons
        self.min_test_size = min_test_size

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups (season_year) required")
        seasons = sorted(set(groups))
        for i, test_season in enumerate(seasons):
            if i < self.min_train_seasons:
                continue
            train_seasons = seasons[:i]
            train_idx = np.where(np.isin(groups, train_seasons))[0]
            test_idx  = np.where(groups == test_season)[0]
            if len(train_idx) >= 30 and len(test_idx) >= self.min_test_size:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        if groups is None:
            return 0
        seasons = sorted(set(groups))
        # Conservative count — actual splits depend on min_test_size
        return max(0, len(seasons) - self.min_train_seasons)


def get_stat_column(stat: str, df: pd.DataFrame) -> str:
    """
    Return the appropriate column name for a stat in game-level data.
    Counting stats use the per-100-snap column ({stat}_p100).
    Rate stats use the raw column ({stat}).
    Falls back to {stat} if _p100 column is absent.
    """
    if stat in SNAP_COUNTING_STATS:
        p100_col = f"{stat}_p100"
        if p100_col in df.columns:
            return p100_col
    return stat


def build_context_features(games: pd.DataFrame,
                             _unused_team_ratings: pd.DataFrame = None,
                             ) -> pd.DataFrame:
    """
    Prepare contextual features for game-level data.

    The game-level contextual features (opp_rolling_*, is_home, is_neutral,
    game_hour, days_rest, temperature, wind_speed, precipitation, game_indoors,
    season_year) are expected to already be present in `games` (joined during
    Step 1). This function just ensures they exist and fills in defaults.

    The `_unused_team_ratings` parameter is kept for API compatibility with
    the season-level version but is ignored — team-side features are not used.
    """
    df = games.copy()

    # is_home / is_neutral — default to 0.5 / 0.0 if missing (season-level fallback)
    if "is_home" not in df.columns:
        df["is_home"] = 0.5
    if "is_neutral" not in df.columns:
        df["is_neutral"] = 0.0

    # Fill numeric defaults for optional weather / time features
    if "game_indoors" not in df.columns:
        df["game_indoors"] = 0.0
    if "temperature" not in df.columns:
        df["temperature"] = np.nan
    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.nan
    if "precipitation" not in df.columns:
        df["precipitation"] = np.nan
    if "humidity" not in df.columns:
        df["humidity"] = np.nan
    if "days_rest" not in df.columns:
        df["days_rest"] = 7.0
    if "game_hour" not in df.columns:
        df["game_hour"] = np.nan

    # Rolling opponent features — default NaN if absent (mean-imputed in-pipeline)
    for col in ["opp_rolling_def_ppa", "opp_rolling_off_ppa",
                "opp_rolling_def_success_rate", "opp_rolling_def_stuff_rate"]:
        if col not in df.columns:
            df[col] = np.nan

    if "opp_days_rest" not in df.columns:
        df["opp_days_rest"] = 7.0

    if "tz_diff" not in df.columns:
        df["tz_diff"] = 0.0

    # Rolling opponent four-factor components — default NaN
    for col in ["opp_rolling_off_success_rate", "opp_rolling_off_explosiveness",
                "opp_rolling_def_explosiveness",
                "opp_rolling_havoc_total", "opp_rolling_havoc_front_seven",
                "opp_rolling_havoc_db"]:
        if col not in df.columns:
            df[col] = np.nan

    # Teammate-QB rolling quality — default NaN (mean-imputed in-pipeline).
    # Activation is gated per-position via CONTEXT_FEATURE_DROP_BY_POSITION in
    # step 4; this just guarantees the columns are present so feature
    # selection in `_get_feature_cols` doesn't trip over missing-col errors.
    for col in ["qb_rolling_passer_rating", "qb_rolling_completion_pct",
                "qb_rolling_ppa_avg_passing", "qb_rolling_ypa",
                "qb_rolling_td_rate", "qb_rolling_int_rate"]:
        if col not in df.columns:
            df[col] = np.nan

    # Own-team rolling context (pace, pass rate, OL run-block) — default NaN.
    # Gated per-position via CONTEXT_FEATURE_DROP_BY_POSITION (pace+pass_rate
    # → skill positions; OL run-block → RB-only). Defaults guarantee the
    # columns exist for `_get_feature_cols` even when step 1 didn't populate
    # them (e.g. on slimmed test fixtures).
    for col in ["team_rolling_pace", "team_rolling_pass_rate",
                "team_rolling_line_yards_avg",
                "team_rolling_second_level_yards_avg",
                "team_rolling_open_field_yards_avg",
                "team_rolling_rushing_power_success"]:
        if col not in df.columns:
            df[col] = np.nan

    # Opponent-defense ALLOWED rolling (run-defense + pass-defense detail,
    # derived from team_box_scores via inverse-perspective). Defaults to NaN
    # for fixtures that don't populate them; gating to skill-pos happens via
    # CONTEXT_FEATURE_DROP_BY_POSITION in step 4.
    for col in ["opp_rolling_def_rushing_stuff_rate",
                "opp_rolling_def_rushing_power_success",
                "opp_rolling_def_line_yards_avg",
                "opp_rolling_def_second_level_yards_avg",
                "opp_rolling_def_open_field_yards_avg",
                "opp_rolling_def_success_rate_passing_downs",
                "opp_rolling_def_success_rate_standard_downs",
                "opp_rolling_def_points_per_opportunity"]:
        if col not in df.columns:
            df[col] = np.nan

    # Own-team rolling scoring environment (scoring_opportunities,
    # points_per_opportunity, avg_starting_field_pos from team_box_scores).
    for col in ["team_rolling_scoring_opportunities",
                "team_rolling_points_per_opportunity",
                "team_rolling_avg_starting_field_pos"]:
        if col not in df.columns:
            df[col] = np.nan

    # Defender-only context: opp offense volume + opp-QB rolling.
    for col in ["opp_rolling_off_pass_att",
                "opp_rolling_off_rush_att",
                "opp_rolling_off_plays",
                "opp_qb_rolling_passer_rating",
                "opp_qb_rolling_completion_pct",
                "opp_qb_rolling_ppa_avg_passing",
                "opp_qb_rolling_ypa",
                "opp_qb_rolling_td_rate",
                "opp_qb_rolling_int_rate"]:
        if col not in df.columns:
            df[col] = np.nan

    # Own-defense teammate rolling (pressure / coverage rates).
    for col in ["def_rolling_pressure_rate",
                "def_rolling_coverage_rate"]:
        if col not in df.columns:
            df[col] = np.nan

    # Venue features (altitude + surface + travel, encoded as differences /
    # distances from the player's school's home stadium — see
    # _add_venue_features() in step 01 for why diffs instead of raw values).
    # Defaults here cover fixtures / legacy gamelogs that pre-date the join.
    if "altitude_diff_from_home_ft" not in df.columns:
        df["altitude_diff_from_home_ft"] = np.nan
    if "surface_diff_from_home" not in df.columns:
        df["surface_diff_from_home"] = np.nan
    if "travel_distance_mi" not in df.columns:
        df["travel_distance_mi"] = np.nan
    if "capacity_diff_from_home" not in df.columns:
        df["capacity_diff_from_home"] = np.nan

    # body_clock_hour: derived in step 01's _add_schedule_features as
    # game_hour - tz_diff. Default falls back to game_hour for fixtures
    # without the tz_diff join (preserves backward-compat).
    if "body_clock_hour" not in df.columns:
        df["body_clock_hour"] = df.get("game_hour", np.nan)

    # week of season: already populated on gamelogs by step 01 for real data;
    # fixtures default to mid-season noise.
    if "week" not in df.columns:
        df["week"] = np.nan

    # Conference tier dummies — captures systematic baseline opponent quality.
    # FCS (tier 3) is the reference level: both dummies = 0.
    # Source: conference_tier column if present; else infer from conference string.
    if "conference_tier" in df.columns:
        tier = df["conference_tier"].fillna(DEFAULT_CONFERENCE_TIER)
    elif "conference" in df.columns:
        tier = df["conference"].map(CONFERENCE_TIERS).fillna(DEFAULT_CONFERENCE_TIER)
    else:
        tier = pd.Series(DEFAULT_CONFERENCE_TIER, index=df.index)
    df["conf_tier_power4"] = (tier == 1).astype(float)
    df["conf_tier_g5"]     = (tier == 2).astype(float)

    return df


def _get_feature_cols(df: pd.DataFrame,
                       pool: Optional[list[str]] = None) -> list[str]:
    """Select the subset of `pool` that's present and non-empty in `df`.
    Defaults to game-grain CONTEXT_FEATURES; season-grain callers pass
    PFF_CONTEXT_FEATURES."""
    candidates = pool if pool is not None else CONTEXT_FEATURES
    return [f for f in candidates if f in df.columns and df[f].notna().any()]


class ContextualAdjustmentModel:
    """
    Per-stat Ridge regression contextual model for one position group.
    Trained on bridge player game logs; applied to all players.
    """

    def __init__(self, stat: str, position_group: str, alpha: float = RIDGE_ALPHA):
        self.stat = stat
        self.position_group = position_group
        self.alpha = alpha
        self.best_alpha: float = alpha
        self.best_l1_ratio: Optional[float] = None   # set when ElasticNet wins
        self.is_adaptive_lasso: bool = False         # set True when Adaptive Lasso wins
        self.pipeline: Optional[Pipeline] = None
        self.feature_cols: list[str] = []
        self.cv_r2: Optional[float] = None
        self._stat_col: str = stat

    def _fit_adaptive_lasso(
        self,
        train: pd.DataFrame,
        w: Optional[np.ndarray],
    ) -> tuple:
        """
        Two-step Adaptive Lasso (Zou 2006). Pilot Ridge produces coefficient
        magnitudes that drive per-feature shrinkage in a final Lasso fit.
        Features driven to exact zero are dropped — the resulting pipeline
        uses a vanilla Lasso on the retained subset, so inference is identical
        to the existing Ridge/ElasticNet path.

        Returns:
            (pipeline, retained_features, cv_r2, best_gamma)
            or (None, feature_cols, -inf, None) if no gamma beats the input.
        """
        # to_numpy(na_value=np.nan): nullable-dtype columns hold pd.NA,
        # which numpy.asarray(float) rejects (np.nan is fine).
        X_raw = train[self.feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y = train[self._stat_col].to_numpy(dtype=float, na_value=np.nan)
        season_groups = train["season_year"].values \
            if "season_year" in train.columns else None

        # Precompute expanding-window-by-season folds. Reused for pilot Ridge
        # alpha tuning, LassoCV alpha tuning, and the final honest CV R² eval —
        # all three steps share the same temporal-aware CV scheme. When
        # sample_weight is provided, drop folds whose train or test set has
        # zero total weight (LassoCV divides by sum(weight) and fails
        # otherwise — common for early-cohort folds).
        n_cv_random = min(5, max(2, len(train) // 10))
        valid_folds: Optional[list] = None
        if season_groups is not None:
            exp_cv = _ExpandingSeasonCV(min_train_seasons=5, min_test_size=30)
            candidate_folds = list(exp_cv.split(X_raw, y, groups=season_groups))
            if w is not None:
                candidate_folds = [
                    (tr_idx, te_idx) for (tr_idx, te_idx) in candidate_folds
                    if w[tr_idx].sum() > 0 and w[te_idx].sum() > 0
                ]
            if len(candidate_folds) >= 3:
                valid_folds = candidate_folds
        cv_for_tuning = valid_folds if valid_folds is not None else n_cv_random

        # Step 1: Pilot Ridge with CV-tuned alpha (expanding-window if available)
        pilot_pipe = Pipeline([
            ("imputer", _make_imputer()),
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ])
        pilot_kw = {"ridge__sample_weight": w} if w is not None else {}
        try:
            pilot_gs = GridSearchCV(
                pilot_pipe, {"ridge__alpha": RIDGE_ALPHA_GRID},
                cv=cv_for_tuning, scoring="r2", refit=True,
            )
            pilot_gs.fit(X_raw, y, **pilot_kw)
        except Exception as exc:
            logger.warning("Adaptive Lasso pilot fit failed for %s/%s: %s",
                           self.position_group, self.stat, exc)
            return (None, list(self.feature_cols), float("-inf"), None)

        pilot_coefs = pilot_gs.best_estimator_["ridge"].coef_
        # Imputer + scaler stages of the fitted pilot (everything but the
        # regressor) — X_scaled is NaN-free for the LassoCV loop below.
        X_scaled = pilot_gs.best_estimator_[:-1].transform(X_raw)

        best_overall = (None, list(self.feature_cols), float("-inf"), None)

        for gamma in ADAPTIVE_LASSO_GAMMAS:
            adaptive_w = (np.abs(pilot_coefs) + ADAPTIVE_LASSO_EPS) ** gamma
            X_adaptive = X_scaled * adaptive_w[np.newaxis, :]

            lasso_kw = {"sample_weight": w} if w is not None else {}
            # Inner LassoCV n_jobs=-1 + outer joblib n_jobs=-1 in
            # fit_contextual_models_for_group is faster on this codebase than
            # serializing either layer (measured: 8:28 nested vs 10:13 outer-only
            # vs 12:53 inner-only). Nested parallelism oversubscribes cores but
            # picks up cycles when outer workers stall on data prep.
            lasso = LassoCV(cv=cv_for_tuning, max_iter=20000, n_jobs=-1)
            try:
                lasso.fit(X_adaptive, y, **lasso_kw)
            except Exception as exc:
                logger.warning("Adaptive Lasso (γ=%s) failed for %s/%s: %s",
                               gamma, self.position_group, self.stat, exc)
                continue

            # Recover original-scale coefs and identify retained features
            final_coefs = lasso.coef_ * adaptive_w
            retained_mask = np.abs(final_coefs) > 1e-8
            if retained_mask.sum() < ADAPTIVE_LASSO_MIN_FEATURES:
                continue
            retained_features = [f for f, m in zip(self.feature_cols, retained_mask)
                                 if m]

            # Refit a clean Lasso pipeline on the retained subset so inference
            # uses the standard Pipeline.predict() path.
            final_pipe = Pipeline([
                ("imputer", _make_imputer()),
                ("scaler", StandardScaler()),
                ("regressor", Lasso(alpha=lasso.alpha_, max_iter=20000)),
            ])
            X_retained = train[retained_features].to_numpy(dtype=float, na_value=np.nan)
            refit_kw = {"regressor__sample_weight": w} if w is not None else {}
            try:
                final_pipe.fit(X_retained, y, **refit_kw)
            except Exception as exc:
                logger.warning("Adaptive Lasso (γ=%s) refit failed for %s/%s: %s",
                               gamma, self.position_group, self.stat, exc)
                continue

            # Honest expanding-window CV R² on the retained subset (reuses the
            # same valid_folds — indices are row-based, feature subset is fine)
            cv_r2_al = float("nan")
            if valid_folds is not None:
                scores = cross_val_score(final_pipe, X_retained, y,
                                         cv=valid_folds, scoring="r2")
                cv_r2_al = float(np.clip(scores, -2.0, 1.0).mean())

            if not np.isnan(cv_r2_al) and cv_r2_al > best_overall[2]:
                best_overall = (final_pipe, retained_features, cv_r2_al, gamma)

        return best_overall

    def fit(self, df: pd.DataFrame,
            sample_weight: Optional[np.ndarray] = None,
            cached_params: Optional[dict] = None,
            feature_pool: Optional[list[str]] = None,
            ) -> "ContextualAdjustmentModel":
        """
        Fit Ridge regression. Uses the per-100-snap column for counting stats,
        raw column for rate stats.

        cached_params, when provided, skips the full GridSearchCV and refits a
        single Ridge / ElasticNet using the cached `best_alpha` (and
        `best_l1_ratio` if present). Adaptive-Lasso entries are not honored —
        when `is_adaptive_lasso=True` is cached, fall through to the full
        path (the AL fit has its own LassoCV which is cheaper than the outer
        grid and mirroring its state from a single alpha is brittle).

        feature_pool, when provided, replaces CONTEXT_FEATURES as the feature
        candidate list (used by the season-grain PFF pipeline to inject
        PFF_CONTEXT_FEATURES instead of the game-grain set).
        """
        stat_col = get_stat_column(self.stat, df)
        if stat_col not in df.columns:
            raise ValueError(f"Stat column {stat_col} not in DataFrame")

        self._stat_col = stat_col
        self.feature_cols = _get_feature_cols(df, pool=feature_pool)
        if not self.feature_cols:
            raise ValueError(f"No contextual features for {self.stat}/{self.position_group}")

        # Target must be observed; features may be NaN (mean-imputed inside
        # the pipeline — see _make_imputer for why this must match predict).
        mask = df[stat_col].notna()
        train = df[mask].copy()

        if len(train) < 30:
            logger.warning("Only %d rows for %s/%s", len(train), self.position_group, self.stat)

        # to_numpy(na_value=np.nan): pd.NA from nullable dtypes must become
        # np.nan before sklearn sees it.
        X = train[self.feature_cols].to_numpy(dtype=float, na_value=np.nan)
        y = train[stat_col].to_numpy(dtype=float, na_value=np.nan)

        w: Optional[np.ndarray] = None
        if sample_weight is not None:
            w = np.asarray(sample_weight)[mask.values]

        n_cv = min(5, max(2, len(train) // 10))
        pipe = Pipeline([
            ("imputer", _make_imputer()),
            ("scaler", StandardScaler()),
            ("regressor", Ridge()),
        ])

        # Cached fast path — single fit with prior best alpha (skip GridSearchCV).
        # Only honored for Ridge/ElasticNet (not Adaptive Lasso) and only when
        # we have enough rows for a reasonable single fit.
        use_cache = (
            cached_params is not None
            and len(train) >= 30
            and not bool(cached_params.get("is_adaptive_lasso", False))
            and cached_params.get("best_alpha") is not None
        )

        if use_cache:
            cached_alpha = float(cached_params["best_alpha"])
            cached_l1 = cached_params.get("best_l1_ratio")
            fit_params = {"regressor__sample_weight": w} if w is not None else {}
            if cached_l1 is not None:
                pipe.set_params(regressor=ElasticNet(
                    alpha=cached_alpha, l1_ratio=float(cached_l1), max_iter=20000,
                ))
            else:
                pipe.set_params(regressor=Ridge(alpha=cached_alpha))
            pipe.fit(X, y, **fit_params)
            self.pipeline = pipe
            self.best_alpha = cached_alpha
            self.best_l1_ratio = float(cached_l1) if cached_l1 is not None else None

            # Honest expanding-window CV R² with the cached single-fit estimator.
            if "season_year" in train.columns:
                season_groups = train["season_year"].values
                exp_cv = _ExpandingSeasonCV(min_train_seasons=5, min_test_size=30)
                valid_folds = list(exp_cv.split(X, y, groups=season_groups))
                if len(valid_folds) >= 3:
                    exp_scores = cross_val_score(
                        self.pipeline, X, y,
                        cv=valid_folds, scoring="r2",
                    )
                    self.cv_r2 = float(np.clip(exp_scores, -2.0, 1.0).mean())
                else:
                    cv_scores = cross_val_score(self.pipeline, X, y, cv=n_cv, scoring="r2")
                    self.cv_r2 = float(cv_scores.mean())
            else:
                cv_scores = cross_val_score(self.pipeline, X, y, cv=n_cv, scoring="r2")
                self.cv_r2 = float(cv_scores.mean())
        elif len(train) >= 30:
            fit_params = {"regressor__sample_weight": w} if w is not None else {}

            # Search over Ridge (all alphas) and ElasticNet (alpha × l1_ratio)
            param_grid = [
                {"regressor": [Ridge()],
                 "regressor__alpha": RIDGE_ALPHA_GRID},
                {"regressor": [ElasticNet(max_iter=20000)],
                 "regressor__alpha": RIDGE_ALPHA_GRID,
                 "regressor__l1_ratio": ELASTICNET_L1_RATIOS},
            ]
            # Alpha selection uses expanding-window CV when season_year is
            # available — without it, K-fold mixes future seasons into the
            # tuning fold, so the chosen alpha is informed by future data.
            # Falls back to K-fold for short panels (<3 valid expanding folds).
            cv_for_grid: object = n_cv
            valid_folds: list[tuple[np.ndarray, np.ndarray]] = []
            if "season_year" in train.columns:
                season_groups = train["season_year"].values
                exp_cv = _ExpandingSeasonCV(min_train_seasons=5, min_test_size=30)
                valid_folds = list(exp_cv.split(X, y, groups=season_groups))
                if len(valid_folds) >= 3:
                    cv_for_grid = valid_folds
            gs = GridSearchCV(pipe, param_grid, cv=cv_for_grid, scoring="r2", refit=True)
            gs.fit(X, y, **fit_params)
            self.pipeline = gs.best_estimator_
            self.best_alpha = float(gs.best_params_["regressor__alpha"])
            self.best_l1_ratio = gs.best_params_.get("regressor__l1_ratio")

            # When alpha was selected via expanding-window CV, reuse those
            # scores as the honest R² report (best_score_ is already the
            # expanding-window-mean). Else fall back to a separate
            # expanding-window pass for reporting only.
            if isinstance(cv_for_grid, list):
                self.cv_r2 = float(np.clip(gs.best_score_, -2.0, 1.0))
            elif "season_year" in train.columns:
                season_groups = train["season_year"].values
                exp_cv = _ExpandingSeasonCV(min_train_seasons=5, min_test_size=30)
                valid_folds = list(exp_cv.split(X, y, groups=season_groups))
                if len(valid_folds) >= 3:
                    exp_scores = cross_val_score(
                        self.pipeline, X, y,
                        cv=valid_folds, scoring="r2",
                    )
                    self.cv_r2 = float(np.clip(exp_scores, -2.0, 1.0).mean())
                else:
                    self.cv_r2 = float(gs.best_score_)
            else:
                self.cv_r2 = float(gs.best_score_)
        else:
            # Too few samples for grid search — use default alpha
            pipe.set_params(regressor__alpha=self.alpha)
            fit_kw = {"regressor__sample_weight": w} if w is not None else {}
            pipe.fit(X, y, **fit_kw)
            self.pipeline = pipe
            self.best_alpha = self.alpha
            cv_scores = cross_val_score(pipe, X, y, cv=n_cv, scoring="r2")
            self.cv_r2 = float(cv_scores.mean())

        # Adaptive Lasso fallback. Attempted whenever pilot CV R² is below
        # ADAPTIVE_LASSO_TRIGGER_R2 (default +∞ → attempt on every head); AL
        # is only swapped in if it strictly beats the pilot below, so the gate
        # is just a cost knob, not a correctness one. Skipped in cached
        # fast-path mode — the user's intent there is to reuse the prior
        # selection, and AL is the most expensive sub-path.
        if (self.cv_r2 is not None
                and self.cv_r2 < ADAPTIVE_LASSO_TRIGGER_R2
                and "season_year" in train.columns
                and not use_cache):
            al_pipe, al_features, al_r2, al_gamma = self._fit_adaptive_lasso(train, w)
            if al_pipe is not None and al_r2 > self.cv_r2:
                n_dropped = len(self.feature_cols) - len(al_features)
                logger.info("%s / %s: Adaptive Lasso (γ=%.1f) dropped %d/%d features, "
                            "CV R² %.4f → %.4f",
                            self.position_group, self.stat, al_gamma,
                            n_dropped, len(self.feature_cols),
                            self.cv_r2, al_r2)
                self.pipeline           = al_pipe
                self.feature_cols       = al_features
                self.cv_r2              = al_r2
                self.best_alpha         = float(al_pipe["regressor"].alpha)
                self.best_l1_ratio      = 1.0
                self.is_adaptive_lasso  = True

        if self.is_adaptive_lasso:
            model_type = "AdaptiveLasso"
        elif self.best_l1_ratio is not None:
            model_type = "ElasticNet"
        else:
            model_type = "Ridge"
        l1_str = f", l1={self.best_l1_ratio}" if self.best_l1_ratio is not None else ""
        logger.info("%s / %s: %d rows, CV R²=%.3f, alpha=%.4g%s [%s] (col=%s)",
                    self.position_group, self.stat, len(train), self.cv_r2,
                    self.best_alpha, l1_str, model_type, stat_col)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted")

        df = df.copy()
        # Absent columns become NaN so the pipeline's imputer fills them with
        # the TRAINING mean — never raw 0.0, which for non-centered features
        # (temperature, success rates) is a several-σ pseudo-observation.
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = np.nan

        X = df[self.feature_cols].to_numpy(dtype=float, na_value=np.nan)
        return pd.Series(self.pipeline.predict(X), index=df.index)

    def compute_residuals(self, df: pd.DataFrame) -> pd.Series:
        expected = self.predict(df)
        stat_col = get_stat_column(self.stat, df)
        raw = df[stat_col] if stat_col in df.columns else pd.Series(np.nan, index=df.index)
        return raw - expected

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"{self.position_group}_{self.stat}_model.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, stat: str, position_group: str,
             path: Optional[Path] = None) -> "ContextualAdjustmentModel":
        path = path or (MODELS_DIR / f"{position_group}_{stat}_model.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_contextual_models_for_group(
        bridge_seasons: pd.DataFrame,
        team_ratings: pd.DataFrame,
        stats: list[str],
        position_group: str,
        save_dir: Optional[Path] = None,
        weight_col: Optional[str] = "bridge_weight",
        n_jobs: int = -1,
        cached_params: Optional[dict[str, dict]] = None,
        feature_pool: Optional[list[str]] = None,
        skip_build_context_features: bool = False,
) -> dict[str, "ContextualAdjustmentModel"]:
    """
    Fit contextual models for all stats in a position group, using bridge
    player game logs only. Parallelized across stats via joblib (each stat is
    an independent Ridge/ElasticNet/AdaptiveLasso fit). n_jobs=1 falls back
    to serial.

    cached_params, when provided, is keyed by stat and supplies the prior
    `best_alpha` / `best_l1_ratio` / `is_adaptive_lasso` for a single-fit
    fast path. Stats missing from the cache fall back to the full grid
    (per-key fallback handles partial staleness, e.g. a newly added stat).

    feature_pool — replaces CONTEXT_FEATURES as the candidate feature list,
    used by the season-grain PFF pipeline.

    skip_build_context_features — when True, callers have already prepared
    the panel (e.g. a season-grain panel that already has conf_tier dummies
    + team-season columns). Avoids running the game-grain
    build_context_features which would inject opp_rolling_* NaN columns
    that are noise for season-grain models.
    """
    df = bridge_seasons if skip_build_context_features else build_context_features(bridge_seasons)

    sample_weight: Optional[np.ndarray] = None
    if weight_col and weight_col in df.columns:
        sample_weight = df[weight_col].fillna(1.0).values

    out_dir = save_dir or MODELS_DIR

    def _fit_one(stat: str):
        stat_col = get_stat_column(stat, df)
        if stat_col not in df.columns:
            logger.warning("Stat %s (col=%s) not in bridge games for %s — skipping",
                           stat, stat_col, position_group)
            return stat, None
        try:
            model = ContextualAdjustmentModel(stat=stat, position_group=position_group)
            cached = cached_params.get(stat) if cached_params else None
            model.fit(df, sample_weight=sample_weight, cached_params=cached,
                      feature_pool=feature_pool)
            model.save(out_dir / f"{position_group}_{stat}_model.pkl")
            return stat, model
        except Exception as exc:
            logger.error("Failed %s/%s: %s", position_group, stat, exc)
            return stat, None

    if n_jobs == 1 or len(stats) <= 1:
        results = [_fit_one(s) for s in stats]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_one)(s) for s in stats
        )

    return {stat: m for stat, m in results if m is not None}
