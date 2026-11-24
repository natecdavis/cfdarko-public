# cfDARKO Pipeline Audit Report

_Generated: 2026-05-04 19:53:39 EDT_

## Summary

| phase | label | runtime (s) | errors | warnings |
| --- | --- | --- | --- | --- |
| 0 | validate.py --all | 2.8 | 0 | 7 |
| 1 | Manifests + Provenance | 0.0 | 1 | 5 |
| 2 | Lineage | 0.6 | 0 | 0 |
| 3 | Feature Flow | 0.9 | 0 | 0 |
| 4 | Output Sanity | 1.8 | 0 | 4 |
| 5 | Model Behavior | 0.7 | 0 | 0 |

## Phase 0 — `pipeline/validate.py --all`

Exit code: `0`

```text
2026-05-04 19:53:33  INFO      ── Step 0: Pre-stage outputs       → 00* (PFF / combine / nfl_avg / hasan / wikidata)
2026-05-04 19:53:33  INFO         Step 0: all 10 checks passed
2026-05-04 19:53:33  INFO      ── Step 1: Data Ingestion          → player_gamelogs.parquet
2026-05-04 19:53:33  INFO         Step 1: all 5 checks passed
2026-05-04 19:53:33  INFO      ── Step 2: Bridge Players          → bridge_players.parquet
2026-05-04 19:53:33  INFO         Step 2: all 6 checks passed
2026-05-04 19:53:33  INFO      ── Step 3: Player Attributes       → player_attributes.parquet (+ PFF features)
2026-05-04 19:53:33  WARNING     WARN height_range: 7 height_in values outside [60, 82]
2026-05-04 19:53:33  WARNING     WARN weight_range: 85 weight_lbs values outside [150, 380]
2026-05-04 19:53:33  INFO         Step 3: all 22 checks passed
2026-05-04 19:53:33  INFO      ── Step 4: Contextual Models       → contextual_models/*.pkl
2026-05-04 19:53:33  WARNING     WARN cv_r2_sane: Could not parse model_summary.json: 'list' object has no attribute 'items'
2026-05-04 19:53:33  INFO         Step 4: all 4 checks passed
2026-05-04 19:53:33  INFO      ── Step 5: Residual Computation    → adjusted_gamelogs.parquet
2026-05-04 19:53:34  WARNING     WARN nan_rate: Mean NaN rate across adjusted_ cols=88.8% (threshold <80%)
2026-05-04 19:53:34  INFO         Step 5: all 4 checks passed
2026-05-04 19:53:34  INFO      ── Step 6: Skill Estimation        → player_skill_snapshot.parquet
2026-05-04 19:53:34  WARNING     WARN kalman_range: Some kalman_mean_ values outside (-500, 500)
2026-05-04 19:53:34  WARNING     WARN n_games_present: n_games median=1.0 (expected >1)
2026-05-04 19:53:34  INFO         Step 6: all 6 checks passed
2026-05-04 19:53:34  INFO      ── Step 7: NFL Labels              → nfl_draft_labels.parquet
2026-05-04 19:53:34  INFO         Step 7: all 5 checks passed
2026-05-04 19:53:34  INFO      ── Step 8: Draft Models            → draft_models/*.pkl
2026-05-04 19:53:35  INFO         Step 8: all 5 checks passed
2026-05-04 19:53:35  INFO      ── Step 9: Comps                   → player_comps.parquet
2026-05-04 19:53:35  INFO         Step 9: all 4 checks passed
2026-05-04 19:53:35  INFO      ── Step 10: Final Output            → projections/latest.parquet
2026-05-04 19:53:35  WARNING     WARN overall_rank_present: overall_rank column absent
2026-05-04 19:53:35  INFO         Step 10: all 5 checks passed
2026-05-04 19:53:35  INFO      ──────────────────────────────────────────────────
2026-05-04 19:53:35  INFO      All ERROR checks passed.
```

## Phase 1 — Manifests + Provenance

**Issues found:**

- `data/processed/decay_betas_manifest.json` — pre-v2
- `data/processed/draft_models/tuned_params_manifest.json` — pre-v2
- `data/processed/draft_models/tuned_params_monotone_manifest.json` — pre-v2
- `data/processed/global_draft_models/tuned_params_manifest.json` — pre-v2
- `data/processed/kalman_hyperparams_draft_manifest.json` — pre-v2
- `data/processed/kalman_hyperparams_draft.json` — expected inline provenance: no _tuning_mode key

### Manifests

| path | version | mode | step | written_at | fingerprint_keys |
| --- | --- | --- | --- | --- | --- |
| `data/processed/contextual_models/model_summary_manifest.json` | 2 | cached | 04_contextual_models | 2026-05-04T11:53:19+00:00 | `bridge_data_shape,bridge_players_file,context_features,elasticnet_l1_ratios,n_mo…` |
| `data/processed/decay_betas_manifest.json` | 1 | — | — | — | `config_hash,written_at` |
| `data/processed/decay_betas_role_manifest.json` | 2 | fresh | 04b_calibrate_betas_role | 2026-05-04T11:55:14+00:00 | `adjusted_gamelogs_file,bounds,cal_season_cutoff,nfl_labels_file,stats` |
| `data/processed/draft_models/tuned_params_manifest.json` | 1 | — | — | — | `hashes_by_key,n_keys,written_at` |
| `data/processed/draft_models/tuned_params_monotone_manifest.json` | 1 | — | — | — | `hashes_by_key,n_keys,written_at` |
| `data/processed/global_draft_models/tuned_params_manifest.json` | 1 | — | — | — | `feature_cols_hash,n_features,written_at` |
| `data/processed/kalman_hyperparams_draft_manifest.json` | 1 | — | — | — | `config_hash,written_at` |

### Provenance sidecars (`_tuning_mode`)

| path | mode | step | stamped_at |
| --- | --- | --- | --- |
| `data/processed/contextual_models/model_summary_provenance.json` | cached | 04_contextual_models | 2026-05-04T11:53:19+00:00 |
| `data/processed/draft_models/training_metrics_provenance.json` | cached | 08_draft_models | 2026-05-04T23:21:30+00:00 |
| `data/processed/global_draft_models/training_metrics_provenance.json` | cached | 08b_global_draft_model | 2026-05-04T23:33:29+00:00 |

### Mode distribution

| mode | count |
| --- | --- |
| `cached` | 3 |

### Expected provenance — missing

| path | kind | note |
| --- | --- | --- |
| `data/processed/kalman_hyperparams_draft.json` | inline | no _tuning_mode key |
## Phase 2 — Lineage

All lineage checks passed.

| check | passed | severity | detail |
| --- | --- | --- | --- |
| `bridge_pid_resolution` | ✓ | WARNING | 6055 bridge pids, 0 not in union(player_seasons, rosters) |
| `attributes_covers_snapshot` | ✓ | WARNING | snapshot has 123893 pids, attributes has 123919; 0 snapshot pids absent from attributes |
| `skill_estimates_match_gamelogs` | ✓ | ERROR | adjusted_gamelogs rows=894,233, skill_estimates rows=894,356, diff=+123 |
| `snapshot_unique_pids` | ✓ | ERROR | snapshot has 123,893 rows, 0 duplicate player_ids |
| `bridge_in_labels` | ✓ | WARNING | bridge has 6,055 pids, labels[is_bridge] has 6,055; 0 bridge pids missing label rows |
| `latest_cohort` | ✓ | ERROR | latest has 123,893 pids, 0 not in snapshot |
| `roster_only_fallback` | ✓ | ERROR | 49,995 pids in rosters but not player_seasons; 49,995 of those reached player_attributes (must be >1,000 for fallback to be working) |
| `n_seasons_recompute` | ✓ | WARNING | sample=100, mismatches=0 (expected 0 — n_seasons should be union over rosters/player_seasons/gamelogs) |
## Phase 3 — Feature Flow

No feature-flow issues found.

### Summary

| kind | OK | WARNING | ERROR |
| --- | --- | --- | --- |
| attribute | 20 | 0 | 0 |
| career_team | 7 | 0 | 0 |
| engineered | 46 | 0 | 0 |
| raw | 216 | 0 | 0 |
## Phase 4 — Output Sanity

**Issues found:**

- (WARN) `position_other_share` — position_group=OTHER is 30.2% of latest (threshold <20%) — many CFBD positions failing to map
- (WARN) `score_mode_density_global_role_score` — global_role_score has 83.0% of values at single rounded bucket — looks compressed (regression model anchoring to mean?)
- (WARN) `score_mode_density_global_impact_apy_score` — global_impact_apy_score has 94.1% of values at single rounded bucket — looks compressed (regression model anchoring to mean?)
- (WARN) `diff_mode_caveat` — diff vs projections_20260504.parquet is best-effort — projection snapshots don't yet carry an explicit _tuning_mode tag

### Tuning-mode of on-disk artifacts

| step | tuning_mode |
| --- | --- |
| `step04_contextual` | cached |
| `step08_draft` | cached |
| `step08b_global` | cached |

### Position distribution

| position_group | count | pct |
| --- | --- | --- |
| OTHER | 37366 | 30.2% |
| DB | 15550 | 12.6% |
| OL | 13798 | 11.1% |
| DL | 12794 | 10.3% |
| WR | 12527 | 10.1% |
| LB | 11310 | 9.1% |
| RB | 7025 | 5.7% |
| TE | 5151 | 4.2% |
| QB | 4279 | 3.5% |
| K | 2591 | 2.1% |
| P | 1502 | 1.2% |

### Score distributions

| head | non_null | p01 | p10 | p50 | p90 | p99 | mode_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `role_score` | 86527 | 0 | 0.0004234 | 0.009332 | 0.05311 | 0.8641 | 12.6% |
| `impact_apy_score` | 86527 | 0.0001274 | 0.0001997 | 0.0004342 | 0.001069 | 0.01416 | 29.7% |
| `impact_wav_score` | 86527 | 0 | 0 | 0.002516 | 0.004242 | 0.233 | 16.5% |
| `global_role_score` | 123893 | 0.01271 | 0.01271 | 0.01271 | 0.01724 | 0.6774 | 83.0% |
| `global_impact_apy_score` | 123893 | 0.0001751 | 0.0001751 | 0.0001751 | 0.0001751 | 0.01579 | 94.1% |
| `global_impact_wav_score` | 123893 | 0.08036 | 0.133 | 0.1622 | 0.2012 | 0.2946 | 16.2% |

### Named-prospect spot check

| player_name | school | position_group | role_score | impact_apy_score | impact_wav_score | impact_apy_dollars | hasan_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fernando Mendoza | Indiana | QB | 0.608 | 0.0438 | 0.544 | 1.32e+07 | 1 |
| Spencer Fano | Utah | OL | 0.848 | 0.0255 | 0.359 | 7.66e+06 | 11 |
| Dametrious Crownover | Texas A&M | OL | 0.801 | 0.00679 | 0.147 | 2.04e+06 | 124 |
| KC Concepcion | Texas A&M | WR | 0.82 | 0.025 | 0.308 | 7.5e+06 | 23 |
| Travis Hunter | Colorado | WR | 0.847 | 0.0375 | 0.317 | 1.13e+07 | 1 |

### Top-100 rank shifts vs prior snapshot (0 > 20 spots)

| player_name | rank_now | rank_prev | delta_rank | impact_apy_score |
| --- | --- | --- | --- | --- |
## Phase 5 — Model Behavior

No feature-availability issues across model bundles.

| bundle | n_features | n_missing | examples | severity |
| --- | --- | --- | --- | --- |
| `DB_impact_model.pkl` | 181 | 0 | `` | OK |
| `DB_impact_wav_model.pkl` | 181 | 0 | `` | OK |
| `DB_role_model.pkl` | 181 | 0 | `` | OK |
| `DL_impact_model.pkl` | 166 | 0 | `` | OK |
| `DL_impact_wav_model.pkl` | 166 | 0 | `` | OK |
| `DL_role_model.pkl` | 166 | 0 | `` | OK |
| `K_impact_model.pkl` | 124 | 0 | `` | OK |
| `K_impact_wav_model.pkl` | 124 | 0 | `` | OK |
| `K_role_model.pkl` | 124 | 0 | `` | OK |
| `LB_impact_model.pkl` | 200 | 0 | `` | OK |
| `LB_impact_wav_model.pkl` | 200 | 0 | `` | OK |
| `LB_role_model.pkl` | 200 | 0 | `` | OK |
| `OL_impact_model.pkl` | 130 | 0 | `` | OK |
| `OL_impact_wav_model.pkl` | 130 | 0 | `` | OK |
| `OL_role_model.pkl` | 130 | 0 | `` | OK |
| `P_impact_model.pkl` | 94 | 0 | `` | OK |
| `P_impact_wav_model.pkl` | 94 | 0 | `` | OK |
| `P_role_model.pkl` | 94 | 0 | `` | OK |
| `QB_impact_model.pkl` | 190 | 0 | `` | OK |
| `QB_impact_wav_model.pkl` | 190 | 0 | `` | OK |
| `QB_role_model.pkl` | 190 | 0 | `` | OK |
| `RB_impact_model.pkl` | 199 | 0 | `` | OK |
| `RB_impact_wav_model.pkl` | 199 | 0 | `` | OK |
| `RB_role_model.pkl` | 199 | 0 | `` | OK |
| `ST_impact_model.pkl` | 56 | 0 | `` | OK |
| `ST_role_model.pkl` | 56 | 0 | `` | OK |
| `TE_impact_model.pkl` | 151 | 0 | `` | OK |
| `TE_impact_wav_model.pkl` | 151 | 0 | `` | OK |
| `TE_role_model.pkl` | 151 | 0 | `` | OK |
| `WR_TE_impact_model.pkl` | 32 | 0 | `` | OK |
| `WR_TE_role_model.pkl` | 32 | 0 | `` | OK |
| `WR_impact_model.pkl` | 154 | 0 | `` | OK |
| `WR_impact_wav_model.pkl` | 154 | 0 | `` | OK |
| `WR_role_model.pkl` | 154 | 0 | `` | OK |
| `global_impact_apy_model.pkl` | 767 | 0 | `` | OK |
| `global_impact_wav_model.pkl` | 767 | 0 | `` | OK |
| `global_role_model.pkl` | 767 | 0 | `` | OK |
| `DB_ints_model.pkl` | 34 | 0 | `` | OK |
| `DB_passes_defended_model.pkl` | 34 | 0 | `` | OK |
| `DB_sacks_model.pkl` | 34 | 0 | `` | OK |
| `DB_tackles_model.pkl` | 34 | 0 | `` | OK |
| `DB_tfl_model.pkl` | 34 | 0 | `` | OK |
| `DL_ints_model.pkl` | 34 | 0 | `` | OK |
| `DL_passes_defended_model.pkl` | 34 | 0 | `` | OK |
| `DL_qb_hurries_model.pkl` | 34 | 0 | `` | OK |
| `DL_sacks_model.pkl` | 34 | 0 | `` | OK |
| `DL_tackles_model.pkl` | 34 | 0 | `` | OK |
| `DL_tfl_model.pkl` | 34 | 0 | `` | OK |
| `K_fg_att_model.pkl` | 34 | 0 | `` | OK |
| `K_fg_made_model.pkl` | 34 | 0 | `` | OK |
| `K_fg_pct_model.pkl` | 34 | 0 | `` | OK |
| `K_kicking_pts_model.pkl` | 34 | 0 | `` | OK |
| `K_xp_att_model.pkl` | 34 | 0 | `` | OK |
| `K_xp_made_model.pkl` | 34 | 0 | `` | OK |
| `K_xp_pct_model.pkl` | 34 | 0 | `` | OK |
| `LB_ints_model.pkl` | 34 | 0 | `` | OK |
| `LB_passes_defended_model.pkl` | 34 | 0 | `` | OK |
| `LB_qb_hurries_model.pkl` | 34 | 0 | `` | OK |
| `LB_sacks_model.pkl` | 34 | 0 | `` | OK |
| `LB_tackles_model.pkl` | 34 | 0 | `` | OK |
| `LB_tfl_model.pkl` | 34 | 0 | `` | OK |
| `P_net_punt_yards_model.pkl` | 34 | 0 | `` | OK |
| `P_punt_yards_model.pkl` | 34 | 0 | `` | OK |
| `P_punts_model.pkl` | 34 | 0 | `` | OK |
| `QB_completion_pct_model.pkl` | 13 | 0 | `` | OK |
| `QB_interceptions_model.pkl` | 34 | 0 | `` | OK |
| `QB_pass_att_model.pkl` | 34 | 0 | `` | OK |
| `QB_pass_share_model.pkl` | 34 | 0 | `` | OK |
| `QB_pass_tds_model.pkl` | 34 | 0 | `` | OK |
| `QB_pass_yards_model.pkl` | 35 | 0 | `` | OK |
| `QB_ppa_avg_passing_model.pkl` | 34 | 0 | `` | OK |
| `QB_rush_att_model.pkl` | 34 | 0 | `` | OK |
| `QB_rush_tds_model.pkl` | 5 | 0 | `` | OK |
| `QB_rush_yards_model.pkl` | 12 | 0 | `` | OK |
| `RB_fumbles_model.pkl` | 34 | 0 | `` | OK |
| `RB_ppa_avg_rushing_model.pkl` | 34 | 0 | `` | OK |
| `RB_rec_model.pkl` | 34 | 0 | `` | OK |
| `RB_rec_share_model.pkl` | 34 | 0 | `` | OK |
| `RB_rec_tds_model.pkl` | 34 | 0 | `` | OK |
| `RB_rec_yards_model.pkl` | 34 | 0 | `` | OK |
| `RB_rush_att_model.pkl` | 34 | 0 | `` | OK |
| `RB_rush_share_model.pkl` | 34 | 0 | `` | OK |
| `RB_rush_tds_model.pkl` | 34 | 0 | `` | OK |
| `RB_rush_yards_model.pkl` | 34 | 0 | `` | OK |
| `RB_yards_per_carry_model.pkl` | 34 | 0 | `` | OK |
| `TE_ppa_avg_model.pkl` | 3 | 0 | `` | OK |
| `TE_rec_model.pkl` | 13 | 0 | `` | OK |
| `TE_rec_share_model.pkl` | 9 | 0 | `` | OK |
| `TE_rec_tds_model.pkl` | 4 | 0 | `` | OK |
| `TE_rec_yards_model.pkl` | 34 | 0 | `` | OK |
| `TE_yards_per_rec_model.pkl` | 5 | 0 | `` | OK |
| `WR_ppa_avg_model.pkl` | 34 | 0 | `` | OK |
| `WR_rec_model.pkl` | 34 | 0 | `` | OK |
| `WR_rec_share_model.pkl` | 34 | 0 | `` | OK |
| `WR_rec_tds_model.pkl` | 34 | 0 | `` | OK |
| `WR_rec_yards_model.pkl` | 34 | 0 | `` | OK |
| `WR_yards_per_rec_model.pkl` | 34 | 0 | `` | OK |

_Note: CV refit comparison against `training_metrics.json` is deferred — needs the step-8 training-data assembly logic to be factored out of `pipeline/08_draft_models.py` first._
