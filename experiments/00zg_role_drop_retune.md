# 00zg — Drop the role heads + full retune (pre-registered 2026-10-31)

## Change
1. Role heads removed from steps 8, 8b, 8c and 10 (role score stopped being an
   output in 8bd28a4; the head was seed-unstable and its target is saturated).
2. 04b (EWMA decay betas) and 04d (Kalman hyperparams) objective switched from
   `role_target` to `impact_target`, the target of the head that ships.
3. Full retune: `run_pipeline.py --retune --from 4 --tune-seeds 3`
   (step 4 alpha grid → 5 → 7 → 04b → 04d → 6 → 8 tune → 8b tune → 8c → 9 → 10).

Arms 2 and 3 are confounded by design (the user asked for the retune with the
role drop folded in); a fail cannot attribute blame between them.

## Baseline
Deployed impact_score = 8b global `impact_apy` OOF,
`_backup_pre_role_drop_retune_20261031/global_draft_models/global_oof_predictions.parquet`
(caf3ffb rebuild, July tuning cache, role-objective 04b/04d).

## Placebo
Seed-only refit of the baseline 8b (LGBM random_state 42 → 43, same params,
same features). It measures fit noise only. Retune noise (Optuna) is larger,
so the bar below is lenient toward the new arm — appropriate because the
change is a scheduled refresh plus a simplification, not a claimed improvement.

## Metrics
Primary: `00u` value referee (within-group Spearman, bridge, K/P excluded),
new arm − baseline.
Secondary: 00u late_snap / late_apy; 00u `--within-class`; merged-board 00zf
top-M and top-100 usage for `merged_dep_cap` vs the old deployed OOF.

## Ship bar (no-regression)
Ship the retuned state unless BOTH:
- Δvalue < −|placebo Δvalue|, and
- the Δvalue 95% bootstrap interval excludes 0.
If late_snap and late_apy both also fall below their placebo |Δ| the result is
a fail even if the interval on value includes 0.

On a fail: restore the backup caches (role heads stay dropped in code), rerun
`--no-tune --from 8`.

## Placebo measured (before the retune finished)
00u, seed 43 vs baseline: value +0.001 [-0.001, +0.003], late_snap +0.002
[-0.003, +0.006], late_apy +0.003 [-0.002, +0.007]. Baseline levels 0.308 /
0.396 / 0.320. So |placebo Δvalue| = 0.001 and the bar is effectively
"Δvalue interval must not sit wholly below 0". Outputs in
`data/processed/_role_drop_retune_20261031/`.

## Result (2026-11-01): PASS on the bar — a tie, shipped
Run: `--retune --from 4 --tune-seeds 3`, 16:37 → 21:45 local (5h08m) plus a
70-minute manual pause. 04b in 5 s (vectorized), 04d 39 s: impact R²_cal
0.051, val 0.000 (old role run also had val 0.000), age-Q effectively off
(max_mult 1.0, halflife 9.75). Role R²_cal on the new features 0.136 vs 0.098.

00u, retune − baseline (placebo in parentheses):
- value      −0.004 [−0.017, +0.009], wins 3/9   (+0.001)
- late_snap  +0.003 [−0.017, +0.024], wins 4/9   (+0.002)
- late_apy   +0.009 [−0.012, +0.032], wins 5/9   (+0.003)
- within-class: −0.008 [−0.021, +0.006] / +0.000 / +0.009
Δvalue is below −|placebo| but its interval contains 0, and both late
referees are above placebo → not a fail. By group: OL +0.017 / QB +0.040
value gains; RB −0.045, DB −0.015 losses.

Merged board (00zf, capped usage merge, new vs old deployed): topM +0.0009
[−0.0036, +0.0028], top100 0.0000, top500 −0.0007 — a tie. Standalone the new
deployed model is slightly weaker deep in the class (usage topM 0.248 vs
0.253, top500 0.159 vs 0.164, auc_contrib 0.964 vs 0.973); the capped merge
absorbs it because the board supplies the top.

Current class vs the old output: within-position rho mean 0.867, top-100
overlap 72, board players 0.796. LB (0.53) and RB (0.67) churned most.

Tuning cache: NOT best-of-N merged with the prior cache — the new 04b/04d
change every Kalman/EWMA feature's values, so the prior cv_mae numbers were
measured on a different surface.

Stale role pickles deleted (copies in the backup).

