# hasan_rank off-board handling — A/B result (2026-07-24)

**Outcome: all three arms null. Nothing shipped.** Production continues to leave
`hasan_rank` NaN for off-board players. Harness: `00n_hasan_offboard_ab.py`.
Result CSV lands in `data/processed/` (gitignored), hence this note.

## Question

`hasan_rank` is the only feature carrying a monotone −1 constraint (8 heads, via
`config.MONOTONE_ENABLED`). It is NaN off-board, and LightGBM chooses its own
missing-direction at each split under that constraint — so there was no guarantee
it learned "unranked = worse", which is what the constraint is meant to encode.

## What NaN actually means (measured first, and it reframed the experiment)

The board covers draft years 2014–2026 only, i.e. `last_college_season_year`
2013–2025. **19.5% of training rows predate it entirely.** Within the board era,
coverage of the full step-8 training cohort is 4–10% per position (K/P: 0.6%).
Overall `hasan_rank` is **2.8% non-null**.

The prior "~50% NaN" figure was measured among *bridge* players, where coverage is
62–82% (K/P: 13–19%). The full cohort is mostly zero-padded non-NFL players. So a
NaN is one of three things: board passed on him / never a prospect (label already
0) / board did not exist yet. That is why the sentinel arm was split into gated
and ungated variants.

## Setup

Four arms plus an A/A placebo, all through step 8's own `train_model_for_group`
(production folds, feature resolution, constraints, holdout), cached
hyperparameters, 3 LightGBM seeds averaged, 33 heads.

Decision metric is `cv_rho_bridge`: mean per-fold Spearman across the 4 tuning
folds, restricted to players who reached the NFL. Per-fold rather than pooled
(folds differ in year range and label scale); bridge-only because Spearman over a
~90%-zero cohort measures "did he ever play", not prospect ranking; 4 folds
because the single holdout fold is too noisy to carry a decision.

## Result

Δρ_bridge, 31 heads with a usable bridge cohort (noise SE = 0.0028):

| arm | Δρ_bridge | wins | vs SE | monotone 8 | wins |
|---|---|---|---|---|---|
| `binary` (`hasan_on_board`, era-gated) | −0.0037 | 9/31 | −1.3 | −0.0065 | 1/8 |
| `sentinel_gated` (NaN→400, board era only) | +0.0007 | 15/31 | +0.3 | +0.0032 | 5/8 |
| `sentinel_ungated` (NaN→400 everywhere) | −0.0029 | 4/31 | −1.1 | −0.0053 | 1/8 |
| **`placebo` (identical data, seed only)** | **−0.0057** | 10/31 | **−2.1** | **−0.0128** | 1/8 |

**The A/A placebo moved more than any real arm.** Per-head seed noise: ±0.005
median, ±0.028 p90, ±0.047 max. The single largest effect in the experiment —
`DL/impact_wav` under `sentinel_gated`, +0.045 — is the size of the placebo's worst
pure-noise swing (`K/impact`, −0.047). Nothing here is distinguishable from noise.

Note the placebo mean is −2.1 SE from zero rather than centered: seeds 42–44 were a
mildly lucky draw for `base`, and every arm was measured against that one draw. A
fixed reference imports its own luck into all comparisons.

The SAFE binary — the a-priori favorite, since it is additive and cannot disturb the
monotone feature — was the arm that lost (1/8 monotone wins, −0.038 on
`RB/impact_wav`). Adding a 95%-zero binary next to the rank column dilutes.

## Kept from the null

1. **If board membership is ever re-encoded, gate it to the board era.**
   `sentinel_gated` beat `sentinel_ungated` 15/31 vs 4/31 (5/8 vs 1/8 monotone).
   Ungated tells the model every pre-2014 player is worse than the worst-ranked
   modern prospect — under monotone −1 that forces two decades below the tail.
2. **The dominant-value filter eats hasan features on K/P under any fill.** 0.6%
   board coverage → 99.4% single-value share → over the 99% threshold, so
   `hasan_rank` is dropped entirely from all 6 K/P heads (and so is the binary).
   Their Δ=0.0000 rows are the filter, not indifference. Any sparse binary added
   in a future batch hits the same wall.
3. **Include an A/A placebo arm in feature-addition A/Bs.** Without it,
   `sentinel_gated` at +0.0007 with 15/31 wins reads as a marginal win worth a
   `--retune-only 8,8.5`.

## Caveat

Cached hyperparameters were tuned on the `base` surface, so the comparison is
biased against the new arms; a genuine but small effect could in principle need a
retune to surface. Not worth paying for here — the effects are below the noise
floor, not merely small.
