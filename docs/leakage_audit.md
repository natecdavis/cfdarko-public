# Data Leakage Audit & Remediation Plan

## TL;DR

For the primary use case — **scoring current (2025/2026) draft prospects** — there is no meaningful
data leakage. Current prospects' games have all been played; every model we apply uses only
historical data. The issues below are real but affect validation metric credibility and historical
simulation, not current prospect ranking.

---

## Do we need per-year retraining?

**No.** The "clean" version of temporal integrity would train a separate contextual model,
Kalman prior, and EWMA beta for every cohort year Y using only data available at time Y. That is
~20 training passes and would make the pipeline prohibitively slow.

The right approach — and the one cdarko uses — is:

> Train everything on all available historical data. Produce projections for the **current year's
> prospects only**. Comps are already backward-looking (comp pool = players from earlier cohorts).

Under this design:

| Component | Uses future data to score 2026 prospects? | Verdict |
|---|---|---|
| Contextual model (Ridge) | No — their games are in the past | ✓ Clean |
| Kalman filter | No — online filter, processes games in order | ✓ Clean |
| EWMA / beta | No — parameters are globally estimated but applied forward-only | ✓ Clean |
| Draft model (LightGBM) | No — trained on ≤2019, applied to current features | ✓ Clean |
| Comps | No — pool is backward-looking by cohort year | ✓ Clean |

The only real issues are in **validation metric credibility**.

---

## Issue 1 — Validation cohort has trajectory-estimated labels (High priority)

**Problem:**

`DRAFT_VAL_START_YEAR = 2020` means the validation set is players with
`last_college_season_year` 2020–2022 (drafted 2021–2023). Many of these players have not yet been
eligible for a second NFL contract. Their `role_target` / `impact_target` labels in
`nfl_draft_labels.parquet` are produced by the **trajectory model**, not from observed second
contracts.

This means the draft model's validation MAE and AUC measure "how well the draft model reproduces
the trajectory model's output" — a partially circular test — not how well it predicts real NFL
outcomes. Validation metrics are optimistic.

**Fix:**

Pull the training/validation cutoffs back so both sets use **observed** labels:

```python
# config.py
DRAFT_TRAIN_CUTOFF_YEAR = 2015   # was 2019 — train on last_college_season_year ≤ 2015
DRAFT_VAL_START_YEAR    = 2016   # was 2020 — validate on 2016–2018 cohorts
```

Players with `last_college_season_year` 2016–2018 were drafted 2017–2019. As of 2026 they have
6–9 years of NFL career data; all second-contract decisions are observed. Validation metrics
computed against these labels are ground truth.

Trade-off: we lose 4 years of training data (2016–2019 cohorts move from train → no longer train,
since they're used as the current-scoring cohort). Training set shrinks from 16 years to 12 years
(2004–2015). That is still a large sample.

**Files changed:** `config.py` (2 constants). Requires re-running Step 8 to retrain draft models.

---

## Issue 2 — Comps StandardScaler fitted on full position group (Low priority)

**Problem:**

In `pipeline/09_comps.py`, `StandardScaler` is fitted on all players in the position group before
the backward-looking distance computation begins. When computing comps for a 2006 prospect, the
feature means and standard deviations include 2007–2026 players.

Since comps affect only display (not model predictions or scores), and since stat distributions are
stable over time, the practical impact is minimal.

**Fix:**

Fit the scaler inside the cohort-year loop, using only the comp pool (players with
`last_college_season_year < Y`):

```python
# In find_comps_for_group(), replace:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# With (inside the unique_years loop, before cdist):
scaler = StandardScaler()
X_pool_scaled = scaler.fit_transform(X_pool_raw)
X_prospects_scaled = scaler.transform(X_prospects_raw)
```

This requires separating `X_raw` into `X_pool_raw` and `X_prospects_raw` per cohort year.

**Files changed:** `pipeline/09_comps.py`. Does not require rerunning any other step.

---

## Issue 3 — Contextual model / Kalman priors trained on all-era data (Low priority)

**Problem:**

The Ridge contextual model (Step 4) is trained on bridge players from all eras. The Kalman
prior variance (`estimate_prior_vars`) and EWMA beta (`optimize_all_betas`) in Step 6 are
estimated on the full adjusted gamelog. Historical players' features are calibrated using patterns
from future players.

**Why it doesn't matter for current scoring:**

For the current use case (score 2025/2026 prospects), using more data to estimate these global
parameters is strictly better. The contextual model coefficients more accurately represent the
true opponent-quality adjustment relationship. The Kalman prior more accurately represents the
true population distribution. More data → better calibration → better features → better
predictions.

**If you want clean backtesting:**

Filter bridge data in Step 4 to `season_year <= DRAFT_TRAIN_CUTOFF_YEAR + 1` before fitting.
Use `--skip-beta-opt` in Step 6 and rely on the hardcoded config values
(`KALMAN_PRIOR_MEAN`, `KALMAN_PRIOR_VAR`).

**Recommendation:** Leave as-is. The improvement to current-year accuracy outweighs the
theoretical backtesting concern.

---

## Implementation Order

1. **`config.py`**: Change `DRAFT_TRAIN_CUTOFF_YEAR = 2015`, `DRAFT_VAL_START_YEAR = 2016`.
2. **Step 8**: Retrain draft models (`python pipeline/08_draft_models.py`).
3. **Step 9**: Fix comps scaler (small refactor in `find_comps_for_group`).
4. **Step 9**: Re-run comps (`python pipeline/09_comps.py`).
5. **Step 10**: Re-run output (`python pipeline/10_output.py`).

Steps 1–2 are the meaningful fix. Steps 3–5 are polish.

---

## What Does Not Need Fixing

- **Kalman filter**: Online, processes games in chronological order. No future data per game.
- **EWMA**: Same — purely backward-looking.
- **Bridge player identification**: A player who played in the NFL and was later cut is correctly
  a bridge player with `role_target=0`. Using 2026 data to confirm historical bridge status is
  correct, not leakage.
- **Career opponent features**: Computed from a player's own past games. No leakage.
- **NFL league average features**: Joined by `last_college_season_year`. Historical data. Clean.
- **Hasan ranks / combine metrics**: Published pre-draft. Clean.
