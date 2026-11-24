# Experiment log

Each experiment was pre-registered (hypothesis, primary metric and decision rule committed before results) and then recorded with its outcome. This log is taken from the private development history, oldest first.

| Date | Entry |
|---|---|
| 2026-07-26 | Record ranking as the primary evaluation metric; flag the Optuna mismatch |
| 2026-10-30 | Pre-register the two-part holdout confirmation and add --holdout |
| 2026-10-30 | Record the two-part holdout confirmation: pass |
| 2026-10-30 | Pre-register the persistent board-failure test |
| 2026-10-30 | Record the board-failure rerun after the combine fix: five traits pass |
| 2026-10-30 | Pre-register the board-plus-corrections test |
| 2026-10-30 | Record the off-board harness rerun after the bridge rebuild |
| 2026-10-31 | Pre-register the merged board test |
| 2026-10-31 | Record the merged board result: fail on the bar, better at the top |
| 2026-10-31 | Pre-register the off-board top cap for the merged board |
| 2026-10-31 | Record the off-board cap result: top 100 fixed, top-M still a tie |
| 2026-10-31 | Pre-register the value-built merged board |
| 2026-10-31 | Record the value-built merged board: fail; the usage build stays |
| 2026-10-31 | Pre-register recency weighting for the snap-share capped merge |
| 2026-10-31 | Record recency weighting for the capped merge: no significant gain |
| 2026-10-31 | Pre-register the dollars-built merged board |
| 2026-10-31 | Record the dollars-built merged board: a dead tie |
| 2026-10-31 | Pre-register board sub-position curves for the merged board |
| 2026-10-31 | Record the board sub-position test: no better than a random split |
| 2026-10-31 | Record the 00zg seed placebo before the retune result |
| 2026-10-31 | Record the 00zg retune result: tie on every referee, shipped |
| 2026-10-31 | Pre-register draft visits and PFF features for the off-board model (00zh) |
| 2026-10-31 | Record 00zh: visits and PFF both fail the bar; PFF lifts off-board NFL ranking |
| 2026-10-31 | Pre-register the seed-bagged PFF confirmation for off-board ranking (00zi) |
| 2026-10-31 | Record 00zi: PFF passes the seed-bagged confirmation; merge check favors the switch |
| 2026-11-01 | Pre-register position-specific PFF grades for the off-board model (00zj) |
| 2026-11-01 | Record 00zj: position-specific PFF grades fail on off-board ranking |
| 2026-11-01 | Record 00zk: all-star invites fail on off-board ranking (+0.009, ns) |
| 2026-11-01 | Pre-register position-normalized usage target for the global model (00zl) |
| 2026-11-01 | Record 00zl: position-normalized usage beats raw on every referee |
| 2026-11-01 | Pre-register retuned normalized usage vs deployed APY target (00zm) |
| 2026-11-01 | Pre-register position-normalized stage-2 target for the off-board model (00zn) |
| 2026-11-01 | Record 00zn: normalized stage-2 target fails on off-board ranking (-0.032) |
| 2026-11-01 | Pre-register the dollars-built merge rerun on fixed usage_y34 (00zf) |
| 2026-11-01 | Record the dollars-built merge rerun: still a dead tie on realized pay |
| 2026-11-01 | Pre-register 00zp: does quality beyond snaps predict the second contract? |
| 2026-11-01 | Record 00zp: quality beyond snaps predicts second-contract size; add NFL mode to PFF downloader |
| 2026-11-01 | Pre-register 00zq: EPA quality-adjusted pricing for QB/RB/WR/TE seasons (00zo --quality epa) |
| 2026-11-01 | 00zq pre-reg amendment (before results): cross-fit pay curves by player fold; OL name matching |
| 2026-11-01 | Pre-register 00zs: merged board in rookie-deal value vs snap share |
| 2026-11-01 | Record 00zs: value-currency board ties the snap-share board; pooled shrinkage compresses position levels |
| 2026-11-01 | Pre-register 00zs level-preserving shrinkage arm |
| 2026-11-01 | Record 00zs level arm: fixes position calibration but loses cross-position rank (pre-reg primary fails); keep pooled shrinkage |
| 2026-11-01 | Record 00zt board drift diagnostics: overall rank beats within-position rank out of time; drift shows in position levels |
| 2026-11-01 | Pre-register 00zu: per-player adjustment on top of the board curve (rookie value) |
| 2026-11-01 | Record 00zu: per-player board adjustment fails its primary (top100 value); player features carry signal but board prices most of it |
| 2026-11-01 | Pre-register 00zv: partial outcomes for recent classes in the board curve |
| 2026-11-01 | Record 00zv: partial outcomes adopt by rule (MSE -3.5%), gain shrinks to ~0 with 5+ complete classes |
| 2026-11-01 | Pre-register 00zw: off-board model stage 2 on rookie value |
| 2026-11-01 | Record 00zw: off-board stage 2 on rookie value fails its primary (top10 -0.0006, P 0.19); keep usage target |
| 2026-11-01 | Pre-register 00zx: P(available at pick n) for board players |
| 2026-11-01 | Record 00zx: availability model, censored log-normal on board rank (B_ln) by rule; C_pos better on conditional Brier |
| 2026-11-01 | Pre-register 00zy: off-board P(available at pick n) |
| 2026-11-01 | 00zy amendment: level-free rank arm (B_rank), pre-registered before its results |
| 2026-11-01 | Record 00zy: off-board availability, level-free rank logistic (B_rank) + F by combine invite |
| 2026-11-01 | Pre-register 00zz: slot curve at the top of the draft (value smoothers, per-pick cost) |
| 2026-11-01 | Record 00zz and adopt: slot curve value via log-link GLM (cubic in log pick), cost via per-pick mean |
| 2026-11-01 | Pre-register 01a: position-level (QB) pricing in the merged board, MSE primary |
| 2026-11-01 | Record 01a and adopt: level-preserving shrinkage for the rookie-value board (QB bias -19% -> 0) |
| 2026-11-01 | Pre-register 01b: do PFF NFL grades add to snap-based value in predicting the 2nd contract, all positions |
| 2026-11-01 | Pre-register 01c: drafting team's roster state vs realized rookie value |
| 2026-11-01 | Record 01c: drafting team's roster state — real but small effect (years 1-2), absorbed by pick; fails MSE primary, not wired |
| 2026-11-01 | Pre-register 01d: QB-only entrenched-starter adjustment (one parameter, permutation null) |
| 2026-11-01 | Record 01d: QB entrenched starter delays years 1-2 (b -0.053, perm pct 0.98) but no 4-year effect; no adjustment |
| 2026-11-02 | Record 01b: PFF grades add to snap value in predicting the 2nd contract at every position (Pearson +0.046, P 1.00) → build v4 label |
| 2026-11-02 | Pre-register 01e: v4 rookie-value label with PFF-grade quality pricing at every position, cross-fitted vs v3.1 |
| 2026-11-02 | Record 01e: v4 label (PFF-grade quality at every position) beats v3.1 (Pearson +0.040, P 1.00) → adopt; production default quality=pff |
| 2026-11-02 | Pre-register 01f: does actual draft position (slide/rise vs board rank) predict rookie value beyond the board? |
| 2026-11-02 | Record 01f: draft position vs board rank predicts rookie value (γ −0.19, MSE −4.1%, P 0.98) → adopt; wire value-at-pick |
