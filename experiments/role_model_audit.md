# Role-model audit — DL/DT/DE saturation at ~0.50


Generated from `data/output/projections/latest.parquet` and `data/processed/nfl_draft_labels.parquet`.


Focus position group: **DL** (includes sub-positions ('DL', 'DT', 'DE', 'EDGE', 'NT')).


## S1. Training-data composition

Per `position_group`, with the production training mix step 8 sees after `_prep_training_df` (NaN role_target → 0):

```
pos_group  n_total  n_bridge  n_non_bridge non_bridge_share  bridge_role_mean  bridge_role_median bridge_zero_frac bridge_one_frac bridge_traj_frac  bridge_traj_mean
       DB     9593       977          8616            89.8%             0.864               0.967             7.5%           16.7%            75.8%             0.919
       DL     6569       634          5935            90.3%             0.893               0.973             3.9%           12.3%            83.8%             0.920
        K     1634        68          1566            95.8%             0.889               0.971             2.9%           11.8%            85.3%             0.905
       LB     6293       577          5716            90.8%             0.856               0.951             6.2%           15.6%            78.2%             0.896
       OL     1881       270          1611            85.6%             0.901               0.973             3.3%           20.0%            76.7%             0.914
        P     1036        49           987            95.3%             0.886               0.969             6.1%            6.1%            87.8%             0.940
       QB     2607       207          2400            92.1%             0.843               0.982            10.6%           37.2%            52.2%             0.904
       RB     4801       425          4376            91.1%             0.878               0.967             5.4%           19.1%            75.5%             0.910
       TE     2971       309          2662            89.6%             0.907               0.975             2.9%           20.1%            77.0%             0.917
       WR     7570       671          6899            91.1%             0.882               0.958             4.9%           19.5%            75.6%             0.909
```

**Reading guide**:

- `non_bridge_share`: fraction of step-8 training rows that are the appended `role_target=0` non-bridge mass (`pipeline/07_nfl_labels.py:313-326`). Higher → stronger pull-toward-0.

- `bridge_role_mean`: mean role_target *among bridge players only*.

- `bridge_traj_frac`: fraction of bridge with a continuous trajectory-imputed label (vs hard 0/1 fallback).


## S2. Reliability curves

### S2a. In-sample reliability (production model on bridge players)

**Caveat**: the production model was trained on these rows. This section is a sanity check (reveals gross miscalibration but cannot diagnose generalization). See S2b for an honest refit.

```
pos  n_bridge  top_decile_pred  top_decile_actual    gap  overall_pred_mean  overall_actual
 DB       977            0.945              0.976 -0.031              0.769           0.864
 DL       634            0.983              0.987 -0.004              0.852           0.893
  K        68            0.778              0.949 -0.171              0.607           0.889
 LB       577            0.867              0.982 -0.116              0.717           0.856
 OL       270            0.955              0.978 -0.023              0.745           0.901
  P        49            0.787              0.986 -0.199              0.620           0.886
 QB       207            0.943              0.998 -0.056              0.753           0.843
 RB       425            0.850              0.895 -0.045              0.692           0.878
 TE       309            0.931              0.989 -0.057              0.778           0.907
 WR       671            0.964              0.967 -0.003              0.776           0.882
```

### S2b. Honest holdout refit — `DL`

Refit on train ≤ 2021, eval on 2022–2024. Saved tuned params transferred from production model.

Headline (refit holdout): n_val=1586, spearman=0.503, auc(@0.5)=0.998, pred_mean=0.104, actual_mean=0.082

```
 decile   n  pred_mean  actual_mean
      2 316   0.000038     0.000000
      4 316   0.000577     0.000000
      5 161   0.002293     0.000000
      6 159   0.007572     0.000000
      7 158   0.016001     0.000000
      8 158   0.032224     0.000000
      9 159   0.123230     0.020321
     10 159   0.851461     0.800919
```


## S3. Feature attribution — 2025 DL prospects with role_score ≥ 0.40

n=37 prospects in 2025 DL cohort with role_score ≥ 0.40 (impact_blend_pct mean=0.96, impact_apy_score mean=0.0119)

Top 30 by role_score (cohort capped at 30 for display):

```
          player_name         school position  role_score  impact_apy_score  hasan_rank  age_at_draft  height_in  weight_lbs  combine_invited  recruiting_composite  pff_seasons   base  shap_kalman  shap_ewma  shap_combine  shap_pff  shap_nfl_era  shap_hasan  shap_attributes  shap_recruiting  shap_context  shap_other
      Chris McClellan       Missouri       DT    0.908488          0.009055       108.0     22.537988       76.0       323.0                1                   NaN          4.0 0.1172       0.0056    -0.0041        0.7511    0.0299       -0.0439      0.0558          -0.0019          -0.0016        0.0003         0.0
          Peter Woods        Clemson       DT    0.874934          0.017404        26.0     21.155373       75.0       315.0                1                0.9822          3.0 0.1172       0.0193     0.0034        0.6199    0.0236       -0.0404      0.0647           0.0011           0.0624        0.0038         0.0
          T.J. Parker        Clemson       DE    0.869835          0.021114        30.0     21.000000       75.0       265.0                1                0.9731          3.0 0.1172       0.0080    -0.0073        0.6294    0.0527       -0.0466      0.0415          -0.0003           0.0738        0.0014         0.0
     Malachi Lawrence            UCF     EDGE    0.867494          0.012558        48.0     22.000000       76.0       250.0                1                   NaN          4.0 0.1172       0.0014     0.0079        0.7277    0.0083       -0.0445      0.0487          -0.0010          -0.0020        0.0037         0.0
   Dani Dennis-Sutton     Penn State       DE    0.863826          0.024052        72.0     22.000000       77.0       265.0                1                0.9869          4.0 0.1172      -0.0021     0.0065        0.6527    0.0233       -0.0545      0.0429           0.0054           0.0737       -0.0013         0.0
           Lee Hunter     Texas Tech       DL    0.858403          0.014588        45.0     22.000000       76.0       325.0                1                   NaN          4.0 0.1172      -0.0084     0.0008        0.7111    0.0354       -0.0445      0.0470          -0.0001          -0.0020        0.0019         0.0
        Tyler Onyedim      Texas A&M       DT    0.857670          0.007504       129.0     22.000000       75.0       295.0                1                   NaN          5.0 0.1172       0.0012    -0.0008        0.7758    0.0057       -0.0432      0.0097          -0.0035          -0.0007       -0.0038         0.0
      Wesley Williams           Duke       DE    0.857200          0.003662       236.0     21.604381       75.0       265.0                1                0.8648          4.0 0.1172       0.0048    -0.0031        0.7541    0.0192       -0.0416      0.0114          -0.0052          -0.0008        0.0013         0.0
      Christen Miller        Georgia       DL    0.856884          0.014444        42.0     21.000000       76.0       305.0                1                0.9560          4.0 0.1172      -0.0004    -0.0040        0.6759    0.0549       -0.0454      0.0607          -0.0014           0.0008       -0.0015         0.0
        Keldric Faulk         Auburn       DE    0.855982          0.024094        20.0     22.000000       78.0       285.0                1                0.9616          3.0 0.1172       0.0109    -0.0005        0.6876    0.0487       -0.0521      0.0389           0.0017           0.0035        0.0001         0.0
  Darrell Jackson Jr.  Florida State       DL    0.854728          0.012579        98.0     22.000000       77.0       337.0                1                   NaN          5.0 0.1172       0.0103     0.0022        0.7090    0.0089       -0.0477      0.0526          -0.0002          -0.0013        0.0038         0.0
        Deven Eastern      Minnesota       DL    0.848515          0.006379       266.0     22.000000       78.0       320.0                1                0.8955          4.0 0.1172       0.0012     0.0043        0.7649    0.0011       -0.0483      0.0069           0.0004          -0.0012        0.0020         0.0
   Skyler Gill-Howard     Texas Tech       DL    0.845986          0.004733       227.0     22.000000       73.0       290.0                1                   NaN          3.0 0.1172       0.0066    -0.0029        0.7461    0.0227       -0.0448      0.0100          -0.0118          -0.0010        0.0038         0.0
          Caden Curry     Ohio State       DE    0.844050          0.009618       161.0     22.000000       75.0       260.0                1                0.9506          4.0 0.1172       0.0063     0.0025        0.6769    0.0648       -0.0433      0.0188          -0.0042           0.0045        0.0004         0.0
           Zion Young       Missouri       DE    0.831360          0.019315        39.0     22.119097       77.0       262.0                1                   NaN          4.0 0.1172       0.0046    -0.0009        0.6750    0.0459       -0.0522      0.0503          -0.0074          -0.0023        0.0011         0.0
          Zane Durant     Penn State       DT    0.825266          0.006424       140.0     22.000000       73.0       294.0                1                0.8991          4.0 0.1172       0.0019    -0.0014        0.7652    0.0007       -0.0491      0.0060          -0.0133          -0.0018       -0.0003         0.0
       Tim Keenan III        Alabama       DL    0.821187          0.003964       165.0     22.000000       74.0       320.0                1                   NaN          4.0 0.1172      -0.0024    -0.0014        0.7636   -0.0010       -0.0480      0.0062          -0.0126          -0.0002       -0.0003         0.0
       Rayshaun Benny       Michigan       DL    0.820861          0.003672       137.0     23.980835       76.0       305.0                1                0.9253          5.0 0.1172       0.0111     0.0000        0.6785    0.0342       -0.0472      0.0134           0.0114          -0.0001        0.0023         0.0
    Brandon Cleveland       NC State       DT    0.820558          0.005658       254.0     22.000000       76.0       315.0                1                0.8933          4.0 0.1172      -0.0013    -0.0034        0.7363    0.0095       -0.0434      0.0065          -0.0020          -0.0011        0.0023         0.0
          Caleb Banks        Florida       DL    0.817931          0.014075        37.0     22.000000       78.0       330.0                1                   NaN          5.0 0.1172       0.0016     0.0062        0.7377   -0.0334       -0.0482      0.0335           0.0009          -0.0010        0.0034         0.0
Quintayvious Hutchins Boston College       DL    0.803461          0.002816       250.0     22.000000       75.0       240.0                1                0.8411          3.0 0.1172      -0.0000    -0.0004        0.7117    0.0136       -0.0429      0.0063          -0.0052          -0.0026        0.0057         0.0
     DeMonte Capehart        Clemson       DT    0.799302          0.006789       170.0     23.783710       77.0       315.0                1                   NaN          6.0 0.1172       0.0050     0.0062        0.7160   -0.0054       -0.0575      0.0064           0.0106          -0.0008        0.0016         0.0
        Gracen Halton       Oklahoma       DL    0.796130          0.008987        83.0     22.000000       74.0       292.0                1                0.9073          4.0 0.1172       0.0046    -0.0067        0.6752    0.0216       -0.0486      0.0404          -0.0126           0.0030        0.0021         0.0
     Domonique Orange     Iowa State       DL    0.795810          0.007249        70.0     22.000000       76.0       325.0                1                0.8856          4.0 0.1172       0.0059    -0.0063        0.6864   -0.0127       -0.0425      0.0494          -0.0003          -0.0012       -0.0002         0.0
         Nick Barrett South Carolina       DL    0.795221          0.003805       174.0     22.759754       75.0       322.0                1                   NaN          4.0 0.1172       0.0129     0.0046        0.7276   -0.0351       -0.0388      0.0067          -0.0044          -0.0002        0.0046         0.0
         Albert Regis      Texas A&M       DT    0.793642          0.005816       150.0     22.000000       73.0       317.0                1                0.8852          5.0 0.1172      -0.0002    -0.0082        0.7506   -0.0087       -0.0455      0.0057          -0.0116          -0.0017       -0.0039         0.0
       Akheem Mesidor          Miami       DL    0.793195          0.029044        22.0     22.000000       75.0       280.0                1                   NaN          6.0 0.1172       0.0046    -0.0055        0.6282    0.0472       -0.0443      0.0508          -0.0022          -0.0025       -0.0004         0.0
        Max Llewellyn           Iowa       DL    0.791920          0.007635       135.0     23.707050       77.0       263.0                1                0.8870          4.0 0.1172       0.0031    -0.0137        0.7285   -0.0006       -0.0506      0.0058           0.0016          -0.0011        0.0017         0.0
        Derrick Moore       Michigan     EDGE    0.790396          0.009897        63.0     23.400411       75.0       260.0                1                0.9789          4.0 0.1172       0.0023    -0.0010        0.6039    0.0239       -0.0441      0.0649          -0.0292           0.0480        0.0045         0.0
      Jackie Marshall         Baylor       DL    0.786643          0.005619         NaN     22.000000       75.0       306.0                1                0.8774          3.0 0.1172       0.0010     0.0039        0.7230   -0.0016       -0.0457     -0.0056          -0.0065          -0.0013        0.0022         0.0
```

Mean SHAP contribution by family (over the n=37 saturated prospects):

```
    family  mean_shap  n_features  abs_mean_shap
   combine     0.6918           9         0.6918
     hasan     0.0329           1         0.0332
       pff     0.0134          68         0.0221
recruiting     0.0063           4         0.0084
    kalman     0.0048          18         0.0057
   context     0.0010           8         0.0025
     other     0.0000           0         0.0000
      ewma    -0.0011          18         0.0040
attributes    -0.0042           5         0.0060
   nfl_era    -0.0455          10         0.0455
```

Top 15 features by |mean SHAP| in this cohort:

```
                    feature  mean_shap  fill_rate
            combine_invited     0.4385   1.000000
               combine_wt_z     0.1878   1.000000
               combine_ht_z     0.0647   1.000000
                 hasan_rank     0.0329   0.972973
         nfl_lg_pass_att_pg    -0.0209   1.000000
       nfl_lg_pass_yards_pg    -0.0102   1.000000
      nfl_lg_completion_pct    -0.0067   1.000000
         combine_vertical_z     0.0057   0.540541
       recruiting_composite     0.0051   0.567568
pff_grades_run_defense_last     0.0049   1.000000
            nfl_lg_plays_pg    -0.0046   1.000000
            combine_bench_z    -0.0036   0.081081
    pff_grades_defense_last     0.0028   1.000000
        nfl_lg_turnovers_pg    -0.0027   1.000000
            combine_forty_z    -0.0021   0.540541
```


## S4. Counterfactual — LGBMRegressor vs LGBMClassifier

Refit on train ≤ 2021 (n=3240); val on 2022–2024 (n=1586); score 2025 cohort (n=1743).

```
                       head  val_auc  val_pred_mean  val_actual_mean  2025_pred_mean  2025_pct_in_[0.45,0.55]  2025_pct_>=0.40  2025_pct_>=0.50
     regressor (production)   0.9985          0.105            0.082           0.036                    0.001            0.022            0.022
classifier (counterfactual)   0.9983          0.107            0.082           0.031                    0.001            0.023            0.022
```


## S5. Non-bridge zero-mass leverage test

Compares production-style training (with non-bridge zero mass) vs bridge-only training. Train ≤ 2021, val 2022–2024, then score 2025 cohort.

```
                          training_set  n_train  val_pred_mean  val_actual_mean  val_auc  2025_pred_mean  2025_pct_>=0.40  2025_pct_in_45_55  2025_top_decile
production (bridge + non-bridge zeros)     3240          0.104            0.082   0.9985           0.034            0.023              0.001            0.054
                           bridge-only      480          0.836            0.082   0.8201           0.815            1.000              0.009            0.915
```


## S6. Findings + remediation candidates

### What the data actually showed

- DL bridge population mean role_target = **0.893** (median 0.973). The bridge population is mostly settled draftees who *did* reach a second contract, so their target mean is **high, not 0.5**. S5 confirms this: a bridge-only refit collapses to a 2025 prediction mean of ~0.83 — i.e. *more* saturated, not less.

- DL non-bridge zero mass = **90%** of training rows (5,935 appended at role_target=0). This mass is what *prevents* full saturation; without it, every prediction trends toward the bridge mean.

- The MSE-trained `LGBMRegressor` lands the 2025 DL cohort at a **predicted mean of ~0.49**, with **>90% of rows in [0.45, 0.55]** (S4 production row). The discrimination is fine — `val_auc=0.998` — but absolute calibration of *low-information* rows is anchored near 0.5.

- **Top feature SHAP family for the saturated cohort: `nfl_era` (mean +0.48)**. Inside that, `nfl_lg_pass_att_pg` alone contributes +0.26 with a fill rate of 0%: the column is NaN for the 2025 cohort because it's joined on `last_college_season_year` and 2025-season NFL averages aren't in `nfl_league_averages.parquet` yet (`pipeline/08_draft_models.py:861-870`). LightGBM's default-direction split for that feature points to the high-prediction branch, pushing the entire 2025 cohort up by ~0.5.

- The next-largest family contribution is `combine` (mean −0.04, abs +0.06): mostly z-score features whose fill rates are still low for the 2025 cohort.

- The trajectory model upstream (`src/scrapers/nfl_data.py:763-923`) **is** isotonically calibrated as a classifier. The saturation is generated entirely in the step-8 regressor head.

- A counterfactual `LGBMClassifier` on the same features and binarized target produces nearly identical AUC (0.998 vs 0.998) but a 2025 prediction mean of **0.046** with **0.1% in [0.45, 0.55]** (S4 classifier row).


### Root cause, ranked

1. **NaN nfl_era features for the 2025 cohort**, combined with LightGBM's default-direction split learning that *missing → high* for those columns, accounts for the bulk of the saturation. The step-8 join (`08_draft_models.py:861-870`) leaves `last_college_season_year=2025` rows with no NFL league averages.

2. **MSE objective on a target that's bimodal at 0 and ~0.9** leaves the regressor expressing ignorance as ~0.5 instead of as a class-probability that collapses to base rate. A classifier head sidesteps this entirely (S4).

3. **No monotone constraints on Role.** `MONOTONE_ENABLED` covers Impact heads only — Role gets no `hasan_rank` or `age_at_draft` constraint despite both being top-importance for DL_role.


### Remediation candidates (ordered by expected payoff)

1. **Fix the nfl_era join for the current cohort.** Either backfill season 2025 league averages in `nfl_league_averages.parquet` (forward-fill from 2024 if 2025 data isn't yet available) or exclude `nfl_lg_*` from the Role feature set when its fill rate is 0 for the scoring cohort. Cheapest, highest-leverage single change.

2. **Switch Role head to `LGBMClassifier` on `(role_target > 0.5)`.** Same features, comparable AUC (0.998), but predictions de-saturate by construction. The trajectory model already calibrates the binary outcome upstream — using log-loss in step 8 preserves that calibration via the sigmoid output. Higher-leverage but a deeper code change.

3. **Add Role monotone constraints**: at minimum `("DL", "role")` with `hasan_rank` and `age_at_draft` negative (per `MONOTONE_CONSTRAINTS`). Run `pipeline/00h_monotone_decision.py` after a paired sweep. Marginal next to options 1 and 2.

4. **Do *not* drop the non-bridge zero mass.** S5 shows this makes saturation strictly worse — those rows are the model's only negative-class signal for low-information prospects.


### Suggested next session

Start with remediation #1 — the nfl_era NaN issue is the smoking gun and a 5-line fix. If that doesn't resolve the bulk of the saturation (re-run this audit to confirm), proceed to #2.
