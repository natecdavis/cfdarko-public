# Combine z-score vs raw — audit
_Thresholds: drift-keep=0.05, drift-floor=0.02, corr-min=0.005_
Decision rule: keep_z if drift_r2 ≥ drift-keep; switch_to_raw if drift_r2 < drift-floor AND Δ|corr| ≥ corr-min.

## Decisions summary
- **insufficient_data**: 23 cells
- **keep_z**: 60 cells
- **switch_to_raw**: 27 cells

## forty

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 180 | -0.052 | 0.027 | 182 | 0.174 | 0.152 | -0.023 | keep_z |
| RB | 378 | -0.021 | 0.011 | 378 | 0.174 | 0.179 | +0.005 | **switch_to_raw** |
| WR | 581 | -0.018 | 0.012 | 582 | 0.083 | 0.068 | -0.015 | keep_z |
| TE | 245 | -0.027 | 0.018 | 245 | 0.180 | 0.111 | -0.069 | keep_z |
| OL | 541 | -0.061 | 0.033 | 542 | 0.216 | 0.138 | -0.078 | keep_z |
| DL | 626 | -0.010 | 0.001 | 626 | 0.102 | 0.071 | -0.031 | keep_z |
| LB | 465 | -0.052 | 0.063 | 466 | 0.154 | 0.125 | -0.030 | keep_z |
| DB | 779 | -0.009 | 0.004 | 779 | 0.090 | 0.061 | -0.029 | keep_z |
| K | 20 | +nan | nan | 20 | nan | nan | +nan | insufficient_data |
| P | 24 | +nan | nan | 24 | nan | nan | +nan | insufficient_data |

## vertical

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 159 | +1.702 | 0.073 | 161 | 0.106 | 0.080 | -0.026 | keep_z |
| RB | 352 | +0.564 | 0.011 | 352 | 0.097 | 0.065 | -0.032 | keep_z |
| WR | 524 | +0.990 | 0.035 | 525 | 0.083 | 0.094 | +0.012 | keep_z |
| TE | 214 | +0.728 | 0.020 | 215 | 0.158 | 0.111 | -0.047 | keep_z |
| OL | 499 | +1.550 | 0.060 | 500 | 0.110 | 0.093 | -0.017 | keep_z |
| DL | 579 | +0.248 | 0.001 | 579 | 0.100 | 0.074 | -0.027 | keep_z |
| LB | 425 | +0.479 | 0.008 | 426 | 0.093 | 0.088 | -0.005 | keep_z |
| DB | 690 | +0.226 | 0.002 | 690 | 0.086 | 0.052 | -0.034 | keep_z |
| K | 3 | +nan | nan | 2 | nan | nan | +nan | insufficient_data |
| P | 7 | +nan | nan | 7 | nan | nan | +nan | insufficient_data |

## broad_jump

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 156 | +5.009 | 0.146 | 158 | 0.254 | 0.189 | -0.065 | keep_z |
| RB | 340 | +3.088 | 0.111 | 340 | 0.175 | 0.208 | +0.033 | keep_z |
| WR | 511 | +3.488 | 0.122 | 512 | 0.042 | 0.087 | +0.045 | keep_z |
| TE | 208 | +2.901 | 0.099 | 209 | 0.155 | 0.119 | -0.036 | keep_z |
| OL | 490 | +4.865 | 0.139 | 491 | 0.152 | 0.098 | -0.055 | keep_z |
| DL | 559 | +2.154 | 0.020 | 559 | 0.127 | 0.077 | -0.050 | keep_z |
| LB | 434 | +3.393 | 0.101 | 435 | 0.117 | 0.104 | -0.013 | keep_z |
| DB | 676 | +2.269 | 0.056 | 676 | 0.084 | 0.066 | -0.019 | keep_z |
| K | 3 | +nan | nan | 3 | nan | nan | +nan | insufficient_data |
| P | 6 | +nan | nan | 6 | nan | nan | +nan | insufficient_data |

## bench

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 10 | +nan | nan | 11 | nan | nan | +nan | insufficient_data |
| RB | 276 | -1.396 | 0.027 | 276 | 0.054 | 0.092 | +0.039 | keep_z |
| WR | 339 | +0.132 | 0.000 | 340 | 0.060 | 0.038 | -0.022 | keep_z |
| TE | 197 | -1.391 | 0.035 | 197 | 0.055 | 0.081 | +0.025 | keep_z |
| OL | 373 | +0.925 | 0.005 | 374 | 0.069 | 0.081 | +0.012 | **switch_to_raw** |
| DL | 490 | -1.188 | 0.013 | 490 | 0.071 | 0.100 | +0.029 | **switch_to_raw** |
| LB | 361 | -1.640 | 0.041 | 361 | 0.033 | 0.054 | +0.021 | keep_z |
| DB | 584 | -0.642 | 0.008 | 584 | 0.013 | 0.019 | +0.006 | **switch_to_raw** |
| K | 3 | +nan | nan | 3 | nan | nan | +nan | insufficient_data |
| P | 4 | +nan | nan | 4 | nan | nan | +nan | insufficient_data |

## cone

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 144 | +0.003 | 0.000 | 145 | 0.145 | 0.127 | -0.018 | keep_z |
| RB | 204 | +0.034 | 0.008 | 204 | 0.113 | 0.133 | +0.020 | **switch_to_raw** |
| WR | 345 | +0.012 | 0.001 | 346 | 0.061 | 0.067 | +0.006 | **switch_to_raw** |
| TE | 165 | +0.026 | 0.006 | 166 | 0.195 | 0.164 | -0.031 | keep_z |
| OL | 391 | -0.062 | 0.011 | 392 | 0.121 | 0.112 | -0.009 | keep_z |
| DL | 422 | +0.092 | 0.017 | 422 | 0.124 | 0.138 | +0.013 | **switch_to_raw** |
| LB | 274 | -0.013 | 0.001 | 275 | 0.136 | 0.120 | -0.016 | keep_z |
| DB | 414 | +0.006 | 0.000 | 414 | 0.076 | 0.123 | +0.047 | **switch_to_raw** |
| K | 0 | +nan | nan | 0 | nan | nan | +nan | insufficient_data |
| P | 2 | +nan | nan | 2 | nan | nan | +nan | insufficient_data |

## shuttle

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 146 | +0.008 | 0.001 | 147 | 0.050 | 0.052 | +0.002 | keep_z |
| RB | 217 | +0.012 | 0.002 | 217 | 0.055 | 0.044 | -0.011 | keep_z |
| WR | 359 | +0.014 | 0.003 | 360 | 0.079 | 0.077 | -0.002 | keep_z |
| TE | 175 | +0.009 | 0.001 | 176 | 0.112 | 0.054 | -0.058 | keep_z |
| OL | 409 | -0.027 | 0.005 | 410 | 0.130 | 0.123 | -0.007 | keep_z |
| DL | 438 | +0.082 | 0.037 | 438 | 0.108 | 0.111 | +0.003 | keep_z |
| LB | 290 | +0.033 | 0.021 | 291 | 0.093 | 0.063 | -0.030 | keep_z |
| DB | 438 | +0.036 | 0.019 | 438 | 0.055 | 0.068 | +0.014 | **switch_to_raw** |
| K | 0 | +nan | nan | 0 | nan | nan | +nan | insufficient_data |
| P | 2 | +nan | nan | 2 | nan | nan | +nan | insufficient_data |

## ht

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 232 | -0.596 | 0.039 | 234 | 0.017 | 0.012 | -0.005 | keep_z |
| RB | 418 | -0.319 | 0.011 | 418 | 0.030 | 0.056 | +0.026 | **switch_to_raw** |
| WR | 635 | +0.064 | 0.000 | 636 | 0.042 | 0.037 | -0.005 | keep_z |
| TE | 285 | +0.194 | 0.008 | 286 | 0.066 | 0.066 | -0.000 | keep_z |
| OL | 643 | +0.284 | 0.009 | 644 | 0.036 | 0.058 | +0.022 | **switch_to_raw** |
| DL | 741 | +0.182 | 0.005 | 741 | 0.108 | 0.091 | -0.017 | keep_z |
| LB | 543 | +0.315 | 0.015 | 544 | 0.063 | 0.058 | -0.005 | keep_z |
| DB | 869 | +0.391 | 0.021 | 869 | 0.025 | 0.029 | +0.004 | keep_z |
| K | 46 | +nan | nan | 46 | 0.147 | 0.133 | -0.014 | insufficient_data |
| P | 38 | +nan | nan | 38 | 0.108 | 0.204 | +0.096 | insufficient_data |

## wt

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 232 | -5.284 | 0.080 | 234 | 0.073 | 0.065 | -0.008 | keep_z |
| RB | 419 | -5.922 | 0.058 | 419 | 0.045 | 0.081 | +0.035 | keep_z |
| WR | 635 | -3.526 | 0.018 | 636 | 0.042 | 0.035 | -0.007 | keep_z |
| TE | 285 | -3.733 | 0.071 | 286 | 0.078 | 0.037 | -0.041 | keep_z |
| OL | 643 | +2.281 | 0.007 | 644 | 0.040 | 0.057 | +0.017 | **switch_to_raw** |
| DL | 742 | -0.181 | 0.000 | 742 | 0.029 | 0.038 | +0.010 | **switch_to_raw** |
| LB | 543 | -2.761 | 0.021 | 544 | 0.035 | 0.078 | +0.044 | keep_z |
| DB | 869 | -3.121 | 0.029 | 869 | 0.027 | 0.047 | +0.020 | keep_z |
| K | 46 | +nan | nan | 46 | 0.191 | 0.189 | -0.002 | insufficient_data |
| P | 38 | +nan | nan | 38 | 0.086 | 0.159 | +0.072 | insufficient_data |

## arm

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 171 | -0.956 | 0.137 | 172 | 0.014 | 0.070 | +0.056 | keep_z |
| RB | 305 | -0.493 | 0.038 | 305 | 0.040 | 0.040 | +0.000 | keep_z |
| WR | 476 | -0.586 | 0.040 | 476 | 0.034 | 0.074 | +0.040 | keep_z |
| TE | 209 | -0.774 | 0.116 | 210 | 0.069 | 0.182 | +0.113 | keep_z |
| OL | 565 | -0.478 | 0.036 | 566 | 0.028 | 0.075 | +0.047 | keep_z |
| DL | 617 | -0.236 | 0.010 | 617 | 0.072 | 0.135 | +0.063 | **switch_to_raw** |
| LB | 406 | -0.208 | 0.006 | 406 | 0.094 | 0.135 | +0.040 | **switch_to_raw** |
| DB | 633 | -0.323 | 0.017 | 633 | 0.061 | 0.078 | +0.017 | **switch_to_raw** |
| K | 12 | +nan | nan | 12 | nan | nan | +nan | insufficient_data |
| P | 15 | +nan | nan | 15 | nan | nan | +nan | insufficient_data |

## hand

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 171 | -0.124 | 0.015 | 172 | 0.024 | 0.035 | +0.011 | **switch_to_raw** |
| RB | 305 | +0.020 | 0.000 | 305 | 0.052 | 0.067 | +0.016 | **switch_to_raw** |
| WR | 476 | -0.037 | 0.001 | 476 | 0.025 | 0.031 | +0.007 | **switch_to_raw** |
| TE | 209 | -0.120 | 0.011 | 210 | 0.022 | 0.038 | +0.016 | **switch_to_raw** |
| OL | 565 | -0.056 | 0.002 | 566 | 0.061 | 0.077 | +0.016 | **switch_to_raw** |
| DL | 616 | -0.039 | 0.001 | 616 | 0.116 | 0.141 | +0.025 | **switch_to_raw** |
| LB | 406 | -0.039 | 0.001 | 406 | 0.105 | 0.114 | +0.009 | **switch_to_raw** |
| DB | 633 | -0.022 | 0.000 | 633 | 0.051 | 0.057 | +0.006 | **switch_to_raw** |
| K | 12 | +nan | nan | 12 | nan | nan | +nan | insufficient_data |
| P | 15 | +nan | nan | 15 | nan | nan | +nan | insufficient_data |

## wingspan

| pos | n_drift | slope/dec | drift_r2 | n_corr | |corr_z| | |corr_raw| | Δ | rec |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| QB | 65 | -0.446 | 0.003 | 65 | 0.074 | 0.124 | +0.050 | **switch_to_raw** |
| RB | 130 | -1.443 | 0.022 | 130 | 0.043 | 0.053 | +0.010 | keep_z |
| WR | 199 | -0.726 | 0.004 | 199 | 0.017 | 0.030 | +0.013 | **switch_to_raw** |
| TE | 96 | +0.493 | 0.005 | 97 | 0.066 | 0.066 | -0.000 | keep_z |
| OL | 249 | +1.664 | 0.029 | 250 | 0.007 | 0.048 | +0.041 | keep_z |
| DL | 270 | +0.926 | 0.011 | 270 | 0.053 | 0.053 | +0.000 | keep_z |
| LB | 177 | +1.099 | 0.010 | 177 | 0.094 | 0.065 | -0.029 | keep_z |
| DB | 306 | +0.801 | 0.008 | 306 | 0.031 | 0.054 | +0.023 | **switch_to_raw** |
| K | 5 | +nan | nan | 5 | nan | nan | +nan | insufficient_data |
| P | 7 | +nan | nan | 7 | nan | nan | +nan | insufficient_data |

