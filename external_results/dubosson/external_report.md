# External Cohort Evaluation: `dubosson` (Engine v2.3, HBML-SFH-Sqrt)

Evaluated conditions: **12**.

## RMSE & Equivalence vs Best Single Expert
| Horizon | Context | N | HBML RMSE | Best Expert | Best RMSE | Diff (Best - HBML) | TOST p (1.0 mg/dL) | Interleaved RMSE | ShortCtx-Pool RMSE |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 0.5 hr | 6 | 7 | 20.05 | XGBoost | 21.69 | +1.64 | 0.939 | 20.79 | 20.43 |
| 0.5 hr | 12 | 7 | 19.15 | ARIMA | 19.45 | +0.31 | 0.1627 | 20.37 | 18.98 |
| 0.5 hr | 24 | 7 | 18.75 | ARIMA | 17.40 | -1.36 | 0.7629 | 20.27 | 17.57 |
| 0.5 hr | full | 7 | 18.72 | ARIMA | 17.18 | -1.54 | 0.8532 | 20.34 | 17.32 |
| 2 hr | 6 | 7 | 64.42 | NODE | 63.82 | -0.60 | 0.3255 | 62.22 | 70.26 |
| 2 hr | 12 | 7 | 64.72 | NODE | 63.82 | -0.90 | 0.4745 | 62.28 | 68.59 |
| 2 hr | 24 | 7 | 64.56 | NODE | 63.82 | -0.75 | 0.3715 | 62.55 | 68.30 |
| 2 hr | full | 7 | 65.02 | NODE | 63.82 | -1.20 | 0.6657 | 62.70 | 67.24 |
| 5 hr | 6 | 7 | 78.12 | RunningMean | 75.37 | -2.75 | 0.9303 | 79.72 | 97.91 |
| 5 hr | 12 | 7 | 78.33 | RunningMean | 75.37 | -2.95 | 0.9995 | 79.44 | 97.62 |
| 5 hr | 24 | 7 | 77.85 | RunningMean | 75.37 | -2.47 | 0.9283 | 79.23 | 89.31 |
| 5 hr | full | 7 | 78.37 | RunningMean | 75.37 | -3.00 | 0.9929 | 80.58 | 95.55 |

## Pre-Onset Hypoglycaemia Detection (<70 mg/dL, Full History)
| Horizon | Model | N | Episodes | Pre-onset Sens (%) | Above Chance (pp) | Cluster PPV (%) | FAR (scored/day) | FAR (cal/day) | Parkes A+B (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 hr | HBML | 7 | 21 | 90.5 | +77.1 | 70.0 | 0.50 | 0.33 | 99.7 |
| 0.5 hr | BestExpert | 7 | 21 | 100.0 | +84.5 | 65.6 | 0.61 | 0.41 | 99.9 |
| 0.5 hr | Persistence | 7 | 21 | 0.0 | -11.7 | 81.8 | 0.22 | 0.15 | 98.9 |
| 2 hr | HBML | 7 | 21 | 14.3 | +6.6 | 63.6 | 0.22 | 0.15 | 87.2 |
| 2 hr | BestExpert | 7 | 21 | 4.8 | +4.0 | 50.0 | 0.06 | 0.04 | 86.6 |
| 2 hr | Persistence | 7 | 21 | 14.3 | -0.7 | 28.6 | 0.83 | 0.55 | 85.9 |
| 5 hr | HBML | 7 | 21 | 0.0 | -0.6 | 0.0 | 0.06 | 0.04 | 80.1 |
| 5 hr | BestExpert | 7 | 21 | 0.0 | +0.0 | nan | 0.00 | 0.00 | 81.4 |
| 5 hr | Persistence | 7 | 21 | 14.3 | -1.8 | 21.1 | 0.83 | 0.55 | 76.4 |
