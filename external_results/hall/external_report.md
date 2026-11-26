# External Cohort Evaluation: `hall` (Engine v2.3, HBML-SFH-Sqrt)

Evaluated conditions: **12**.

## RMSE & Equivalence vs Best Single Expert
| Horizon | Context | N | HBML RMSE | Best Expert | Best RMSE | Diff (Best - HBML) | TOST p (1.0 mg/dL) | Interleaved RMSE | ShortCtx-Pool RMSE |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 0.5 hr | 6 | 57 | 12.52 | NODE | 13.04 | +0.53 | 0.0002 | 12.44 | 12.88 |
| 0.5 hr | 12 | 57 | 12.54 | NODE | 13.04 | +0.51 | 9.999e-05 | 12.46 | 12.88 |
| 0.5 hr | 24 | 57 | 12.48 | NODE | 13.04 | +0.57 | 0.0003 | 12.41 | 12.78 |
| 0.5 hr | full | 57 | 12.42 | ARIMA | 12.71 | +0.30 | 9.999e-05 | 12.38 | 12.52 |
| 2 hr | 6 | 57 | 18.79 | NODE | 19.15 | +0.35 | 9.999e-05 | 18.88 | 20.85 |
| 2 hr | 12 | 57 | 18.88 | NODE | 19.15 | +0.26 | 9.999e-05 | 18.80 | 20.71 |
| 2 hr | 24 | 57 | 18.84 | NODE | 19.15 | +0.30 | 9.999e-05 | 18.76 | 20.82 |
| 2 hr | full | 57 | 18.88 | NODE | 19.15 | +0.27 | 9.999e-05 | 18.75 | 20.12 |
| 5 hr | 6 | 57 | 20.04 | RunningMean | 19.58 | -0.46 | 9.999e-05 | 20.43 | 23.55 |
| 5 hr | 12 | 57 | 19.88 | RunningMean | 19.58 | -0.30 | 9.999e-05 | 19.99 | 22.41 |
| 5 hr | 24 | 57 | 19.87 | RunningMean | 19.58 | -0.29 | 9.999e-05 | 19.90 | 22.10 |
| 5 hr | full | 57 | 19.92 | RunningMean | 19.58 | -0.34 | 9.999e-05 | 19.85 | 21.61 |

## Pre-Onset Hypoglycaemia Detection (<70 mg/dL, Full History)
| Horizon | Model | N | Episodes | Pre-onset Sens (%) | Above Chance (pp) | Cluster PPV (%) | FAR (scored/day) | FAR (cal/day) | Parkes A+B (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 hr | HBML | 57 | 252 | 5.2 | -5.9 | 72.2 | 0.23 | 0.04 | 99.9 |
| 0.5 hr | BestExpert | 57 | 252 | 36.1 | +22.1 | 53.8 | 0.57 | 0.10 | 99.9 |
| 0.5 hr | Persistence | 57 | 252 | 3.6 | -10.7 | 62.2 | 0.46 | 0.08 | 99.8 |
| 2 hr | HBML | 57 | 253 | 5.1 | +1.5 | 45.8 | 0.10 | 0.02 | 99.3 |
| 2 hr | BestExpert | 57 | 253 | 2.8 | -0.3 | 34.8 | 0.11 | 0.02 | 99.3 |
| 2 hr | Persistence | 57 | 253 | 25.3 | +7.3 | 32.3 | 0.83 | 0.14 | 98.5 |
| 5 hr | HBML | 57 | 243 | 2.9 | +0.4 | 24.4 | 0.12 | 0.02 | 99.1 |
| 5 hr | BestExpert | 57 | 243 | 0.0 | -0.1 | 0.0 | 0.00 | 0.00 | 99.1 |
| 5 hr | Persistence | 57 | 243 | 22.6 | +4.2 | 25.8 | 0.93 | 0.15 | 98.3 |
