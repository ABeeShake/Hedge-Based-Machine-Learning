# Cached Engine v2.3 Validation & Missing-Data Audit

## 1. Cohort Retention Sensitivity (`retained_ge288` vs `all_scorable_ge1` vs `common_all_conditions`)
| Dataset | Horizon | Context | Cohort Set | N | HBML RMSE | Best Expert | Diff | TOST p |
|---|---|---|---|---:|---:|---|---:|---:|
| weinstock | 0.5 hr | full | retained_ge288 | 197 | 24.38 | ARIMA (24.58) | +0.20 | 0.0004998 |
| weinstock | 0.5 hr | full | all_scorable_ge1 | 200 | 24.31 | ARIMA (24.51) | +0.19 | 0.0004998 |
| weinstock | 0.5 hr | full | common_all_conditions | 195 | 24.35 | ARIMA (24.54) | +0.19 | 0.0004998 |
| weinstock | 2 hr | full | retained_ge288 | 195 | 60.60 | NODE (60.98) | +0.38 | 0.0009995 |
| weinstock | 2 hr | full | all_scorable_ge1 | 200 | 60.51 | NODE (60.93) | +0.43 | 0.001999 |
| weinstock | 2 hr | full | common_all_conditions | 195 | 60.60 | NODE (60.98) | +0.38 | 0.0009995 |
| weinstock | 5 hr | full | retained_ge288 | 195 | 76.84 | RunningMean (76.67) | -0.17 | 0.0004998 |
| weinstock | 5 hr | full | all_scorable_ge1 | 200 | 76.51 | RunningMean (76.40) | -0.11 | 0.0004998 |
| weinstock | 5 hr | full | common_all_conditions | 195 | 76.84 | RunningMean (76.67) | -0.17 | 0.0004998 |
| cgmacros | 0.5 hr | full | retained_ge288 | 44 | 16.79 | ARIMA (16.96) | +0.17 | 0.0004998 |
| cgmacros | 0.5 hr | full | all_scorable_ge1 | 44 | 16.79 | ARIMA (16.96) | +0.17 | 0.0004998 |
| cgmacros | 0.5 hr | full | common_all_conditions | 44 | 16.79 | ARIMA (16.96) | +0.17 | 0.0004998 |
| cgmacros | 2 hr | full | retained_ge288 | 44 | 28.42 | NODE (29.15) | +0.73 | 0.07146 |
| cgmacros | 2 hr | full | all_scorable_ge1 | 44 | 28.42 | NODE (29.15) | +0.73 | 0.07146 |
| cgmacros | 2 hr | full | common_all_conditions | 44 | 28.42 | NODE (29.15) | +0.73 | 0.07146 |
| cgmacros | 5 hr | full | retained_ge288 | 44 | 29.21 | RunningMean (28.93) | -0.28 | 0.0004998 |
| cgmacros | 5 hr | full | all_scorable_ge1 | 44 | 29.21 | RunningMean (28.93) | -0.28 | 0.0004998 |
| cgmacros | 5 hr | full | common_all_conditions | 44 | 29.21 | RunningMean (28.93) | -0.28 | 0.0004998 |

## 2. Gap-Burden Stratification (Full-History Context)
| Dataset | Horizon | Stratum | N | Missing (%) | Bridged (%) | HBML RMSE | Best RMSE | Diff | Hypo Sens (%) | Above Chance (pp) | Cluster PPV (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weinstock | 0.5 hr | T1_LowGap | 66 | 5.6 | 2.3 | 23.87 | 24.05 | +0.18 | 54.2 | +38.4 | 71.9 |
| weinstock | 0.5 hr | T2_MedGap | 66 | 16.7 | 5.3 | 24.69 | 24.89 | +0.19 | 48.3 | +35.3 | 67.9 |
| weinstock | 0.5 hr | T3_HighGap | 65 | 43.5 | 7.4 | 24.57 | 24.80 | +0.23 | 48.0 | +35.1 | 67.3 |
| weinstock | 2 hr | T1_LowGap | 65 | 5.6 | 2.3 | 58.72 | 58.75 | +0.03 | 13.8 | +4.3 | 46.0 |
| weinstock | 2 hr | T2_MedGap | 65 | 16.4 | 5.2 | 61.79 | 62.01 | +0.21 | 9.8 | +2.3 | 45.0 |
| weinstock | 2 hr | T3_HighGap | 65 | 41.8 | 7.1 | 61.28 | 62.17 | +0.89 | 7.5 | +1.8 | 34.4 |
| weinstock | 5 hr | T1_LowGap | 65 | 5.6 | 2.3 | 73.64 | 73.42 | -0.23 | 3.7 | +0.5 | 27.4 |
| weinstock | 5 hr | T2_MedGap | 65 | 16.4 | 5.2 | 78.89 | 78.32 | -0.57 | 2.2 | -0.5 | 23.2 |
| weinstock | 5 hr | T3_HighGap | 65 | 41.8 | 7.1 | 77.97 | 78.27 | +0.30 | 1.8 | -0.6 | 17.6 |
| cgmacros | 0.5 hr | T1_LowGap | 15 | 0.0 | 0.0 | 14.64 | 14.79 | +0.16 | 0.0 | -0.6 | 4.8 |
| cgmacros | 0.5 hr | T2_MedGap | 15 | 0.0 | 0.0 | 16.93 | 17.09 | +0.15 | 0.0 | -0.5 | 6.2 |
| cgmacros | 0.5 hr | T3_HighGap | 14 | 0.0 | 0.0 | 18.96 | 19.15 | +0.19 | 12.5 | +12.0 | 11.8 |
| cgmacros | 2 hr | T1_LowGap | 15 | 0.0 | 0.0 | 23.15 | 23.35 | +0.20 | 0.0 | -1.2 | 11.8 |
| cgmacros | 2 hr | T2_MedGap | 15 | 0.0 | 0.0 | 26.04 | 26.44 | +0.40 | 0.0 | -1.2 | 0.0 |
| cgmacros | 2 hr | T3_HighGap | 14 | 0.0 | 0.0 | 36.62 | 37.70 | +1.09 | 0.0 | -1.2 | 0.0 |
| cgmacros | 5 hr | T1_LowGap | 15 | 0.0 | 0.0 | 23.40 | 23.21 | -0.19 | 25.0 | +23.2 | 6.7 |
| cgmacros | 5 hr | T2_MedGap | 15 | 0.0 | 0.0 | 26.41 | 26.16 | -0.25 | 0.0 | -2.7 | 0.0 |
| cgmacros | 5 hr | T3_HighGap | 14 | 0.0 | 0.0 | 38.45 | 38.03 | -0.42 | 12.5 | +10.3 | 14.3 |
