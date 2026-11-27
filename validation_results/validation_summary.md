# Cached Engine v2.3 Validation & Missing-Data Audit

## 1. Cohort Retention Sensitivity (`retained_ge288` vs `all_scorable_ge1` vs `common_all_conditions`)
| Dataset | Horizon | Context | Cohort Set | N | HBML RMSE | Best Expert | Diff | TOST p |
|---|---|---|---|---:|---:|---|---:|---:|
| weinstock | 0.5 hr | full | retained_ge288 | 197 | 24.40 | ARIMA (24.61) | +0.21 | 0.0004998 |
| weinstock | 0.5 hr | full | all_scorable_ge1 | 200 | 24.34 | ARIMA (24.54) | +0.20 | 0.0004998 |
| weinstock | 0.5 hr | full | common_all_conditions | 195 | 24.38 | ARIMA (24.57) | +0.20 | 0.0004998 |
| weinstock | 2 hr | full | retained_ge288 | 195 | 60.86 | NODE (61.22) | +0.36 | 0.0009995 |
| weinstock | 2 hr | full | all_scorable_ge1 | 200 | 60.76 | NODE (61.17) | +0.41 | 0.0009995 |
| weinstock | 2 hr | full | common_all_conditions | 195 | 60.86 | NODE (61.22) | +0.36 | 0.0009995 |
| weinstock | 5 hr | full | retained_ge288 | 195 | 77.35 | RunningMean (77.16) | -0.19 | 0.0004998 |
| weinstock | 5 hr | full | all_scorable_ge1 | 200 | 77.02 | RunningMean (76.88) | -0.14 | 0.0009995 |
| weinstock | 5 hr | full | common_all_conditions | 195 | 77.35 | RunningMean (77.16) | -0.19 | 0.0004998 |
| cgmacros | 0.5 hr | full | retained_ge288 | 44 | 16.79 | ARIMA (16.98) | +0.19 | 0.0004998 |
| cgmacros | 0.5 hr | full | all_scorable_ge1 | 44 | 16.79 | ARIMA (16.98) | +0.19 | 0.0004998 |
| cgmacros | 0.5 hr | full | common_all_conditions | 44 | 16.79 | ARIMA (16.98) | +0.19 | 0.0004998 |
| cgmacros | 2 hr | full | retained_ge288 | 44 | 28.47 | RunningMean (29.27) | +0.80 | 0.2009 |
| cgmacros | 2 hr | full | all_scorable_ge1 | 44 | 28.47 | RunningMean (29.27) | +0.80 | 0.2009 |
| cgmacros | 2 hr | full | common_all_conditions | 44 | 28.47 | RunningMean (29.27) | +0.80 | 0.2009 |
| cgmacros | 5 hr | full | retained_ge288 | 44 | 29.71 | RunningMean (29.31) | -0.41 | 0.0004998 |
| cgmacros | 5 hr | full | all_scorable_ge1 | 44 | 29.71 | RunningMean (29.31) | -0.41 | 0.0004998 |
| cgmacros | 5 hr | full | common_all_conditions | 44 | 29.71 | RunningMean (29.31) | -0.41 | 0.0004998 |

## 2. Gap-Burden Stratification (Full-History Context)
| Dataset | Horizon | Stratum | N | Missing (%) | Bridged (%) | HBML RMSE | Best RMSE | Diff | Hypo Sens (%) | Above Chance (pp) | Cluster PPV (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weinstock | 0.5 hr | T1_LowGap | 66 | 5.6 | 2.3 | 23.91 | 24.10 | +0.19 | 54.2 | +38.6 | 75.3 |
| weinstock | 0.5 hr | T2_MedGap | 66 | 16.7 | 5.3 | 24.71 | 24.92 | +0.20 | 48.1 | +35.4 | 70.8 |
| weinstock | 0.5 hr | T3_HighGap | 65 | 43.5 | 7.4 | 24.58 | 24.82 | +0.24 | 47.4 | +34.8 | 70.4 |
| weinstock | 2 hr | T1_LowGap | 65 | 5.6 | 2.3 | 59.12 | 59.16 | +0.04 | 11.7 | +3.3 | 48.3 |
| weinstock | 2 hr | T2_MedGap | 65 | 16.4 | 5.2 | 61.99 | 62.20 | +0.21 | 8.9 | +2.6 | 49.0 |
| weinstock | 2 hr | T3_HighGap | 65 | 41.8 | 7.1 | 61.48 | 62.30 | +0.82 | 6.3 | +1.4 | 38.9 |
| weinstock | 5 hr | T1_LowGap | 65 | 5.6 | 2.3 | 74.28 | 74.08 | -0.21 | 1.5 | +0.4 | 32.8 |
| weinstock | 5 hr | T2_MedGap | 65 | 16.4 | 5.2 | 79.49 | 78.86 | -0.62 | 0.6 | -0.3 | 21.4 |
| weinstock | 5 hr | T3_HighGap | 65 | 41.8 | 7.1 | 78.29 | 78.55 | +0.26 | 0.5 | -0.1 | 18.8 |
| cgmacros | 0.5 hr | T1_LowGap | 15 | 0.0 | 0.0 | 14.64 | 14.81 | +0.18 | 0.0 | -0.3 | 16.7 |
| cgmacros | 0.5 hr | T2_MedGap | 15 | 0.0 | 0.0 | 16.93 | 17.11 | +0.18 | 0.0 | +0.0 | 100.0 |
| cgmacros | 0.5 hr | T3_HighGap | 14 | 0.0 | 0.0 | 18.95 | 19.16 | +0.21 | 12.5 | +12.4 | 50.0 |
| cgmacros | 2 hr | T1_LowGap | 15 | 0.0 | 0.0 | 23.15 | 23.44 | +0.29 | 0.0 | -0.1 | 100.0 |
| cgmacros | 2 hr | T2_MedGap | 15 | 0.0 | 0.0 | 26.07 | 26.53 | +0.46 | 0.0 | +0.0 | nan |
| cgmacros | 2 hr | T3_HighGap | 14 | 0.0 | 0.0 | 36.75 | 37.89 | +1.14 | 0.0 | +0.0 | nan |
| cgmacros | 5 hr | T1_LowGap | 15 | 0.0 | 0.0 | 23.76 | 23.49 | -0.27 | 0.0 | +0.0 | nan |
| cgmacros | 5 hr | T2_MedGap | 15 | 0.0 | 0.0 | 26.92 | 26.55 | -0.37 | 0.0 | +0.0 | nan |
| cgmacros | 5 hr | T3_HighGap | 14 | 0.0 | 0.0 | 39.09 | 38.49 | -0.60 | 0.0 | +0.0 | nan |
