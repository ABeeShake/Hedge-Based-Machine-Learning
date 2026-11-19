# Aggregator & Fixed-Share comparison

RMSE is per participant, mg/dL. Ratios are variant RMSE / primary HBML-SFH-Sqrt RMSE (below 1 = better than the primary algorithm).

## Average over conditions

| dataset | variant | conditions | mean % change in RMSE vs current | conditions where it beats current |
|---|---|---|---|---|
| cgmacros | alg1_fs_c0.001 | 12 | +6.74 | 0 |
| cgmacros | alg1_fs_c0.01 | 12 | +6.70 | 0 |
| cgmacros | alg1_fs_c0.05 | 12 | +6.02 | 0 |
| cgmacros | alg1_fs_d | 12 | +6.82 | 0 |
| cgmacros | alg1_fs_fd | 12 | +6.00 | 0 |
| cgmacros | alg1_noshare | 12 | +3.73 | 0 |
| cgmacros | cum_decaymax | 12 | +2.21 | 0 |
| cgmacros | current | 12 | +2.29 | 0 |
| cgmacros | ftl | 12 | +2.71 | 0 |
| cgmacros | sfh_sqrt | 12 | +0.00 | 0 |
| cgmacros | uniform | 12 | +4.82 | 2 |
| weinstock | alg1_fs_c0.001 | 12 | +6.28 | 0 |
| weinstock | alg1_fs_c0.01 | 12 | +5.87 | 0 |
| weinstock | alg1_fs_c0.05 | 12 | +4.51 | 0 |
| weinstock | alg1_fs_d | 12 | +6.13 | 0 |
| weinstock | alg1_fs_fd | 12 | +5.19 | 0 |
| weinstock | alg1_noshare | 12 | +2.72 | 0 |
| weinstock | cum_decaymax | 12 | +2.49 | 0 |
| weinstock | current | 12 | +2.55 | 0 |
| weinstock | ftl | 12 | +3.18 | 0 |
| weinstock | sfh_sqrt | 12 | +0.00 | 0 |
| weinstock | uniform | 12 | +2.47 | 4 |

## Held-out choice (odd / even participant IDs)

| dataset | chosen on | chosen variant | ratio on the selection half | ratio on the held-out half |
|---|---|---|---|---|
| cgmacros | odd | sfh_sqrt | 1.0000 | 1.0000 |
| cgmacros | even | sfh_sqrt | 1.0000 | 1.0000 |
| weinstock | odd | sfh_sqrt | 1.0000 | 1.0000 |
| weinstock | even | sfh_sqrt | 1.0000 | 1.0000 |

## Per condition (mean RMSE)

| dataset | horizon | context | n | best expert | current | cum_decaymax | sfh_sqrt | ftl | alg1_noshare | alg1_fs_fd | alg1_fs_d | alg1_fs_c0.001 | alg1_fs_c0.01 | alg1_fs_c0.05 | uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cgmacros | half | 6 | 44 | 18.01 | 17.87 | 17.85 | 17.06 | 18.01 | 18.22 | 17.98 | 17.82 | 17.84 | 17.70 | 17.48 | 16.97 |
| cgmacros | half | 12 | 44 | 18.01 | 17.80 | 17.77 | 17.03 | 17.94 | 18.12 | 17.93 | 17.84 | 17.84 | 17.70 | 17.48 | 17.00 |
| cgmacros | half | 24 | 44 | 17.87 | 17.55 | 17.54 | 16.95 | 17.67 | 17.99 | 17.87 | 17.75 | 17.75 | 17.59 | 17.39 | 17.02 |
| cgmacros | half | full | 44 | 16.98 | 17.06 | 17.04 | 16.79 | 17.13 | 17.65 | 17.56 | 17.54 | 17.54 | 17.48 | 17.32 | 17.01 |
| cgmacros | 2 | 6 | 44 | 29.27 | 29.15 | 29.13 | 28.34 | 29.22 | 29.37 | 30.52 | 30.54 | 30.52 | 30.37 | 30.20 | 30.24 |
| cgmacros | 2 | 12 | 44 | 29.27 | 29.12 | 29.11 | 28.38 | 29.20 | 29.47 | 30.47 | 30.68 | 30.63 | 30.56 | 30.31 | 29.96 |
| cgmacros | 2 | 24 | 44 | 29.27 | 29.27 | 29.25 | 28.47 | 29.35 | 29.46 | 30.23 | 30.42 | 30.38 | 30.34 | 30.07 | 29.54 |
| cgmacros | 2 | full | 44 | 29.27 | 29.15 | 29.14 | 28.47 | 29.23 | 29.31 | 30.13 | 30.39 | 30.38 | 30.33 | 30.03 | 29.34 |
| cgmacros | 5 | 6 | 44 | 29.31 | 29.94 | 29.92 | 29.84 | 30.06 | 30.45 | 31.77 | 32.41 | 32.32 | 32.66 | 32.90 | 33.88 |
| cgmacros | 5 | 12 | 44 | 29.31 | 29.92 | 29.91 | 29.74 | 30.03 | 30.20 | 31.54 | 32.32 | 32.26 | 32.67 | 32.72 | 32.86 |
| cgmacros | 5 | 24 | 44 | 29.31 | 29.92 | 29.91 | 29.69 | 29.98 | 30.13 | 31.47 | 32.23 | 32.19 | 32.28 | 32.17 | 31.93 |
| cgmacros | 5 | full | 44 | 29.31 | 29.90 | 29.88 | 29.71 | 29.98 | 30.10 | 31.45 | 32.15 | 32.14 | 32.34 | 32.21 | 31.53 |
| weinstock | half | 6 | 197 | 26.41 | 25.70 | 25.67 | 24.56 | 26.07 | 25.84 | 25.50 | 25.24 | 25.26 | 25.06 | 24.90 | 26.13 |
| weinstock | half | 12 | 197 | 25.65 | 25.43 | 25.40 | 24.50 | 25.72 | 25.62 | 25.36 | 25.12 | 25.13 | 25.01 | 24.94 | 26.15 |
| weinstock | half | 24 | 197 | 25.31 | 25.07 | 25.04 | 24.43 | 25.30 | 25.32 | 25.17 | 25.03 | 25.01 | 24.96 | 24.95 | 26.36 |
| weinstock | half | full | 197 | 24.61 | 24.60 | 24.58 | 24.40 | 24.76 | 24.79 | 24.74 | 24.79 | 24.78 | 24.87 | 24.95 | 26.17 |
| weinstock | 2 | 6 | 195 | 61.15 | 62.90 | 62.87 | 60.79 | 63.26 | 62.24 | 64.13 | 64.99 | 65.24 | 64.54 | 62.78 | 58.83 |
| weinstock | 2 | 12 | 195 | 61.15 | 62.88 | 62.85 | 60.83 | 63.19 | 62.11 | 63.93 | 64.81 | 65.04 | 64.43 | 62.77 | 58.76 |
| weinstock | 2 | 24 | 195 | 61.15 | 63.00 | 62.97 | 60.81 | 63.33 | 62.04 | 63.93 | 64.84 | 65.06 | 64.44 | 62.75 | 58.59 |
| weinstock | 2 | full | 195 | 61.15 | 62.64 | 62.61 | 60.87 | 62.91 | 61.88 | 63.54 | 64.58 | 64.79 | 64.27 | 62.79 | 58.87 |
| weinstock | 5 | 6 | 195 | 77.19 | 78.23 | 78.22 | 77.25 | 78.49 | 79.23 | 83.29 | 85.12 | 85.15 | 85.41 | 84.54 | 81.33 |
| weinstock | 5 | 12 | 195 | 77.19 | 78.39 | 78.37 | 77.29 | 78.64 | 79.26 | 83.29 | 84.70 | 84.79 | 84.77 | 83.81 | 80.39 |
| weinstock | 5 | 24 | 195 | 77.19 | 78.58 | 78.57 | 77.34 | 78.84 | 79.24 | 83.24 | 84.61 | 84.66 | 84.56 | 83.47 | 79.47 |
| weinstock | 5 | full | 195 | 77.19 | 78.13 | 78.11 | 77.35 | 78.33 | 78.77 | 82.97 | 84.20 | 84.26 | 84.31 | 83.45 | 79.57 |
