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
| weinstock | alg1_fs_c0.01 | 12 | +5.88 | 0 |
| weinstock | alg1_fs_c0.05 | 12 | +4.52 | 0 |
| weinstock | alg1_fs_d | 12 | +6.15 | 0 |
| weinstock | alg1_fs_fd | 12 | +5.19 | 0 |
| weinstock | alg1_noshare | 12 | +2.71 | 0 |
| weinstock | cum_decaymax | 12 | +2.57 | 0 |
| weinstock | current | 12 | +2.63 | 0 |
| weinstock | ftl | 12 | +3.27 | 0 |
| weinstock | sfh_sqrt | 12 | +0.00 | 0 |
| weinstock | uniform | 12 | +2.48 | 4 |

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
| weinstock | half | 6 | 197 | 26.41 | 25.73 | 25.70 | 24.58 | 26.10 | 25.85 | 25.52 | 25.25 | 25.27 | 25.06 | 24.91 | 26.14 |
| weinstock | half | 12 | 197 | 25.65 | 25.44 | 25.41 | 24.50 | 25.73 | 25.63 | 25.38 | 25.15 | 25.15 | 25.03 | 24.95 | 26.16 |
| weinstock | half | 24 | 197 | 25.31 | 25.08 | 25.05 | 24.43 | 25.33 | 25.33 | 25.20 | 25.04 | 25.03 | 24.97 | 24.95 | 26.35 |
| weinstock | half | full | 197 | 24.61 | 24.60 | 24.58 | 24.40 | 24.77 | 24.80 | 24.75 | 24.79 | 24.79 | 24.86 | 24.95 | 26.18 |
| weinstock | 2 | 6 | 195 | 61.22 | 62.95 | 62.92 | 60.79 | 63.29 | 62.18 | 64.00 | 64.99 | 65.22 | 64.59 | 62.84 | 58.84 |
| weinstock | 2 | 12 | 195 | 61.22 | 63.06 | 63.03 | 60.85 | 63.39 | 62.24 | 63.99 | 64.94 | 65.15 | 64.55 | 62.85 | 58.79 |
| weinstock | 2 | 24 | 195 | 61.22 | 63.09 | 63.06 | 60.85 | 63.42 | 62.13 | 63.99 | 65.00 | 65.18 | 64.58 | 62.86 | 58.62 |
| weinstock | 2 | full | 195 | 61.22 | 62.70 | 62.68 | 60.86 | 62.99 | 61.91 | 63.56 | 64.61 | 64.81 | 64.35 | 62.82 | 58.86 |
| weinstock | 5 | 6 | 195 | 77.16 | 78.33 | 78.32 | 77.23 | 78.58 | 79.14 | 83.29 | 85.10 | 85.11 | 85.33 | 84.47 | 81.29 |
| weinstock | 5 | 12 | 195 | 77.16 | 78.45 | 78.45 | 77.32 | 78.70 | 79.19 | 83.28 | 84.66 | 84.74 | 84.76 | 83.85 | 80.42 |
| weinstock | 5 | 24 | 195 | 77.16 | 78.53 | 78.52 | 77.26 | 78.78 | 79.08 | 83.30 | 84.48 | 84.49 | 84.42 | 83.38 | 79.53 |
| weinstock | 5 | full | 195 | 77.16 | 78.25 | 78.24 | 77.35 | 78.42 | 78.75 | 82.68 | 84.09 | 84.11 | 84.20 | 83.38 | 79.54 |
