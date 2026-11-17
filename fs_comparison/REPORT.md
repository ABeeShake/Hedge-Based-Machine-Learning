# Fixed-Share comparison

RMSE is per participant, mg/dL. Ratios are variant RMSE / current HBML-SFHDF RMSE (below 1 = better than the current algorithm).

## Average over conditions

| dataset | variant | conditions | mean % change in RMSE vs current | conditions where it beats current |
|---|---|---|---|---|
| cgmacros | alg1_fs_c0.001 | 12 | +4.60 | 0 |
| cgmacros | alg1_fs_c0.01 | 12 | +4.48 | 2 |
| cgmacros | alg1_fs_c0.05 | 12 | +4.04 | 3 |
| cgmacros | alg1_fs_d | 12 | +4.67 | 0 |
| cgmacros | alg1_fs_fd | 12 | +4.12 | 0 |
| cgmacros | alg1_noshare | 12 | +2.09 | 0 |
| cgmacros | cum_decaymax | 12 | -0.04 | 7 |
| cgmacros | current | 12 | +0.00 | 0 |
| cgmacros | ftl | 12 | +0.64 | 0 |
| cgmacros | sfh_sqrt | 12 | -2.10 | 8 |
| cgmacros | uniform | 12 | +4.18 | 5 |
| weinstock | alg1_fs_c0.001 | 12 | +3.76 | 2 |
| weinstock | alg1_fs_c0.01 | 12 | +2.76 | 6 |
| weinstock | alg1_fs_c0.05 | 12 | +0.91 | 6 |
| weinstock | alg1_fs_d | 12 | +3.54 | 2 |
| weinstock | alg1_fs_fd | 12 | +3.60 | 2 |
| weinstock | alg1_noshare | 12 | +0.66 | 4 |
| weinstock | cum_decaymax | 12 | -0.06 | 12 |
| weinstock | current | 12 | +0.00 | 0 |
| weinstock | ftl | 12 | +1.05 | 0 |
| weinstock | sfh_sqrt | 12 | -4.18 | 12 |
| weinstock | uniform | 12 | -0.75 | 4 |

## Held-out choice (odd / even participant IDs)

| dataset | chosen on | chosen variant | ratio on the selection half | ratio on the held-out half |
|---|---|---|---|---|
| cgmacros | odd | sfh_sqrt | 0.9784 | 0.9832 |
| cgmacros | even | sfh_sqrt | 0.9832 | 0.9784 |
| weinstock | odd | sfh_sqrt | 0.9588 | 0.9581 |
| weinstock | even | sfh_sqrt | 0.9581 | 0.9588 |

## Per condition (mean RMSE)

| dataset | horizon | context | n | best expert | current | cum_decaymax | sfh_sqrt | ftl | alg1_noshare | alg1_fs_fd | alg1_fs_d | alg1_fs_c0.001 | alg1_fs_c0.01 | alg1_fs_c0.05 | uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cgmacros | half | 6 | 44 | 18.01 | 17.91 | 17.89 | 17.11 | 18.07 | 18.30 | 18.02 | 17.92 | 17.94 | 17.80 | 17.55 | 17.02 |
| cgmacros | half | 12 | 44 | 18.01 | 17.82 | 17.79 | 17.04 | 17.98 | 18.20 | 17.94 | 17.85 | 17.87 | 17.73 | 17.51 | 17.10 |
| cgmacros | half | 24 | 44 | 17.87 | 17.59 | 17.57 | 16.99 | 17.71 | 17.99 | 17.82 | 17.76 | 17.78 | 17.64 | 17.42 | 17.14 |
| cgmacros | half | full | 44 | 16.98 | 17.06 | 17.05 | 16.81 | 17.14 | 17.46 | 17.46 | 17.47 | 17.48 | 17.49 | 17.35 | 17.10 |
| cgmacros | 2 | 6 | 44 | 29.27 | 29.51 | 29.50 | 28.54 | 29.63 | 30.81 | 31.49 | 31.30 | 31.33 | 30.95 | 30.58 | 30.28 |
| cgmacros | 2 | 12 | 44 | 29.27 | 29.59 | 29.59 | 28.56 | 29.82 | 30.57 | 31.64 | 31.40 | 31.47 | 31.02 | 30.64 | 30.04 |
| cgmacros | 2 | 24 | 44 | 29.27 | 29.71 | 29.70 | 28.66 | 29.89 | 30.42 | 31.39 | 31.19 | 31.27 | 30.92 | 30.43 | 29.64 |
| cgmacros | 2 | full | 44 | 29.27 | 29.60 | 29.59 | 28.61 | 29.80 | 30.57 | 31.45 | 31.18 | 31.28 | 30.95 | 30.46 | 29.51 |
| cgmacros | 5 | 6 | 44 | 29.31 | 30.04 | 30.05 | 30.17 | 30.30 | 30.30 | 31.72 | 32.24 | 32.12 | 32.54 | 33.30 | 35.50 |
| cgmacros | 5 | 12 | 44 | 29.31 | 29.88 | 29.89 | 30.10 | 30.06 | 30.03 | 31.20 | 32.15 | 31.93 | 32.57 | 33.09 | 34.56 |
| cgmacros | 5 | 24 | 44 | 29.31 | 29.96 | 29.96 | 30.11 | 30.06 | 30.20 | 31.21 | 32.28 | 32.13 | 32.50 | 32.62 | 33.69 |
| cgmacros | 5 | full | 44 | 29.31 | 29.84 | 29.85 | 30.02 | 29.99 | 30.02 | 31.27 | 32.03 | 31.88 | 32.21 | 32.38 | 33.31 |
| weinstock | half | 6 | 200 | 26.33 | 25.70 | 25.68 | 24.52 | 26.11 | 25.82 | 25.49 | 25.23 | 25.26 | 25.08 | 24.97 | 26.46 |
| weinstock | half | 12 | 200 | 25.55 | 25.36 | 25.33 | 24.42 | 25.67 | 25.57 | 25.35 | 25.12 | 25.13 | 25.05 | 25.01 | 26.65 |
| weinstock | half | 24 | 200 | 25.23 | 25.01 | 24.99 | 24.37 | 25.31 | 25.30 | 25.14 | 25.02 | 25.02 | 25.01 | 25.04 | 27.02 |
| weinstock | half | full | 200 | 24.54 | 24.54 | 24.53 | 24.32 | 24.77 | 24.73 | 24.72 | 24.77 | 24.77 | 24.89 | 25.03 | 26.60 |
| weinstock | 2 | 6 | 200 | 66.21 | 66.84 | 66.80 | 61.56 | 67.73 | 66.33 | 68.62 | 67.89 | 68.19 | 66.20 | 63.38 | 59.64 |
| weinstock | 2 | 12 | 200 | 66.21 | 66.62 | 66.59 | 61.57 | 67.39 | 66.44 | 68.62 | 67.76 | 68.07 | 66.12 | 63.33 | 59.62 |
| weinstock | 2 | 24 | 200 | 66.21 | 66.54 | 66.50 | 61.51 | 67.43 | 66.38 | 68.84 | 67.76 | 68.09 | 66.03 | 63.20 | 59.49 |
| weinstock | 2 | full | 200 | 64.11 | 65.33 | 65.29 | 61.53 | 65.95 | 65.13 | 68.02 | 67.49 | 67.80 | 66.02 | 63.37 | 59.81 |
| weinstock | 5 | 6 | 200 | 76.92 | 79.24 | 79.22 | 77.40 | 79.94 | 80.61 | 84.89 | 86.44 | 86.55 | 86.78 | 85.84 | 81.82 |
| weinstock | 5 | 12 | 200 | 76.92 | 79.38 | 79.36 | 77.48 | 79.77 | 80.49 | 85.04 | 86.30 | 86.32 | 86.53 | 85.51 | 80.89 |
| weinstock | 5 | 24 | 200 | 76.92 | 79.51 | 79.49 | 77.43 | 80.19 | 80.58 | 85.87 | 86.85 | 87.04 | 86.65 | 85.16 | 79.91 |
| weinstock | 5 | full | 200 | 76.92 | 79.27 | 79.24 | 77.56 | 79.79 | 80.63 | 85.07 | 86.30 | 86.40 | 86.59 | 85.62 | 80.26 |
