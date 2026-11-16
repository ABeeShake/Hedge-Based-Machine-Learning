# Fixed-Share comparison

RMSE is per participant, mg/dL. Ratios are variant RMSE / current HBML-SFHDF RMSE (below 1 = better than the current algorithm).

## Average over conditions

| dataset | variant | conditions | mean % change in RMSE vs current | conditions where it beats current |
|---|---|---|---|---|
| cgmacros | alg1_fs_c0.001 | 12 | +2.25 | 4 |
| cgmacros | alg1_fs_c0.01 | 12 | +1.70 | 5 |
| cgmacros | alg1_fs_c0.05 | 12 | +0.55 | 8 |
| cgmacros | alg1_fs_d | 12 | +2.17 | 4 |
| cgmacros | alg1_fs_fd | 12 | +2.06 | 2 |
| cgmacros | alg1_noshare | 12 | +3.05 | 1 |
| cgmacros | cum_decaymax | 12 | -0.06 | 11 |
| cgmacros | current | 12 | +0.00 | 0 |
| cgmacros | uniform | 12 | +0.94 | 5 |
| weinstock | alg1_fs_c0.001 | 12 | +3.05 | 4 |
| weinstock | alg1_fs_c0.01 | 12 | +2.91 | 3 |
| weinstock | alg1_fs_c0.05 | 12 | +1.89 | 3 |
| weinstock | alg1_fs_d | 12 | +2.96 | 4 |
| weinstock | alg1_fs_fd | 12 | +2.09 | 4 |
| weinstock | alg1_noshare | 12 | +0.06 | 5 |
| weinstock | cum_decaymax | 12 | -0.07 | 12 |
| weinstock | current | 12 | +0.00 | 0 |
| weinstock | uniform | 12 | +3.85 | 3 |

## Held-out choice (odd / even participant IDs)

| dataset | chosen on | chosen variant | ratio on the selection half | ratio on the held-out half |
|---|---|---|---|---|
| cgmacros | odd | alg1_fs_c0.05 | 0.9994 | 0.9986 |
| cgmacros | even | alg1_fs_c0.05 | 0.9986 | 0.9994 |
| weinstock | odd | alg1_noshare | 0.9992 | 1.0015 |
| weinstock | even | cum_decaymax | 0.9992 | 0.9994 |

## Per condition (mean RMSE)

| dataset | horizon | context | n | best expert | current | alg1_fs_fd | alg1_noshare | alg1_fs_d | alg1_fs_c0.001 | alg1_fs_c0.01 | alg1_fs_c0.05 | cum_decaymax | uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cgmacros | half | 6 | 44 | 22.59 | 21.95 | 21.92 | 22.15 | 21.69 | 21.70 | 21.65 | 21.47 | 21.92 | 21.40 |
| cgmacros | half | 12 | 44 | 22.04 | 21.32 | 21.39 | 21.75 | 21.27 | 21.28 | 21.25 | 21.11 | 21.30 | 21.20 |
| cgmacros | half | 24 | 44 | 21.34 | 20.75 | 20.83 | 21.26 | 20.80 | 20.82 | 20.88 | 20.80 | 20.74 | 20.79 |
| cgmacros | half | full | 44 | 20.74 | 20.44 | 20.56 | 21.24 | 20.52 | 20.55 | 20.47 | 20.41 | 20.43 | 20.46 |
| cgmacros | 2 | 6 | 44 | 39.76 | 39.72 | 41.34 | 42.38 | 40.49 | 40.53 | 39.57 | 38.56 | 39.67 | 37.58 |
| cgmacros | 2 | 12 | 44 | 36.68 | 37.35 | 37.85 | 38.60 | 37.27 | 37.32 | 36.79 | 36.03 | 37.32 | 35.32 |
| cgmacros | 2 | 24 | 44 | 34.70 | 35.57 | 36.08 | 35.98 | 36.12 | 36.19 | 35.75 | 35.05 | 35.54 | 34.36 |
| cgmacros | 2 | full | 44 | 32.12 | 32.62 | 34.25 | 34.49 | 34.35 | 34.44 | 34.22 | 33.89 | 32.59 | 33.60 |
| cgmacros | 5 | 6 | 44 | 41.14 | 44.35 | 47.04 | 46.94 | 49.66 | 49.59 | 49.66 | 48.69 | 44.33 | 49.91 |
| cgmacros | 5 | 12 | 44 | 39.48 | 40.35 | 42.24 | 41.81 | 41.30 | 41.39 | 40.85 | 40.16 | 40.30 | 40.53 |
| cgmacros | 5 | 24 | 44 | 36.75 | 37.53 | 36.69 | 37.12 | 37.21 | 37.24 | 37.29 | 37.13 | 37.50 | 38.33 |
| cgmacros | 5 | full | 44 | 32.72 | 34.20 | 35.24 | 34.85 | 35.80 | 35.75 | 35.96 | 36.01 | 34.25 | 37.66 |
| weinstock | half | 6 | 200 | 35.05 | 34.47 | 34.16 | 34.57 | 34.25 | 34.25 | 34.36 | 34.26 | 34.42 | 36.34 |
| weinstock | half | 12 | 200 | 34.61 | 33.78 | 33.65 | 33.80 | 33.69 | 33.67 | 33.74 | 33.79 | 33.75 | 36.23 |
| weinstock | half | 24 | 200 | 34.45 | 33.39 | 33.25 | 33.38 | 33.37 | 33.39 | 33.42 | 33.22 | 33.36 | 36.19 |
| weinstock | half | full | 200 | 33.95 | 32.89 | 32.80 | 33.01 | 32.83 | 32.81 | 32.87 | 32.94 | 32.87 | 35.69 |
| weinstock | 2 | 6 | 200 | 72.10 | 73.46 | 78.47 | 74.29 | 80.38 | 80.53 | 80.28 | 78.71 | 73.42 | 75.20 |
| weinstock | 2 | 12 | 200 | 72.43 | 73.19 | 76.06 | 73.24 | 76.63 | 76.67 | 76.35 | 75.25 | 73.15 | 72.71 |
| weinstock | 2 | 24 | 200 | 69.84 | 71.40 | 73.05 | 71.88 | 73.50 | 73.59 | 73.34 | 72.36 | 71.36 | 71.08 |
| weinstock | 2 | full | 200 | 67.50 | 68.88 | 70.34 | 69.50 | 70.73 | 70.86 | 70.83 | 70.21 | 68.86 | 69.58 |
| weinstock | 5 | 6 | 200 | 99.13 | 103.68 | 111.15 | 103.53 | 113.35 | 113.23 | 112.95 | 111.00 | 103.63 | 113.93 |
| weinstock | 5 | 12 | 200 | 99.24 | 104.08 | 107.11 | 103.77 | 107.36 | 107.47 | 106.46 | 104.40 | 104.00 | 103.57 |
| weinstock | 5 | 24 | 200 | 91.40 | 95.99 | 96.35 | 94.76 | 97.35 | 97.48 | 97.18 | 95.93 | 95.93 | 96.44 |
| weinstock | 5 | full | 200 | 86.73 | 89.12 | 90.42 | 88.24 | 91.89 | 92.20 | 92.45 | 91.84 | 89.08 | 93.14 |
