# Fixed-Share comparison

RMSE is per participant, mg/dL. Ratios are variant RMSE / current HBML-SFHDF RMSE (below 1 = better than the current algorithm).

**Left out of this report** because the forecasts in this folder do not reproduce the HBML means in `overleaf/tables/rmse_pivot_mean.csv` (stale or partial copy; run these where the outputs are complete): cgmacros h=2 c=12, cgmacros h=2 c=24, cgmacros h=2 c=6, cgmacros h=2 c=full, cgmacros h=5 c=12, cgmacros h=5 c=6.

## Average over conditions

| dataset | variant | conditions | mean % change in RMSE vs current | conditions where it beats current |
|---|---|---|---|---|
| cgmacros | alg1_fs_c0.001 | 6 | +0.36 | 3 |
| cgmacros | alg1_fs_c0.01 | 6 | +0.37 | 3 |
| cgmacros | alg1_fs_c0.05 | 6 | +0.27 | 4 |
| cgmacros | alg1_fs_d | 6 | +0.34 | 4 |
| cgmacros | alg1_fs_fd | 6 | +0.20 | 2 |
| cgmacros | alg1_noshare | 6 | +1.56 | 1 |
| cgmacros | cum_decaymax | 6 | -0.06 | 5 |
| cgmacros | current | 6 | +0.00 | 0 |
| cgmacros | uniform | 6 | +2.94 | 0 |
| weinstock | alg1_fs_c0.001 | 12 | +2.91 | 4 |
| weinstock | alg1_fs_c0.01 | 12 | +2.77 | 4 |
| weinstock | alg1_fs_c0.05 | 12 | +1.92 | 4 |
| weinstock | alg1_fs_d | 12 | +2.83 | 4 |
| weinstock | alg1_fs_fd | 12 | +2.04 | 4 |
| weinstock | alg1_noshare | 12 | +0.08 | 4 |
| weinstock | cum_decaymax | 12 | -0.08 | 12 |
| weinstock | current | 12 | +0.00 | 0 |
| weinstock | uniform | 12 | +4.96 | 3 |

## Held-out choice (odd / even participant IDs)

| dataset | chosen on | chosen variant | ratio on the selection half | ratio on the held-out half |
|---|---|---|---|---|
| cgmacros | odd | cum_decaymax | 0.9994 | 0.9991 |
| cgmacros | even | alg1_fs_fd | 0.9974 | 1.0041 |
| weinstock | odd | cum_decaymax | 0.9993 | 0.9991 |
| weinstock | even | cum_decaymax | 0.9991 | 0.9993 |

## Per condition (mean RMSE)

| dataset | horizon | context | n | best expert | current | alg1_fs_fd | alg1_noshare | alg1_fs_d | alg1_fs_c0.001 | alg1_fs_c0.01 | alg1_fs_c0.05 | cum_decaymax | uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cgmacros | half | 6 | 44 | 22.59 | 21.93 | 21.93 | 22.14 | 21.67 | 21.65 | 21.58 | 21.60 | 21.87 | 22.54 |
| cgmacros | half | 12 | 44 | 22.04 | 21.30 | 21.37 | 21.74 | 21.23 | 21.24 | 21.17 | 21.15 | 21.28 | 21.70 |
| cgmacros | half | 24 | 44 | 21.34 | 20.75 | 20.85 | 21.27 | 20.75 | 20.77 | 20.84 | 20.81 | 20.74 | 21.01 |
| cgmacros | half | full | 44 | 20.74 | 20.44 | 20.55 | 21.24 | 20.53 | 20.55 | 20.47 | 20.42 | 20.43 | 20.49 |
| cgmacros | 5 | 24 | 44 | 38.08 | 37.87 | 36.66 | 37.12 | 37.28 | 37.32 | 37.38 | 37.21 | 37.84 | 38.36 |
| cgmacros | 5 | full | 44 | 32.72 | 34.20 | 35.24 | 34.85 | 35.80 | 35.75 | 35.96 | 36.01 | 34.25 | 37.66 |
| weinstock | half | 6 | 200 | 35.05 | 34.43 | 34.19 | 34.58 | 34.08 | 34.05 | 34.08 | 34.26 | 34.34 | 38.60 |
| weinstock | half | 12 | 200 | 34.61 | 33.77 | 33.62 | 33.83 | 33.58 | 33.56 | 33.57 | 33.75 | 33.72 | 37.22 |
| weinstock | half | 24 | 200 | 34.45 | 33.38 | 33.23 | 33.39 | 33.28 | 33.30 | 33.33 | 33.24 | 33.34 | 36.71 |
| weinstock | half | full | 200 | 33.95 | 32.89 | 32.80 | 33.02 | 32.80 | 32.79 | 32.85 | 32.95 | 32.86 | 35.78 |
| weinstock | 2 | 6 | 200 | 72.10 | 73.42 | 77.98 | 74.18 | 80.09 | 80.24 | 80.23 | 78.94 | 73.38 | 76.34 |
| weinstock | 2 | 12 | 200 | 72.43 | 73.18 | 76.02 | 73.19 | 76.57 | 76.61 | 76.32 | 75.27 | 73.14 | 72.85 |
| weinstock | 2 | 24 | 200 | 69.84 | 71.40 | 73.04 | 71.89 | 73.48 | 73.57 | 73.34 | 72.38 | 71.36 | 71.14 |
| weinstock | 2 | full | 200 | 67.50 | 68.88 | 70.32 | 69.45 | 70.74 | 70.87 | 70.86 | 70.24 | 68.85 | 69.62 |
| weinstock | 5 | 6 | 200 | 99.59 | 104.18 | 111.51 | 104.02 | 113.71 | 113.57 | 113.31 | 111.37 | 104.13 | 114.37 |
| weinstock | 5 | 12 | 200 | 99.24 | 104.09 | 107.24 | 103.76 | 107.39 | 107.53 | 106.49 | 104.43 | 104.01 | 103.60 |
| weinstock | 5 | 24 | 200 | 91.40 | 96.06 | 96.39 | 94.93 | 97.38 | 97.52 | 97.21 | 95.97 | 96.00 | 96.46 |
| weinstock | 5 | full | 200 | 86.73 | 89.19 | 90.46 | 88.29 | 91.92 | 92.24 | 92.46 | 91.86 | 89.15 | 93.14 |
