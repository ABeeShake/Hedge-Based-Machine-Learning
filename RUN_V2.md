# Engine v2.1: re-running the study

Engine v2 (`fast_engine/`) changes how the experts are trained and adds two experts:

- **Burn-in.** Each participant's first 288 readings (24 h) train the experts; forecasts start after that.
- **Daily refits.** Every expert is refitted every 288 steps (24 h) on the readings observed so far
  (`--retrain_interval` to change). Between refits ARIMA/ETS update their Kalman state each step;
  the other experts only condition on new data.
- **Trained models.** N-HITS and NODE are now trained (Adam; warm-started at each refit). XGBoost uses lagged
  levels, multi-scale deltas and rolling statistics and predicts the change over the horizon.
- **New experts.** `Persistence` (y(t)) and `RunningMean` (mean of y(0..t)) are part of the pool.
- **Timing fix.** v1 stored the forecast made from y(0..t-1) at row t+h, so every expert was really
  forecasting h+1 steps ahead while persistence used y(t). v2 forecasts y(t+h) from y(0..t).
- **Clipping.** All expert forecasts are clipped to the sensor range, 40-400 mg/dL.
- **Scoring window.** Every script scores rows `GlobalValues.eval_start` (= 288 + 60 = 348) to `T-10`, the same
  time points for all horizons. All protocol constants live in `ExpMethods/globals.py`.
- **Context window.** `context` is now the look-back an expert conditions on (ARIMA/ETS are fitted to it, N-HITS
  uses it as its input length). NODE, N-HITS and XGBoost are trained on all readings up to the refit time.

- **5-minute grid (v2.1).** Weinstock has missing readings (27% of grid slots; median 10 segments per
  participant), so in v1/v2.0 a "step" was not always 5 minutes and horizons were not clock-time horizons.
  Every participant is now put on a 5-minute grid (`ExpMethods/timegrid.py`):
  1. each reading advances round(interval / 5 min) slots from the previous one;
  2. gaps of up to 30 min between readings (`max_gap_min`) are linearly filled, **as model inputs only**;
  3. a longer gap starts a new segment. It is bridged by a straight line for model inputs, but no
     training target ever lies inside it;
  4. forecasts are issued only from slots with a real reading, after the burn-in (the first 288 real
     readings), and at least 1 h (`min_segment_history` = 12 slots) into a segment;
  5. only rows with a real reading at t+h and an issued forecast are scored (`ExpMethods/evaluation.scored_mask`);
     the aggregators get no feedback on other rows.
  Because filling only uses readings up to the end of each gap and origins are always real readings,
  no expert ever sees a value after its origin (`tests/test_engine.py::test_grid_no_lookahead`).
  Forecast files have one row per grid slot; rows without an issued forecast are 0.
  `analyze_segments.py` (runs locally, raw data only) reports gaps, segments and coverage and writes
  `overleaf/tables/grid_summary.csv` -> `\Grid...` macros. CGMacros is already on a gap-free 5-minute
  grid, so nothing changes for it.
- **Why bridged windows for N-HITS.** Requiring the whole input window to lie in one segment would keep only
  57-77% of Weinstock forecasts (input 24 h / 10 h); bridging long gaps and waiting 1 h after each keeps 97%,
  and fewer than 3% of kept forecasts have an input window that is more than half bridged.

Each forecast file gets a `<id>_forecasts.meta.json` with the engine version, settings and per-expert
timings. A forecast counts as finished only if its metadata matches the current engine, so old (v1)
files are never mistaken for v2 results.

## Steps on the VM (SLURM)

To submit the entire Engine v2.1 simulation grid, secondary analyses, and post-processing pipeline with automatic SLURM `afterok` dependencies:

```bash
git pull
bash slurm/submit_v2_pipeline.sh --chunk_size 20 --throttle 4
```

Or submit each stage individually as SLURM jobs:

```bash
git pull
python3 tests/test_engine.py            # must end with the full-pool smoke test printing RMSE and fit times
python3 analyze_segments.py             # grid summary (also fine to run locally)
bash archive_v1_outputs.sh              # moves Outputs/<dataset>/h-* to Outputs_v1/
bash slurm/submit_grid.sh 0-23 20 4     # main simulations (all 24 conditions, chunk size 20)
python3 run_simulation_chunk.py --list_conditions     # wait until every condition is DONE

# secondary analyses (can run alongside the main grid)
sbatch --array=0-5 slurm/secondary_population.slurm              # within-cohort population models
SCHEME=cross sbatch --array=0-5 slurm/secondary_population.slurm # cross-cohort population models
sbatch --array=0-5 slurm/secondary_cadence.slurm                 # refit every 6 / 24 / 48 h, 20 participants per cohort
sbatch --array=0-5 slurm/secondary_seeds.slurm                   # 3 random seeds (0, 1, 2), 20 participants per cohort

sbatch slurm/run_on_vm.slurm            # post-processing, tables, clinical events, regimes, timings, secondary evaluations, macros
```

Before launching the full grid, time one participant to size the job:

```bash
python3 run_simulation_chunk.py --condition_idx 23 --patient_ids 001 --n_workers 1
cat Outputs/weinstock/h-5hr/context-fullhr/forecasts/001_forecasts.meta.json   # fit_s_mean per expert
```

Training cost per expert is set by `first_fit_steps` / `fit_steps` in `fast_engine/models.py`.

## Secondary analyses

| Script | Question | Output tables |
|---|---|---|
| `secondary_population_models.py` | How does personal HBML compare with N-HITS / XGBoost trained on other participants, and does adding them as experts help? Participant-level 5-fold (or cross-cohort) training; no participant in both train and test. | `population_comparison.csv`, `table_population_comparison.tex` |
| `secondary_retrain_cadence.py` | Do results depend on refitting every 24 h? Same pipeline with refits every 6, 24 and 48 h on a fixed random subset (20 per cohort, 30-min and 5-h horizons, full history). | `retrain_cadence.csv`, `table_retrain_cadence.tex` |
| `secondary_seed_stability.py` | How much run-to-run variation arises from stochastic mini-batch training (`NODE`, `NHITS`, `XGBoost`)? Evaluates 3 random seeds (`0, 1, 2`) on 20 participants per cohort. | `seed_stability.csv`, `table_seed_stability.tex` |
| `evaluate_clinical_and_weights.py` | Event-level hypoglycaemia ($<70$~mg/dL) detection sensitivity, warning lead time, false alerts/day, Parkes error grid, weight concentration, Follow-the-Leader (`FTL`), Orabona--P\'al `SFH-Sqrt`, and regime summary tables. | `hypoglycemia_events.csv`, `table_hypoglycemia_events.tex`, `weight_diagnostics.csv`, `regime_summary_table.csv`, `table_regime_summary.tex` |

All write macros (`\Pop...`, `\Cadence...`, `\Seed...`, `\Hypo...`, `\Weight...`, `\Regime...`) through `generate_results_macros.py`.
