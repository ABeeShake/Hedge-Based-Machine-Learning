# HBML: Hedge-Based Machine Learning for Personalized CGM Glucose Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the full implementation for the paper:

> **[Paper Title]**
> [Author Names]
> *npj Digital Medicine*, [Year].
> [DOI / Link]

HBML is an adaptive online ensemble framework for real-time, personalized
continuous glucose monitoring (CGM) forecasting. It combines six heterogeneous
expert forecasting models under the AdaHedge algorithm with Fixed-Share mixing,
enabling the ensemble to track the best-performing expert as a patient's glucose
dynamics evolve over time - without any population-level pretraining or
patient-specific hyperparameter tuning.

---

## Repository Structure

```
HedgeProject/
├── run_simulation.py          # Main entry point: per-patient online simulation
├── process_results.py         # Post-hoc HBML aggregation, regrets, CEG plots
├── generate_figures.py        # Paper figures (RMSE tables, regret plots, CEG)
├── generate_results_macros.py # LaTeX \newcommand values from result CSVs
├── plot_and_format_volatility.py
├── plot_regime_analysis.py    # Physiological regime stacked bar charts
├── plot_subgroups.py          # Per-subgroup RMSE / CEG bar charts
├── plot_weights.py            # Expert weight evolution stackplot
├── analyze_regimes.py         # Regime classification and summary statistics
├── compute_regimes.py         # Assigns regime labels to each timestep
├── compute_volatility.py      # Patient-level CGM variability metrics
├── parse_timings.py           # Parse timing logs from simulation output
├── summarize_timings.py       # Summarize and format timing statistics
├── create_hbml_flowchart.py   # Generate the HBML pipeline diagram
├── reproduce.sh               # One-command end-to-end reproduction
├── cgmacros-horizons.sh       # Simulation runner: CGMacros dataset
├── run_all_process_results.sh # Post-processing runner: all output directories
├── requirements.txt           # Direct Python dependencies
│
├── ExpMethods/                # Core library
│   ├── simulate.py            # HBML algorithm, Hedge/FS mixing, eta schedules
│   ├── models.py              # Expert model wrappers (NODE, NHITS, ETS, ARIMA, XGBoost)
│   ├── data.py                # CGM data loading and PyTorch dataset classes
│   ├── utils.py               # I/O helpers, array utilities, result aggregation
│   ├── timing.py              # Wall-clock profiling (Timer, TimingRegistry)
│   ├── visualizations.py      # Plotting utilities (Clarke Error Grid, weights, etc.)
│   └── globals.py             # Shared constants, model defaults, color palettes
│
├── cgmacros/                  # Per-patient CGMacros CSVs (see Data Setup below)
├── weinstock/                 # Per-patient Weinstock 2016 CSVs (see Data Setup below)
│
├── overleaf/                  # LaTeX paper source
│   ├── tables/                # Result CSV files imported by LaTeX
│   └── sections/              # Paper section .tex files
│
├── rmse.csv                   # Aggregated RMSE results (all patients, all settings)
├── maxae.csv                  # Aggregated Max-AE results
│
└── archive/                   # Development/exploratory scripts (not part of pipeline)
```

---

## Data Setup

### CGMacros (2025)

CGMacros is publicly available on PhysioNet:

```
https://physionet.org/content/cgmacros/
```

Download the dataset, extract the per-patient CSV files, and place them in
`./cgmacros/`. Each file should be named `<patient_id>.csv` and contain at
minimum a `Timestamp` column and a `Libre.GL` column (FreeStyle Libre glucose
readings in mg/dL).

### Weinstock 2016

Weinstock et al. (2016) is available through the study data repository.
Place per-patient CSV files in `./weinstock/`. Each file should contain a
`Timestamp` column and a `Dexcom.GL` column.

**Preprocessing**: Both datasets were downsampled to 5-minute intervals (every
5th reading for CGMacros), glucose readings were clipped to [40, 400] mg/dL,
and all covariates other than the glucose time series were excluded. These steps
are applied automatically by `ExpMethods/data.py`.

---

## Environment Setup

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

**GPU note**: `torch`, `lightning`, `neuralforecast`, and `torchdyn` all
support CUDA automatically when a compatible GPU and CUDA toolkit are available.
On a CPU-only machine everything still runs; Neural ODE and N-HITS will be
substantially slower.

---

## Reproducing Results

### Option A — Full reproduction (requires data files)

```bash
# Reproduce everything end-to-end
bash reproduce.sh

# Or skip the multi-hour simulation if you have pre-computed outputs
bash reproduce.sh --skip-sim

# Or skip both simulation and post-processing (figures only)
bash reproduce.sh --skip-sim --skip-process
```

### Option B — Step by step

**Step 1: Run the online simulation** (generates per-patient forecast CSVs)

```bash
# CGMacros (runs all horizons × context window combinations)
bash cgmacros-horizons.sh

# Weinstock 2016
bash weinstock-horizons.sh
```

Forecasts are written to `Outputs/<dataset>/h-<horizon>hr/context-<context>hr/forecasts/`.

**Step 2: Post-process** (compute HBML weights, regrets, Clarke Error Grid plots)

```bash
bash run_all_process_results.sh
```

**Step 3: Generate figures**

```bash
python3 generate_figures.py          # RMSE tables, regret plots, Clarke Error Grids
python3 plot_regime_analysis.py      # Regime stacked bar charts
python3 plot_subgroups.py            # Subgroup RMSE / CEG plots
python3 plot_weights.py              # Expert weight evolution
python3 plot_and_format_volatility.py
```

**Step 4: Generate LaTeX macros** (writes `\newcommand` values to `overleaf/macros/`)

```bash
python3 generate_results_macros.py
```

---

## Key CLI Arguments for `run_simulation.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | *(required)* | Directory with per-patient CSVs |
| `--output_dir` | *(required)* | Directory for forecast output |
| `--model_dir` | `./` | Load pre-trained checkpoints from here (warm-start) |
| `--horizon` | `6` | Forecast horizon in 5-min steps (6=30 min, 24=2 hr, 60=5 hr) |
| `--context_len` | `120` | Rolling context window in steps (288=24 hr, −1=full history) |
| `--epochs` | `50` | Max training epochs per periodic retrain |
| `--n_workers` | `511` | DataLoader workers (use `0` on macOS or CPU-only machines) |
| `--log_n_steps` | `500` | Checkpoint interval (enables crash recovery) |
| `--debug` | `False` | Truncate to 100 steps per patient for quick testing |

---

## Expert Models

| Expert | Library | Type | Retrain interval |
|--------|---------|------|-----------------|
| Neural ODE (NODE) | `torchdyn` + `lightning` | Deep learning | Every 20 steps |
| N-HITS | `neuralforecast` | Deep learning | Every 50 steps |
| Auto-ARIMA | `statsforecast` | Statistical | Every 10 steps |
| Auto-ETS | `statsforecast` | Statistical | Every 10 steps |
| XGBoost | `xgboost` | Gradient-boosted tree | Every 10 steps |

The HBML ensemble weight update uses the AdaHedge learning rate schedule with
Fixed-Share mixing (α_t = 1/t²). See `ExpMethods/simulate.py` for full
implementation details and `ExpMethods/globals.py` for all default hyperparameters.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{[citekey],
  title   = {[Paper Title]},
  author  = {[Authors]},
  journal = {npj Digital Medicine},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
