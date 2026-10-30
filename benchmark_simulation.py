"""benchmark_simulation.py — Empirical Benchmark: Legacy vs Fast Engine.

Directly compares the original simulation implementation against the redesigned
fast engine on identical participant CGM sequences. Measures:
- Wall-clock execution time
- Latency per 5-minute timestep across all expert models
- Output fidelity and concordance (correlation, RMSD, MAD)
- Downstream Hedge ensemble performance
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import lightning as L

# Legacy imports
import ExpMethods.models as m
import ExpMethods.data as data
import ExpMethods.simulate as sim
from statsforecast.models import AutoETS, AutoARIMA
from xgboost import XGBRegressor
from neuralforecast.models import NHITS

# Fast Engine imports
from fast_engine import (
    FastRollingSimulation,
    FastOnlineHedge,
)


def run_benchmark(
    cgm_path: str = "cgmacros/001.csv",
    n_steps: int = 30,
    horizon: int = 6,
    context_len: int = 120,
    start_step: int = 20,
):
    print("=" * 80)
    print("     HBML ONLINE SIMULATION BENCHMARK: LEGACY vs REDESIGNED FAST ENGINE     ")
    print("=" * 80)
    print(f"Data file:      {cgm_path}")
    print(f"Horizon:        {horizon} steps ({horizon * 5} minutes)")
    print(f"Context length: {context_len} steps ({context_len * 5 / 60:.1f} hours)")
    print(f"Start step:     {start_step}")
    print(f"Number of steps:{n_steps} timesteps (t={start_step} to {start_step + n_steps})")
    print("-" * 80)

    # 1. Load and prepare data
    raw_df = pd.read_csv(cgm_path)
    X = data.transform_minute_data(raw_df).reshape(-1, 1)
    end_step = start_step + n_steps

    # ---------------------------------------------------------
    # PART 1: LEGACY SIMULATION
    # ---------------------------------------------------------
    print("\n>>> [1/2] Running Legacy Simulation Pipeline...")
    torch.set_float32_matmul_precision("high")
    trainer_params = sim.DefaultSimulationParams.trainer_params(max_epochs=1)
    trainer = L.Trainer(**trainer_params)

    node_params = m.DefaultModelParams.node_params(horizon=horizon)
    xgboost_params = m.DefaultModelParams.xgboost_params()
    nhits_params = m.DefaultModelParams.nf_params(
        omit=["train_n_steps", "hidden_size"],
        input_size=context_len if (context_len is not None and context_len > 0) else 120,
        h=horizon,
    )

    legacy_models = {
        "NODE": m.NODEForecaster(m.NODE(**node_params)),
        "HoltWinters": m.StatsForecaster(AutoETS, horizon),
        "ARIMA": m.StatsForecaster(AutoARIMA, horizon),
        "XGBoost": m.XGBoostForecaster(XGBRegressor(**xgboost_params), horizon),
        "NHITS": m.NNForecaster(NHITS(**nhits_params)),
    }

    legacy_forecasts = {k: np.zeros(len(X)) for k in legacy_models.keys()}
    legacy_timings = {k: 0.0 for k in legacy_models.keys()}

    t_legacy_total_start = time.perf_counter()
    for step_idx, t in enumerate(range(start_step, end_step)):
        t_min = max(0, t - context_len) if context_len != -1 else 0
        x_train = X[:t]
        x_test = X[t:]

        data_params = dict(
            x_train=x_train[t_min:],
            x_test=x_test,
            batch_size=10,
            max_horizon=horizon,
            h_first=True,
            num_workers=0,
        )
        dm = data.MinuteDataLightningDataModule(**data_params)

        for name, model in legacy_models.items():
            t0 = time.perf_counter()
            pred = sim.sim_step(model, dm, trainer, x_train, t_min)
            dt = time.perf_counter() - t0
            legacy_forecasts[name][t + horizon] = pred
            legacy_timings[name] += dt

        print(
            f"   Legacy step {step_idx + 1:2d}/{n_steps} (t={t}) done in "
            f"{sum(legacy_timings.values()):.2f}s accumulated"
        )

    t_legacy_total = time.perf_counter() - t_legacy_total_start

    # ---------------------------------------------------------
    # PART 2: REDESIGNED FAST ENGINE
    # ---------------------------------------------------------
    print("\n>>> [2/2] Running Redesigned Fast Engine Pipeline...")
    fast_sim = FastRollingSimulation(
        horizon=horizon,
        context_len=context_len,
        start_step=start_step,
        end_step=end_step,
    )

    t_fast_total_start = time.perf_counter()
    fast_result = fast_sim.run_patient(X, verbose=False)
    t_fast_total = time.perf_counter() - t_fast_total_start

    fast_forecasts = fast_result["forecasts"]
    fast_timings = fast_result["timings"]

    # ---------------------------------------------------------
    # PART 3: COMPARATIVE ANALYSIS & FIDELITY
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    EMPIRICAL BENCHMARK RESULTS                     ")
    print("=" * 80)

    # Per-model metrics
    table_rows = []
    eval_slice = slice(start_step + horizon, end_step + horizon)

    for name in ["NODE", "HoltWinters", "ARIMA", "XGBoost", "NHITS"]:
        leg_time_s = legacy_timings[name]
        fast_time_s = fast_timings[name]
        leg_ms_step = (leg_time_s / n_steps) * 1000.0
        fast_ms_step = (fast_time_s / n_steps) * 1000.0
        speedup = leg_time_s / max(fast_time_s, 1e-6)

        # Statistical concordance
        f_leg = legacy_forecasts[name][eval_slice]
        f_fast = fast_forecasts[name][eval_slice]

        rmsd = np.sqrt(np.mean((f_leg - f_fast) ** 2))
        mad = np.mean(np.abs(f_leg - f_fast))
        # Correlation
        if np.std(f_leg) > 1e-6 and np.std(f_fast) > 1e-6:
            corr = np.corrcoef(f_leg, f_fast)[0, 1]
        else:
            corr = 1.0

        table_rows.append({
            "Model": name,
            "Legacy (ms/step)": f"{leg_ms_step:.1f}",
            "Fast (ms/step)": f"{fast_ms_step:.2f}",
            "Speedup Multiple": f"{speedup:.1f}x",
            "Correlation (r)": f"{corr:.4f}",
            "RMSD (mg/dL)": f"{rmsd:.2f}",
            "MAD (mg/dL)": f"{mad:.2f}",
        })

    results_df = pd.DataFrame(table_rows)
    print(results_df.to_string(index=False))

    overall_speedup = t_legacy_total / max(t_fast_total, 1e-6)
    print("\n" + "-" * 80)
    print(f"Total Legacy Runtime:     {t_legacy_total:.2f} s ({t_legacy_total/n_steps*1000:.1f} ms / timestep)")
    print(f"Total Fast Engine Runtime:{t_fast_total:.2f} s ({t_fast_total/n_steps*1000:.2f} ms / timestep)")
    print(f"OVERALL WALL-CLOCK SPEEDUP: {overall_speedup:.1f}x FASTER")
    print("-" * 80)

    # Full dataset extrapolation
    full_steps_per_patient = 2800
    n_patients = 100
    total_paper_steps = full_steps_per_patient * n_patients

    legacy_est_hours = (t_legacy_total / n_steps * total_paper_steps) / 3600.0
    fast_est_minutes = (t_fast_total / n_steps * total_paper_steps) / 60.0

    print(f"\nProjected Paper Runtime for 100 Patients ({total_paper_steps:,} steps):")
    print(f"  • Legacy Architecture:    {legacy_est_hours:.1f} hours ({legacy_est_hours / 24:.1f} days)")
    print(f"  • Redesigned Fast Engine: {fast_est_minutes:.1f} minutes ({fast_est_minutes / 60:.2f} hours)")

    # ---------------------------------------------------------
    # PART 4: DOWNSTREAM HEDGE EVALUATION
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("              DOWNSTREAM ONLINE HEDGE AGGREGATION BENCHMARK              ")
    print("=" * 80)

    targets = np.asarray(X.cpu().numpy() if isinstance(X, torch.Tensor) else X, dtype=float).flatten()
    model_names = list(legacy_models.keys())
    F_leg = np.column_stack([legacy_forecasts[m] for m in model_names])
    F_fast = np.column_stack([fast_forecasts[m] for m in model_names])

    hedge = FastOnlineHedge(horizon=horizon, start_step=start_step, alpha=0.5, eta=1.0)

    t0_hedge = time.perf_counter()
    hedge_res_fast = hedge.evaluate(F_fast, targets, expert_names=model_names)
    t_hedge_s = time.perf_counter() - t0_hedge

    hedge_res_leg = hedge.evaluate(F_leg, targets, expert_names=model_names)

    # Compute MSE / RMSE of ensemble
    rmse_leg = np.sqrt(np.mean((hedge_res_leg["ensemble_forecasts"][eval_slice] - targets[eval_slice]) ** 2))
    rmse_fast = np.sqrt(np.mean((hedge_res_fast["ensemble_forecasts"][eval_slice] - targets[eval_slice]) ** 2))

    print(f"Fast Online Hedge Execution Time: {t_hedge_s * 1000:.3f} ms for {n_steps} steps")
    print(f"Legacy-derived Ensemble RMSE:     {rmse_leg:.2f} mg/dL")
    print(f"Fast-derived Ensemble RMSE:       {rmse_fast:.2f} mg/dL")
    print(f"RMSE delta:                       {abs(rmse_leg - rmse_fast):.2f} mg/dL")

    # Save summary table
    os.makedirs("scratch", exist_ok=True)
    results_df.to_csv("scratch/benchmark_speedup_results.csv", index=False)
    print("\nSaved detailed benchmark CSV to scratch/benchmark_speedup_results.csv")

    return results_df, overall_speedup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empirical Simulation Benchmark")
    parser.add_argument("--steps", type=int, default=25, help="Number of timesteps to simulate")
    parser.add_argument("--cgm", type=str, default="cgmacros/001.csv", help="CGM CSV path")
    args = parser.parse_args()

    run_benchmark(cgm_path=args.cgm, n_steps=args.steps)
