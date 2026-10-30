"""fast_engine/simulation.py — Vectorized Walk-Forward Rolling Simulator.

Orchestrates sequential rolling walk-forward glucose forecasts across
all expert models with minimal per-step overhead.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from .models import (
    FastNODE,
    FastNHITS,
    FastStatsForecaster,
    FastXGBoost,
)


class FastRollingSimulation:
    """High-throughput online walk-forward simulator."""
    
    def __init__(
        self,
        horizon: int = 6,
        context_len: int = 120,
        start_step: int = 20,
        end_step: Optional[int] = None,
        include_nhits: bool = True,
        include_node: bool = True,
        include_arima: bool = True,
        include_ets: bool = True,
        include_xgboost: bool = True,
    ):
        self.horizon = horizon
        self.context_len = context_len
        self.start_step = start_step
        self.end_step = end_step
        
        self.include_nhits = include_nhits
        self.include_node = include_node
        self.include_arima = include_arima
        self.include_ets = include_ets
        self.include_xgboost = include_xgboost
        
    def run_patient(
        self,
        X: np.ndarray,
        verbose: bool = False,
        log_interval: int = 100,
    ) -> Dict[str, Any]:
        """Run sequential online forecasting for a single participant.
        
        Args:
            X: 1D or 2D array of CGM values of length T.
            verbose: If True, prints progress updates.
            log_interval: Interval for logging progress.
            
        Returns:
            Dictionary containing:
                - 'forecasts': Dict of {model_name: np.ndarray of shape (T,)}
                - 'timings': Dict of {model_name: total_wall_clock_seconds}
                - 'per_step_latency_ms': Dict of {model_name: mean_ms_per_step}
                - 'total_time_s': Total simulation wall-clock time in seconds
                - 'steps_completed': Number of evaluated steps
        """
        X_arr = np.asarray(X, dtype=float).flatten()
        T = len(X_arr)
        h = self.horizon
        start = self.start_step
        end = self.end_step if self.end_step is not None else (T - h)
        end = min(end, T - h)
        
        # Instantiate fast models
        models = {}
        if self.include_node:
            models["NODE"] = FastNODE(horizon=h)
        if self.include_ets:
            models["HoltWinters"] = FastStatsForecaster(model_type="ets", horizon=h)
        if self.include_arima:
            models["ARIMA"] = FastStatsForecaster(model_type="arima", horizon=h)
        if self.include_xgboost:
            models["XGBoost"] = FastXGBoost(horizon=h)
        if self.include_nhits:
            nhits_input = self.context_len if (self.context_len is not None and self.context_len > 0) else 120
            models["NHITS"] = FastNHITS(horizon=h, input_size=nhits_input)
            
        forecasts = {name: np.zeros(T, dtype=float) for name in models.keys()}
        model_timings = {name: 0.0 for name in models.keys()}
        
        total_steps = end - start
        t_sim_start = time.perf_counter()
        
        for step_idx, t in enumerate(range(start, end)):
            if self.context_len == -1 or self.context_len > t:
                t_min = 0
            else:
                t_min = t - self.context_len
                
            x_context = X_arr[t_min:t]
            
            for name, model in models.items():
                t0 = time.perf_counter()
                pred = model.step_and_forecast(x_context)
                dt = time.perf_counter() - t0
                
                forecasts[name][t + h] = pred
                model_timings[name] += dt
                
            if verbose and (step_idx + 1) % log_interval == 0:
                elapsed = time.perf_counter() - t_sim_start
                rate = (step_idx + 1) / max(elapsed, 1e-4)
                print(f"Step {step_idx + 1}/{total_steps} (t={t}) | {rate:.1f} steps/s | Elapsed: {elapsed:.2f}s")
                
        total_sim_time = time.perf_counter() - t_sim_start
        per_step_latency = {
            name: (total_s / max(total_steps, 1)) * 1000.0
            for name, total_s in model_timings.items()
        }
        
        return {
            "forecasts": forecasts,
            "timings": model_timings,
            "per_step_latency_ms": per_step_latency,
            "total_time_s": total_sim_time,
            "steps_completed": total_steps,
        }


def _worker_run_patient(args_tuple):
    """Worker task for running a single patient in a multi-process pool."""
    csv_path, horizon, context_len, start_step, end_step, output_dir, skip_existing = args_tuple
    
    # Enforce 1 thread per worker process to prevent thread oversubscription on shared HPC nodes
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    import torch
    torch.set_num_threads(1)
    
    # Extract patient ID
    base = os.path.basename(csv_path)
    id_num = base[: base.find(".csv")]
    if "-" in id_num:
        id_num = id_num[id_num.find("-") + 1 : id_num.find("-") + 4]
        
    fc_path = os.path.join(output_dir, "forecasts", f"{id_num}_forecasts.csv") if output_dir else None
    
    # Check if already completed with new fast engine (on or after Oct 15, 2026 18:47 UTC)
    FAST_ENGINE_CUTOFF = 1792090000.0
    if skip_existing and fc_path and os.path.exists(fc_path) and os.path.getsize(fc_path) > 500:
        if os.path.getmtime(fc_path) >= FAST_ENGINE_CUTOFF:
            return id_num, 0.0
        
    import ExpMethods.data as data
    
    raw_df = pd.read_csv(csv_path)
    X = data.transform_minute_data(raw_df).reshape(-1, 1)
    sim = FastRollingSimulation(
        horizon=horizon,
        context_len=context_len,
        start_step=start_step,
        end_step=end_step,
    )
    result = sim.run_patient(X, verbose=False)
        
    if output_dir:
        fc_dir = os.path.join(output_dir, "forecasts")
        os.makedirs(fc_dir, exist_ok=True)
        fc_df = pd.DataFrame(result["forecasts"])
        fc_df.to_csv(fc_path, index=False)
        
    return id_num, result["total_time_s"]


def run_cohort(
    csv_paths: List[str],
    horizon: int = 6,
    context_len: int = 120,
    start_step: int = 20,
    end_step: Optional[int] = None,
    n_workers: int = 4,
    output_dir: Optional[str] = None,
    skip_existing: bool = True,
) -> Dict[str, float]:
    """Execute high-throughput rolling simulation across a cohort of participants.
    
    Args:
        csv_paths: List of file paths to participant CSVs.
        horizon: Forecast horizon in 5-min steps.
        context_len: Historical window length in steps.
        start_step: Initial evaluation step.
        end_step: Final evaluation step.
        n_workers: Number of concurrent worker processes.
        output_dir: Directory where forecast CSVs will be saved.
        skip_existing: If True, skips participants whose forecast CSV already exists.
        
    Returns:
        Dictionary mapping patient_id to simulation wall-clock time in seconds.
    """
    from concurrent.futures import ProcessPoolExecutor
    
    tasks = [
        (path, horizon, context_len, start_step, end_step, output_dir, skip_existing)
        for path in csv_paths
    ]
    
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for id_num, t_sec in executor.map(_worker_run_patient, tasks):
            results[id_num] = t_sec
            
    return results
