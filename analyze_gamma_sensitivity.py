#!/usr/bin/env python3
"""analyze_gamma_sensitivity.py — Sensitivity analysis for decay forgetting parameter gamma.

Evaluates HBML-SFHDF performance across the full grid of decay parameter candidates
gamma in [0.0, 1.0] across horizons and context windows.
Demonstrates stability around the default hyperparameter gamma = 0.2.

Outputs:
  - overleaf/tables/gamma_sensitivity.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import ExpMethods.simulate as sim
from calculate_baselines import reconstruct_ground_truth

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(BASE_DIR, "overleaf", "tables")
OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs")

GAMMA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def run_gamma_sensitivity():
    print("Running Gamma Sensitivity Analysis across gamma in [0.0, 1.0]...")
    
    # We evaluate representative benchmarks:
    benchmarks = [
        ("cgmacros", "h-2hr", "context-fullhr", 24, 2.0),
        ("cgmacros", "h-5hr", "context-fullhr", 60, 5.0),
        ("weinstock", "h-5hr", "context-12hr", 60, 5.0),
        ("weinstock", "h-2hr", "context-12hr", 24, 2.0),
    ]
    
    records = []
    
    for ds, h_dir, ctx_dir, h_steps, h_val in benchmarks:
        dir_path = os.path.join(OUTPUTS_DIR, ds, h_dir, ctx_dir)
        fc_dir = os.path.join(dir_path, "forecasts")
        ls_dir = os.path.join(dir_path, "losses")
        
        if not os.path.exists(fc_dir):
            continue
            
        fc_files = sorted(glob.glob(os.path.join(fc_dir, "*_forecasts.csv")))[:20]
        if not fc_files:
            continue
            
        # Accumulate patient data
        patients = []
        for f_path in fc_files:
            pid = os.path.basename(f_path)[:3]
            l_path = os.path.join(ls_dir, f"{pid}_losses.csv")
            if not os.path.exists(l_path):
                continue
                
            fc_df = pd.read_csv(f_path)
            ls_df = pd.read_csv(l_path)
            if "HoltWinters" in fc_df.columns:
                fc_df["ETS"] = fc_df["HoltWinters"]
            if "HoltWinters" in ls_df.columns:
                ls_df["ETS"] = ls_df["HoltWinters"]
                
            res = reconstruct_ground_truth(fc_df, ls_df)
            if res is None:
                continue
            y_rec, mask = res
            
            models = [m for m in ["NODE", "ARIMA", "ETS", "XGBoost", "NHITS"] if m in fc_df.columns]
            f_dict = {m: fc_df[m].values for m in models}
            l_dict = {m: ls_df[m].values for m in models if m in ls_df.columns}
            
            patients.append((pid, f_dict, l_dict, y_rec, mask))
            
        if not patients:
            continue
            
        print(f"Evaluating {ds} {h_dir} {ctx_dir} with {len(patients)} patients...")
        
        for gamma in GAMMA_GRID:
            patient_rmses = []
            for pid, f_dict, l_dict, y_rec, mask in patients:
                # Run SFH-DF
                exp_f, exp_l = sim.scale_free_hedge_df_forecast(
                    f_dict, l_dict, gamma=gamma,
                    save_weights=False, forecast_type="mean",
                    targets=y_rec, horizon=h_steps, start=100, end=len(y_rec) - 10
                )
                
                val_idx = np.arange(100, len(y_rec) - 10)
                val_idx = val_idx[mask[val_idx]]
                if len(val_idx) < 30:
                    continue
                    
                pred = exp_f[val_idx]
                target = y_rec[val_idx]
                rmse = np.sqrt(np.mean((pred - target) ** 2))
                patient_rmses.append(rmse)
                
            mean_rmse = np.mean(patient_rmses) if patient_rmses else np.nan
            records.append({
                "dataset": ds,
                "horizon": h_val,
                "context": ctx_dir.replace("context-", "").replace("hr", ""),
                "gamma": gamma,
                "mean_rmse": mean_rmse,
                "n_patients": len(patient_rmses)
            })
            
    df_res = pd.DataFrame(records)
    out_path = os.path.join(TABLES_DIR, "gamma_sensitivity.csv")
    df_res.to_csv(out_path, index=False)
    print(f"Saved gamma sensitivity to {out_path}")
    
    # Pivot for clean viewing
    if not df_res.empty:
        pivot = df_res.pivot_table(index="gamma", columns=["dataset", "horizon", "context"], values="mean_rmse")
        print("\nGamma Sensitivity Pivot (Mean RMSE):")
        print(pivot.round(2))

if __name__ == "__main__":
    run_gamma_sensitivity()
