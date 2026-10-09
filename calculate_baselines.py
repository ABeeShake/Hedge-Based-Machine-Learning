#!/usr/bin/env python3
"""calculate_baselines.py — Evaluate Persistence and Uniform Ensemble Baselines.

Computes essential comparator baselines requested during peer review:
  1. Persistence Baseline: y_hat_{t+h} = y_t
  2. Uniform Ensemble Average: y_hat_{t+h} = (1/K) sum_{k=1}^K f_{k,t+h}
  3. Static Best Expert per cohort

Saves results to:
  - overleaf/tables/baselines_comparison.csv
  - overleaf/tables/table_baselines_comparison.tex
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(BASE_DIR, "overleaf", "tables")
OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs")

DATASETS = ["weinstock", "cgmacros"]
HORIZONS = {0.5: 6, 2.0: 24, 5.0: 60}  # horizon in 5-min steps: 30min=6, 2hr=24, 5hr=60
CONTEXTS = ["6", "12", "24", "full"]

HORIZON_MAP = {0.5: "0p5", 2.0: "2", 5.0: "5"}
HORIZON_DIR_MAP = {0.5: "h-halfhr", 2.0: "h-2hr", 5.0: "h-5hr"}
DATASET_LABEL = {"weinstock": "Weinstock", "cgmacros": "CGMacros"}
HORIZON_LABEL = {0.5: "0.5 hr", 2.0: "2 hr", 5.0: "5 hr"}

MODELS = ["NODE", "ARIMA", "ETS", "XGBoost", "NHITS"]


def reconstruct_ground_truth(fc_df, ls_df):
    """Reconstruct exact true target y from constituent expert forecasts and losses."""
    models_avail = [m for m in ["NODE", "ARIMA", "ETS", "XGBoost", "NHITS"] if m in fc_df.columns and m in ls_df.columns]
    if len(models_avail) < 2:
        return None
    
    # Try all pairs to find best numerical separation
    best_pair = None
    max_diff_spread = 0.0
    for i in range(len(models_avail)):
        for j in range(i + 1, len(models_avail)):
            m1, m2 = models_avail[i], models_avail[j]
            diff = np.abs(fc_df[m1].values - fc_df[m2].values)
            spread = np.mean(diff)
            if spread > max_diff_spread:
                max_diff_spread = spread
                best_pair = (m1, m2)
                
    if best_pair is None or max_diff_spread < 0.1:
        return None
        
    m1, m2 = best_pair
    f1 = fc_df[m1].values
    f2 = fc_df[m2].values
    l1 = ls_df[m1].values
    l2 = ls_df[m2].values
    
    diff = f1 - f2
    y_rec = np.zeros_like(f1)
    mask = (np.abs(diff) > 0.1) & (l1 >= 0)
    
    y_rec[mask] = (f1[mask] + f2[mask]) / 2.0 - (l1[mask] - l2[mask]) / (2.0 * diff[mask])
    # CGM values are integers or tenths
    y_rec[mask] = np.round(y_rec[mask], 1)
    return y_rec, mask


def evaluate_baselines():
    records = []
    
    for ds in DATASETS:
        for h_val, h_steps in HORIZONS.items():
            h_dir = HORIZON_DIR_MAP[h_val]
            for ctx in CONTEXTS:
                ctx_dir = f"context-{ctx}hr"
                dir_path = os.path.join(OUTPUTS_DIR, ds, h_dir, ctx_dir)
                fc_dir = os.path.join(dir_path, "forecasts")
                ls_dir = os.path.join(dir_path, "losses")
                
                if not os.path.exists(fc_dir):
                    continue
                    
                fc_files = glob.glob(os.path.join(fc_dir, "*_forecasts.csv"))
                if not fc_files:
                    continue
                    
                hbml_rmses = []
                unif_rmses = []
                pers_rmses = []
                best_exp_rmses = []
                
                for f_path in fc_files:
                    pid_match = re.search(r"/(\d{3})_forecasts\.csv", f_path)
                    if not pid_match:
                        continue
                    pid = pid_match.group(1)
                    
                    l_path = os.path.join(ls_dir, f"{pid}_losses.csv")
                    hbml_path = os.path.join(fc_dir, f"{pid}_expforecasts_advanced.csv")
                    if not os.path.exists(l_path) or not os.path.exists(hbml_path):
                        continue
                        
                    fc_df = pd.read_csv(f_path)
                    ls_df = pd.read_csv(l_path)
                    hbml_df = pd.read_csv(hbml_path)
                    
                    # Clean model names (HoltWinters -> ETS)
                    if "HoltWinters" in fc_df.columns:
                        fc_df["ETS"] = fc_df["HoltWinters"]
                    if "HoltWinters" in ls_df.columns:
                        ls_df["ETS"] = ls_df["HoltWinters"]
                        
                    res = reconstruct_ground_truth(fc_df, ls_df)
                    if res is None:
                        continue
                    y_rec, mask = res
                    
                    val_idx = np.arange(100, len(fc_df) - 10)
                    val_idx = val_idx[mask[val_idx]]
                    if len(val_idx) < 30:
                        continue
                        
                    y_true = y_rec[val_idx]
                    
                    # 1. HBML forecast
                    hbml_col = "HBML-SFHDF" if "HBML-SFHDF" in hbml_df.columns else hbml_df.columns[0]
                    h_pred = hbml_df[hbml_col].iloc[val_idx].values
                    h_rmse = np.sqrt(np.mean((h_pred - y_true) ** 2))
                    
                    # 2. Uniform Ensemble
                    cols = [c for c in MODELS if c in fc_df.columns]
                    u_pred = fc_df[cols].iloc[val_idx].mean(axis=1).values
                    u_rmse = np.sqrt(np.mean((u_pred - y_true) ** 2))
                    
                    # 3. Persistence
                    valid_pers = [t for t in val_idx if (t - h_steps) >= 0 and mask[t - h_steps]]
                    if len(valid_pers) >= 20:
                        p_pred = np.array([y_rec[t - h_steps] for t in valid_pers])
                        p_true = np.array([y_rec[t] for t in valid_pers])
                        p_rmse = np.sqrt(np.mean((p_pred - p_true) ** 2))
                    else:
                        p_rmse = np.nan
                        
                    # 4. Expert RMSEs
                    exp_errs = [np.sqrt(np.mean((fc_df[c].iloc[val_idx].values - y_true) ** 2)) for c in cols]
                    best_exp_rmse = min(exp_errs) if exp_errs else np.nan
                    
                    hbml_rmses.append(h_rmse)
                    unif_rmses.append(u_rmse)
                    pers_rmses.append(p_rmse)
                    best_exp_rmses.append(best_exp_rmse)
                    
                if not hbml_rmses:
                    continue
                    
                n = len(hbml_rmses)
                h_mean = np.nanmean(hbml_rmses)
                u_mean = np.nanmean(unif_rmses)
                p_mean = np.nanmean(pers_rmses)
                b_mean = np.nanmean(best_exp_rmses)
                
                # Paired test: HBML vs Uniform
                diff_unif = np.array(unif_rmses) - np.array(hbml_rmses)
                try:
                    w_u = wilcoxon(hbml_rmses, unif_rmses).pvalue
                except Exception:
                    w_u = np.nan
                    
                # Paired test: HBML vs Persistence
                valid_mask = ~np.isnan(pers_rmses)
                if np.sum(valid_mask) > 10:
                    try:
                        w_p = wilcoxon(np.array(hbml_rmses)[valid_mask], np.array(pers_rmses)[valid_mask]).pvalue
                    except Exception:
                        w_p = np.nan
                else:
                    w_p = np.nan
                    
                # Improvements
                pct_better_unif = (np.sum(diff_unif > 0) / n) * 100.0
                unif_red = ((u_mean - h_mean) / u_mean) * 100.0
                pers_red = ((p_mean - h_mean) / p_mean) * 100.0 if not np.isnan(p_mean) else np.nan
                
                record = {
                    "dataset": ds,
                    "horizon": h_val,
                    "context": ctx,
                    "n": n,
                    "hbml_rmse": h_mean,
                    "uniform_rmse": u_mean,
                    "persistence_rmse": p_mean,
                    "best_expert_rmse": b_mean,
                    "diff_uniform": np.nanmean(diff_unif),
                    "pct_improved_vs_uniform": pct_better_unif,
                    "hbml_reduction_vs_uniform_pct": unif_red,
                    "hbml_reduction_vs_persistence_pct": pers_red,
                    "wilcoxon_p_uniform": w_u,
                    "wilcoxon_p_persistence": w_p,
                }
                records.append(record)
                print(f"{ds} h={h_val} ctx={ctx} (n={n}): HBML={h_mean:.2f} | Uniform={u_mean:.2f} (-{unif_red:.1f}%) | Pers={p_mean:.2f} (-{pers_red:.1f}%)")

    df_out = pd.DataFrame(records)
    out_csv = os.path.join(TABLES_DIR, "baselines_comparison.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved baselines comparison to {out_csv}")
    
    # Generate LaTeX table
    tex_path = os.path.join(TABLES_DIR, "table_baselines_comparison.tex")
    generate_latex_table(df_out, tex_path)
    print(f"Saved LaTeX baselines table to {tex_path}")


def generate_latex_table(df, output_tex):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Horizon & Context & Persistence & Uniform Ensemble & Static Best Expert & HBML & Gain vs. Uniform (\%) \\")
    lines.append(r"\midrule")
    
    current_ds = None
    for _, row in df.iterrows():
        ds = DATASET_LABEL[row["dataset"]]
        h = HORIZON_LABEL[row["horizon"]]
        ctx = f"{row['context']} hr" if row["context"] != "full" else "Full"
        
        p_val = f"{row['persistence_rmse']:.2f}" if not pd.isna(row["persistence_rmse"]) else "---"
        u_val = f"{row['uniform_rmse']:.2f}"
        b_val = f"{row['best_expert_rmse']:.2f}"
        h_val = f"\\textbf{{{row['hbml_rmse']:.2f}}}"
        red = f"+{row['hbml_reduction_vs_uniform_pct']:.1f}\\%"
        
        if current_ds is not None and current_ds != ds:
            lines.append(r"\midrule")
        current_ds = ds
        
        line = f"{ds} & {h} & {ctx} & {p_val} & {u_val} & {b_val} & {h_val} & {red} \\\\"
        lines.append(line)
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{Empirical comparison of HBML against standard forecasting baselines.} "
        r"Baselines include naive persistence ($\hat{y}_{t+h} = y_t$), an equally weighted uniform ensemble average "
        r"($\hat{y}_{t+h} = \frac{1}{K}\sum_{k=1}^K \hat{f}_{k,t+h}$), and the static best constituent expert per participant. "
        r"Values report mean participant RMSE (mg/dL). Relative gain denotes percentage error reduction achieved by HBML over uniform averaging.}"
    )
    lines.append(r"\label{tab:baselines_comparison}")
    lines.append(r"\end{table*}")
    
    with open(output_tex, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    evaluate_baselines()
