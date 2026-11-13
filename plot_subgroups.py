import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ExpMethods.globals import GlobalValues

COLORS = GlobalValues.color_params
# map HBML aliases
for var in ["HBML-SFHDF", "HBML-SFH", "HBML (c)", "HBML (d)", "HBML (fd)"]:
    COLORS[var] = COLORS.get("HBML", "#d62728")

def get_diabetic_status(patient_id):
    pid_str = str(patient_id).zfill(3)
    if pid_str in GlobalValues.CGMacros.get("diabetic_ids", []):
        return "T2D"
    if pid_str in GlobalValues.CGMacros.get("prediabetic_ids", []):
        return "pre-T2D"
    if pid_str in GlobalValues.CGMacros.get("nondiabetic_ids", []):
        return "non-T2D"
    return "Unknown"

def clean_data(df, is_ceg=False):
    # filter and remap models
    if "model" not in df.columns:
        return df
    
    # drop old/non-primary hbml variants
    df = df[~df["model"].isin(["HBML (c)", "HBML (d)", "HBML (fd)", "HBML-SFH", "HBML-SFH-SD", "HBML-SFH-Meta", "LSTM", "Autoformer", "FS (Start)", "FS (Uniform)", "FS (Decay)"])].copy()
    df.loc[df["model"].isin(["HBML-SFHDF"]), "model"] = "HBML"
    df.loc[df["horizon"] == "half", "horizon"] = "0.5"
    
    df["status"] = df["id"].apply(get_diabetic_status)
    df = df[df["status"] != "Unknown"]
    
    # Drop rows with NaN RMSE or CEG
    if is_ceg:
        df = df.dropna(subset=["A", "B", "C"])
    else:
        df = df.dropna(subset=["rmse"])
    
    return df

def plot_subgroup_rmse(rmse_path, out_dir):
    if not os.path.exists(rmse_path):
        print(f"Skipping RMSE subgroup plot: {rmse_path} not found.")
        return
        
    df = pd.read_csv(rmse_path)
    # Only keep cgmacros dataset
    if "dataset" in df.columns:
        df = df[df["dataset"] == "cgmacros"]
    df = clean_data(df)
    
    if df.empty:
        print("No valid CGMacros data found for RMSE subgroup plot.")
        return
    
    # Get optimal params for each model/patient (like generate_figures.py does)
    min_idx = df.groupby(["horizon","context","model","id"])["rmse"].idxmin()
    df = df.loc[min_idx].drop(columns=["eta"], errors='ignore')
    
    # Aggregate over contexts (or you can filter to context=24 if desired)
    # Here we take the median RMSE over contexts per model/patient
    agg = df.groupby(["status", "horizon", "model", "id"])["rmse"].median().reset_index()
    agg = agg.groupby(["status", "horizon", "model"])["rmse"].median().reset_index()
    
    # Set categorical orders
    agg["status"] = pd.Categorical(agg["status"], categories=["non-T2D", "pre-T2D", "T2D"], ordered=True)
    agg["horizon"] = pd.Categorical(agg["horizon"], categories=["0.5", "2", "5"], ordered=True)
    
    # define model colors
    model_order = ["ARIMA", "ETS", "NHITS", "NODE", "XGBoost", "HBML"]
    model_order = [m for m in model_order if m in agg["model"].unique()]
    palette = [COLORS.get(m, "#333333") for m in model_order]
    
    # Create plot using matplotlib
    plt.rcParams.update(GlobalValues.plot_params)
    statuses = ["non-T2D", "pre-T2D", "T2D"]
    horizons = ["0.5", "2", "5"]
    fig, axes = plt.subplots(1, len(horizons), figsize=(15, 5), sharey=False)
    n_models = len(model_order)
    bar_width = 0.8 / max(n_models, 1)
    x_pos = np.arange(len(statuses))

    for h_idx, h_val in enumerate(horizons):
        ax = axes[h_idx]
        h_df = agg[agg["horizon"] == h_val]
        for m_idx, m_name in enumerate(model_order):
            m_df = h_df[h_df["model"] == m_name].set_index("status")
            vals = [m_df.loc[s, "rmse"] if s in m_df.index else 0.0 for s in statuses]
            offset = (m_idx - (n_models - 1) / 2.0) * bar_width
            ax.bar(
                x_pos + offset, vals, width=bar_width,
                color=COLORS.get(m_name, "#333333"), edgecolor="white", linewidth=0.5,
                label=m_name
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(statuses)
        ax.set_xlabel("Diabetic Status")
        if h_idx == 0:
            ax.set_ylabel("Median RMSE (mg/dL)")
        ax.set_title(f"Horizon: {h_val} hr")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("CGMacros: Predictive Accuracy across Patient Phenotypes", y=1.05, fontsize=GlobalValues.plot_params.get("figure.titlesize", 22))
    plt.tight_layout()
    out_file = Path(out_dir) / "subgroup_rmse.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved RMSE subgroup plot -> {out_file}")

def load_cgmacros_ceg():
    import glob
    import re
    files = glob.glob("./Outputs/cgmacros/h-*/context-*/ceg.csv") or glob.glob("../cgmacros/h-*/context-*/ceg.csv")
    dfs = []
    for f in files:
        m = re.match(r".*/h-([^/]+?)hr/context-([^/]+?)hr/ceg\.csv", f)
        if not m:
            continue
        try:
            df = pd.read_csv(f, on_bad_lines="skip")
            df["horizon"] = m.group(1)
            df["context"] = m.group(2)
            df["dataset"] = "cgmacros"
            dfs.append(df)
        except Exception as e:
            pass
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def plot_subgroup_ceg(out_dir):
    df = load_cgmacros_ceg()
    
    if df.empty:
        print("No valid CGMacros data found for CEG subgroup plot.")
        return
    
    df = clean_data(df, is_ceg=True)
    
    if df.empty:
        print("No valid CGMacros data found for CEG subgroup plot after cleaning.")
        return
    
    df["A+B"] = df["A"] + df["B"]
    df["total"] = df[["A", "B", "C", "D1", "D2", "E1", "E2"]].sum(axis=1)
    df["A+B_pct"] = (df["A+B"] / df["total"]) * 100
    
    # Filter to optimal models (max A+B)
    min_idx = df.groupby(["horizon","context","model","id"])["A+B_pct"].idxmax()
    df = df.loc[min_idx].drop(columns=["eta"], errors='ignore')
    
    # Aggregate over contexts
    agg = df.groupby(["status", "horizon", "model", "id"])["A+B_pct"].mean().reset_index()
    agg = agg.groupby(["status", "horizon", "model"])["A+B_pct"].mean().reset_index()
    
    # Set categorical orders
    agg["status"] = pd.Categorical(agg["status"], categories=["non-T2D", "pre-T2D", "T2D"], ordered=True)
    agg["horizon"] = pd.Categorical(agg["horizon"], categories=["0.5", "2", "5"], ordered=True)
    
    model_order = ["ARIMA", "ETS", "NHITS", "NODE", "XGBoost", "HBML"]
    model_order = [m for m in model_order if m in agg["model"].unique()]
    palette = [COLORS.get(m, "#333333") for m in model_order]
    
    # Create plot using matplotlib
    plt.rcParams.update(GlobalValues.plot_params)
    statuses = ["non-T2D", "pre-T2D", "T2D"]
    horizons = ["0.5", "2", "5"]
    fig, axes = plt.subplots(1, len(horizons), figsize=(15, 5), sharey=False)
    n_models = len(model_order)
    bar_width = 0.8 / max(n_models, 1)
    x_pos = np.arange(len(statuses))

    for h_idx, h_val in enumerate(horizons):
        ax = axes[h_idx]
        h_df = agg[agg["horizon"] == h_val]
        for m_idx, m_name in enumerate(model_order):
            m_df = h_df[h_df["model"] == m_name].set_index("status")
            vals = [m_df.loc[s, "A+B_pct"] if s in m_df.index else 0.0 for s in statuses]
            offset = (m_idx - (n_models - 1) / 2.0) * bar_width
            ax.bar(
                x_pos + offset, vals, width=bar_width,
                color=COLORS.get(m_name, "#333333"), edgecolor="white", linewidth=0.5,
                label=m_name
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(statuses)
        ax.set_xlabel("Diabetic Status")
        if h_idx == 0:
            ax.set_ylabel("Zone A+B (%)")
        ax.set_title(f"Horizon: {h_val} hr")
        ax.set_ylim(90, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, title="Model", bbox_to_anchor=(1.05, 0.5), loc="center left")

    plt.suptitle("CGMacros: Clinical Safety across Patient Phenotypes", y=1.05, fontsize=GlobalValues.plot_params.get("figure.titlesize", 22))
    plt.tight_layout()
    out_file = Path(out_dir) / "subgroup_ceg.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved CEG subgroup plot -> {out_file}")

if __name__ == "__main__":
    out_dir = "overleaf/images/subgroups" if os.path.exists("overleaf") else "../overleaf/images/subgroups"
    os.makedirs(out_dir, exist_ok=True)
    
    rmse_path = "rmse.csv"
    for cand in ["./cgmacros/rmse.csv", "../cgmacros/rmse.csv", "./rmse.csv", "../rmse.csv"]:
        if os.path.exists(cand):
            rmse_path = cand
            break
        
    print("Generating Subgroup Visualization Plots...")
    plot_subgroup_rmse(rmse_path, out_dir)
    plot_subgroup_ceg(out_dir)
    print(f"Done. Plots saved to {out_dir}")
