import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # drop old hbml variants
    df = df[~df["model"].isin(["HBML (c)", "HBML (d)", "LSTM", "Autoformer", "FS (Start)", "FS (Uniform)", "FS (Decay)"])]
    df.loc[df["model"].isin(["HBML-SFHDF", "HBML (fd)"]), "model"] = "HBML"
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
    
    # Create plot
    plt.rcParams.update(GlobalValues.plot_params)
    sns.set_style("whitegrid")
    
    g = sns.catplot(
        data=agg, x="status", y="rmse", hue="model", col="horizon", 
        kind="bar", palette=palette, hue_order=model_order,
        height=5, aspect=0.9, sharey=False, edgecolor="white", linewidth=0.5,
        legend=False
    )
    g.set_axis_labels("Diabetic Status", "Median RMSE (mg/dL)")
    g.set_titles("Horizon: {col_name} hr")
    plt.suptitle("CGMacros: Predictive Accuracy across Patient Phenotypes", y=1.05, fontsize=GlobalValues.plot_params.get("figure.titlesize", 22))
    
    out_file = Path(out_dir) / "subgroup_rmse.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved RMSE subgroup plot -> {out_file}")

def load_cgmacros_ceg():
    import glob
    import re
    files = glob.glob("../cgmacros/h-*/context-*/ceg.csv")
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
    
    # Create plot
    plt.rcParams.update(GlobalValues.plot_params)
    sns.set_style("whitegrid")

    g = sns.catplot(
        data=agg, x="status", y="A+B_pct", hue="model", col="horizon", 
        kind="bar", palette=palette, hue_order=model_order,
        height=5, aspect=0.9, sharey=False, edgecolor="white", linewidth=0.5,
        legend=False
    )
    g.set_axis_labels("Diabetic Status", "Zone A+B (%)")
    g.set_titles("Horizon: {col_name} hr")
    
    g.add_legend(title="Model", bbox_to_anchor=(1.05, 0.5), loc="center left")
    
    # Set consistent y-limits for CEG to allow easy visual comparison across horizons
    for ax in g.axes.flat:
        ax.set_ylim(90, 100)
            
    plt.suptitle("CGMacros: Clinical Safety across Patient Phenotypes", y=1.05, fontsize=GlobalValues.plot_params.get("figure.titlesize", 22))
    
    out_file = Path(out_dir) / "subgroup_ceg.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved CEG subgroup plot -> {out_file}")

if __name__ == "__main__":
    out_dir = "../overleaf/images/subgroups"
    os.makedirs(out_dir, exist_ok=True)
    
    rmse_path = "../cgmacros/rmse.csv"
    if not os.path.exists(rmse_path) and os.path.exists("../rmse.csv"):
        rmse_path = "../rmse.csv"
        
    print("Generating Subgroup Visualization Plots...")
    plot_subgroup_rmse(rmse_path, out_dir)
    plot_subgroup_ceg(out_dir)
    print(f"Done. Plots saved to {out_dir}")
