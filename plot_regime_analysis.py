#!/usr/bin/env python3
"""
plot_regime_analysis.py
=======================
Generates visualizations highlighting HBML's effectiveness across physiological regimes.
Implements Figures 2, 3, and 4 from the regime analysis proposal.

Usage
-----
    python plot_regime_analysis.py \
        --results_dir ./Outputs/regime_analysis \
        --tag weinstock_h0.5_c6 \
        --out_dir ./Outputs/regime_analysis/plots
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ExpMethods.globals import GlobalValues
    COLORS = GlobalValues.color_params
except ImportError:
    COLORS = {
        "NODE": "#1f77b4", "ARIMA": "#2ca02c", "HBML": "#d62728",
        "NHITS": "#ff7f0e", "ETS": "#9467bd", "XGBoost": "#8c564b"
    }

# Ensure HBML variants map to the base color
for var in ["HBML-SFHDF", "HBML-SFH", "HBML (c)", "HBML (d)", "HBML (fd)"]:
    COLORS[var] = COLORS.get("HBML", "#d62728")

REGIME_COLORS = {
    "STABLE": "#2ca02c",      # Green
    "RISING": "#ff7f0e",      # Orange
    "FALLING": "#1f77b4",     # Blue
    "HIGHVAR": "#d62728",     # Red
    "RECOVERY": "#9467bd",    # Purple
    "HYPO_TRANS": "#8c564b",  # Brown
}

REGIME_ORDER = ["STABLE", "RISING", "FALLING", "HIGHVAR", "RECOVERY", "HYPO_TRANS"]

def get_hbml_col(models):
    """Find the HBML column name present in the data."""
    if "HBML" in models:
        return "HBML"
    raise ValueError(f"No HBML model found in {models}")

def get_expert_cols(models, hbml_col):
    return [m for m in models if m != hbml_col and m not in ["LSTM", "Autoformer", "FS (Start)", "FS (Uniform)", "FS (Decay)"]]

def plot_figure2_scatter(detail_rmse_path, out_dir, tag):
    """Figure 2: HBML vs. Best Expert Regime Scatter"""
    if not os.path.exists(detail_rmse_path):
        print(f"Skipping Figure 2: {detail_rmse_path} not found.")
        return

    df = pd.read_csv(detail_rmse_path)
    if df.empty:
        return

    models = df["model"].unique()
    hbml_col = get_hbml_col(models)
    expert_cols = get_expert_cols(models, hbml_col)

    # Pivot so columns are models, rows are patient+regime
    pivot_df = df.pivot_table(index=["patient_id", "regime"], columns="model", values="rmse").reset_index()
    
    # Identify the Overall Best Fixed Expert per patient
    overall_df = pivot_df[pivot_df["regime"] == "OVERALL"].copy()
    if overall_df.empty:
        # Fallback to per-regime best if OVERALL is missing (e.g. old data)
        pivot_df["Best_Expert_RMSE"] = pivot_df[expert_cols].min(axis=1)
    else:
        overall_best_experts = overall_df.set_index("patient_id")[expert_cols].idxmin(axis=1)
        
        def get_baseline_rmse(row):
            pid = row["patient_id"]
            if pid in overall_best_experts.index:
                best_mod = overall_best_experts.loc[pid]
                return row[best_mod]
            return np.nan
            
        pivot_df["Best_Expert_RMSE"] = pivot_df.apply(get_baseline_rmse, axis=1)

    pivot_df["HBML_RMSE"] = pivot_df[hbml_col]

    # Filter out NaNs and the OVERALL regime from the plot itself
    plot_df = pivot_df.dropna(subset=["Best_Expert_RMSE", "HBML_RMSE"])
    plot_df = plot_df[plot_df["regime"] != "OVERALL"]

    plt.figure(figsize=(8, 8))
    
    for regime in REGIME_ORDER:
        subset = plot_df[plot_df["regime"] == regime]
        if not subset.empty:
            plt.scatter(
                subset["Best_Expert_RMSE"], subset["HBML_RMSE"],
                label=regime,
                color=REGIME_COLORS.get(regime, "gray"),
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5,
                s=50
            )

    # Diagonal line
    max_val = max(plot_df["Best_Expert_RMSE"].max(), plot_df["HBML_RMSE"].max())
    min_val = min(plot_df["Best_Expert_RMSE"].min(), plot_df["HBML_RMSE"].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Parity (HBML = Best Expert)")

    plt.xlabel("Best Static Expert RMSE (mg/dL)")
    plt.ylabel("HBML RMSE (mg/dL)")
    plt.title("HBML vs. Best Expert by Regime", pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title="Physiological Regime", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    out_path = Path(out_dir) / f"fig2_scatter_rmse_{tag}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 → {out_path}")

def plot_figure3_stacked_bars(best_rmse_path, best_ab_path, out_dir, tag):
    """Figure 3: Best-Expert Frequency Stacked Bars"""
    if not os.path.exists(best_rmse_path) or not os.path.exists(best_ab_path):
        print("Skipping Figure 3: required aggregate files not found.")
        return

    df_rmse = pd.read_csv(best_rmse_path)
    df_ab = pd.read_csv(best_ab_path)

    plt.rcParams.update(GlobalValues.plot_params)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    def _plot_stacked(ax, df, title):
        if df.empty: return
        df = df[df["regime"] != "OVERALL"].copy()
        # Ensure we order regimes correctly
        df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER[::-1], ordered=True)
        # Pivot for stacked bar
        pivot = df.pivot_table(index="regime", columns="model", values="freq_mean", fill_value=0, observed=False)
        
        # Determine color mapping for models
        model_colors = [COLORS.get(m, "#333333") for m in pivot.columns]
        
        pivot.plot(kind="barh", stacked=True, ax=ax, color=model_colors, width=0.8, edgecolor="white", linewidth=0.5, legend=False)
        
        ax.set_title(title, pad=10)
        ax.set_xlabel("Average Frequency (Fraction of Timesteps)")
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    _plot_stacked(axes[0], df_rmse, "Frequency of Best RMSE")
    _plot_stacked(axes[1], df_ab, "Frequency of Best A+B% Zone")

    # Legend placement logic based on dataset tag
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        if tag.startswith("cgmacros"):
            pass # Exclude legend from all CGMacros plots
        elif tag.startswith("weinstock"):
            # Exclude from RMSE (done by default via legend=False in _plot_stacked)
            # Place legend on right side of A+B% (ax[1])
            axes[1].legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Model")

    plt.suptitle("Model Selection Frequency by Regime", y=1.02)
    plt.tight_layout()
    out_path = Path(out_dir) / f"fig3_stacked_freq_{tag}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3 → {out_path}")


def plot_figure4_radar(agg_ceg_path, out_dir, tag):
    """Figure 4: Clinical Safety Radar Chart (Zone A+B%)"""
    if not os.path.exists(agg_ceg_path):
        print(f"Skipping Figure 4: {agg_ceg_path} not found.")
        return

    df = pd.read_csv(agg_ceg_path)
    if df.empty or "AB_mean" not in df.columns:
        return

    models = df["model"].unique()
    hbml_col = get_hbml_col(models)
    expert_cols = get_expert_cols(models, hbml_col)

    # Find HBML AB% per regime
    hbml_data = df[df["model"] == hbml_col].set_index("regime")["AB_mean"]
    
    # Find Best Overall Static Expert
    experts_df = df[df["model"].isin(expert_cols)]
    overall_df = experts_df[experts_df["regime"] == "OVERALL"]
    
    if overall_df.empty:
        best_expert_data = experts_df.groupby("regime")["AB_mean"].max()
        best_expert_label = "Best Static Expert (per regime)"
    else:
        best_overall_model = overall_df.loc[overall_df["AB_mean"].idxmax(), "model"]
        best_expert_data = experts_df[experts_df["model"] == best_overall_model].set_index("regime")["AB_mean"]
        best_expert_label = f"Best Overall Expert ({best_overall_model})"

    # Align with REGIME_ORDER, filling missing with NaN
    regimes_present = [r for r in REGIME_ORDER if r in hbml_data.index and r in best_expert_data.index and r != "OVERALL"]
    if not regimes_present:
        return

    hbml_vals = [hbml_data.get(r, 0) * 100 for r in regimes_present] # convert to %
    expert_vals = [best_expert_data.get(r, 0) * 100 for r in regimes_present]

    # Close the loop for radar chart
    angles = np.linspace(0, 2 * np.pi, len(regimes_present), endpoint=False).tolist()
    hbml_vals += hbml_vals[:1]
    expert_vals += expert_vals[:1]
    angles += angles[:1]
    labels = regimes_present + [regimes_present[0]] # For x-ticks

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    # Plot HBML
    ax.plot(angles, hbml_vals, color=COLORS.get("HBML", "red"), linewidth=2.5, linestyle='solid', label="HBML")
    ax.fill(angles, hbml_vals, color=COLORS.get("HBML", "red"), alpha=0.1)

    # Plot Best Expert
    ax.plot(angles, expert_vals, color="gray", linewidth=2, linestyle='dashed', label=best_expert_label)
    
    # Fix axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw labels
    ax.set_xticks(angles[:-1])
    # Replace underscores with spaces for prettier labels
    pretty_labels = [r.replace("_", " ").title() for r in regimes_present]
    ax.set_xticklabels(pretty_labels, fontweight='bold')

    # Configure radial ticks
    ax.set_ylim(80, 100) # Assuming AB% is generally high. Adjust if needed.
    ax.set_yticks([80, 85, 90, 95, 100])
    ax.set_yticklabels(["80%", "85%", "90%", "95%", "100%"], color="grey")

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("HBML Clinical Safety (A+B%) by Regime", pad=20, y=1.1)
    
    plt.tight_layout()
    out_path = Path(out_dir) / f"fig4_radar_ceg_{tag}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4 → {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot regime analysis results.")
    parser.add_argument("--results_dir", required=True, help="Directory containing analyze_regimes CSVs")
    parser.add_argument("--tag", required=True, help="Tag used for the files, e.g., weinstock_h0.5_c6")
    parser.add_argument("--out_dir", required=True, help="Directory to save plots")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    d_rmse = os.path.join(args.results_dir, f"detail_rmse_{args.tag}.csv")
    a_rmse_freq = os.path.join(args.results_dir, f"aggregate_best_rmse_freq_{args.tag}.csv")
    a_ab_freq = os.path.join(args.results_dir, f"aggregate_best_ab_freq_{args.tag}.csv")
    a_ceg = os.path.join(args.results_dir, f"aggregate_ceg_{args.tag}.csv")

    print(f"Plotting figures for tag: {args.tag}")
    plot_figure2_scatter(d_rmse, args.out_dir, args.tag)
    plot_figure3_stacked_bars(a_rmse_freq, a_ab_freq, args.out_dir, args.tag)
    plot_figure4_radar(a_ceg, args.out_dir, args.tag)

if __name__ == "__main__":
    main()
