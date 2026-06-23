#!/usr/bin/env python3
"""
Visualize glucose volatility and format results for LaTeX.

Generates:
1. Boxplots comparing volatility metrics across datasets.
2. Violin plots comparing volatility metrics across datasets.
3. A formatted CSV table for LaTeX csvsimple package.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import mannwhitneyu

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ExpMethods.globals import GlobalValues

plt.rcParams.update(GlobalValues.plot_params)

def load_data(filepath):
    """Load volatility results."""
    return pd.read_csv(filepath)

def compute_p_values(df):
    """Compute Mann-Whitney U test p-values for each metric."""
    metrics = ['sd', 'cov', 'tir', 'mage', 'modd']
    datasets = df['dataset'].unique()
    
    if len(datasets) != 2:
        print("Warning: Need exactly 2 datasets for comparison. Skipping stats.")
        return {}
        
    ds1, ds2 = datasets[0], datasets[1]
    p_values = {}
    
    print(f"Comparing {ds1} vs {ds2}...")
    
    for metric in metrics:
        group1 = df[df['dataset'] == ds1][metric].dropna()
        group2 = df[df['dataset'] == ds2][metric].dropna()
        
        # Mann-Whitney U test (non-parametric t-test equivalent)
        stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
        p_values[metric] = p
        print(f"  {metric.upper()}: p={p:.2e}")
        
    return p_values

def plot_volatility(df, output_dir):
    """Generate boxplots and violin plots using matplotlib."""
    metrics = {
        'sd': 'Standard Deviation (mg/dL)',
        'cov': 'Coefficient of Variation (%)',
        'tir': 'Time in Range (%)',
        'mage': 'MAGE (mg/dL)',
        'modd': 'MODD (mg/dL)'
    }
    
    unique_datasets = df['dataset'].unique()
    dataset_colors = ['lightblue', 'lightgreen']
    
    # rcParams already applied via GlobalValues.plot_params at module level
    
    # 1. Boxplots
    fig, axes = plt.subplots(1, 5, figsize=(22, 7)) # Increased figure size slightly
    plt.subplots_adjust(wspace=0.4)
    
    for i, (metric, label) in enumerate(metrics.items()):
        data_to_plot = [df[df['dataset'] == ds][metric].dropna() for ds in unique_datasets]
        
        # Create boxplot
        # changed labels -> tick_labels to fix warning
        bp = axes[i].boxplot(data_to_plot, patch_artist=True, tick_labels=unique_datasets)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], dataset_colors):
            patch.set_facecolor(color)
            
        # Add jittered points (strip plot equivalent)
        for j, ds in enumerate(unique_datasets):
            y = df[df['dataset'] == ds][metric].dropna()
            x = np.random.normal(j + 1, 0.04, size=len(y))
            axes[i].plot(x, y, 'r.', alpha=0.5)
            
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel(label)
        
    plt.suptitle('Glucose Volatility Comparison: CGMacros vs Weinstock 2016')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.savefig(os.path.join(output_dir, 'data/volatility_boxplot.pdf'), dpi=300)
    plt.close()
    
    # 2. Violin Plots
    fig, axes = plt.subplots(1, 5, figsize=(22, 7))
    plt.subplots_adjust(wspace=0.4)
    
    for i, (metric, label) in enumerate(metrics.items()):
        data_to_plot = [df[df['dataset'] == ds][metric].dropna() for ds in unique_datasets]
        
        # Create violin plot
        parts = axes[i].violinplot(data_to_plot, showmeans=False, showmedians=True)
        
        # Color the bodies
        for pc, color in zip(parts['bodies'], dataset_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            
        axes[i].set_xticks([1, 2])
        axes[i].set_xticklabels(unique_datasets)
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel(label)

    plt.suptitle('Glucose Volatility Distributions: CGMacros vs Weinstock 2016')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'data/volatility_violin.pdf'), dpi=300)
    plt.close()

    print(f"Plots saved to {output_dir}")

def format_latex_table(df, p_values, output_path):
    """Format data for LaTeX csvsimple table."""
    metrics = ['sd', 'cov', 'tir', 'mage', 'modd']
    metric_names = ['SD', 'CoV', 'TIR', 'MAGE', 'MODD']
    
    datasets = ['CGMacros', 'Weinstock 2016']
    dataset_labels = {'CGMacros': 'CGMacros', 'Weinstock 2016': 'Weinstock'}
    
    rows = []
    
    for i, metric in enumerate(metrics):
        metric_label = metric_names[i]
        p_val = p_values.get(metric, 1.0)
        
        # Format p-value column
        if p_val < 0.001:
            p_str = "$<0.001$"
        elif p_val < 0.01:
            p_str = f"{p_val:.3f}"
        else:
            p_str = f"{p_val:.3f}"
        
        for j, dataset in enumerate(datasets):
            subset = df[df['dataset'] == dataset]
            mean_val = subset[metric].mean()
            sd_val = subset[metric].std()
            
            # Format value string: "Mean (SD)"
            val_str = f"{mean_val:.2f} ({sd_val:.2f})"
            
            # Formatting commands
            # Formatting commands
            if j == 0:
                # First row of the metric group
                metric_col = f"\\multirow{{2}}{{*}}{{{metric_label}}}"
                p_col = f"\\multirow{{2}}{{*}}{{{p_str}}}"  # P-value spans 2 rows
                style = ""
            else:
                # Second row
                metric_col = ""
                p_col = ""
                # Add midrule only if it's not the very last row
                if i < len(metrics) - 1:
                    style = "\\midrule"
                else:
                    style = "" # No midrule after last group
            
            rows.append({
                'Metric': metric_col,
                'Dataset': dataset_labels[dataset],
                'Value': val_str,
                'P-value': p_col,
                'Style': style
            })
            
    # Create DataFrame
    table_df = pd.DataFrame(rows)
    
    # Save to CSV
    table_df.to_csv(output_path, index=False)
    print(f"Formatted table saved to {output_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'volatility_results.csv')
    images_dir = os.path.join(base_dir, 'overleaf', 'images')
    tables_dir = os.path.join(base_dir, 'overleaf', 'tables')
    
    # Ensure directories exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    # Load data
    df = load_data(input_file)
    
    # Compute P-values
    p_values = compute_p_values(df)
    
    # Generate plots
    plot_volatility(df, images_dir)
    
    # Format table
    format_latex_table(df, p_values, os.path.join(tables_dir, 'volatility_summary.csv'))

if __name__ == "__main__":
    main()
