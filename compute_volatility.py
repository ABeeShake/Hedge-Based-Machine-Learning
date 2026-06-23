#!/usr/bin/env python3
"""
Compute glucose volatility metrics for CGM datasets.

This script computes multiple glucose variability metrics:
- Standard Deviation (SD)
- Coefficient of Variation (CoV)
- Time in Range (TIR)
- Mean Amplitude of Glycemic Excursions (MAGE)
- Mean of Daily Differences (MODD)

Usage:
    python compute_volatility.py --cgmacros_dir ./cgmacros --weinstock_dir ./weinstock
"""

import os  
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def load_glucose_data(file_path: str) -> pd.DataFrame:
    """Load glucose data from CSV file."""
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    # Identify glucose column (could be 'Dexcom.GL', 'Libre.GL', 'CGM', 'Glucose', etc.)
    glucose_col = 'Dexcom.GL' if 'Dexcom.GL' in df.columns else 'Libre.GL'
    
    if glucose_col and glucose_col != 'Glucose':
        df['Glucose'] = df[glucose_col]
    
    return df


def compute_sd(glucose_values: np.ndarray) -> float:
    """Compute Standard Deviation."""
    return np.std(glucose_values, ddof=1)


def compute_cov(glucose_values: np.ndarray) -> float:
    """Compute Coefficient of Variation (%)."""
    mean = np.mean(glucose_values)
    sd = np.std(glucose_values, ddof=1)
    return (sd / mean) * 100 if mean > 0 else 0


def compute_tir(glucose_values: np.ndarray, lower=70, upper=180) -> float:
    """
    Compute Time in Range (%).
    Default range: 70-180 mg/dL (standard for diabetes)
    """
    in_range = np.sum((glucose_values >= lower) & (glucose_values <= upper))
    return (in_range / len(glucose_values)) * 100


def compute_mage(glucose_values: np.ndarray) -> float:
    """
    Compute Mean Amplitude of Glycemic Excursions (MAGE).
    
    MAGE measures intraday glycemic variability by averaging the amplitude
    of glucose excursions that exceed one standard deviation.
    """
    if len(glucose_values) < 3:
        return np.nan
    
    sd = np.std(glucose_values, ddof=1)
    
    # Find local peaks and valleys
    peaks_valleys = []
    for i in range(1, len(glucose_values) - 1):
        if glucose_values[i] > glucose_values[i-1] and glucose_values[i] > glucose_values[i+1]:
            peaks_valleys.append(('peak', i, glucose_values[i]))
        elif glucose_values[i] < glucose_values[i-1] and glucose_values[i] < glucose_values[i+1]:
            peaks_valleys.append(('valley', i, glucose_values[i]))
    
    if len(peaks_valleys) < 2:
        return np.nan
    
    # Calculate excursions between consecutive peaks and valleys
    excursions = []
    for i in range(len(peaks_valleys) - 1):
        excursion = abs(peaks_valleys[i][2] - peaks_valleys[i+1][2])
        if excursion > sd:  # Only count excursions > 1 SD
            excursions.append(excursion)
    
    return np.mean(excursions) if len(excursions) > 0 else 0


def compute_modd(df: pd.DataFrame, glucose_col='Glucose') -> float:
    """
    Compute Mean of Daily Differences (MODD).
    
    MODD measures interday glycemic variability by computing the average
    absolute difference between glucose values at the same time on consecutive days.
    """
    df = df.copy()
    
    # Round timestamps to nearest 5 minutes to ensure alignment across days
    df['Timestamp_rounded'] = df['Timestamp'].dt.round('5min')
    df['Date'] = df['Timestamp_rounded'].dt.date
    df['Time'] = df['Timestamp_rounded'].dt.time
    
    # Pivot to get glucose values by time and date  
    pivot = df.pivot_table(index='Time', columns='Date', values=glucose_col, aggfunc='first')
    
    if pivot.shape[1] < 2:
        return np.nan
    
    # Compute differences between consecutive days
    differences = []
    for i in range(pivot.shape[1] - 1):
        day1 = pivot.iloc[:, i]
        day2 = pivot.iloc[:, i+1]
        
        # Only compute for times present in both days
        mask = ~(day1.isna() | day2.isna())
        if mask.sum() > 0:
            diffs = np.abs(day1[mask] - day2[mask])
            differences.extend(diffs)
    
    return np.mean(differences) if len(differences) > 0 else np.nan


def compute_all_metrics(df: pd.DataFrame, patient_id: str) -> Dict:
    """Compute all volatility metrics for a patient."""
    # Use standardized 'Glucose' column
    if 'Glucose' not in df.columns:
        return None
    
    glucose = df['Glucose'].values
    glucose = glucose[~np.isnan(glucose)]  # Remove NaN values
    
    if len(glucose) == 0:
        return None
    
    metrics = {
        'patient_id': patient_id,
        'n_measurements': len(glucose),
        'mean_glucose': np.mean(glucose),
        'sd': compute_sd(glucose),
        'cov': compute_cov(glucose),
        'tir': compute_tir(glucose),
        'mage': compute_mage(glucose),
        'modd': compute_modd(df)
    }
    
    return metrics


def process_dataset(data_dir: str, dataset_name: str) -> pd.DataFrame:
    """Process all CSV files in a dataset directory."""
    results = []
    
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob('*.csv'))
    
    print(f"\nProcessing {dataset_name} dataset...")
    print(f"Found {len(csv_files)} patient files")
    
    for csv_file in csv_files:
        patient_id = csv_file.stem
        try:
            df = load_glucose_data(str(csv_file))
            metrics = compute_all_metrics(df, patient_id)
            
            if metrics:
                metrics['dataset'] = dataset_name
                results.append(metrics)
                print(f"  Processed {patient_id}: CV={metrics['cov']:.2f}%, TIR={metrics['tir']:.2f}%")
        except Exception as e:
            print(f"  Error processing {patient_id}: {e}")
    
    return pd.DataFrame(results)


def compare_datasets(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str):
    """Compare volatility metrics between two datasets."""
    print(f"\n{'='*80}")
    print(f"VOLATILITY COMPARISON: {name1} vs {name2}")
    print(f"{'='*80}\n")
    
    metrics_to_compare = ['mean_glucose', 'sd', 'cov', 'tir', 'mage', 'modd']
    
    for metric in metrics_to_compare:
        val1 = df1[metric].mean()
        val2 = df2[metric].mean()
        std1 = df1[metric].std()
        std2 = df2[metric].std()
        
        print(f"{metric.upper()}")
        print(f"  {name1:15s}: {val1:8.2f} (±{std1:.2f})")
        print(f"  {name2:15s}: {val2:8.2f} (±{std2:.2f})")
        
        diff_pct = ((val2 - val1) / val1 * 100) if val1 != 0 else 0
        print(f"  Difference:      {val2-val1:8.2f} ({diff_pct:+.1f}%)\n")


def main():
    parser = argparse.ArgumentParser(description='Compute glucose volatility metrics')
    parser.add_argument('--cgmacros_dir', type=str, default='./cgmacros',
                        help='Directory containing CGMacros dataset')
    parser.add_argument('--weinstock_dir', type=str, default='./weinstock',
                        help='Directory containing Weinstock dataset')
    parser.add_argument('--output', type=str, default='volatility_results.csv',
                        help='Output CSV file for results')
    
    args = parser.parse_args()
    
    # Process both datasets
    cgmacros_results = process_dataset(args.cgmacros_dir, 'CGMacros')
    weinstock_results = process_dataset(args.weinstock_dir, 'Weinstock 2016')
    
    # Combine results
    all_results = pd.concat([cgmacros_results, weinstock_results], ignore_index=True)
    
    # Save to CSV
    all_results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Print comparison (only if both datasets have data)
    if len(weinstock_results) > 0 and len(cgmacros_results) > 0:
        compare_datasets(cgmacros_results, weinstock_results, 'CGMacros', 'Weinstock 2016')
    else:
        print("\nWarning: Could not compare datasets - one or both datasets are empty")
        if len(cgmacros_results) > 0:
            print(f"\nCGMacros Summary (n={len(cgmacros_results)}):\n{cgmacros_results.describe()}")
        if len(weinstock_results) > 0:
            print(f"\nWeinstock Summary (n={len(weinstock_results)}):\n{weinstock_results.describe()}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*80}\n")
    print("Standard Deviation (SD): Absolute measure of glucose variability (mg/dL)")
    print("  - Lower is better (less variability)")
    print("\nCoefficient of Variation (CoV): Normalized variability (%)")
    print("  - <36% is considered stable")
    print("  - >36% indicates high variability")
    print("\nTime in Range (TIR): % of time in 70-180 mg/dL")
    print("  - >70% is considered good control")
    print("  - >50% is acceptable")
    print("\nMAGE: Mean amplitude of major glucose excursions (mg/dL)")
    print("  - Measures intraday variability")
    print("  - Lower is better")
    print("\nMODD: Mean of daily differences (mg/dL)")
    print("  - Measures day-to-day consistency")
    print("  - Lower is better (more predictable)")


if __name__ == '__main__':
    main()
