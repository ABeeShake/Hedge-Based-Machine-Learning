#!/usr/bin/env python3
"""
compute_regimes.py
==================
Classify each CGM timestep into a dynamic-state physiological regime based on:
  - Glucose level (euglycemia / hypo / hyper bounds)
  - Rate of change  (rolling linear slope over a short window)
  - Local volatility (rolling variance over a medium window)

Regime taxonomy
---------------
Regime                  | Short code | Primary criteria
------------------------|------------|-----------------------------------------
Stable Euglycemia       | STABLE     | in-range glucose, low |slope|, low variance
Rising Excursion        | RISING     | positive slope above threshold
Falling Excursion       | FALLING    | negative slope below threshold
High Variability        | HIGHVAR    | rolling variance above threshold
Hypoglycemic Transition | HYPO_TRANS | near-hypo glucose with downward trend
Recovery Phase          | RECOVERY   | rebound after any excursion (glucose
                        |            | returning toward range from outside)

Priority order (applied top-to-bottom):
  1. Hypoglycemic Transition  – safety-critical, checked first
  2. Recovery Phase           – rebound signal
  3. High Variability         – local chaos overrides direction labels
  4. Rising / Falling         – directional excursions
  5. Stable Euglycemia        – default when nothing else fires

Usage
-----
    # Classify a single patient file
    python compute_regimes.py --file ./weinstock/data/103.csv

    # Classify all patients in both datasets and save summaries
    python compute_regimes.py \
        --cgmacros_dir ./cgmacros/data \
        --weinstock_dir ./weinstock/data \
        --out_dir ./Outputs/regimes

Outputs
-------
  <out_dir>/<dataset>/<patient_id>_regimes.csv   — per-timestep regime labels
  <out_dir>/regime_summary.csv                   — per-patient regime distribution (%)
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Default thresholds  (all in mg/dL or mg/dL per 5-min interval)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # Glucose range boundaries
    hypo_thresh      = 80,    # mg/dL  — "near-hypo" upper bound
    hyper_thresh     = 180,   # mg/dL  — hyperglycaemia lower bound
    in_range_low     = 70,    # mg/dL
    in_range_high    = 180,   # mg/dL

    # Slope thresholds (mg/dL per 5-min interval)
    slope_rise_thresh  =  1.5,   # positive slope → Rising Excursion
    slope_fall_thresh  = -1.5,   # negative slope → Falling Excursion
    slope_down_hypo    = -1.0,   # downward trend near hypo → Hypo Transition

    # Rolling windows (number of 5-min intervals)
    slope_window  = 6,    # 30 min  — window for linear slope estimation
    var_window    = 12,   # 60 min  — window for rolling variance

    # Variance threshold (mg/dL²)
    var_thresh = 100,     # ≈ SD of 10 mg/dL over the rolling window

    # Recovery: glucose must be moving back toward range
    recovery_margin = 20,  # mg/dL outside range where recovery can be declared
)

REGIME_CODES = [
    "HYPO_TRANS",
    "RECOVERY",
    "HIGHVAR",
    "RISING",
    "FALLING",
    "STABLE",
]


# ---------------------------------------------------------------------------
# Data loading  (mirrors compute_volatility.py)
# ---------------------------------------------------------------------------

def load_glucose_series(file_path: str) -> pd.Series:
    """
    Load a patient CSV and return a clean pd.Series of glucose values
    indexed by Timestamp, sorted chronologically.
    """
    df = pd.read_csv(file_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp")

    glucose_col = next(
        (c for c in ("Dexcom.GL", "Libre.GL", "CGM", "Glucose") if c in df.columns),
        None,
    )
    if glucose_col is None:
        raise ValueError(f"No recognised glucose column in {file_path}")

    series = df.set_index("Timestamp")[glucose_col].astype(float)
    return series


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Estimate local rate-of-change as the OLS slope over a rolling window.
    Units: mg/dL per interval (1 interval = 5 min for standard CGM).
    """
    def _slope(y: np.ndarray) -> float:
        n = len(y)
        if n < 2 or np.all(np.isnan(y)):
            return np.nan
        x = np.arange(n, dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        xm, ym = x[mask] - x[mask].mean(), y[mask] - y[mask].mean()
        denom = (xm ** 2).sum()
        return float((xm * ym).sum() / denom) if denom > 0 else 0.0

    return series.rolling(window, min_periods=max(2, window // 2)).apply(
        _slope, raw=True
    )


def compute_features(
    glucose: pd.Series,
    slope_window: int,
    var_window: int,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns: glucose, slope, rolling_var.
    All indexed identically to `glucose`.
    """
    slope = _rolling_slope(glucose, slope_window)
    rolling_var = glucose.rolling(var_window, min_periods=max(2, var_window // 2)).var()

    return pd.DataFrame(
        {"glucose": glucose, "slope": slope, "rolling_var": rolling_var},
        index=glucose.index,
    )


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regimes(features: pd.DataFrame, thresholds: dict) -> pd.Series:
    """
    Apply the priority-ordered regime taxonomy to a feature DataFrame.

    Parameters
    ----------
    features : DataFrame with columns [glucose, slope, rolling_var]
    thresholds : dict of threshold parameters (see DEFAULTS)

    Returns
    -------
    pd.Series of regime codes, same index as `features`.
    """
    g   = features["glucose"]
    s   = features["slope"]
    var = features["rolling_var"]

    t = thresholds   # shorthand

    in_range  = (g >= t["in_range_low"]) & (g <= t["in_range_high"])
    near_hypo = g < t["hypo_thresh"]
    hyper     = g > t["hyper_thresh"]
    low_slope = s.abs() < t["slope_rise_thresh"]
    low_var   = var < t["var_thresh"]

    # --- Recovery: glucose was outside range but is now moving back toward it ---
    # Rising from hypo side (glucose < in_range_low but slope > 0)
    recovery_from_low  = (g < t["in_range_low"]) & (g > t["in_range_low"] - t["recovery_margin"]) & (s > 0)
    # Falling from hyper side (glucose > in_range_high but slope < 0)
    recovery_from_high = (g > t["in_range_high"]) & (g < t["in_range_high"] + t["recovery_margin"]) & (s < 0)
    recovery = recovery_from_low | recovery_from_high

    # --- Priority ordering ---
    regime = pd.Series("STABLE", index=features.index, dtype=str)

    # 5. Falling Excursion
    regime[s < t["slope_fall_thresh"]] = "FALLING"

    # 4. Rising Excursion
    regime[s > t["slope_rise_thresh"]] = "RISING"

    # 3. High Variability (overrides directional labels unless safety-critical)
    regime[var >= t["var_thresh"]] = "HIGHVAR"

    # 2. Recovery Phase
    regime[recovery] = "RECOVERY"

    # 1. Hypoglycemic Transition (highest priority)
    hypo_trans = near_hypo & (s < t["slope_down_hypo"])
    regime[hypo_trans] = "HYPO_TRANS"

    return regime


# ---------------------------------------------------------------------------
# Per-patient pipeline
# ---------------------------------------------------------------------------

def classify_patient(
    file_path: str,
    thresholds: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load a patient CSV, compute features, classify regimes.

    Returns
    -------
    DataFrame with columns [glucose, slope, rolling_var, regime].
    """
    t = {**DEFAULTS, **(thresholds or {})}
    glucose  = load_glucose_series(file_path)
    features = compute_features(glucose, t["slope_window"], t["var_window"])
    regimes  = classify_regimes(features, t)
    features["regime"] = regimes
    return features


def regime_distribution(regimes: pd.Series) -> pd.Series:
    """
    Return the fraction (0–1) of timesteps in each regime category.
    All six codes are always present (zero-filled if absent).
    """
    counts = regimes.value_counts()
    dist   = counts.reindex(REGIME_CODES, fill_value=0) / len(regimes)
    return dist


# ---------------------------------------------------------------------------
# Dataset-level pipeline
# ---------------------------------------------------------------------------

def process_dataset(
    data_dir: str,
    dataset_name: str,
    out_dir: Optional[str] = None,
    thresholds: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Classify regimes for all patient CSVs in `data_dir`.

    Returns
    -------
    DataFrame with one row per patient containing regime distribution columns
    plus dataset and patient_id metadata.
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))
    print(f"\nProcessing {dataset_name} ({len(csv_files)} patients)...")

    if out_dir:
        patient_out_dir = Path(out_dir) / dataset_name
        patient_out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for csv_file in csv_files:
        patient_id = csv_file.stem
        try:
            df = classify_patient(str(csv_file), thresholds)

            if out_dir:
                out_path = patient_out_dir / f"{patient_id}_regimes.csv"
                df.to_csv(out_path)
                print(f"  Saved per-timestep regimes → {out_path}")

            dist = regime_distribution(df["regime"])
            row  = dist.to_dict()
            row["patient_id"] = patient_id
            row["dataset"]    = dataset_name
            row["n_timesteps"] = len(df)
            summary_rows.append(row)

            regime_counts = df["regime"].value_counts().to_dict()
            pct_stable = 100 * row.get("STABLE", 0)
            print(f"  {patient_id}: STABLE={pct_stable:.1f}%  {regime_counts}")

        except Exception as e:
            print(f"  Error processing {patient_id}: {e}")

    summary = pd.DataFrame(summary_rows)
    # Reorder columns
    meta_cols   = ["patient_id", "dataset", "n_timesteps"]
    regime_cols = [c for c in REGIME_CODES if c in summary.columns]
    summary = summary[meta_cols + regime_cols]
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Classify CGM timesteps into dynamic-state physiological regimes."
    )
    p.add_argument("--file",          type=str, help="Single patient CSV to classify.")
    p.add_argument("--cgmacros_dir",  type=str, default=None, help="CGMacros data directory.")
    p.add_argument("--weinstock_dir", type=str, default=None, help="Weinstock data directory.")
    p.add_argument("--out_dir",       type=str, default="./Outputs/regimes",
                   help="Root output directory for per-patient regime CSVs.")
    p.add_argument("--summary_out",   type=str, default="./Outputs/regimes/regime_summary.csv",
                   help="Output path for the per-patient summary CSV.")

    # Threshold overrides
    p.add_argument("--hypo_thresh",       type=float, default=DEFAULTS["hypo_thresh"])
    p.add_argument("--hyper_thresh",      type=float, default=DEFAULTS["hyper_thresh"])
    p.add_argument("--slope_rise_thresh", type=float, default=DEFAULTS["slope_rise_thresh"])
    p.add_argument("--slope_fall_thresh", type=float, default=DEFAULTS["slope_fall_thresh"])
    p.add_argument("--slope_down_hypo",   type=float, default=DEFAULTS["slope_down_hypo"])
    p.add_argument("--var_thresh",        type=float, default=DEFAULTS["var_thresh"])
    p.add_argument("--slope_window",      type=int,   default=DEFAULTS["slope_window"])
    p.add_argument("--var_window",        type=int,   default=DEFAULTS["var_window"])
    return p


def main():
    args   = build_parser().parse_args()
    thresh = {k: getattr(args, k) for k in DEFAULTS if hasattr(args, k)}

    # --- Single file mode ---
    if args.file:
        df = classify_patient(args.file, thresh)
        print(df["regime"].value_counts())
        print(f"\nRegime distribution:\n{regime_distribution(df['regime']).mul(100).round(2)}")
        out = Path(args.out_dir) / (Path(args.file).stem + "_regimes.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out)
        print(f"Saved → {out}")
        return

    # --- Dataset mode ---
    summaries = []

    if args.cgmacros_dir:
        s = process_dataset(args.cgmacros_dir, "cgmacros", args.out_dir, thresh)
        summaries.append(s)

    if args.weinstock_dir:
        s = process_dataset(args.weinstock_dir, "weinstock", args.out_dir, thresh)
        summaries.append(s)

    if not summaries:
        print("No data directories specified. Use --file, --cgmacros_dir, or --weinstock_dir.")
        return

    combined = pd.concat(summaries, ignore_index=True)

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.summary_out, index=False)
    print(f"\nSummary saved → {args.summary_out}")

    # Print cross-dataset comparison
    print("\n=== Mean regime distribution by dataset (%) ===")
    print(
        combined.groupby("dataset")[REGIME_CODES]
        .mean()
        .mul(100)
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
