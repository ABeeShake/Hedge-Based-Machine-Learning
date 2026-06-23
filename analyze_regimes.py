#!/usr/bin/env python3
"""
analyze_regimes.py
==================
Assess HBML forecasting effectiveness broken down by dynamic-state regime.

Requires
--------
1. Per-patient regime files produced by compute_regimes.py
   (e.g. Outputs/regimes/<dataset>/<patient_id>_regimes.csv)
   Columns: Timestamp, glucose, slope, rolling_var, regime

2. Per-patient full-forecast files produced by process_results.py
   (e.g. <dataset_dir>/h-<horizon>/context-<context>/forecasts/<id>_fullforecasts_advanced.csv)
   Columns: one per model (ARIMA, ETS, NHITS, NODE, XGBoost, HBML-SFHDF, …)
   Row order aligns chronologically with the trimmed glucose series.

3. Per-patient raw glucose CSV files
   (e.g. weinstock/001.csv)
   Columns include Timestamp, Libre.GL or Dexcom.GL

Outputs (saved to --out_dir)
-------------------------------
  regime_rmse_<dataset>_h<h>_c<c>.csv      — per-patient + aggregate RMSE by regime
  regime_ceg_<dataset>_h<h>_c<c>.csv       — per-patient + aggregate CEG zones by regime
  regime_best_expert_<dataset>_h<h>_c<c>.csv — best-expert frequency by regime

Usage
-----
    python analyze_regimes.py \\
        --dataset weinstock \\
        --data_dir ./weinstock \\
        --regimes_dir ./Outputs/regimes/weinstock \\
        --forecast_root ./weinstock \\
        --horizon 0.5 \\
        --context 6 \\
        --out_dir ./Outputs/regime_analysis
"""

import os
import sys
import argparse
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# CEG zone classification  (extracted from ExpMethods/visualizations.py)
# ---------------------------------------------------------------------------

def classify_ceg_zone(r: float, p: float) -> str:
    """Classify a single (reference, prediction) pair into a Clarke Error Grid zone."""
    if np.isnan(r) or np.isnan(p):
        return "unknown"
    if (abs(r - p) <= 0.2 * r) or (r < 70 and p < 70):
        return "A"
    if r <= 70 and p >= 180:
        return "E1"
    if r >= 180 and p <= 70:
        return "E2"
    if ((r >= 70 and r <= 290) and p >= r + 110) or \
       ((r >= 130 and r <= 180) and (p <= (7 / 5) * r - 182)):
        return "C"
    if r <= 70 and r > 0 and (70 <= p <= 180):
        return "D1"
    if r >= 240 and (70 <= p <= 180):
        return "D2"
    return "B"


CEG_ZONES   = ["A", "B", "C", "D1", "D2", "E1", "E2"]
REGIME_CODES = ["HYPO_TRANS", "RECOVERY", "HIGHVAR", "RISING", "FALLING", "STABLE"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_glucose(data_dir: str, patient_id: str) -> pd.Series:
    """Return a chronologically sorted glucose Series indexed by Timestamp."""
    path = os.path.join(data_dir, f"{patient_id}.csv")
    df   = pd.read_csv(path, parse_dates=["Timestamp"]).sort_values("Timestamp")
    col  = next(
        (c for c in ("Dexcom.GL", "Libre.GL", "CGM", "Glucose") if c in df.columns),
        None,
    )
    if col is None:
        raise ValueError(f"No glucose column found in {path}")
    return df.set_index("Timestamp")[col].astype(float)


def _load_regimes(regimes_dir: str, patient_id: str) -> pd.DataFrame:
    """Load the per-timestep regime file for a patient."""
    path = os.path.join(regimes_dir, f"{patient_id}_regimes.csv")
    df   = pd.read_csv(path, parse_dates=["Timestamp"]).set_index("Timestamp")
    return df[["regime"]]


def _load_forecasts(forecast_root: str, patient_id: str,
                    horizon: str, context: str) -> pd.DataFrame:
    """
    Load full-forecasts CSV.  Tries several naming conventions used
    by process_results.py in order of preference.
    """
    horizon_str  = str(horizon).replace(".", "p")
    horizon_full = f"h-{horizon_str}hr"
    context_full = f"context-{context}hr"

    candidates = [
        # advanced (AdaHedge) variant – preferred
        os.path.join(forecast_root, horizon_full, context_full,
                     "forecasts", f"{patient_id}_fullforecasts_advanced.csv"),
        # adaptive eta variant
        os.path.join(forecast_root, horizon_full, context_full,
                     "forecasts", f"{patient_id}_fullforecasts_adaptive_eta.csv"),
        # fixed-eta variant (fallback)
        os.path.join(forecast_root, horizon_full, context_full,
                     "forecasts", f"{patient_id}_fullforecasts_eta10.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"No forecast file found for patient {patient_id} "
        f"(horizon={horizon}, context={context}) in {forecast_root}"
    )


def _hbml_col(df: pd.DataFrame) -> str:
    """Return the name of the HBML column. Enforces HBML-SFHDF and renames it to HBML."""
    if "HBML-SFHDF" in df.columns:
        df.rename(columns={"HBML-SFHDF": "HBML"}, inplace=True)
        return "HBML"
    elif "HBML" in df.columns:
        return "HBML"
    raise ValueError(f"No HBML-SFHDF column found. Columns: {list(df.columns)}")


def _expert_cols(df: pd.DataFrame, hbml_col: str) -> list:
    """Return the list of non-HBML model columns."""
    skip = {"HBML", "HBML (c)", "HBML (d)", "HBML (fd)", "HBML-SFH",
            "HBML-SFHDF", "FS (Start)", "FS (Uniform)", "FS (Decay)",
            "LSTM", "Autoformer"}
    return [c for c in df.columns if c not in skip]


# ---------------------------------------------------------------------------
# Per-patient analysis
# ---------------------------------------------------------------------------

def analyze_patient(
    patient_id: str,
    glucose: pd.Series,
    regimes: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> dict:
    """
    Align glucose, regimes, and forecasts; compute per-regime metrics.

    Returns a dict with keys:
      rmse   : DataFrame — regime × model RMSE values
      ceg    : DataFrame — regime × (model × zone) CEG fractions
      best_rmse  : DataFrame — regime × model best-RMSE frequency
      best_ab    : DataFrame — regime × model best-A+B% frequency
    """
    hbml = _hbml_col(forecasts)
    experts = _expert_cols(forecasts, hbml)
    all_models = experts + [hbml]

    # ----- Align to forecast length -----
    n = len(forecasts)
    glucose_arr = glucose.values[-n:]
    regime_arr  = regimes["regime"].values[-n:] if len(regimes) >= n else None

    if regime_arr is None or len(regime_arr) != n:
        # Try inner join on position (regime series may have been computed
        # on the full glucose series; trim to last n rows)
        regime_arr = regimes["regime"].iloc[-n:].values

    forecasts = forecasts.reset_index(drop=True)

    # ----- Build a long DataFrame for fast groupby -----
    rows = []
    for t in range(n):
        ref    = glucose_arr[t]
        regime = regime_arr[t] if t < len(regime_arr) else "STABLE"
        if not isinstance(regime, str):
            regime = "STABLE"

        for model in all_models:
            if model not in forecasts.columns:
                continue
            pred = forecasts.at[t, model]
            if pd.isna(ref) or pd.isna(pred):
                continue
            rows.append({
                "t":      t,
                "regime": regime,
                "model":  model,
                "ref":    ref,
                "pred":   pred,
                "se":     (pred - ref) ** 2,
                "zone":   classify_ceg_zone(ref, pred),
            })

    long = pd.DataFrame(rows)
    if long.empty:
        return {"rmse": pd.DataFrame(), "ceg": pd.DataFrame(),
                "best_rmse": pd.DataFrame(), "best_ab": pd.DataFrame()}

    # ---- 1. RMSE per regime per model ----
    rmse_df = (
        long.groupby(["regime", "model"])["se"]
        .apply(lambda x: np.sqrt(x.mean()))
        .rename("rmse")
        .reset_index()
    )
    
    overall_rmse = (
        long.groupby(["model"])["se"]
        .apply(lambda x: np.sqrt(x.mean()))
        .rename("rmse")
        .reset_index()
    )
    overall_rmse["regime"] = "OVERALL"
    rmse_df = pd.concat([rmse_df, overall_rmse], ignore_index=True)

    # ---- 2. CEG zone percentages per regime per model ----
    zone_counts = (
        long.groupby(["regime", "model", "zone"])
        .size()
        .reset_index(name="count")
    )
    zone_totals = long.groupby(["regime", "model"]).size().reset_index(name="total")
    ceg_df = zone_counts.merge(zone_totals, on=["regime", "model"])
    ceg_df["pct"] = ceg_df["count"] / ceg_df["total"]
    # Pivot to wide form: one column per zone
    ceg_wide = ceg_df.pivot_table(
        index=["regime", "model"], columns="zone", values="pct", fill_value=0
    ).reset_index()
    ceg_wide.columns.name = None
    for z in CEG_ZONES:
        if z not in ceg_wide.columns:
            ceg_wide[z] = 0.0
    ceg_wide["AB"] = ceg_wide.get("A", 0) + ceg_wide.get("B", 0)

    # ---- Add OVERALL CEG ----
    overall_counts = (
        long.groupby(["model", "zone"])
        .size()
        .reset_index(name="count")
    )
    overall_totals = long.groupby(["model"]).size().reset_index(name="total")
    overall_ceg = overall_counts.merge(overall_totals, on=["model"])
    overall_ceg["pct"] = overall_ceg["count"] / overall_ceg["total"]
    overall_ceg_wide = overall_ceg.pivot_table(
        index=["model"], columns="zone", values="pct", fill_value=0
    ).reset_index()
    overall_ceg_wide.columns.name = None
    for z in CEG_ZONES:
        if z not in overall_ceg_wide.columns:
            overall_ceg_wide[z] = 0.0
    overall_ceg_wide["AB"] = overall_ceg_wide.get("A", 0) + overall_ceg_wide.get("B", 0)
    overall_ceg_wide["regime"] = "OVERALL"
    
    ceg_wide = pd.concat([ceg_wide, overall_ceg_wide], ignore_index=True)

    # ---- 3. Best-expert frequency per regime ----
    # Give HBML priority in tie-breakers so it gets credit when it copies the best expert
    long["is_hbml"] = (long["model"] == hbml).astype(int)

    # For each (t, regime), which model has the lowest SE? (RMSE proxy)
    best_rmse_model = (
        long.sort_values(["t", "se", "is_hbml"], ascending=[True, True, False])
        .groupby(["t", "regime"])
        .first()
        .reset_index()[["t", "regime", "model"]]
    )
    best_rmse_model.columns = ["t", "regime", "best_rmse_model"]
    best_rmse_freq = (
        best_rmse_model.groupby(["regime", "best_rmse_model"])
        .size()
        .reset_index(name="count")
    )
    best_rmse_total = best_rmse_model.groupby("regime").size().reset_index(name="total")
    best_rmse_freq  = best_rmse_freq.merge(best_rmse_total, on="regime")
    best_rmse_freq["freq"] = best_rmse_freq["count"] / best_rmse_freq["total"]

    # For A+B%: each step, which model has the best CEG zone? Break ties with squared error.
    zone_ranks = {"A": 1, "B": 2, "C": 3, "D1": 4, "D2": 4, "E1": 5, "E2": 5}
    long["zone_rank"] = long["zone"].map(zone_ranks)
    
    best_ab_model = (
        long.sort_values(["t", "zone_rank", "se", "is_hbml"], ascending=[True, True, True, False])
        .groupby(["t", "regime"])
        .first()
        .reset_index()[["t", "regime", "model"]]
    )
    best_ab_model.columns = ["t", "regime", "best_ab_model"]
    best_ab_freq = (
        best_ab_model.groupby(["regime", "best_ab_model"])
        .size()
        .reset_index(name="count")
    )
    best_ab_total = best_ab_model.groupby("regime").size().reset_index(name="total")
    best_ab_freq  = best_ab_freq.merge(best_ab_total, on="regime")
    best_ab_freq["freq"] = best_ab_freq["count"] / best_ab_freq["total"]

    return dict(
        rmse=rmse_df,
        ceg=ceg_wide,
        best_rmse=best_rmse_freq.rename(columns={"best_rmse_model": "model"}),
        best_ab=best_ab_freq.rename(columns={"best_ab_model": "model"}),
    )


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

def aggregate(per_patient: list, key: str) -> pd.DataFrame:
    """
    Concatenate per-patient DataFrames and compute mean ± std across patients
    for each (regime, model) group.
    """
    frames = [d[key] for d in per_patient if d and not d[key].empty]
    if not frames:
        return pd.DataFrame()

    cat = pd.concat(frames, keys=range(len(frames)), names=["patient_idx"])
    cat = cat.reset_index(level="patient_idx")

    # numeric columns to aggregate
    num_cols = [c for c in cat.columns
                if c not in ("patient_idx", "regime", "model", "best_rmse_model",
                             "best_ab_model", "zone", "count", "total")]

    group_cols = [c for c in ("regime", "model") if c in cat.columns]
    agg = cat.groupby(group_cols)[num_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regime-stratified analysis of HBML forecasting performance."
    )
    p.add_argument("--dataset",       required=True,
                   help="Dataset label, e.g. 'weinstock' or 'cgmacros'.")
    p.add_argument("--data_dir",      required=True,
                   help="Directory containing raw patient glucose CSVs.")
    p.add_argument("--regimes_dir",   required=True,
                   help="Directory of per-patient regime CSVs from compute_regimes.py.")
    p.add_argument("--forecast_root", required=True,
                   help="Root directory for forecast outputs (contains h-* subdirs).")
    p.add_argument("--horizon",       default="0.5",
                   help="Forecast horizon string, e.g. '0.5', '2', '5'.")
    p.add_argument("--context",       default="6",
                   help="Context window, e.g. '6', '12', '24', 'full'.")
    p.add_argument("--out_dir",       default="./Outputs/regime_analysis",
                   help="Output directory for results CSVs.")
    p.add_argument("--patient_ids",   nargs="*", default=None,
                   help="Subset of patient IDs to analyze (default: all).")
    return p


def main():
    args = build_parser().parse_args()
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tag = f"{args.dataset}_h{args.horizon}_c{args.context}"

    # Discover patients from regime files
    regime_files = sorted(glob(os.path.join(args.regimes_dir, "*_regimes.csv")))
    patient_ids  = [Path(f).stem.replace("_regimes", "") for f in regime_files]
    if args.patient_ids:
        patient_ids = [p for p in patient_ids if p in args.patient_ids]

    print(f"\nAnalyzing {len(patient_ids)} patients "
          f"[dataset={args.dataset}, h={args.horizon}, ctx={args.context}]")

    per_patient_results = []
    per_patient_rmse    = []
    per_patient_ceg     = []
    per_patient_best_r  = []
    per_patient_best_ab = []

    for pid in patient_ids:
        print(f"  Processing {pid}...", end=" ", flush=True)
        try:
            glucose   = _load_glucose(args.data_dir, pid)
            regimes   = _load_regimes(args.regimes_dir, pid)
            forecasts = _load_forecasts(args.forecast_root, pid,
                                        args.horizon, args.context)
        except FileNotFoundError as e:
            print(f"SKIP ({e})")
            continue
        except Exception as e:
            print(f"ERROR ({e})")
            continue

        result = analyze_patient(pid, glucose, regimes, forecasts)
        print(f"OK — regimes: {sorted(result['rmse']['regime'].unique()) if not result['rmse'].empty else 'none'}")

        # Tag with patient id for per-patient CSVs
        for key, df in result.items():
            if not df.empty:
                df.insert(0, "patient_id", pid)

        per_patient_results.append(result)

        # Accumulate per-patient frames
        if not result["rmse"].empty:
            per_patient_rmse.append(result["rmse"])
        if not result["ceg"].empty:
            per_patient_ceg.append(result["ceg"])
        if not result["best_rmse"].empty:
            per_patient_best_r.append(result["best_rmse"])
        if not result["best_ab"].empty:
            per_patient_best_ab.append(result["best_ab"])

    # ---- Save per-patient detail ----
    def _save_detail(frames, name):
        if frames:
            df = pd.concat(frames, ignore_index=True)
            path = out / f"detail_{name}_{tag}.csv"
            df.to_csv(path, index=False)
            print(f"  Saved per-patient {name} → {path}")
            return df
        return pd.DataFrame()

    detail_rmse    = _save_detail(per_patient_rmse,    "rmse")
    detail_ceg     = _save_detail(per_patient_ceg,     "ceg")
    detail_best_r  = _save_detail(per_patient_best_r,  "best_rmse")
    detail_best_ab = _save_detail(per_patient_best_ab, "best_ab")

    # ---- Aggregate across patients ----
    def _agg_and_save(detail_df, value_cols, group_cols, name):
        if detail_df.empty:
            return
        agg = (
            detail_df.groupby(group_cols)[value_cols]
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        agg.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                       for c in agg.columns]
        path = out / f"aggregate_{name}_{tag}.csv"
        agg.to_csv(path, index=False)
        print(f"  Saved aggregate {name} → {path}")

    print("\nAggregating across patients...")

    if not detail_rmse.empty:
        _agg_and_save(detail_rmse, ["rmse"], ["regime", "model"], "rmse")

    if not detail_ceg.empty:
        ceg_val_cols = [c for c in detail_ceg.columns
                        if c in CEG_ZONES + ["AB"]]
        _agg_and_save(detail_ceg, ceg_val_cols, ["regime", "model"], "ceg")

    if not detail_best_r.empty:
        _agg_and_save(detail_best_r, ["freq"], ["regime", "model"], "best_rmse_freq")

    if not detail_best_ab.empty:
        _agg_and_save(detail_best_ab, ["freq"], ["regime", "model"], "best_ab_freq")

    print(f"\nDone. All outputs in {out}/")


if __name__ == "__main__":
    main()
