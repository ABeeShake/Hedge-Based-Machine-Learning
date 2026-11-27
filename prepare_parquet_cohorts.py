#!/usr/bin/env python3
"""prepare_parquet_cohorts.py -- Convert harmonized T1D Parquet releases into engine-ready CSVs.

Pre-registered T1D replication (see reviews/preregistration_T1D_replication.md). Converts
AZT1D.parquet, OhioT1DM.parquet and BrisT1D.parquet (same 39-column harmonized schema:
id, date [ns], CGM [mg/dL], cgm_device, insulin_delivery_modality, ...) into the per-participant
layout read by ExpMethods.timegrid.Grid.from_csv:

    <dataset>/<pid>.csv          columns id,Timestamp,Dexcom.GL   (pid = 001, 002, ...)
    <dataset>/<dataset>_manifest.csv

Rules (fixed before any forecasts were run on these cohorts):
  * every row with a non-missing CGM value is kept, in time order, for the whole record; the
    benchmark train/test labels (is_test, subject_split_across_traintest) are ignored, because the
    online protocol scores each participant's full record causally;
  * glucose is clipped to the CGM sensor range [40, 400] mg/dL (counts recorded in the manifest);
  * exact duplicate timestamps keep the last value (the engine also drops readings < 2.5 min apart);
  * participants are numbered 001.. in ascending order of their source id (numeric when possible);
  * sampling stratum: a participant whose median interval between consecutive readings exceeds
    10 minutes is a 15-minute participant. For BrisT1D these participants are written to the
    separate sensitivity dataset `brist1d_15min`; all others go to `brist1d`. AZT1D and OhioT1DM
    contain no 15-minute participants (the script stops if one appears).

Usage:
  python3 prepare_parquet_cohorts.py                     # all three cohorts, pandas/pyarrow reader
  python3 prepare_parquet_cohorts.py --reader pure       # dependency-free fallback reader
  python3 prepare_parquet_cohorts.py --cohorts azt1d
"""
import argparse
import csv
import datetime as dt
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP = (40.0, 400.0)
FIFTEEN_MIN_THRESHOLD = 10.0  # minutes; median inter-reading interval above this -> 15-min stratum
COLUMNS = ["id", "date", "CGM", "cgm_device", "insulin_delivery_device", "insulin_delivery_modality"]

COHORTS = {
    "azt1d": dict(file="AZT1D.parquet", source="AZT1D (harmonized Parquet release)", split_15min=None),
    "ohiot1dm": dict(file="OhioT1DM.parquet", source="OhioT1DM (Marling & Bunescu 2020; harmonized Parquet release)",
                     split_15min=None),
    "brist1d": dict(file="BrisT1D.parquet", source="BrisT1D (harmonized Parquet release)",
                    split_15min="brist1d_15min"),
}


def read_columns(path, reader):
    if reader == "pandas":
        import pandas as pd
        df = pd.read_parquet(path, columns=COLUMNS)
        out = {c: df[c].tolist() for c in COLUMNS}
        if np.issubdtype(df["date"].dtype, np.datetime64):
            out["date"] = df["date"].values.astype("datetime64[ns]").astype("int64").tolist()
        out["CGM"] = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in out["CGM"]]
        for c in COLUMNS[3:]:
            out[c] = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in out[c]]
        return out
    sys.path.insert(0, os.path.join(BASE_DIR, "tools"))
    from parquet_pure import read_table
    return read_table(path, COLUMNS)


def _sort_key(sid):
    s = str(sid)
    digits = "".join(ch for ch in s if ch.isdigit())
    return (0, int(digits), s) if digits else (1, 0, s)


def build_series(cols):
    """Group rows by source id -> sorted (ns_timestamp, glucose) arrays plus metadata sets."""
    ids = np.array(cols["id"], dtype=object)
    t = np.array(cols["date"], dtype="int64")
    g = np.array([np.nan if v is None else v for v in cols["CGM"]], dtype=float)
    meta_cols = {c: np.array(cols[c], dtype=object)
                 for c in ("cgm_device", "insulin_delivery_device", "insulin_delivery_modality")}
    out = {}
    for sid in sorted(set(ids.tolist()), key=_sort_key):
        m = ids == sid
        tt, gg = t[m], g[m]
        keep = np.isfinite(gg)
        tt, gg = tt[keep], gg[keep]
        order = np.argsort(tt, kind="stable")
        tt, gg = tt[order], gg[order]
        if len(tt) == 0:
            continue
        # exact duplicate timestamps: keep the last value
        last = np.r_[tt[1:] != tt[:-1], True]
        tt, gg = tt[last], gg[last]
        meta = {c: sorted({str(v) for v in arr[m].tolist() if v is not None}) for c, arr in meta_cols.items()}
        out[sid] = (tt, gg, meta)
    return out


def write_dataset(dataset, items, source, out_root):
    out_dir = os.path.join(out_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for idx, (sid, (tt, gg, meta, med_step)) in enumerate(items, start=1):
        pid = f"{idx:03d}"
        n_low, n_high = int((gg < CLIP[0]).sum()), int((gg > CLIP[1]).sum())
        gc = np.clip(gg, *CLIP)
        stamps = tt.astype("datetime64[ns]").astype("datetime64[s]").astype(dt.datetime)
        with open(os.path.join(out_dir, f"{pid}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "Timestamp", "Dexcom.GL"])
            for ts, val in zip(stamps, gc):
                w.writerow([pid, ts.strftime("%Y-%m-%d %H:%M:%S"), f"{val:.1f}"])
        rows.append({
            "pid": pid, "source_id": sid, "diagnosis": "T1D", "n_points": len(tt),
            "start_ts": stamps[0].strftime("%Y-%m-%d %H:%M:%S"),
            "end_ts": stamps[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "span_days": round((tt[-1] - tt[0]) / 8.64e13, 2),
            "median_step_min": round(med_step, 2),
            "n_clipped_low": n_low, "n_clipped_high": n_high,
            "cgm_device": ";".join(meta["cgm_device"]),
            "insulin_delivery_device": ";".join(meta["insulin_delivery_device"]),
            "insulin_delivery_modality": ";".join(meta["insulin_delivery_modality"]),
            "source": source,
        })
    with open(os.path.join(out_dir, f"{dataset}_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{dataset}: {len(rows)} participants -> {out_dir}")
    return rows


def prepare(cohort, reader, in_dir, out_root):
    spec = COHORTS[cohort]
    cols = read_columns(os.path.join(in_dir, spec["file"]), reader)
    series = build_series(cols)
    primary, fifteen = [], []
    for sid, (tt, gg, meta) in series.items():
        med = float(np.median(np.diff(tt)) / 6e10) if len(tt) > 1 else float("nan")
        (fifteen if med > FIFTEEN_MIN_THRESHOLD else primary).append((sid, (tt, gg, meta, med)))
    if fifteen and spec["split_15min"] is None:
        raise SystemExit(f"{cohort}: unexpected 15-minute participants {[s for s, _ in fifteen]}")
    write_dataset(cohort, primary, spec["source"], out_root)
    if fifteen:
        write_dataset(spec["split_15min"], fifteen, spec["source"] + "; 15-minute sensitivity stratum", out_root)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohorts", default="azt1d,ohiot1dm,brist1d")
    ap.add_argument("--reader", choices=["pandas", "pure"], default="pandas")
    ap.add_argument("--input-dir", default=BASE_DIR, help="Directory holding the .parquet files.")
    ap.add_argument("--output-root", default=BASE_DIR)
    a = ap.parse_args()
    for c in [x.strip() for x in a.cohorts.split(",") if x.strip()]:
        prepare(c, a.reader, a.input_dir, a.output_root)


if __name__ == "__main__":
    main()
