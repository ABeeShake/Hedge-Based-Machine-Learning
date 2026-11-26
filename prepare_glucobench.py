#!/usr/bin/env python3
"""prepare_glucobench.py -- Fetch and convert public GlucoBench datasets for Engine v2.3.

Downloads `raw_data.zip` from the official IrinaStatsLab/GlucoBench GitHub repository
(or reads a local `--zip-path`) and exports canonical per-participant 5-minute-ready CSVs
(`id,Timestamp,Dexcom.GL`) plus `<dataset>/<dataset>_manifest.csv`.

Supported datasets inside GlucoBench `raw_data.zip`:
  - hall     : Hall et al. 2018 (PLOS Biology, CC BY 4.0) -- 57 adults (normoglycaemic,
               pre-diabetic, and T2D; 1,584-2,361 Dexcom G4 readings each, 4.2% < 70 mg/dL).
               Primary external cohort: all 57 subjects exceed burn-in + 288 scored points.
  - dubosson : Dubosson et al. 2018 / D1NAMO (CC BY-SA 4.0) -- 9 T1D adults (7 with 909-1,413
               readings >= 3.1 days; 7.4% < 70 mg/dL, 35.8% > 180 mg/dL).

(GlucoBench also contains `colas` (<= 576 points/subject, <= 48h, too short for 24h burn-in
 + 288 scored points), `iglu` (5 subjects, 0.2% < 70 mg/dL), and `weinstock` (already
 in the primary evaluation).)

Examples:
  python3 prepare_glucobench.py --dataset hall
  python3 prepare_glucobench.py --dataset hall,dubosson
"""
import argparse
import csv
import datetime as dt
import io
import os
import urllib.request
import zipfile
from collections import defaultdict

GLUCOBENCH_ZIP_URL = "https://github.com/IrinaStatsLab/GlucoBench/raw/main/raw_data.zip"

HALL_DIAGNOSIS_MAP = {
    "0": "normoglycemic",
    "1": "pre-diabetic",
    "2": "T2D",
}

DATASET_SPECS = {
    "hall": {
        "zip_member": "raw_data/hall.csv",
        "diabetes_default": "mixed",
        "source_label": "Hall et al. 2018 (CC BY 4.0)",
    },
    "dubosson": {
        "zip_member": "raw_data/dubosson.csv",
        "diabetes_default": "T1D",
        "source_label": "Dubosson et al. 2018 / D1NAMO (CC BY-SA 4.0)",
    },
}


def _parse_ts(raw_ts: str) -> dt.datetime:
    s = raw_ts.strip().replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return dt.datetime.fromisoformat(s).replace(tzinfo=None)


def load_zip_bytes(zip_path: str = None, url: str = GLUCOBENCH_ZIP_URL) -> bytes:
    """Read raw_data.zip bytes from a local file or download from GlucoBench GitHub."""
    if zip_path:
        with open(zip_path, "rb") as f:
            return f.read()
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def export_glucobench_dataset(zf: zipfile.ZipFile, dataset: str, output_root: str = "."):
    """Convert one dataset inside an open GlucoBench ZipFile into <dataset>/001.csv..."""
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported GlucoBench dataset {dataset!r}; choose from {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[dataset]
    member = spec["zip_member"]
    if member not in zf.namelist():
        raise FileNotFoundError(f"{member} not found in archive")

    by_id = defaultdict(dict)
    diag_by_id = {}
    with io.TextIOWrapper(zf.open(member), encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("id") or "").strip()
            raw_gl = (row.get("gl") or "").strip()
            raw_t = (row.get("time") or "").strip()
            if not sid or not raw_gl or not raw_t:
                continue
            try:
                val = float(raw_gl)
            except ValueError:
                continue
            if not (0.0 < val < 1000.0):
                continue
            t = _parse_ts(raw_t)
            by_id[sid][t] = val
            if "diagnosis" in row and (row["diagnosis"] or "").strip():
                diag_by_id[sid] = HALL_DIAGNOSIS_MAP.get(row["diagnosis"].strip(), row["diagnosis"].strip())

    out_dir = os.path.join(output_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    manifest_rows = []
    for idx, sid in enumerate(sorted(by_id.keys()), start=1):
        pid = f"{idx:03d}"
        series = sorted(by_id[sid].items(), key=lambda kv: kv[0])
        out_csv = os.path.join(out_dir, f"{pid}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "Timestamp", "Dexcom.GL"])
            for t, val in series:
                w.writerow([pid, t.strftime("%Y-%m-%d %H:%M:%S"), f"{val:.1f}"])
        manifest_rows.append({
            "pid": pid,
            "source_id": sid,
            "diagnosis": diag_by_id.get(sid, spec["diabetes_default"]),
            "n_points": len(series),
            "start_ts": series[0][0].strftime("%Y-%m-%d %H:%M:%S"),
            "end_ts": series[-1][0].strftime("%Y-%m-%d %H:%M:%S"),
            "source": spec["source_label"],
        })

    manifest_path = os.path.join(out_dir, f"{dataset}_manifest.csv")
    if manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
    return manifest_rows


def prepare_glucobench(datasets, zip_path: str = None, output_root: str = "."):
    """Download/open GlucoBench raw_data.zip and export all requested datasets."""
    raw_bytes = load_zip_bytes(zip_path=zip_path)
    results = {}
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        for ds in datasets:
            results[ds] = export_glucobench_dataset(zf, ds, output_root=output_root)
    return results


def main():
    parser = argparse.ArgumentParser(description="Download and prepare GlucoBench external CGM cohorts.")
    parser.add_argument("--dataset", default="hall",
                        help="Comma-separated datasets from {hall, dubosson} (default: hall).")
    parser.add_argument("--zip-path", default=None, help="Optional local path to GlucoBench raw_data.zip.")
    parser.add_argument("--output-root", default=".", help="Workspace root where <dataset>/ will be written.")
    args = parser.parse_args()

    datasets = [d.strip().lower() for d in args.dataset.split(",") if d.strip()]
    res = prepare_glucobench(datasets, zip_path=args.zip_path, output_root=args.output_root)
    for ds, rows in res.items():
        pts = [r["n_points"] for r in rows]
        print(f"Prepared {len(rows)} subjects in {os.path.join(args.output_root, ds)}/ "
              f"(points min/max = {min(pts)}/{max(pts)})")


if __name__ == "__main__":
    main()
