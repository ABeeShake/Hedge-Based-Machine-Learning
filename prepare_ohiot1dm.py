#!/usr/bin/env python3
"""prepare_ohiot1dm.py -- Convert OhioT1DM XML or tabular CSV files into engine-ready format.

Produces per-subject CSVs (`ohiot1dm/001.csv`..`012.csv`) with canonical columns
`id,Timestamp,Dexcom.GL` (compatible with ExpMethods.timegrid.Grid.from_csv,
ExpMethods.evaluation.forecast_files' 3-char ID convention, and process_results.py)
plus `ohiot1dm/ohiot1dm_manifest.csv` recording source IDs, cohort year, split counts,
and time spans.

Supported raw layouts inside `--input-dir`:
  1. Official OhioT1DM XML release (`5xx-ws-training.xml`, `5xx-ws-testing.xml`,
     including nested `2018/` and `2020/` subdirectories).
  2. Tabular CSV exports (e.g. GlucoBench curated OhioT1DM CSVs) with timestamp
     and glucose columns.

Examples:
  python3 prepare_ohiot1dm.py --input-dir ~/data/OhioT1DM --output-dir ohiot1dm
  python3 prepare_ohiot1dm.py --input-dir ~/data/OhioT1DM --output-dir ohiot1dm --split testing
"""
import argparse
import csv
import datetime as dt
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

OHIO_2018_IDS = {"559", "563", "570", "575", "588", "591"}
OHIO_2020_IDS = {"540", "544", "552", "567", "584", "596"}


def _parse_ohio_timestamp(raw_ts: str) -> dt.datetime:
    """Parse timestamps formatted as DD-MM-YYYY HH:MM:SS (Ohio XML) or ISO-8601 variants."""
    s = raw_ts.strip()
    for fmt in ("%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y %H:%M"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T")).replace(tzinfo=None)


def parse_ohio_xml(xml_path: str):
    """Extract chronological (datetime, glucose_float) pairs from an OhioT1DM XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    gl_node = root.find("glucose_level")
    events = gl_node.findall("event") if gl_node is not None else root.findall(".//glucose_level/event")
    for ev in events:
        ts_raw = ev.attrib.get("ts")
        val_raw = ev.attrib.get("value")
        if ts_raw is None or val_raw is None:
            continue
        try:
            val = float(val_raw)
        except ValueError:
            continue
        if not (0.0 < val < 1000.0):
            continue
        rows.append((_parse_ohio_timestamp(ts_raw), val))
    rows.sort(key=lambda r: r[0])
    return rows


def parse_ohio_csv(csv_path: str):
    """Extract chronological (datetime, glucose_float) pairs from a tabular CGM CSV."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        ts_col = next((c for c in ("Timestamp", "timestamp", "ts", "datetime", "date", "time") if c in fields), None)
        gl_col = next(
            (c for c in ("Dexcom.GL", "Libre.GL", "glucose", "GlucoseValue", "CGM", "gl", "value", "bg") if c in fields),
            None,
        )
        if ts_col is None or gl_col is None:
            raise ValueError(f"Unrecognized CSV columns in {csv_path}: {fields}")
        for r in reader:
            raw_t = (r.get(ts_col) or "").strip()
            raw_g = (r.get(gl_col) or "").strip()
            if not raw_t or not raw_g:
                continue
            try:
                val = float(raw_g)
            except ValueError:
                continue
            if not (0.0 < val < 1000.0):
                continue
            rows.append((_parse_ohio_timestamp(raw_t), val))
    rows.sort(key=lambda r: r[0])
    return rows


def _infer_subject_and_split(path: str):
    """Infer (subject_id, split_role) from filename such as 540-ws-training.xml."""
    base = os.path.basename(path)
    lower = base.lower()
    m = re.match(r"(\d{3})(?:-ws-(training|testing))?", lower)
    if m:
        sid = m.group(1)
        role = m.group(2)
        if role is None:
            if "test" in lower:
                role = "testing"
            elif "train" in lower:
                role = "training"
            else:
                role = "combined"
        return sid, role
    stem = os.path.splitext(base)[0]
    digits = re.findall(r"\d+", stem)
    sid = digits[0] if digits else stem
    role = "testing" if "test" in lower else ("training" if "train" in lower else "combined")
    return sid, role


def discover_ohio_sources(input_dir: str):
    """Find OhioT1DM XML or CSV source files grouped by source subject ID."""
    grouped = defaultdict(dict)
    csv_fallbacks = defaultdict(list)
    for root, _, files in os.walk(input_dir):
        for fn in sorted(files):
            lower = fn.lower()
            full = os.path.join(root, fn)
            if lower.endswith(".xml") and "glucose" not in lower:
                sid, role = _infer_subject_and_split(fn)
                grouped[sid][role] = full
            elif lower.endswith(".csv") and "manifest" not in lower:
                sid, role = _infer_subject_and_split(fn)
                csv_fallbacks[sid].append((role, full))

    if grouped:
        return "xml", dict(sorted(grouped.items()))
    if csv_fallbacks:
        return "csv", {sid: dict(pairs) for sid, pairs in sorted(csv_fallbacks.items())}
    raise FileNotFoundError(f"No OhioT1DM .xml or .csv files found under {input_dir}")


def prepare_ohiot1dm(input_dir: str, output_dir: str = "ohiot1dm", split: str = "combined"):
    """Convert raw OhioT1DM XML/CSV files into canonical engine CSVs and a manifest."""
    fmt, sources = discover_ohio_sources(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []

    for idx, (sid, role_map) in enumerate(sources.items(), start=1):
        pid = f"{idx:03d}"
        selected_roles = (
            [split] if split in role_map
            else ([r for r in ("training", "testing", "combined") if r in role_map] if split == "combined" else [])
        )
        if not selected_roles:
            continue

        merged = {}
        used_files = []
        # Parse training before testing so testing overwrites on rare boundary duplicates
        for role in selected_roles:
            src_path = role_map[role]
            used_files.append(os.path.basename(src_path))
            pts = parse_ohio_xml(src_path) if fmt == "xml" else parse_ohio_csv(src_path)
            for t, val in pts:
                merged[t] = val

        if not merged:
            continue
        series = sorted(merged.items(), key=lambda kv: kv[0])
        out_csv = os.path.join(output_dir, f"{pid}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "Timestamp", "Dexcom.GL"])
            for t, val in series:
                w.writerow([pid, t.strftime("%Y-%m-%d %H:%M:%S"), f"{val:.1f}"])

        year = "2018" if sid in OHIO_2018_IDS else ("2020" if sid in OHIO_2020_IDS else "unknown")
        manifest_rows.append({
            "pid": pid,
            "source_id": sid,
            "cohort_year": year,
            "split_mode": split,
            "n_points": len(series),
            "start_ts": series[0][0].strftime("%Y-%m-%d %H:%M:%S"),
            "end_ts": series[-1][0].strftime("%Y-%m-%d %H:%M:%S"),
            "source_files": ";".join(used_files),
        })

    manifest_path = os.path.join(output_dir, "ohiot1dm_manifest.csv")
    if manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description="Prepare OhioT1DM XML/CSV files for HBML Engine v2.3.")
    parser.add_argument("--input-dir", required=True, help="Directory containing OhioT1DM XML or CSV files.")
    parser.add_argument("--output-dir", default="ohiot1dm", help="Output dataset directory (default: ohiot1dm).")
    parser.add_argument("--split", choices=["combined", "testing", "training"], default="combined",
                        help="Which split(s) to export (default: combined chronological stream).")
    args = parser.parse_args()

    rows = prepare_ohiot1dm(args.input_dir, args.output_dir, split=args.split)
    print(f"Prepared {len(rows)} OhioT1DM participant CSVs in {args.output_dir}/")
    for r in rows:
        print(f"  {r['pid']} (Ohio {r['source_id']}, {r['cohort_year']}): {r['n_points']} pts "
              f"[{r['start_ts']} .. {r['end_ts']}]")


if __name__ == "__main__":
    main()
