#!/usr/bin/env python3
"""analyze_segments.py -- the 5-minute grid: gaps, segments and how many forecasts survive.

Runs locally from the raw participant files (no simulation outputs needed):
    python3 analyze_segments.py                       # thresholds 30 and 60 min, both datasets
For each dataset and gap threshold it reports
  * timestamp jitter / duplicates removed by snapping, filled and long-gap slots;
  * segment counts and lengths;
  * scorable forecasts (real reading at the origin and at the target, past the burn-in) and the share
    kept by the engine's rule (origin >= `min_segment_history` slots into its segment);
  * the alternative we did not adopt -- windows confined to one segment -- for N-HITS input sizes;
  * among kept origins, how often an N-HITS input window is mostly bridged (lies in long gaps).
Outputs: Outputs/segments/{segment_lengths,coverage}.csv and overleaf/tables/grid_summary.csv
(read by generate_results_macros.py -> \\Grid... macros).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from ExpMethods.globals import GlobalValues as G
from ExpMethods.timegrid import Grid, read_raw, snap

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, "overleaf", "tables")
HORIZONS = {"half": 6, "2": 24, "5": 60}
SEG_LOCAL = [72, 120, 144, 288]      # N-HITS input sizes: 6 h, 10 h (full history), 12 h, 24 h


def files(ds):
    return sorted(f for f in glob.glob(os.path.join(BASE, ds, "*.csv")) if os.path.basename(f)[0].isdigit())


def one(path, thr):
    ts, g = read_raw(path)
    slot, gs, nd = snap(ts, g, G.grid_step_min)
    dt = np.diff(ts) / np.timedelta64(1, "s") / 60.0
    # irregular sampling: consecutive readings closer than 7.5 min but not 5 +/- 0.5 min apart
    irregular = int(((dt < 1.5 * G.grid_step_min) & (np.abs(dt - G.grid_step_min) > 0.5)).sum())
    grid = Grid(slot, gs, ts[0], thr, G.grid_step_min, nd)
    T = len(grid)
    pid = os.path.basename(path)[:-4]
    bounds = grid.segment_bounds()
    segs = [dict(id=pid, seg=i, start=a, slots=b - a, n_obs=int(grid.observed[a:b].sum())) for i, (a, b) in enumerate(bounds)]
    bad = np.r_[0, np.cumsum(~grid.valid)]
    rec = dict(id=pid, n_readings=len(ts), n_irregular=irregular, n_duplicates=nd,
               n_slots=T, n_observed=int(grid.observed.sum()), n_filled=int(grid.filled.sum()),
               n_long_gap=int((~grid.valid).sum()), n_segments=len(bounds))
    first = grid.first_origin(G.burn_in_steps)
    for hk, h in HORIZONS.items():
        t = np.arange(first, max(T - h, first))
        cand = t[grid.observed[t] & grid.observed[t + h]]
        have = cand - grid.segment_start[cand] + 1
        kept = cand[have >= G.min_segment_history]
        rec[f"cand_{hk}"], rec[f"kept_{hk}"] = len(cand), len(kept)
        for L in SEG_LOCAL:
            rec[f"seglocal{L}_{hk}"] = int((have >= L).sum())
            lo = np.maximum(kept - L + 1, 0)
            frac = (bad[kept + 1] - bad[lo]) / L
            rec[f"bridged50_{L}_{hk}"] = int((frac > 0.5).sum())
            rec[f"bridged_mean_{L}_{hk}"] = float(frac.sum())
    return rec, segs


def summarize(per, seg, ds, thr):
    s = per.sum(numeric_only=True)
    hrs = seg.slots * G.grid_step_min / 60
    nseg = per.n_segments
    out = dict(dataset=ds, max_gap_min=thr, warmup_min=G.min_segment_history * G.grid_step_min,
               n_participants=len(per), n_readings=int(s.n_readings),
               irregular_pct=100 * s.n_irregular / max(s.n_readings - len(per), 1), n_duplicates=int(s.n_duplicates),
               filled_pct=100 * s.n_filled / s.n_slots, long_gap_pct=100 * s.n_long_gap / s.n_slots,
               missing_pct=100 * (1 - s.n_observed / s.n_slots),
               segments_median=nseg.median(), segments_q1=nseg.quantile(.25), segments_q3=nseg.quantile(.75),
               segment_hours_median=hrs.median(), segment_hours_q1=hrs.quantile(.25), segment_hours_q3=hrs.quantile(.75),
               time_in_segments_ge_10h_pct=100 * seg.slots[hrs >= 10].sum() / seg.slots.sum())
    for hk in HORIZONS:
        out[f"kept_pct_{hk}"] = 100 * s[f"kept_{hk}"] / s[f"cand_{hk}"]
        out[f"n_kept_{hk}"] = int(s[f"kept_{hk}"])
        for L in SEG_LOCAL:
            out[f"seglocal{L}_pct_{hk}"] = 100 * s[f"seglocal{L}_{hk}"] / s[f"cand_{hk}"]
            out[f"bridged50_{L}_pct_{hk}"] = 100 * s[f"bridged50_{L}_{hk}"] / s[f"kept_{hk}"]
            out[f"bridged_mean_{L}_pct_{hk}"] = 100 * s[f"bridged_mean_{L}_{hk}"] / s[f"kept_{hk}"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["weinstock", "cgmacros"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=sorted({G.max_gap_min, 60.0}))
    a = ap.parse_args()
    out_dir = os.path.join(BASE, "Outputs", "segments")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    rows, all_per, all_seg = [], [], []
    for ds in a.datasets:
        fl = files(ds)
        for thr in a.thresholds:
            per, seg = [], []
            for f in fl:
                try:
                    r, sg = one(f, thr)
                except OSError as e:
                    print(f"  skipped {f}: {e}")
                    continue
                per.append(r)
                seg += sg
            if not per:
                continue
            per, seg = pd.DataFrame(per), pd.DataFrame(seg)
            per["dataset"] = seg["dataset"] = ds
            per["max_gap_min"] = seg["max_gap_min"] = thr
            all_per.append(per)
            all_seg.append(seg)
            rows.append(summarize(per, seg, ds, thr))
    if not rows:
        raise SystemExit("no participant files could be read")
    pd.concat(all_per).to_csv(os.path.join(out_dir, "coverage.csv"), index=False)
    pd.concat(all_seg).to_csv(os.path.join(out_dir, "segment_lengths.csv"), index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(TABLES, "grid_summary.csv"), index=False)
    for r in rows:
        print(f"\n== {r['dataset']} ({r['n_participants']} participants), gaps <= {r['max_gap_min']:.0f} min filled, "
              f"{r['warmup_min']:.0f}-min warm-up after long gaps ==")
        print(f"readings {r['n_readings']:,}; irregular intervals (not 5 +/- 0.5 min) {r['irregular_pct']:.2f}%; "
              f"readings dropped as a second reading in one slot {r['n_duplicates']}")
        print(f"slots missing {r['missing_pct']:.1f}% (filled {r['filled_pct']:.1f}%, long gaps {r['long_gap_pct']:.1f}%)")
        print(f"segments per participant median {r['segments_median']:.0f} (IQR {r['segments_q1']:.0f}-{r['segments_q3']:.0f}); "
              f"segment length median {r['segment_hours_median']:.1f} h (IQR {r['segment_hours_q1']:.1f}-{r['segment_hours_q3']:.1f}); "
              f"{r['time_in_segments_ge_10h_pct']:.1f}% of segment time in segments >= 10 h")
        for hk, h in HORIZONS.items():
            print(f"  h={h:>2}: forecasts kept {r[f'kept_pct_{hk}']:.1f}% ({r[f'n_kept_{hk}']:,}); "
                  "segment-local windows would keep " +
                  ", ".join(f"L{L}: {r[f'seglocal{L}_pct_{hk}']:.1f}%" for L in SEG_LOCAL) +
                  "; kept origins with >50% bridged window " +
                  ", ".join(f"L{L}: {r[f'bridged50_{L}_pct_{hk}']:.1f}%" for L in SEG_LOCAL))


if __name__ == "__main__":
    main()
