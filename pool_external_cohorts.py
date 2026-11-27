#!/usr/bin/env python3
"""pool_external_cohorts.py -- Pre-registered pooled T1D replication analysis.

See reviews/preregistration_T1D_replication.md (analysis P2). Reads the per-cohort outputs written
by the locked evaluate_external_cohort.py and, for each of the 12 horizon x context configurations,
pools participants across the primary T1D replication cohorts:

    d_i = RMSE_i(best single expert of participant i's own cohort) - RMSE_i(HBML)

where each cohort's best single expert is the one recorded in that cohort's rmse_summary.csv (lowest
cohort-mean RMSE in that configuration, exactly as in the locked per-cohort analysis). The pooled
differences go through calculate_statistics.paired_stats unchanged (participant bootstrap TOST
within +/-1 mg/dL as the primary estimand, BCa 95% CI of the mean, Wilcoxon, Hodges-Lehmann).
Benjamini-Hochberg adjustment is applied to the 12 pooled Wilcoxon p-values.

Only the columns HBML and the named best expert are read; the exploratory variant columns that the
locked script also writes (HBML_Interleaved, HBML_ShortCtx, HBML_NoXGBNODE, FTL, Uniform,
HBML_SFHDF) are never used.

Usage:
  python3 pool_external_cohorts.py                    # azt1d, ohiot1dm, brist1d
  python3 pool_external_cohorts.py --results-root external_results --out external_results/pooled_t1d
"""
import argparse
import csv
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import calculate_statistics as cstat  # noqa: E402

PRIMARY_COHORTS = ["azt1d", "ohiot1dm", "brist1d"]
HORIZONS = ["0.5", "2", "5"]
CONTEXTS = ["6", "12", "24", "full"]


def _h(v):
    s = str(v).strip()
    return "0.5" if s in ("0.5", ".5") else s.replace(".0", "")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def classify(st):
    """Pre-registered three-way verdict for one configuration."""
    if st["equiv_boot"]:
        return "equivalent"
    if np.isfinite(st["ci95_mean_high"]) and st["ci95_mean_high"] < 0:
        return "HBML worse"
    if np.isfinite(st["ci95_mean_low"]) and st["ci95_mean_low"] > 0:
        return "HBML better"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohorts", default=",".join(PRIMARY_COHORTS))
    ap.add_argument("--results-root", default=os.path.join(BASE_DIR, "external_results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    cohorts = [c.strip() for c in a.cohorts.split(",") if c.strip()]
    out_dir = a.out or os.path.join(a.results_root, "pooled_t1d")
    os.makedirs(out_dir, exist_ok=True)

    part = {c: read_csv(os.path.join(a.results_root, c, "rmse_participant.csv")) for c in cohorts}
    summ = {c: read_csv(os.path.join(a.results_root, c, "rmse_summary.csv")) for c in cohorts}

    rows = []
    for h in HORIZONS:
        for ctx in CONTEXTS:
            ex, hb, n_by = [], [], {}
            best_by = {}
            for c in cohorts:
                srow = [s for s in summ[c] if _h(s["horizon"]) == h and s["context"] == ctx]
                if not srow:
                    raise SystemExit(f"{c}: no summary row for h={h} ctx={ctx}")
                best = srow[0]["best_expert"]
                best_by[c] = best
                prows = [r for r in part[c] if _h(r["horizon"]) == h and r["context"] == ctx]
                n_by[c] = len(prows)
                ex += [float(r[best]) for r in prows]
                hb += [float(r["HBML"]) for r in prows]
            st = cstat.paired_stats(np.array(ex), np.array(hb), n_boot=a.n_boot)
            row = {"horizon": h, "context": ctx}
            row.update({f"n_{c}": n_by[c] for c in cohorts})
            row.update({f"best_expert_{c}": best_by[c] for c in cohorts})
            row.update(st)
            row["verdict"] = classify(st)
            rows.append(row)

    p_adj = cstat.bh_adjust(np.array([r["wilcoxon_p"] for r in rows], dtype=float))
    for r, q in zip(rows, p_adj):
        r["wilcoxon_p_bh"] = float(q)

    path = os.path.join(out_dir, "pooled_statistical_tests.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("equivalent", "HBML worse", "HBML better", "inconclusive")}
    print(f"Wrote {path}")
    print("Pooled verdicts over 12 configurations:", counts)


if __name__ == "__main__":
    main()
