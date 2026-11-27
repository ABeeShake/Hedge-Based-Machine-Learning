#!/usr/bin/env python3
"""analyze_transported_weights.py -- can fixed convex weights replace online HBML? (round 5, Major 3)

Fixed simplex weights fitted by 5-fold CV within a cohort beat online HBML in every development
cell (Table S17). The manuscript argued such weights need labelled participants from the target
population. This script tests that directly on cached Engine v2.3 forecasts (full-history context):

  Transported-<src>   weights fitted once on ALL scored rows of source cohort <src>, applied unchanged
                      to every participant of a different target cohort (zero target labels)
  Personal-burnin     each participant's own weights, fitted on their first `--fit_targets` scored
                      targets (default 288 = one day) and then frozen
  HBML / Uniform      references

Every variant is scored on the same rows: scored targets AFTER each participant's fitting window,
so the personal variant is never scored on the data it was fitted on. Participants with fewer than
`--fit_targets` + 288 scored targets are skipped (reported as n).

Outputs (overleaf/tables/):
  transported_weights_summary.csv   target x horizon x variant: n, mean RMSE, paired diff vs HBML
                                    with participant bootstrap 95% CI
  transported_weights_fitted.csv    the fitted source weights per cohort x horizon

Usage: python analyze_transported_weights.py [--cohorts weinstock,cgmacros,hall,dubosson]
Set HBML_EVAL_ENGINE to evaluate forecasts written by another engine version.
"""
import argparse
import csv
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ExpMethods import evaluation as ev  # noqa: E402
import evaluate_clinical_and_weights as ecw  # noqa: E402
from analyze_pool_mechanism import TABLES_DIR, boot_ci, hbml_rmse, load_config, simplex_ls  # noqa: E402


def post_fit_mask(mask, n_fit):
    """Scored rows after the participant's first n_fit scored targets."""
    idx = np.flatnonzero(mask)
    out = np.zeros_like(mask)
    if len(idx) > n_fit:
        out[idx[n_fit:]] = True
    return out, (idx[:n_fit] if len(idx) >= n_fit else idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="weinstock,cgmacros,hall,dubosson")
    ap.add_argument("--fit_targets", type=int, default=288)
    ap.add_argument("--min_eval", type=int, default=288)
    ap.add_argument("--n_boot", type=int, default=5000)
    a = ap.parse_args()
    cohorts = [c.strip() for c in a.cohorts.split(",") if c.strip()]
    retained = ecw.load_retained_ids()
    summ, fitted = [], []
    for h_str, h_disp, h in ecw.HORIZONS:
        data = {}
        for ds in cohorts:
            parts = load_config(ds, h_str, h_disp, "full", retained.get((ds, h_disp, "full")))
            if parts:
                data[ds] = parts
        if not data:
            continue
        cols = next(iter(data.values()))[0]["cols"]
        for ds, parts in data.items():
            if any(p["cols"] != cols for p in parts):
                raise SystemExit(f"Expert columns differ in {ds}; align them before transporting weights.")
        # weights fitted on every scored row of each source cohort
        w_src = {}
        for src, parts in data.items():
            F = np.vstack([p["F"][p["mask"]] for p in parts])
            y = np.concatenate([p["y"][p["mask"]] for p in parts])
            w_src[src] = simplex_ls(F, y)
            fitted.append(dict(source=src, horizon=h_disp, **{c: float(v) for c, v in zip(cols, w_src[src])}))
        for tgt, parts in data.items():
            per = {}
            n_skip = 0
            for p in parts:
                m_eval, fit_idx = post_fit_mask(p["mask"], a.fit_targets)
                if m_eval.sum() < a.min_eval:
                    n_skip += 1
                    continue
                F, y = p["F"], p["y"]
                res = {"HBML": hbml_rmse(F, y, m_eval, h),
                       "Uniform": ev.rmse((F.mean(axis=1) - y) ** 2, m_eval)}
                w_own = simplex_ls(F[fit_idx], y[fit_idx])
                res["Personal-burnin"] = ev.rmse((F @ w_own - y) ** 2, m_eval)
                for src, w in w_src.items():
                    if src != tgt:
                        res[f"Transported-{src}"] = ev.rmse((F @ w - y) ** 2, m_eval)
                for v, r in res.items():
                    per.setdefault(v, {})[p["pid"]] = r
            if "HBML" not in per:
                continue
            base = per["HBML"]
            print(f"h={h_disp} target={tgt}: {len(base)} participants evaluated, {n_skip} too short", flush=True)
            for v, d in per.items():
                ids = sorted(set(d) & set(base))
                diff = np.array([d[i] - base[i] for i in ids])     # negative = variant better than HBML
                lo, hi = boot_ci(diff, a.n_boot)
                summ.append(dict(target=tgt, horizon=h_disp, variant=v, n=len(ids), n_skipped=n_skip,
                                 mean_rmse=float(np.mean([d[i] for i in ids])), diff_vs_hbml=float(diff.mean()),
                                 diff_ci95_low=lo, diff_ci95_high=hi,
                                 n_better_than_hbml=int((diff < 0).sum())))
    if not summ:
        raise SystemExit(f"No engine-{ecw.ENGINE_VERSION} forecasts found under Outputs/.")
    os.makedirs(TABLES_DIR, exist_ok=True)
    for name, rows in (("transported_weights_summary.csv", summ), ("transported_weights_fitted.csv", fitted)):
        with open(os.path.join(TABLES_DIR, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("Wrote transported_weights_summary.csv and transported_weights_fitted.csv")


if __name__ == "__main__":
    main()
