#!/usr/bin/env python3
"""analyze_pool_mechanism.py -- why does HBML win or lose? (response to review, point 7)

Runs on the cached engine forecasts only (no re-simulation). For every dataset x horizon x context
and every retained participant it computes the RMSE, on the shared scored rows, of:

  HBML                  primary aggregator (globals.primary_hbml) on the full seven-expert pool
  HBML-minus-<expert>   the same aggregator with one expert removed (leave-one-expert-out)
  HBML-no-reference     the same aggregator without Persistence and RunningMean (pool fixed a priori)
  Uniform               equal weights on the full pool
  FixedConvex-CV        fixed simplex weights fitted by least squares on the *other* participants'
                        scored rows (5-fold participant-level cross-validation), then applied unchanged

Outputs (overleaf/tables/):
  pool_mechanism_participant.csv   one row per participant x configuration x variant
  pool_mechanism_summary.csv       cohort means, paired difference vs HBML with bootstrap 95% CI

Usage:  python analyze_pool_mechanism.py [--full-only] [--n_boot 5000]
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ExpMethods import aggregators as agg  # noqa: E402
from ExpMethods import evaluation as ev  # noqa: E402
from ExpMethods.globals import GlobalValues as G  # noqa: E402
import evaluate_clinical_and_weights as ecw  # noqa: E402

TABLES_DIR = os.path.join(BASE_DIR, "overleaf", "tables")
OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs")
PRIMARY = G.primary_hbml
START = int(G.burn_in_steps)
REFERENCE_EXPERTS = ("Persistence", "RunningMean")


def simplex_ls(F, y, n_iter=20000, tol=1e-12):
    """argmin_w ||F w - y||^2 subject to w >= 0, sum w = 1.

    On the simplex F w - y = E w with E = F - y (the expert errors), so we minimise w' (E'E / n) w by
    projected gradient with an exact simplex projection (E is well conditioned, unlike F).
    """
    E = F - y[:, None]
    A = E.T @ E / len(y)
    K = A.shape[0]
    w = np.ones(K) / K
    step = 1.0 / max(np.linalg.eigvalsh(A).max(), 1e-12)
    for _ in range(n_iter):
        v = w - step * (A @ w)
        u = np.sort(v)[::-1]
        css = np.cumsum(u) - 1.0
        rho = np.nonzero(u - css / np.arange(1, K + 1) > 0)[0][-1]
        w_new = np.maximum(v - css[rho] / (rho + 1.0), 0.0)
        if np.abs(w_new - w).max() < tol:
            return w_new
        w = w_new
    return w


def hbml_rmse(F, y, mask, h):
    L = agg.online_losses(F, y, START, h)
    _, loss = agg.run_named(PRIMARY, F, L, y, start=START, horizon=h)
    return ev.rmse(loss, mask)


def load_config(ds, h_str, h_disp, ctx, keep):
    fc_dir = os.path.join(OUTPUTS_DIR, ds, f"h-{h_str}hr", f"context-{ctx}hr", "forecasts")
    out = []
    for f_path in sorted(glob.glob(os.path.join(fc_dir, "*_forecasts.csv"))):
        pid = os.path.basename(f_path)[:3]
        if os.path.basename(f_path)[3:] != "_forecasts.csv" or (keep and pid not in keep):
            continue
        meta = ev.read_meta(f_path)
        if not meta or meta.get("settings", {}).get("engine_version") != ecw.ENGINE_VERSION:
            continue
        y = ev.load_targets(ds, pid, root=BASE_DIR)
        if y is None:
            continue
        cols, F = ecw.load_forecasts_matrix(f_path)
        n = min(len(y), len(F))
        y, F = y[:n], F[:n]
        mask = ev.scored_mask(F, y)
        if mask.sum() < int(getattr(G, "min_scored_points", 288)):
            continue
        out.append(dict(pid=pid, y=y, F=F, cols=cols, mask=mask))
    return out


def boot_ci(d, n_boot, seed=11):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-only", action="store_true")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    retained = ecw.load_retained_ids()
    rows = []
    for ds in ("weinstock", "cgmacros"):
        for h_str, h_disp, h in ecw.HORIZONS:
            for ctx in (["full"] if a.full_only else ecw.CONTEXTS):
                parts = load_config(ds, h_str, h_disp, ctx, retained.get((ds, h_disp, ctx)))
                if not parts:
                    continue
                cols = parts[0]["cols"]
                print(f"{ds} h={h_disp} ctx={ctx}: {len(parts)} participants", flush=True)
                # fixed convex weights, participant-level CV
                rng = np.random.default_rng(0)
                fold = rng.permutation(len(parts)) % a.folds
                w_cv = {}
                for k in range(a.folds):
                    tr = [p for p, f in zip(parts, fold) if f != k]
                    Ftr = np.vstack([p["F"][p["mask"]] for p in tr])
                    ytr = np.concatenate([p["y"][p["mask"]] for p in tr])
                    wk = simplex_ls(Ftr, ytr)
                    for p, f in zip(parts, fold):
                        if f == k:
                            w_cv[p["pid"]] = wk
                for p in parts:
                    F, y, m = p["F"], p["y"], p["mask"]
                    res = {"HBML": hbml_rmse(F, y, m, h),
                           "Uniform": ev.rmse((F.mean(axis=1) - y) ** 2, m),
                           "FixedConvex-CV": ev.rmse((F @ w_cv[p["pid"]] - y) ** 2, m)}
                    for j, c in enumerate(cols):
                        keep = [i for i in range(len(cols)) if i != j]
                        res[f"HBML-minus-{c}"] = hbml_rmse(F[:, keep], y, m, h)
                    keep = [i for i, c in enumerate(cols) if c not in REFERENCE_EXPERTS]
                    res["HBML-no-reference"] = hbml_rmse(F[:, keep], y, m, h)
                    for v, r in res.items():
                        rows.append(dict(dataset=ds, horizon=h_disp, context=ctx, id=p["pid"], variant=v, rmse=r))
    if not rows:
        raise SystemExit("No engine-%s forecasts found under Outputs/." % ecw.ENGINE_VERSION)
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "pool_mechanism_participant.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = []
    keys = sorted({(r["dataset"], r["horizon"], r["context"]) for r in rows})
    for key in keys:
        sub = [r for r in rows if (r["dataset"], r["horizon"], r["context"]) == key]
        by = {}
        for r in sub:
            by.setdefault(r["variant"], {})[r["id"]] = r["rmse"]
        base = by["HBML"]
        for v, d in by.items():
            ids = sorted(set(d) & set(base))
            diff = np.array([d[i] - base[i] for i in ids])   # positive = variant worse than HBML
            lo, hi = boot_ci(diff, a.n_boot)
            summ.append(dict(dataset=key[0], horizon=key[1], context=key[2], variant=v, n=len(ids),
                             mean_rmse=float(np.mean([d[i] for i in ids])),
                             diff_vs_hbml=float(diff.mean()), diff_ci95_low=lo, diff_ci95_high=hi))
    with open(os.path.join(TABLES_DIR, "pool_mechanism_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    print(f"Wrote pool_mechanism_participant.csv ({len(rows)} rows) and pool_mechanism_summary.csv")


if __name__ == "__main__":
    main()
