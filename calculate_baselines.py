#!/usr/bin/env python3
"""calculate_baselines.py -- naive comparators for HBML, computed from the raw forecasts.

Baselines (all causal, all evaluated on exactly the rows used for HBML):
  persistence   y_hat(t+h) = y(t)
  climatology   y_hat(t+h) = mean of y(0..t)            (running participant mean)
  uniform       y_hat(t+h) = mean of the K expert forecasts
HBML itself is re-aggregated here with ExpMethods/aggregators.py using the primary
configuration named in ExpMethods/globals.py, so every method is scored on the same
participants and time points.

Nothing in this file is hard-coded: a configuration whose forecasts are missing, or whose
forecasts do not reproduce the HBML mean in rmse_pivot_mean.csv (stale copy), is written
as NaN and shows up as a visible gap in the table and macros.

Needs Outputs/<dataset>/h-*/context-*/forecasts and the raw glucose files, so run it on
the machine that holds the simulation outputs (after generate_figures.py --rmse_only).

Outputs: overleaf/tables/baselines_comparison.csv, baselines_participant.csv,
         table_baselines_comparison.tex
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from ExpMethods import aggregators as agg  # noqa: E402
from ExpMethods.globals import GlobalValues  # noqa: E402

TABLES_DIR = os.path.join(BASE_DIR, "overleaf", "tables")
OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs")
DATASETS = ["weinstock", "cgmacros"]
HORIZONS = [0.5, 2.0, 5.0]
CONTEXTS = ["6", "12", "24", "full"]
HORIZON_DIR = {0.5: "h-halfhr", 2.0: "h-2hr", 5.0: "h-5hr"}
HORIZON_STEPS = {0.5: 6, 2.0: 24, 5.0: 60}
DATASET_LABEL = {"weinstock": "Weinstock", "cgmacros": "CGMacros"}
HORIZON_LABEL = {0.5: "0.5 hr", 2.0: "2 hr", 5.0: "5 hr"}
EXPERTS = ["NODE", "ARIMA", "ETS", "XGBoost", "NHITS"]
START = 21
EVAL = slice(100, -10)
BASELINES = ["persistence", "climatology", "uniform"]


def _rmse(sq):
    v = sq[EVAL]
    return float(np.sqrt(np.nanmean(v))) if len(v) else np.nan


def wilcoxon_p(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 6:
        return np.nan
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:  # SciPy missing: normal approximation with tie correction
        a = pd.Series(np.abs(d))
        r = a.rank().to_numpy()
        n = len(d)
        ties = a.value_counts().to_numpy()
        var = n * (n + 1) * (2 * n + 1) / 24 - (ties ** 3 - ties).sum() / 48
        z = (r[d > 0].sum() - n * (n + 1) / 4) / math.sqrt(var)
        return float(math.erfc(abs(z) / math.sqrt(2)))


def load_targets(ds, pid):
    for d in (os.path.join(BASE_DIR, ds), os.path.join(OUTPUTS_DIR, ds, "data")):
        p = os.path.join(d, f"{pid}.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
            except OSError:
                continue
            return df["Libre.GL" if "Libre.GL" in df.columns else "Dexcom.GL"].to_numpy(dtype=float)
    return None


def participant_ids(ds, h, ctx):
    """IDs retained in the main tables for this configuration (None = no restriction available)."""
    p = os.path.join(TABLES_DIR, "rmse_participant.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, dtype={"id": str, "context": str, "horizon": str})
        df["h"] = df["horizon"].map(lambda v: 0.5 if v in ("half", "0.5") else float(v))
        sel = df[(df.dataset == ds) & (df.h == h) & (df.context == ctx)]
        return set(sel["id"].str.zfill(3)) if len(sel) else None
    p = os.path.join(TABLES_DIR, f"rmse_dev_{ds}_h{str(h).replace('.0','').replace('.','p')}_c{ctx}.csv")
    if os.path.exists(p):
        return set(pd.read_csv(p, dtype={"id": str})["id"].str.zfill(3))
    return None


def table_hbml_mean(ds, h, ctx):
    p = os.path.join(TABLES_DIR, "rmse_pivot_mean.csv")
    if not os.path.exists(p):
        return np.nan
    pv = pd.read_csv(p, dtype=str)
    row = pv[(pv.dataset == ds) & (pv.model == "HBML") & (pv.horizon.astype(float) == h)]
    if row.empty:
        return np.nan
    try:
        return float(str(row.iloc[0][f"mean.{ctx}"]).replace("\\textbf{", "").replace("}", ""))
    except (ValueError, KeyError):
        return np.nan


def _one_participant(task):
    ds, h, ctx, f = task
    steps = HORIZON_STEPS[h]
    pid = os.path.basename(f)[:3]
    y = load_targets(ds, pid)
    try:
        fdf = pd.read_csv(f)
    except OSError:
        return None
    if y is None:
        return None
    if "HoltWinters" in fdf.columns and "ETS" not in fdf.columns:
        fdf = fdf.rename(columns={"HoltWinters": "ETS"})
    cols = [c for c in EXPERTS if c in fdf.columns]
    if len(cols) < 2:
        return None
    f_mat = fdf[cols].to_numpy(dtype=float)
    n = min(len(f_mat), len(y))
    if n < 120 + steps:
        return None
    f_mat, y = f_mat[:n], y[:n]
    l_mat = agg.online_losses(f_mat, y, START, steps)
    _, hb = agg.run_named(GlobalValues.primary_hbml, f_mat, l_mat, y, start=START, horizon=steps)
    pers = np.full(n, np.nan)
    pers[steps:] = y[:-steps]
    clim = np.full(n, np.nan)
    clim[steps:] = (np.cumsum(y) / np.arange(1, n + 1))[:-steps]
    rec = dict(dataset=ds, horizon=h, context=ctx, id=pid, hbml=_rmse(hb),
               persistence=_rmse((pers - y) ** 2), climatology=_rmse((clim - y) ** 2),
               uniform=_rmse((np.nanmean(f_mat, axis=1) - y) ** 2))
    for i, c in enumerate(cols):
        rec[c] = _rmse(l_mat[:, i])
    return rec


WORKERS = max(1, (os.cpu_count() or 2) - 1)


def evaluate_condition(ds, h, ctx):
    fc_dir = os.path.join(OUTPUTS_DIR, ds, HORIZON_DIR[h], f"context-{ctx}hr", "forecasts")
    keep = participant_ids(ds, h, ctx)
    tasks = []
    for f in sorted(glob.glob(os.path.join(fc_dir, "*_forecasts.csv"))):
        name = os.path.basename(f)
        if name[3:] != "_forecasts.csv" or (keep is not None and name[:3] not in keep):
            continue
        tasks.append((ds, h, ctx, f))
    if WORKERS > 1 and len(tasks) > 8:
        from multiprocessing import Pool
        with Pool(WORKERS) as pool:
            res = pool.map(_one_participant, tasks, chunksize=4)
    else:
        res = [_one_participant(t) for t in tasks]
    return pd.DataFrame([r for r in res if r is not None])


def evaluate_baselines(allow_mismatch=False):
    parts, records = [], []
    for ds in DATASETS:
        for h in HORIZONS:
            for ctx in CONTEXTS:
                df = evaluate_condition(ds, h, ctx)
                rec = dict(dataset=ds, horizon=h, context=ctx, n=len(df), status="ok")
                ref = table_hbml_mean(ds, h, ctx)
                if df.empty:
                    rec["status"] = "no_forecasts"
                elif not allow_mismatch and not np.isnan(ref) and abs(df["hbml"].mean() - ref) > 0.005 * ref:
                    rec["status"] = "stale_forecasts"
                    print(f"WARNING {ds} h={h} ctx={ctx}: HBML recomputed here = {df['hbml'].mean():.2f} but the main "
                          f"table says {ref:.2f}. These forecasts are not the ones behind the tables; row left empty.")
                if rec["status"] != "ok":
                    if rec["status"] == "no_forecasts":
                        print(f"WARNING {ds} h={h} ctx={ctx}: no forecast files found; row left empty.")
                    for k in ["hbml_rmse", "best_expert_rmse"] + [f"{b}_rmse" for b in BASELINES]:
                        rec[k] = np.nan
                    rec["best_expert_name"] = ""
                    records.append(rec)
                    continue
                parts.append(df)
                experts = [c for c in EXPERTS if c in df.columns]
                best = min(experts, key=lambda c: df[c].mean())
                rec.update(hbml_rmse=df["hbml"].mean(), best_expert_name=best, best_expert_rmse=df[best].mean())
                for b in BASELINES:
                    d = df[b] - df["hbml"]  # positive = HBML better
                    rec[f"{b}_rmse"] = df[b].mean()
                    rec[f"hbml_reduction_vs_{b}_pct"] = 100.0 * (df[b].mean() - df["hbml"].mean()) / df[b].mean()
                    rec[f"pct_improved_vs_{b}"] = 100.0 * float((d > 1e-3).mean())
                    rec[f"wilcoxon_p_vs_{b}"] = wilcoxon_p(d.to_numpy())
                records.append(rec)
                print(f"{ds:9s} h={h:<3} ctx={ctx:<4s} n={len(df):<3d} persistence={rec['persistence_rmse']:6.2f} "
                      f"climatology={rec['climatology_rmse']:6.2f} uniform={rec['uniform_rmse']:6.2f} "
                      f"best({best})={rec['best_expert_rmse']:6.2f} HBML={rec['hbml_rmse']:6.2f}")
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(TABLES_DIR, "baselines_comparison.csv"), index=False)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(TABLES_DIR, "baselines_participant.csv"), index=False)
    generate_latex_table(out, os.path.join(TABLES_DIR, "table_baselines_comparison.tex"))
    n_bad = int((out["status"] != "ok").sum())
    print(f"Saved baselines for {len(out) - n_bad}/{len(out)} configurations"
          + (f"; {n_bad} left empty (see warnings)." if n_bad else "."))
    return out


def generate_latex_table(df, output_tex):
    cols = ["persistence_rmse", "climatology_rmse", "uniform_rmse", "best_expert_rmse", "hbml_rmse"]
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{lllcccccc}", r"\toprule",
             r"Dataset & Horizon & Context & Persistence & Running mean & Uniform ensemble & Best expert & HBML & HBML vs.\ persistence (\%) \\",
             r"\midrule"]
    cur = None
    for _, r in df.iterrows():
        ds = DATASET_LABEL[r["dataset"]]
        if cur is not None and cur != ds:
            lines.append(r"\midrule")
        cur = ds
        vals = [r.get(c, np.nan) for c in cols]
        if all(pd.isna(v) for v in vals):
            cells = ["---"] * 5 + ["---"]
        else:
            lo = np.nanmin(vals)
            cells = [("---" if pd.isna(v) else (f"\\textbf{{{v:.2f}}}" if abs(v - lo) < 5e-3 else f"{v:.2f}")) for v in vals]
            red = r.get("hbml_reduction_vs_persistence_pct", np.nan)
            cells.append("---" if pd.isna(red) else f"{red:+.1f}")
        lines.append(f"{ds} & {HORIZON_LABEL[r['horizon']]} & {'Full' if r['context'] == 'full' else r['context'] + ' hr'} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{HBML compared with naive baselines.} Mean participant-level RMSE (mg/dL) for naive "
              r"persistence ($\hat{y}_{t+h} = y_t$), the running participant mean ($\hat{y}_{t+h} = \bar{y}_{0:t}$), an "
              r"equally weighted average of the five experts, the best constituent expert (lowest cohort mean RMSE), and "
              r"HBML. All methods are scored on the same participants and time points. Bold marks the lowest value in "
              r"each row. The last column is the relative RMSE change of HBML against persistence (positive: HBML lower). "
              r"Dashes mark configurations that could not be evaluated.}",
              r"\label{tab:baselines_comparison}", r"\end{table*}"]
    with open(output_tex, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_mismatch", action="store_true",
                    help="Keep rows even if the recomputed HBML mean differs from rmse_pivot_mean.csv.")
    ap.add_argument("--tables_dir", default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if a.tables_dir:
        TABLES_DIR = a.tables_dir
    WORKERS = a.workers
    evaluate_baselines(a.allow_mismatch)
