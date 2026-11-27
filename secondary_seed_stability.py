#!/usr/bin/env python3
"""secondary_seed_stability.py -- secondary analysis: stochastic run-to-run seed stability.

Evaluates run-to-run variation of the stochastic experts (NODE, NHITS, XGBoost) and HBML
across three random seeds (default seeds = 0, 1, 2) on a fixed random subset of participants
(default 20 per cohort, 30-min and 5-h horizons, full-history context).

Stages
  simulate  writes Outputs/secondary/seeds/s<seed>/<dataset>/h-<h>hr/context-<c>hr/forecasts/
            (resumable; completed participants are skipped). Needs the full engine (VM).
  evaluate  writes overleaf/tables/seed_stability.csv, seed_stability_participant.csv and
            table_seed_stability.tex (NumPy/pandas only).

Examples
  python3 secondary_seed_stability.py --stage simulate --n_workers 8
  python3 secondary_seed_stability.py --stage evaluate
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from ExpMethods import aggregators as agg  # noqa: E402
from ExpMethods import evaluation as ev  # noqa: E402
from ExpMethods.globals import GlobalValues as G  # noqa: E402
from fast_engine.simulation import ENGINE_VERSION  # noqa: E402

OUT_ROOT = os.path.join(BASE, "Outputs", "secondary", "seeds")
TABLES = os.path.join(BASE, "overleaf", "tables")
CTX = {"6": 72, "12": 144, "24": 288, "full": -1}


def subset(ds, n, seed=0):
    files = ev.participant_files(ds)
    if n >= len(files):
        return files
    idx = np.random.default_rng(seed).choice(len(files), n, replace=False)
    return [files[i] for i in sorted(idx)]


def cond_dir(seed_val, ds, hk, ctx):
    return os.path.join(OUT_ROOT, f"s{seed_val}", ds, f"h-{hk}hr", f"context-{ctx}hr")


def stage_simulate(a):
    from fast_engine import run_cohort
    for ds in a.datasets:
        files = subset(ds, a.n_participants, seed=0)
        for hk in a.horizons:
            for s in a.seeds:
                d = cond_dir(s, ds, hk, a.context)
                print(f"== {ds} h={hk} context={a.context} seed={s}: {len(files)} participants", flush=True)
                run_cohort(files, horizon=ev.HORIZON_STEPS[hk], context_len=CTX[a.context], output_dir=d,
                           n_workers=a.n_workers, skip_existing=True, seed=s)


def stage_evaluate(a):
    recs = []
    for s in a.seeds:
        for ds in a.datasets:
            for hk in a.horizons:
                d = cond_dir(s, ds, hk, a.context)
                h = ev.HORIZON_STEPS[hk]
                for f in ev.forecast_files(d):
                    pid = os.path.basename(f)[:3]
                    meta = ev.read_meta(f)
                    if not meta or meta["settings"].get("seed") != s or meta["settings"].get("engine_version") != ENGINE_VERSION:
                        continue
                    y = ev.load_targets(ds, pid)
                    F = pd.read_csv(f)
                    experts = [e for e in G.experts if e in F.columns]
                    n = min(len(F), len(y))
                    P, yy = F[experts].to_numpy(float)[:n], y[:n]
                    rec = dict(seed=s, dataset=ds, horizon=ev.HORIZON_LABEL[hk], context=a.context, id=pid)
                    mask = ev.scored_mask(P, yy)
                    with np.errstate(invalid="ignore"):
                        for i, e in enumerate(experts):
                            rec[e] = ev.rmse((P[:, i] - yy) ** 2, mask)
                    L = agg.online_losses(P, yy, G.burn_in_steps, h)
                    _, l = agg.run_named(G.primary_hbml, P, L, yy, start=G.burn_in_steps, horizon=h)
                    rec["HBML"] = ev.rmse(l, mask)
                    recs.append(rec)
    if not recs:
        raise SystemExit("Nothing to evaluate: run --stage simulate first.")
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(TABLES, "seed_stability_participant.csv"), index=False)
    models = [e for e in G.experts if e in df.columns] + ["HBML"]
    rows = []
    for (ds, hz), g in df.groupby(["dataset", "horizon"], sort=False):
        r = dict(dataset=ds, horizon=hz, context=a.context, n_participants=g["id"].nunique(),
                 n_seeds=g["seed"].nunique())
        for m in models:
            piv = g.pivot(index="id", columns="seed", values=m).dropna()
            if piv.empty:
                continue
            r[f"{m}_mean_rmse"] = float(piv.values.mean())
            # mean across participants of their across-seed SD
            r[f"{m}_within_patient_seed_sd"] = float(piv.std(axis=1, ddof=1).mean())
            # SD of the cohort-mean RMSE across seeds
            r[f"{m}_cohort_mean_seed_sd"] = float(piv.mean(axis=0).std(ddof=1))
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "seed_stability.csv"), index=False)
    write_tex(out)
    print(out.round(4).to_string(index=False))


def write_tex(df):
    lab = {"weinstock": "Weinstock", "cgmacros": "CGMacros"}
    stoch = [m for m in ("NHITS", "NODE", "XGBoost", "HBML") if f"{m}_mean_rmse" in df.columns]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} llcc" + "c" * len(stoch) + "}", r"\toprule",
             "Dataset & Horizon (hr) & $N$ & Seeds & " + " & ".join(stoch) + r" \\",
             r"\midrule"]
    for _, r in df.iterrows():
        cells = [f"{r[f'{m}_mean_rmse']:.2f} ($\\pm${r[f'{m}_within_patient_seed_sd']:.3f})" for m in stoch]
        lines.append(f"{lab[r.dataset]} & {r.horizon} & {int(r.n_participants)} & {int(r.n_seeds)} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}",
              r"\caption{\textbf{Run-to-run stochastic seed stability.} Mean participant-level RMSE (mg/dL) and mean "
              r"within-participant across-seed standard deviation (in parentheses) across three random seeds on a fixed "
              r"subset of participants (full-history context).}",
              r"\label{tab:seed_stability}", r"\end{table}"]
    with open(os.path.join(TABLES, "table_seed_stability.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["simulate", "evaluate"], required=True)
    ap.add_argument("--dataset", choices=["weinstock", "cgmacros", "all"], default="all")
    ap.add_argument("--horizons", nargs="+", default=["half", "5"], choices=["half", "2", "5"])
    ap.add_argument("--context", default="full", choices=list(CTX))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_participants", type=int, default=20)
    ap.add_argument("--n_workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()
    a.datasets = ["weinstock", "cgmacros"] if a.dataset == "all" else [a.dataset]
    stage_simulate(a) if a.stage == "simulate" else stage_evaluate(a)


if __name__ == "__main__":
    main()

