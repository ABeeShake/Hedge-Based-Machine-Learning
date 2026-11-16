#!/usr/bin/env python3
"""secondary_retrain_cadence.py -- secondary analysis: does the refit interval change the results?

Re-runs the full expert pool on a fixed random subset of participants with several refit
intervals (default 6 h, 24 h and 48 h = 72, 288, 576 steps; 24 h is the main analysis) and
compares expert and HBML RMSE with the 24-hour reference, together with the training cost.

Stages
  simulate  writes Outputs/secondary/cadence/i<interval>/<dataset>/h-<h>hr/context-<c>hr/forecasts/
            (resumable; completed participants are skipped). Needs the full engine (VM).
  evaluate  writes overleaf/tables/retrain_cadence.csv, retrain_cadence_participant.csv and
            table_retrain_cadence.tex (NumPy/pandas only).

Examples
  python3 secondary_retrain_cadence.py --stage simulate --n_workers 8
  python3 secondary_retrain_cadence.py --stage simulate --intervals 576 --dataset cgmacros
  python3 secondary_retrain_cadence.py --stage evaluate
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

OUT_ROOT = os.path.join(BASE, "Outputs", "secondary", "cadence")
TABLES = os.path.join(BASE, "overleaf", "tables")
CTX = {"6": 72, "12": 144, "24": 288, "full": -1}


def subset(ds, n, seed):
    files = ev.participant_files(ds)
    if n >= len(files):
        return files
    idx = np.random.default_rng(seed).choice(len(files), n, replace=False)
    return [files[i] for i in sorted(idx)]


def cond_dir(interval, ds, hk, ctx):
    return os.path.join(OUT_ROOT, f"i{interval}", ds, f"h-{hk}hr", f"context-{ctx}hr")


def stage_simulate(a):
    from fast_engine import run_cohort
    for ds in a.datasets:
        files = subset(ds, a.n_participants, a.seed)
        for hk in a.horizons:
            for I in a.intervals:
                d = cond_dir(I, ds, hk, a.context)
                print(f"== {ds} h={hk} context={a.context} refit every {I} steps: {len(files)} participants", flush=True)
                run_cohort(files, horizon=ev.HORIZON_STEPS[hk], context_len=CTX[a.context], output_dir=d,
                           n_workers=a.n_workers, skip_existing=True, retrain_interval=I)


def stage_evaluate(a):
    recs = []
    for I in a.intervals:
        for ds in a.datasets:
            for hk in a.horizons:
                d = cond_dir(I, ds, hk, a.context)
                h = ev.HORIZON_STEPS[hk]
                for f in ev.forecast_files(d):
                    pid = os.path.basename(f)[:3]
                    meta = ev.read_meta(f)
                    if not meta or meta["settings"].get("retrain_interval") != I:
                        continue
                    y = ev.load_targets(ds, pid)
                    F = pd.read_csv(f)
                    experts = [e for e in G.experts if e in F.columns]
                    n = min(len(F), len(y))
                    P, yy = F[experts].to_numpy(float)[:n], y[:n]
                    rec = dict(interval=I, dataset=ds, horizon=ev.HORIZON_LABEL[hk], context=a.context, id=pid,
                               fit_s_total=sum(t["fit_s_total"] for t in meta["timings"].values()),
                               n_fits=sum(t["n_fits"] for t in meta["timings"].values()))
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
    df.to_csv(os.path.join(TABLES, "retrain_cadence_participant.csv"), index=False)
    models = [e for e in G.experts if e in df.columns] + ["HBML"]
    rows = []
    for (ds, hz), g in df.groupby(["dataset", "horizon"], sort=False):
        ref = g[g.interval == a.reference].set_index("id")
        for I, gi in g.groupby("interval"):
            gi = gi.set_index("id")
            common = ref.index.intersection(gi.index)
            r = dict(dataset=ds, horizon=hz, context=a.context, interval=int(I), hours=I * 5 / 60, n=len(common),
                     fit_s_per_participant=gi.loc[common, "fit_s_total"].mean())
            for m in models:
                r[f"{m}_rmse"] = gi.loc[common, m].mean()
                if I != a.reference:
                    s = ev.paired_summary(ref.loc[common, m], gi.loc[common, m])
                    r[f"{m}_delta_vs_ref"] = s["delta_mean"]
                    r[f"{m}_rel_pct_vs_ref"] = s["rel_pct"]
                    r[f"{m}_p_vs_ref"] = s["wilcoxon_p"]
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "retrain_cadence.csv"), index=False)
    write_tex(out, models)
    print(out[["dataset", "horizon", "hours", "n", "fit_s_per_participant", "HBML_rmse"]
              + [f"{m}_rmse" for m in models if m != "HBML"]].round(2).to_string(index=False))


def write_tex(df, models):
    lab = {"weinstock": "Weinstock", "cgmacros": "CGMacros"}
    show = [m for m in models if m not in ("Persistence", "RunningMean")]
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\begin{tabular}{llc" + "c" * len(show) + "cc}", r"\toprule",
             "Dataset & Horizon (hr) & Refit every & " + " & ".join(show) + r" & HBML $p$ vs 24 h & Training (s) \\",
             r"\midrule"]
    for _, r in df.iterrows():
        cells = [f"{r[f'{m}_rmse']:.2f}" for m in show]
        p = r.get("HBML_p_vs_ref", np.nan)
        lines.append(f"{lab[r.dataset]} & {r.horizon} & {r.hours:g} h & " + " & ".join(cells)
                     + f" & {ev.tex_p(p) if r.interval != 288 else 'ref.'} & {r.fit_s_per_participant:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Sensitivity to the expert refit interval.} Mean participant-level RMSE (mg/dL) on a fixed "
              r"random subset of participants, full-history context, with experts refitted every 6, 24 (main analysis) or "
              r"48 hours. $p$: two-sided Wilcoxon signed-rank test of HBML against the 24-hour interval. Training: total "
              r"expert refit time per participant (seconds).}",
              r"\label{tab:retrain_cadence}", r"\end{table*}"]
    with open(os.path.join(TABLES, "table_retrain_cadence.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["simulate", "evaluate"], required=True)
    ap.add_argument("--dataset", choices=["weinstock", "cgmacros", "all"], default="all")
    ap.add_argument("--horizons", nargs="+", default=["half", "5"], choices=["half", "2", "5"])
    ap.add_argument("--context", default="full", choices=list(CTX))
    ap.add_argument("--intervals", nargs="+", type=int, default=[72, 288, 576])
    ap.add_argument("--reference", type=int, default=288)
    ap.add_argument("--n_participants", type=int, default=20, help="Per dataset; fixed random subset.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()
    a.datasets = ["weinstock", "cgmacros"] if a.dataset == "all" else [a.dataset]
    if a.stage == "evaluate" and a.reference not in a.intervals:
        a.intervals.append(a.reference)
    stage_simulate(a) if a.stage == "simulate" else stage_evaluate(a)


if __name__ == "__main__":
    main()
