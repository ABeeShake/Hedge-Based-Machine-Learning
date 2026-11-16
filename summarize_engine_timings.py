#!/usr/bin/env python3
"""summarize_engine_timings.py -- wall-clock costs of the v2 engine, from the simulation metadata.

Reads every Outputs/<dataset>/h-*/context-*/forecasts/<id>_forecasts.meta.json written by
fast_engine/simulation.py and times the HBML weight update directly, then writes
overleaf/tables/timings_summary.csv (format read by generate_results_macros.py).

Per expert: mean (SD across participant-configurations) of
  Forecasting  seconds per 5-minute step, excluding refits
  Training     seconds per refit
HBML rows: Hedge Update (seconds per step), All-Expert Forecasting (sum of expert forecasting),
All-Expert Training (sum of mean refit times, i.e. a step at which every expert refits) and
End-to-End Step (w/ Train) = all-expert forecasting + all-expert training + hedge update.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from ExpMethods import aggregators as agg  # noqa: E402
from ExpMethods.globals import GlobalValues as G  # noqa: E402
from fast_engine.simulation import ENGINE_VERSION  # noqa: E402

DISPLAY = {"ARIMA": "AutoARIMA", "ETS": "AutoETS"}


def hedge_update_seconds(K, T=3000, reps=5):
    rng = np.random.default_rng(0)
    y = 140 + rng.normal(0, 30, T)
    F = y[:, None] + rng.normal(0, 20, (T, K))
    L = agg.online_losses(F, y, G.burn_in_steps, 6)
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        agg.run_named(G.primary_hbml, F, L, y, start=G.burn_in_steps, horizon=6)
        per.append((time.perf_counter() - t0) / (T - G.burn_in_steps))
    return float(np.mean(per)), float(np.std(per, ddof=1))


def fmt(m, s):
    nd = 6 if m < 1e-3 else 4
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def main():
    rows = []
    for mp in glob.glob(os.path.join(BASE, "Outputs", "*", "h-*", "context-*", "forecasts", "*_forecasts.meta.json")):
        try:
            meta = json.load(open(mp))
        except (OSError, ValueError):
            continue
        if meta.get("settings", {}).get("engine_version") != ENGINE_VERSION:
            continue
        for name, t in meta["timings"].items():
            rows.append(dict(expert=name, forecast_s=t["forecast_s_mean"],
                             fit_s=t["fit_s_mean"] if t["n_fits"] else np.nan,
                             fallbacks=t.get("n_fallback", 0), steps=t.get("n_steps", 0), error=t.get("first_error")))
    if not rows:
        raise SystemExit("No current-engine metadata found under Outputs/. Run the simulations first.")
    df = pd.DataFrame(rows)
    fb = df.groupby("expert")[["fallbacks", "steps"]].sum()
    for name, r in fb[fb.fallbacks > 0].iterrows():
        err = df[(df.expert == name) & df.error.notna()].error.iloc[0]
        print(f"WARNING: {name} failed and fell back to persistence on {int(r.fallbacks)} of {int(r.steps)} steps "
              f"({100 * r.fallbacks / max(r.steps, 1):.2f}%). First error: {err}")
    out = []
    fc_tot, fit_tot = 0.0, 0.0
    fc_var, fit_var = 0.0, 0.0
    for name in [e for e in G.experts if e in set(df.expert)]:
        d = df[df.expert == name]
        fm_, fs_ = d.forecast_s.mean(), d.forecast_s.std(ddof=1)
        out.append(("\\multirow{2}{*}{%s}" % DISPLAY.get(name, name), "Forecasting", fmt(fm_, fs_)))
        if d.fit_s.notna().any():
            tm_, ts_ = d.fit_s.mean(), d.fit_s.std(ddof=1)
            out.append(("", "Training", fmt(tm_, ts_)))
            fit_tot += tm_; fit_var += ts_ ** 2
        else:
            out.append(("", "Training", "---"))
        fc_tot += fm_; fc_var += fs_ ** 2
    hm, hs = hedge_update_seconds(len(G.experts))
    out += [("\\multirow{4}{*}{\\textbf{HBML (Total)}}", "Hedge Update", fmt(hm, hs)),
            ("", "All-Expert Forecasting", fmt(fc_tot, np.sqrt(fc_var))),
            ("", "All-Expert Training", fmt(fit_tot, np.sqrt(fit_var))),
            ("", "End-to-End Step (w/ Train)", fmt(fc_tot + fit_tot + hm, np.sqrt(fc_var + fit_var + hs ** 2)))]
    path = os.path.join(BASE, "overleaf", "tables", "timings_summary.csv")
    pd.DataFrame(out, columns=["Model", "Category", "Time (s)"]).to_csv(path, index=False)
    print(f"Wrote {path} from {len(df)} expert-participant records.")


if __name__ == "__main__":
    main()
