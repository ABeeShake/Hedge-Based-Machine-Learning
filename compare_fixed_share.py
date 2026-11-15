#!/usr/bin/env python3
"""compare_fixed_share.py -- should HBML use Fixed-Share mixing?

Re-aggregates the saved expert forecasts with several aggregators and compares
participant-level RMSE. Nothing is re-simulated: only ``*_forecasts.csv`` and
the raw glucose files are read, so this needs NumPy and pandas only (SciPy is
used for exact Wilcoxon p-values when it is installed).

Variants
--------
current        HBML-SFHDF as used for the results so far (no Fixed-Share)
alg1_fs_fd     Algorithm 1 of the manuscript: recursive update, decayed-max
               scale, Fixed-Share with alpha_n = 1/(n+1)^2
alg1_noshare   Algorithm 1 with the sharing step removed (isolates Fixed-Share)
alg1_fs_d      Algorithm 1 with alpha_n = 1/(n+1)
alg1_fs_c*     Algorithm 1 with constant alpha (0.001, 0.01, 0.05)
cum_decaymax   current update rule with the decayed-max scale (isolates the scale)
uniform        equal-weight average of the experts (reference)

Evaluation mirrors process_results.py: start step 21, squared error of the
weighted-mean forecast, RMSE over ``loss[100:-10]``.

Choosing a variant on the same participants that are then reported would be
selection on the test set. The report therefore also splits participants into
two fixed halves (odd / even ID), picks the best variant on one half and shows
its performance on the other.

Usage
-----
  python3 compare_fixed_share.py                 # every complete condition found
  python3 compare_fixed_share.py --dataset weinstock --horizon 5
  python3 compare_fixed_share.py --report-only   # rebuild the summary from saved parts
Outputs go to ./fs_comparison/ .
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ExpMethods import aggregators as agg  # noqa: E402

START = 21
EVAL = slice(100, -10)
FAST_ENGINE_CUTOFF = 1792090000.0  # same cutoff as run_simulation_chunk.py
EXPERTS = ["NODE", "ARIMA", "XGBoost", "NHITS", "ETS"]
HORIZONS = {"half": 6, "2": 24, "5": 60}
CONTEXTS = ["6", "12", "24", "full"]
VARIANTS = {
    "current": dict(update="cumulative", scale="ema", gamma=0.2, alpha=0.0),
    "alg1_fs_fd": dict(update="recursive", scale="decay_max", gamma=0.2, alpha="fd"),
    "alg1_noshare": dict(update="recursive", scale="decay_max", gamma=0.2, alpha=0.0),
    "alg1_fs_d": dict(update="recursive", scale="decay_max", gamma=0.2, alpha="d"),
    "alg1_fs_c0.001": dict(update="recursive", scale="decay_max", gamma=0.2, alpha=0.001),
    "alg1_fs_c0.01": dict(update="recursive", scale="decay_max", gamma=0.2, alpha=0.01),
    "alg1_fs_c0.05": dict(update="recursive", scale="decay_max", gamma=0.2, alpha=0.05),
    "cum_decaymax": dict(update="cumulative", scale="decay_max", gamma=0.2, alpha=0.0),
}
FS_VARIANTS = [v for v in VARIANTS if "_fs_" in v]


def find_targets(root, dataset, pid):
    for d in (os.path.join(root, dataset), os.path.join(root, "Outputs", dataset, "data")):
        p = os.path.join(d, f"{pid}.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
            except OSError:
                continue
            col = "Libre.GL" if "Libre.GL" in df.columns else "Dexcom.GL"
            return df[col].to_numpy(dtype=float)
    return None


def rmse(loss):
    v = loss[EVAL]
    return float(np.sqrt(np.mean(v))) if len(v) else float("nan")


def _one_participant(task):
    root, dataset, hkey, ctx, f = task
    h = HORIZONS[hkey]
    pid = os.path.basename(f)[:3]
    try:
        fdf = pd.read_csv(f)
    except OSError:
        return []
    if "HoltWinters" in fdf.columns:
        fdf = fdf.rename(columns={"HoltWinters": "ETS"})
    cols = [c for c in EXPERTS if c in fdf.columns]
    y = find_targets(root, dataset, pid)
    if len(cols) < 2 or y is None:
        return []
    f_mat = fdf[cols].to_numpy(dtype=float)
    n = min(len(f_mat), len(y))
    f_mat, y = f_mat[:n], y[:n]
    l_mat = agg.online_losses(f_mat, y, START, h)
    rec = {c: rmse(l_mat[:, i]) for i, c in enumerate(cols)}
    uni = np.nanmean(f_mat, axis=1)
    ul = (uni - y) ** 2
    ul[: START + h] = 0.0
    rec["uniform"] = rmse(ul)
    for name, spec in VARIANTS.items():
        _, loss = agg.run_hedge(f_mat, l_mat, y, start=START, horizon=h, **spec)
        rec[name] = rmse(loss)
    return [dict(dataset=dataset, horizon=hkey, context=ctx, id=pid, variant=k, rmse=v) for k, v in rec.items()]


def run_condition(root, dataset, hkey, ctx, min_mtime, workers=1):
    fc_dir = os.path.join(root, "Outputs", dataset, f"h-{hkey}hr", f"context-{ctx}hr", "forecasts")
    files = sorted(f for f in glob.glob(os.path.join(fc_dir, "*_forecasts.csv"))
                   if os.path.basename(f)[3:] == "_forecasts.csv")
    fresh = [f for f in files if os.path.getmtime(f) >= min_mtime]
    info = dict(n_files=len(files), n_fresh=len(fresh))
    if not files or len(fresh) < 0.5 * len(files):
        return pd.DataFrame(), info
    tasks = [(root, dataset, hkey, ctx, f) for f in fresh]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_one_participant, tasks, chunksize=4)
    else:
        res = [_one_participant(t) for t in tasks]
    return pd.DataFrame([r for rows in res for r in rows]), info


def wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank p-value (SciPy if present, else normal approximation)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 6:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        r = pd.Series(np.abs(d)).rank().to_numpy()
        n = len(d)
        w = r[d > 0].sum()
        mu = n * (n + 1) / 4
        ties = pd.Series(np.abs(d)).value_counts().to_numpy()
        var = n * (n + 1) * (2 * n + 1) / 24 - (ties ** 3 - ties).sum() / 48
        z = (w - mu) / math.sqrt(var)
        return float(math.erfc(abs(z) / math.sqrt(2)))


def build_report(out_dir, drop_mismatched=False):
    parts = sorted(glob.glob(os.path.join(out_dir, "parts", "*.csv")))
    if not parts:
        print("No per-condition results found; nothing to report.")
        return
    long = pd.concat([pd.read_csv(p, dtype={"id": str, "horizon": str, "context": str}) for p in parts],
                     ignore_index=True)
    long.to_csv(os.path.join(out_dir, "per_participant.csv"), index=False)
    wide = long.pivot_table(index=["dataset", "horizon", "context", "id"], columns="variant",
                            values="rmse").reset_index()
    experts = [e for e in EXPERTS if e in wide.columns]
    rows = []
    for (ds, h, c), g in wide.groupby(["dataset", "horizon", "context"]):
        best_exp = min(experts, key=lambda e: g[e].mean())
        base = g["current"]
        for v in list(VARIANTS) + ["uniform"]:
            d = base - g[v]  # positive = variant better than current
            rows.append(dict(
                dataset=ds, horizon=h, context=c, n=len(g), variant=v,
                mean_rmse=g[v].mean(), median_rmse=g[v].median(),
                best_expert=best_exp, best_expert_mean_rmse=g[best_exp].mean(),
                delta_vs_current=d.mean(), pct_change_vs_current=100 * (g[v].mean() - base.mean()) / base.mean(),
                pct_participants_better_than_current=100 * float((d > 1e-9).mean()),
                wilcoxon_p_vs_current=wilcoxon_p(d.to_numpy()) if v != "current" else float("nan"),
                delta_vs_best_expert=(g[best_exp] - g[v]).mean(),
                pct_participants_better_than_best_expert=100 * float((g[best_exp] - g[v] > 1e-9).mean()),
            ))
    summ = pd.DataFrame(rows)

    # Consistency check against the manuscript tables: if the 'current' aggregator does not
    # reproduce the HBML mean in overleaf/tables/rmse_pivot_mean.csv, the forecasts in this
    # folder are not the ones behind the tables (e.g. a partial sync from the VM).
    summ["matches_tables"] = "unknown"
    pivot_path = os.path.join(os.path.dirname(out_dir.rstrip(os.sep)), "overleaf", "tables", "rmse_pivot_mean.csv")
    if os.path.exists(pivot_path):
        pv = pd.read_csv(pivot_path, dtype=str)
        pv = pv[pv["model"] == "HBML"]
        hmap = {"0.5": "half", "2": "2", "5": "5", "2.0": "2", "5.0": "5"}
        ref = {}
        for _, r in pv.iterrows():
            for c in CONTEXTS:
                val = str(r.get(f"mean.{c}", "")).replace("\\textbf{", "").replace("}", "")
                try:
                    ref[(r["dataset"], hmap.get(r["horizon"], r["horizon"]), c)] = float(val)
                except ValueError:
                    pass
        cur = summ[summ["variant"] == "current"].set_index(["dataset", "horizon", "context"])["mean_rmse"]
        def _match(r):
            k = (r["dataset"], r["horizon"], r["context"])
            if k not in ref or k not in cur.index:
                return "unknown"
            return "yes" if abs(cur[k] - ref[k]) <= 0.005 * ref[k] else "no"
        summ["matches_tables"] = summ.apply(_match, axis=1)
    if drop_mismatched:
        bad = summ[summ["matches_tables"] == "no"][["dataset", "horizon", "context"]].drop_duplicates()
        mismatched = [tuple(x) for x in bad.to_numpy()]
        keep = ~summ.set_index(["dataset", "horizon", "context"]).index.isin(mismatched)
        summ = summ[keep].copy()
        wide = wide[~wide.set_index(["dataset", "horizon", "context"]).index.isin(mismatched)].copy()
    else:
        mismatched = []
    order_h = {"half": 0, "2": 1, "5": 2}
    order_c = {c: i for i, c in enumerate(CONTEXTS)}
    summ = summ.sort_values(["dataset", "horizon", "context"],
                            key=lambda s: s.map({**order_h, **order_c}).fillna(0) if s.name != "dataset" else s)
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Held-out choice: ratio to current, per participant-condition.
    ratio = wide.copy()
    for v in VARIANTS:
        ratio[v] = wide[v] / wide["current"]
    ratio["half"] = np.where(ratio["id"].astype(int) % 2 == 0, "even", "odd")
    lines = ["# Fixed-Share comparison", "",
             "RMSE is per participant, mg/dL. Ratios are variant RMSE / current HBML-SFHDF RMSE "
             "(below 1 = better than the current algorithm).", ""]
    if mismatched:
        lines += ["**Left out of this report** because the forecasts in this folder do not reproduce the HBML "
                  "means in `overleaf/tables/rmse_pivot_mean.csv` (stale or partial copy; run these where the "
                  "outputs are complete): " + ", ".join(f"{d} h={h} c={c}" for d, h, c in mismatched) + ".", ""]
    n_unk = int((summ.drop_duplicates(["dataset", "horizon", "context"])["matches_tables"] == "no").sum())
    if n_unk:
        lines += [f"**Warning:** {n_unk} condition(s) below do not reproduce the manuscript tables "
                  "(column `matches_tables` in summary.csv). Use --drop-mismatched to leave them out.", ""]
    cond = summ.groupby(["dataset", "variant"]).agg(
        conditions=("mean_rmse", "size"), mean_pct_change=("pct_change_vs_current", "mean"),
        wins=("delta_vs_current", lambda s: int((s > 0).sum()))).reset_index()
    lines += ["## Average over conditions", "",
              "| dataset | variant | conditions | mean % change in RMSE vs current | conditions where it beats current |",
              "|---|---|---|---|---|"]
    for _, r in cond.iterrows():
        lines.append(f"| {r.dataset} | {r.variant} | {r.conditions} | {r.mean_pct_change:+.2f} | {r.wins} |")
    lines += ["", "## Held-out choice (odd / even participant IDs)", "",
              "| dataset | chosen on | chosen variant | ratio on the selection half | ratio on the held-out half |",
              "|---|---|---|---|---|"]
    gm = lambda s: float(np.exp(np.nanmean(np.log(s))))
    for ds, g in ratio.groupby("dataset"):
        for dev, test in (("odd", "even"), ("even", "odd")):
            gd, gt = g[g["half"] == dev], g[g["half"] == test]
            if gd.empty or gt.empty:
                continue
            score = {v: gm(gd[v]) for v in VARIANTS}
            pick = min(score, key=score.get)
            lines.append(f"| {ds} | {dev} | {pick} | {score[pick]:.4f} | {gm(gt[pick]):.4f} |")
    lines += ["", "## Per condition (mean RMSE)", ""]
    tab = summ.pivot_table(index=["dataset", "horizon", "context"], columns="variant", values="mean_rmse")
    tab = tab[[v for v in list(VARIANTS) + ["uniform"] if v in tab.columns]]
    be = summ.drop_duplicates(["dataset", "horizon", "context"]).set_index(["dataset", "horizon", "context"])
    tab.insert(0, "best expert", be["best_expert_mean_rmse"])
    tab.insert(0, "n", be["n"])
    tab = tab.reset_index()
    tab = tab.sort_values(["dataset", "horizon", "context"],
                          key=lambda s: s.map({**order_h, **order_c}).fillna(0) if s.name != "dataset" else s)
    lines.append("| " + " | ".join(tab.columns) + " |")
    lines.append("|" + "---|" * len(tab.columns))
    for _, r in tab.iterrows():
        lines.append("| " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) + " |")
    with open(os.path.join(out_dir, "REPORT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {out_dir}/per_participant.csv, summary.csv and REPORT.md "
          f"({long.groupby(['dataset','horizon','context']).ngroups} conditions).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--dataset", choices=["weinstock", "cgmacros", "all"], default="all")
    ap.add_argument("--horizon", choices=["half", "2", "5", "all"], default="all")
    ap.add_argument("--context", choices=CONTEXTS + ["all"], default="all")
    ap.add_argument("--out", default="fs_comparison")
    ap.add_argument("--min-complete", type=float, default=0.95,
                    help="Skip a condition unless this share of its forecast files comes from the current engine.")
    ap.add_argument("--allow-legacy", action="store_true", help="Also use forecasts older than the fast-engine cutoff.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--drop-mismatched", action="store_true",
                    help="Leave out conditions whose forecasts do not reproduce the HBML means in the manuscript tables.")
    a = ap.parse_args()
    out_dir = os.path.join(a.root, a.out)
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    if not a.report_only:
        min_mtime = 0.0 if a.allow_legacy else FAST_ENGINE_CUTOFF
        for ds in (["weinstock", "cgmacros"] if a.dataset == "all" else [a.dataset]):
            for hk in (list(HORIZONS) if a.horizon == "all" else [a.horizon]):
                for c in (CONTEXTS if a.context == "all" else [a.context]):
                    part = os.path.join(out_dir, "parts", f"{ds}_h{hk}_c{c}.csv")
                    if os.path.exists(part) and not a.overwrite:
                        print(f"[skip] {ds} h={hk} c={c}: already done")
                        continue
                    df, info = run_condition(a.root, ds, hk, c, min_mtime, a.workers)
                    if info["n_files"] == 0 or info["n_fresh"] < a.min_complete * info["n_files"] or df.empty:
                        print(f"[incomplete] {ds} h={hk} c={c}: {info['n_fresh']}/{info['n_files']} current-engine "
                              f"forecast files here; run this condition where the outputs are complete")
                        continue
                    df.to_csv(part, index=False)
                    print(f"[done] {ds} h={hk} c={c}: {df['id'].nunique()} participants", flush=True)
    build_report(out_dir, a.drop_mismatched)


if __name__ == "__main__":
    main()
