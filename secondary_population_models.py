#!/usr/bin/env python3
"""secondary_population_models.py -- secondary analysis: HBML vs population-trained models.

Question: how does the purely personal HBML (no population pretraining) compare with models
trained on other people, and does adding a population model as an extra expert help?

Population experts (same architectures and features as the personal experts):
  Pop-NHITS    N-HITS trained on windows pooled from the training participants
  Pop-XGBoost  XGBoost on the same lag/delta features, pooled
They are trained once per fold and NOT updated online; at test time they forecast y[t+h]
from y[0..t] for t from the end of the burn-in, exactly like the personal experts.

Training schemes (no participant is ever in both training and test data):
  within  k-fold split of participants within the dataset (default k = 5)
  cross   train on the other dataset (Weinstock <-> CGMacros)

Stages
  train     fit the population models and write held-out forecasts to
            Outputs/secondary/population/<scheme>/<dataset>/h-<h>hr/context-<c>hr/forecasts/<id>_popforecasts.csv
  evaluate  combine with the personal-pool forecasts in Outputs/<dataset>/... (engine v2) and write
            overleaf/tables/population_comparison.csv, population_participant.csv and
            table_population_comparison.tex
Needs torch, neuralforecast and xgboost for `train` (run on the VM); `evaluate` needs NumPy/pandas.

Examples
  python3 secondary_population_models.py --stage train --dataset weinstock --horizon half
  python3 secondary_population_models.py --stage evaluate
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from ExpMethods import aggregators as agg  # noqa: E402
from ExpMethods import evaluation as ev  # noqa: E402
from ExpMethods.globals import GlobalValues as G  # noqa: E402
from fast_engine import features as ft  # noqa: E402
from fast_engine.simulation import ENGINE_VERSION, load_grid  # noqa: E402

OUT_ROOT = os.path.join(BASE, "Outputs", "secondary", "population")
TABLES = os.path.join(BASE, "overleaf", "tables")
POP = ["Pop-NHITS", "Pop-XGBoost"]
CTX = {"6": 72, "12": 144, "24": 288, "full": -1}


def folds(ids, k, seed):
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(sorted(ids)))
    return {pid: i % k for i, pid in enumerate(perm)}


def train_pop_nhits(series, h, input_size, steps, batch, lr, seed, max_windows):
    import torch
    from neuralforecast.models import NHITS
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = NHITS(h=h, input_size=input_size, max_steps=1, mlp_units=3 * [[256, 256]])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X, Y = [], []
    for g in series:
        y = g.inputs
        ends = ft.eligible_ends(g.valid, len(y) - 1, h, need_path=True)   # targets never in a long gap
        if len(ends) > max_windows:
            ends = rng.choice(ends, max_windows, replace=False)
        Xn, Yn, _ = ft.neural_windows(y, ends, input_size, h)
        X.append(Xn); Y.append(Yn)
    X = torch.as_tensor(np.vstack(X), dtype=torch.float32)
    Y = torch.as_tensor(np.vstack(Y), dtype=torch.float32)
    net.train()
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(X), batch))
        x = X[idx].unsqueeze(-1)
        out = net({"insample_y": x, "insample_mask": torch.ones_like(x), "futr_exog": None,
                   "hist_exog": None, "stat_exog": None})
        loss = torch.nn.functional.mse_loss(out.reshape(len(idx), h, -1)[:, :, 0], Y[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    return net


def nhits_predict(net, y, h, input_size, ends):
    import torch
    out = np.empty(len(ends))
    for i in range(0, len(ends), 2048):
        e = ends[i:i + 2048]
        Xn, base = ft.neural_windows(y, e, input_size)
        x = torch.as_tensor(Xn, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            o = net({"insample_y": x, "insample_mask": torch.ones_like(x), "futr_exog": None,
                     "hist_exog": None, "stat_exog": None})
        out[i:i + len(e)] = o.reshape(len(e), h, -1)[:, -1, 0].numpy() * ft.NEURAL_SCALE + base
    return out


def train_pop_xgb(series, h, rounds, seed, threads):
    import xgboost as xgb
    sets = [ft.xgb_training_set(g.inputs, len(g) - 1, h, g.valid) for g in series]
    X = np.vstack([s[0] for s in sets if s[0] is not None])
    Y = np.concatenate([s[1] for s in sets if s[0] is not None])
    params = {"max_depth": 6, "eta": 0.05, "objective": "reg:squarederror", "tree_method": "hist",
              "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": 0, "nthread": threads, "seed": seed}
    return xgb.train(params, xgb.DMatrix(X, label=Y), num_boost_round=rounds)


def stage_train(a):
    datasets = ["weinstock", "cgmacros"] if a.dataset == "all" else [a.dataset]
    horizons = ["half", "2", "5"] if a.horizon == "all" else [a.horizon]
    lo, hi = G.forecast_clip
    for ds in datasets:
        files = ev.participant_files(ds)
        test = {os.path.basename(f)[:3]: load_grid(f) for f in files}
        if a.scheme == "cross":
            other = "cgmacros" if ds == "weinstock" else "weinstock"
            pool = [load_grid(f) for f in ev.participant_files(other)]
            fold_of = {pid: 0 for pid in test}
            n_folds = 1
        else:
            fold_of = folds(test, a.folds, a.seed)
            n_folds = a.folds
        for hk in horizons:
            h = ev.HORIZON_STEPS[hk]
            input_size = CTX[a.context] if CTX[a.context] > 0 else 120
            out_dir = os.path.join(OUT_ROOT, a.scheme, ds, f"h-{hk}hr", f"context-{a.context}hr", "forecasts")
            os.makedirs(out_dir, exist_ok=True)
            for k in range(n_folds):
                held = [p for p, f in fold_of.items() if f == k]
                todo = [p for p in held if a.overwrite or not os.path.exists(os.path.join(out_dir, f"{p}_popforecasts.csv"))]
                if not todo:
                    continue
                train = pool if a.scheme == "cross" else [y for p, y in test.items() if fold_of[p] != k]
                t0 = time.perf_counter()
                net = train_pop_nhits(train, h, input_size, a.nhits_steps, 256, 1e-3, a.seed, a.max_windows)
                booster = train_pop_xgb(train, h, a.xgb_rounds, a.seed, a.threads)
                fit_s = time.perf_counter() - t0
                import xgboost as xgb
                for p in todo:
                    g = test[p]
                    y, T = g.inputs, len(g)
                    # same origins as the personal engine (real reading, past burn-in, 1 h into segment)
                    ends = g.origins(G.burn_in_steps, h, G.min_segment_history)
                    fc = {m: np.zeros(T) for m in POP}
                    fc["Pop-NHITS"][ends + h] = np.clip(nhits_predict(net, y, h, input_size, ends), lo, hi)
                    fc["Pop-XGBoost"][ends + h] = np.clip(
                        y[ends] + booster.predict(xgb.DMatrix(ft.xgb_features(y, ends))), lo, hi)
                    pd.DataFrame(fc).to_csv(os.path.join(out_dir, f"{p}_popforecasts.csv"), index=False)
                print(f"[{a.scheme}] {ds} h={hk} fold {k + 1}/{n_folds}: trained on {len(train)} participants "
                      f"in {fit_s:.0f}s, wrote {len(todo)} held-out forecast files", flush=True)


def stage_evaluate(a):
    import json
    ret_path = os.path.join(TABLES, "rmse_participant.csv")
    retained = {}
    if os.path.exists(ret_path):
        rdf = pd.read_csv(ret_path, dtype={"id": str, "context": str, "horizon": str})
        for (rds, rhz, rctx), rg in rdf.groupby(["dataset", "horizon", "context"]):
            retained[(rds, "half" if rhz in ("0.5", "half") else rhz, rctx)] = set(rg["id"].str.zfill(3))
    recs, rows = [], []
    for scheme in ("within", "cross"):
        for ds in ("weinstock", "cgmacros"):
            for hk in ("half", "2", "5"):
                pop_dir = os.path.join(OUT_ROOT, scheme, ds, f"h-{hk}hr", f"context-{a.context}hr")
                if not os.path.isdir(pop_dir):
                    continue
                h = ev.HORIZON_STEPS[hk]
                pers_dir = os.path.join(BASE, "Outputs", ds, f"h-{hk}hr", f"context-{a.context}hr")
                keep = retained.get((ds, hk, str(a.context)))
                for pf in ev.forecast_files(pop_dir, "_popforecasts.csv"):
                    pid = os.path.basename(pf)[:3]
                    if keep is not None and pid not in keep:
                        continue
                    ff = os.path.join(pers_dir, "forecasts", f"{pid}_forecasts.csv")
                    meta = ev.read_meta(ff) if os.path.exists(ff) else None
                    if not meta or meta["settings"].get("engine_version") != ENGINE_VERSION:
                        continue
                    y = ev.load_targets(ds, pid)
                    Fp = pd.read_csv(ff)
                    Fq = pd.read_csv(pf)
                    experts = [e for e in G.experts if e in Fp.columns]
                    n = min(len(Fp), len(Fq), len(y))
                    P = Fp[experts].to_numpy(float)[:n]
                    Q = Fq[POP].to_numpy(float)[:n]
                    yy = y[:n]
                    rec = dict(scheme=scheme, dataset=ds, horizon=ev.HORIZON_LABEL[hk], context=a.context, id=pid)
                    mask = ev.scored_mask(np.column_stack([P, Q]), yy)
                    with np.errstate(invalid="ignore"):
                        for i, e in enumerate(experts):
                            rec[e] = ev.rmse((P[:, i] - yy) ** 2, mask)
                        for i, e in enumerate(POP):
                            rec[e] = ev.rmse((Q[:, i] - yy) ** 2, mask)
                    for tag, F in (("HBML", P), ("HBML+Pop", np.column_stack([P, Q]))):
                        L = agg.online_losses(F, yy, G.burn_in_steps, h)
                        _, l = agg.run_named(G.primary_hbml, F, L, yy, start=G.burn_in_steps, horizon=h)
                        rec[tag] = ev.rmse(l, mask)
                    recs.append(rec)
    if not recs:
        raise SystemExit("Nothing to evaluate: run --stage train first (and the main v2 simulations).")
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(TABLES, "population_participant.csv"), index=False)
    experts = [e for e in G.experts if e in df.columns]
    for (scheme, ds, hz, c), g in df.groupby(["scheme", "dataset", "horizon", "context"], sort=False):
        best = min(experts, key=lambda e: g[e].mean())
        r = dict(scheme=scheme, dataset=ds, horizon=hz, context=c, n=len(g), best_personal_expert=best,
                 best_personal_rmse=g[best].mean())
        for m in POP + ["HBML", "HBML+Pop"]:
            r[f"{m}_rmse"] = g[m].mean()
            r[f"{m}_sd"] = g[m].std(ddof=1)
        for ref, alt, key in (("Pop-NHITS", "HBML", "hbml_vs_popnhits"), ("Pop-XGBoost", "HBML", "hbml_vs_popxgb"),
                              ("HBML", "HBML+Pop", "pop_added_vs_hbml"), (best, "HBML+Pop", "pop_added_vs_best")):
            for k_, v in ev.paired_summary(g[ref], g[alt]).items():
                r[f"{key}_{k_}"] = v
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "population_comparison.csv"), index=False)
    write_tex(out)
    print(out[["scheme", "dataset", "horizon", "n", "Pop-NHITS_rmse", "Pop-XGBoost_rmse", "HBML_rmse",
               "HBML+Pop_rmse", "best_personal_rmse"]].round(2).to_string(index=False))


def write_tex(df):
    lab = {"weinstock": "Weinstock", "cgmacros": "CGMacros"}
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{lllcccccc}", r"\toprule",
             r"Training & Dataset & Horizon (hr) & $N$ & Pop-NHITS & Pop-XGBoost & HBML & HBML + population & $p$ (added) \\",
             r"\midrule"]
    for _, r in df.iterrows():
        vals = [r["Pop-NHITS_rmse"], r["Pop-XGBoost_rmse"], r["HBML_rmse"], r["HBML+Pop_rmse"]]
        lo = min(vals)
        cells = [(f"\\textbf{{{v:.2f}}}" if abs(v - lo) < 5e-3 else f"{v:.2f}") for v in vals]
        lines.append(f"{'Within-cohort' if r.scheme == 'within' else 'Cross-cohort'} & {lab[r.dataset]} & {r.horizon} & "
                     f"{int(r.n)} & " + " & ".join(cells) + f" & {ev.tex_p(r['pop_added_vs_hbml_wilcoxon_p'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Personal HBML compared with population-trained models.} Mean participant-level RMSE "
              r"(mg/dL), full-history context. Pop-NHITS and Pop-XGBoost are trained on other participants (within-cohort: "
              r"participant-level k-fold; cross-cohort: trained on the other dataset) and applied without online updates. "
              r"HBML uses only each participant's own data; HBML + population adds the two population models to its expert "
              r"pool. $p$: two-sided Wilcoxon signed-rank test of HBML + population against HBML.}",
              r"\label{tab:population_comparison}", r"\end{table*}"]
    with open(os.path.join(TABLES, "table_population_comparison.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train", "evaluate"], required=True)
    ap.add_argument("--scheme", choices=["within", "cross"], default="within")
    ap.add_argument("--dataset", choices=["weinstock", "cgmacros", "all"], default="all")
    ap.add_argument("--horizon", choices=["half", "2", "5", "all"], default="all")
    ap.add_argument("--context", default="full", choices=list(CTX))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nhits_steps", type=int, default=3000)
    ap.add_argument("--xgb_rounds", type=int, default=400)
    ap.add_argument("--max_windows", type=int, default=4000, help="Training windows sampled per participant.")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    try:
        import torch
        torch.set_num_threads(a.threads)
    except ImportError:
        pass
    stage_train(a) if a.stage == "train" else stage_evaluate(a)


if __name__ == "__main__":
    main()
