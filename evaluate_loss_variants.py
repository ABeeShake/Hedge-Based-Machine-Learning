#!/usr/bin/env python3
"""evaluate_loss_variants.py -- Alternative online loss functions & mixture learners on cached v2.3 forecasts.

Runs on existing cached Outputs/<dataset>/h-*hr/context-*hr/forecasts/ (zero base-expert retraining)
across development (weinstock, cgmacros) and untouched GlucoBench (hall, dubosson) cohorts to test:
  1. MAE / Huber online Hedge losses (preventing O(e^4) denominator freeze on volatile T1D streams)
  2. Asymmetric hypoglycemia & Pinball-20% online Hedge losses (30-min matched-FAR hypo recall vs Persistence)
  3. Mixture-Loss Exponentiated Gradient (HBML-MixEG & interleaved) optimizing ensemble loss directly
     to test capturing cross-expert error cancellation vs Uniform (2-hr Weinstock/Dubosson, 30-min Hall)

Usage:
  python3 evaluate_loss_variants.py --datasets weinstock,cgmacros,hall,dubosson --n-boot 2000
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ExpMethods import aggregators as agg  # noqa: E402
from ExpMethods import evaluation as ev  # noqa: E402
from ExpMethods.globals import GlobalValues as G  # noqa: E402
import calculate_statistics as cstat  # noqa: E402
import evaluate_clinical_and_weights as ecw  # noqa: E402
import analyze_pool_mechanism as apm  # noqa: E402

START = int(G.eval_start)
MIN_SCORED = int(G.min_scored_points)
ENGINE_VERSION = ecw.ENGINE_VERSION

VARIANTS = (
    "HBML-SE",
    "HBML-MAE",
    "HBML-Huber15",
    "HBML-Huber25",
    "HBML-Pinball20",
    "HBML-AsymHypo",
    "HBML-MixEG",
    "HBML-MixEG-MAE",
    "HBML-MixEG-interleaved",
    "Uniform",
    "FTL",
)


def eval_variant(name, F, y, mask, h):
    """Return (predictions, rmse_val, mae_val) for one named online loss/mixture variant."""
    if name == "Uniform":
        pred = np.nanmean(F, axis=1)
    elif name == "FTL":
        L = agg.online_losses(F, y, START, h, loss_fn="squared")
        pred, _ = agg.run_named("FTL", F, L, y, start=START, horizon=h)
    elif name.startswith("HBML-MixEG"):
        lfn = "mae" if name.endswith("-MAE") else "squared"
        if name.endswith("-interleaved"):
            pred, _ = agg.run_interleaved_mixture_eg(F, np.zeros_like(F), y, start=START, horizon=h, loss_fn=lfn)
        else:
            pred, _ = agg.run_mixture_eg(F, np.zeros_like(F), y, start=START, horizon=h, loss_fn=lfn)
    else:
        loss_map = {
            "HBML-SE": "squared",
            "HBML-MAE": "mae",
            "HBML-Huber15": "huber15",
            "HBML-Huber25": "huber25",
            "HBML-Pinball20": "pinball20",
            "HBML-AsymHypo": "asym_hypo",
        }
        L = agg.online_losses(F, y, START, h, loss_fn=loss_map[name])
        pred, _ = agg.run_named(G.primary_hbml, F, L, y, start=START, horizon=h)

    err = pred[mask] - y[mask]
    ok = np.isfinite(err)
    rmse_v = float(np.sqrt(np.mean(err[ok] ** 2))) if ok.any() else float("nan")
    mae_v = float(np.mean(np.abs(err[ok]))) if ok.any() else float("nan")
    return pred, rmse_v, mae_v


def load_eligible_participants(ds, h_str, ctx, outputs_root):
    fc_dir = os.path.join(outputs_root, ds, f"h-{h_str}hr", f"context-{ctx}hr", "forecasts")
    out = []
    for fc_path in sorted(glob.glob(os.path.join(fc_dir, "*_forecasts.csv"))):
        base = os.path.basename(fc_path)
        if base[3:] != "_forecasts.csv":
            continue
        pid = base[:3]
        meta = ev.read_meta(fc_path)
        if not meta or meta.get("settings", {}).get("engine_version") != ENGINE_VERSION:
            continue
        y = ev.load_targets(ds, pid, root=BASE_DIR)
        if y is None:
            continue
        cols, F = ecw.load_forecasts_matrix(fc_path)
        n = min(len(y), len(F))
        y, F = y[:n], F[:n]
        mask = ev.scored_mask(F, y)
        if int(mask.sum()) < MIN_SCORED:
            continue
        out.append(dict(pid=pid, y=y, F=F, cols=cols, mask=mask))
    return out


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_loss_variants(datasets, outputs_root, out_dir, n_boot=2000, full_only=False):
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    hypo_sweep_rows = []
    matched_far_rows = []

    t2d_pids_cache = {}
    def _d_type(ds, pid):
        if ds not in t2d_pids_cache:
            try:
                t2d_pids_cache[ds] = set(ecw.load_t2d_pids(ds))
            except Exception:
                t2d_pids_cache[ds] = set()
        return 2 if pid in t2d_pids_cache[ds] else 1

    ctxs = ["full"] if full_only else ecw.CONTEXTS
    for ds in datasets:
        for h_str, h_disp, h_steps in ecw.HORIZONS:
            for ctx in ctxs:
                parts = load_eligible_participants(ds, h_str, ctx, outputs_root)
                if not parts:
                    continue
                cols = parts[0]["cols"]
                rmses = {v: [] for v in VARIANTS}
                maes = {v: [] for v in VARIANTS}
                exp_rmses = {c: [] for c in cols}
                preds_by_var = {v: [] for v in VARIANTS}

                for p in parts:
                    F, y, mask = p["F"], p["y"], p["mask"]
                    L_sq = agg.online_losses(F, y, START, h_steps, loss_fn="squared")
                    for j, c in enumerate(cols):
                        exp_rmses[c].append(ev.rmse(L_sq[:, j], mask))
                    for v in VARIANTS:
                        pr, rv, mv = eval_variant(v, F, y, mask, h_steps)
                        rmses[v].append(rv)
                        maes[v].append(mv)
                        if h_steps == 6 and ctx == "full":
                            preds_by_var[v].append((p, pr))

                best_exp_name = min(cols, key=lambda c: float(np.mean(exp_rmses[c])))
                arr_se = np.asarray(rmses["HBML-SE"], dtype=float)
                arr_uni = np.asarray(rmses["Uniform"], dtype=float)
                arr_best = np.asarray(exp_rmses[best_exp_name], dtype=float)

                for v in VARIANTS:
                    av = np.asarray(rmses[v], dtype=float)
                    am = np.asarray(maes[v], dtype=float)
                    d_se = av - arr_se
                    lo_se, hi_se = apm.boot_ci(d_se, n_boot=n_boot, seed=11)
                    st_best = cstat.paired_stats(av, arr_best, n_boot=n_boot)
                    d_uni = av - arr_uni
                    lo_uni, hi_uni = apm.boot_ci(d_uni, n_boot=n_boot, seed=13)
                    summary_rows.append({
                        "dataset": ds,
                        "horizon": h_disp,
                        "context": ctx,
                        "n": len(parts),
                        "variant": v,
                        "mean_rmse": float(np.mean(av)),
                        "sd_rmse": float(np.std(av, ddof=1)) if len(av) > 1 else np.nan,
                        "mean_mae": float(np.mean(am)),
                        "diff_vs_se": float(np.mean(d_se)),
                        "ci95_low_vs_se": lo_se,
                        "ci95_high_vs_se": hi_se,
                        "best_expert": best_exp_name,
                        "best_expert_rmse": float(np.mean(arr_best)),
                        "diff_best_minus_var": float(st_best["paired_diff_mean"]),
                        "wilcoxon_p_vs_best": st_best["wilcoxon_p"],
                        "tost_boot_p": st_best["tost_boot_p"],
                        "tost_rel5_p": st_best["tost_rel5_p"],
                        "diff_vs_uniform": float(np.mean(d_uni)),
                        "ci95_low_vs_uniform": lo_uni,
                        "ci95_high_vs_uniform": hi_uni,
                    })

                # 30-min full-history matched-FAR hypoglycemia evaluation across all loss variants + Persistence + ARIMA
                if h_steps == 6 and ctx == "full":
                    pp_rows = []
                    sweep_models = list(VARIANTS) + ["Persistence", "BestExpert"]
                    pers_idx = cols.index("Persistence")
                    best_idx = cols.index(best_exp_name)
                    for p_idx, p in enumerate(parts):
                        y, F, mask, pid = p["y"], p["F"], p["mask"], p["pid"]
                        dt_type = _d_type(ds, pid)
                        model_preds = {v: preds_by_var[v][p_idx][1] for v in VARIANTS}
                        model_preds["Persistence"] = F[:, pers_idx]
                        model_preds["BestExpert"] = F[:, best_idx]
                        for m_name in sweep_models:
                            pr = model_preds[m_name]
                            for thr in ecw.SWEEP_THRESHOLDS:
                                r = ecw.hypo_events_patient(
                                    y, pr, mask, h_steps, thresh=thr,
                                    compute_parkes=False, diabetes_type=dt_type,
                                    warning_minutes=ecw.WARNING_MINUTES,
                                )
                                r["model"] = m_name
                                r["threshold_mgdl"] = thr
                                r["pid"] = pid
                                pp_rows.append(r)

                    for m_name in sweep_models:
                        for thr in ecw.SWEEP_THRESHOLDS:
                            sub = [r for r in pp_rows if r["model"] == m_name and r["threshold_mgdl"] == thr]
                            ep_sum = sum(r["n_episodes"] for r in sub)
                            det_sum = sum(r["n_detected"] for r in sub)
                            ch_sum = sum(r["chance_detected_sum"] for r in sub)
                            days_sum = sum(r["days"] for r in sub)
                            fc_sum = sum(r["false_clusters"] for r in sub)
                            tc_sum = sum(r["true_clusters"] for r in sub)
                            cl_sum = sum(r["n_alert_clusters"] for r in sub)
                            sens = 100.0 * det_sum / ep_sum if ep_sum > 0 else np.nan
                            ch_s = 100.0 * ch_sum / ep_sum if ep_sum > 0 else np.nan
                            hypo_sweep_rows.append({
                                "dataset": ds, "model": m_name, "threshold_mgdl": thr,
                                "n_participants": len(parts), "n_episodes": ep_sum,
                                "episode_sensitivity_pct": sens,
                                "above_chance_pp": sens - ch_s if ep_sum > 0 else np.nan,
                                "cluster_ppv_pct": 100.0 * tc_sum / cl_sum if cl_sum > 0 else np.nan,
                                "false_alerts_per_day": fc_sum / days_sum if days_sum > 0 else np.nan,
                            })

                    total_ep = sum(r["n_episodes"] for r in pp_rows if r["model"] == "Persistence" and r["threshold_mgdl"] == 70.0)
                    if total_ep >= 10:
                        mfar = ecw.matched_far_comparison(
                            pp_rows,
                            far_targets=(0.45, 0.55, 0.65, 0.75, 0.85, 1.0),
                            warning_minutes=ecw.WARNING_MINUTES,
                            n_boot=min(n_boot, 2000),
                            seed=7,
                            reference="Persistence",
                        )
                        for r in mfar:
                            r["dataset"] = ds
                            dval = r.pop("diff_vs_reference_pp", np.nan)
                            dlo = r.pop("diff_ci95_low", np.nan)
                            dhi = r.pop("diff_ci95_high", np.nan)
                            if dval is not None and np.isfinite(dval):
                                r["diff_vs_persistence_pp"] = -float(dval)
                                r["diff_ci95_low_vs_pers"] = -float(dhi) if np.isfinite(dhi) else np.nan
                                r["diff_ci95_high_vs_pers"] = -float(dlo) if np.isfinite(dlo) else np.nan
                            else:
                                r["diff_vs_persistence_pp"] = np.nan
                                r["diff_ci95_low_vs_pers"] = np.nan
                                r["diff_ci95_high_vs_pers"] = np.nan
                            matched_far_rows.append(r)

    if summary_rows:
        _write_csv(os.path.join(out_dir, "loss_variants_rmse_summary.csv"), summary_rows, list(summary_rows[0].keys()))
    if hypo_sweep_rows:
        _write_csv(os.path.join(out_dir, "loss_variants_hypo_sweep.csv"), hypo_sweep_rows, list(hypo_sweep_rows[0].keys()))
    if matched_far_rows:
        _write_csv(os.path.join(out_dir, "loss_variants_hypo_matched_far.csv"), matched_far_rows, list(matched_far_rows[0].keys()))

    md = ["# Alternative Loss & Mixture-Rule Simulation Report (Cached Engine v2.3)", ""]
    md.append("## 1. Full-History RMSE Across Alternative Losses & Mixture-EG")
    md.append("| Cohort | Horizon | HBML-SE | HBML-MAE | HBML-Huber15 | HBML-Huber25 | HBML-MixEG | MixEG-Interl | Uniform | BestExpert |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    lu = {(r["dataset"], r["horizon"], r["context"], r["variant"]): r for r in summary_rows}
    for ds in datasets:
        for _, hd, _ in ecw.HORIZONS:
            r0 = lu.get((ds, hd, "full", "HBML-SE"))
            if not r0:
                continue
            vals = [f"{lu[(ds, hd, 'full', v)]['mean_rmse']:.2f}" for v in
                    ("HBML-SE", "HBML-MAE", "HBML-Huber15", "HBML-Huber25", "HBML-MixEG", "HBML-MixEG-interleaved", "Uniform")]
            md.append(f"| {ds} | {hd} | " + " | ".join(vals) + f" | {r0['best_expert']} ({r0['best_expert_rmse']:.2f}) |")

    if matched_far_rows:
        md += ["", "## 2. 30-Min Pre-Onset Hypoglycemia Sensitivity at Matched FAR (diff vs Persistence in pp)"]
        md.append("| Cohort | Min Warn | FAR/day | HBML-SE | HBML-AsymHypo | HBML-Pinball20 | HBML-MixEG | Persistence |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        mlu = {(r["dataset"], int(r["min_warning_min"]), round(float(r["false_alerts_per_day"]), 2), r["model"]): r
               for r in matched_far_rows}
        for ds in datasets:
            for warn in (0, 15):
                for far in (0.65, 0.75, 1.0):
                    if not (ds, warn, far, "Persistence") in mlu:
                        continue
                    def _c(m):
                        rr = mlu.get((ds, warn, far, m))
                        if not rr or not np.isfinite(rr["sensitivity_pct"]):
                            return "---"
                        dv = rr.get("diff_vs_persistence_pp")
                        dstr = f" ({dv:+.1f})" if dv is not None and np.isfinite(dv) else ""
                        return f"{rr['sensitivity_pct']:.1f}%{dstr}"
                    md.append(f"| {ds} | >={max(warn,5)}m | {far:.2f} | {_c('HBML-SE')} | {_c('HBML-AsymHypo')} | {_c('HBML-Pinball20')} | {_c('HBML-MixEG')} | {_c('Persistence')} |")

    with open(os.path.join(out_dir, "loss_variants_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote alternative-loss results to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="weinstock,cgmacros,hall,dubosson")
    ap.add_argument("--outputs-root", default=os.path.join(BASE_DIR, "Outputs"))
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "loss_results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--full-only", action="store_true")
    a = ap.parse_args()
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    run_loss_variants(datasets, a.outputs_root, a.out_dir, n_boot=a.n_boot, full_only=a.full_only)


if __name__ == "__main__":
    main()
