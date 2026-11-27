#!/usr/bin/env python3
"""validate_engine_outputs.py -- Zero-retraining validation and missing-data audits on cached Outputs/.

Runs entirely from saved engine-v2.3 forecast CSVs, `.meta.json` telemetry, and raw participant
5-minute grids:
  1. Fit-window & early-stopping audit (`validation_fit_windows.csv`):
     Summarizes `fit_windows_train_median`, `fit_windows_train_min`, `fit_windows_val_median`,
     `n_fits`, `n_fallback`, and `n_fits_without_early_stopping` across participants.
  2. Participant eligibility & common-cohort sensitivity (`validation_cohort_sensitivity.csv`):
     Compares RMSE and paired Wilcoxon / bootstrap TOST (+/- equiv_margin_mgdl) across
     (a) `retained_ge288` (`n_scored >= 288`), (b) `all_scorable_ge1` (`n_scored >= 1`, e.g. N=200
     on Weinstock), and (c) `common_all_conditions` (strict ID intersection across all 12
     conditions per dataset, e.g. N=195 on Weinstock).
  3. Missing-data & gap-burden stratification (`validation_gap_burden.csv`):
     Stratifies retained participants by total missing-slot fraction tertiles (plus bridged-input
     share and segment count) at full-history context across horizons, and reports cohort-mean
     RMSE + 30-min pre-onset hypoglycaemia sensitivity, cluster PPV, and false-alert rate per tertile.

Usage:
  python3 validate_engine_outputs.py [--datasets weinstock,cgmacros] [--n-boot 2000]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ExpMethods import aggregators as agg  # noqa: E402
from ExpMethods import evaluation as ev  # noqa: E402
from ExpMethods.globals import GlobalValues as G  # noqa: E402
import calculate_statistics as cstat  # noqa: E402
import evaluate_clinical_and_weights as ecw  # noqa: E402

START = int(G.burn_in_steps)
MIN_SCORED = int(G.min_scored_points)
HORIZONS = [("half", "0.5", 6), ("2", "2", 24), ("5", "5", 60)]
CONTEXTS = ["6", "12", "24", "full"]


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def eval_participant_unfiltered(dataset, pid, fc_path, h_steps, min_scored=1):
    """Like ecw.run_participant but retains any participant with n_scored >= min_scored."""
    meta = ev.read_meta(fc_path)
    if not meta or meta.get("settings", {}).get("engine_version") != ecw.ENGINE_VERSION:
        return None
    y = ev.load_targets(dataset, pid, root=BASE_DIR)
    if y is None:
        return None
    cols, f_mat = ecw.load_forecasts_matrix(fc_path)
    if len(cols) < 2:
        return None
    n = min(len(y), len(f_mat))
    if n <= int(G.eval_start) + int(G.eval_tail) + 10:
        return None
    y = y[:n]
    f_mat = f_mat[:n]
    mask = ev.scored_mask(f_mat, y)
    n_scored = int(mask.sum())
    if n_scored < min_scored:
        return None
    l_mat = agg.online_losses(f_mat, y, START, h_steps)
    pred_prim, loss_prim = agg.run_named(ecw.PRIMARY_HBML, f_mat, l_mat, y, start=START, horizon=h_steps)
    return {
        "pid": pid,
        "meta": meta,
        "y": y,
        "F": f_mat,
        "cols": cols,
        "scored_mask": mask,
        "n_scored": n_scored,
        "pred_primary": pred_prim,
        "hbml_rmse": ev.rmse(loss_prim, mask),
        "exp_rmses": {cols[k]: ev.rmse(l_mat[:, k], mask) for k in range(len(cols))},
    }


def summarize_fit_windows(datasets, outputs_dir):
    """Audit neural and learned-expert refit window counts & early-stopping skips."""
    rows = []
    for ds in datasets:
        for h_str, h_disp, _ in HORIZONS:
            for ctx in CONTEXTS:
                cond_dir = os.path.join(outputs_dir, ds, f"h-{h_str}hr", f"context-{ctx}hr")
                files = ev.forecast_files(cond_dir) if os.path.isdir(cond_dir) else []
                by_exp = defaultdict(list)
                for fc_path in files:
                    meta = ev.read_meta(fc_path)
                    if not meta or meta.get("settings", {}).get("engine_version") != ecw.ENGINE_VERSION:
                        continue
                    timings = meta.get("timings", {})
                    for exp, stats in timings.items():
                        if isinstance(stats, dict):
                            by_exp[exp].append(stats)
                for exp, slist in sorted(by_exp.items()):
                    if not any(k in s for s in slist for k in ("fit_windows_train_median", "n_fits")):
                        continue
                    def _vals(key):
                        return np.array([float(s[key]) for s in slist if key in s and s[key] is not None], dtype=float)
                    tr_med = _vals("fit_windows_train_median")
                    tr_min = _vals("fit_windows_train_min")
                    va_med = _vals("fit_windows_val_median")
                    n_fits = _vals("n_fits")
                    n_noes = _vals("n_fits_without_early_stopping")
                    n_fb = _vals("n_fallback")
                    tot_fits = float(np.nansum(n_fits)) if len(n_fits) else 0.0
                    tot_noes = float(np.nansum(n_noes)) if len(n_noes) else 0.0
                    rows.append({
                        "dataset": ds,
                        "horizon": h_disp,
                        "context": ctx,
                        "expert": exp,
                        "n_participants": len(slist),
                        "train_win_med_of_medians": float(np.nanmedian(tr_med)) if len(tr_med) else float("nan"),
                        "train_win_p10_of_medians": float(np.nanpercentile(tr_med, 10)) if len(tr_med) else float("nan"),
                        "train_win_global_min": float(np.nanmin(tr_min)) if len(tr_min) else float("nan"),
                        "val_win_med_of_medians": float(np.nanmedian(va_med)) if len(va_med) else float("nan"),
                        "val_win_p10_of_medians": float(np.nanpercentile(va_med, 10)) if len(va_med) else float("nan"),
                        "total_refits": int(tot_fits),
                        "total_fallbacks": int(np.nansum(n_fb)) if len(n_fb) else 0,
                        "refits_without_early_stopping": int(tot_noes),
                        "pct_refits_without_early_stopping": (100.0 * tot_noes / tot_fits) if tot_fits > 0 else float("nan"),
                    })
    return rows


def compute_gap_profile(dataset, pid, root=BASE_DIR):
    """Missing-slot share, bridged-input share, and segment count on the 5-minute grid."""
    csv_path = os.path.join(root, dataset, f"{pid}.csv")
    if not os.path.exists(csv_path):
        return None
    g = ev.load_grid(dataset, pid, root=root)
    n_slots = len(g.raw)
    n_obs = int(g.observed.sum())
    n_valid = int(g.valid.sum())
    n_bridged = int((g.valid & ~g.observed).sum())
    return {
        "n_slots": n_slots,
        "n_obs": n_obs,
        "missing_pct": 100.0 * (1.0 - n_obs / max(n_slots, 1)),
        "bridged_of_valid_pct": 100.0 * n_bridged / max(n_valid, 1),
        "n_segments": int(g.segment[-1]) + 1 if n_slots > 0 else 0,
    }


def run_validation(datasets, outputs_dir, out_dir, n_boot=2000):
    os.makedirs(out_dir, exist_ok=True)
    fit_rows = summarize_fit_windows(datasets, outputs_dir)

    cohort_rows = []
    gap_rows = []
    gap_cache = {}

    for ds in datasets:
        cond_cache = {}
        retained_sets = []
        for h_str, h_disp, h_steps in HORIZONS:
            for ctx in CONTEXTS:
                cond_dir = os.path.join(outputs_dir, ds, f"h-{h_str}hr", f"context-{ctx}hr")
                files = ev.forecast_files(cond_dir) if os.path.isdir(cond_dir) else []
                p_all = []
                for fc_path in files:
                    pid = os.path.basename(fc_path)[:3]
                    res = eval_participant_unfiltered(ds, pid, fc_path, h_steps, min_scored=1)
                    if res:
                        p_all.append(res)
                if p_all:
                    cond_cache[(h_str, h_disp, h_steps, ctx)] = p_all
                    retained_sets.append({r["pid"] for r in p_all if r["n_scored"] >= MIN_SCORED})

        common_ids = set.intersection(*retained_sets) if retained_sets else set()

        for (h_str, h_disp, h_steps, ctx), p_all in cond_cache.items():
            # Sensitivity comparison across participant sets
            subsets = [
                ("retained_ge288", [r for r in p_all if r["n_scored"] >= MIN_SCORED]),
                ("all_scorable_ge1", p_all),
                ("common_all_conditions", [r for r in p_all if r["pid"] in common_ids]),
            ]
            for label, sub in subsets:
                if not sub:
                    continue
                cols = sub[0]["cols"]
                hb = np.array([r["hbml_rmse"] for r in sub], dtype=float)
                exp_means = {c: float(np.mean([r["exp_rmses"][c] for r in sub])) for c in cols}
                best_exp = min(exp_means, key=exp_means.get)
                be = np.array([r["exp_rmses"][best_exp] for r in sub], dtype=float)
                st = cstat.paired_stats(be, hb, n_boot=n_boot)
                cohort_rows.append({
                    "dataset": ds,
                    "horizon": h_disp,
                    "context": ctx,
                    "cohort_set": label,
                    "n": len(sub),
                    "best_expert": best_exp,
                    "hbml_rmse": float(np.mean(hb)),
                    "best_expert_rmse": exp_means[best_exp],
                    "mean_diff_best_minus_hbml": st["paired_diff_mean"],
                    "wilcoxon_p": st["wilcoxon_p"],
                    "tost_boot_p": st["tost_boot_p"],
                    "equivalent_1mgdl": st["equiv_boot"],
                })

            # Gap-burden stratification on retained participants in full-history context
            if ctx == "full":
                ret = [r for r in p_all if r["n_scored"] >= MIN_SCORED]
                enriched = []
                for r in ret:
                    key = (ds, r["pid"])
                    if key not in gap_cache:
                        gap_cache[key] = compute_gap_profile(ds, r["pid"], root=BASE_DIR)
                    gp = gap_cache[key]
                    if gp is not None:
                        enriched.append((gp["missing_pct"], gp, r))
                enriched.sort(key=lambda x: x[0])
                if len(enriched) >= 6:
                    tertile_splits = np.array_split(np.arange(len(enriched)), 3)
                    tertile_names = ["T1_LowGap", "T2_MedGap", "T3_HighGap"]
                else:
                    tertile_splits = [np.arange(len(enriched))]
                    tertile_names = ["All"]

                cols = ret[0]["cols"]
                for tname, idxs in zip(tertile_names, tertile_splits):
                    if len(idxs) == 0:
                        continue
                    sub_items = [enriched[i] for i in idxs]
                    sub_r = [it[2] for it in sub_items]
                    hb_mean = float(np.mean([r["hbml_rmse"] for r in sub_r]))
                    exp_means = {c: float(np.mean([r["exp_rmses"][c] for r in sub_r])) for c in cols}
                    best_exp = min(exp_means, key=exp_means.get)

                    ep_tot = det_tot = fc_tot = tc_tot = 0
                    chance_tot = days_tot = 0.0
                    for r in sub_r:
                        dtype = 2 if (ds == "cgmacros" and r["pid"] in ecw.T2D_IDS) else 1
                        ev_r = ecw.evaluate_hypo_trajectory(
                            r["y"], r["pred_primary"], r["scored_mask"], h_steps,
                            diabetes_type=dtype, compute_parkes=False,
                        )
                        if not ev_r:
                            continue
                        ep_tot += ev_r["n_episodes"]
                        det_tot += ev_r["n_detected"]
                        chance_tot += ev_r["chance_detected_sum"]
                        days_tot += ev_r["days"]
                        fc_tot += ev_r["false_clusters"]
                        tc_tot += ev_r["true_clusters"]
                    ep_sens = 100.0 * det_tot / ep_tot if ep_tot > 0 else float("nan")
                    ch_sens = 100.0 * chance_tot / ep_tot if ep_tot > 0 else float("nan")
                    cl_tot = tc_tot + fc_tot
                    gap_rows.append({
                        "dataset": ds,
                        "horizon": h_disp,
                        "context": ctx,
                        "stratum": tname,
                        "n": len(sub_r),
                        "mean_missing_pct": float(np.mean([it[1]["missing_pct"] for it in sub_items])),
                        "max_missing_pct": float(np.max([it[1]["missing_pct"] for it in sub_items])),
                        "mean_bridged_pct": float(np.mean([it[1]["bridged_of_valid_pct"] for it in sub_items])),
                        "mean_segments": float(np.mean([it[1]["n_segments"] for it in sub_items])),
                        "hbml_rmse": hb_mean,
                        "best_expert": best_exp,
                        "best_expert_rmse": exp_means[best_exp],
                        "diff_best_minus_hbml": exp_means[best_exp] - hb_mean,
                        "n_episodes": ep_tot,
                        "hypo_sens_pct": ep_sens,
                        "above_chance_pp": (ep_sens - ch_sens) if ep_tot > 0 else float("nan"),
                        "cluster_ppv_pct": 100.0 * tc_tot / cl_tot if cl_tot > 0 else float("nan"),
                        "false_alerts_per_day": fc_tot / max(days_tot, 1e-6),
                    })

    write_csv(os.path.join(out_dir, "validation_fit_windows.csv"), fit_rows)
    write_csv(os.path.join(out_dir, "validation_cohort_sensitivity.csv"), cohort_rows)
    write_csv(os.path.join(out_dir, "validation_gap_burden.csv"), gap_rows)

    # Concise Markdown summary
    md = [
        f"# Cached Engine v{ecw.ENGINE_VERSION} Validation & Missing-Data Audit",
        "",
        "## 1. Cohort Retention Sensitivity (`retained_ge288` vs `all_scorable_ge1` vs `common_all_conditions`)",
        "| Dataset | Horizon | Context | Cohort Set | N | HBML RMSE | Best Expert | Diff | TOST p |",
        "|---|---|---|---|---:|---:|---|---:|---:|",
    ]
    for r in cohort_rows:
        if r["context"] == "full":
            md.append(
                f"| {r['dataset']} | {r['horizon']} hr | {r['context']} | {r['cohort_set']} | {r['n']} | "
                f"{r['hbml_rmse']:.2f} | {r['best_expert']} ({r['best_expert_rmse']:.2f}) | "
                f"{r['mean_diff_best_minus_hbml']:+.2f} | {r['tost_boot_p']:.4g} |"
            )
    if gap_rows:
        md += [
            "",
            "## 2. Gap-Burden Stratification (Full-History Context)",
            "| Dataset | Horizon | Stratum | N | Missing (%) | Bridged (%) | HBML RMSE | Best RMSE | Diff | Hypo Sens (%) | Above Chance (pp) | Cluster PPV (%) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in gap_rows:
            md.append(
                f"| {r['dataset']} | {r['horizon']} hr | {r['stratum']} | {r['n']} | {r['mean_missing_pct']:.1f} | "
                f"{r['mean_bridged_pct']:.1f} | {r['hbml_rmse']:.2f} | {r['best_expert_rmse']:.2f} | "
                f"{r['diff_best_minus_hbml']:+.2f} | {r['hypo_sens_pct']:.1f} | {r['above_chance_pp']:+.1f} | {r['cluster_ppv_pct']:.1f} |"
            )

    with open(os.path.join(out_dir, "validation_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote validation_fit_windows.csv ({len(fit_rows)} rows), "
          f"validation_cohort_sensitivity.csv ({len(cohort_rows)} rows), "
          f"validation_gap_burden.csv ({len(gap_rows)} rows) to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Validate cached Engine v2.3 outputs without retraining.")
    parser.add_argument("--datasets", default="weinstock,cgmacros", help="Comma-separated dataset names.")
    parser.add_argument("--outputs-dir", default=os.path.join(BASE_DIR, "Outputs"))
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "Outputs", "validation"))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    run_validation(datasets, args.outputs_dir, args.out_dir, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
